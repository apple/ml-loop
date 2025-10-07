#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import dataclasses
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from collections import deque
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import BufferedReader, TextIOWrapper
from pathlib import Path
from typing import IO, Any, Literal, cast, overload

import boto3
import fsspec
import lz4.frame
from boto3 import Session as Boto3Session

__path_to_str_re = re.compile(r"^(?P<scheme>\w+?):/(?P<nonslash>[^/])")


# Optional: set AWS_PROFILE externally via env var if needed, or rely on IAM/default config
AWS_PROFILE = os.environ.get("AWS_PROFILE")  # e.g. "default" or "your-profile"

# Thread-local cache to avoid redundant session creation
thread_local_cache = threading.local()

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))


def filesystem_for_scheme(scheme: str) -> Any:
    if scheme != "s3":
        return fsspec.filesystem(scheme)

    boto3_session: Boto3Session | None = getattr(thread_local_cache, "boto3_session", None)
    if boto3_session is None or (
        boto3_session.get_credentials() and boto3_session.get_credentials().refresh_needed()
    ):
        boto3_session = boto3.Session(profile_name=AWS_PROFILE)
        thread_local_cache.boto3_session = boto3_session

        creds = boto3_session.get_credentials().get_frozen_credentials()
        thread_local_cache.boto3_fs = fsspec.filesystem(
            scheme,
            key=creds.access_key,
            secret=creds.secret_key,
            token=creds.token,
            use_listings_cache=False,
        )

    return thread_local_cache.boto3_fs


def path_to_str(path: Path | str) -> str:
    """Safely convert a path to string.

    Path doesn't work with URI's, it will convert "s3://my/path" to "s3:/my/path" which causes
    other utilities to fail. Here we just convert to string and add back in the forward slash.
    """
    if isinstance(path, Path):
        path_s = path.as_posix()
    elif isinstance(path, str):
        path_s = path
    else:
        raise Exception(f"Unrecognized input type {type(path)}")

    return __path_to_str_re.sub(r"\g<scheme>://\g<nonslash>", path_s)


def get_scheme_and_path(uri_path: Path | str) -> tuple[str, str]:
    """Split a URI into scheme and path.

    If no scheme is specified it defaults to 'file'

    Examples: 's3://foo/bar' --> ('s3', 'foo/bar')
              'foo/bar' --> ('file', 'foo/bar')

    Args:
        uri_path: The URI.

    Returns:
        Scheme and path of the given URI.

    """
    uri_path = path_to_str(uri_path)
    parts = uri_path.split("://")
    if len(parts) == 1:
        parts = ["file"] + parts
    if len(parts) != 2:
        raise RuntimeError(f'The provided string "{uri_path}" is not a valid URI')
    scheme, path = parts
    return scheme, path


def get_scheme(uri_path: Path | str) -> str:
    scheme, _ = get_scheme_and_path(uri_path)
    return scheme


def get_fs(uri_path: Path | str) -> Any:
    scheme, _ = get_scheme_and_path(uri_path)
    return filesystem_for_scheme(scheme)


@contextmanager
def uri_open(
    uri_path: Path | str, mode: str = "r", use_temp_file: bool = False
) -> Generator[IO[Any], None, None]:
    uri_path = path_to_str(uri_path)
    scheme, path = get_scheme_and_path(uri_path)
    if scheme == "file":
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    fs = filesystem_for_scheme(scheme)
    if use_temp_file:
        with tempfile.NamedTemporaryFile(suffix=Path(uri_path).suffix) as tmp:
            temp_filename = tmp.name

        if "r" in mode or ("x" in mode and fs.exists(path)):
            copy(uri_path, temp_filename)

        with open(temp_filename, mode) as fh:
            yield fh

        if "w" in mode or "x" in mode:
            copy(temp_filename, uri_path)

        Path(temp_filename).unlink()
    else:
        with fs.open(path, mode) as fh:
            yield fh


@contextmanager
def uri_write_filename(uri_path: Path | str) -> Generator[str, None, None]:
    uri_path = path_to_str(uri_path)
    scheme, path = get_scheme_and_path(uri_path)
    if scheme == "file":
        yield path
    else:
        # Create a named temporary file and close it immediately
        with tempfile.NamedTemporaryFile(delete=True, suffix=Path(uri_path).suffix) as tmp:
            temp_filename = tmp.name

        yield temp_filename

        # Upload it
        copy(temp_filename, uri_path)
        # delete the local file
        Path(temp_filename).unlink()


def exists(uri_path: Path | str) -> bool:
    uri_path = path_to_str(uri_path)
    scheme, path = get_scheme_and_path(uri_path)
    fs = filesystem_for_scheme(scheme)
    return cast(bool, fs.exists(path))


def is_dir(uri_path: Path | str) -> bool:
    fs = get_fs(path_to_str(uri_path))
    return cast(bool, fs.isdir(get_scheme_and_path(uri_path)[1]))


def glob(uri_path: Path | str) -> list[str]:
    uri_path = path_to_str(uri_path)
    scheme, _ = get_scheme_and_path(uri_path)
    fs = filesystem_for_scheme(scheme)
    return [f"{scheme}://{file}" for file in fs.glob(uri_path)]


def listdir(uri_path: Path | str, keep_file_scheme: bool = True) -> list[str]:
    uri_path = path_to_str(uri_path)
    scheme, _ = get_scheme_and_path(uri_path)
    fs = filesystem_for_scheme(scheme)
    if keep_file_scheme:
        return [f"{scheme}://{file}" for file in fs.ls(uri_path)]
    else:
        return [file if scheme == "file" else f"{scheme}://{file}" for file in fs.ls(uri_path)]


def listdir_path(uri_path: Path | str) -> list[Path]:
    # Can't keep file scheme because then we don't know if it's absolute or relative because Path
    # eats all the extra ///
    return [Path(f) for f in listdir(uri_path, keep_file_scheme=False)]


def delete(uri_path: Path | str, recursive: bool = False) -> None:
    uri_path = path_to_str(uri_path)
    scheme, path = get_scheme_and_path(uri_path)
    if scheme == "file" and Path(path).is_dir():
        shutil.rmtree(path)
        return

    fs = filesystem_for_scheme(scheme)
    if fs.exists(path):
        fs.delete(path, recursive=recursive)


def copy(src_uri: Path | str, dst_uri: Path | str) -> None:
    src_uri = path_to_str(src_uri)
    dst_uri = path_to_str(dst_uri)
    src_scheme, src_path = get_scheme_and_path(src_uri)
    dst_scheme, dst_path = get_scheme_and_path(dst_uri)

    if src_uri.startswith("s3://") or dst_uri.startswith("s3://"):
        """Copy files with a uri, such as s3:// or file://

        This is expected to work with any combination of local and remote.
        """
        assert src_scheme in ("s3", "file")
        assert dst_scheme in ("s3", "file")
        s3_fs = filesystem_for_scheme("s3")
        if src_scheme == "s3" and dst_scheme == "s3":
            s3_fs.copy(src_path, dst_path, recursive=True)
        elif src_scheme == "s3":
            s3_fs.get(src_path, dst_path, recursive=True)
        else:
            s3_fs.put(src_path, dst_path, recursive=True)
    else:
        if Path(src_path).is_file():
            shutil.copy(src_path, dst_path)
        else:
            safe_mkdir(Path(dst_uri).parent, exist_ok=True, parents=True)
            shutil.copytree(src_path, dst_path)


def mkdir(uri_path: Path | str, parents: bool = False, exist_ok: bool = False) -> None:
    uri_path = path_to_str(uri_path)
    get_fs(uri_path).mkdir(uri_path, parents=parents, exist_ok=exist_ok)


def safe_mkdir(uri_path: Path | str, parents: bool = False, exist_ok: bool = False) -> None:
    """Some filesystems (i.e. Lustre) seem to ignore exist_ok. So we're trying to be extra safe in this function."""
    if exists(uri_path):
        return

    try:
        mkdir(uri_path, parents, exist_ok)
    except FileExistsError:
        if not exist_ok:
            raise


OpenTextReadingMode = Literal["r"]
OpenBinaryReadingMode = Literal["rb"]


@overload
def open_readonly(filename: Path | str) -> TextIOWrapper: ...


@overload
def open_readonly(filename: Path | str, mode: OpenTextReadingMode) -> TextIOWrapper: ...


@overload
def open_readonly(filename: Path | str, mode: OpenBinaryReadingMode) -> BufferedReader: ...


def open_readonly(
    filename: Path | str,
    mode: OpenTextReadingMode | OpenBinaryReadingMode = "r",
) -> TextIOWrapper | BufferedReader:
    assert "r" in mode

    return cast(
        TextIOWrapper | BufferedReader,
        os.fdopen(os.open(filename, os.O_RDONLY | os.O_NOATIME), mode),
    )


def open_zip_readonly(zip_name: Path | str, in_memory: bool = False) -> zipfile.ZipFile:
    if in_memory:
        with open_readonly(zip_name, "rb") as f:
            return zipfile.ZipFile(io.BytesIO(f.read()), "r")
    else:
        return zipfile.ZipFile(open_readonly(zip_name, "rb"), "r")


def open_maybe_zipped(
    filepath: Path,
    opened_cache: dict[str, zipfile.ZipFile] | None = None,
    in_memory: bool = False,
) -> IO[bytes]:
    filepath_s = path_to_str(filepath)
    if ".zip/" in filepath_s:
        zip_name, filename = filepath_s.split(".zip/")
        zip_name = zip_name + ".zip"

        if opened_cache is None:
            with open_zip_readonly(zip_name, in_memory) as z:
                return z.open(filename)
        else:
            if zip_name not in opened_cache:
                opened_cache[zip_name] = open_zip_readonly(zip_name, in_memory)

            return opened_cache[zip_name].open(filename)
    else:
        return open_readonly(filepath, "rb")


def read_maybe_zipped(
    filepath: Path,
    opened_cache: dict[str, zipfile.ZipFile] | None = None,
    in_memory: bool = False,
) -> bytes:
    with open_maybe_zipped(filepath, opened_cache, in_memory) as f:
        return f.read()


def zip_folder(folder: Path, root_dir: Path, clean: bool = True) -> Path:
    relative_folder = folder.relative_to(root_dir)

    args = ["zip", "-0", "-r"]

    if clean:
        args.extend(("--move", "--test"))

    args.extend((relative_folder.stem, path_to_str(relative_folder)))

    _ = subprocess.check_output(
        args,
        cwd=root_dir,
    )

    return (root_dir / folder.stem).with_suffix(".zip")


def zip_folders_of_lz4(source_folder: Path) -> list[tuple[Path, Path]]:
    zips_and_manifests = []
    for folder in set(f.parent for f in source_folder.rglob("*.lz4")):
        zip_name = zip_folder(folder, folder.parent)
        manifest_name = zip_name.parent / (zip_name.stem + "_lz4.json.manifest")

        with open_zip_readonly(zip_name) as zf:
            readers = [
                construct_file_in_zip_reader(zip_name, zf, name)
                for name in zf.namelist()
                if name.endswith(".lz4")
            ]

        with lz4.frame.open(manifest_name, "wt") as f:
            json.dump(
                [
                    dataclasses.asdict(dataclasses.replace(r, arc_name=Path(r.arc_name.name)))
                    for r in readers
                ],
                f,
            )

        zips_and_manifests.append((zip_name, manifest_name))

    return zips_and_manifests


@dataclasses.dataclass(frozen=True)
class FileInArchive:
    arc_name: Path
    name: str
    offset_to_file: int
    file_size: int

    def read(self, shared_file_handles: dict[str, BufferedReader] | None = None) -> bytes:
        if shared_file_handles is None:
            with open_readonly(self.arc_name, "rb") as f:
                f.seek(self.offset_to_file)
                return f.read(self.file_size)
        else:
            arc_name_s = self.arc_name.as_posix()
            if arc_name_s in shared_file_handles:
                f = shared_file_handles[arc_name_s]
            else:
                f = open_readonly(self.arc_name, "rb")
                shared_file_handles[arc_name_s] = f

            f.seek(self.offset_to_file)
            return f.read(self.file_size)

    @classmethod
    def from_dict(cls, folder: Path, d) -> "FileInArchive":  # type: ignore
        arc_name = folder / str(d.pop("arc_name"))
        return cls(arc_name=arc_name, **d)


def load_archive_manifest(arc_filename: Path | str) -> list[FileInArchive]:
    if isinstance(arc_filename, str):
        arc_filename = Path(arc_filename)

    maybe_manifest_name = arc_filename.parent / (arc_filename.stem + "_lz4.json.manifest")
    arc_filename = arc_filename.absolute().resolve()
    if maybe_manifest_name.exists():
        with open_readonly(maybe_manifest_name, "rb") as f:
            data = f.read()
        return [
            FileInArchive.from_dict(arc_filename.parent, ele)
            for ele in json.loads(lz4.frame.decompress(data).decode("utf-8"))
        ]
    else:
        with open_zip_readonly(arc_filename) as zf:
            return [
                construct_file_in_zip_reader(arc_filename, zf, name)
                for name in zf.namelist()
                if name.endswith(".lz4")
            ]


def construct_file_in_zip_reader(zip_name: Path, zf: zipfile.ZipFile, name: str) -> FileInArchive:
    zinfo = zf.getinfo(name)
    assert zf.compression == zipfile.ZIP_STORED
    assert zinfo.compress_type == zipfile.ZIP_STORED
    assert zinfo.file_size == zinfo.compress_size

    ##
    # NB: After opening the zip file, the file pointer will be a the start of the data
    # for the file in the zip.
    opened_file = zf.open(name)

    return FileInArchive(zip_name, name, opened_file._fileobj.tell(), zinfo.file_size)  # type: ignore


def get_git_root(path: Path | str) -> Path:
    from git.repo import Repo

    git_repo = Repo(path_to_str(path), search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return Path(git_root)


class FileRetention:
    """Utility class that implements a retention policy for artifacts created during training.

    Features:
    * Optionally back up files to S3.
    * Delete backed-up files from the local storage to save disk space.
    * Keep N most recent files for debugging.
    * If S3 backup is not set up, keep all local copies.
    """

    def __init__(
        self,
        save_to_s3: bool,
        s3_base_path: str | None = None,
        keep_n_local: int = 3,
        verbose: bool = False,
    ):
        if save_to_s3:
            assert s3_base_path is not None

        self.s3_base_path = s3_base_path if save_to_s3 else None

        # how many local copies to keep for each file type
        self.keep_n_local = keep_n_local
        self.recent_local_files: dict[str, deque] = dict()  # type: ignore

        self.verbose = verbose

    def upload_to_s3(self, local_path: str) -> None:
        if self.verbose:
            print(f"Uploading {Path(local_path).name} to S3...")

        assert self.s3_base_path is not None
        copy(local_path, Path(self.s3_base_path) / Path(local_path).name)

    def backup_and_cleanup(self, local_paths: dict[str, str]) -> None:
        # we keep all local copies unless backup is set up
        if self.s3_base_path is None:
            return

        with ThreadPoolExecutor(max_workers=5, thread_name_prefix="file_s3_upload") as executor:
            futures = [executor.submit(self.upload_to_s3, path) for path in local_paths.values()]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred during file upload: {e}")

        self.schedule_cleanup(local_paths)
        self.cleanup()

    def schedule_cleanup(self, local_paths: dict[str, str]) -> None:
        for file_type, local_path in local_paths.items():
            if file_type not in self.recent_local_files:
                self.recent_local_files[file_type] = deque([])
            self.recent_local_files[file_type].append(local_path)

    def cleanup(self) -> None:
        for local_files in self.recent_local_files.values():
            while len(local_files) > self.keep_n_local:
                to_delete = local_files.popleft()
                if self.verbose:
                    print(f"Deleting old artifact {Path(to_delete).name}...")

                if not Path(to_delete).exists():
                    print(f"{to_delete=} not found!")
                elif Path(to_delete).is_dir():
                    shutil.rmtree(to_delete)
                else:
                    delete(to_delete)


def project_source_root() -> Path:
    """
    Returns the source root directory of the project as a Path object.
    Assumes this function is in agents/phi_agents/utils/file_utils.py.
    """
    return Path(__file__).resolve().parents[2]


def get_path_to_python_env_bin(bin_name: str, override_path: str | None = None) -> Path:
    """
    Returns:
        Path: Path to the bin_name in the current python env.
    """
    if override_path and Path(override_path).exists():
        logger.info(f"Using override_path: '{override_path}' for {bin_name}")
        return Path(override_path)

    local_python_env_bin_path = Path(sys.executable).with_name(bin_name)
    # if local_python_env_bin_path doesn't exist fall back to whatever is on the path
    if not local_python_env_bin_path.exists():
        logger.warning(
            f"'{bin_name}' not found at '{local_python_env_bin_path}' falling back to relative."
        )
        return Path(bin_name)
    return local_python_env_bin_path


def read_directory_files(source_dir: str | Path) -> list[tuple[bytes, str]]:
    source_path = Path(source_dir)
    files_data = []
    for file_path in source_path.glob("**/*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(source_path)
            content = file_path.read_bytes()
            files_data.append((content, str(rel_path)))
    return files_data


def lora_path(checkpoint_dir: Path | None) -> Path | None:
    """LoRA weights path relative to the checkpoint dir."""
    if checkpoint_dir is None:
        return None
    else:
        return checkpoint_dir / "lora"
