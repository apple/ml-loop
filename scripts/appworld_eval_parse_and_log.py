#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""This script wraps the appworld eval script and allows us to capture its stdout of the final
results into a pandas Dataframe.
"""

import logging
import subprocess
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import hydra
import pandas as pd

# needed to get the config cs.store registered
import phi_agents.rl.config  # noqa: F401
import phi_agents.utils.file_utils as fu
from phi_agents.inference.config import MainInferenceConfig

logger = logging.getLogger(__name__)


DEFAULT_APPWORLD_EXE = "appworld-env/bin/appworld"


@hydra.main(version_base=None, config_path="../phi_agents/rl/conf", config_name="appworld_eval")  # type: ignore[misc]
def main(cfg: MainInferenceConfig) -> None:
    # Run the bash command and capture the output
    df = extract_df(cfg)
    log_df_to_all_results_txt(cfg, df)


def extract_df(cfg: MainInferenceConfig) -> pd.DataFrame:
    appworld_cli = fu.get_path_to_python_env_bin("appworld", override_path=DEFAULT_APPWORLD_EXE)
    command = (
        str(appworld_cli),
        "evaluate",
        cfg.experiment_name,
        cfg.scenario_sampler.dataset_name,
        "--root",
        "$APPWORLD_ROOT",
    )
    result = subprocess.run(
        " ".join(command),
        text=True,
        shell=True,
        stdout=subprocess.PIPE,  # Stream stdout to the console
        stderr=sys.stderr,  # Stream stderr to the console
        check=True,  # Raises CalledProcessError on non-zero exit
    )
    output = result.stdout
    print(output)
    # Remove output above the matrix
    result_str = output[output.find("type") :]
    # Get ride of the -------- line
    result_str_list = result_str.split("\n")
    result_str_list.pop(1)
    result_str = "\n".join(result_str_list)
    # Remove whitespace
    result_str = result_str.replace(" ", "").strip()
    # Use StringIO to read the string as if it's a CSV file
    df = pd.read_csv(StringIO(result_str), sep="|").set_index("type")
    return df


def log_df_to_all_results_txt(cfg: MainInferenceConfig, df: pd.DataFrame) -> None:
    # Remove the -Instruct and stuff after to shorten the name, we can add this back later if we
    # ever end up using a non-Instruct model
    base_model = Path(cfg.llm.base_model_path).name.split("-Instruct")[0]
    output_path = Path(cfg.log_dir) / "all_results.txt"
    logger.info(f"Logging results to {output_path}")
    print(f"Logging results to {output_path}")
    with fu.uri_open(output_path, "a") as fh:
        date_time_string = datetime.now().strftime("%Y-%m-%d %H:%M")
        fh.write(
            f"{date_time_string} {base_model:<14} "
            f"{cfg.scenario_sampler.dataset_name:<11} "
            f"{cfg.experiment_name:<50} "
            f"TGC: {df.loc['aggregate', 'task_goal_completion']:<4}  "
            f"SGC: {df.loc['aggregate', 'scenario_goal_completion']:<4}  "
            f"TGC_1: {df.loc['difficulty_1', 'task_goal_completion']:<4} "
            f"TGC_2: {df.loc['difficulty_2', 'task_goal_completion']:<4} "
            f"TGC_3: {df.loc['difficulty_3', 'task_goal_completion']:<4} "
            f"SGC_1: {df.loc['difficulty_1', 'scenario_goal_completion']:<4} "
            f"SGC_2: {df.loc['difficulty_2', 'scenario_goal_completion']:<4} "
            f"SGC_3: {df.loc['difficulty_3', 'scenario_goal_completion']:<4} "
            "\n"
        )


if __name__ == "__main__":
    phi_agents.rl.config.register_hydra_resolvers()
    main()
