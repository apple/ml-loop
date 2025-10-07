#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import inspect
from collections.abc import Callable
from functools import wraps
from types import NoneType, UnionType
from typing import Any, Literal, NoReturn, TypeVar, Union, cast, get_args, get_origin

from fire import Fire as UntypedFire

FuncT = TypeVar("FuncT", bound=Callable[..., Any])


def type_cast(v: Any, T: type) -> Any:
    if get_origin(T) is Literal:
        return v
    if get_origin(T) in [Union, UnionType]:
        if v is None and NoneType in get_args(T):
            return v
        for TT in get_args(T):
            try:
                return type_cast(v, TT)
            except ValueError:
                pass
    return T(v)


def _type_wrap(f: FuncT) -> FuncT:
    signature = inspect.signature(f)

    @wraps(f)
    def _w(*args: Any, **kwds: Any) -> Any:
        ba = signature.bind(*args, **kwds)
        arguments = []
        kw_arguments = {}
        for k, v in ba.arguments.items():
            p = signature.parameters[k]
            T = p.annotation
            try:
                if p.kind in [
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.POSITIONAL_ONLY,
                ]:
                    arguments.append(type_cast(v, T))
                elif p.kind == inspect.Parameter.VAR_POSITIONAL:
                    arguments.extend([type_cast(i, T) for i in v])
                elif p.kind == inspect.Parameter.KEYWORD_ONLY:
                    kw_arguments[k] = type_cast(v, T)
                elif p.kind == inspect.Parameter.VAR_KEYWORD:
                    for n, i in v.items():
                        kw_arguments[n] = type_cast(i, T)
            except ValueError:
                raise ValueError(
                    f"Invalid argument for parameter {k!r}. Expected {T}, got {type(v)}"
                ) from None
        rval = f(*arguments, **kw_arguments)
        exit(rval or 0)

    return cast(FuncT, _w)


FireType = Callable[..., Any] | dict[str, Callable[..., Any]] | list[Callable[..., Any]]


def Fire(component: FireType, *args: Any, **kwds: Any) -> NoReturn:
    match component:
        case dict():
            UntypedFire({k: _type_wrap(v) for k, v in component.items()}, *args, **kwds)
        case list():
            UntypedFire([_type_wrap(v) for v in component], *args, **kwds)
        case func if callable(func):
            UntypedFire(_type_wrap(func), *args, **kwds)
        case _:
            raise ValueError("Component should be dict, list, or callable")
    exit(255)


def test(a: str, b: int = 0, c: float = 1.0) -> None:
    print(type(a), type(b), type(c))


if __name__ == "__main__":
    Fire(test)
