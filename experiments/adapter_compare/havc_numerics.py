from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch


def require_finite(value: object, stage: str) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"HAVC produced a nonfinite tensor at {stage}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite(item, f"{stage}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{stage}.{key}")


def guard_method(target: object, name: str, stage: str) -> None:
    original = cast(Callable[..., object], getattr(target, name))

    def guarded(*args: object, **kwargs: object) -> object:
        require_finite((args, kwargs), stage + ".input")
        result = original(*args, **kwargs)
        require_finite(result, stage + ".output")
        return result

    setattr(target, name, guarded)


def guard_pipeline(pipeline: dict[str, object], diffusion: torch.nn.Module) -> None:
    guard_method(pipeline["clip"], "encode_from_tokens_scheduled", "clip")
    guard_method(pipeline["vae"], "encode", "vae.encode")
    guard_method(pipeline["vae"], "decode", "vae.decode")
    guard_method(diffusion, "_forward", "transformer")
