from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ClipRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Path
    output: Path | None = None
    overwrite: bool = False
    resume: bool = False


@dataclass(frozen=True)
class ClipInfo:
    width: int
    height: int
    fps: Fraction
    frames: int
    duration: float
    audio: bool


@dataclass(frozen=True)
class Shot:
    start: int
    end: int
    references: tuple[int, ...]


@dataclass(frozen=True)
class ModelAssets:
    flux: Path
    cmnet: Path


@dataclass(frozen=True)
class Reference:
    frame: int
    source: Path
    colored: Path
