from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw["paths"]

    @property
    def runtime(self) -> dict[str, Any]:
        return self.raw["runtime"]

    @property
    def video(self) -> dict[str, Any]:
        return self.raw["video"]

    @property
    def scenes(self) -> dict[str, Any]:
        return self.raw["scenes"]

    @property
    def compression(self) -> dict[str, Any]:
        return self.raw.get("compression", {})


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text()) or {}
    for key in ("model", "runtime", "video", "scenes", "paths"):
        if key not in data:
            raise ValueError(f"Config file is missing required top-level key: {key}")

    return AppConfig(raw=data, path=path.resolve())
