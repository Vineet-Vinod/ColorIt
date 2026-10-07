from __future__ import annotations

import platform
import shutil
import sys
import zipfile
from pathlib import Path

from src.experimental.experimental_dataclasses import ModelAssets
from src.pipeline.weights import experimental_weight_targets


def require_apple_silicon() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "Experimental FLUX + CMNET2 currently requires Apple Silicon macOS."
        )


def prepare_assets(root: Path, *, force: bool = False) -> ModelAssets:
    for target in experimental_weight_targets(root):
        if not target.destination.is_file():
            raise FileNotFoundError(
                f"Missing model file: {target.destination}. "
                "Run `colorit download-weights --experimental` first."
            )
    directory = root / "models/cmnet2"
    weights = directory / "weights"
    backbone = weights / "dinov3-vitb16"
    if force or not all(
        (backbone / name).is_file() for name in ("config.json", "model.safetensors")
    ):
        with zipfile.ZipFile(directory / "dinov3-vitb16.zip") as handle:
            for name in ("config.json", "model.safetensors"):
                zip_member = next(
                    item for item in handle.namelist() if item.endswith("/" + name)
                )
                backbone.mkdir(exist_ok=True)
                with (
                    handle.open(zip_member) as incoming,
                    (backbone / name).open("wb") as outgoing,
                ):
                    shutil.copyfileobj(incoming, outgoing)
    return ModelAssets(flux=root / "models/flux", cmnet=directory)
