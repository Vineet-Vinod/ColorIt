from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from src.experimental.experimental_dataclasses import ModelAssets
from src.pipeline.weights import (
    CMNET_CHECKPOINT,
    CMNET_REVISION,
    experimental_weight_targets,
)


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
    source = root / "models/cmnet2"
    patch = Path(__file__).with_name("cmnet2.patch")
    patch_digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    marker = source / ".colorit-patch"
    if force or not marker.exists():
        archive = source / "cmnet2-source.tar.gz"
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                relative = Path(member.name).relative_to(f"cmnet2-{CMNET_REVISION}")
                if not relative.parts or relative.parts[0] not in {
                    "colormnet",
                    "README.md",
                }:
                    continue
                if not member.isfile() or ".." in relative.parts:
                    continue
                contents = handle.extractfile(member)
                if contents is None:
                    continue
                destination = source / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with contents, destination.open("wb") as output:
                    shutil.copyfileobj(contents, output)
        subprocess.run(
            ["patch", "-p1", "--batch", "-i", str(patch)], cwd=source, check=True
        )
        marker.write_text(patch_digest)
    elif marker.read_text() != patch_digest:
        raise RuntimeError(
            f"CMNET2 adapter changed. Use a fresh model directory: {source}"
        )
    weights = source / "weights"
    weights.mkdir(exist_ok=True)
    backbone = weights / "dinov3-vitb16"
    if force or not all(
        (backbone / name).is_file() for name in ("config.json", "model.safetensors")
    ):
        with zipfile.ZipFile(source / "dinov3-vitb16.zip") as handle:
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
    (source / "colormnet/models.json").write_text(
        json.dumps({"cmnet2": {"dinov3": {"checkpoint": CMNET_CHECKPOINT}}})
    )
    return ModelAssets(flux=root / "models/flux", cmnet=source)
