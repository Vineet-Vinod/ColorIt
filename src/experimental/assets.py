from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from pydantic import TypeAdapter

from src.experimental.experimental_dataclasses import Artifact, ModelAssets

CMNET_REVISION = "e0d51432d224476769babfbff5e90f531a454939"


def require_apple_silicon() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "Experimental FLUX + CMNET2 currently requires Apple Silicon macOS."
        )


def verify_artifact(path: Path, artifact: Artifact) -> None:
    if path.stat().st_size != artifact.size:
        raise ValueError(f"Unexpected size for {path}")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != artifact.sha256:
        raise ValueError(f"Checksum mismatch for {path}")


def download_artifact(root: Path, artifact: Artifact) -> Path:
    destination = root / artifact.path
    if destination.is_file():
        verify_artifact(destination, artifact)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {artifact.path}", flush=True)
    request = urllib.request.Request(artifact.url, headers={"User-Agent": "ColorIt"})
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            partial.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
        verify_artifact(partial, artifact)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def prepare_assets(root: Path) -> ModelAssets:
    require_apple_silicon()
    directory = root / "data/experimental/models"
    artifacts = TypeAdapter(tuple[Artifact, ...]).validate_json(
        Path(__file__).with_name("assets.json").read_text()
    )
    print("Checking pinned FLUX and CMNET2 assets.", flush=True)
    for artifact in artifacts:
        download_artifact(directory, artifact)
    source = directory / "cmnet2"
    patch = Path(__file__).with_name("cmnet2.patch")
    patch_digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    marker = source / ".colorit-patch"
    if not marker.exists():
        archive = directory / "cmnet2-source.tar.gz"
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
    checkpoint = "DINOv3FeatureV6_LocalAtten_p374099.pth"
    if not (weights / checkpoint).exists():
        (weights / checkpoint).symlink_to(directory / checkpoint)
    backbone = weights / "dinov3-vitb16"
    if not all(
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
    (source / "colormnet/models.json").write_text(
        json.dumps({"cmnet2": {"dinov3": {"checkpoint": checkpoint}}})
    )
    return ModelAssets(flux=directory / "flux2-klein-4b", cmnet=source)
