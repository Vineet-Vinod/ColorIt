from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from src.pipeline.config import AppConfig
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths


DEFAULT_SPENSERCAI_URL = (
    "https://huggingface.co/spensercai/DeOldify/resolve/main/ColorizeVideo_gen.pth"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_weights_manifest(manifest_path: Path, payload: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            delete=False,
            prefix=destination.name + ".",
            suffix=".part",
        ) as temp_handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                temp_handle.write(chunk)
            temp_path = Path(temp_handle.name)

    temp_path.replace(destination)


def run_download_weights(
    *,
    config: AppConfig,
    config_path: Path,
    url_override: str | None,
    force: bool,
) -> int:
    paths = resolve_project_paths(config)
    ensure_runtime_directories(paths)

    destination = paths.weights_path
    url = url_override or DEFAULT_SPENSERCAI_URL
    manifest_path = paths.manifest_dir / "weights.json"

    print(f"Config: {config_path.resolve()}")
    print(f"Download source: {url}")
    print(f"Destination: {destination}")
    print("Model loading: intentionally disabled in this bootstrap phase")

    if destination.exists() and not force:
        print("Weights already exist. Reusing the existing file.")
    else:
        download_file(url, destination)
        print("Weights downloaded successfully.")

    payload = {
        "filename": destination.name,
        "repo_id": config.model["repo_id"],
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "source_url": url,
        "weights_path": str(destination),
    }
    write_weights_manifest(manifest_path, payload)
    print(f"Manifest written to {manifest_path}")
    return 0
