from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

from src.pipeline.config import AppConfig
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths
from src.pipeline.pipeline_dataclasses import WeightDownload

DEFAULT_SPENSERCAI_URL = (
    "https://huggingface.co/spensercai/DeOldify/resolve/main/ColorizeVideo_gen.pth"
)
DEFAULT_DDCOLOR_URL = (
    "https://huggingface.co/piddnad/ddcolor_modelscope/resolve/main/pytorch_model.bin"
)
DEFAULT_DDCOLOR_WEIGHTS_PATH = Path("models/ddcolor/pytorch_model.bin")
FLUX_REPO_ID = "black-forest-labs/FLUX.2-klein-4B"
FLUX_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
FLUX_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
CMNET_REVISION = "e0d51432d224476769babfbff5e90f531a454939"
CMNET_CHECKPOINT = "DINOv3FeatureV6_LocalAtten_p374099.pth"


def experimental_weight_targets(root: Path) -> tuple[WeightDownload, ...]:
    flux = root / "models/flux"
    cmnet = root / "models/cmnet2"
    return (
        *(
            WeightDownload(
                "flux",
                FLUX_REPO_ID,
                f"https://huggingface.co/{FLUX_REPO_ID}/resolve/{FLUX_REVISION}/{name}",
                flux / name,
            )
            for name in FLUX_FILES
        ),
        WeightDownload(
            "cmnet2-source",
            "dan64/cmnet2",
            f"https://codeload.github.com/dan64/cmnet2/tar.gz/{CMNET_REVISION}",
            cmnet / "cmnet2-source.tar.gz",
        ),
        WeightDownload(
            "cmnet2",
            "dan64/cmnet2",
            f"https://github.com/dan64/cmnet2/releases/download/v1.3.0/{CMNET_CHECKPOINT}",
            cmnet / "weights" / CMNET_CHECKPOINT,
        ),
        WeightDownload(
            "dinov3",
            "dan64/cmnet2",
            "https://github.com/dan64/cmnet2/releases/download/v1.1.0/dinov3-vitb16.zip",
            cmnet / "dinov3-vitb16.zip",
        ),
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


def write_weights_manifest(manifest_path: Path, payload: dict[str, object]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def download_file(url: str, destination: Path) -> None:
    with (
        urllib.request.urlopen(url) as response,
        tempfile.NamedTemporaryFile(
            dir=destination.parent,
            delete=False,
            prefix=destination.name + ".",
            suffix=".part",
        ) as temp_handle,
    ):
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
    experimental: bool = False,
) -> int:
    paths = resolve_project_paths(config)
    ensure_runtime_directories(paths)

    deoldify_url = url_override or DEFAULT_SPENSERCAI_URL
    manifest_path = paths.manifest_dir / "weights.json"
    targets = (
        experimental_weight_targets(paths.root)
        if experimental
        else (
            WeightDownload(
                "deoldify",
                str(config.model["repo_id"]),
                deoldify_url,
                paths.weights_path,
            ),
            WeightDownload(
                "ddcolor",
                "piddnad/ddcolor_modelscope",
                DEFAULT_DDCOLOR_URL,
                (paths.root / DEFAULT_DDCOLOR_WEIGHTS_PATH).resolve(),
            ),
        )
    )

    print(f"Config: {config_path.resolve()}")
    print("Model loading: intentionally disabled in this bootstrap phase")

    weights = []
    for target in targets:
        destination = target.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"{target.name} source: {target.url}")
        print(f"{target.name} destination: {destination}")
        if destination.is_file() and not force:
            print(f"{target.name} weights already exist. Reusing the existing file.")
        else:
            download_file(target.url, destination)
            print(f"{target.name} weights downloaded successfully.")

        weights.append(
            {
                "name": target.name,
                "filename": destination.name,
                "repo_id": target.repo_id,
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
                "source_url": target.url,
                "weights_path": str(destination),
            }
        )

    if experimental:
        from src.experimental.assets import prepare_assets

        prepare_assets(paths.root, force=force)

    payload: dict[str, object] = {
        "weights": weights,
    }
    write_weights_manifest(manifest_path, payload)
    print(f"Manifest written to {manifest_path}")
    return 0
