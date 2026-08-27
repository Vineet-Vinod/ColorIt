"""Safe, restricted conversion for the official Control-Color checkpoint.

The authors publish ``main_model.ckpt`` as a PyTorch zip checkpoint.  PyTorch
checkpoints are pickle containers, so callers must never load it with the
default unpickler.  This module accepts only a tensor-only state dictionary
through PyTorch's ``weights_only`` loader and emits independent safetensors
files for the inference components used by the MLX base path.

The optional content-guided deformable VAE is intentionally excluded.  It uses
deformable convolution and is not part of the automatic default path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


OFFICIAL_REPOSITORY = "https://github.com/ZhexinLiang/Control-Color"
OFFICIAL_REPOSITORY_REVISION = "f21054af54f524591f7a3c0862fa90392f7d33c0"
OFFICIAL_DRIVE_FILE_ID = "1sYnRwm6aVs2rRlRukrVeeNqA14zdYSwi"
OFFICIAL_EXPECTED_BYTES = 8_601_357_057


@dataclass(frozen=True)
class ConvertedComponent:
    name: str
    tensor_count: int
    sha256: str
    path: str


@dataclass(frozen=True)
class ConversionManifest:
    format: str
    source: str
    source_bytes: int
    source_sha256: str
    official_repository: str
    official_repository_revision: str
    official_drive_file_id: str
    components: list[ConvertedComponent]


_COMPONENT_PREFIXES = {
    "unet": "model.diffusion_model.",
    "controlnet": "control_model.",
    "vae": "first_stage_model.",
    "text_encoder": "cond_stage_model.",
}

_REQUIRED_KEYS = {
    "unet": "model.diffusion_model.input_blocks.0.0.weight",
    "controlnet": "control_model.input_blocks.0.0.weight",
    "vae": "first_stage_model.encoder.conv_in.weight",
    "text_encoder": "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
}


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a checkpoint."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tensor_state_dict(checkpoint: Path) -> dict[str, Any]:
    """Load only tensors from a PyTorch zip checkpoint.

    ``weights_only=True`` prevents arbitrary pickle globals from executing.
    The explicit validation below also rejects optimizer state, model objects,
    and non-string keys before anything is written to disk.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError("Control-Color conversion requires the optional torch package") from exc

    try:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Refusing to load Control-Color with PyTorch's unsafe pickle loader. "
            "The checkpoint must be a PyTorch zip state_dict accepted by "
            "torch.load(weights_only=True, mmap=True)."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Control-Color checkpoint must contain a dictionary")
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, dict):
        raise ValueError("Control-Color state_dict must be a dictionary")

    tensors: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise ValueError("Control-Color state_dict contains a non-string key")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Control-Color state_dict contains non-tensor value at {key!r}")
        if value.layout != torch.strided:
            raise ValueError(f"Control-Color state_dict contains unsupported layout at {key!r}")
        if value.is_sparse:
            raise ValueError(f"Control-Color state_dict contains sparse tensor at {key!r}")
        tensors[key] = value.detach().contiguous()
    return tensors


def _component_tensors(name: str, state_dict: dict[str, Any]) -> dict[str, Any]:
    prefix = _COMPONENT_PREFIXES[name]
    required = _REQUIRED_KEYS[name]
    if required not in state_dict:
        raise ValueError(
            f"Checkpoint does not match the released Control-Color inference architecture: "
            f"missing {required!r}"
        )
    tensors = {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not tensors:
        raise ValueError(f"Checkpoint has no {name} tensors")
    return tensors


def convert_checkpoint(checkpoint: Path, output_dir: Path) -> ConversionManifest:
    """Convert the verified official checkpoint to separate safetensors files.

    The source size check detects a partial Google Drive download.  It is not a
    substitute for a vendor hash, which the authors have not published.  The
    returned manifest records a local source hash and hashes every converted
    component so subsequent MLX runs never need to open the pickle container.
    """
    checkpoint = checkpoint.resolve(strict=True)
    if checkpoint.name != "main_model.ckpt":
        raise ValueError("Expected the official file name 'main_model.ckpt'")
    source_bytes = checkpoint.stat().st_size
    if source_bytes != OFFICIAL_EXPECTED_BYTES:
        raise ValueError(
            f"Expected official main_model.ckpt size {OFFICIAL_EXPECTED_BYTES} bytes, "
            f"got {source_bytes}. Refusing partial or substituted checkpoint."
        )

    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError("Control-Color conversion requires safetensors") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    source_sha256 = sha256_file(checkpoint)
    state_dict = _load_tensor_state_dict(checkpoint)

    components: list[ConvertedComponent] = []
    for name in _COMPONENT_PREFIXES:
        destination = output_dir / f"{name}.safetensors"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing converted weights: {destination}")
        tensors = _component_tensors(name, state_dict)
        save_file(tensors, str(destination), metadata={"format": "pt"})
        components.append(
            ConvertedComponent(
                name=name,
                tensor_count=len(tensors),
                sha256=sha256_file(destination),
                path=destination.name,
            )
        )

    manifest = ConversionManifest(
        format="control-color-mlx-components-v1",
        source=checkpoint.name,
        source_bytes=source_bytes,
        source_sha256=source_sha256,
        official_repository=OFFICIAL_REPOSITORY,
        official_repository_revision=OFFICIAL_REPOSITORY_REVISION,
        official_drive_file_id=OFFICIAL_DRIVE_FILE_ID,
        components=components,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    manifest = convert_checkpoint(args.checkpoint, args.output_dir)
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
