"""FireRed Image Edit 1.1 support through MFLUX's Qwen edit runtime.

FireRed 1.1 uses the QwenImageTransformer2DModel, Qwen 2.5 VL encoder,
and Qwen Image VAE.  Its tensor names and shapes match Qwen Image Edit
2511, while its transformer config keeps ``zero_cond_t`` disabled.  MFLUX's
Qwen edit implementation already follows that non-zero conditioning path, so
loading FireRed requires a model configuration rather than a model rewrite.

This module deliberately accepts only a local snapshot.  Weight downloads are
handled separately so callers can pin and verify the Hugging Face revision
before this code opens any multi-gigabyte files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FIRERED_REPO_ID = "FireRedTeam/FireRed-Image-Edit-1.1"
FIRERED_REVISION = "3bc3f2a12722fd9883eb6357500de191d56baaf5"
FIRERED_LICENSE = "apache-2.0"
FIRERED_WEIGHT_SHA256 = {
    "text_encoder/model-00001-of-00004.safetensors": "d725335e4ea2399be706469e4b8807716a8fa64bd03468252e9f7acf2415fee4",
    "text_encoder/model-00002-of-00004.safetensors": "b1830db6908dcc76df3a71492acbcf2b8cac130114cf1f3c2d9edae8de8c6de3",
    "text_encoder/model-00003-of-00004.safetensors": "09c1807c6d00d7cab94f7db39d4c02ebb8537225ccde383861ac48db97945aa6c",
    "text_encoder/model-00004-of-00004.safetensors": "5dd068336d14d45ffb43cef374d286cc6ba9d8741b028f90a7d040d847961f4a",
    "transformer/diffusion_pytorch_model-00001-of-00005.safetensors": "cd6f0d78a3a8c21792538d0abae604bd7abbca1508a2e8c778ea359f5fabd180",
    "transformer/diffusion_pytorch_model-00002-of-00005.safetensors": "bb6b283ea5954aa16e8df94fbbd37368c48c07ff3cfcf3a514117333c3753463",
    "transformer/diffusion_pytorch_model-00003-of-00005.safetensors": "e0602f9a002cf2807080bfb6d055cbcd6991887be7e89c82fa9411f938195923",
    "transformer/diffusion_pytorch_model-00004-of-00005.safetensors": "ae1d1ec1a35f5f59b086c1947dbf62d67d972d25cf7c771640921c9aa97ee492",
    "transformer/diffusion_pytorch_model-00005-of-00005.safetensors": "9d304d6539e7dad0647346cecf68d1fd5dc0efc7319626cdde2fdfc3e1533417",
    "vae/diffusion_pytorch_model.safetensors": "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344",
}

_EXPECTED_TRANSFORMER = {
    "_class_name": "QwenImageTransformer2DModel",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": False,
    "in_channels": 64,
    "joint_attention_dim": 3584,
    "num_attention_heads": 24,
    "num_layers": 60,
    "out_channels": 16,
    "patch_size": 2,
}
_EXPECTED_COMPONENTS = {
    "processor": ["transformers", "Qwen2VLProcessor"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
    "tokenizer": ["transformers", "Qwen2Tokenizer"],
    "transformer": ["diffusers", "QwenImageTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLQwenImage"],
}
_EXPECTED_TRANSFORMER_KEYS = 1933
_EXPECTED_TRANSFORMER_BYTES = 40_860_802_176
_EXPECTED_TEXT_ENCODER_KEYS = 729
_EXPECTED_TEXT_ENCODER_BYTES = 16_584_333_312


class FireRedCheckpointError(ValueError):
    """Raised before loading when a snapshot is not FireRed 1.1."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FireRedCheckpointError(f"Missing FireRed checkpoint file: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FireRedCheckpointError(f"Invalid JSON in FireRed checkpoint: {path}") from exc


@dataclass(frozen=True)
class FireRedCheckpointInfo:
    path: Path
    transformer_keys: int
    transformer_bytes: int
    text_encoder_keys: int
    text_encoder_bytes: int


def validate_firered_checkpoint(model_path: str | Path) -> FireRedCheckpointInfo:
    """Validate cheap architecture metadata before opening weight shards.

    A pinned downloader still needs to verify each downloaded LFS object's
    SHA-256.  This check prevents accidental use of another Qwen-family model
    after download and catches incomplete snapshots without reading 58 GB.
    """

    root = Path(model_path).expanduser().resolve()
    model_index = _read_json(root / "model_index.json")
    for component, expected in _EXPECTED_COMPONENTS.items():
        if model_index.get(component) != expected:
            raise FireRedCheckpointError(
                f"Unexpected {component} in {root / 'model_index.json'}: "
                f"{model_index.get(component)!r}, expected {expected!r}"
            )

    transformer = _read_json(root / "transformer" / "config.json")
    for field, expected in _EXPECTED_TRANSFORMER.items():
        if transformer.get(field) != expected:
            raise FireRedCheckpointError(
                f"Unexpected transformer {field}: {transformer.get(field)!r}, expected {expected!r}"
            )
    if transformer.get("zero_cond_t", False) is not False:
        raise FireRedCheckpointError("FireRed 1.1 requires zero_cond_t=false")

    transformer_index = _read_json(
        root / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    )
    text_index = _read_json(root / "text_encoder" / "model.safetensors.index.json")
    transformer_keys = len(transformer_index.get("weight_map", {}))
    transformer_bytes = transformer_index.get("metadata", {}).get("total_size")
    text_encoder_keys = len(text_index.get("weight_map", {}))
    text_encoder_bytes = text_index.get("metadata", {}).get("total_size")

    actual = (transformer_keys, transformer_bytes, text_encoder_keys, text_encoder_bytes)
    expected = (
        _EXPECTED_TRANSFORMER_KEYS,
        _EXPECTED_TRANSFORMER_BYTES,
        _EXPECTED_TEXT_ENCODER_KEYS,
        _EXPECTED_TEXT_ENCODER_BYTES,
    )
    if actual != expected:
        raise FireRedCheckpointError(f"Unexpected FireRed weight index metadata: {actual}, expected {expected}")

    for index, subdir in ((transformer_index, "transformer"), (text_index, "text_encoder")):
        missing = sorted(
            filename
            for filename in set(index["weight_map"].values())
            if not (root / subdir / filename).is_file()
        )
        if missing:
            raise FireRedCheckpointError(f"Missing {subdir} weight shards: {', '.join(missing)}")
    if not (root / "vae" / "diffusion_pytorch_model.safetensors").is_file():
        raise FireRedCheckpointError("Missing FireRed VAE weights")

    return FireRedCheckpointInfo(
        path=root,
        transformer_keys=transformer_keys,
        transformer_bytes=transformer_bytes,
        text_encoder_keys=text_encoder_keys,
        text_encoder_bytes=text_encoder_bytes,
    )


def create_firered_mlx(
    model_path: str | Path,
    *,
    quantize: int | None = 8,
    lora_paths: list[str] | None = None,
    lora_scales: list[float] | None = None,
):
    """Create a reusable FireRed 1.1 MLX pipeline.

    Keep the returned object alive across keyframes.  Reloading its 58 GB
    checkpoint for every image overwhelms any denoising optimization.  Eight
    bit transformer quantization is the quality-oriented default; MFLUX keeps
    the Qwen 2.5 VL encoder in bfloat16 because lower precision harms prompt
    semantics.
    """

    checkpoint = validate_firered_checkpoint(model_path)
    try:
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
    except ImportError as exc:
        raise RuntimeError(
            "FireRed MLX inference requires mflux with Qwen Image Edit support"
        ) from exc

    model_config = ModelConfig(
        priority=0,
        aliases=["firered-image-edit-1.1", "firered-1.1"],
        model_name=FIRERED_REPO_ID,
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=None,
        requires_sigma_shift=True,
        sigma_max_shift=0.9,
        sigma_max_seq_len=8192,
        sigma_shift_terminal=0.02,
    )
    return QwenImageEdit(
        quantize=quantize,
        model_path=str(checkpoint.path),
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        model_config=model_config,
    )
