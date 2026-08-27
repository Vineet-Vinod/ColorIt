"""Local, reproducible Qwen Image Edit inference through MLX and mflux.

The model stays outside the Python package.  This module pins both the official
Hugging Face revision and the mflux release so a run cannot silently change when
either project's default moves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
from typing import Any

from src.pipeline.image_edit_weights import (
    IMAGE_EDIT_MODELS,
    download_model_snapshot,
    verify_model_snapshot,
)


MFLUX_VERSION = "0.19.1"
_MODEL_NAME = "qwen_image_edit_2511"
_MODEL = IMAGE_EDIT_MODELS[_MODEL_NAME]
QWEN_IMAGE_EDIT_2511: dict[str, Any] = {
    "repo_id": _MODEL.repo_id,
    "revision": _MODEL.revision,
    "license": _MODEL.license,
    "total_bytes": 57_720_463_453,
    "source": "Official Qwen organization on Hugging Face",
}
DEFAULT_MODEL_DIR = Path("models/image-edit/qwen-image-edit-2511")


class QwenMlxError(RuntimeError):
    """Raised when the pinned model or runner cannot be validated."""


@dataclass(frozen=True)
class QwenEditConfig:
    """Settings for a reproducible edit run.

    Qwen Image Edit has no graph-compile switch in mflux 0.19.1.  Keeping one
    runner alive across keyframes therefore gives the useful throughput win:
    model weights, tokenizers, and MLX allocations stay resident.
    """

    model_dir: Path = DEFAULT_MODEL_DIR
    seed: int = 42
    steps: int = 20
    guidance: float = 2.5
    quantize: int | None = 8
    width: int | None = None
    height: int | None = None
    mlx_cache_limit_gb: int | None = None
    scheduler: str = "linear"
    cfg_mode: str = "auto"
    fused_cfg_max_batch: int = 1

    def __post_init__(self) -> None:
        if self.quantize not in {None, 3, 4, 5, 6, 8}:
            raise ValueError("quantize must be one of None, 3, 4, 5, 6, or 8")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.guidance < 0:
            raise ValueError("guidance must be non-negative")
        if self.mlx_cache_limit_gb is not None and self.mlx_cache_limit_gb < 1:
            raise ValueError("mlx_cache_limit_gb must be at least 1")
        if self.cfg_mode not in {"auto", "fused", "separate"}:
            raise ValueError("cfg_mode must be auto, fused, or separate")
        if self.fused_cfg_max_batch < 1:
            raise ValueError("fused_cfg_max_batch must be positive")
        for name, value in (("width", self.width), ("height", self.height)):
            if value is not None and (value < 16 or value % 16):
                raise ValueError(f"{name} must be a multiple of 16 and at least 16")

    def metadata(self) -> dict[str, Any]:
        result = asdict(self)
        result["model_dir"] = str(self.model_dir)
        result["model"] = QWEN_IMAGE_EDIT_2511
        result["mflux_version"] = MFLUX_VERSION
        result["compile"] = True
        result["compile_note"] = "ColorIt's Qwen loop compiles fixed-shape CFG transformer calls."
        return result


def validate_official_weights(model_dir: Path = DEFAULT_MODEL_DIR) -> dict[str, Any]:
    """Verify with ColorIt's shared immutable, allowlisted weight registry."""

    model_dir = Path(model_dir).resolve()
    try:
        verify_model_snapshot(_MODEL, model_dir)
    except ValueError as exc:
        raise QwenMlxError(str(exc)) from exc
    return {
        "model_dir": str(model_dir),
        "revision": _MODEL.revision,
        "verified_files": len(_MODEL.files),
        "verified_weight_bytes": sum(item.size_bytes or 0 for item in _MODEL.files),
    }


def download_official_weights(model_dir: Path = DEFAULT_MODEL_DIR) -> dict[str, Any]:
    """Download the immutable official revision, then verify every weight hash."""

    model_dir = model_dir.resolve()
    if model_dir.name != _MODEL.destination_name:
        raise QwenMlxError(
            f"model_dir must end with {_MODEL.destination_name!r} to use the shared registry"
        )
    try:
        manifest = download_model_snapshot(_MODEL_NAME, model_dir.parent)
    except ValueError as exc:
        raise QwenMlxError(str(exc)) from exc
    return {
        "model_dir": str(model_dir),
        "revision": _MODEL.revision,
        "manifest": manifest,
    }


def _require_mflux() -> None:
    try:
        installed = importlib.metadata.version("mflux")
    except importlib.metadata.PackageNotFoundError as error:
        raise QwenMlxError(
            f"mflux=={MFLUX_VERSION} is required. Run with uvx or install it in the inference environment."
        ) from error
    if installed != MFLUX_VERSION:
        raise QwenMlxError(f"mflux {installed} is installed; this adapter requires {MFLUX_VERSION}")


class QwenImageEditRunner:
    """Reusable in-process Qwen Image Edit runner for a batch of keyframes."""

    def __init__(self, config: QwenEditConfig = QwenEditConfig()) -> None:
        self.config = config
        _require_mflux()
        if not self.config.model_dir.is_dir():
            raise QwenMlxError(f"model directory is missing: {self.config.model_dir}")
        # Safetensors avoid pickle execution, but an immutable revision alone
        # does not prove the local files still match it. Verify the complete
        # allowlist before MFLUX opens any shard.
        validate_official_weights(self.config.model_dir)
        import mlx.core as mx
        from mflux.models.common.config import ModelConfig
        from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
        from .qwen_2511 import enable_zero_cond_t
        from .optimized import QwenCompiledEditLoop

        if self.config.mlx_cache_limit_gb is not None:
            mx.metal.set_cache_limit(self.config.mlx_cache_limit_gb * 1024**3)
        self._mx = mx
        self._model = QwenImageEdit(
            quantize=self.config.quantize,
            model_path=str(self.config.model_dir),
            model_config=ModelConfig.qwen_image_edit(),
        )
        enable_zero_cond_t(self._model)
        self.compiled_loop = QwenCompiledEditLoop(
            self._model,
            cfg_mode=self.config.cfg_mode,
            fused_cfg_max_batch=self.config.fused_cfg_max_batch,
        )

    def generate_batch(
        self,
        image_path: Path,
        prompt: str,
        *,
        negative_prompt: str = "",
        seeds: list[int] | None = None,
    ) -> list[object]:
        """Generate every seed with one prepared source image and prompt."""

        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        use_seeds = [self.config.seed] if seeds is None else list(seeds)
        if not use_seeds:
            raise ValueError("seeds must not be empty")
        images = self.compiled_loop.generate_batch(
            image_path=str(image_path),
            prompt=prompt,
            negative_prompt=negative_prompt,
            seeds=use_seeds,
            width=self.config.width,
            height=self.config.height,
            steps=self.config.steps,
            guidance=self.config.guidance,
            scheduler=self.config.scheduler,
        )
        self._mx.eval(self._mx.array(0))
        return images

    def edit_batch(
        self,
        image_path: Path,
        prompt: str,
        output_paths: list[Path],
        *,
        negative_prompt: str = "",
        seeds: list[int] | None = None,
    ) -> list[Path]:
        """Write one output per seed through the compiled batch path."""

        use_seeds = [self.config.seed] if seeds is None else list(seeds)
        if len(output_paths) != len(use_seeds):
            raise ValueError("output_paths and seeds must have the same length")
        generated = self.generate_batch(
            image_path,
            prompt,
            negative_prompt=negative_prompt,
            seeds=use_seeds,
        )
        resolved = [path.resolve() for path in output_paths]
        for image, output_path in zip(generated, resolved, strict=True):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(output_path), export_json_metadata=True)
        return resolved

    def edit(
        self,
        image_path: Path,
        prompt: str,
        output_path: Path,
        *,
        negative_prompt: str = "",
        seed: int | None = None,
    ) -> Path:
        """Color one image through the same compiled batch path used by sweeps."""

        return self.edit_batch(
            image_path,
            prompt,
            [output_path],
            negative_prompt=negative_prompt,
            seeds=[self.config.seed if seed is None else seed],
        )[0]


def smoke_check() -> dict[str, str]:
    """Verify the pinned mflux CLI without downloading or loading the 58 GB model."""

    uvx = shutil.which("uvx")
    if uvx is None:
        raise QwenMlxError("uvx is required for the isolated mflux CLI smoke test")
    command = [uvx, "--from", f"mflux=={MFLUX_VERSION}", "mflux-generate-qwen-edit", "--help"]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or "Qwen Image Edit" not in completed.stdout:
        raise QwenMlxError(completed.stderr.strip() or "mflux Qwen CLI smoke test failed")
    return {"mflux": MFLUX_VERSION, "entrypoint": "mflux-generate-qwen-edit"}
