"""Modern MLX image editors used as DeepRemaster palette references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EDIT_KEYFRAME_MODELS = (
    "flux2_klein_4b",
)

DEFAULT_COLORIZE_PROMPT = (
    "Colorize this black-and-white archival film frame. Preserve its people, faces, "
    "costumes, objects, composition, lighting, and period photographic realism. "
    "Use believable material colors and natural skin tones. Do not crop, redraw, "
    "add, remove, or restyle anything."
)

DEFAULT_PALETTE_PROMPT = (
    "Colorize the first black-and-white archival film frame using the exact costume, "
    "skin, object, foliage, architecture, and sky palette from the second colored frame. "
    "The images are from the same scene. Preserve the first frame's people, faces, poses, "
    "objects, composition, and lighting. Do not copy the second frame's geometry."
)


@dataclass(frozen=True)
class ImageEditKeyframeOptions:
    width: int = 1024
    height: int = 576
    quantize: int = 8
    seed: int = 101
    flux_steps: int = 4
    flux_batch_size: int = 4
    qwen_steps: int = 20
    firered_steps: int = 20
    qwen_guidance: float = 2.5
    prompt: str = DEFAULT_COLORIZE_PROMPT
    palette_anchor: bool = False
    palette_prompt: str = DEFAULT_PALETTE_PROMPT
    align_reference_luma: bool = True

    @classmethod
    def from_settings(cls, settings: object) -> "ImageEditKeyframeOptions":
        raw = settings if isinstance(settings, dict) else {}
        options = cls(
            width=int(raw.get("width", 1024)),
            height=int(raw.get("height", 576)),
            quantize=int(raw.get("quantize", 8)),
            seed=int(raw.get("seed", 101)),
            flux_steps=int(raw.get("flux_steps", 4)),
            flux_batch_size=int(raw.get("flux_batch_size", 4)),
            qwen_steps=int(raw.get("qwen_steps", 20)),
            firered_steps=int(raw.get("firered_steps", 20)),
            qwen_guidance=float(raw.get("qwen_guidance", 2.5)),
            prompt=str(raw.get("prompt", DEFAULT_COLORIZE_PROMPT)),
            palette_anchor=bool(raw.get("palette_anchor", False)),
            palette_prompt=str(raw.get("palette_prompt", DEFAULT_PALETTE_PROMPT)),
            align_reference_luma=bool(raw.get("align_reference_luma", True)),
        )
        options.validate()
        return options

    def validate(self) -> None:
        for name, value in (("width", self.width), ("height", self.height)):
            if value < 64 or value % 16:
                raise ValueError(f"DeepRemaster image_edit.{name} must be a multiple of 16 and at least 64")
        if self.quantize not in {4, 6, 8}:
            raise ValueError("DeepRemaster image_edit.quantize must be 4, 6, or 8")
        if self.seed < 0:
            raise ValueError("DeepRemaster image_edit.seed must be non-negative")
        if min(self.flux_steps, self.qwen_steps, self.firered_steps) < 1:
            raise ValueError("DeepRemaster image editor step counts must be positive")
        if self.flux_batch_size < 1:
            raise ValueError("DeepRemaster image_edit.flux_batch_size must be positive")
        if self.qwen_guidance < 0:
            raise ValueError("DeepRemaster image_edit.qwen_guidance must be non-negative")
        if not self.prompt.strip() or not self.palette_prompt.strip():
            raise ValueError("DeepRemaster image edit prompts must not be empty")


class ImageEditKeyframeColorizer:
    """Hold one verified MLX editor resident across all scene references."""

    def __init__(self, *, model: str, model_root: Path, options: ImageEditKeyframeOptions):
        if model not in IMAGE_EDIT_KEYFRAME_MODELS:
            raise ValueError(f"Unsupported MLX image editor: {model}")
        self.model_name = model
        self.model_root = model_root.expanduser().resolve()
        self.options = options
        self._runner: Any | None = None

    def colorize(self, source_paths: list[Path], colored_paths: list[Path]) -> None:
        if len(source_paths) != len(colored_paths):
            raise ValueError("Source and output keyframe counts must match")
        if not source_paths:
            return
        if self.model_name == "flux2_klein_4b" and self.options.palette_anchor and len(source_paths) > 1:
            self._colorize_flux_palette_bank(source_paths, colored_paths)
            return
        if self.model_name == "flux2_klein_4b":
            self._colorize_flux_batches(source_paths, colored_paths)
            return
        for source, output in zip(source_paths, colored_paths, strict=True):
            self._generate_one(source=source, output=output, prompt=self.options.prompt)

    def _colorize_flux_batches(self, source_paths: list[Path], colored_paths: list[Path]) -> None:
        runner = self._get_runner()
        batch_size = self.options.flux_batch_size
        for offset in range(0, len(source_paths), batch_size):
            sources = source_paths[offset : offset + batch_size]
            outputs = colored_paths[offset : offset + batch_size]
            images = runner.generate_batch(
                source_images=sources,
                prompt=self.options.prompt,
                seeds=[self.options.seed] * len(sources),
                width=self.options.width,
                height=self.options.height,
            )
            if len(images) != len(outputs):
                raise RuntimeError("FLUX.2 batch returned an unexpected image count")
            for image, output in zip(images, outputs, strict=True):
                if not isinstance(image, Image.Image):
                    raise TypeError("flux2_klein_4b returned a non-image keyframe")
                output.parent.mkdir(parents=True, exist_ok=True)
                image.convert("RGB").save(output)

    def _colorize_flux_palette_bank(self, source_paths: list[Path], colored_paths: list[Path]) -> None:
        anchor_index = len(source_paths) // 2
        self._generate_one(
            source=source_paths[anchor_index],
            output=colored_paths[anchor_index],
            prompt=self.options.prompt,
        )
        anchor = colored_paths[anchor_index]
        for index, (source, output) in enumerate(zip(source_paths, colored_paths, strict=True)):
            if index == anchor_index:
                continue
            self._generate_one(
                source=source,
                output=output,
                prompt=self.options.palette_prompt,
                reference_images=[anchor],
            )

    def _generate_one(
        self,
        *,
        source: Path,
        output: Path,
        prompt: str,
        reference_images: list[Path] | None = None,
    ) -> None:
        runner = self._get_runner()
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.model_name == "flux2_klein_4b":
            image = runner.generate(
                source_image=source,
                reference_images=reference_images,
                prompt=prompt,
                seed=self.options.seed,
                width=self.options.width,
                height=self.options.height,
            )
        elif self.model_name == "qwen_image_edit_2511":
            result = runner.generate_batch(source, prompt, seeds=[self.options.seed])[0]
            image = getattr(result, "image", result)
        else:
            result = runner.generate_batch(
                image_path=str(source),
                prompt=prompt,
                negative_prompt="",
                seeds=[self.options.seed],
                width=self.options.width,
                height=self.options.height,
                steps=self.options.firered_steps,
                guidance=self.options.qwen_guidance,
                scheduler="linear",
            )[0]
            image = getattr(result, "image", result)
        if not isinstance(image, Image.Image):
            raise TypeError(f"{self.model_name} returned a non-image keyframe")
        image.convert("RGB").save(output)

    def _get_runner(self):
        if self._runner is not None:
            return self._runner
        if self.model_name == "flux2_klein_4b":
            from src.pipeline.flux2_klein_mlx import Flux2KleinMLXColorizer, Flux2KleinMLXOptions

            self._runner = Flux2KleinMLXColorizer(
                self.model_root / "flux2-klein-4b",
                Flux2KleinMLXOptions(quantize=self.options.quantize, steps=self.options.flux_steps),
            )
            self._runner.warmup()
        elif self.model_name == "qwen_image_edit_2511":
            from src.vendor.qwen_mlx.adapter import QwenEditConfig, QwenImageEditRunner

            self._runner = QwenImageEditRunner(
                QwenEditConfig(
                    model_dir=self.model_root / "qwen-image-edit-2511",
                    seed=self.options.seed,
                    steps=self.options.qwen_steps,
                    guidance=self.options.qwen_guidance,
                    quantize=self.options.quantize,
                    width=self.options.width,
                    height=self.options.height,
                )
            )
        else:
            from src.pipeline.firered_mlx import create_firered_mlx
            from src.vendor.qwen_mlx.optimized import QwenCompiledEditLoop

            model = create_firered_mlx(
                self.model_root / "firered-image-edit-1.1",
                quantize=self.options.quantize,
            )
            self._runner = QwenCompiledEditLoop(model)
        return self._runner


def keyframe_model_fingerprint(model: str) -> str:
    """Stable model provenance included in resume/run identities."""

    if model == "flux2_klein_4b":
        from src.pipeline.flux2_klein_mlx import FLUX2_KLEIN_4B_REVISION, MFLUX_VERSION

        return f"{FLUX2_KLEIN_4B_REVISION}:mflux-{MFLUX_VERSION}"
    if model == "qwen_image_edit_2511":
        from src.vendor.qwen_mlx.adapter import MFLUX_VERSION, QWEN_IMAGE_EDIT_2511

        return f"{QWEN_IMAGE_EDIT_2511['revision']}:mflux-{MFLUX_VERSION}:zero-cond-t"
    if model == "firered_image_edit_1_1":
        from src.pipeline.firered_mlx import FIRERED_REVISION
        from src.vendor.qwen_mlx.adapter import MFLUX_VERSION

        return f"{FIRERED_REVISION}:mflux-{MFLUX_VERSION}"
    return f"legacy-{model}-v1"
