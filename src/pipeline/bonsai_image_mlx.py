"""Prism ML Bonsai Image inference with reference-conditioned FLUX.2 editing.

Prism's public demo constructs the packed transformer only for text-to-image.
This adapter uses the same immutable packed checkpoint and native low-bit MLX
kernel in MFLUX's reference-conditioned ``Flux2KleinEdit`` path, so comparisons
against dense FLUX.2 use identical conditioning rather than unrelated prompts.

The 1-bit kernel requires Prism's MLX fork. Keep that runtime in an isolated
environment; importing this module alone does not import or modify MLX/MFLUX.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from src.pipeline.image_edit_weights import IMAGE_EDIT_MODELS, verify_model_snapshot


PRISM_IMAGE_STUDIO_REVISION = "31b02171634c16b5da0eec6aea075e7489d5fb39"
PRISM_MFLUX_REVISION = "bcd13e83b7fcfd76186c98ef322dd9cf28e996c1"
PRISM_MLX_REVISION = "b9effaf65ecce0e99a2fb23c8962d34eff95b625"

BonsaiVariant = Literal["binary", "ternary"]
BonsaiConditioning = Literal["edit", "img2img"]
_MODEL_NAMES = {
    "binary": "bonsai_image_binary_4b_mlx_1bit",
    "ternary": "bonsai_image_ternary_4b_mlx_2bit",
}
_PRECISIONS = {"binary": "1bit", "ternary": "2bit"}


@dataclass(frozen=True)
class BonsaiImageOptions:
    variant: BonsaiVariant
    conditioning: BonsaiConditioning = "edit"
    image_strength: float = 0.5
    steps: int = 4
    guidance: float = 1.0

    def __post_init__(self) -> None:
        if self.variant not in _MODEL_NAMES:
            raise ValueError("variant must be 'binary' or 'ternary'")
        if self.conditioning not in ("edit", "img2img"):
            raise ValueError("conditioning must be 'edit' or 'img2img'")
        if not 0.0 < self.image_strength <= 1.0:
            raise ValueError("image_strength must be in (0, 1]")
        if self.steps != 4:
            raise ValueError("Bonsai Image is distilled for exactly four denoising steps")
        if self.guidance != 1.0:
            raise ValueError("Bonsai Image requires guidance=1.0")


class BonsaiImageMLXColorizer:
    """Reusable packed Bonsai model using FLUX.2 edit conditioning."""

    def __init__(
        self,
        model_dir: Path,
        options: BonsaiImageOptions,
        *,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self.options = options
        self._model_factory = model_factory
        self._model: Any | None = None

    def warmup(self) -> None:
        self._get_model()

    @property
    def kernel_mode(self) -> str:
        model = self._get_model()
        return str(getattr(model, "_colorit_kernel_mode", "prism_native"))

    def generate(
        self,
        *,
        source_image: Path,
        prompt: str,
        seed: int,
        width: int,
        height: int,
    ) -> Image.Image:
        if not prompt.strip():
            raise ValueError("Bonsai Image requires a non-empty edit prompt")
        if width < 64 or height < 64 or width % 16 or height % 16:
            raise ValueError("width and height must be multiples of 16 and at least 64")
        source = source_image.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Source image not found: {source}")

        model = self._get_model()
        # Prism's older edit wrapper does not perform the lazy transition that
        # its text-to-image wrapper does. Load once here and retain both parts.
        loader = getattr(model, "load_transformer_and_vae", None)
        if callable(loader):
            loader()
        common = {
            "seed": seed,
            "prompt": prompt,
            "num_inference_steps": self.options.steps,
            "width": width,
            "height": height,
            "guidance": self.options.guidance,
            "scheduler": "flow_match_euler_discrete",
        }
        if self.options.conditioning == "edit":
            generated = model.generate_image(**common, image_paths=[source])
        else:
            generated = model.generate_image(
                **common,
                image_path=source,
                image_strength=self.options.image_strength,
            )
        image = getattr(generated, "image", generated)
        if not isinstance(image, Image.Image):
            raise TypeError("Prism MFLUX returned an unexpected image result")
        return image.convert("RGB")

    def _get_model(self) -> Any:
        if self._model is None:
            record = IMAGE_EDIT_MODELS[_MODEL_NAMES[self.options.variant]]
            verify_model_snapshot(record, self.model_dir)
            factory = self._model_factory or _prism_model_factory(self.options.conditioning)
            self._model = factory(
                model_path=str(self.model_dir),
                precision=_PRECISIONS[self.options.variant],
            )
        return self._model


def _prism_model_factory(conditioning: BonsaiConditioning = "edit") -> Callable[..., Any]:
    """Build a packed-transformer edit model inside Prism's isolated runtime."""

    try:
        import gc

        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_unflatten
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.common.resolution.path_resolution import PathResolution
        from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
        from mflux.models.common.weights.loading.weight_applier import WeightApplier
        from mflux.models.common.weights.loading.weight_loader import WeightLoader
        from mflux.models.flux2.flux2_initializer import FULL_DECODER_CHANNELS, Flux2Initializer
        from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
        from mflux.models.flux2.model.flux2_transformer.klein_fast import blocks as fast_blocks
        from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
        from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
    except ImportError as exc:
        raise RuntimeError(
            "Bonsai Image requires the isolated Prism MLX/MFLUX runtime pinned "
            f"to {PRISM_MLX_REVISION[:8]} and {PRISM_MFLUX_REVISION[:8]}."
        ) from exc

    def install_one_bit_fallback() -> bool:
        """Use cached MLX dequantization when Prism's native 1-bit op is absent.

        The fallback is numerically the same affine 1-bit checkpoint. It keeps
        dequantized bfloat16 matrices resident after their first use, trading
        memory for much faster subsequent denoising steps. This is only needed
        on machines without a build of Prism's MLX fork.
        """

        try:
            fast_blocks._require_native_quantized_matmul(1, 128)
            return False
        except RuntimeError:
            pass

        original_require = fast_blocks._require_native_quantized_matmul
        original_call = fast_blocks.QuantizedLinearKernel.__call__

        def require_native(bits: int, group_size: int) -> None:
            if bits == 1:
                return
            original_require(bits, group_size)

        def low_bit_call(kernel: Any, value: Any) -> Any:
            if kernel.bits != 1:
                return original_call(kernel, value)
            dense = getattr(kernel, "_colorit_dense_weight", None)
            if dense is None:
                shifts = mx.arange(32, dtype=mx.uint32)
                unpacked = ((kernel.packed_weight[..., None] >> shifts) & 1).reshape(
                    kernel.packed_weight.shape[0], -1
                )
                scales = mx.repeat(kernel.scales, kernel.group_size, axis=1)
                biases = mx.repeat(kernel.biases, kernel.group_size, axis=1)
                dense = (unpacked.astype(mx.bfloat16) * scales + biases).astype(mx.bfloat16)
                mx.eval(dense)
                kernel._colorit_dense_weight = dense
            original_shape = value.shape
            flat = value.reshape((-1, value.shape[-1])).astype(mx.bfloat16)
            output = flat @ dense.transpose()
            return output.reshape((*original_shape[:-1], dense.shape[0]))

        fast_blocks._require_native_quantized_matmul = require_native
        fast_blocks.QuantizedLinearKernel.__call__ = low_bit_call
        return True

    binary_fallback = install_one_bit_fallback()

    def load_text_encoder(model: Any) -> None:
        root = Path(model._model_path) / "text_encoder-mlx-4bit"
        path = root / "model.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundled Bonsai text encoder: {path}")
        raw = mx.load(str(path))
        stripped = {key[len("model.") :]: value for key, value in raw.items() if key.startswith("model.")}
        text_encoder = Qwen3TextEncoder(**model.model_config.text_encoder_overrides)
        nn.quantize(
            text_encoder,
            class_predicate=lambda _path, layer: hasattr(layer, "to_quantized"),
            bits=4,
            group_size=64,
        )
        text_encoder.update(tree_unflatten(list(stripped.items())))
        model.text_encoder = text_encoder

    def load_transformer_and_vae(model: Any) -> None:
        if model.vae is None:
            model.vae = Flux2VAE(decoder_block_out_channels=FULL_DECODER_CHANNELS)
            root = PathResolution.resolve(
                path=model._model_path,
                patterns=Flux2KleinWeightDefinition.get_download_patterns(),
            )
            if root is None:
                raise FileNotFoundError(f"Cannot resolve Bonsai checkpoint: {model._model_path}")
            vae_component = next(
                component
                for component in Flux2KleinWeightDefinition.get_components()
                if component.name == "vae"
            )
            vae_weights, _, _ = WeightLoader._load_component(root, vae_component)
            WeightApplier.apply_and_quantize(
                weights=LoadedWeights(
                    components={"vae": vae_weights},
                    meta_data=MetaData(quantization_level=None, mflux_version=None),
                ),
                quantize_arg=None,
                weight_definition=Flux2KleinWeightDefinition,
                models={"vae": model.vae},
            )
        if model.transformer is None:
            Flux2Initializer._load_klein_fast_transformer_weights(
                model,
                model._model_path,
                precision=model._klein_fast_precision,
            )
        gc.collect()
        mx.clear_cache()

    Flux2Initializer.reload_text_encoder = staticmethod(load_text_encoder)

    class PackedFlux2KleinEdit(Flux2KleinEdit):
        def __init__(self, *, model_path: str, precision: str) -> None:
            nn.Module.__init__(self)
            Flux2Initializer.init(
                model=self,
                quantize=None,
                model_path=model_path,
                model_config=ModelConfig.flux2_klein_4b(),
                use_klein_fast_transformer=True,
                klein_fast_precision=precision,
                vae_variant="full",
                evict_text_encoder=False,
                lazy_components=False,
                bucketed_seq_len=False,
            )
            self._colorit_kernel_mode = (
                "mlx_cached_bfloat16_fallback"
                if precision == "1bit" and binary_fallback
                else "prism_native_packed"
            )

        def load_transformer_and_vae(self) -> None:
            load_transformer_and_vae(self)

    Flux2Initializer.load_transformer_and_vae = staticmethod(load_transformer_and_vae)

    if conditioning == "edit":
        return PackedFlux2KleinEdit

    def packed_img2img(*, model_path: str, precision: str) -> Any:
        model = Flux2Klein(
            model_path=model_path,
            quantize=None,
            use_klein_fast_transformer=True,
            klein_fast_precision=precision,
            vae_variant="full",
            evict_text_encoder=False,
            lazy_components=False,
            bucketed_seq_len=False,
        )
        model._colorit_kernel_mode = (
            "mlx_cached_bfloat16_fallback"
            if precision == "1bit" and binary_fallback
            else "prism_native_packed"
        )
        return model

    return packed_img2img


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run verified Bonsai Image MLX editing.")
    parser.add_argument("--variant", choices=tuple(_MODEL_NAMES), required=True)
    parser.add_argument("--conditioning", choices=("edit", "img2img"), default="edit")
    parser.add_argument("--image-strength", type=float, default=0.5)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--metrics", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    colorizer = BonsaiImageMLXColorizer(
        args.model_dir,
        BonsaiImageOptions(
            variant=args.variant,
            conditioning=args.conditioning,
            image_strength=args.image_strength,
        ),
    )
    started = time.perf_counter()
    colorizer.warmup()
    loaded = time.perf_counter()
    image = colorizer.generate(
        source_image=args.input,
        prompt=args.prompt,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )
    completed = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(
            json.dumps(
                {
                    "variant": args.variant,
                    "load_seconds": loaded - started,
                    "inference_seconds": completed - loaded,
                    "total_seconds": completed - started,
                    "seed": args.seed,
                    "width": args.width,
                    "height": args.height,
                    "steps": 4,
                    "guidance": 1.0,
                    "kernel_mode": colorizer.kernel_mode,
                    "conditioning": args.conditioning,
                    "image_strength": args.image_strength,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
