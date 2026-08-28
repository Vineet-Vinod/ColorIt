"""Pinned, local MLX inference for FLUX.2 [klein] 4B image editing.

The official Black Forest Labs checkpoint is downloaded only from its immutable
Hugging Face revision.  MFLUX supplies the native MLX implementation; this
module keeps the model on device between calls and relies on MFLUX's compiled
denoiser path on M3-class Macs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from src.pipeline.weights import download_file, verify_file

FLUX2_KLEIN_4B_REPO_ID = "black-forest-labs/FLUX.2-klein-4B"
FLUX2_KLEIN_4B_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
MFLUX_VERSION = "0.19.1"
# Annotated upstream tag v.0.19.1.  uv.lock pins the corresponding PyPI wheel
# by SHA-256; retain the source revision for auditability.
MFLUX_GIT_REVISION = "e3d313ec2191f1960b3c9f3745299dcc58d2fdeb"


@dataclass(frozen=True)
class Flux2KleinFile:
    path: str
    size_bytes: int
    sha256: str

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/{FLUX2_KLEIN_4B_REPO_ID}/resolve/"
            f"{FLUX2_KLEIN_4B_REVISION}/{self.path}"
        )


# The 7.75 GB root single-file checkpoint is a Diffusers convenience file.  MFLUX
# consumes the component files below, so retaining it would duplicate the weights.
FLUX2_KLEIN_4B_FILES: tuple[Flux2KleinFile, ...] = (
    Flux2KleinFile("model_index.json", 446, "51a76cb1cf3ed37423a1128c79c22faee8e6fbe7f5aaeb737f0a258930dbaac0"),
    Flux2KleinFile("scheduler/scheduler_config.json", 486, "067afb012cef64553a763447d1efd93daeffcc0123ca7e25b09f8de20b90762e"),
    Flux2KleinFile("text_encoder/config.json", 1_536, "214b4c29a0d975e9fddf9994a5673f22cb2c4c5750352f9227c2c3251ebeab40"),
    Flux2KleinFile("text_encoder/generation_config.json", 214, "4347b1aeed2b2b78bc059920a0b7f5fec71482e1344952b76d7665d638d71f13"),
    Flux2KleinFile("text_encoder/model-00001-of-00002.safetensors", 4_967_215_360, "8c0506e7f4936fa7e26183a4fd8da4e2bdbc5990ba64ae441f965d51228f36ea"),
    Flux2KleinFile("text_encoder/model-00002-of-00002.safetensors", 3_077_766_632, "82f2bd839378541b0557bfabaf37c7d3d637071fdcb73302dedd7cf61162ce07"),
    Flux2KleinFile("text_encoder/model.safetensors.index.json", 32_855, "06b3d5319b6d76d1a4a2433419180016cfd54ed62d086a5e6567a809f8c82634"),
    Flux2KleinFile("tokenizer/added_tokens.json", 707, "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680"),
    Flux2KleinFile("tokenizer/chat_template.jinja", 4_168, "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"),
    Flux2KleinFile("tokenizer/merges.txt", 1_671_853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    Flux2KleinFile("tokenizer/special_tokens_map.json", 613, "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd"),
    Flux2KleinFile("tokenizer/tokenizer_config.json", 5_404, "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0"),
    Flux2KleinFile("tokenizer/vocab.json", 2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    Flux2KleinFile("transformer/config.json", 541, "09733c74a3da6d17dd0a0472a091a8950c7c6935889c32c16cc800ede05029de"),
    Flux2KleinFile("transformer/diffusion_pytorch_model.safetensors", 7_751_109_744, "9f29f9edcfdae452a653ffb51a534ca4decd389952c225724ff3b94042612a6e"),
    Flux2KleinFile("vae/config.json", 821, "0d6dfb69ae95a5e2ac9836284bbb63d8b38ce67b25ba2dff380752b2a10ab948"),
    Flux2KleinFile("vae/diffusion_pytorch_model.safetensors", 168_120_878, "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04"),
)


def flux2_klein_4b_model_dir(project_root: Path) -> Path:
    return project_root.expanduser().resolve() / "models" / "flux2-klein-4b"


def flux2_klein_4b_required_bytes() -> int:
    return sum(item.size_bytes for item in FLUX2_KLEIN_4B_FILES)


def verify_flux2_klein_4b_weights(model_dir: Path) -> None:
    """Verify every required local file before MFLUX opens a checkpoint."""
    root = model_dir.expanduser().resolve()
    for item in FLUX2_KLEIN_4B_FILES:
        candidate = root / item.path
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing FLUX.2 Klein 4B file: {candidate}")
        verify_file(
            candidate,
            expected_sha256=item.sha256,
            expected_size=item.size_bytes,
        )


def download_flux2_klein_4b_weights(model_dir: Path) -> None:
    """Download the MFLUX component layout from BFL's immutable HF revision."""
    root = model_dir.expanduser().resolve()
    for item in FLUX2_KLEIN_4B_FILES:
        destination = root / item.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            verify_file(
                destination,
                expected_sha256=item.sha256,
                expected_size=item.size_bytes,
            )
            continue
        download_file(
            item.url,
            destination,
            expected_sha256=item.sha256,
            expected_size=item.size_bytes,
        )


@dataclass(frozen=True)
class Flux2KleinMLXOptions:
    """Throughput-oriented FLUX.2 [klein] settings for reference-image editing."""

    quantize: int | None = 8
    steps: int = 4
    guidance: float = 1.0
    use_kv_cache: bool = False

    def __post_init__(self) -> None:
        if self.quantize not in (None, 4, 6, 8):
            raise ValueError("quantize must be one of None, 4, 6, or 8")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.guidance != 1.0:
            raise ValueError("The distilled FLUX.2 Klein 4B checkpoint requires guidance=1.0")
        if self.use_kv_cache:
            raise ValueError("FLUX.2 Klein 4B is not the KV-cache-tuned 9B variant")


class Flux2KleinMLXColorizer:
    """Reusable MFLUX resident model for image-conditioned color-reference creation."""

    def __init__(
        self,
        model_dir: Path,
        options: Flux2KleinMLXOptions | None = None,
        *,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self.options = options or Flux2KleinMLXOptions()
        self._model_factory = model_factory
        self._model: Any | None = None

    def warmup(self) -> None:
        """Load once before a batch so quantization and MLX compilation are amortized."""
        self._get_model()

    def generate(
        self,
        *,
        source_image: Path,
        reference_images: list[Path] | None = None,
        prompt: str,
        seed: int,
        width: int,
        height: int,
    ) -> Image.Image:
        if not prompt.strip():
            raise ValueError("FLUX.2 requires a non-empty editing prompt")
        if width < 64 or height < 64:
            raise ValueError("width and height must both be at least 64 pixels")

        source = source_image.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Source image not found: {source}")
        references = [path.expanduser().resolve() for path in (reference_images or [])]
        for reference in references:
            if not reference.is_file():
                raise FileNotFoundError(f"Reference image not found: {reference}")

        generated = self._get_model().generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=self.options.steps,
            width=width,
            height=height,
            guidance=self.options.guidance,
            image_paths=[source, *references],
            scheduler="flow_match_euler_discrete",
            use_kv_cache=self.options.use_kv_cache,
        )
        image = getattr(generated, "image", generated)
        if not isinstance(image, Image.Image):
            raise TypeError("MFLUX returned an unexpected FLUX.2 image result")
        return image.convert("RGB")

    def generate_batch(
        self,
        *,
        source_images: list[Path],
        prompt: str,
        seeds: list[int],
        width: int,
        height: int,
    ) -> list[Image.Image]:
        """Generate independent same-prompt edits in one MLX batch.

        MFLUX's public edit API treats a list of images as multiple references
        for one output. This path instead batches independent source images,
        sharing text encoding and compiled denoiser evaluations while retaining
        one conditioning image and seed per output.
        """

        if not source_images:
            raise ValueError("FLUX.2 batch requires at least one source image")
        if len(source_images) != len(seeds):
            raise ValueError("FLUX.2 batch requires one seed per source image")
        if not prompt.strip():
            raise ValueError("FLUX.2 requires a non-empty editing prompt")
        if width < 64 or height < 64:
            raise ValueError("width and height must both be at least 64 pixels")
        sources = [path.expanduser().resolve() for path in source_images]
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(f"Source image not found: {source}")
        return _generate_batch_mflux(
            self._get_model(),
            source_images=sources,
            prompt=prompt,
            seeds=seeds,
            width=width,
            height=height,
            steps=self.options.steps,
            guidance=self.options.guidance,
        )

    def _get_model(self) -> Any:
        if self._model is None:
            verify_flux2_klein_4b_weights(self.model_dir)
            factory = self._model_factory or _mflux_flux2_edit_factory()
            self._model = factory(
                model_path=str(self.model_dir),
                quantize=self.options.quantize,
            )
        return self._model


def _mflux_flux2_edit_factory() -> Callable[..., Any]:
    try:
        from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
    except ImportError as exc:
        raise RuntimeError(
            f"FLUX.2 MLX support requires mflux=={MFLUX_VERSION}. "
            "Install the project's image-edit extra before running this adapter."
        ) from exc
    return Flux2KleinEdit


def _generate_batch_mflux(
    model: Any,
    *,
    source_images: list[Path],
    prompt: str,
    seeds: list[int],
    width: int,
    height: int,
    steps: int,
    guidance: float,
) -> list[Image.Image]:
    """Native MLX batch implementation for pinned MFLUX 0.19.1."""

    try:
        from mflux.models.common.config.config import Config
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
        from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import (
            _Flux2KleinEditHelpers,
        )
        from mflux.utils.image_util import ImageUtil
    except ImportError as exc:  # pragma: no cover - guarded by optional extra.
        raise RuntimeError(
            f"FLUX.2 MLX batching requires mflux=={MFLUX_VERSION}"
        ) from exc

    batch_size = len(source_images)
    config = Config(
        model_config=model.model_config,
        num_inference_steps=steps,
        height=height,
        width=width,
        guidance=guidance,
        scheduler="flow_match_euler_discrete",
    )
    prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids = (
        model._encode_prompt_pair(
            prompt=prompt,
            negative_prompt=" ",
            guidance=guidance,
        )
    )

    def broadcast_batch(array: Any | None) -> Any | None:
        if array is None or array.shape[0] == batch_size:
            return array
        if array.shape[0] != 1:
            raise ValueError("FLUX.2 conditioning batch cannot be broadcast")
        return mx.broadcast_to(array, (batch_size, *array.shape[1:]))

    prompt_embeds = broadcast_batch(prompt_embeds)
    text_ids = broadcast_batch(text_ids)
    negative_prompt_embeds = broadcast_batch(negative_prompt_embeds)
    negative_text_ids = broadcast_batch(negative_text_ids)

    latent_rows = []
    latent_ids = None
    latent_height = 0
    latent_width = 0
    for seed in seeds:
        row, row_ids, latent_height, latent_width = Flux2LatentCreator.prepare_packed_latents(
            seed=seed,
            height=config.height,
            width=config.width,
            batch_size=1,
        )
        latent_rows.append(row)
        if latent_ids is None:
            latent_ids = row_ids
    latents = mx.concatenate(latent_rows, axis=0)
    assert latent_ids is not None
    latent_ids = mx.broadcast_to(latent_ids, (batch_size, *latent_ids.shape[1:]))

    prepared_images = [
        _Flux2KleinEditHelpers.prepare_reference_image(ImageUtil.load_image(path))
        for path in source_images
    ]
    reference_sizes = {(image.width, image.height) for image in prepared_images}
    if len(reference_sizes) != 1:
        raise ValueError("FLUX.2 batch source images must resolve to one reference size")
    reference_arrays = mx.concatenate([ImageUtil.to_array(image) for image in prepared_images], axis=0)
    encoded = VAEUtil.encode(
        vae=model.vae,
        image=reference_arrays,
        tiling_config=model.tiling_config,
    )
    encoded = _Flux2KleinEditHelpers.ensure_4d_latents(encoded)
    encoded = _Flux2KleinEditHelpers.crop_to_even_spatial(encoded)
    encoded = Flux2LatentCreator.patchify_latents(encoded)
    encoded = _Flux2KleinEditHelpers.bn_normalize_vae_encoded_latents(encoded, vae=model.vae)
    image_latents = Flux2LatentCreator.pack_latents(encoded)
    image_latent_ids = Flux2LatentCreator.prepare_grid_ids(encoded, t_coord=10)

    predict = model._predict(model.transformer)
    for timestep in config.time_steps:
        noise = predict(
            latents=latents,
            image_latents=image_latents,
            latent_ids=latent_ids,
            image_latent_ids=image_latent_ids,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            guidance=guidance,
            timestep=config.scheduler.timesteps[timestep],
        )
        latents = config.scheduler.step(
            noise=noise,
            timestep=timestep,
            latents=latents,
            sigmas=config.scheduler.sigmas,
        )
        mx.eval(latents)

    packed = latents.reshape(
        batch_size,
        latent_height,
        latent_width,
        latents.shape[-1],
    ).transpose(0, 3, 1, 2)
    decoded = model.vae.decode_packed_latents(packed)
    mx.eval(decoded)
    decoded_array = np.asarray(decoded.astype(mx.float32))
    if decoded_array.ndim == 5 and decoded_array.shape[2] == 1:
        decoded_array = np.squeeze(decoded_array, axis=2)
    decoded_array = np.transpose(decoded_array, (0, 2, 3, 1))
    decoded_array = np.clip(decoded_array / 2.0 + 0.5, 0.0, 1.0)
    return [
        Image.fromarray(np.rint(image * 255.0).astype(np.uint8))
        for image in decoded_array
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or verify the local FLUX.2 Klein 4B MLX adapter.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Safely download the pinned official checkpoint.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--quantize", type=int, choices=(4, 6, 8), default=8)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.download:
        download_flux2_klein_4b_weights(args.model_dir)
    verify_flux2_klein_4b_weights(args.model_dir)
    if args.input is None:
        print(f"Verified {len(FLUX2_KLEIN_4B_FILES)} FLUX.2 Klein 4B files.")
        return 0
    if args.output is None or args.prompt is None:
        raise SystemExit("--input requires --output and --prompt")

    colorizer = Flux2KleinMLXColorizer(
        args.model_dir,
        Flux2KleinMLXOptions(quantize=args.quantize),
    )
    colorizer.warmup()
    image = colorizer.generate(
        source_image=args.input,
        prompt=args.prompt,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
