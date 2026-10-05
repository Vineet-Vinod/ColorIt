from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    import torch

DEFAULT_PROMPT = (
    "Add color to this black-and-white image without hesitation regarding the appropriate colors. "
    "For any subject, garment, object, or setting where the color is common knowledge or established "
    "by convention, confidently apply the expected color; otherwise use natural colors. "
    "Color the image by strictly preserving all shapes, outlines, and background details."
)


class ReferenceRecord(BaseModel):
    frame: int
    path: Path
    source_path: Path
    model_input_path: Path


class InputReference(BaseModel):
    frame: int
    path: Path
    source_path: Path | None = None
    prompt: str | None = None
    seed: int = 42


def cached_dequantizer(
    original: Callable[[torch.Tensor | None, torch.dtype], torch.Tensor | None],
) -> Callable[[torch.Tensor | None, torch.dtype], torch.Tensor | None]:
    cached_key: tuple[int, torch.dtype, torch.device] | None = None
    cached_value: torch.Tensor | None = None

    def get_weight(tensor: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
        nonlocal cached_key, cached_value
        if tensor is None:
            return original(tensor, dtype)
        key = (tensor.data_ptr(), dtype, tensor.device)
        if key != cached_key:
            cached_value = None
            cached_value = original(tensor, dtype)
            cached_key = key
        return cached_value

    return get_weight


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Author HAVC Qwen2.1/Viggle recipe; BF16 UNet and CPU FP32 VAE."
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frame", type=int)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help="JSON array of {frame, path} raw reference records.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Author FastPipeline: pair refs side by side at1280/gap16.",
    )
    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path("tmp/adapter_compare_90_120/upstream/HAVCServerDiT"),
    )
    parser.add_argument(
        "--models", type=Path, default=Path("data/adapter_compare_90_120/havcdit")
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, choices=[2, 4, 6, 8], default=6)
    parser.add_argument("--backend", choices=["mps", "mlx"], default="mps")
    parser.add_argument("--precision", choices=["bf16", "int8"], default="bf16")
    parser.add_argument("--enhance-prompt", action="store_true")
    parser.add_argument("--preserve-colors", action="store_true")
    parser.add_argument("--cache-text-weights", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check dependencies and files without importing the inference runtime.",
    )
    parser.add_argument(
        "--import-check",
        action="store_true",
        help="Import upstream on CPU, without loading models or starting GPU work.",
    )
    args = parser.parse_args()
    models = args.models.resolve()
    unet_name = (
        "qwen_image_2.1_bf16.safetensors"
        if args.precision == "bf16"
        else "qwen_image_2.1_int8_convrot.safetensors"
    )
    unet = models / "diffusion_models" / unet_name
    if args.backend == "mlx" and args.precision != "bf16":
        parser.error(
            "The MLX transformer port currently accepts BF16 checkpoint weights only"
        )
    clip = models / "text_encoders/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"
    projector = models / "text_encoders/Qwen3-VL-8B-Instruct-mmproj-BF16.gguf"
    vae = models / "vae/qwen_image_2.1_vae_bf16.safetensors"
    lora = (
        models / "loras/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
    )
    paths = [unet, clip, projector, vae, lora]
    if args.check:
        dependencies = [
            "torch",
            "transformers",
            "comfy_kitchen",
            "comfy_aimdo",
            "gguf",
            "safetensors",
            "PIL",
        ]
        print(
            json.dumps(
                {
                    "missing_dependencies": [
                        name
                        for name in dependencies
                        if importlib.util.find_spec(name) is None
                    ],
                    "missing_weights": [
                        str(path) for path in paths if not path.exists()
                    ],
                    "precision": args.precision,
                    "steps": args.steps,
                },
                indent=2,
            )
        )
        return
    sys.path.insert(0, str(args.upstream.resolve()))
    importlib.import_module("comfy_bridge")
    from comfy.cli_args import args as comfy_args

    comfy_args.cpu = args.import_check
    comfy_args.disable_dynamic_vram = True
    comfy_args.use_pytorch_cross_attention = True
    # Large Qwen VAE images corrupt on MPS; CPU FP32 preserves its native math.
    comfy_args.cpu_vae = True
    comfy_args.fp32_vae = True
    import folder_paths

    for category, directory in [
        ("diffusion_models", unet.parent),
        ("text_encoders", clip.parent),
        ("vae", vae.parent),
        ("loras", lora.parent),
    ]:
        folder_paths.add_model_folder_path(category, str(directory), is_default=True)
    if args.import_check:
        for module in [
            "nodes",
            "comfy_extras.nodes_qwen",
            "comfy_extras.nodes_custom_sampler",
            "viggle_turbo",
        ]:
            importlib.import_module(module)
        print("CPU import check passed; no model loaded or GPU work started.")
        return
    origins: dict[int, Path] = {}
    prompts: dict[int, str] = {}
    seeds: dict[int, int] = {}
    if args.input_manifest:
        if not args.output_dir or not args.reference_manifest:
            parser.error(
                "--input-manifest requires --output-dir and --reference-manifest"
            )
        manifest_data = json.loads(args.input_manifest.read_text())
        raw_records = (
            manifest_data["references"]
            if isinstance(manifest_data, dict)
            else manifest_data
        )
        inputs = sorted(
            [InputReference.model_validate(item) for item in raw_records],
            key=lambda item: item.frame,
        )
        if len({item.frame for item in inputs}) != len(inputs):
            parser.error("Reference frame positions must be unique")
        origins = {
            item.frame: (item.source_path or item.path).resolve() for item in inputs
        }
        prompts = {item.frame: item.prompt or args.prompt for item in inputs}
        seeds = {item.frame: item.seed for item in inputs}
        jobs = [
            (
                item.frame,
                item.path.resolve(),
                args.output_dir.resolve() / f"ref_{item.frame:06d}.png",
            )
            for item in inputs
        ]
        record_path = args.output_dir / "image_batch.json"
    else:
        if not args.image or not args.output:
            parser.error(
                "--image and --output, or --input-manifest, are required for inference"
            )
        jobs = [(args.frame, args.image.resolve(), args.output.resolve())]
        record_path = args.output.with_suffix(".json")
    if not jobs:
        parser.error("No input references supplied")
    if missing := [str(path) for path in paths if not path.exists()]:
        raise RuntimeError(f"Missing verified weights: {missing}")
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    torch.mps.set_per_process_memory_fraction(0.35)
    torch.set_num_threads(8)
    mlx_backend = None
    if args.backend == "mlx":
        import mlx.core as mx

        mx.set_memory_limit(80 * 1024**3)
        mx.set_cache_limit(1024**3)
    os.environ["COMFY_AUTO_DOWNLOAD"] = "0"
    from dit_colorize_main import (
        colorize_image,
        load_viggle_pipeline,
        process_image,
        process_image_pair,
    )
    from PIL import Image

    for _, source, output in jobs:
        if not source.is_file():
            raise RuntimeError(f"Missing source image: {source}")
        output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    pipeline = load_viggle_pipeline(
        "qwen21-viggle",
        str(unet),
        str(clip),
        lora_path=str(lora),
        vae_name=vae.name,
        clip_mmproj=str(projector),
    )
    cached_linears = 0
    if args.cache_text_weights:
        with torch.inference_mode():
            for module in pipeline["clip"].cond_stage_model.modules():
                quantized = getattr(module, "is_ggml_quantized", None)
                get_weight = getattr(module, "get_weight", None)
                if not isinstance(module, torch.nn.Linear) or not callable(quantized) or not quantized():
                    continue
                if not callable(get_weight):
                    raise TypeError("Quantized text layer lacks its native dequantizer")
                probe = torch.linspace(-1, 1, module.in_features, device="mps", dtype=torch.float32).unsqueeze(0)
                expected = module(probe)
                module.__dict__["get_weight"] = cached_dequantizer(get_weight)
                actual = module(probe)
                if not torch.equal(expected, actual):
                    raise RuntimeError("Cached text linear failed exact FP32 output parity")
                cached_linears += 1
                torch.mps.synchronize()
        print("Cached quantized text linears with exact FP32 parity:", cached_linears, flush=True)
    if args.backend == "mlx":
        from comfy.patcher_extension import WrappersMP

        from .havc_mlx import install_mlx_transformer

        diffusion_model = pipeline["model"].model.diffusion_model
        mlx_backend = install_mlx_transformer(diffusion_model, lora_path=lora)
        pipeline["model"].remove_wrappers_with_key(
            WrappersMP.DIFFUSION_MODEL, "viggle_turbo_lora"
        )
    from .havc_numerics import guard_pipeline

    guard_pipeline(pipeline, pipeline["model"].model.diffusion_model)
    enhancement_records: list[dict[str, str | float | bool]] = []
    if args.enhance_prompt:
        bridge = importlib.import_module("comfy_bridge")
        original_enhance = bridge._viggle_enhance_prompt

        def enhance(clip: object, image_tensor: torch.Tensor, prompt: str, seed: int = 42) -> str:
            began = time.monotonic()
            rewritten = str(original_enhance(clip, image_tensor, prompt, seed=seed))
            entry: dict[str, str | float | bool] = {"input_prompt": prompt, "rewritten_prompt": rewritten,
                     "changed": rewritten != prompt, "seconds": time.monotonic() - began}
            enhancement_records.append(entry)
            print("Prompt enhancement", json.dumps(entry), flush=True)
            return rewritten

        bridge.__dict__["_viggle_enhance_prompt"] = enhance
    load_seconds = time.monotonic() - start
    elapsed = 0.0
    references = []
    stride = 2 if args.paired else 1
    for begin in range(0, len(jobs), stride):
        batch = jobs[begin : begin + stride]
        if len(batch) == 2:
            if args.preserve_colors or any(prompts.get(item[0], args.prompt) != args.prompt for item in batch):
                raise ValueError("Per-reference edits require single-image inference")
            pair_dir = batch[0][2].parent / f"pair_{begin:06d}"
            pair_dir.mkdir(exist_ok=True)
            seconds = process_image_pair(
                pipeline,
                batch[0][1],
                batch[1][1],
                pair_dir,
                args.prompt,
                gap_px=16,
                steps=args.steps,
                enhance_prompt=args.enhance_prompt,
            )
            for frame, source, output in batch:
                generated = pair_dir / (source.stem + ".jpg")
                if generated.exists():
                    target = output.with_suffix(".jpg")
                    generated.replace(target)
                    references.append(
                        ReferenceRecord(
                            frame=frame,
                            path=target,
                            source_path=origins.get(frame, source),
                            model_input_path=source,
                        )
                    )
        else:
            frame, source, output = batch[0]
            prompt = prompts.get(frame, args.prompt)
            seed = seeds.get(frame, 42)
            if args.preserve_colors or seed != 42:
                image_start = time.monotonic()
                with Image.open(source) as original:
                    pixels = original.convert("RGB")
                if not args.preserve_colors:
                    pixels = pixels.convert("L").convert("RGB")
                colored = colorize_image(pipeline, pixels, prompt, args.steps,
                                         seed=seed, enhance_prompt=args.enhance_prompt)
                colored.resize(pixels.size, Image.Resampling.LANCZOS).save(output)
                seconds = time.monotonic() - image_start
            else:
                seconds = process_image(
                    source,
                    output,
                    pipeline,
                    prompt,
                    img_size=0,
                    steps=args.steps,
                    enhance_prompt=args.enhance_prompt,
                )
            if output.exists() and frame is not None:
                references.append(
                    ReferenceRecord(
                        frame=frame,
                        path=output,
                        source_path=origins.get(frame, source),
                        model_input_path=source,
                    )
                )
        elapsed += seconds
        print(
            f"Colored reference batch {begin + 1}/{len(jobs)} in {seconds:.2f}s",
            flush=True,
        )
        if args.reference_manifest:
            args.reference_manifest.parent.mkdir(parents=True, exist_ok=True)
            args.reference_manifest.write_text(
                json.dumps(
                    {
                        "references": [
                            item.model_dump(mode="json") for item in references
                        ]
                    },
                    indent=2,
                )
                + "\n"
            )
    torch.mps.synchronize()
    result = {
        "model": "Qwen-Image-2.1 + Viggle Turbo v0.2.1 rank128",
        "device": f"{args.backend}-transformer+mps-text-encoder+cpu-vae",
        "vae_device": "cpu",
        "vae_precision": "fp32",
        "unet_precision": args.precision,
        "input_pre_resize_long_side": 0,
        "steps": args.steps,
        "seed": 42,
        "resolution": 1280 if args.paired else 1024,
        "paired": args.paired,
        "preserve_input_colors": args.preserve_colors,
        "enhance_prompt": args.enhance_prompt,
        "cached_text_linears": cached_linears,
        "prompt_enhancement_records": enhancement_records,
        "prompts": prompts,
        "seeds": seeds,
        "load_seconds": load_seconds,
        "inference_seconds_author_boundary": elapsed,
        "total_seconds": time.monotonic() - start,
        "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
        "reference_count": len(references),
        "input_count": len(jobs),
        "prefix_cache_hits": mlx_backend.prefix_cache_hits
        if mlx_backend is not None
        else None,
        "prefix_cache_misses": mlx_backend.prefix_cache_misses
        if mlx_backend is not None
        else None,
    }
    record_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
