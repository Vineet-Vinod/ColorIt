from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol, cast

import torch
from pydantic import BaseModel

from .havc_image import DEFAULT_PROMPT, InputReference


class ModelPatcher(Protocol):
    model: torch.nn.Module

    def remove_wrappers_with_key(self, kind: object, name: str) -> None: ...


class Clip(Protocol):
    cond_stage_model: torch.nn.Module


class VAE(Protocol):
    vae_dtype: torch.dtype
    first_stage_model: torch.nn.Module

    def encode(self, pixels: torch.Tensor) -> torch.Tensor: ...

    def decode(self, samples: torch.Tensor) -> torch.Tensor: ...


class DiagnosticConfig(BaseModel):
    input_manifest: Path
    output_dir: Path
    pairs: list[tuple[int, int]]
    backend: Literal["mps", "mlx"]
    bf16_text_enc: bool
    upstream: Path
    models: Path
    import_check: bool
    trace_vae: bool
    vae_cycle_only: bool
    disable_vae_strips: bool
    fp32_vae: bool
    cpu_vae: bool


def tensor_records(value: object, label: str) -> list[dict[str, object]]:
    if isinstance(value, torch.Tensor):
        finite = torch.isfinite(value)
        count = int(finite.sum().item())
        record: dict[str, object] = {
            "name": label,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "finite": count == value.numel(),
            "nonfinite_count": value.numel() - count,
        }
        if value.numel() and count == value.numel():
            record["abs_max"] = float(value.detach().abs().max().item())
        return [record]
    result: list[dict[str, object]] = []
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.extend(tensor_records(item, f"{label}[{index}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            result.extend(tensor_records(item, f"{label}.{key}"))
    return result


def trace_method(target: object, name: str, label: str, log: Path) -> None:
    original = cast(Callable[..., object], getattr(target, name))
    calls = 0

    def inspect(value: object, stage: str) -> None:
        records = tensor_records(value, stage)
        event = {"stage": stage, "call": calls, "tensors": records}
        line = json.dumps(event)
        print(line, flush=True)
        with log.open("a") as stream:
            stream.write(line + "\n")
        if any(record["finite"] is False for record in records):
            raise FloatingPointError(f"First nonfinite tensor at {stage}; see {log}")

    def traced(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        inspect({"args": args, "kwargs": kwargs}, label + ".input")
        result = original(*args, **kwargs)
        inspect(result, label + ".output")
        if label == "vae.encode" and isinstance(result, torch.Tensor):
            torch.save(result, log.parent / f"vae_latent_{calls:02d}.pt")
        return result

    setattr(target, name, traced)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace HAVC numerical failure without changing its recipe."
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path(
            "tmp/adapter_compare_90_120/havc_diagnose/source_archive/automatic/input_manifest.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pairs", nargs="+", default=["2250,2266", "2250,2266", "2278,2300"]
    )
    parser.add_argument("--backend", choices=["mps", "mlx"], default="mlx")
    parser.add_argument("--bf16-text-enc", action="store_true")
    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path("tmp/adapter_compare_90_120/upstream/HAVCServerDiT"),
    )
    parser.add_argument(
        "--models", type=Path, default=Path("data/adapter_compare_90_120/havcdit")
    )
    parser.add_argument("--import-check", action="store_true")
    parser.add_argument("--trace-vae", action="store_true")
    parser.add_argument("--vae-cycle-only", action="store_true")
    parser.add_argument("--disable-vae-strips", action="store_true")
    parser.add_argument("--fp32-vae", action="store_true")
    parser.add_argument("--cpu-vae", action="store_true")
    args = parser.parse_args()
    config = DiagnosticConfig.model_validate(
        {
            **vars(args),
            "pairs": [
                tuple(int(frame) for frame in pair.split(",")) for pair in args.pairs
            ],
        }
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(config.upstream.resolve()))
    importlib.import_module("comfy_bridge")
    from comfy.cli_args import args as comfy_args

    comfy_args.cpu = config.import_check
    comfy_args.disable_dynamic_vram = True
    comfy_args.use_pytorch_cross_attention = True
    comfy_args.bf16_text_enc = config.bf16_text_enc
    comfy_args.fp32_vae = config.fp32_vae
    comfy_args.cpu_vae = config.cpu_vae
    import folder_paths

    for category in ["diffusion_models", "text_encoders", "vae", "loras"]:
        folder_paths.add_model_folder_path(
            category, str((config.models / category).resolve()), is_default=True
        )
    author = importlib.import_module("dit_colorize_main")
    if config.import_check:
        print("CPU import check passed; no model loaded or GPU work started.")
        return
    if config.disable_vae_strips:
        import comfy.ldm.wan.vae2_2 as wan_vae

        wan_vae.STRIP_ELEMS = 2**60
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    torch.mps.set_per_process_memory_fraction(0.35)
    torch.set_num_threads(8)
    if config.backend == "mlx":
        import mlx.core as mx

        mx.set_memory_limit(80 * 1024**3)
        mx.set_cache_limit(1024**3)
    os.environ["COMFY_AUTO_DOWNLOAD"] = "0"
    raw = json.loads(config.input_manifest.read_text())
    records = raw["references"] if isinstance(raw, dict) else raw
    inputs = {
        item.frame: item.path.resolve()
        for item in map(InputReference.model_validate, records)
    }
    models = config.models.resolve()
    lora = (
        models / "loras/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
    )
    load = cast(Callable[..., dict[str, object]], author.load_viggle_pipeline)
    pipeline = load(
        "qwen21-viggle",
        str(models / "diffusion_models/qwen_image_2.1_bf16.safetensors"),
        str(models / "text_encoders/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"),
        lora_path=str(lora),
        vae_name="qwen_image_2.1_vae_bf16.safetensors",
        clip_mmproj=str(models / "text_encoders/Qwen3-VL-8B-Instruct-mmproj-BF16.gguf"),
    )
    patcher = cast(ModelPatcher, pipeline["model"])
    diffusion = patcher.model.get_submodule("diffusion_model")
    backend = None
    if config.backend == "mlx":
        from comfy.patcher_extension import WrappersMP

        from .havc_mlx import install_mlx_transformer

        backend = install_mlx_transformer(diffusion, lora_path=lora)
        patcher.remove_wrappers_with_key(
            WrappersMP.DIFFUSION_MODEL, "viggle_turbo_lora"
        )
    log = config.output_dir / "finite_trace.jsonl"
    clip = cast(Clip, pipeline["clip"])
    clip_model = clip.cond_stage_model
    transformer = clip_model.get_submodule("qwen3vl_8b.transformer")
    print(
        json.dumps(
            {
                "encoder_declared_dtype": str(getattr(transformer, "dtype", None)),
                "encoder_norm_dtype": str(
                    transformer.get_parameter("model.norm.weight").dtype
                ),
                "vae_dtype": str(cast(VAE, pipeline["vae"]).vae_dtype),
                "bf16_text_enc_requested": config.bf16_text_enc,
            }
        ),
        flush=True,
    )
    trace_method(clip, "encode_from_tokens_scheduled", "clip", log)
    trace_method(pipeline["vae"], "encode", "vae.encode", log)
    trace_method(pipeline["vae"], "decode", "vae.decode", log)
    trace_method(diffusion, "_forward", "transformer", log)
    vae_model = cast(VAE, pipeline["vae"])
    if config.trace_vae:
        def trace_layer(
            module: torch.nn.Module, inputs: tuple[object, ...], result: object
        ) -> None:
            records = tensor_records(result, str(module.diagnostic_name))
            line = json.dumps({"stage": "vae.layer", "tensors": records})
            with log.open("a") as stream:
                stream.write(line + "\n")
            if any(record["finite"] is False for record in records):
                weights = tensor_records(dict(module.named_parameters()), "weights")
                print(line, json.dumps(weights), flush=True)
                raise FloatingPointError(line)

        for name, module in vae_model.first_stage_model.named_modules():
            if not list(module.children()):
                module.diagnostic_name = name
                module.register_forward_hook(trace_layer)
    if config.vae_cycle_only:
        import numpy as np
        from PIL import Image

        pixels = torch.from_numpy(
            np.array(
                Image.open(inputs[config.pairs[0][0]])
                .convert("RGB")
                .resize((2432, 672)),
                dtype=np.float32,
            )
            / 255
        ).unsqueeze(0)
        with torch.inference_mode():
            for index in range(3):
                latent = vae_model.encode(pixels)
                decoded = vae_model.decode(latent)
                Image.fromarray(
                    (decoded[0, ..., :3].numpy().clip(0, 1) * 255).astype(np.uint8)
                ).save(config.output_dir / f"vae_reconstruction_{index}.png")
        return
    process = cast(Callable[..., float], author.process_image_pair)
    for index, (first, second) in enumerate(config.pairs):
        destination = config.output_dir / f"pair_{index:02d}_{first}_{second}"
        destination.mkdir(exist_ok=True)
        start = time.monotonic()
        process(
            pipeline,
            inputs[first],
            inputs[second],
            destination,
            DEFAULT_PROMPT,
            gap_px=16,
            steps=6,
            enhance_prompt=False,
        )
        print(
            json.dumps(
                {
                    "pair": [first, second],
                    "seconds": time.monotonic() - start,
                    "prefix_cache_hits": backend.prefix_cache_hits if backend else None,
                    "prefix_cache_misses": backend.prefix_cache_misses
                    if backend
                    else None,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
