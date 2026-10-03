from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter

import mlx.core as mx
import numpy as np
import torch
from ltx_core.model.transformer.attention import AttentionOps, MPSSdpaAttention
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import (
    BasicAVTransformerBlock,
    TransformerConfig,
)
from ltx_core.model.transformer.transformer_args import TransformerArgs

from .gpu_guard import configure_gpu_limits
from .ltx_mlx import MLXBlock, from_torch, modality_arrays


def arguments(dim: int, head_dim: int, tokens: int) -> TransformerArgs:
    dtype = torch.bfloat16
    angles = torch.randn(1, 32, tokens, head_dim // 2, dtype=dtype)
    cross_angles = torch.randn(1, 32, tokens, 32, dtype=dtype)
    return TransformerArgs(
        x=torch.randn(1, tokens, dim, dtype=dtype),
        context=torch.randn(1, 256, dim, dtype=dtype),
        context_mask=torch.zeros(1, 1, 1, 256, dtype=dtype),
        timesteps=torch.randn(1, 1, 9 * dim, dtype=dtype) * 0.1,
        embedded_timestep=torch.zeros(1, 1, dim, dtype=dtype),
        positional_embeddings=(angles.cos(), angles.sin()),
        cross_positional_embeddings=(cross_angles.cos(), cross_angles.sin()),
        cross_scale_shift_timestep=torch.randn(1, 1, 4 * dim, dtype=dtype) * 0.1,
        cross_gate_timestep=torch.randn(1, 1, dim, dtype=dtype) * 0.1,
        prompt_timestep=torch.randn(1, 1, 2 * dim, dtype=dtype) * 0.1,
        cross_attn_perturbation_mask=torch.ones(1, 1, 1, dtype=dtype),
        enabled=True,
    )


def to_mps(args: TransformerArgs) -> TransformerArgs:
    assert args.cross_positional_embeddings is not None
    assert args.cross_scale_shift_timestep is not None
    assert args.cross_gate_timestep is not None
    assert args.prompt_timestep is not None
    assert args.cross_attn_perturbation_mask is not None
    return replace(
        args,
        x=args.x.to("mps"),
        context=args.context.to("mps"),
        context_mask=args.context_mask.to("mps"),
        timesteps=args.timesteps.to("mps"),
        embedded_timestep=args.embedded_timestep.to("mps"),
        positional_embeddings=(
            args.positional_embeddings[0].to("mps"),
            args.positional_embeddings[1].to("mps"),
        ),
        cross_positional_embeddings=(
            args.cross_positional_embeddings[0].to("mps"),
            args.cross_positional_embeddings[1].to("mps"),
        ),
        cross_scale_shift_timestep=args.cross_scale_shift_timestep.to("mps"),
        cross_gate_timestep=args.cross_gate_timestep.to("mps"),
        prompt_timestep=args.prompt_timestep.to("mps"),
        cross_attn_perturbation_mask=args.cross_attn_perturbation_mask.to("mps"),
    )


@torch.inference_mode()
def benchmark(output: Path, tokens: int, repeats: int) -> None:
    configure_gpu_limits()
    torch.manual_seed(17)
    attention = MPSSdpaAttention()
    block = (
        BasicAVTransformerBlock(
            TransformerConfig(4096, 32, 128, 4096, True, True),
            TransformerConfig(2048, 32, 64, 2048, True, True),
            LTXRopeType.SPLIT,
            attention_ops=AttentionOps(
                attention_function=attention, masked_attention_function=attention
            ),
        )
        .eval()
        .to(torch.bfloat16)
    )
    for name, parameter in block.named_parameters():
        if name.endswith(("q_norm.weight", "k_norm.weight")):
            parameter.fill_(1)
        else:
            parameter.normal_(0, 0.008)
    backend = MLXBlock(block)
    video, audio = arguments(4096, 128, tokens), arguments(2048, 64, 128)
    vx, ax = from_torch(video.x), from_torch(audio.x)
    va, aa = modality_arrays(video), modality_arrays(audio)
    timings: dict[str, list[float]] = {}
    for compiled in (False, True):
        backend.compiled = compiled
        key = "mlx_fused" if compiled else "mlx_unfused"
        timings[key] = []
        for iteration in range(repeats + 1):
            started = perf_counter()
            result_video, result_audio = backend(vx, ax, va, aa)
            mx.eval(result_video, result_audio)
            elapsed = perf_counter() - started
            if iteration:
                timings[key].append(elapsed)
        print(key, timings[key], flush=True)
    block.to("mps")
    video = to_mps(video)
    audio = to_mps(audio)
    timings["pytorch_mps"] = []
    for iteration in range(repeats + 1):
        started = perf_counter()
        expected_video, expected_audio = block(video, audio)
        torch.mps.synchronize()
        elapsed = perf_counter() - started
        if iteration:
            timings["pytorch_mps"].append(elapsed)
    assert result_video is not None and result_audio is not None
    assert expected_video is not None and expected_audio is not None
    errors = {}
    for name, actual, expected in (
        ("video", result_video, expected_video.x),
        ("audio", result_audio, expected_audio.x),
    ):
        difference = np.abs(
            np.array(actual.astype(mx.float32)) - expected.float().cpu().numpy()
        )
        errors[name] = {
            "mean_absolute": float(difference.mean()),
            "max_absolute": float(difference.max()),
        }
    record = {
        "scope": "One randomly initialized official-size block; not trained-weight or full-pipeline performance",
        "video_tokens": tokens,
        "audio_tokens": 128,
        "context_tokens": 256,
        "dtype": "bfloat16",
        "mps_attention": attention.label,
        "parameters": sum(parameter.numel() for parameter in block.parameters()),
        "seconds": timings,
        "median_seconds": {key: median(values) for key, values in timings.items()},
        "errors_against_mps": errors,
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(record, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    options = parser.parse_args()
    benchmark(options.output, options.tokens, options.repeats)
