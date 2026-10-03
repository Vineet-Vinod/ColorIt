from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import cast

import mlx.core as mx
import torch
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.loader.registry import DummyRegistry
from ltx_core.model.transformer import X0Model
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_core.tools import LatentTools
from ltx_pipelines.chunks import ChunkConfig
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from pydantic import BaseModel, ConfigDict, Field

from .gpu_guard import configure_gpu_limits
from .ltx_mlx import install_mlx_blocks
from .windows import SceneManifest


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Path
    weights: Path
    unused_upsampler: Path
    scene_manifest: Path | None = None
    prompts: Path | None = None
    prompt: str = "Reference shows a black and white film. Edited shows the same scene with natural colors restored. COLORIZE the people, clothing and surroundings with vivid, plausible colors that remain consistent throughout the film. Preserve subject identity, framing, motion and background geometry, changing only color."
    output: Path
    frames: int = Field(default=750, ge=1)
    source_offset: int = Field(default=2250, ge=0)
    seed: int = 42
    mlx: bool = True
    width: int = Field(default=960, ge=64, multiple_of=32)
    height: int = Field(default=544, ge=64, multiple_of=32)


def wrap_model(model: torch.nn.Module, tools: LatentTools | None) -> torch.nn.Module:
    del tools
    wrapped = cast(X0Model, model)
    core = cast(LTXModel, wrapped.velocity_model)
    started = perf_counter()
    backend = install_mlx_blocks(core, disposal_owner=wrapped)
    print(
        f"MLX installed: {len(backend.blocks)} blocks in {perf_counter() - started:.2f}s",
        flush=True,
    )
    return model


def load_pipeline(options: Options) -> ICLoraPipeline:
    weights = options.weights
    paths = ModelPaths.from_split(
        transformer_path=str(
            weights
            / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
        ),
        text_encoder_path=str(
            weights / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
        ),
        video_vae_path=str(weights / "vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
        audio_vae_path=str(weights / "vae/ltx-2.5-audio-vae-bf16.safetensors"),
    )
    adapter = weights / "ltx-2.5-22b-ic-lora-colorization-0.9.safetensors"
    for path in (
        paths.transformer(),
        paths.text_encoder(),
        paths.video_vae(),
        paths.audio_vae(),
        str(adapter),
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Download gated LTX-2.5 weights before inference: {path}"
            )
    # Cached shells reload weights on every build. Replacing their blocks with MLX
    # would invalidate the shell, so this experiment uses fresh official builders.
    pipeline = ICLoraPipeline(
        model_paths=paths,
        spatial_upsampler_path=str(options.unused_upsampler),
        loras=[
            LoraPathStrengthAndSDOps(str(adapter), 1.0, LTXV_LORA_COMFY_RENAMING_MAP)
        ],
        device=torch.device("mps"),
        registry=DummyRegistry(),
    )
    if options.mlx:
        pipeline._diffusion_stage = pipeline._diffusion_stage.with_model_wrapper(
            wrap_model
        )
    return pipeline


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


@torch.inference_mode()
def colorize(options: Options) -> None:
    configure_gpu_limits()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline(options)
    boundaries = (
        SceneManifest.model_validate_json(
            options.scene_manifest.read_text()
        ).scene_boundaries
        if options.scene_manifest is not None
        else []
    )
    cuts = sorted(
        {
            0,
            options.frames,
            *(
                cut - options.source_offset
                for cut in boundaries
                if options.source_offset < cut < options.source_offset + options.frames
            ),
        }
    )
    shots = []
    started = perf_counter()
    for shot, (begin, end) in enumerate(pairwise(cuts)):
        count = end - begin
        model_frames = math.ceil((count - 1) / 8) * 8 + 1
        source = options.output.parent / f"ltx_source_shot_{shot:02d}.mp4"
        raw = options.output.parent / f"ltx_raw_shot_{shot:02d}.mp4"
        trimmed = options.output.parent / f"ltx_shot_{shot:02d}.mp4"
        prompt = (
            (options.prompts / f"shot_{shot:02d}.txt").read_text().strip()
            if options.prompts is not None
            else options.prompt
        )
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(options.source),
                "-an",
                "-vf",
                f"trim=start_frame={begin}:end_frame={end},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={model_frames - count}",
                "-frames:v",
                str(model_frames),
                "-c:v",
                "libx264",
                "-crf",
                "10",
                "-pix_fmt",
                "yuv444p",
                str(source),
            ]
        )
        shot_started = perf_counter()
        result = pipeline(
            prompt=prompt,
            seed=options.seed,
            height=options.height * 2,
            width=options.width * 2,
            num_frames=model_frames,
            frame_rate=25,
            images=[],
            video_conditioning=[(str(source), 1.0)],
            tiling_config=AUTO_TILING,
            skip_stage_2=True,
            chunk_config=ChunkConfig(
                chunk_pixel_frames=121,
                next_video_carry_frames=17,
                overlap_blend_frames=17,
            ),
        )
        encode_video(
            video=result.video,
            fps=25,
            audio=None,
            output_path=str(raw),
            video_chunks_number=get_video_chunks_number(
                result.num_frames, result.tiling_config
            ),
            crf=16,
            preset="medium",
        )
        elapsed = perf_counter() - shot_started
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(raw),
                "-frames:v",
                str(count),
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "16",
                str(trimmed),
            ]
        )
        record = {
            "shot": shot,
            "begin": begin,
            "end": end,
            "model_frames": model_frames,
            "seconds": elapsed,
            "prompt": prompt,
            "output": str(trimmed),
            "mlx_peak_bytes": mx.get_peak_memory() if options.mlx else None,
        }
        shots.append(record)
        options.output.with_suffix(".progress.json").write_text(
            json.dumps(shots, indent=2) + "\n"
        )
        print(record, flush=True)
        gc.collect()
        torch.mps.empty_cache()
        mx.clear_cache()
    concat = options.output.with_suffix(".concat.txt")
    concat.write_text(
        "".join(f"file '{Path(str(shot['output'])).resolve()}'\n" for shot in shots)
    )
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(options.output),
        ]
    )
    record = {
        **options.model_dump(mode="json"),
        "seconds": perf_counter() - started,
        "shots": shots,
        "width": options.width,
        "height": options.height,
        "fps": 25,
        "recipe": "Official stage-1 distilled colorization, source video and text only, strength 1, no CFG, convolutional VAE; 121-frame chunks with 17-frame decoded overlap",
        "backend": "MLX transformer blocks with official MPS text encoder, VAE and transformer input/output layers"
        if options.mlx
        else "Official PyTorch MPS",
        "case": "assisted" if options.prompts is not None else "automatic",
    }
    options.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(record, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in (
        "source",
        "weights",
        "unused-upsampler",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--prompt", default=Options.model_fields["prompt"].default)
    parser.add_argument("--frames", type=int, default=750)
    parser.add_argument("--source-offset", type=int, default=2250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--mlx", action=argparse.BooleanOptionalAction, default=True)
    colorize(Options.model_validate(vars(parser.parse_args())))
