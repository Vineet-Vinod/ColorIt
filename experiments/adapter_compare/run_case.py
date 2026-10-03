from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .gpu_guard import supervise


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["ltx", "havc"]
    case: Literal["automatic", "assisted"]
    backend: Literal["mlx", "mps"] = "mlx"
    source: Path = Path("deep_rem.mp4")
    destination: Path = Path("tmp/adapter_compare_90_120/cases")


def run(options: Options) -> int:
    if options.model == "havc":
        return run_havc(options)
    folder = options.destination / f"{options.model}_{options.case}"
    folder.mkdir(parents=True, exist_ok=True)
    protocol = folder / "protocol.json"
    if protocol.exists():
        raise FileExistsError(
            f"This case was already launched: {protocol}. Use a new destination for a labelled rerun."
        )
    source = folder / "source.mp4"
    command = [
        sys.executable,
        "-u",
        "-m",
        "experiments.adapter_compare.ltx",
        "--source",
        str(source),
        "--weights",
        "data/adapter_compare_90_120/ltx25",
        "--unused-upsampler",
        "tmp/reference_video_models/ltx_colorization_weights/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "--output",
        str(folder / "output.mp4"),
    ]
    if options.case == "assisted":
        command += [
            "--scene-manifest",
            "experiments/adapter_compare/scene_manifest.json",
            "--prompts",
            "experiments/adapter_compare/prompts",
        ]
    if options.backend == "mps":
        command.append("--no-mlx")
    protocol.write_text(
        json.dumps(
            {
                **options.model_dump(mode="json"),
                "source_frames": [2250, 3000],
                "fps": 25,
                "command": command,
                "manual_input": options.case == "assisted",
                "automatic_policy": "Fixed generic prompt, seed 42, continuous chunked clip, first completed output retained. No manual reference or palette input.",
                "shared_setup": "Mac MLX port, official convolutional VAE, stage-1 recipe at 960x544, 121-frame chunks with 17-frame overlap.",
            },
            indent=2,
        )
        + "\n"
    )
    subprocess.run(
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
            "trim=start_frame=2250:end_frame=3000,setpts=PTS-STARTPTS,scale=960:544:flags=lanczos",
            "-frames:v",
            "750",
            "-c:v",
            "libx264",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv444p",
            str(source),
        ],
        check=True,
    )
    result = supervise(command, folder / "guard.json")
    if result != 0:
        return result
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(folder / "output.mp4"),
            "-ss",
            "90",
            "-i",
            str(options.source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-t",
            "30",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(folder / "review.mp4"),
        ],
        check=True,
    )
    return 0


def run_havc(options: Options) -> int:
    folder = (options.destination / f"havc_{options.case}").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    protocol = folder / "protocol.json"
    if protocol.exists():
        raise FileExistsError(f"This case was already launched: {protocol}")
    sources = folder / "sources"
    references = folder / "references"
    manifest = folder / "references.json"
    commands = [
        [
            "tmp/havc_cpu/.venv/bin/python",
            "-m",
            "experiments.adapter_compare.havc_sources",
            "--source",
            str(options.source.resolve()),
            "--output",
            str(sources),
            "--mode",
            options.case,
        ],
        [
            "tmp/adapter_compare_90_120/havc_env/bin/python",
            "-u",
            "-m",
            "experiments.adapter_compare.havc_image",
            "--input-manifest",
            str(sources / "input_manifest.json"),
            "--output-dir",
            str(references),
            "--reference-manifest",
            str(manifest),
            "--paired",
            "--backend",
            options.backend,
        ],
        [
            ".venv/bin/python",
            "-u",
            "-m",
            "experiments.adapter_compare.havc_video",
            "--source",
            str(options.source.resolve()),
            "--references",
            str(references),
            "--manifest",
            str(manifest),
            "--output",
            str(folder / "output.mp4"),
            "--mode",
            options.case,
        ],
    ]
    if options.case == "assisted":
        commands[1] += [
            "--prompt",
            Path("experiments/adapter_compare/prompts/havc.txt").read_text().strip(),
        ]
    protocol.write_text(
        json.dumps(
            {
                **options.model_dump(mode="json"),
                "source_frames": [2250, 3000],
                "commands": commands,
                "manual_input": options.case == "assisted",
                "implementation": "Mac integration of author Qwen2.1/Viggle and CMNET2; no upstream video CLI.",
                "vae": "Native Qwen VAE on CPU FP32; large-image MPS VAE reconstruction is numerically invalid.",
            },
            indent=2,
        )
        + "\n"
    )
    subprocess.run(commands[0], check=True)
    for name, command in zip(("references", "propagation"), commands[1:], strict=True):
        result = supervise(command, folder / f"{name}_guard.json")
        if result != 0:
            return result
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["ltx", "havc"])
    parser.add_argument("--case", choices=["automatic", "assisted"], required=True)
    parser.add_argument("--backend", choices=["mlx", "mps"], default="mlx")
    parser.add_argument("--source", type=Path, default=Path("deep_rem.mp4"))
    parser.add_argument(
        "--destination", type=Path, default=Path("tmp/adapter_compare_90_120/cases")
    )
    raise SystemExit(run(Options.model_validate(vars(parser.parse_args()))))
