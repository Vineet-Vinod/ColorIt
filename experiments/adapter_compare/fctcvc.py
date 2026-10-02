from __future__ import annotations

import argparse
import importlib
import json
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from .windows import frame_windows


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Path
    anchors: Path
    checkpoint: Path
    upstream: Path
    output: Path
    start: int = Field(default=2250, ge=0)
    frames: int = Field(default=750, ge=3)
    width: int = Field(default=683, ge=64)
    height: int = Field(default=384, ge=64)
    window: int = Field(default=15, ge=3)
    device: Literal["mps", "cpu"] = "mps"
    scene_manifest: Path | None = None


class SceneManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    scene_boundaries: list[int]


def load_model(options: Options) -> torch.nn.Module:
    sys.path.insert(0, str(options.upstream.resolve() / "FGST"))
    module = importlib.import_module(
        "tools.mmedit.models.backbones.sr_backbones.FGT_colorization"
    )
    model = cast(
        torch.nn.Module, module.FGT_colorization(dim=32, GMAnet_pretrained=None)
    )
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("generator."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("generator.")
    }
    model.load_state_dict(state, strict=True)
    return model.eval().to(options.device)


def read_frames(path: Path, start: int, count: int) -> list[NDArray[np.uint8]]:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read {count} frames at {start} from {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def colorize(options: Options) -> None:
    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    model = load_model(options)
    print(
        "Loaded parameters",
        sum(weight.numel() for weight in model.parameters()),
        flush=True,
    )
    source = read_frames(options.source, options.start, options.frames)
    anchors = read_frames(options.anchors, options.start, options.frames)
    output_frames = []
    timings = []
    started = perf_counter()
    cuts = []
    if options.scene_manifest is not None:
        manifest = SceneManifest.model_validate_json(options.scene_manifest.read_text())
        cuts = [boundary - options.start for boundary in manifest.scene_boundaries]
    with torch.inference_mode():
        for segment in frame_windows(options.frames, options.window, cuts):
            begin, end = segment.begin, segment.end
            window_started = perf_counter()
            images = []
            for index in range(begin, end):
                rgb = cv2.resize(
                    source[index],
                    (options.width, options.height),
                    interpolation=cv2.INTER_LINEAR,
                )
                lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
                if index in (begin, end - 1):
                    reference = cv2.resize(
                        anchors[index],
                        (options.width, options.height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    lab[:, :, 1:] = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)[:, :, 1:]
                else:
                    lab[:, :, 1:] = 128
                images.append(lab)
            while len(images) < 3:
                images.append(images[-1].copy())
            inputs = (
                torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2).copy())
                .float()
                .unsqueeze(0)
                .to(options.device)
                / 255
            )
            result = model(inputs)[0].float().cpu().numpy()[0]
            for local in range(segment.skip, end - begin):
                predicted = np.clip(
                    result[local].transpose(1, 2, 0) * 255, 0, 255
                ).astype(np.uint8)
                output_frames.append(cv2.cvtColor(predicted, cv2.COLOR_LAB2RGB))
            elapsed = perf_counter() - window_started
            timings.append(elapsed)
            progress = {
                "frames": len(output_frames),
                "total": options.frames,
                "seconds": perf_counter() - started,
                "window_seconds": elapsed,
            }
            options.output.with_suffix(".progress.json").write_text(
                json.dumps(progress) + "\n"
            )
            print(progress, flush=True)
            if options.device == "mps":
                torch.mps.empty_cache()
    if len(output_frames) != options.frames:
        raise RuntimeError(
            f"Coverage mismatch: {len(output_frames)} / {options.frames}"
        )
    inference_seconds = perf_counter() - started
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{options.width}x{options.height}",
            "-r",
            "25",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "16",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(options.output),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    for frame in output_frames:
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("Encoding failed")
    result_record = {
        **options.model_dump(mode="json"),
        "seconds": inference_seconds,
        "fps": options.frames / inference_seconds,
        "window_seconds": timings,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "anchor_policy": "Aligned CMNET2 output at both ends of each window; not an independent automatic palette test",
    }
    options.output.with_suffix(".json").write_text(
        json.dumps(result_record, indent=2) + "\n"
    )
    np.save(options.output.with_suffix(".npy"), np.stack(output_frames))
    print(result_record, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("source", "anchors", "checkpoint", "upstream", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name, default in (
        ("start", 2250),
        ("frames", 750),
        ("width", 683),
        ("height", 384),
        ("window", 15),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--scene-manifest", type=Path)
    colorize(Options.model_validate(vars(parser.parse_args())))


if __name__ == "__main__":
    main()
