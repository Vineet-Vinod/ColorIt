from __future__ import annotations

import math
import re
import subprocess
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, Field

from src.experimental.experimental_dataclasses import ClipInfo, Shot


class ProbeStream(BaseModel):
    codec_type: str
    width: int = 0
    height: int = 0
    avg_frame_rate: str = "0/0"
    r_frame_rate: str = "0/0"
    duration: float = 0
    nb_frames: int = 0
    nb_read_frames: int = 0


class ProbeFormat(BaseModel):
    duration: float = Field(default=0, ge=0)


class ProbeResult(BaseModel):
    streams: list[ProbeStream]
    format: ProbeFormat


def probe_clip(source: Path, *, count_frames: bool = False) -> ClipInfo:
    command = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json"]
    if count_frames:
        command.append("-count_frames")
    result = subprocess.run(
        [*command, str(source)], capture_output=True, text=True, check=True
    )
    parsed = ProbeResult.model_validate_json(result.stdout)
    stream = next((item for item in parsed.streams if item.codec_type == "video"), None)
    if stream is None:
        raise ValueError("Input has no video stream.")
    fps = Fraction(
        stream.avg_frame_rate if stream.avg_frame_rate != "0/0" else stream.r_frame_rate
    )
    duration = max(stream.duration, parsed.format.duration)
    if (
        fps <= 0
        or stream.width <= 0
        or stream.height <= 0
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError(
            "Input must have valid video dimensions, frame rate, and duration."
        )
    if stream.width % 2 or stream.height % 2:
        raise ValueError(
            "Experimental delivery currently requires even video dimensions."
        )
    return ClipInfo(
        stream.width,
        stream.height,
        fps,
        stream.nb_read_frames or stream.nb_frames,
        duration,
        any(item.codec_type == "audio" for item in parsed.streams),
    )


def make_shots(boundaries: list[int], frames: int) -> list[Shot]:
    cuts = sorted({0, frames, *(frame for frame in boundaries if 0 < frame < frames)})
    shots = []
    for start, end in pairwise(cuts):
        references = tuple(
            sorted(
                {
                    start + min(end - start - 1, int((end - start) * position))
                    for position in (0.2, 0.5, 0.8)
                }
            )
        )
        shots.append(Shot(start, end, references))
    return shots


def detect_shots(source: Path, info: ClipInfo) -> list[Shot]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(source),
            "-vf",
            "select='gt(scene,0.1)',metadata=print:file=-",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    candidates = re.findall(
        r"pts_time:([0-9.]+)\s+lavfi.scene_score=([0-9.]+)", result.stdout
    )
    cuts: list[tuple[int, float]] = []
    minimum_gap = max(1, round(float(info.fps) * 0.5))
    for time, score in candidates:
        frame = round(float(time) * float(info.fps))
        if frame < minimum_gap or frame >= info.frames:
            continue
        if cuts and frame - cuts[-1][0] < minimum_gap:
            if float(score) > cuts[-1][1]:
                cuts[-1] = (frame, float(score))
        else:
            cuts.append((frame, float(score)))
    return make_shots([frame for frame, _ in cuts], info.frames)


def restore_luminance(
    source: NDArray[np.uint8], prediction: Image.Image
) -> NDArray[np.uint8]:
    source_lab = cv2.cvtColor(source.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    predicted_lab = cv2.cvtColor(
        np.asarray(prediction.convert("RGB"), dtype=np.float32) / 255, cv2.COLOR_RGB2LAB
    )
    source_lab[:, :, 1:] = cv2.resize(
        predicted_lab[:, :, 1:], (source.shape[1], source.shape[0])
    )
    output = np.clip(cv2.cvtColor(source_lab, cv2.COLOR_LAB2RGB) * 255, 0, 255).astype(
        np.uint8
    )
    return output


def deliver_clip(raw: Path, source: Path, output: Path, info: ClipInfo) -> None:
    budget = source.stat().st_size * 2
    for crf in (18, 23, 28):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(raw),
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "libx265",
                "-tag:v",
                "hvc1",
                "-preset",
                "medium",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-af",
                "apad",
                "-t",
                str(info.frames / float(info.fps)),
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
        if output.stat().st_size <= budget:
            break
    else:
        raise RuntimeError(
            "Delivery exceeds twice the input size at the supported quality settings."
        )
    result = probe_clip(output, count_frames=True)
    if (
        result.frames != info.frames
        or result.fps != info.fps
        or result.audio != info.audio
    ):
        raise RuntimeError(
            "Delivery changed frame count, frame rate, or audio availability."
        )
    if abs(result.duration - info.frames / float(info.fps)) > max(
        0.05, 1 / float(info.fps)
    ):
        raise RuntimeError("Delivery changed the clip duration.")
