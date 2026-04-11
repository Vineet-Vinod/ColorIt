from __future__ import annotations

import json
import subprocess
from pathlib import Path


def extract_single_frame(*, input_path: Path, output_path: Path, time_seconds: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(time_seconds),
        "-i",
        str(input_path),
        "-frames:v",
        "1",
        str(output_path),
    ]
    _run(command)


def extract_clip(
    *,
    input_path: Path,
    output_path: Path,
    start_time: str,
    end_time: str,
    video_codec: str,
    crf: int,
    pixel_format: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ss",
        start_time,
        "-to",
        end_time,
        "-c:v",
        video_codec,
        "-crf",
        str(crf),
        "-pix_fmt",
        pixel_format,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_path),
    ]
    _run(command)


def ffprobe_media(path: Path) -> dict[str, str | int | float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe command failed with code {result.returncode}: {result.stderr.strip()}"
        )

    payload = json.loads(result.stdout)
    video_stream = next(
        stream for stream in payload["streams"] if stream.get("codec_type") == "video"
    )
    return {
        "duration_seconds": float(payload["format"]["duration"]),
        "fps": str(video_stream["avg_frame_rate"]),
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
    }


def get_media_duration_seconds(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe duration probe failed with code {result.returncode}: {result.stderr.strip()}"
        )
    return float(result.stdout.strip())


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg command failed with code {result.returncode}: {result.stderr.strip()}"
        )
