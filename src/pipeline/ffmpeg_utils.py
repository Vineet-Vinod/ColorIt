from __future__ import annotations

import json
import subprocess
from pathlib import Path
import shlex


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


def detect_scene_change_times(*, movie_path: Path, threshold: float) -> list[float]:
    escaped_path = str(movie_path).replace(",", "\\,")
    lavfi = f"movie={escaped_path},select=gt(scene\\,{threshold:.4f})"
    command = [
        "ffprobe",
        "-v",
        "error",
        "-f",
        "lavfi",
        lavfi,
        "-show_entries",
        "frame=pts_time",
        "-of",
        "csv=p=0",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe scene detection failed with code {result.returncode}: {result.stderr.strip()}"
        )

    change_times: list[float] = []
    for line in result.stdout.splitlines():
        value = line.strip().rstrip(",")
        if not value:
            continue
        change_times.append(float(value))
    return change_times


def extract_frames(
    *,
    input_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        str(output_dir / "%06d.png"),
    ]
    _run(command)


def encode_video_from_frames(
    *,
    frame_dir: Path,
    output_path: Path,
    fps: str,
    video_codec: str,
    crf: int,
    pixel_format: str,
    audio_input_path: Path | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        fps_to_decimal_string(fps),
        "-i",
        str(frame_dir / "%06d.png"),
    ]
    if audio_input_path is not None:
        command.extend(["-i", str(audio_input_path), "-map", "0:v:0", "-map", "1:a:0?"])
    command.extend(
        [
            "-c:v",
            video_codec,
            "-crf",
            str(crf),
            "-pix_fmt",
            pixel_format,
        ]
    )
    if audio_input_path is not None:
        command.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
    command.append(str(output_path))
    _run(command)


def concat_videos(
    *,
    input_list_path: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(input_list_path),
        "-c",
        "copy",
        str(output_path),
    ]
    _run(command)


def compress_video(
    *,
    input_path: Path,
    output_path: Path,
    video_codec: str,
    preset: str,
    crf: int,
    audio_codec: str,
    audio_bitrate: str,
    faststart: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        video_codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        audio_codec,
        "-b:a",
        audio_bitrate,
    ]
    if faststart:
        command.extend(["-movflags", "+faststart"])
    command.append(str(output_path))
    _run(command)


def fps_to_decimal_string(value: str) -> str:
    if "/" not in value:
        return value
    numerator, denominator = value.split("/", 1)
    return str(float(numerator) / float(denominator))


def quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg command failed with code {result.returncode}: {result.stderr.strip()}"
        )
