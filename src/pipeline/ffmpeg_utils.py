from __future__ import annotations

import json
import subprocess
from pathlib import Path

MAX_OUTPUT_FPS = 30.0


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
        "frame_count": _parse_frame_count(video_stream),
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


def open_rawvideo_reader(
    *,
    input_path: Path,
) -> subprocess.Popen[bytes]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(input_path),
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def open_rawvideo_writer(
    *,
    output_path: Path,
    width: int,
    height: int,
    fps: str,
    video_codec: str,
    crf: int,
    pixel_format: str,
    preset: str | None = None,
    audio_input_path: Path | None = None,
) -> subprocess.Popen[bytes]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        fps_to_decimal_string(fps),
        "-i",
        "-",
    ]
    if audio_input_path is not None:
        command.extend(["-i", str(audio_input_path), "-map", "0:v:0", "-map", "1:a:0?"])
    fps_filter = output_fps_filter(fps)
    if fps_filter is not None:
        command.extend(["-vf", fps_filter])
    command.extend(
        [
            "-c:v",
            video_codec,
        ]
    )
    if preset is not None:
        command.extend(["-preset", preset])
    command.extend(
        [
            "-crf",
            str(crf),
            "-pix_fmt",
            pixel_format,
        ]
    )
    if audio_input_path is not None:
        command.extend(["-c:a", "aac", "-b:a", "192k", "-af", "apad", "-shortest"])
    command.append(str(output_path))
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


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


def output_fps_filter(value: str) -> str | None:
    fps = float(fps_to_decimal_string(value))
    if fps > MAX_OUTPUT_FPS:
        return f"fps={int(MAX_OUTPUT_FPS)}"
    return None


def _parse_frame_count(video_stream: dict) -> int:
    frame_count = video_stream.get("nb_frames")
    if frame_count is None or frame_count == "N/A":
        return 0
    try:
        return int(frame_count)
    except ValueError:
        return 0


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg command failed with code {result.returncode}: {result.stderr.strip()}"
        )
