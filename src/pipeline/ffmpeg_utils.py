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


def encode_scene_mezzanine(
    *,
    input_path: Path,
    output_path: Path,
    keyframe_times: list[str],
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
    ]
    if keyframe_times:
        command.extend(["-force_key_frames", ",".join(keyframe_times)])
    command.append(str(output_path))
    _run(command)


def copy_clip(
    *,
    input_path: Path,
    output_path: Path,
    start_time: str,
    duration_seconds: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        start_time,
        "-i",
        str(input_path),
        "-t",
        f"{duration_seconds:.3f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]
    _run(command)


def segment_copy_clips(
    *,
    input_path: Path,
    output_pattern: Path,
    segment_times: list[str],
) -> None:
    output_pattern.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c",
        "copy",
        "-f",
        "segment",
        "-reset_timestamps",
        "1",
    ]
    if segment_times:
        command.extend(["-segment_times", ",".join(segment_times)])
    command.append(str(output_pattern))
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
    format_duration = float(payload["format"]["duration"])
    return {
        "duration_seconds": format_duration,
        "video_duration_seconds": _parse_duration(video_stream.get("duration"), format_duration),
        "fps": str(video_stream["avg_frame_rate"]),
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "frame_count": _parse_frame_count(video_stream),
    }


def count_video_frames(path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe frame count failed with code {result.returncode}: {result.stderr.strip()}"
        )
    return int(result.stdout.strip())


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


def normalize_cfr_video(
    *,
    input_path: Path,
    audio_source_path: Path,
    output_path: Path,
    fps: str,
    frame_count: int,
    video_codec: str,
    crf: int,
    pixel_format: str,
    preset: str | None = None,
    audio_bitrate: str = "192k",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps_decimal = fps_to_decimal_string(fps)
    duration_seconds = frame_count / float(fps_decimal)
    video_filter = (
        f"fps=fps={fps_decimal},"
        "tpad=stop_mode=clone:stop_duration=1,"
        f"trim=end_frame={frame_count},"
        f"setpts=N/({fps_decimal}*TB)"
    )
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-i",
        str(audio_source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-vf",
        video_filter,
        "-fps_mode",
        "cfr",
        "-c:v",
        video_codec,
    ]
    if preset is not None:
        command.extend(["-preset", preset])
    command.extend(
        [
            "-crf",
            str(crf),
            "-pix_fmt",
            pixel_format,
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-af",
            "apad",
            "-t",
            f"{duration_seconds:.6f}",
            str(output_path),
        ]
    )
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
        "-fps_mode",
        "cfr",
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


def _parse_duration(value: object, fallback: float) -> float:
    if value is None or value == "N/A":
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg command failed with code {result.returncode}: {result.stderr.strip()}"
        )
