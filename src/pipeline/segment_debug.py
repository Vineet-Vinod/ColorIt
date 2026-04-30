from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from src.pipeline.ffmpeg_utils import ffprobe_media, open_rawvideo_reader, open_rawvideo_writer
from src.pipeline.segments import load_segment_manifest, resolve_manifest_path


def run_render_segment_debug(
    *,
    input_path: Path,
    segment_manifest_path: Path,
    output_path: Path,
    include_labels: list[str],
    include_tracks: list[str],
    exclude_labels: list[str],
    alpha: float,
    overwrite: bool,
) -> int:
    input_path = input_path.expanduser().resolve()
    segment_manifest_path = segment_manifest_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input clip not found: {input_path}")
    if not segment_manifest_path.exists():
        raise FileNotFoundError(f"Segment manifest not found: {segment_manifest_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    manifest = load_segment_manifest(segment_manifest_path)
    media_info = ffprobe_media(input_path)
    width = int(media_info["width"])
    height = int(media_info["height"])
    frame_bytes = width * height * 3
    if width != int(manifest["width"]) or height != int(manifest["height"]):
        raise ValueError(
            f"Input clip dimensions {width}x{height} do not match manifest "
            f"{manifest['width']}x{manifest['height']}."
        )

    frames_by_index = {
        int(frame["frame_index"]): frame
        for frame in manifest["frames"]
    }
    include_label_set = set(include_labels)
    include_track_set = set(include_tracks)
    exclude_label_set = set(exclude_labels)

    reader = open_rawvideo_reader(input_path=input_path)
    writer = open_rawvideo_writer(
        output_path=output_path,
        width=width,
        height=height,
        fps=str(media_info["fps"]),
        video_codec="libx264",
        crf=18,
        pixel_format="yuv420p",
        audio_input_path=None,
    )
    if reader.stdout is None or reader.stderr is None:
        raise RuntimeError("ffmpeg rawvideo reader failed to expose stdout/stderr pipes.")
    if writer.stdin is None or writer.stderr is None:
        raise RuntimeError("ffmpeg rawvideo writer failed to expose stdin/stderr pipes.")

    frame_index = 0
    try:
        while True:
            frame_data = reader.stdout.read(frame_bytes)
            if not frame_data:
                break
            if len(frame_data) != frame_bytes:
                raise RuntimeError(
                    f"Unexpected end of rawvideo stream; expected {frame_bytes} bytes, got {len(frame_data)}."
                )
            frame_rgb = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
            overlay_rgb = render_segment_overlay_frame(
                frame_rgb=frame_rgb,
                frame_payload=frames_by_index.get(frame_index, {"instances": []}),
                manifest_path=segment_manifest_path,
                include_labels=include_label_set,
                include_tracks=include_track_set,
                exclude_labels=exclude_label_set,
                alpha=alpha,
            )
            writer.stdin.write(np.ascontiguousarray(overlay_rgb).tobytes())
            frame_index += 1

        writer.stdin.close()
        writer_returncode = writer.wait()
        reader_returncode = reader.wait()
    finally:
        if reader.stdout is not None:
            reader.stdout.close()
        if writer.stdin is not None:
            writer.stdin.close()

    if reader_returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo reader failed: {reader.stderr.read().decode().strip()}")
    if writer_returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo writer failed: {writer.stderr.read().decode().strip()}")
    reader.stderr.close()
    writer.stderr.close()

    print(f"Segment debug overlay written: {output_path}")
    return 0


def render_segment_overlay_frame(
    *,
    frame_rgb: np.ndarray,
    frame_payload: dict,
    manifest_path: Path,
    include_labels: set[str],
    include_tracks: set[str],
    exclude_labels: set[str],
    alpha: float,
) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    output = frame_rgb.copy()
    for instance in _filtered_instances(
        frame_payload.get("instances", []),
        include_labels=include_labels,
        include_tracks=include_tracks,
        exclude_labels=exclude_labels,
    ):
        mask_path = resolve_manifest_path(
            manifest_path=manifest_path,
            relative_path=str(instance["mask_path"]),
        )
        mask = np.asarray(Image.open(mask_path).convert("L"))
        if mask.shape[:2] != output.shape[:2]:
            mask = cv2.resize(mask, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_NEAREST)
        selected = mask > 0
        if not np.any(selected):
            continue
        color = np.array(_track_color(str(instance["track_id"])), dtype=np.float32)
        output[selected] = (
            (1.0 - alpha) * output[selected].astype(np.float32) + alpha * color
        ).astype(np.uint8)
    return output


def _filtered_instances(
    instances: Iterable[dict],
    *,
    include_labels: set[str],
    include_tracks: set[str],
    exclude_labels: set[str],
) -> Iterable[dict]:
    for instance in instances:
        label = str(instance.get("label", ""))
        track_id = str(instance.get("track_id", ""))
        if include_labels and label not in include_labels:
            continue
        if include_tracks and track_id not in include_tracks:
            continue
        if exclude_labels and label in exclude_labels:
            continue
        yield instance


def _track_color(track_id: str) -> tuple[int, int, int]:
    colors = [
        (230, 45, 68),
        (39, 110, 241),
        (28, 158, 92),
        (235, 153, 34),
        (164, 83, 214),
        (34, 185, 210),
        (225, 77, 166),
    ]
    index = sum(ord(character) for character in track_id) % len(colors)
    return colors[index]
