from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.pipeline.ffmpeg_utils import ffprobe_media, open_rawvideo_reader
from src.pipeline.segments import (
    SegmentFrame,
    SegmentInstance,
    SegmentManifest,
    SegmentTrack,
    mask_bbox,
    write_segment_manifest,
)


def run_polygon_segmentation(
    *,
    input_path: Path,
    tracks_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> SegmentManifest:
    input_path = input_path.expanduser().resolve()
    tracks_path = tracks_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input clip not found: {input_path}")
    if not tracks_path.exists():
        raise FileNotFoundError(f"Polygon track file not found: {tracks_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Segment output dir is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    track_payloads = _load_polygon_tracks(tracks_path)
    media_info = ffprobe_media(input_path)
    width = int(media_info["width"])
    height = int(media_info["height"])
    fps = str(media_info["fps"])
    frame_bytes = width * height * 3

    tracks = {
        str(track["track_id"]): SegmentTrack(
            track_id=str(track["track_id"]),
            label=str(track.get("label", "segment")),
            kind=str(track.get("kind", "polygon")),
            parent_track_id=(
                str(track["parent_track_id"])
                if track.get("parent_track_id") is not None
                else None
            ),
            confidence=(
                float(track["confidence"])
                if track.get("confidence") is not None
                else None
            ),
            metadata=_track_metadata(track),
        )
        for track in track_payloads
    }

    reader = open_rawvideo_reader(input_path=input_path)
    if reader.stdout is None or reader.stderr is None:
        raise RuntimeError("ffmpeg rawvideo reader failed to expose stdout/stderr pipes.")

    frames: list[SegmentFrame] = []
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

            instances: list[SegmentInstance] = []
            for track in track_payloads:
                mask = _render_track_mask(
                    track=track,
                    frame_rgb=frame_rgb,
                    frame_index=frame_index,
                    width=width,
                    height=height,
                )
                if mask is None or not np.any(mask):
                    continue

                track_id = str(track["track_id"])
                mask_path = output_dir / "masks" / f"frame_{frame_index:06d}_{track_id}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(mask_path)
                instances.append(
                    SegmentInstance(
                        track_id=track_id,
                        label=str(track.get("label", "segment")),
                        kind=str(track.get("kind", "polygon")),
                        parent_track_id=(
                            str(track["parent_track_id"])
                            if track.get("parent_track_id") is not None
                            else None
                        ),
                        mask_path=str(mask_path.relative_to(output_dir)),
                        bbox=mask_bbox(mask),
                        confidence=float(track.get("confidence", 1.0)),
                        metadata=_instance_metadata(track),
                    )
                )

            frames.append(SegmentFrame(frame_index=frame_index, instances=instances))
            frame_index += 1
        reader_returncode = reader.wait()
    finally:
        if reader.stdout is not None:
            reader.stdout.close()

    if reader_returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo reader failed: {reader.stderr.read().decode().strip()}")

    manifest = SegmentManifest(
        source_clip=str(input_path),
        backend="polygon",
        frame_count=frame_index,
        width=width,
        height=height,
        fps=fps,
        tracks=tracks,
        frames=frames,
        metadata={"source_tracks": str(tracks_path)},
    )
    write_segment_manifest(output_dir / "segment_manifest.json", manifest)
    return manifest


def _load_polygon_tracks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    tracks = payload.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError(f"Polygon track file must contain a tracks list: {path}")
    for track in tracks:
        if not isinstance(track, dict):
            raise ValueError(f"Polygon track entries must be objects: {path}")
        if not track.get("track_id"):
            raise ValueError(f"Polygon track is missing track_id: {path}")
        if not isinstance(track.get("keyframes"), list) or not track["keyframes"]:
            raise ValueError(f"Polygon track is missing keyframes: {track.get('track_id')}")
    return tracks


def _render_track_mask(
    *,
    track: dict[str, Any],
    frame_rgb: np.ndarray,
    frame_index: int,
    width: int,
    height: int,
) -> np.ndarray | None:
    keyframes = track["keyframes"]
    first = keyframes[0]
    last = keyframes[-1]
    if frame_index < int(first["frame"]) or frame_index > int(last["frame"]):
        return None

    lower = first
    upper = last
    for index, keyframe in enumerate(keyframes):
        if int(keyframe["frame"]) <= frame_index:
            lower = keyframe
        if int(keyframe["frame"]) >= frame_index:
            upper = keyframe
            break
        if index == len(keyframes) - 1:
            upper = keyframe

    polygon = _interpolate_polygon(
        lower=lower,
        upper=upper,
        frame_index=frame_index,
        width=width,
        height=height,
    )
    if polygon is None:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    if str(track.get("refine", "")).lower() == "grabcut":
        mask = _refine_with_grabcut(
            frame_rgb=frame_rgb,
            seed_mask=mask,
            sure_foreground_erode_px=int(track.get("grabcut_sure_fg_erode_px", 18)),
            iterations=int(track.get("grabcut_iterations", 3)),
        )

    blur_px = int(track.get("preblur_px", 0))
    if blur_px > 0:
        kernel_size = blur_px * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    return mask


def _refine_with_grabcut(
    *,
    frame_rgb: np.ndarray,
    seed_mask: np.ndarray,
    sure_foreground_erode_px: int,
    iterations: int,
) -> np.ndarray:
    if not np.any(seed_mask):
        return seed_mask

    grabcut_mask = np.full(seed_mask.shape, cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[seed_mask > 0] = cv2.GC_PR_FGD

    if sure_foreground_erode_px > 0:
        kernel_size = sure_foreground_erode_px * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        sure_foreground = cv2.erode(seed_mask, kernel, iterations=1)
        grabcut_mask[sure_foreground > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.grabCut(
        frame_bgr,
        grabcut_mask,
        None,
        bgd_model,
        fgd_model,
        max(1, iterations),
        cv2.GC_INIT_WITH_MASK,
    )
    selected = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    return cv2.bitwise_and(selected, seed_mask)


def _interpolate_polygon(
    *,
    lower: dict[str, Any],
    upper: dict[str, Any],
    frame_index: int,
    width: int,
    height: int,
) -> np.ndarray | None:
    lower_points = lower.get("points")
    upper_points = upper.get("points")
    if not isinstance(lower_points, list) or not isinstance(upper_points, list):
        return None
    if len(lower_points) != len(upper_points):
        raise ValueError("Interpolated polygon keyframes must have the same point count.")

    lower_frame = int(lower["frame"])
    upper_frame = int(upper["frame"])
    if lower_frame == upper_frame:
        weight = 0.0
    else:
        weight = (frame_index - lower_frame) / float(upper_frame - lower_frame)

    points = []
    for lower_point, upper_point in zip(lower_points, upper_points, strict=True):
        lx, ly = _scale_point(lower_point, width, height)
        ux, uy = _scale_point(upper_point, width, height)
        points.append(
            [
                int(round((1.0 - weight) * lx + weight * ux)),
                int(round((1.0 - weight) * ly + weight * uy)),
            ]
        )
    return np.asarray(points, dtype=np.int32)


def _scale_point(point: list[float] | tuple[float, float], width: int, height: int) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError(f"Polygon points must be [x, y], got: {point}")
    x = float(point[0])
    y = float(point[1])
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return x * width, y * height
    return x, y


def _track_metadata(track: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in track.items()
        if key
        not in {
            "track_id",
            "label",
            "kind",
            "parent_track_id",
            "confidence",
            "keyframes",
        }
    }


def _instance_metadata(track: dict[str, Any]) -> dict[str, Any]:
    metadata = _track_metadata(track)
    color = track.get("color")
    if color is not None:
        metadata["color"] = str(color)
    return metadata
