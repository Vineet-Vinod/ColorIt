from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.pipeline.ffmpeg_utils import ffprobe_media, open_rawvideo_reader, open_rawvideo_writer
from src.pipeline.segments import load_segment_manifest, resolve_manifest_path


def run_recolor_segments(
    *,
    input_path: Path,
    segment_manifest_path: Path,
    output_path: Path,
    color_hex: str,
    include_labels: list[str],
    chroma_blend: float,
    mask_erode_px: int,
    mask_feather_px: int,
    temporal_mask_blend: float,
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
    if not include_label_set:
        include_label_set = {str(track["label"]) for track in manifest["tracks"].values()}

    frame_bytes = width * height * 3
    reader = open_rawvideo_reader(input_path=input_path)
    writer = open_rawvideo_writer(
        output_path=output_path,
        width=width,
        height=height,
        fps=str(media_info["fps"]),
        video_codec="libx264",
        crf=16,
        pixel_format="yuv420p",
        audio_input_path=input_path,
    )
    if reader.stdout is None or reader.stderr is None:
        raise RuntimeError("ffmpeg rawvideo reader failed to expose stdout/stderr pipes.")
    if writer.stdin is None or writer.stderr is None:
        raise RuntimeError("ffmpeg rawvideo writer failed to expose stdin/stderr pipes.")

    previous_alpha: np.ndarray | None = None
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
            alpha = _frame_alpha(
                frame_payload=frames_by_index.get(frame_index, {"instances": []}),
                segment_manifest_path=segment_manifest_path,
                include_labels=include_label_set,
                width=width,
                height=height,
                erode_px=mask_erode_px,
                feather_px=mask_feather_px,
            )
            if previous_alpha is not None and temporal_mask_blend > 0.0:
                blend = float(np.clip(temporal_mask_blend, 0.0, 0.95))
                alpha = (1.0 - blend) * alpha + blend * previous_alpha
            previous_alpha = alpha

            recolored = _apply_lab_chroma(
                base_rgb=frame_rgb,
                target_hex=color_hex,
                mask_alpha=alpha,
                chroma_blend=chroma_blend,
            )
            writer.stdin.write(np.ascontiguousarray(recolored).tobytes())
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

    print(f"Segment recolor written: {output_path}")
    return 0


def _frame_alpha(
    *,
    frame_payload: dict,
    segment_manifest_path: Path,
    include_labels: set[str],
    width: int,
    height: int,
    erode_px: int,
    feather_px: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for instance in frame_payload.get("instances", []):
        if str(instance.get("label", "")) not in include_labels:
            continue
        mask_path = resolve_manifest_path(
            manifest_path=segment_manifest_path,
            relative_path=str(instance["mask_path"]),
        )
        instance_mask = np.asarray(Image.open(mask_path).convert("L"))
        if instance_mask.shape[:2] != (height, width):
            instance_mask = cv2.resize(instance_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = cv2.bitwise_or(mask, np.where(instance_mask > 0, 255, 0).astype(np.uint8))

    alpha = mask.astype(np.float32) / 255.0
    if erode_px > 0:
        alpha = cv2.erode(alpha, _kernel(erode_px), iterations=1)
    if feather_px > 0:
        kernel_size = feather_px * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), 0)
    return np.clip(alpha, 0.0, 1.0)


def _apply_lab_chroma(
    *,
    base_rgb: np.ndarray,
    target_hex: str,
    mask_alpha: np.ndarray,
    chroma_blend: float,
) -> np.ndarray:
    blend = np.clip(mask_alpha * float(np.clip(chroma_blend, 0.0, 1.0)), 0.0, 1.0)
    if not np.any(blend > 0.0):
        return base_rgb

    base_lab = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_rgb = np.array([[_hex_to_rgb(target_hex)]], dtype=np.uint8)
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    base_lab[:, :, 1] = (1.0 - blend) * base_lab[:, :, 1] + blend * target_lab[1]
    base_lab[:, :, 2] = (1.0 - blend) * base_lab[:, :, 2] + blend * target_lab[2]
    return cv2.cvtColor(np.clip(base_lab, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB color, got: {value}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _kernel(radius_px: int) -> np.ndarray:
    size = max(1, radius_px * 2 + 1)
    return np.ones((size, size), dtype=np.uint8)
