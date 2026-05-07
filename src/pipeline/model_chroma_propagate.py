from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.pipeline.ffmpeg_utils import ffprobe_media, open_rawvideo_reader, open_rawvideo_writer


def run_model_chroma_propagate(
    *,
    source_path: Path,
    model_color_path: Path,
    output_path: Path,
    keyframe_stride: int,
    chroma_blend: float,
    fallback_color_hex: str | None,
    fallback_strength: float,
    fallback_uncertainty: str,
    disagreement_start: float,
    disagreement_end: float,
    overwrite: bool,
) -> int:
    source_path = source_path.expanduser().resolve()
    model_color_path = model_color_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source clip not found: {source_path}")
    if not model_color_path.exists():
        raise FileNotFoundError(f"Model color clip not found: {model_color_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    fallback_ab = _hex_to_lab_ab(fallback_color_hex) if fallback_color_hex else None

    source_info = ffprobe_media(source_path)
    model_info = ffprobe_media(model_color_path)
    width = int(source_info["width"])
    height = int(source_info["height"])
    if width != int(model_info["width"]) or height != int(model_info["height"]):
        raise ValueError("Source and model color clips must have matching dimensions.")

    source_frames = _read_frames(source_path, width=width, height=height)
    model_frames = _read_frames(model_color_path, width=width, height=height)
    frame_count = min(len(source_frames), len(model_frames))
    source_frames = source_frames[:frame_count]
    model_frames = model_frames[:frame_count]
    if frame_count == 0:
        raise ValueError("No frames available for chroma propagation.")

    keyframe_indices = list(range(0, frame_count, max(1, keyframe_stride)))
    if keyframe_indices[-1] != frame_count - 1:
        keyframe_indices.append(frame_count - 1)
    output_frames = _propagate_chroma(
        source_frames=source_frames,
        model_frames=model_frames,
        keyframe_indices=keyframe_indices,
        chroma_blend=chroma_blend,
        fallback_ab=fallback_ab,
        fallback_strength=fallback_strength,
        fallback_uncertainty=fallback_uncertainty,
        disagreement_start=disagreement_start,
        disagreement_end=disagreement_end,
    )
    _write_frames(
        output_path=output_path,
        frames=output_frames,
        width=width,
        height=height,
        fps=str(source_info["fps"]),
        audio_input_path=source_path,
    )
    print(f"Model chroma propagation written: {output_path}")
    print(f"Frames: {frame_count}")
    print(f"Keyframes: {len(keyframe_indices)}")
    return 0


def _propagate_chroma(
    *,
    source_frames: list[np.ndarray],
    model_frames: list[np.ndarray],
    keyframe_indices: list[int],
    chroma_blend: float,
    fallback_ab: np.ndarray | None,
    fallback_strength: float,
    fallback_uncertainty: str,
    disagreement_start: float,
    disagreement_end: float,
) -> list[np.ndarray]:
    if fallback_uncertainty not in {"ab-delta", "hue"}:
        raise ValueError(f"Unsupported fallback uncertainty mode: {fallback_uncertainty}")
    source_gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in source_frames]
    source_l = [cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)[:, :, :1].astype(np.float32) for frame in source_frames]
    model_ab_by_key = {
        index: cv2.cvtColor(model_frames[index], cv2.COLOR_RGB2LAB)[:, :, 1:3].astype(np.float32)
        for index in keyframe_indices
    }

    forward_ab: dict[int, np.ndarray] = {}
    for start, end in zip(keyframe_indices, keyframe_indices[1:]):
        ab = model_ab_by_key[start]
        forward_ab[start] = ab
        for frame_index in range(start + 1, end + 1):
            ab = _warp_ab(previous_ab=ab, previous_gray=source_gray[frame_index - 1], current_gray=source_gray[frame_index])
            forward_ab[frame_index] = ab

    backward_ab: dict[int, np.ndarray] = {}
    for start, end in zip(reversed(keyframe_indices[:-1]), reversed(keyframe_indices[1:])):
        ab = model_ab_by_key[end]
        backward_ab[end] = ab
        for frame_index in range(end - 1, start - 1, -1):
            ab = _warp_ab(previous_ab=ab, previous_gray=source_gray[frame_index + 1], current_gray=source_gray[frame_index])
            backward_ab[frame_index] = ab

    output_frames: list[np.ndarray] = []
    blend = float(np.clip(chroma_blend, 0.0, 1.0))
    for frame_index in range(len(source_frames)):
        left_key = max(index for index in keyframe_indices if index <= frame_index)
        right_key = min(index for index in keyframe_indices if index >= frame_index)
        if left_key == right_key:
            propagated_ab = model_ab_by_key[left_key]
        else:
            t = (frame_index - left_key) / max(right_key - left_key, 1)
            forward = forward_ab[frame_index]
            backward = backward_ab[frame_index]
            propagated_ab = (1.0 - t) * forward + t * backward
            if fallback_ab is not None and fallback_strength > 0.0:
                disagreement = _chroma_disagreement(forward=forward, backward=backward, mode=fallback_uncertainty)
                uncertainty = _smoothstep(disagreement_start, disagreement_end, disagreement)
                uncertainty = cv2.GaussianBlur(uncertainty.astype(np.float32), (0, 0), 2.0)
                fallback_mix = np.clip(uncertainty * fallback_strength, 0.0, 1.0)[:, :, None]
                propagated_ab = (1.0 - fallback_mix) * propagated_ab + fallback_mix * fallback_ab
        deoldify_ab = cv2.cvtColor(source_frames[frame_index], cv2.COLOR_RGB2LAB)[:, :, 1:3].astype(np.float32)
        output_ab = (1.0 - blend) * deoldify_ab + blend * propagated_ab
        output_lab = np.concatenate([source_l[frame_index], output_ab], axis=2)
        output_rgb = cv2.cvtColor(np.clip(output_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        output_frames.append(output_rgb)
    return output_frames


def _chroma_disagreement(*, forward: np.ndarray, backward: np.ndarray, mode: str) -> np.ndarray:
    if mode == "ab-delta":
        return np.linalg.norm(forward - backward, axis=2)

    forward_centered = forward - 128.0
    backward_centered = backward - 128.0
    forward_norm = np.linalg.norm(forward_centered, axis=2)
    backward_norm = np.linalg.norm(backward_centered, axis=2)
    norm_product = np.maximum(forward_norm * backward_norm, 1e-6)
    cosine = np.sum(forward_centered * backward_centered, axis=2) / norm_product
    hue_angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    low_chroma = 1.0 - _smoothstep(8.0, 24.0, np.minimum(forward_norm, backward_norm))
    return np.maximum(hue_angle, low_chroma * 45.0)


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    value = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _hex_to_lab_ab(value: str | None) -> np.ndarray:
    if value is None:
        raise ValueError("Expected a #RRGGBB fallback color.")
    value = value.strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB fallback color, got: {value}")
    rgb = np.array([[[int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)]]], dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    return lab[1:3]


def _warp_ab(*, previous_ab: np.ndarray, previous_gray: np.ndarray, current_gray: np.ndarray) -> np.ndarray:
    current_to_previous_flow = cv2.calcOpticalFlowFarneback(
        current_gray,
        previous_gray,
        None,
        0.5,
        3,
        21,
        3,
        5,
        1.2,
        0,
    )
    height, width = current_gray.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x + current_to_previous_flow[:, :, 0]
    map_y = grid_y + current_to_previous_flow[:, :, 1]
    channels = [
        cv2.remap(
            previous_ab[:, :, channel],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        for channel in range(2)
    ]
    return np.stack(channels, axis=2)


def _read_frames(path: Path, *, width: int, height: int) -> list[np.ndarray]:
    frame_bytes = width * height * 3
    reader = open_rawvideo_reader(input_path=path)
    if reader.stdout is None or reader.stderr is None:
        raise RuntimeError("ffmpeg rawvideo reader failed to expose stdout/stderr pipes.")
    frames: list[np.ndarray] = []
    try:
        while True:
            frame_data = reader.stdout.read(frame_bytes)
            if not frame_data:
                break
            if len(frame_data) != frame_bytes:
                raise RuntimeError(f"Unexpected rawvideo frame size: {len(frame_data)}")
            frames.append(np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3)).copy())
        returncode = reader.wait()
    finally:
        if reader.stdout is not None:
            reader.stdout.close()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo reader failed: {reader.stderr.read().decode().strip()}")
    if reader.stderr is not None:
        reader.stderr.close()
    return frames


def _write_frames(
    *,
    output_path: Path,
    frames: list[np.ndarray],
    width: int,
    height: int,
    fps: str,
    audio_input_path: Path,
) -> None:
    writer = open_rawvideo_writer(
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
        video_codec="libx264",
        crf=16,
        pixel_format="yuv420p",
        audio_input_path=audio_input_path,
    )
    if writer.stdin is None or writer.stderr is None:
        raise RuntimeError("ffmpeg rawvideo writer failed to expose stdin/stderr pipes.")
    try:
        for frame in frames:
            writer.stdin.write(np.ascontiguousarray(frame).tobytes())
        writer.stdin.close()
        returncode = writer.wait()
    finally:
        if writer.stdin is not None:
            writer.stdin.close()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo writer failed: {writer.stderr.read().decode().strip()}")
    writer.stderr.close()
