from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

from src.pipeline.ffmpeg_utils import ffprobe_media, open_rawvideo_reader, open_rawvideo_writer


DEFAULT_BACKGROUND_STABILIZATION = {
    "enabled": True,
    "strength": 0.85,
    "chroma_damping": 0.55,
    "min_mask_fraction": 0.06,
    "max_chroma": 64.0,
    "highlight_luma_min": 120.0,
    "highlight_max_chroma": 180.0,
    "highlight_neutral_damping": 0.15,
    "highlight_gradient_multiplier": 3.0,
    "shadow_luma_max": 82.0,
    "shadow_max_chroma": 72.0,
    "shadow_neutral_damping": 0.55,
    "shadow_gradient_multiplier": 2.2,
    "edge_min_gradient_multiplier": 3.2,
    "edge_max_chroma": 88.0,
    "edge_neutral_damping": 0.68,
    "static_enabled": True,
    "static_luma_tolerance": 10.0,
    "static_update_alpha": 0.04,
    "static_min_confidence": 0.25,
    "static_chroma_blend": 0.82,
    "static_confidence_gain": 0.08,
    "static_confidence_decay": 0.05,
    "static_reference_update_alpha": 0.01,
    "static_max_chroma": 190.0,
    "max_gradient": 7.5,
    "shot_change_threshold": 18.0,
    "feather_radius": 9,
}


def run_background_stabilize(
    *,
    input_path: Path,
    output_path: Path,
    settings: dict[str, object] | None = None,
    overwrite: bool,
) -> int:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input clip not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    merged_settings = dict(DEFAULT_BACKGROUND_STABILIZATION)
    if settings is not None:
        merged_settings.update(settings)

    media_info = ffprobe_media(input_path)
    started = time.perf_counter()
    frame_stats = _measure_background_chroma(input_path=input_path, media_info=media_info, settings=merged_settings)
    if not frame_stats:
        raise ValueError("No frames available for background stabilization.")

    shot_segments = _build_shot_segments(
        frame_stats,
        shot_change_threshold=float(merged_settings["shot_change_threshold"]),
    )
    targets = _build_shot_targets(frame_stats, shot_segments=shot_segments)
    static_models = _build_static_chroma_models(
        input_path=input_path,
        media_info=media_info,
        settings=merged_settings,
        shot_segments=shot_segments,
    )
    static_model_indexes = _build_static_model_indexes(
        frame_count=len(targets),
        shot_segments=shot_segments,
    )
    frame_count = _apply_background_stabilization(
        input_path=input_path,
        output_path=output_path,
        media_info=media_info,
        settings=merged_settings,
        targets=targets,
        static_models=static_models,
        static_model_indexes=static_model_indexes,
        audio_input_path=input_path,
    )

    print(f"Background chroma stabilization written: {output_path}")
    print(f"Frames: {frame_count}")
    print(f"Runtime seconds: {time.perf_counter() - started:.2f}")
    return 0


def _measure_background_chroma(
    *,
    input_path: Path,
    media_info: dict[str, str | int | float],
    settings: dict[str, object],
) -> list[dict[str, np.ndarray | float]]:
    width = int(media_info["width"])
    height = int(media_info["height"])
    frame_bytes = width * height * 3
    reader = open_rawvideo_reader(input_path=input_path)
    _ensure_pipe(reader.stdout, reader.stderr, "rawvideo reader")

    stats: list[dict[str, np.ndarray | float]] = []
    previous_luma: np.ndarray | None = None
    try:
        while True:
            frame_data = _read_exact_or_none(reader.stdout, frame_bytes)
            if frame_data is None:
                break
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
            small = _analysis_frame(frame)
            lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
            luma = lab[:, :, 0]
            ab = lab[:, :, 1:3].astype(np.float32)
            mask = _background_mask(
                lab=lab,
                max_chroma=float(settings["max_chroma"]),
                highlight_luma_min=float(settings["highlight_luma_min"]),
                highlight_max_chroma=float(settings["highlight_max_chroma"]),
                max_gradient=float(settings["max_gradient"]),
            )
            mask_fraction = float(mask.mean())
            mean_ab = _masked_mean_ab(ab=ab, mask=mask, fallback=np.array([128.0, 128.0], dtype=np.float32))
            luma_delta = 0.0 if previous_luma is None else float(np.mean(np.abs(luma.astype(np.float32) - previous_luma)))
            previous_luma = luma.copy()
            stats.append(
                {
                    "mean_ab": mean_ab,
                    "mask_fraction": mask_fraction,
                    "luma_delta": luma_delta,
                }
            )
    finally:
        _close_pipe(reader.stdout)

    if reader.wait() != 0:
        raise RuntimeError(f"ffmpeg rawvideo reader failed: {_read_stderr(reader)}")
    _close_pipe(reader.stderr)
    return stats


def _build_shot_targets(
    frame_stats: list[dict[str, np.ndarray | float]],
    *,
    shot_segments: list[tuple[int, int]],
) -> list[np.ndarray]:
    targets = [np.array([128.0, 128.0], dtype=np.float32) for _ in frame_stats]
    for start, end in shot_segments:
        segment = frame_stats[start:end]
        usable = [
            np.asarray(stats["mean_ab"], dtype=np.float32)
            for stats in segment
            if float(stats["mask_fraction"]) > 0.0
        ]
        if not usable:
            continue
        target = np.median(np.stack(usable, axis=0), axis=0).astype(np.float32)
        for index in range(start, end):
            targets[index] = target
    return targets


def _build_shot_segments(
    frame_stats: list[dict[str, np.ndarray | float]],
    *,
    shot_change_threshold: float,
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = 0
    for index, stats in enumerate(frame_stats[1:], start=1):
        if float(stats["luma_delta"]) >= shot_change_threshold:
            segments.append((start, index))
            start = index
    segments.append((start, len(frame_stats)))
    return segments


def _build_static_chroma_models(
    *,
    input_path: Path,
    media_info: dict[str, str | int | float],
    settings: dict[str, object],
    shot_segments: list[tuple[int, int]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    if not bool(settings["static_enabled"]):
        return []

    width = int(media_info["width"])
    height = int(media_info["height"])
    frame_bytes = width * height * 3
    segment_by_frame: dict[int, int] = {}
    models: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = [None] * len(shot_segments)
    for segment_index, (start, end) in enumerate(shot_segments):
        for frame_index in range(start, end):
            segment_by_frame[frame_index] = segment_index

    reader = open_rawvideo_reader(input_path=input_path)
    _ensure_pipe(reader.stdout, reader.stderr, "rawvideo reader")
    frame_index = 0
    try:
        while True:
            frame_data = _read_exact_or_none(reader.stdout, frame_bytes)
            if frame_data is None:
                break
            segment_index = segment_by_frame.get(frame_index)
            if segment_index is None:
                frame_index += 1
                continue

            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            luma = lab[:, :, 0].astype(np.float32)
            ab = lab[:, :, 1:3].astype(np.float32)
            model = models[segment_index]
            if model is None:
                reference_luma = luma.copy()
                stable_ab = ab.copy()
                confidence = np.zeros((height, width), dtype=np.float32)
                models[segment_index] = (reference_luma, stable_ab, confidence)
                frame_index += 1
                continue

            reference_luma, stable_ab, confidence = model
            stable_mask = np.abs(luma - reference_luma) <= float(settings["static_luma_tolerance"])
            chroma = np.linalg.norm(ab - 128.0, axis=2)
            stable_mask &= chroma <= float(settings["static_max_chroma"])
            update_alpha = float(settings["static_update_alpha"])
            reference_alpha = float(settings["static_reference_update_alpha"])
            stable_ab[stable_mask] = (1.0 - update_alpha) * stable_ab[stable_mask] + update_alpha * ab[stable_mask]
            reference_luma[stable_mask] = (
                (1.0 - reference_alpha) * reference_luma[stable_mask] + reference_alpha * luma[stable_mask]
            )
            confidence[stable_mask] = np.minimum(
                1.0,
                confidence[stable_mask] + float(settings["static_confidence_gain"]),
            )
            confidence[~stable_mask] = np.maximum(
                0.0,
                confidence[~stable_mask] - float(settings["static_confidence_decay"]),
            )
            frame_index += 1
    finally:
        _close_pipe(reader.stdout)

    if reader.wait() != 0:
        raise RuntimeError(f"ffmpeg rawvideo reader failed: {_read_stderr(reader)}")
    _close_pipe(reader.stderr)
    return models


def _build_static_model_indexes(
    *,
    frame_count: int,
    shot_segments: list[tuple[int, int]],
) -> list[int | None]:
    indexes: list[int | None] = [None] * frame_count
    for segment_index, (start, end) in enumerate(shot_segments):
        for frame_index in range(start, min(end, frame_count)):
            indexes[frame_index] = segment_index
    return indexes


def _apply_background_stabilization(
    *,
    input_path: Path,
    output_path: Path,
    media_info: dict[str, str | int | float],
    settings: dict[str, object],
    targets: list[np.ndarray],
    static_models: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    static_model_indexes: list[int | None],
    audio_input_path: Path,
) -> int:
    width = int(media_info["width"])
    height = int(media_info["height"])
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
        audio_input_path=audio_input_path,
    )
    _ensure_pipe(reader.stdout, reader.stderr, "rawvideo reader")
    _ensure_pipe(writer.stdin, writer.stderr, "rawvideo writer")

    frame_count = 0
    try:
        for target_ab in targets:
            frame_data = _read_exact_or_none(reader.stdout, frame_bytes)
            if frame_data is None:
                break
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
            model_index = static_model_indexes[frame_count] if frame_count < len(static_model_indexes) else None
            static_model = (
                static_models[model_index]
                if model_index is not None and model_index < len(static_models)
                else None
            )
            output_frame = _stabilize_frame(
                frame=frame,
                target_ab=target_ab,
                static_model=static_model,
                settings=settings,
            )
            writer.stdin.write(np.ascontiguousarray(output_frame).tobytes())
            frame_count += 1

        writer.stdin.close()
        writer_returncode = writer.wait()
        reader_returncode = reader.wait()
    finally:
        _close_pipe(reader.stdout)
        _close_pipe(writer.stdin)

    if reader_returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo reader failed: {_read_stderr(reader)}")
    if writer_returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo writer failed: {_read_stderr(writer)}")
    _close_pipe(reader.stderr)
    _close_pipe(writer.stderr)
    return frame_count


def _stabilize_frame(
    *,
    frame: np.ndarray,
    target_ab: np.ndarray,
    static_model: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    settings: dict[str, object],
) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    ab = lab[:, :, 1:3].astype(np.float32)
    lab_float = lab.astype(np.float32)
    mask = _background_mask(
        lab=lab,
        max_chroma=float(settings["max_chroma"]),
        highlight_luma_min=float(settings["highlight_luma_min"]),
        highlight_max_chroma=float(settings["highlight_max_chroma"]),
        max_gradient=float(settings["max_gradient"]),
    )

    if float(mask.mean()) >= float(settings["min_mask_fraction"]):
        mean_ab = _masked_mean_ab(ab=ab, mask=mask, fallback=target_ab)
        strength = float(np.clip(settings["strength"], 0.0, 1.0))
        chroma_damping = float(np.clip(settings["chroma_damping"], 0.0, 1.0))
        shifted = ab + (target_ab - mean_ab) * strength
        damped = target_ab + (shifted - target_ab) * chroma_damping
        stabilized_ab = (1.0 - strength) * shifted + strength * damped

        feather = _feather_mask(mask, radius=int(settings["feather_radius"]))
        lab_float[:, :, 1:3] = ab * (1.0 - feather[:, :, None]) + stabilized_ab * feather[:, :, None]

    if static_model is not None:
        lab_float[:, :, 1:3] = _apply_static_chroma_anchor(
            lab_float=lab_float,
            static_model=static_model,
            settings=settings,
        )

    lab_float[:, :, 1:3] = _neutralize_unstable_chroma(
        lab_float=lab_float,
        settings=settings,
    )
    return cv2.cvtColor(np.clip(lab_float, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _apply_static_chroma_anchor(
    *,
    lab_float: np.ndarray,
    static_model: tuple[np.ndarray, np.ndarray, np.ndarray],
    settings: dict[str, object],
) -> np.ndarray:
    reference_luma, stable_ab, confidence = static_model
    luma = lab_float[:, :, 0]
    ab = lab_float[:, :, 1:3]
    luma_match = np.abs(luma - reference_luma) <= float(settings["static_luma_tolerance"])
    confident = confidence >= float(settings["static_min_confidence"])
    chroma = np.linalg.norm(ab - 128.0, axis=2)
    mask = luma_match & confident & (chroma <= float(settings["static_max_chroma"]))
    if not np.any(mask):
        return ab
    alpha = _feather_mask(mask, radius=max(3, int(settings["feather_radius"]) // 2))
    alpha *= float(np.clip(settings["static_chroma_blend"], 0.0, 1.0))
    return ab * (1.0 - alpha[:, :, None]) + stable_ab * alpha[:, :, None]


def _neutralize_unstable_chroma(
    *,
    lab_float: np.ndarray,
    settings: dict[str, object],
) -> np.ndarray:
    luma = lab_float[:, :, 0].astype(np.uint8)
    ab = lab_float[:, :, 1:3]
    chroma = np.linalg.norm(ab - 128.0, axis=2)
    gradient_x = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    highlight_mask = (
        (luma >= float(settings["highlight_luma_min"]))
        & (chroma <= float(settings["highlight_max_chroma"]))
        & (gradient <= float(settings["max_gradient"]) * float(settings["highlight_gradient_multiplier"]))
    )
    shadow_mask = (
        (luma <= float(settings["shadow_luma_max"]))
        & (chroma <= float(settings["shadow_max_chroma"]))
        & (gradient <= float(settings["max_gradient"]) * float(settings["shadow_gradient_multiplier"]))
    )
    edge_mask = (
        (luma > 20)
        & (luma < 235)
        & (chroma <= float(settings["edge_max_chroma"]))
        & (gradient >= float(settings["max_gradient"]) * float(settings["edge_min_gradient_multiplier"]))
    )

    output_ab = ab
    output_ab = _apply_neutral_damping(
        ab=output_ab,
        mask=highlight_mask,
        damping=float(settings["highlight_neutral_damping"]),
        radius=int(settings["feather_radius"]),
    )
    output_ab = _apply_neutral_damping(
        ab=output_ab,
        mask=shadow_mask,
        damping=float(settings["shadow_neutral_damping"]),
        radius=int(settings["feather_radius"]),
    )
    return _apply_neutral_damping(
        ab=output_ab,
        mask=edge_mask,
        damping=float(settings["edge_neutral_damping"]),
        radius=max(3, int(settings["feather_radius"]) // 2),
    )


def _apply_neutral_damping(
    *,
    ab: np.ndarray,
    mask: np.ndarray,
    damping: float,
    radius: int,
) -> np.ndarray:
    if not np.any(mask):
        return ab
    clipped_damping = float(np.clip(damping, 0.0, 1.0))
    feather = _feather_mask(mask, radius=radius)
    neutral_ab = 128.0 + (ab - 128.0) * clipped_damping
    return ab * (1.0 - feather[:, :, None]) + neutral_ab * feather[:, :, None]


def _background_mask(
    *,
    lab: np.ndarray,
    max_chroma: float,
    highlight_luma_min: float,
    highlight_max_chroma: float,
    max_gradient: float,
) -> np.ndarray:
    luma = lab[:, :, 0]
    ab = lab[:, :, 1:3].astype(np.float32)
    chroma = np.linalg.norm(ab - 128.0, axis=2)
    gradient_x = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    flat = gradient <= max_gradient
    low_chroma_background = chroma <= max_chroma
    bright_overcolored_background = (luma >= highlight_luma_min) & (chroma <= highlight_max_chroma)
    mask = (luma > 35) & (luma < 245) & flat & (low_chroma_background | bright_overcolored_background)
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(bool)


def _feather_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    kernel_size = radius if radius % 2 == 1 else radius + 1
    feather = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
    return np.clip(feather, 0.0, 1.0)


def _analysis_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, 360.0 / float(max(width, height)))
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def _masked_mean_ab(*, ab: np.ndarray, mask: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return fallback.astype(np.float32)
    return ab[mask].mean(axis=0).astype(np.float32)


def _read_exact_or_none(stream, size: int) -> bytes | None:
    buffer = bytearray()
    while len(buffer) < size:
        chunk = stream.read(size - len(buffer))
        if not chunk:
            if not buffer:
                return None
            raise RuntimeError(f"Unexpected rawvideo frame size: {len(buffer)}")
        buffer.extend(chunk)
    return bytes(buffer)


def _ensure_pipe(pipe, stderr, name: str) -> None:
    if pipe is None or stderr is None:
        raise RuntimeError(f"ffmpeg {name} failed to expose required pipes.")


def _close_pipe(pipe) -> None:
    if pipe is not None and not pipe.closed:
        pipe.close()


def _read_stderr(process) -> str:
    if process.stderr is None:
        return ""
    return process.stderr.read().decode().strip()
