from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from src.pipeline.config import AppConfig
from src.pipeline.ffmpeg_utils import (
    encode_video_from_frames,
    extract_frames,
    ffprobe_media,
)
from src.pipeline.inference import colorize_pil_image
from src.pipeline.manifest import write_json_manifest
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths


@dataclass(frozen=True)
class ClipRunRecord:
    input_path: str
    output_path: str
    config_path: str
    render_factor: int
    backend: str
    runtime_seconds: float
    frame_count: int
    fps: str
    width: int
    height: int
    status: str


def run_colorize_clip(
    *,
    config: AppConfig,
    config_path: Path,
    input_path: Path,
    output_path: Path,
    manifest_path: Path | None,
    overwrite: bool,
) -> int:
    paths = resolve_project_paths(config)
    ensure_runtime_directories(paths)

    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input clip not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    media_info = ffprobe_media(input_path)
    bundle = load_colorizer_bundle(config)

    run_hash = sha256(f"{input_path}:{output_path}:{config_path.resolve()}".encode()).hexdigest()[:12]
    frame_root = paths.frames_dir / f"clip_{run_hash}"
    source_frames_dir = frame_root / "source"
    colorized_frames_dir = frame_root / "colorized"

    print(f"Input clip: {input_path}")
    print(f"Output clip: {output_path}")
    print(f"Backend: {bundle.backend}")
    print(f"Render factor: {config.model['render_factor']}")

    started = time.perf_counter()
    extract_frames(input_path=input_path, output_dir=source_frames_dir)

    frame_paths = sorted(source_frames_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError("No frames were extracted from the input clip.")

    colorized_frames_dir.mkdir(parents=True, exist_ok=True)
    previous_smoothed_frame: np.ndarray | None = None
    postprocess_config = config.raw["postprocess"]
    for frame_path in frame_paths:
        input_image = Image.open(frame_path).convert("RGB")
        result = colorize_pil_image(
            model_bundle=bundle,
            input_image=input_image,
            render_factor=int(config.model["render_factor"]),
            postprocess_config=postprocess_config,
        )
        result_np = np.asarray(result)
        if bool(postprocess_config.get("temporal_smoothing", False)):
            result_np, previous_smoothed_frame = apply_temporal_smoothing(
                current_frame=result_np,
                previous_frame=previous_smoothed_frame,
                strength=float(postprocess_config.get("smoothing_strength", 0.0)),
                chroma_threshold=float(postprocess_config.get("smoothing_chroma_threshold", 24.0)),
                adaptive_boost=float(postprocess_config.get("adaptive_smoothing_boost", 0.0)),
            )
        else:
            previous_smoothed_frame = result_np

        Image.fromarray(result_np).save(colorized_frames_dir / frame_path.name)

    encode_video_from_frames(
        frame_dir=colorized_frames_dir,
        output_path=output_path,
        fps=str(media_info["fps"]),
        video_codec=str(config.raw["video"]["output_codec"]),
        crf=int(config.raw["video"]["crf"]),
        pixel_format=str(config.raw["video"]["pixel_format"]),
        audio_input_path=input_path,
    )
    runtime_seconds = time.perf_counter() - started

    record = ClipRunRecord(
        input_path=str(input_path),
        output_path=str(output_path),
        config_path=str(config_path.resolve()),
        render_factor=int(config.model["render_factor"]),
        backend=bundle.backend,
        runtime_seconds=runtime_seconds,
        frame_count=len(frame_paths),
        fps=str(media_info["fps"]),
        width=int(media_info["width"]),
        height=int(media_info["height"]),
        status="succeeded",
    )

    manifest_path = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else paths.manifest_dir / "probe_runs.json"
    )
    update_probe_runs_manifest(manifest_path, record)
    print(f"Clip colorization succeeded in {runtime_seconds:.2f}s")
    print(f"Run manifest updated: {manifest_path}")
    return 0


def update_probe_runs_manifest(manifest_path: Path, record: ClipRunRecord) -> None:
    if manifest_path.exists():
        import json

        payload = json.loads(manifest_path.read_text())
    else:
        payload = {"runs": []}

    payload.setdefault("runs", []).append(asdict(record))
    write_json_manifest(manifest_path, payload)


def apply_temporal_smoothing(
    *,
    current_frame: np.ndarray,
    previous_frame: np.ndarray | None,
    strength: float,
    chroma_threshold: float = 24.0,
    adaptive_boost: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    strength = float(np.clip(strength, 0.0, 1.0))
    if previous_frame is None or strength <= 0.0:
        return current_frame, current_frame

    current_yuv = cv2.cvtColor(current_frame, cv2.COLOR_RGB2YUV).astype(np.float32)
    previous_yuv = cv2.cvtColor(previous_frame, cv2.COLOR_RGB2YUV).astype(np.float32)

    smoothed_yuv = current_yuv.copy()
    chroma_delta = np.abs(current_yuv[:, :, 1:3] - previous_yuv[:, :, 1:3]).mean(axis=2)
    adaptive_component = np.clip(
        (chroma_delta - chroma_threshold) / max(1.0, 255.0 - chroma_threshold),
        0.0,
        1.0,
    ) * float(np.clip(adaptive_boost, 0.0, 1.0))
    blend = np.clip(strength + adaptive_component, 0.0, 0.9)[:, :, None]
    smoothed_yuv[:, :, 1:3] = (
        (1.0 - blend) * current_yuv[:, :, 1:3] + blend * previous_yuv[:, :, 1:3]
    )
    smoothed_rgb = cv2.cvtColor(
        np.clip(smoothed_yuv, 0.0, 255.0).astype(np.uint8),
        cv2.COLOR_YUV2RGB,
    )
    return smoothed_rgb, smoothed_rgb
