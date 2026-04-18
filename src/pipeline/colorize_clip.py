from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
import cv2
import numpy as np

from src.pipeline.config import AppConfig
from src.pipeline.ffmpeg_utils import ffprobe_media, open_rawvideo_reader, open_rawvideo_writer
from src.pipeline.inference import colorize_rgb_batch
from src.pipeline.manifest import load_json_manifest, utc_now_iso, write_json_manifest
from src.pipeline.model_loader import ModelBundle, load_colorizer_bundle
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
    model_bundle: ModelBundle | None = None,
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
    bundle = model_bundle or load_colorizer_bundle(config)

    print(f"Input clip: {input_path}")
    print(f"Output clip: {output_path}")
    print(f"Backend: {bundle.backend}")
    print(f"Render factor: {config.model['render_factor']}")
    print("Frame transport: pipe")

    started = time.perf_counter()
    frame_count = _run_pipe_transport(
        input_path=input_path,
        output_path=output_path,
        media_info=media_info,
        config=config,
        bundle=bundle,
    )
    runtime_seconds = time.perf_counter() - started

    record = ClipRunRecord(
        input_path=str(input_path),
        output_path=str(output_path),
        config_path=str(config_path.expanduser().resolve()),
        render_factor=int(config.model["render_factor"]),
        backend=bundle.backend,
        runtime_seconds=runtime_seconds,
        frame_count=frame_count,
        fps=str(media_info["fps"]),
        width=int(media_info["width"]),
        height=int(media_info["height"]),
        status="succeeded",
    )
    manifest_destination = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else paths.manifest_dir / "clip_runs.json"
    )
    update_clip_runs_manifest(manifest_destination, record)

    print(f"Clip colorization succeeded in {runtime_seconds:.2f}s")
    print(f"Run manifest updated: {manifest_destination}")
    return 0


def _run_pipe_transport(
    *,
    input_path: Path,
    output_path: Path,
    media_info: dict[str, str | int | float],
    config: AppConfig,
    bundle: ModelBundle,
) -> int:
    width = int(media_info["width"])
    height = int(media_info["height"])
    frame_bytes = width * height * 3
    previous_smoothed_frame: np.ndarray | None = None
    frame_count = 0
    postprocess_config = config.raw["postprocess"]

    reader = open_rawvideo_reader(input_path=input_path)
    writer = open_rawvideo_writer(
        output_path=output_path,
        width=width,
        height=height,
        fps=str(media_info["fps"]),
        video_codec=str(config.raw["video"]["output_codec"]),
        crf=int(config.raw["video"]["crf"]),
        pixel_format=str(config.raw["video"]["pixel_format"]),
        audio_input_path=input_path,
    )
    if reader.stdout is None or reader.stderr is None:
        raise RuntimeError("ffmpeg rawvideo reader failed to expose stdout/stderr pipes.")
    if writer.stdin is None or writer.stderr is None:
        raise RuntimeError("ffmpeg rawvideo writer failed to expose stdin/stderr pipes.")

    inference_batch_size = max(1, int(config.runtime.get("inference_batch_size", 1)))
    try:
        while True:
            batch_frames: list[np.ndarray] = []
            for _ in range(inference_batch_size):
                frame_data = _read_exact(reader.stdout, frame_bytes)
                if frame_data is None:
                    break
                batch_frames.append(
                    np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
                )
            if not batch_frames:
                break

            result_batch = colorize_rgb_batch(
                model_bundle=bundle,
                input_rgbs=batch_frames,
                render_factor=int(config.model["render_factor"]),
                postprocess_config=postprocess_config,
            )
            for result_np in result_batch:
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

                writer.stdin.write(np.ascontiguousarray(result_np).tobytes())
                frame_count += 1

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
    return frame_count


def _read_exact(stream, size: int) -> bytes | None:
    buffer = bytearray()
    while len(buffer) < size:
        chunk = stream.read(size - len(buffer))
        if not chunk:
            if not buffer:
                return None
            raise RuntimeError(
                f"Unexpected end of rawvideo stream; expected {size} bytes, got {len(buffer)}."
            )
        buffer.extend(chunk)
    return bytes(buffer)


def update_clip_runs_manifest(manifest_path: Path, record: ClipRunRecord) -> None:
    payload = load_json_manifest(manifest_path, {"runs": [], "updated_at": utc_now_iso()})
    if not isinstance(payload, dict):
        payload = {"runs": []}

    payload["updated_at"] = utc_now_iso()
    runs = payload.setdefault("runs", [])
    serialized = asdict(record)
    for index, existing in enumerate(runs):
        if existing.get("output_path") == record.output_path:
            runs[index] = serialized
            break
    else:
        runs.append(serialized)
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
