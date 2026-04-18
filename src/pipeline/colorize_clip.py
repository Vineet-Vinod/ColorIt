from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image

from src.pipeline.config import AppConfig
from src.pipeline.costume_palette import CostumePaletteRuntime, build_costume_palette_runtime
from src.pipeline.ffmpeg_utils import (
    encode_video_from_frames,
    extract_frames,
    ffprobe_media,
    open_rawvideo_reader,
    open_rawvideo_writer,
)
from src.pipeline.inference import (
    colorize_rgb_batch,
    colorize_rgb_batch_profiled,
    colorize_pil_image,
    colorize_pil_image_profiled,
    colorize_rgb_frame,
    colorize_rgb_frame_profiled,
)
from src.pipeline.manifest import write_json_manifest
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
    frame_transport: str
    status: str


@dataclass(frozen=True)
class ClipStageProfile:
    model_load_seconds: float
    frame_extract_seconds: float
    frame_decode_seconds: float
    inference_preprocess_seconds: float
    inference_preprocess_upload_seconds: float
    inference_model_seconds: float
    inference_postprocess_seconds: float
    inference_postprocess_upload_seconds: float
    inference_postprocess_graph_seconds: float
    inference_postprocess_download_seconds: float
    inference_postprocess_cpu_seconds: float
    costume_palette_seconds: float
    temporal_smoothing_seconds: float
    frame_save_seconds: float
    encode_seconds: float
    cleanup_seconds: float
    total_runtime_seconds: float
    process_cpu_seconds: float
    effective_fps: float
    per_frame_seconds: float


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
    result = run_colorize_clip_profiled(
        config=config,
        config_path=config_path,
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        overwrite=overwrite,
        collect_profile=False,
        model_bundle=model_bundle,
    )
    print(f"Clip colorization succeeded in {result.run_record.runtime_seconds:.2f}s")
    print(f"Run manifest updated: {result.manifest_path}")
    return 0


@dataclass(frozen=True)
class ClipExecutionResult:
    run_record: ClipRunRecord
    stage_profile: ClipStageProfile | None
    manifest_path: Path


def run_colorize_clip_profiled(
    *,
    config: AppConfig,
    config_path: Path,
    input_path: Path,
    output_path: Path,
    manifest_path: Path | None,
    overwrite: bool,
    collect_profile: bool,
    model_bundle: ModelBundle | None = None,
) -> ClipExecutionResult:
    paths = resolve_project_paths(config)
    ensure_runtime_directories(paths)

    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input clip not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    media_info = ffprobe_media(input_path)
    load_started = time.perf_counter()
    process_cpu_started = time.process_time()
    if model_bundle is None:
        bundle = load_colorizer_bundle(config)
        model_load_seconds = time.perf_counter() - load_started
    else:
        bundle = model_bundle
        model_load_seconds = 0.0
    frame_transport = str(config.raw.get("runtime", {}).get("frame_transport", "png")).lower()

    run_hash = sha256(f"{input_path}:{output_path}:{config_path.resolve()}".encode()).hexdigest()[:12]
    frame_root = paths.frames_dir / f"clip_{run_hash}"
    source_frames_dir = frame_root / "source"
    colorized_frames_dir = frame_root / "colorized"

    print(f"Input clip: {input_path}")
    print(f"Output clip: {output_path}")
    print(f"Backend: {bundle.backend}")
    print(f"Render factor: {config.model['render_factor']}")
    print(f"Frame transport: {frame_transport}")

    started = time.perf_counter()
    cleanup_frames = bool(config.raw.get("runtime", {}).get("cleanup_frames", True))
    frame_extract_seconds = 0.0
    frame_decode_seconds = 0.0
    inference_preprocess_seconds = 0.0
    inference_preprocess_upload_seconds = 0.0
    inference_model_seconds = 0.0
    inference_postprocess_seconds = 0.0
    inference_postprocess_upload_seconds = 0.0
    inference_postprocess_graph_seconds = 0.0
    inference_postprocess_download_seconds = 0.0
    inference_postprocess_cpu_seconds = 0.0
    costume_palette_seconds = 0.0
    temporal_smoothing_seconds = 0.0
    frame_save_seconds = 0.0
    encode_seconds = 0.0
    cleanup_seconds = 0.0
    frame_count = 0
    try:
        previous_smoothed_frame: np.ndarray | None = None
        postprocess_config = config.raw["postprocess"]
        costume_palette_runtime = build_costume_palette_runtime(postprocess_config=postprocess_config, device=bundle.device)
        if frame_transport == "pipe":
            (
                frame_count,
                frame_decode_seconds,
                inference_preprocess_seconds,
                inference_preprocess_upload_seconds,
                inference_model_seconds,
                inference_postprocess_seconds,
                inference_postprocess_upload_seconds,
                inference_postprocess_graph_seconds,
                inference_postprocess_download_seconds,
                inference_postprocess_cpu_seconds,
                costume_palette_seconds,
                temporal_smoothing_seconds,
                frame_save_seconds,
                encode_seconds,
            ) = _run_pipe_transport(
                input_path=input_path,
                output_path=output_path,
                media_info=media_info,
                config=config,
                bundle=bundle,
                postprocess_config=postprocess_config,
                collect_profile=collect_profile,
                costume_palette_runtime=costume_palette_runtime,
            )
        else:
            extract_started = time.perf_counter()
            extract_frames(input_path=input_path, output_dir=source_frames_dir)
            frame_extract_seconds = time.perf_counter() - extract_started

            frame_paths = sorted(source_frames_dir.glob("*.png"))
            if not frame_paths:
                raise RuntimeError("No frames were extracted from the input clip.")
            frame_count = len(frame_paths)

            colorized_frames_dir.mkdir(parents=True, exist_ok=True)
            for frame_path in frame_paths:
                frame_decode_started = time.perf_counter()
                input_image = Image.open(frame_path).convert("RGB")
                frame_decode_seconds += time.perf_counter() - frame_decode_started
                if collect_profile:
                    result, inference_profile = colorize_pil_image_profiled(
                        model_bundle=bundle,
                        input_image=input_image,
                        render_factor=int(config.model["render_factor"]),
                        postprocess_config=postprocess_config,
                    )
                    inference_preprocess_seconds += inference_profile.preprocess_seconds
                    inference_preprocess_upload_seconds += inference_profile.preprocess_upload_seconds
                    inference_model_seconds += inference_profile.model_seconds
                    inference_postprocess_seconds += inference_profile.postprocess_seconds
                    inference_postprocess_upload_seconds += inference_profile.postprocess_upload_seconds
                    inference_postprocess_graph_seconds += inference_profile.postprocess_graph_seconds
                    inference_postprocess_download_seconds += inference_profile.postprocess_download_seconds
                    inference_postprocess_cpu_seconds += inference_profile.postprocess_cpu_seconds
                else:
                    result = colorize_pil_image(
                        model_bundle=bundle,
                        input_image=input_image,
                        render_factor=int(config.model["render_factor"]),
                        postprocess_config=postprocess_config,
                    )
                result_np = np.asarray(result)
                if costume_palette_runtime is not None:
                    palette_started = time.perf_counter()
                    result_np = costume_palette_runtime.apply(
                        image_rgb=result_np,
                        source_rgb=np.asarray(input_image),
                    )
                    costume_palette_seconds += time.perf_counter() - palette_started
                if bool(postprocess_config.get("temporal_smoothing", False)):
                    temporal_started = time.perf_counter()
                    result_np, previous_smoothed_frame = apply_temporal_smoothing(
                        current_frame=result_np,
                        previous_frame=previous_smoothed_frame,
                        strength=float(postprocess_config.get("smoothing_strength", 0.0)),
                        chroma_threshold=float(postprocess_config.get("smoothing_chroma_threshold", 24.0)),
                        adaptive_boost=float(postprocess_config.get("adaptive_smoothing_boost", 0.0)),
                    )
                    temporal_smoothing_seconds += time.perf_counter() - temporal_started
                else:
                    previous_smoothed_frame = result_np

                frame_save_started = time.perf_counter()
                Image.fromarray(result_np).save(colorized_frames_dir / frame_path.name)
                frame_save_seconds += time.perf_counter() - frame_save_started

            encode_started = time.perf_counter()
            encode_video_from_frames(
                frame_dir=colorized_frames_dir,
                output_path=output_path,
                fps=str(media_info["fps"]),
                video_codec=str(config.raw["video"]["output_codec"]),
                crf=int(config.raw["video"]["crf"]),
                pixel_format=str(config.raw["video"]["pixel_format"]),
                audio_input_path=input_path,
            )
            encode_seconds = time.perf_counter() - encode_started
        runtime_seconds = time.perf_counter() - started
    finally:
        cleanup_started = time.perf_counter()
        if cleanup_frames and frame_root.exists():
            shutil.rmtree(frame_root, ignore_errors=True)
        cleanup_seconds = time.perf_counter() - cleanup_started
    process_cpu_seconds = time.process_time() - process_cpu_started

    record = ClipRunRecord(
        input_path=str(input_path),
        output_path=str(output_path),
        config_path=str(config_path.resolve()),
        render_factor=int(config.model["render_factor"]),
        backend=bundle.backend,
        runtime_seconds=runtime_seconds,
        frame_count=frame_count,
        fps=str(media_info["fps"]),
        width=int(media_info["width"]),
        height=int(media_info["height"]),
        frame_transport=frame_transport,
        status="succeeded",
    )

    manifest_path = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else paths.manifest_dir / "probe_runs.json"
    )
    update_probe_runs_manifest(manifest_path, record)
    stage_profile = (
        ClipStageProfile(
            model_load_seconds=model_load_seconds,
            frame_extract_seconds=frame_extract_seconds,
            frame_decode_seconds=frame_decode_seconds,
            inference_preprocess_seconds=inference_preprocess_seconds,
            inference_preprocess_upload_seconds=inference_preprocess_upload_seconds,
            inference_model_seconds=inference_model_seconds,
            inference_postprocess_seconds=inference_postprocess_seconds,
            inference_postprocess_upload_seconds=inference_postprocess_upload_seconds,
            inference_postprocess_graph_seconds=inference_postprocess_graph_seconds,
            inference_postprocess_download_seconds=inference_postprocess_download_seconds,
            inference_postprocess_cpu_seconds=inference_postprocess_cpu_seconds,
            costume_palette_seconds=costume_palette_seconds,
            temporal_smoothing_seconds=temporal_smoothing_seconds,
            frame_save_seconds=frame_save_seconds,
            encode_seconds=encode_seconds,
            cleanup_seconds=cleanup_seconds,
            total_runtime_seconds=runtime_seconds,
            process_cpu_seconds=process_cpu_seconds,
            effective_fps=(frame_count / runtime_seconds) if runtime_seconds > 0 else 0.0,
            per_frame_seconds=(runtime_seconds / frame_count) if frame_count else 0.0,
        )
        if collect_profile
        else None
    )
    return ClipExecutionResult(
        run_record=record,
        stage_profile=stage_profile,
        manifest_path=manifest_path,
    )


def _run_pipe_transport(
    *,
    input_path: Path,
    output_path: Path,
    media_info: dict[str, str | int | float],
    config: AppConfig,
    bundle,
    postprocess_config: dict,
    collect_profile: bool,
    costume_palette_runtime: CostumePaletteRuntime | None,
) -> tuple[int, float, float, float, float, float, float, float, float, float, float, float, float, float]:
    width = int(media_info["width"])
    height = int(media_info["height"])
    frame_bytes = width * height * 3
    previous_smoothed_frame: np.ndarray | None = None
    frame_count = 0
    frame_decode_seconds = 0.0
    inference_preprocess_seconds = 0.0
    inference_preprocess_upload_seconds = 0.0
    inference_model_seconds = 0.0
    inference_postprocess_seconds = 0.0
    inference_postprocess_upload_seconds = 0.0
    inference_postprocess_graph_seconds = 0.0
    inference_postprocess_download_seconds = 0.0
    inference_postprocess_cpu_seconds = 0.0
    costume_palette_seconds = 0.0
    temporal_smoothing_seconds = 0.0
    frame_save_seconds = 0.0

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
    inference_batch_size = max(1, int(config.raw.get("runtime", {}).get("inference_batch_size", 1)))

    try:
        while True:
            batch_frames: list[np.ndarray] = []
            batch_read_started = time.perf_counter()
            for _ in range(inference_batch_size):
                frame_data = _read_exact(reader.stdout, frame_bytes)
                if frame_data is None:
                    break
                batch_frames.append(
                    np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
                )
            frame_decode_seconds += time.perf_counter() - batch_read_started
            if not batch_frames:
                break

            if collect_profile:
                result_batch, inference_profile = colorize_rgb_batch_profiled(
                    model_bundle=bundle,
                    input_rgbs=batch_frames,
                    render_factor=int(config.model["render_factor"]),
                    postprocess_config=postprocess_config,
                )
                inference_preprocess_seconds += inference_profile.preprocess_seconds
                inference_preprocess_upload_seconds += inference_profile.preprocess_upload_seconds
                inference_model_seconds += inference_profile.model_seconds
                inference_postprocess_seconds += inference_profile.postprocess_seconds
                inference_postprocess_upload_seconds += inference_profile.postprocess_upload_seconds
                inference_postprocess_graph_seconds += inference_profile.postprocess_graph_seconds
                inference_postprocess_download_seconds += inference_profile.postprocess_download_seconds
                inference_postprocess_cpu_seconds += inference_profile.postprocess_cpu_seconds
            else:
                result_batch = colorize_rgb_batch(
                    model_bundle=bundle,
                    input_rgbs=batch_frames,
                    render_factor=int(config.model["render_factor"]),
                    postprocess_config=postprocess_config,
                )
            for source_rgb, result_np in zip(batch_frames, result_batch, strict=True):
                if costume_palette_runtime is not None:
                    palette_started = time.perf_counter()
                    result_np = costume_palette_runtime.apply(
                        image_rgb=result_np,
                        source_rgb=source_rgb,
                    )
                    costume_palette_seconds += time.perf_counter() - palette_started
                if bool(postprocess_config.get("temporal_smoothing", False)):
                    temporal_started = time.perf_counter()
                    result_np, previous_smoothed_frame = apply_temporal_smoothing(
                        current_frame=result_np,
                        previous_frame=previous_smoothed_frame,
                        strength=float(postprocess_config.get("smoothing_strength", 0.0)),
                        chroma_threshold=float(postprocess_config.get("smoothing_chroma_threshold", 24.0)),
                        adaptive_boost=float(postprocess_config.get("adaptive_smoothing_boost", 0.0)),
                    )
                    temporal_smoothing_seconds += time.perf_counter() - temporal_started
                else:
                    previous_smoothed_frame = result_np

                frame_write_started = time.perf_counter()
                writer.stdin.write(result_np.tobytes())
                frame_save_seconds += time.perf_counter() - frame_write_started
                frame_count += 1

        writer.stdin.close()
        encode_started = time.perf_counter()
        writer_returncode = writer.wait()
        encode_seconds = time.perf_counter() - encode_started
        reader_returncode = reader.wait()
    finally:
        if reader.stdout:
            reader.stdout.close()
        if writer.stdin:
            writer.stdin.close()

    if reader_returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo reader failed: {reader.stderr.read().decode().strip()}")
    if writer_returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo writer failed: {writer.stderr.read().decode().strip()}")

    return (
        frame_count,
        frame_decode_seconds,
        inference_preprocess_seconds,
        inference_preprocess_upload_seconds,
        inference_model_seconds,
        inference_postprocess_seconds,
        inference_postprocess_upload_seconds,
        inference_postprocess_graph_seconds,
        inference_postprocess_download_seconds,
        inference_postprocess_cpu_seconds,
        costume_palette_seconds,
        temporal_smoothing_seconds,
        frame_save_seconds,
        encode_seconds,
    )


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
