from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1
from pathlib import Path
import math
import time
from typing import Any

from src.pipeline.ffmpeg_utils import (
    compress_video,
    concat_videos,
    extract_clip,
    get_media_duration_seconds,
)
from src.pipeline.manifest import utc_now_iso, write_json_manifest
from src.pipeline.ddcolor_clip import run_ddcolor_clip
from src.pipeline.model_chroma_propagate import run_model_chroma_propagate
from src.pipeline.segment_clip import run_segment_clip


DEFAULT_SEMANTIC_CONSENSUS_LABELS = [
    "upper_clothes",
    "dress",
    "skirt",
    "pants",
    "coat",
]
DEFAULT_SEMANTIC_PROTECT_LABELS = ["face", "hair", "left_arm", "right_arm", "left_leg", "right_leg"]
DEFAULT_SEMANTIC_SPLIT_LABELS = ["face", "hair"]


@dataclass(frozen=True)
class ChunkRun:
    chunk_id: str
    start_seconds: float
    end_seconds: float
    source_clip: str
    ddcolor_clip: str
    segment_manifest: str
    propagated_clip: str
    status: str
    runtime_seconds: float
    prep_runtime_seconds: float
    propagation_runtime_seconds: float


@dataclass(frozen=True)
class ChunkPrep:
    chunk_id: str
    start_seconds: float
    end_seconds: float
    source_clip_path: Path
    ddcolor_clip_path: Path
    segment_manifest_path: Path
    prep_runtime_seconds: float


def run_fast_semantic_movie(
    *,
    input_path: Path,
    output_path: Path | None,
    weights_path: Path,
    work_dir: Path | None,
    chunk_seconds: float,
    limit_chunks: int | None,
    ddcolor_input_size: int,
    ddcolor_device: str,
    human_parser_model_id: str | None,
    human_parser_device: str,
    human_parser_frame_stride: int,
    propagation_mode: str,
    keyframe_stride: int,
    output_preset: str,
    chroma_blend: float,
    semantic_consensus_strength: float,
    semantic_consensus_labels: list[str],
    semantic_protect_labels: list[str],
    semantic_split_labels: list[str],
    max_workers: int,
    propagation_workers: int,
    compress: bool,
    compression_preset: str,
    compression_crfs: list[int],
    max_size_multiplier: float,
    overwrite: bool,
) -> int:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input movie not found: {input_path}")

    output_path = _resolve_output_path(input_path=input_path, output_path=output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    run_id = _build_run_id(input_path)
    root_dir = (
        work_dir.expanduser().resolve()
        if work_dir is not None
        else Path.home() / "ColorIt" / "data" / "fast_semantic_movie" / run_id
    )
    chunks_dir = root_dir / "chunks"
    ddcolor_dir = root_dir / "ddcolor"
    segments_dir = root_dir / "segments"
    propagated_dir = root_dir / "propagated"
    manifests_dir = root_dir / "manifests"
    for directory in (chunks_dir, ddcolor_dir, segments_dir, propagated_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=True)

    duration = get_media_duration_seconds(input_path)
    chunks = _build_chunks(duration_seconds=duration, chunk_seconds=chunk_seconds)
    if limit_chunks is not None:
        chunks = chunks[:limit_chunks]
    if not chunks:
        raise ValueError("No chunks to process.")

    assembly_output_path = root_dir / f"{input_path.stem}_fast_semantic_assembly.mp4" if compress else output_path
    manifest_path = manifests_dir / "fast_semantic_movie_manifest.json"
    payload: dict[str, Any] = {
        "run_id": run_id,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "work_dir": str(root_dir),
        "chunk_seconds": chunk_seconds,
        "chunk_count": len(chunks),
        "prep_workers": max_workers,
        "propagation_workers": propagation_workers,
        "status": "running",
        "started_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "chunks": [],
    }
    write_json_manifest(manifest_path, payload)

    print(f"Input movie: {input_path}")
    print(f"Output movie: {output_path}")
    print(f"Work dir: {root_dir}")
    print(f"Chunks: {len(chunks)}")

    chunk_sources: dict[int, Path] = {}
    for index, (start_seconds, end_seconds) in enumerate(chunks):
        chunk_id = f"chunk_{index:04d}"
        chunk_sources[index] = _extract_chunk_source(
            input_path=input_path,
            chunk_id=chunk_id,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            chunks_dir=chunks_dir,
            overwrite=overwrite,
        )

    completed_by_index: dict[int, Path] = {}
    chunk_records: dict[int, ChunkRun] = {}
    prep_workers = max(1, int(max_workers))
    prop_workers = max(1, int(propagation_workers))
    try:
        with ThreadPoolExecutor(max_workers=prep_workers) as prep_executor, ThreadPoolExecutor(max_workers=prop_workers) as prop_executor:
            prep_futures = {
                prep_executor.submit(
                    _prepare_chunk,
                    chunk_id=f"chunk_{index:04d}",
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    source_clip_path=chunk_sources[index],
                    ddcolor_dir=ddcolor_dir,
                    segments_dir=segments_dir,
                    weights_path=weights_path,
                    ddcolor_input_size=ddcolor_input_size,
                    ddcolor_device=ddcolor_device,
                    human_parser_model_id=human_parser_model_id,
                    human_parser_device=human_parser_device,
                    human_parser_frame_stride=human_parser_frame_stride,
                    output_preset=output_preset,
                    semantic_consensus_labels=semantic_consensus_labels or DEFAULT_SEMANTIC_CONSENSUS_LABELS,
                    semantic_protect_labels=semantic_protect_labels or DEFAULT_SEMANTIC_PROTECT_LABELS,
                    semantic_split_labels=semantic_split_labels or DEFAULT_SEMANTIC_SPLIT_LABELS,
                    overwrite=overwrite,
                ): index
                for index, (start_seconds, end_seconds) in enumerate(chunks)
            }
            prop_futures = {}
            for future in as_completed(prep_futures):
                index = prep_futures[future]
                prep = future.result()
                prop_future = prop_executor.submit(
                    _propagate_prepared_chunk,
                    prep=prep,
                    propagated_dir=propagated_dir,
                    propagation_mode=propagation_mode,
                    keyframe_stride=keyframe_stride,
                    output_preset=output_preset,
                    chroma_blend=chroma_blend,
                    semantic_consensus_strength=semantic_consensus_strength,
                    semantic_consensus_labels=semantic_consensus_labels or DEFAULT_SEMANTIC_CONSENSUS_LABELS,
                    semantic_protect_labels=semantic_protect_labels or DEFAULT_SEMANTIC_PROTECT_LABELS,
                    semantic_split_labels=semantic_split_labels or DEFAULT_SEMANTIC_SPLIT_LABELS,
                    overwrite=overwrite,
                )
                prop_futures[prop_future] = index

            for future in as_completed(prop_futures):
                index = prop_futures[future]
                chunk_run = future.result()
                completed_by_index[index] = Path(chunk_run.propagated_clip)
                chunk_records[index] = chunk_run
                payload["chunks"] = [asdict(chunk_records[item]) for item in sorted(chunk_records)]
                payload["updated_at"] = utc_now_iso()
                write_json_manifest(manifest_path, payload)

        concat_list_path = manifests_dir / "concat_chunks.txt"
        completed_chunks = [completed_by_index[index] for index in range(len(chunks))]
        concat_list_path.write_text(
            "".join(f"file '{clip.as_posix()}'\n" for clip in completed_chunks),
            encoding="utf-8",
        )
        concat_videos(input_list_path=concat_list_path, output_path=assembly_output_path)

        compression_result: dict[str, Any] | None = None
        if compress:
            compression_result = _compress_to_size_target(
                input_path=assembly_output_path,
                output_path=output_path,
                source_path=input_path,
                preset=compression_preset,
                crfs=compression_crfs,
                max_size_multiplier=max_size_multiplier,
            )

        payload["status"] = "succeeded"
        payload["assembly_output_path"] = str(assembly_output_path)
        payload["concat_list_path"] = str(concat_list_path)
        payload["compression"] = compression_result
        payload["completed_at"] = utc_now_iso()
        payload["updated_at"] = utc_now_iso()
        write_json_manifest(manifest_path, payload)
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = str(exc)
        payload["updated_at"] = utc_now_iso()
        write_json_manifest(manifest_path, payload)
        raise

    print(f"Fast semantic movie written: {output_path}")
    print(f"Manifest written: {manifest_path}")
    return 0


def _extract_chunk_source(
    *,
    input_path: Path,
    chunk_id: str,
    start_seconds: float,
    end_seconds: float,
    chunks_dir: Path,
    overwrite: bool,
) -> Path:
    source_clip_path = chunks_dir / f"{chunk_id}.mp4"
    print(f"[{chunk_id}] Extracting {start_seconds:.2f}s-{end_seconds:.2f}s")
    if not source_clip_path.exists() or overwrite:
        extract_clip(
            input_path=input_path,
            output_path=source_clip_path,
            start_time=_format_seconds(start_seconds),
            end_time=_format_seconds(end_seconds),
            video_codec="libx264",
            crf=16,
            pixel_format="yuv420p",
        )
    return source_clip_path


def _prepare_chunk(
    *,
    chunk_id: str,
    start_seconds: float,
    end_seconds: float,
    source_clip_path: Path,
    ddcolor_dir: Path,
    segments_dir: Path,
    weights_path: Path,
    ddcolor_input_size: int,
    ddcolor_device: str,
    human_parser_model_id: str | None,
    human_parser_device: str,
    human_parser_frame_stride: int,
    output_preset: str,
    semantic_consensus_labels: list[str],
    semantic_protect_labels: list[str],
    semantic_split_labels: list[str],
    overwrite: bool,
) -> ChunkPrep:
    started = time.perf_counter()
    ddcolor_clip_path = ddcolor_dir / f"{chunk_id}.mp4"
    segment_dir = segments_dir / chunk_id

    print(f"[{chunk_id}] Running DDColor")
    run_ddcolor_clip(
        input_path=source_clip_path,
        output_path=ddcolor_clip_path,
        weights_path=weights_path,
        input_size=ddcolor_input_size,
        device=ddcolor_device,
        output_preset=output_preset,
        overwrite=overwrite,
    )

    print(f"[{chunk_id}] Running human-parser segmentation")
    run_segment_clip(
        input_path=source_clip_path,
        backend="human-parser",
        output_dir=segment_dir,
        tracks_path=None,
        model_id=human_parser_model_id,
        device=human_parser_device,
        frame_stride=human_parser_frame_stride,
        include_labels=sorted(set(semantic_consensus_labels + semantic_protect_labels + semantic_split_labels)),
        score_threshold=0.70,
        mask_threshold=0.50,
        min_area=3000,
        iou_threshold=0.10,
        max_center_distance=260.0,
        max_missing_frames=8,
        overwrite=overwrite,
    )

    return ChunkPrep(
        chunk_id=chunk_id,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        source_clip_path=source_clip_path,
        ddcolor_clip_path=ddcolor_clip_path,
        segment_manifest_path=segment_dir / "segment_manifest.json",
        prep_runtime_seconds=time.perf_counter() - started,
    )


def _propagate_prepared_chunk(
    *,
    prep: ChunkPrep,
    propagated_dir: Path,
    propagation_mode: str,
    keyframe_stride: int,
    output_preset: str,
    chroma_blend: float,
    semantic_consensus_strength: float,
    semantic_consensus_labels: list[str],
    semantic_protect_labels: list[str],
    semantic_split_labels: list[str],
    overwrite: bool,
) -> ChunkRun:
    started = time.perf_counter()
    propagated_clip_path = propagated_dir / f"{prep.chunk_id}.mp4"
    print(f"[{prep.chunk_id}] Propagating model chroma with semantic consensus")
    run_model_chroma_propagate(
        source_path=prep.source_clip_path,
        model_color_path=prep.ddcolor_clip_path,
        output_path=propagated_clip_path,
        keyframe_stride=keyframe_stride,
        propagation_mode=propagation_mode,
        output_preset=output_preset,
        chroma_blend=chroma_blend,
        fallback_color_hex=None,
        fallback_strength=0.85,
        fallback_uncertainty="hue",
        disagreement_start=20.0,
        disagreement_end=70.0,
        scene_cut_threshold=0.0,
        scene_keyframe_window=2,
        chroma_smooth_diameter=0,
        chroma_smooth_sigma_color=16.0,
        chroma_smooth_sigma_space=7.0,
        dark_fill_strength=0.0,
        dark_fill_luma_end=92.0,
        dark_fill_chroma_end=22.0,
        dark_fill_sigma=8.0,
        model_fill_strength=0.25,
        model_fill_chroma_end=28.0,
        model_fill_disagreement_start=25.0,
        model_fill_disagreement_end=80.0,
        model_fill_blur_sigma=1.5,
        model_fill_chroma_floor=0.0,
        component_fill_strength=0.0,
        component_fill_luma_end=100.0,
        component_fill_min_area=1800,
        component_fill_model_chroma_min=14.0,
        blue_suppress_strength=0.0,
        blue_suppress_hue_start=85.0,
        blue_suppress_hue_end=132.0,
        semantic_consensus_manifest_path=prep.segment_manifest_path,
        semantic_consensus_labels=semantic_consensus_labels,
        semantic_protect_labels=semantic_protect_labels,
        semantic_protect_dilate=3,
        semantic_split_labels=semantic_split_labels,
        semantic_consensus_strength=semantic_consensus_strength,
        semantic_consensus_min_area=600,
        semantic_consensus_model_chroma_min=8.0,
        semantic_consensus_feather_sigma=1.2,
        semantic_consensus_diversify_strength=0.65,
        semantic_consensus_diversify_threshold=20.0,
        overwrite=overwrite,
    )
    propagation_runtime = time.perf_counter() - started
    return ChunkRun(
        chunk_id=prep.chunk_id,
        start_seconds=prep.start_seconds,
        end_seconds=prep.end_seconds,
        source_clip=str(prep.source_clip_path),
        ddcolor_clip=str(prep.ddcolor_clip_path),
        segment_manifest=str(prep.segment_manifest_path),
        propagated_clip=str(propagated_clip_path),
        status="succeeded",
        runtime_seconds=prep.prep_runtime_seconds + propagation_runtime,
        prep_runtime_seconds=prep.prep_runtime_seconds,
        propagation_runtime_seconds=propagation_runtime,
    )


def _build_chunks(*, duration_seconds: float, chunk_seconds: float) -> list[tuple[float, float]]:
    chunk_seconds = max(1.0, float(chunk_seconds))
    count = int(math.ceil(duration_seconds / chunk_seconds))
    return [
        (index * chunk_seconds, min(duration_seconds, (index + 1) * chunk_seconds))
        for index in range(count)
        if index * chunk_seconds < duration_seconds
    ]


def _compress_to_size_target(
    *,
    input_path: Path,
    output_path: Path,
    source_path: Path,
    preset: str,
    crfs: list[int],
    max_size_multiplier: float,
) -> dict[str, Any]:
    source_size_bytes = source_path.stat().st_size
    max_size_bytes = int(source_size_bytes * max_size_multiplier)
    unique_crfs = list(dict.fromkeys(crfs or [20, 23, 26, 28]))
    selected_crf = unique_crfs[-1]
    final_size_bytes = 0

    for crf in unique_crfs:
        selected_crf = int(crf)
        compress_video(
            input_path=input_path,
            output_path=output_path,
            video_codec="libx264",
            preset=preset,
            crf=selected_crf,
            audio_codec="aac",
            audio_bitrate="160k",
            faststart=True,
        )
        final_size_bytes = output_path.stat().st_size
        if final_size_bytes <= max_size_bytes:
            break

    return {
        "output_path": str(output_path),
        "selected_crf": selected_crf,
        "source_size_bytes": source_size_bytes,
        "final_size_bytes": final_size_bytes,
        "max_size_bytes": max_size_bytes,
        "within_size_target": final_size_bytes <= max_size_bytes,
    }


def _resolve_output_path(*, input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path.expanduser().resolve()
    return input_path.with_stem(f"{input_path.stem}_fast_semantic_color")


def _build_run_id(input_path: Path) -> str:
    safe_stem = "".join(character if character.isalnum() else "_" for character in input_path.stem).strip("_")
    location_hash = sha1(str(input_path).encode("utf-8")).hexdigest()[:8]
    return f"{safe_stem}_{location_hash}"


def _format_seconds(value: float) -> str:
    return f"{value:.3f}"
