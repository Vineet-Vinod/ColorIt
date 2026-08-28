from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import cv2

from src.pipeline.colorize_clip import run_deoldify_clip
from src.pipeline.config import AppConfig
from src.pipeline.ddcolor_clip import DEFAULT_DDCOLOR_WEIGHTS_PATH, run_ddcolor_clip
from src.pipeline.deepremaster_clip import DeepRemasterRunner, clip_record_to_dict
from src.pipeline.ffmpeg_utils import (
    FrameSplitSpec,
    encode_scene_mezzanine,
    ffprobe_media,
    fps_to_decimal_string,
    split_video_by_frame_counts,
)
from src.pipeline.manifest import load_json_manifest, utc_now_iso, write_json_manifest
from src.pipeline.keyframes import (
    KeyframeColorizer,
    keyframe_colored_paths,
    keyframe_record_to_dict,
    normalize_keyframe_positions,
)
from src.pipeline.model_chroma_propagate import run_model_chroma_propagate
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths
from src.pipeline.preprocess import equalize_clip_luma_clahe
from src.pipeline.progress import ProgressBar
from src.pipeline.scenes import load_scene_manifest
from src.pipeline.weights import DEFAULT_DEEPREMASTER_WEIGHTS_PATH


SCENE_EXTRACTION_VERSION = "scene-frame-split-v1"


@dataclass(frozen=True)
class BatchSceneStatus:
    scene_id: str
    input_clip: str
    output_clip: str
    status: str
    runtime_seconds: float | None = None
    stage_runtime_seconds: dict[str, float] | None = None
    error: str | None = None
    keyframe: dict[str, Any] | None = None
    keyframes: list[dict[str, Any]] | None = None
    deepremaster: dict[str, Any] | None = None


def run_colorize_batch(
    *,
    config: AppConfig,
    config_path: Path,
    movie_path: Path,
    scene_manifest_path: Path,
    resume: bool,
    limit: int | None,
    pipeline: str = "default",
    coloring_model: str = "deoldify",
) -> int:
    paths = resolve_project_paths(config)
    ensure_runtime_directories(paths)

    movie_path = movie_path.expanduser().resolve()
    scene_manifest_path = scene_manifest_path.expanduser().resolve()
    if not movie_path.exists():
        raise FileNotFoundError(f"Movie file not found: {movie_path}")

    manifest = load_scene_manifest(scene_manifest_path)
    scenes = manifest["scenes"]
    if limit is not None:
        scenes = scenes[:limit]

    run_id = scene_manifest_path.stem
    scene_output_dir = paths.scene_dir / run_id
    colorized_output_dir = paths.colorized_dir / "scenes" / run_id
    deoldify_output_dir = paths.colorized_dir / "deoldify" / run_id
    ddcolor_output_dir = paths.colorized_dir / "ddcolor" / run_id
    equalized_output_dir = paths.colorized_dir / "clahe" / run_id
    keyframe_source_dir = paths.colorized_dir / "keyframes" / coloring_model / run_id / "source"
    keyframe_colored_dir = paths.colorized_dir / "keyframes" / coloring_model / run_id / "colored"
    keyframe_reference_dir = paths.colorized_dir / "keyframes" / coloring_model / run_id / "references"
    for directory in (
        scene_output_dir,
        colorized_output_dir,
        deoldify_output_dir,
        ddcolor_output_dir,
        equalized_output_dir,
        keyframe_source_dir,
        keyframe_colored_dir,
        keyframe_reference_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    cleanup_scene_clips = bool(config.raw.get("runtime", {}).get("cleanup_scene_clips", True))
    deepremaster_settings = config.raw.get("deep_remaster", {})
    keyframe_positions = (
        normalize_keyframe_positions(
            deepremaster_settings.get("keyframe_positions", [0.2, 0.5, 0.8])
        )
        if pipeline == "deepremaster"
        else []
    )

    batch_manifest_path = paths.manifest_dir / f"full_run_{run_id}.json"
    scene_runs_manifest_path = paths.manifest_dir / f"scene_runs_{run_id}.json"
    batch_payload = _load_batch_manifest(
        batch_manifest_path=batch_manifest_path,
        movie_path=movie_path,
        config_path=config_path,
        scene_manifest_path=scene_manifest_path,
        scene_count=len(scenes),
        resume=resume,
        limit=limit,
        pipeline=pipeline,
        coloring_model=coloring_model,
        keyframe_positions=keyframe_positions,
    )
    batch_payload["status"] = "running"
    batch_payload["updated_at"] = utc_now_iso()
    write_json_manifest(batch_manifest_path, batch_payload)

    print(f"Movie: {movie_path}")
    print(f"Scene manifest: {scene_manifest_path}")
    print(f"Batch scene count: {len(scenes)}")
    print(f"Resume mode: {resume}")
    shared_bundle = None
    keyframe_colorizer = (
        KeyframeColorizer(config=config, model=coloring_model, root=paths.root)
        if pipeline == "deepremaster"
        else None
    )
    deepremaster_runner = None
    scene_mezzanine_path = scene_output_dir / "source_mezzanine.mp4"
    scene_extraction_manifest_path = scene_output_dir / "scene_extraction_manifest.json"
    scene_extraction_current = _scene_extraction_manifest_matches(
        manifest_path=scene_extraction_manifest_path,
        movie_path=movie_path,
        scenes=scenes,
    )
    if not scene_extraction_current:
        _invalidate_scene_artifacts(
            scenes=scenes,
            scene_output_dir=scene_output_dir,
            colorized_output_dir=colorized_output_dir,
            deoldify_output_dir=deoldify_output_dir,
            ddcolor_output_dir=ddcolor_output_dir,
            equalized_output_dir=equalized_output_dir,
            keyframe_source_dir=keyframe_source_dir,
            keyframe_colored_dir=keyframe_colored_dir,
            keyframe_reference_dir=keyframe_reference_dir,
            scene_extraction_manifest_path=scene_extraction_manifest_path,
        )
        batch_payload["scene_runs"] = []
        _refresh_batch_summary(batch_payload, expected_scene_count=len(scenes))
        write_json_manifest(batch_manifest_path, batch_payload)
    scene_clips_prepared = False
    scene_preparation_runtimes: dict[str, float] = {}
    resumable_scene_ids = {
        str(scene["scene_id"])
        for scene in scenes
        if _can_resume_completed_scene(
            scene=scene,
            batch_payload=batch_payload,
            scene_output_dir=scene_output_dir,
            colorized_output_dir=colorized_output_dir,
            resume=resume,
            pipeline=pipeline,
            expected_reference_paths=(
                keyframe_colored_paths(
                    scene_id=str(scene["scene_id"]),
                    colored_dir=keyframe_colored_dir,
                    positions=keyframe_positions,
                )
                if pipeline == "deepremaster"
                else []
            ),
        )
    }
    resumed_duration = sum(
        float(scene["duration_seconds"])
        for scene in scenes
        if str(scene["scene_id"]) in resumable_scene_ids
    )
    batch_progress = ProgressBar(
        "Batch colorization",
        total=sum(float(scene["duration_seconds"]) for scene in scenes),
        unit="video-s",
        initial=resumed_duration,
    )

    for scene in scenes:
        scene_id = scene["scene_id"]
        scene_clip_path = scene_output_dir / f"{scene_id}.mp4"
        equalized_clip_path = equalized_output_dir / f"{scene_id}.mp4"
        deoldify_clip_path = deoldify_output_dir / f"{scene_id}.mp4"
        ddcolor_clip_path = ddcolor_output_dir / f"{scene_id}.mp4"
        colorized_clip_path = colorized_output_dir / f"{scene_id}.mp4"
        existing_status = _get_scene_status(batch_payload, scene_id)

        if scene_id in resumable_scene_ids:
            print(f"Skipping completed scene: {scene_id}")
            if existing_status is None or existing_status.get("status") != "succeeded":
                _upsert_scene_status(
                    batch_payload,
                    BatchSceneStatus(
                        scene_id=scene_id,
                        input_clip=str(scene_clip_path),
                        output_clip=str(colorized_clip_path),
                        status="succeeded",
                        runtime_seconds=0.0,
                        stage_runtime_seconds={
                            "scene_extraction": 0.0,
                            "deoldify": 0.0,
                            "ddcolor": 0.0,
                            "chroma_propagation": 0.0,
                        },
                    ),
                )
            _refresh_batch_summary(batch_payload, expected_scene_count=len(scenes))
            write_json_manifest(batch_manifest_path, batch_payload)
            continue

        started = time.perf_counter()
        stage_runtimes: dict[str, float] = {}
        try:
            if not _is_usable_video(scene_clip_path):
                stage_started = time.perf_counter()
                if not scene_clips_prepared:
                    scene_preparation_runtimes = _prepare_scene_clips(
                        movie_path=movie_path,
                        scene_output_dir=scene_output_dir,
                        mezzanine_path=scene_mezzanine_path,
                        scenes=scenes,
                        config=config,
                        extraction_manifest_path=scene_extraction_manifest_path,
                    )
                    scene_clips_prepared = True
                if not _is_usable_video(scene_clip_path):
                    raise FileNotFoundError(f"Missing frame-exact scene clip after extraction: {scene_clip_path}")
                stage_runtimes["scene_extraction"] = time.perf_counter() - stage_started
                stage_runtimes.update(scene_preparation_runtimes)
                scene_preparation_runtimes = {}
                print(f"Extracted scene clip: {scene_clip_path.name}")
            else:
                stage_runtimes["scene_extraction"] = 0.0

            if pipeline == "deepremaster":
                if keyframe_colorizer is None:
                    raise RuntimeError("DeepRemaster keyframe colorizer was not initialized")
                stage_started = time.perf_counter()
                keyframe_records = keyframe_colorizer.colorize_scene_references(
                    scene_id=scene_id,
                    clip_path=scene_clip_path,
                    source_dir=keyframe_source_dir,
                    colored_dir=keyframe_colored_dir,
                    reference_dir=keyframe_reference_dir,
                    positions=keyframe_positions,
                    reuse_existing=resume,
                )
                stage_runtimes["keyframe_colorization"] = time.perf_counter() - stage_started
                for keyframe_record in keyframe_records:
                    if keyframe_record.reused:
                        print(f"Reusing {coloring_model} keyframe: {Path(keyframe_record.colored_path).name}")
                keyframe_paths = [
                    Path(record.reference_path or record.colored_path)
                    for record in keyframe_records
                ]

                if deepremaster_runner is None:
                    precision = str(deepremaster_settings.get("precision", "float16"))
                    deepremaster_runner = DeepRemasterRunner(
                        checkpoint_path=paths.root / DEFAULT_DEEPREMASTER_WEIGHTS_PATH,
                        converted_weights_path=(
                            paths.root
                            / "models/deepremaster"
                            / f"remasternet.{precision}.mlx.safetensors"
                        ),
                        precision=precision,
                        min_dimension=int(deepremaster_settings.get("min_dimension", 192)),
                        reference_min_dimension=int(
                            deepremaster_settings.get("reference_min_dimension", 256)
                        ),
                        temporal_block_size=int(deepremaster_settings.get("temporal_block_size", 5)),
                        compile_model=bool(deepremaster_settings.get("compile", True)),
                        dense_max_scores=int(deepremaster_settings.get("dense_max_scores", 32_000_000)),
                        source_tile_size=int(deepremaster_settings.get("source_tile_size", 1024)),
                        reference_tile_size=int(deepremaster_settings.get("reference_tile_size", 2048)),
                        restoration_strength=float(deepremaster_settings.get("restoration_strength", 1.0)),
                        chroma_gain=float(deepremaster_settings.get("chroma_gain", 1.0)),
                        target_chroma_ratio=_optional_positive_float(
                            deepremaster_settings.get("target_chroma_ratio")
                        ),
                        max_chroma_gain=float(deepremaster_settings.get("max_chroma_gain", 2.5)),
                        lab_backend=str(deepremaster_settings.get("lab_backend", "opencv")),
                    )
                stage_started = time.perf_counter()
                deepremaster_record = deepremaster_runner.run_clip(
                    input_path=scene_clip_path,
                    output_path=colorized_clip_path,
                    reference_paths=keyframe_paths,
                    reference_times_seconds=[record.time_seconds for record in keyframe_records],
                    overwrite=True,
                    progress_label=f"DeepRemaster {coloring_model} {scene_id}",
                    output_crf=int(
                        deepremaster_settings.get("intermediate_crf", config.raw["video"]["crf"])
                    ),
                    output_preset=str(
                        deepremaster_settings.get("intermediate_preset", "ultrafast")
                    ),
                    output_pixel_format=str(
                        deepremaster_settings.get(
                            "intermediate_pixel_format", config.raw["video"]["pixel_format"]
                        )
                    ),
                )
                stage_runtimes["deepremaster"] = time.perf_counter() - stage_started
                status = BatchSceneStatus(
                    scene_id=scene_id,
                    input_clip=str(scene_clip_path),
                    output_clip=str(colorized_clip_path),
                    status="succeeded",
                    runtime_seconds=time.perf_counter() - started,
                    stage_runtime_seconds=stage_runtimes,
                    keyframe=(
                        keyframe_record_to_dict(keyframe_records[0])
                        if len(keyframe_records) == 1
                        else None
                    ),
                    keyframes=[keyframe_record_to_dict(record) for record in keyframe_records],
                    deepremaster=clip_record_to_dict(deepremaster_record),
                )
                if cleanup_scene_clips and scene_clip_path.exists():
                    scene_clip_path.unlink()
                _upsert_scene_status(batch_payload, status)
                _refresh_batch_summary(batch_payload, expected_scene_count=len(scenes))
                write_json_manifest(batch_manifest_path, batch_payload)
                batch_progress.update(float(scene["duration_seconds"]))
                continue

            if resume and _is_usable_video(equalized_clip_path, reference_path=scene_clip_path):
                print(f"Reusing CLAHE source clip: {equalized_clip_path.name}")
                stage_runtimes["clahe"] = 0.0
            else:
                stage_started = time.perf_counter()
                frame_count = equalize_clip_luma_clahe(
                    input_path=scene_clip_path,
                    output_path=equalized_clip_path,
                    clip_limit=2.0,
                    tile_grid_size=8,
                    strength=0.70,
                    crf=int(config.raw["video"]["crf"]),
                    include_audio=True,
                    progress_label=f"CLAHE {scene_id}",
                )
                print(f"CLAHE source clip written: {equalized_clip_path.name} ({frame_count} frames)")
                stage_runtimes["clahe"] = time.perf_counter() - stage_started

            if resume and _is_usable_video(deoldify_clip_path, reference_path=equalized_clip_path):
                print(f"Reusing DeOldify clip: {deoldify_clip_path.name}")
                stage_runtimes["deoldify"] = 0.0
            else:
                if shared_bundle is None:
                    print("Loading colorizer model once for batch reuse...")
                    shared_bundle = load_colorizer_bundle(config)
                stage_started = time.perf_counter()
                run_deoldify_clip(
                    config=config,
                    config_path=config_path,
                    input_path=equalized_clip_path,
                    output_path=deoldify_clip_path,
                    manifest_path=scene_runs_manifest_path,
                    overwrite=True,
                    model_bundle=shared_bundle,
                    include_audio=False,
                    progress_label=f"DeOldify {scene_id}",
                )
                stage_runtimes["deoldify"] = time.perf_counter() - stage_started

            if resume and _is_usable_video(ddcolor_clip_path, reference_path=equalized_clip_path):
                print(f"Reusing DDColor clip: {ddcolor_clip_path.name}")
                stage_runtimes["ddcolor"] = 0.0
            else:
                stage_started = time.perf_counter()
                run_ddcolor_clip(
                    input_path=equalized_clip_path,
                    output_path=ddcolor_clip_path,
                    weights_path=paths.root / DEFAULT_DDCOLOR_WEIGHTS_PATH,
                    input_size=256,
                    device="auto",
                    output_preset="ultrafast",
                    overwrite=True,
                    include_audio=False,
                    progress_label=f"DDColor {scene_id}",
                )
                stage_runtimes["ddcolor"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            run_model_chroma_propagate(
                source_path=deoldify_clip_path,
                model_color_path=ddcolor_clip_path,
                output_path=colorized_clip_path,
                keyframe_stride=35,
                chroma_blend=1.0,
                audio_input_path=equalized_clip_path,
                overwrite=True,
                progress_label=f"Chroma propagation {scene_id}",
            )
            stage_runtimes["chroma_propagation"] = time.perf_counter() - stage_started
            status = BatchSceneStatus(
                scene_id=scene_id,
                input_clip=str(scene_clip_path),
                output_clip=str(colorized_clip_path),
                status="succeeded",
                runtime_seconds=time.perf_counter() - started,
                stage_runtime_seconds=stage_runtimes,
            )
            if cleanup_scene_clips and scene_clip_path.exists():
                scene_clip_path.unlink()
        except Exception as exc:
            status = BatchSceneStatus(
                scene_id=scene_id,
                input_clip=str(scene_clip_path),
                output_clip=str(colorized_clip_path),
                status="failed",
                runtime_seconds=time.perf_counter() - started,
                stage_runtime_seconds=stage_runtimes,
                error=str(exc),
            )
            print(f"Scene failed: {scene_id}: {exc}")

        _upsert_scene_status(batch_payload, status)
        _refresh_batch_summary(batch_payload, expected_scene_count=len(scenes))
        write_json_manifest(batch_manifest_path, batch_payload)
        batch_progress.update(float(scene["duration_seconds"]))

    batch_progress.finish()
    _refresh_batch_summary(batch_payload, expected_scene_count=len(scenes))
    batch_payload["status"] = "failed" if int(batch_payload["failed_scene_count"]) > 0 else "succeeded"
    batch_payload["updated_at"] = utc_now_iso()
    write_json_manifest(batch_manifest_path, batch_payload)

    print(f"Succeeded: {batch_payload['succeeded_scene_count']}")
    print(f"Failed: {batch_payload['failed_scene_count']}")
    print(f"Batch manifest written to {batch_manifest_path}")
    if int(batch_payload["failed_scene_count"]) > 0:
        raise RuntimeError(f"{batch_payload['failed_scene_count']} scene(s) failed during batch colorization.")
    return 0


def _can_resume_completed_scene(
    *,
    scene: dict[str, Any],
    batch_payload: dict[str, Any],
    scene_output_dir: Path,
    colorized_output_dir: Path,
    resume: bool,
    pipeline: str,
    expected_reference_paths: list[Path],
) -> bool:
    if not resume:
        return False
    scene_id = str(scene["scene_id"])
    scene_clip_path = scene_output_dir / f"{scene_id}.mp4"
    colorized_clip_path = colorized_output_dir / f"{scene_id}.mp4"
    if not _is_usable_video(colorized_clip_path, reference_path=scene_clip_path):
        return False
    existing_status = _get_scene_status(batch_payload, scene_id)
    if pipeline == "deepremaster":
        if existing_status is None or existing_status.get("status") != "succeeded":
            return False
        if not all(_is_usable_image(path) for path in expected_reference_paths):
            return False
        return _recorded_keyframe_paths(existing_status) == [str(path) for path in expected_reference_paths]
    return (
        existing_status is None
        or existing_status.get("status") == "succeeded"
        or existing_status.get("output_clip") == str(colorized_clip_path)
    )


def _prepare_scene_clips(
    *,
    movie_path: Path,
    scene_output_dir: Path,
    mezzanine_path: Path,
    scenes: list[dict[str, Any]],
    config: AppConfig,
    extraction_manifest_path: Path,
) -> dict[str, float]:
    runtimes: dict[str, float] = {}
    if not _is_usable_video(mezzanine_path):
        started = time.perf_counter()
        _encode_scene_mezzanine(
            movie_path=movie_path,
            output_path=mezzanine_path,
            scenes=scenes,
            config=config,
        )
        runtimes["scene_mezzanine_encode"] = time.perf_counter() - started
        print(f"Encoded scene mezzanine: {mezzanine_path.name}")

    started = time.perf_counter()
    _split_scene_clips_by_frame_count(
        mezzanine_path=mezzanine_path,
        scene_output_dir=scene_output_dir,
        scenes=scenes,
        config=config,
    )
    runtimes["scene_frame_split"] = time.perf_counter() - started
    print(f"Frame-exact scene clips written from mezzanine: {len(scenes)}")
    _write_scene_extraction_manifest(
        manifest_path=extraction_manifest_path,
        movie_path=movie_path,
        scenes=scenes,
    )
    return runtimes


def _split_scene_clips_by_frame_count(
    *,
    mezzanine_path: Path,
    scene_output_dir: Path,
    scenes: list[dict[str, Any]],
    config: AppConfig,
) -> None:
    media_info = ffprobe_media(mezzanine_path)
    fps = str(media_info["fps"])
    fps_value = float(fps_to_decimal_string(fps))
    split_specs = [
        FrameSplitSpec(
            output_path=scene_output_dir / f"{scene['scene_id']}.mp4",
            frame_count=max(1, int(round(float(scene["duration_seconds"]) * fps_value))),
        )
        for scene in scenes
    ]
    split_video_by_frame_counts(
        input_path=mezzanine_path,
        split_specs=split_specs,
        fps=fps,
        video_codec=str(config.raw["video"]["output_codec"]),
        crf=int(config.raw["video"]["crf"]),
        pixel_format=str(config.raw["video"]["pixel_format"]),
    )


def _scene_extraction_manifest_matches(
    *,
    manifest_path: Path,
    movie_path: Path,
    scenes: list[dict[str, Any]],
) -> bool:
    payload = load_json_manifest(manifest_path, {})
    if not isinstance(payload, dict):
        return False
    if payload.get("version") != SCENE_EXTRACTION_VERSION:
        return False
    if payload.get("movie") != str(movie_path):
        return False
    if payload.get("scene_count") != len(scenes):
        return False
    return payload.get("scene_signature") == _scene_signature(scenes)


def _write_scene_extraction_manifest(
    *,
    manifest_path: Path,
    movie_path: Path,
    scenes: list[dict[str, Any]],
) -> None:
    write_json_manifest(
        manifest_path,
        {
            "version": SCENE_EXTRACTION_VERSION,
            "movie": str(movie_path),
            "scene_count": len(scenes),
            "scene_signature": _scene_signature(scenes),
            "updated_at": utc_now_iso(),
        },
    )


def _scene_signature(scenes: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "scene_id": str(scene["scene_id"]),
            "start_time": str(scene["start_time"]),
            "end_time": str(scene["end_time"]),
            "duration_seconds": str(scene["duration_seconds"]),
        }
        for scene in scenes
    ]


def _invalidate_scene_artifacts(
    *,
    scenes: list[dict[str, Any]],
    scene_output_dir: Path,
    colorized_output_dir: Path,
    deoldify_output_dir: Path,
    ddcolor_output_dir: Path,
    equalized_output_dir: Path,
    keyframe_source_dir: Path,
    keyframe_colored_dir: Path,
    keyframe_reference_dir: Path,
    scene_extraction_manifest_path: Path,
) -> None:
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        for directory in (
            scene_output_dir,
            colorized_output_dir,
            deoldify_output_dir,
            ddcolor_output_dir,
            equalized_output_dir,
        ):
            candidate = directory / f"{scene_id}.mp4"
            if candidate.exists():
                candidate.unlink()
        for directory in (keyframe_source_dir, keyframe_colored_dir, keyframe_reference_dir):
            for candidate in (directory / f"{scene_id}.png", *directory.glob(f"{scene_id}__p*.png")):
                if candidate.exists():
                    candidate.unlink()
    if scene_extraction_manifest_path.exists():
        scene_extraction_manifest_path.unlink()


def _encode_scene_mezzanine(
    *,
    movie_path: Path,
    output_path: Path,
    scenes: list[dict[str, Any]],
    config: AppConfig,
) -> None:
    keyframe_times = sorted(
        {
            timecode
            for scene in scenes
            for timecode in (str(scene["start_time"]), str(scene["end_time"]))
        }
    )
    encode_scene_mezzanine(
        input_path=movie_path,
        output_path=output_path,
        keyframe_times=keyframe_times,
        video_codec=str(config.raw["video"]["output_codec"]),
        crf=int(config.raw["video"]["crf"]),
        pixel_format=str(config.raw["video"]["pixel_format"]),
    )


def _load_batch_manifest(
    *,
    batch_manifest_path: Path,
    movie_path: Path,
    config_path: Path,
    scene_manifest_path: Path,
    scene_count: int,
    resume: bool,
    limit: int | None,
    pipeline: str,
    coloring_model: str,
    keyframe_positions: list[float],
) -> dict[str, Any]:
    if resume:
        payload = load_json_manifest(batch_manifest_path, {})
        if isinstance(payload, dict):
            if "scene_runs" not in payload and isinstance(payload.get("runs"), list):
                payload["scene_runs"] = payload.pop("runs")
            same_movie = payload.get("movie") == str(movie_path)
            same_scene_manifest = payload.get("scene_manifest_path") == str(scene_manifest_path)
            same_limit = payload.get("limit") == limit
            same_pipeline = payload.get("pipeline", "default") == pipeline
            same_coloring_model = payload.get("coloring_model", "deoldify") == coloring_model
            stored_positions = payload.get("deepremaster_keyframe_positions", [0.5])
            same_keyframe_positions = (
                pipeline != "deepremaster" or stored_positions == keyframe_positions
            )
            if (
                same_movie
                and same_scene_manifest
                and same_limit
                and same_pipeline
                and same_coloring_model
                and same_keyframe_positions
            ):
                payload.setdefault("scene_runs", [])
                payload["scene_count"] = scene_count
                payload["updated_at"] = utc_now_iso()
                return payload

    return {
        "movie": str(movie_path),
        "config_path": str(config_path.resolve()),
        "scene_manifest_path": str(scene_manifest_path),
        "scene_count": scene_count,
        "limit": limit,
        "pipeline": pipeline,
        "coloring_model": coloring_model,
        "deepremaster_keyframe_positions": keyframe_positions if pipeline == "deepremaster" else [],
        "status": "pending",
        "succeeded_scene_count": 0,
        "failed_scene_count": 0,
        "remaining_scene_count": scene_count,
        "scene_runs": [],
        "started_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }


def _get_scene_status(payload: dict[str, Any], scene_id: str) -> dict[str, Any] | None:
    for entry in payload.get("scene_runs", []):
        if entry.get("scene_id") == scene_id:
            return entry
    return None


def _upsert_scene_status(payload: dict[str, Any], status: BatchSceneStatus) -> None:
    payload["updated_at"] = utc_now_iso()
    runs = payload.setdefault("scene_runs", [])
    serialized = asdict(status)
    for index, entry in enumerate(runs):
        if entry.get("scene_id") == status.scene_id:
            runs[index] = serialized
            return
    runs.append(serialized)


def _refresh_batch_summary(payload: dict[str, Any], *, expected_scene_count: int) -> None:
    runs = payload.get("scene_runs", [])
    succeeded = sum(1 for entry in runs if entry.get("status") == "succeeded")
    failed = sum(1 for entry in runs if entry.get("status") == "failed")
    remaining = max(0, expected_scene_count - succeeded - failed)

    payload["scene_count"] = expected_scene_count
    payload["succeeded_scene_count"] = succeeded
    payload["failed_scene_count"] = failed
    payload["remaining_scene_count"] = remaining
    payload["stage_runtime_seconds"] = _sum_stage_runtimes(runs)


def _sum_stage_runtimes(runs: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for entry in runs:
        stage_runtimes = entry.get("stage_runtime_seconds")
        if not isinstance(stage_runtimes, dict):
            continue
        for stage_name, runtime_seconds in stage_runtimes.items():
            totals[stage_name] = totals.get(stage_name, 0.0) + float(runtime_seconds)
    return {stage_name: round(runtime_seconds, 3) for stage_name, runtime_seconds in totals.items()}


def _optional_positive_float(value: object) -> float | None:
    """Normalize a nullable config setting before constructing the MLX runner."""
    if value is None:
        return None
    result = float(value)
    if result <= 0.0:
        raise ValueError("DeepRemaster target_chroma_ratio must be positive when configured")
    return result


def _recorded_keyframe_paths(status: dict[str, Any]) -> list[str]:
    keyframes = status.get("keyframes")
    if isinstance(keyframes, list):
        return [
            str(record["colored_path"])
            for record in keyframes
            if isinstance(record, dict) and isinstance(record.get("colored_path"), str)
        ]
    keyframe = status.get("keyframe")
    if isinstance(keyframe, dict) and isinstance(keyframe.get("colored_path"), str):
        return [str(keyframe["colored_path"])]
    return []


def _is_usable_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image is not None and image.size > 0


def _is_usable_video(path: Path, *, reference_path: Path | None = None) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        media_info = ffprobe_media(path)
        reference_info = ffprobe_media(reference_path) if reference_path is not None and reference_path.exists() else None
    except Exception:
        return False
    if float(media_info["duration_seconds"]) <= 0.0:
        return False
    if reference_info is None:
        return True

    frame_count = int(media_info.get("frame_count", 0))
    reference_frame_count = int(reference_info.get("frame_count", 0))
    if frame_count > 0 and reference_frame_count > 0:
        return frame_count == reference_frame_count

    duration_delta = abs(float(media_info["duration_seconds"]) - float(reference_info["duration_seconds"]))
    return duration_delta <= 0.05
