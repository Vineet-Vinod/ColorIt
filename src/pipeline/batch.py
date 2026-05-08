from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

from src.pipeline.colorize_clip import run_colorize_clip
from src.pipeline.config import AppConfig
from src.pipeline.ddcolor_clip import DEFAULT_DDCOLOR_WEIGHTS_PATH, run_ddcolor_clip
from src.pipeline.ffmpeg_utils import extract_clip
from src.pipeline.manifest import load_json_manifest, utc_now_iso, write_json_manifest
from src.pipeline.model_chroma_propagate import run_model_chroma_propagate
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths
from src.pipeline.scenes import load_scene_manifest
from src.pipeline.segment_clip import run_segment_clip


GARMENT_LABELS = ["upper_clothes", "dress", "skirt", "pants", "coat"]
PROTECT_LABELS = ["face", "hair", "left_arm", "right_arm", "left_leg", "right_leg"]
SPLIT_LABELS = ["face", "hair"]


@dataclass(frozen=True)
class BatchSceneStatus:
    scene_id: str
    input_clip: str
    output_clip: str
    status: str
    runtime_seconds: float | None = None
    error: str | None = None


def run_colorize_batch(
    *,
    config: AppConfig,
    config_path: Path,
    movie_path: Path,
    scene_manifest_path: Path,
    resume: bool,
    limit: int | None,
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
    segment_output_dir = paths.colorized_dir / "segments" / run_id
    for directory in (scene_output_dir, colorized_output_dir, deoldify_output_dir, ddcolor_output_dir, segment_output_dir):
        directory.mkdir(parents=True, exist_ok=True)
    cleanup_scene_clips = bool(config.raw.get("runtime", {}).get("cleanup_scene_clips", True))

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
    )
    batch_payload["status"] = "running"
    batch_payload["updated_at"] = utc_now_iso()
    write_json_manifest(batch_manifest_path, batch_payload)

    print(f"Movie: {movie_path}")
    print(f"Scene manifest: {scene_manifest_path}")
    print(f"Batch scene count: {len(scenes)}")
    print(f"Resume mode: {resume}")
    print("Loading colorizer model once for batch reuse...")
    shared_bundle = load_colorizer_bundle(config)

    for scene in scenes:
        scene_id = scene["scene_id"]
        scene_clip_path = scene_output_dir / f"{scene_id}.mp4"
        deoldify_clip_path = deoldify_output_dir / f"{scene_id}.mp4"
        ddcolor_clip_path = ddcolor_output_dir / f"{scene_id}.mp4"
        segment_dir = segment_output_dir / scene_id
        colorized_clip_path = colorized_output_dir / f"{scene_id}.mp4"
        existing_status = _get_scene_status(batch_payload, scene_id)

        if (
            resume
            and existing_status is not None
            and existing_status.get("status") == "succeeded"
            and colorized_clip_path.exists()
        ):
            print(f"Skipping completed scene: {scene_id}")
            continue

        started = time.perf_counter()
        try:
            if not scene_clip_path.exists():
                extract_clip(
                    input_path=movie_path,
                    output_path=scene_clip_path,
                    start_time=scene["start_time"],
                    end_time=scene["end_time"],
                    video_codec=str(config.raw["video"]["output_codec"]),
                    crf=int(config.raw["video"]["crf"]),
                    pixel_format=str(config.raw["video"]["pixel_format"]),
                )
                print(f"Extracted scene clip: {scene_clip_path.name}")

            run_colorize_clip(
                config=config,
                config_path=config_path,
                input_path=scene_clip_path,
                output_path=deoldify_clip_path,
                manifest_path=scene_runs_manifest_path,
                overwrite=True,
                model_bundle=shared_bundle,
            )
            run_ddcolor_clip(
                input_path=scene_clip_path,
                output_path=ddcolor_clip_path,
                weights_path=paths.root / DEFAULT_DDCOLOR_WEIGHTS_PATH,
                input_size=256,
                device="auto",
                output_preset="ultrafast",
                overwrite=True,
            )
            run_segment_clip(
                input_path=scene_clip_path,
                output_dir=segment_dir,
                model_id=None,
                device="auto",
                frame_stride=8,
                include_labels=sorted(set(GARMENT_LABELS + PROTECT_LABELS + SPLIT_LABELS)),
                overwrite=True,
            )
            run_model_chroma_propagate(
                source_path=deoldify_clip_path,
                model_color_path=ddcolor_clip_path,
                output_path=colorized_clip_path,
                keyframe_stride=35,
                chroma_blend=0.80,
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
                semantic_consensus_manifest_path=segment_dir / "segment_manifest.json",
                semantic_consensus_labels=GARMENT_LABELS,
                semantic_protect_labels=PROTECT_LABELS,
                semantic_protect_dilate=3,
                semantic_split_labels=SPLIT_LABELS,
                semantic_consensus_strength=0.92,
                semantic_consensus_min_area=600,
                semantic_consensus_model_chroma_min=8.0,
                semantic_consensus_feather_sigma=1.2,
                semantic_consensus_diversify_strength=0.65,
                semantic_consensus_diversify_threshold=20.0,
                overwrite=True,
            )
            status = BatchSceneStatus(
                scene_id=scene_id,
                input_clip=str(scene_clip_path),
                output_clip=str(colorized_clip_path),
                status="succeeded",
                runtime_seconds=time.perf_counter() - started,
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
                error=str(exc),
            )
            print(f"Scene failed: {scene_id}: {exc}")

        _upsert_scene_status(batch_payload, status)
        _refresh_batch_summary(batch_payload, expected_scene_count=len(scenes))
        write_json_manifest(batch_manifest_path, batch_payload)

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


def _load_batch_manifest(
    *,
    batch_manifest_path: Path,
    movie_path: Path,
    config_path: Path,
    scene_manifest_path: Path,
    scene_count: int,
    resume: bool,
    limit: int | None,
) -> dict[str, Any]:
    if resume:
        payload = load_json_manifest(batch_manifest_path, {})
        if isinstance(payload, dict):
            if "scene_runs" not in payload and isinstance(payload.get("runs"), list):
                payload["scene_runs"] = payload.pop("runs")
            same_movie = payload.get("movie") == str(movie_path)
            same_scene_manifest = payload.get("scene_manifest_path") == str(scene_manifest_path)
            same_limit = payload.get("limit") == limit
            if same_movie and same_scene_manifest and same_limit:
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
