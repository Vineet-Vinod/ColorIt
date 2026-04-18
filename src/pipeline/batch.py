from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

from src.pipeline.colorize_clip import run_colorize_clip
from src.pipeline.config import AppConfig
from src.pipeline.ffmpeg_utils import extract_clip
from src.pipeline.manifest import load_json_manifest, utc_now_iso, write_json_manifest
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths
from src.pipeline.scenes import load_scene_manifest


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
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    colorized_output_dir.mkdir(parents=True, exist_ok=True)
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
                output_path=colorized_clip_path,
                manifest_path=scene_runs_manifest_path,
                overwrite=True,
                model_bundle=shared_bundle,
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
