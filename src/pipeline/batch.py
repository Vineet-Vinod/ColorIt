from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import time
from typing import Any

from src.pipeline.colorize_clip import run_colorize_clip
from src.pipeline.config import AppConfig
from src.pipeline.ddcolor_clip import DEFAULT_DDCOLOR_WEIGHTS_PATH, run_ddcolor_clip
from src.pipeline.ffmpeg_utils import copy_clip, encode_scene_mezzanine, ffprobe_media, segment_copy_clips
from src.pipeline.manifest import load_json_manifest, utc_now_iso, write_json_manifest
from src.pipeline.model_chroma_propagate import run_model_chroma_propagate
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths
from src.pipeline.scenes import load_scene_manifest


SCENE_EXTRACTION_VERSION = "scene-copy-cfr-v3-per-scene-copy"


@dataclass(frozen=True)
class BatchSceneStatus:
    scene_id: str
    input_clip: str
    output_clip: str
    status: str
    runtime_seconds: float | None = None
    stage_runtime_seconds: dict[str, float] | None = None
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
    for directory in (scene_output_dir, colorized_output_dir, deoldify_output_dir, ddcolor_output_dir):
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
    shared_bundle = None
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
            scene_extraction_manifest_path=scene_extraction_manifest_path,
        )
        batch_payload["scene_runs"] = []
        _refresh_batch_summary(batch_payload, expected_scene_count=len(scenes))
        write_json_manifest(batch_manifest_path, batch_payload)
    scene_clips_prepared = False
    scene_preparation_runtimes: dict[str, float] = {}

    for scene in scenes:
        scene_id = scene["scene_id"]
        scene_clip_path = scene_output_dir / f"{scene_id}.mp4"
        deoldify_clip_path = deoldify_output_dir / f"{scene_id}.mp4"
        ddcolor_clip_path = ddcolor_output_dir / f"{scene_id}.mp4"
        colorized_clip_path = colorized_output_dir / f"{scene_id}.mp4"
        existing_status = _get_scene_status(batch_payload, scene_id)

        if (
            resume
            and _is_usable_video(colorized_clip_path, reference_path=scene_clip_path)
            and (
                existing_status is None
                or existing_status.get("status") == "succeeded"
                or existing_status.get("output_clip") == str(colorized_clip_path)
            )
        ):
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
                        use_segment_copy=False,
                    )
                    scene_clips_prepared = True
                if not _is_usable_video(scene_clip_path):
                    copy_clip(
                        input_path=scene_mezzanine_path,
                        output_path=scene_clip_path,
                        start_time=scene["start_time"],
                        duration_seconds=float(scene["duration_seconds"]),
                    )
                stage_runtimes["scene_extraction"] = time.perf_counter() - stage_started
                stage_runtimes.update(scene_preparation_runtimes)
                scene_preparation_runtimes = {}
                print(f"Extracted scene clip: {scene_clip_path.name}")
            else:
                stage_runtimes["scene_extraction"] = 0.0

            if resume and _is_usable_video(deoldify_clip_path, reference_path=scene_clip_path):
                print(f"Reusing DeOldify clip: {deoldify_clip_path.name}")
                stage_runtimes["deoldify"] = 0.0
            else:
                if shared_bundle is None:
                    print("Loading colorizer model once for batch reuse...")
                    shared_bundle = load_colorizer_bundle(config)
                stage_started = time.perf_counter()
                run_colorize_clip(
                    config=config,
                    config_path=config_path,
                    input_path=scene_clip_path,
                    output_path=deoldify_clip_path,
                    manifest_path=scene_runs_manifest_path,
                    overwrite=True,
                    model_bundle=shared_bundle,
                    include_audio=False,
                )
                stage_runtimes["deoldify"] = time.perf_counter() - stage_started

            if resume and _is_usable_video(ddcolor_clip_path, reference_path=scene_clip_path):
                print(f"Reusing DDColor clip: {ddcolor_clip_path.name}")
                stage_runtimes["ddcolor"] = 0.0
            else:
                stage_started = time.perf_counter()
                run_ddcolor_clip(
                    input_path=scene_clip_path,
                    output_path=ddcolor_clip_path,
                    weights_path=paths.root / DEFAULT_DDCOLOR_WEIGHTS_PATH,
                    input_size=256,
                    device="auto",
                    output_preset="ultrafast",
                    overwrite=True,
                    include_audio=False,
                )
                stage_runtimes["ddcolor"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            run_model_chroma_propagate(
                source_path=deoldify_clip_path,
                model_color_path=ddcolor_clip_path,
                output_path=colorized_clip_path,
                keyframe_stride=35,
                chroma_blend=1.0,
                audio_input_path=scene_clip_path,
                overwrite=True,
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


def _prepare_scene_clips(
    *,
    movie_path: Path,
    scene_output_dir: Path,
    mezzanine_path: Path,
    scenes: list[dict[str, Any]],
    config: AppConfig,
    extraction_manifest_path: Path,
    use_segment_copy: bool,
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

    if use_segment_copy and _can_segment_scenes(scenes):
        started = time.perf_counter()
        _segment_scene_clips(
            mezzanine_path=mezzanine_path,
            scene_output_dir=scene_output_dir,
            scenes=scenes,
        )
        runtimes["scene_stream_copy_split"] = time.perf_counter() - started
        print(f"Stream-copied scene clips from mezzanine: {len(scenes)}")
        _write_scene_extraction_manifest(
            manifest_path=extraction_manifest_path,
            movie_path=movie_path,
            scenes=scenes,
        )
    else:
        _write_scene_extraction_manifest(
            manifest_path=extraction_manifest_path,
            movie_path=movie_path,
            scenes=scenes,
        )
    return runtimes


def _can_segment_scenes(scenes: list[dict[str, Any]]) -> bool:
    previous_end: str | None = None
    for scene in scenes:
        start_time = str(scene["start_time"])
        if previous_end is not None and start_time != previous_end:
            return False
        previous_end = str(scene["end_time"])
    return True


def _segment_scene_clips(
    *,
    mezzanine_path: Path,
    scene_output_dir: Path,
    scenes: list[dict[str, Any]],
) -> None:
    segment_dir = scene_output_dir / "_segments"
    if segment_dir.exists():
        shutil.rmtree(segment_dir)
    segment_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = segment_dir / "scene_%06d.mp4"
    segment_times = [str(scene["end_time"]) for scene in scenes[:-1]]
    segment_copy_clips(
        input_path=mezzanine_path,
        output_pattern=output_pattern,
        segment_times=segment_times,
    )
    for index, scene in enumerate(scenes):
        source_path = segment_dir / f"scene_{index:06d}.mp4"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing stream-copied scene segment: {source_path}")
        target_path = scene_output_dir / f"{scene['scene_id']}.mp4"
        if target_path.exists():
            target_path.unlink()
        source_path.replace(target_path)
    shutil.rmtree(segment_dir, ignore_errors=True)


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
    scene_extraction_manifest_path: Path,
) -> None:
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        for directory in (scene_output_dir, colorized_output_dir, deoldify_output_dir, ddcolor_output_dir):
            candidate = directory / f"{scene_id}.mp4"
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
