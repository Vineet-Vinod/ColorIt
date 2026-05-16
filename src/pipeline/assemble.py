from __future__ import annotations

from pathlib import Path

from src.pipeline.config import AppConfig
from src.pipeline.ffmpeg_utils import (
    concat_videos,
    count_video_frames,
    ffprobe_media,
    fps_to_decimal_string,
    normalize_cfr_video,
    normalize_silent_cfr_video,
)
from src.pipeline.manifest import write_json_manifest
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths
from src.pipeline.scenes import load_scene_manifest


def run_assemble_final(
    *,
    config: AppConfig,
    scene_manifest_path: Path,
    source_movie_path: Path,
    output_path: Path | None,
    limit: int | None,
) -> int:
    paths = resolve_project_paths(config)
    ensure_runtime_directories(paths)

    scene_manifest_path = scene_manifest_path.expanduser().resolve()
    source_movie_path = source_movie_path.expanduser().resolve()
    manifest = load_scene_manifest(scene_manifest_path)
    scenes = manifest["scenes"]
    if limit is not None:
        scenes = scenes[:limit]

    run_id = scene_manifest_path.stem
    colorized_scene_dir = paths.colorized_dir / "scenes" / run_id
    assembly_scene_dir = paths.final_dir / f"{run_id}_assembly_clips"
    if not colorized_scene_dir.exists():
        raise FileNotFoundError(f"Colorized scene directory not found: {colorized_scene_dir}")

    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else paths.final_dir / f"{run_id}_assembled.mp4"
    )
    concat_list_path = paths.manifest_dir / f"concat_{run_id}.txt"
    raw_concat_path = paths.final_dir / f"{run_id}_concat_raw{output_path.suffix}"

    source_info = ffprobe_media(source_movie_path)
    fps = str(source_info["fps"])
    fps_value = float(fps_to_decimal_string(fps))
    assembly_scene_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    assembly_items: list[dict[str, str | int]] = []
    for index, scene in enumerate(scenes, start=1):
        clip_path = colorized_scene_dir / f"{scene['scene_id']}.mp4"
        if not clip_path.exists():
            raise FileNotFoundError(f"Missing colorized scene clip: {clip_path}")
        expected_scene_frames = int(round(float(scene["duration_seconds"]) * fps_value))
        normalized_clip_path = assembly_scene_dir / clip_path.name
        if not _clip_matches_frame_count(normalized_clip_path, expected_scene_frames):
            normalize_silent_cfr_video(
                input_path=clip_path,
                output_path=normalized_clip_path,
                fps=fps,
                frame_count=expected_scene_frames,
                video_codec=str(config.compression.get("video_codec", config.raw["video"]["output_codec"])),
                crf=int(config.compression.get("crf", config.raw["video"]["crf"])),
                pixel_format=str(config.raw["video"]["pixel_format"]),
                preset=str(config.compression.get("preset", "veryfast")),
            )
        lines.append(f"file '{normalized_clip_path.as_posix()}'")
        assembly_items.append(
            {
                "index": index,
                "scene_id": scene["scene_id"],
                "clip_path": str(normalized_clip_path),
                "source_clip_path": str(clip_path),
                "frame_count": expected_scene_frames,
            }
        )

    concat_list_path.write_text("\n".join(lines) + "\n")
    concat_videos(input_list_path=concat_list_path, output_path=raw_concat_path)

    if limit is None:
        frame_count = int(source_info.get("frame_count", 0))
        if frame_count <= 0:
            frame_count = count_video_frames(source_movie_path)
        if frame_count <= 0:
            frame_count = int(round(float(source_info["video_duration_seconds"]) * fps_value))
    else:
        frame_count = int(
            round(sum(float(scene["duration_seconds"]) for scene in scenes) * fps_value)
        )
    normalize_cfr_video(
        input_path=raw_concat_path,
        audio_source_path=source_movie_path,
        output_path=output_path,
        fps=fps,
        frame_count=frame_count,
        video_codec=str(config.compression.get("video_codec", config.raw["video"]["output_codec"])),
        crf=int(config.compression.get("crf", config.raw["video"]["crf"])),
        pixel_format=str(config.raw["video"]["pixel_format"]),
        preset=str(config.compression.get("preset", "veryfast")),
        audio_bitrate=str(config.compression.get("audio_bitrate", "192k")),
    )
    if raw_concat_path.exists():
        raw_concat_path.unlink()

    write_json_manifest(
        paths.manifest_dir / f"final_assembly_{run_id}.json",
        {
            "scene_manifest_path": str(scene_manifest_path),
            "source_movie_path": str(source_movie_path),
            "output_path": str(output_path),
            "raw_concat_path": str(raw_concat_path),
            "concat_list_path": str(concat_list_path),
            "assembly_scene_dir": str(assembly_scene_dir),
            "fps": fps,
            "frame_count": frame_count,
            "scene_count": len(assembly_items),
            "scenes": assembly_items,
        },
    )
    print(f"Final assembly written to {output_path}")
    return 0


def _clip_matches_frame_count(path: Path, frame_count: int) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        return int(ffprobe_media(path).get("frame_count", 0)) == frame_count
    except Exception:
        return False
