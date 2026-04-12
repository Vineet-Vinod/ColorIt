from __future__ import annotations

from pathlib import Path

from src.pipeline.config import AppConfig
from src.pipeline.ffmpeg_utils import concat_videos
from src.pipeline.manifest import write_json_manifest
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths
from src.pipeline.scenes import load_scene_manifest


def run_assemble_final(
    *,
    config: AppConfig,
    scene_manifest_path: Path,
    output_path: Path | None,
    limit: int | None,
) -> int:
    paths = resolve_project_paths(config)
    ensure_runtime_directories(paths)

    scene_manifest_path = scene_manifest_path.expanduser().resolve()
    manifest = load_scene_manifest(scene_manifest_path)
    scenes = manifest["scenes"]
    if limit is not None:
        scenes = scenes[:limit]

    run_id = scene_manifest_path.stem
    colorized_scene_dir = paths.colorized_dir / "scenes" / run_id
    if not colorized_scene_dir.exists():
        raise FileNotFoundError(f"Colorized scene directory not found: {colorized_scene_dir}")

    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else paths.final_dir / f"{run_id}_assembled.mp4"
    )
    concat_list_path = paths.manifest_dir / f"concat_{run_id}.txt"

    lines: list[str] = []
    assembly_items: list[dict[str, str | int]] = []
    for index, scene in enumerate(scenes, start=1):
        clip_path = colorized_scene_dir / f"{scene['scene_id']}.mp4"
        if not clip_path.exists():
            raise FileNotFoundError(f"Missing colorized scene clip: {clip_path}")
        lines.append(f"file '{clip_path.as_posix()}'")
        assembly_items.append(
            {
                "index": index,
                "scene_id": scene["scene_id"],
                "clip_path": str(clip_path),
            }
        )

    concat_list_path.write_text("\n".join(lines) + "\n")
    concat_videos(input_list_path=concat_list_path, output_path=output_path)

    write_json_manifest(
        paths.manifest_dir / f"final_assembly_{run_id}.json",
        {
            "scene_manifest_path": str(scene_manifest_path),
            "output_path": str(output_path),
            "concat_list_path": str(concat_list_path),
            "scene_count": len(assembly_items),
            "scenes": assembly_items,
        },
    )
    print(f"Final assembly written to {output_path}")
    return 0
