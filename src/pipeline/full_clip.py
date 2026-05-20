from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import time

from src.pipeline.colorize_clip import run_deoldify_clip
from src.pipeline.config import AppConfig
from src.pipeline.ddcolor_clip import DEFAULT_DDCOLOR_WEIGHTS_PATH, run_ddcolor_clip
from src.pipeline.ffmpeg_utils import compress_video
from src.pipeline.model_chroma_propagate import run_model_chroma_propagate
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths
from src.pipeline.preprocess import equalize_clip_luma_clahe


CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = 8
CLAHE_STRENGTH = 0.70


def run_full_colorize_clip(
    *,
    config: AppConfig,
    config_path: Path,
    input_path: Path,
    output_path: Path,
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

    run_id = _build_clip_run_id(input_path)
    work_dir = paths.root / "tmp" / "full_colorize_clip" / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    equalized_source_path = work_dir / "source_clahe.mp4"
    deoldify_path = work_dir / "deoldify.mp4"
    ddcolor_path = work_dir / "ddcolor.mp4"
    chroma_path = work_dir / "chroma_propagated.mp4"

    print(f"Input clip: {input_path}")
    print(f"Output clip: {output_path}")
    print("Pipeline: CLAHE source -> DeOldify + DDColor -> chroma propagation -> compression")
    print(
        "CLAHE: "
        f"clip_limit={CLAHE_CLIP_LIMIT}, tile_grid_size={CLAHE_TILE_GRID_SIZE}, strength={CLAHE_STRENGTH}"
    )

    started = time.perf_counter()
    stage_started = time.perf_counter()
    frame_count = equalize_clip_luma_clahe(
        input_path=input_path,
        output_path=equalized_source_path,
        clip_limit=CLAHE_CLIP_LIMIT,
        tile_grid_size=CLAHE_TILE_GRID_SIZE,
        strength=CLAHE_STRENGTH,
        crf=int(config.video["crf"]),
        include_audio=True,
    )
    print(f"CLAHE source written: {equalized_source_path}")
    print(f"CLAHE frames: {frame_count}")
    print(f"CLAHE runtime seconds: {time.perf_counter() - stage_started:.2f}")

    stage_started = time.perf_counter()
    run_deoldify_clip(
        config=config,
        config_path=config_path,
        input_path=equalized_source_path,
        output_path=deoldify_path,
        manifest_path=paths.manifest_dir / "clip_runs.json",
        overwrite=True,
        include_audio=False,
    )
    print(f"DeOldify stage seconds: {time.perf_counter() - stage_started:.2f}")

    stage_started = time.perf_counter()
    run_ddcolor_clip(
        input_path=equalized_source_path,
        output_path=ddcolor_path,
        weights_path=paths.root / DEFAULT_DDCOLOR_WEIGHTS_PATH,
        input_size=256,
        device="auto",
        output_preset="ultrafast",
        overwrite=True,
        include_audio=False,
    )
    print(f"DDColor stage seconds: {time.perf_counter() - stage_started:.2f}")

    stage_started = time.perf_counter()
    run_model_chroma_propagate(
        source_path=deoldify_path,
        model_color_path=ddcolor_path,
        output_path=chroma_path,
        keyframe_stride=35,
        chroma_blend=1.0,
        audio_input_path=equalized_source_path,
        overwrite=True,
    )
    print(f"Chroma propagation stage seconds: {time.perf_counter() - stage_started:.2f}")

    stage_started = time.perf_counter()
    compression = config.compression
    compress_video(
        input_path=chroma_path,
        output_path=output_path,
        video_codec=str(compression.get("video_codec", config.video["output_codec"])),
        preset=str(compression.get("preset", "medium")),
        crf=int(compression.get("crf", config.video["crf"])),
        audio_codec=str(compression.get("audio_codec", "aac")),
        audio_bitrate=str(compression.get("audio_bitrate", "160k")),
        faststart=bool(compression.get("faststart", True)),
    )
    print(f"Compression stage seconds: {time.perf_counter() - stage_started:.2f}")
    print(f"Full clip colorization complete in {time.perf_counter() - started:.2f}s")
    return 0


def _build_clip_run_id(input_path: Path) -> str:
    safe_stem = "".join(character if character.isalnum() else "_" for character in input_path.stem).strip("_")
    location_hash = sha1(str(input_path).encode("utf-8")).hexdigest()[:8]
    return f"{safe_stem}_{location_hash}"
