from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import time

from PIL import Image

from src.pipeline.config import AppConfig
from src.pipeline.ffmpeg_utils import (
    encode_video_from_frames,
    extract_frames,
    ffprobe_media,
)
from src.pipeline.inference import colorize_pil_image
from src.pipeline.manifest import write_json_manifest
from src.pipeline.model_loader import load_colorizer_bundle
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
    bundle = load_colorizer_bundle(config)

    run_hash = sha256(f"{input_path}:{output_path}:{config_path.resolve()}".encode()).hexdigest()[:12]
    frame_root = paths.frames_dir / f"clip_{run_hash}"
    source_frames_dir = frame_root / "source"
    colorized_frames_dir = frame_root / "colorized"

    print(f"Input clip: {input_path}")
    print(f"Output clip: {output_path}")
    print(f"Backend: {bundle.backend}")
    print(f"Render factor: {config.model['render_factor']}")

    started = time.perf_counter()
    extract_frames(input_path=input_path, output_dir=source_frames_dir)

    frame_paths = sorted(source_frames_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError("No frames were extracted from the input clip.")

    colorized_frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in frame_paths:
        input_image = Image.open(frame_path).convert("RGB")
        result = colorize_pil_image(
            model_bundle=bundle,
            input_image=input_image,
            render_factor=int(config.model["render_factor"]),
        )
        result.save(colorized_frames_dir / frame_path.name)

    encode_video_from_frames(
        frame_dir=colorized_frames_dir,
        output_path=output_path,
        fps=str(media_info["fps"]),
        video_codec=str(config.raw["video"]["output_codec"]),
        crf=int(config.raw["video"]["crf"]),
        pixel_format=str(config.raw["video"]["pixel_format"]),
        audio_input_path=input_path,
    )
    runtime_seconds = time.perf_counter() - started

    record = ClipRunRecord(
        input_path=str(input_path),
        output_path=str(output_path),
        config_path=str(config_path.resolve()),
        render_factor=int(config.model["render_factor"]),
        backend=bundle.backend,
        runtime_seconds=runtime_seconds,
        frame_count=len(frame_paths),
        fps=str(media_info["fps"]),
        width=int(media_info["width"]),
        height=int(media_info["height"]),
        status="succeeded",
    )

    manifest_path = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else paths.manifest_dir / "probe_runs.json"
    )
    update_probe_runs_manifest(manifest_path, record)
    print(f"Clip colorization succeeded in {runtime_seconds:.2f}s")
    print(f"Run manifest updated: {manifest_path}")
    return 0


def update_probe_runs_manifest(manifest_path: Path, record: ClipRunRecord) -> None:
    if manifest_path.exists():
        import json

        payload = json.loads(manifest_path.read_text())
    else:
        payload = {"runs": []}

    payload.setdefault("runs", []).append(asdict(record))
    write_json_manifest(manifest_path, payload)
