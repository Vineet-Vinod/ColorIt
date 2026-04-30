from __future__ import annotations

from pathlib import Path

from src.pipeline.segment_backends import (
    DEFAULT_HUMAN_PARSER_MODEL_ID,
    run_human_parser_segmentation,
    run_polygon_segmentation,
)


def run_segment_clip(
    *,
    input_path: Path,
    backend: str,
    output_dir: Path,
    tracks_path: Path | None,
    model_id: str | None,
    device: str,
    overwrite: bool,
) -> int:
    backend = backend.lower()
    if backend == "polygon":
        if tracks_path is None:
            raise ValueError("--tracks is required when --backend polygon is used.")
        manifest = run_polygon_segmentation(
            input_path=input_path,
            tracks_path=tracks_path,
            output_dir=output_dir,
            overwrite=overwrite,
        )
    elif backend == "human-parser":
        manifest = run_human_parser_segmentation(
            input_path=input_path,
            output_dir=output_dir,
            model_id=model_id or DEFAULT_HUMAN_PARSER_MODEL_ID,
            device=device,
            overwrite=overwrite,
        )
    else:
        raise ValueError(f"Unsupported segmentation backend: {backend}")

    manifest_path = output_dir.expanduser().resolve() / "segment_manifest.json"
    print(f"Segmented {manifest.frame_count} frame(s) with backend '{backend}'.")
    print(f"Segment manifest written: {manifest_path}")
    return 0
