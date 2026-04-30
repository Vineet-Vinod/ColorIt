from __future__ import annotations

from pathlib import Path

from src.pipeline.segment_backends import run_polygon_segmentation


def run_segment_clip(
    *,
    input_path: Path,
    backend: str,
    output_dir: Path,
    tracks_path: Path | None,
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
    else:
        raise ValueError(f"Unsupported segmentation backend: {backend}")

    manifest_path = output_dir.expanduser().resolve() / "segment_manifest.json"
    print(f"Segmented {manifest.frame_count} frame(s) with backend '{backend}'.")
    print(f"Segment manifest written: {manifest_path}")
    return 0
