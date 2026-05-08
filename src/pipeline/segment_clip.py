from __future__ import annotations

from pathlib import Path

from src.pipeline.segment_backends import (
    DEFAULT_HUMAN_PARSER_MODEL_ID,
    run_human_parser_segmentation,
    run_person_maskrcnn_segmentation,
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
    frame_stride: int,
    include_labels: list[str] | None,
    score_threshold: float,
    mask_threshold: float,
    min_area: int,
    iou_threshold: float,
    max_center_distance: float,
    max_missing_frames: int,
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
            frame_stride=frame_stride,
            include_labels=include_labels,
            overwrite=overwrite,
        )
    elif backend == "person-maskrcnn":
        manifest = run_person_maskrcnn_segmentation(
            input_path=input_path,
            output_dir=output_dir,
            device=device,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            min_area=min_area,
            iou_threshold=iou_threshold,
            max_center_distance=max_center_distance,
            max_missing_frames=max_missing_frames,
            overwrite=overwrite,
        )
    else:
        raise ValueError(f"Unsupported segmentation backend: {backend}")

    manifest_path = output_dir.expanduser().resolve() / "segment_manifest.json"
    print(f"Segmented {manifest.frame_count} frame(s) with backend '{backend}'.")
    print(f"Segment manifest written: {manifest_path}")
    return 0
