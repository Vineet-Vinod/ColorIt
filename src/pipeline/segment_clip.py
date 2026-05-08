from __future__ import annotations

from pathlib import Path

from src.pipeline.segment_backends import DEFAULT_HUMAN_PARSER_MODEL_ID, run_human_parser_segmentation


def run_segment_clip(
    *,
    input_path: Path,
    output_dir: Path,
    model_id: str | None,
    device: str,
    frame_stride: int,
    include_labels: list[str] | None,
    overwrite: bool,
) -> int:
    manifest = run_human_parser_segmentation(
        input_path=input_path,
        output_dir=output_dir,
        model_id=model_id or DEFAULT_HUMAN_PARSER_MODEL_ID,
        device=device,
        frame_stride=frame_stride,
        include_labels=include_labels,
        overwrite=overwrite,
    )
    manifest_path = output_dir.expanduser().resolve() / "segment_manifest.json"
    print(f"Segmented {manifest.frame_count} frame(s) with human-parser.")
    print(f"Segment manifest written: {manifest_path}")
    return 0
