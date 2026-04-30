from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.pipeline.segments import (
    SegmentFrame,
    SegmentInstance,
    SegmentManifest,
    SegmentTrack,
    load_segment_manifest,
    mask_bbox,
    resolve_manifest_path,
    write_segment_manifest,
)


def run_filter_segments(
    *,
    segment_manifest_path: Path,
    output_dir: Path,
    include_labels: list[str],
    veto_labels: list[str],
    output_label: str,
    min_area: int,
    close_px: int,
    erode_px: int,
    dilate_px: int,
    overwrite: bool,
) -> int:
    segment_manifest_path = segment_manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not segment_manifest_path.exists():
        raise FileNotFoundError(f"Segment manifest not found: {segment_manifest_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Segment output dir is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = load_segment_manifest(segment_manifest_path)
    include_label_set = set(include_labels)
    veto_label_set = set(veto_labels)
    if not include_label_set:
        raise ValueError("At least one --include-label value is required.")

    track_id = f"filtered_{output_label}"
    output_track = SegmentTrack(
        track_id=track_id,
        label=output_label,
        kind="filtered_segment",
        metadata={
            "source_manifest": str(segment_manifest_path),
            "include_labels": sorted(include_label_set),
            "veto_labels": sorted(veto_label_set),
            "min_area": min_area,
            "close_px": close_px,
            "erode_px": erode_px,
            "dilate_px": dilate_px,
        },
    )

    frames: list[SegmentFrame] = []
    width = int(source_manifest["width"])
    height = int(source_manifest["height"])
    for frame_payload in source_manifest["frames"]:
        frame_index = int(frame_payload["frame_index"])
        include_mask = np.zeros((height, width), dtype=np.uint8)
        veto_mask = np.zeros((height, width), dtype=np.uint8)

        for instance in frame_payload.get("instances", []):
            label = str(instance.get("label", ""))
            if label not in include_label_set and label not in veto_label_set:
                continue
            mask = _load_instance_mask(
                segment_manifest_path=segment_manifest_path,
                instance=instance,
                width=width,
                height=height,
            )
            if label in include_label_set:
                include_mask = cv2.bitwise_or(include_mask, mask)
            if label in veto_label_set:
                veto_mask = cv2.bitwise_or(veto_mask, mask)

        output_mask = cv2.bitwise_and(include_mask, cv2.bitwise_not(veto_mask))
        output_mask = _clean_mask(
            output_mask,
            close_px=close_px,
            erode_px=erode_px,
            dilate_px=dilate_px,
            min_area=min_area,
        )

        instances: list[SegmentInstance] = []
        if np.any(output_mask):
            mask_path = output_dir / "masks" / f"frame_{frame_index:06d}_{track_id}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(output_mask).save(mask_path)
            instances.append(
                SegmentInstance(
                    track_id=track_id,
                    label=output_label,
                    kind="filtered_segment",
                    mask_path=str(mask_path.relative_to(output_dir)),
                    bbox=mask_bbox(output_mask),
                    confidence=1.0,
                    metadata={
                        "source_frame_index": frame_index,
                    },
                )
            )
        frames.append(SegmentFrame(frame_index=frame_index, instances=instances))

    manifest = SegmentManifest(
        source_clip=str(source_manifest["source_clip"]),
        backend="filter-segments",
        frame_count=int(source_manifest["frame_count"]),
        width=width,
        height=height,
        fps=str(source_manifest["fps"]),
        tracks={track_id: output_track},
        frames=frames,
        metadata={
            "source_manifest": str(segment_manifest_path),
            "source_backend": source_manifest.get("backend"),
        },
    )
    write_segment_manifest(output_dir / "segment_manifest.json", manifest)
    print(f"Filtered segment manifest written: {output_dir / 'segment_manifest.json'}")
    return 0


def _load_instance_mask(
    *,
    segment_manifest_path: Path,
    instance: dict,
    width: int,
    height: int,
) -> np.ndarray:
    mask_path = resolve_manifest_path(
        manifest_path=segment_manifest_path,
        relative_path=str(instance["mask_path"]),
    )
    mask = np.asarray(Image.open(mask_path).convert("L"))
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _clean_mask(
    mask: np.ndarray,
    *,
    close_px: int,
    erode_px: int,
    dilate_px: int,
    min_area: int,
) -> np.ndarray:
    output = mask
    if close_px > 0:
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, _kernel(close_px))
    if erode_px > 0:
        output = cv2.erode(output, _kernel(erode_px), iterations=1)
    if dilate_px > 0:
        output = cv2.dilate(output, _kernel(dilate_px), iterations=1)
    if min_area > 0:
        output = _remove_small_components(output, min_area=min_area)
    return output


def _remove_small_components(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros(mask.shape, dtype=np.uint8)
    for component_index in range(1, component_count):
        if int(stats[component_index, cv2.CC_STAT_AREA]) >= min_area:
            output[labels == component_index] = 255
    return output


def _kernel(radius_px: int) -> np.ndarray:
    size = max(1, radius_px * 2 + 1)
    return np.ones((size, size), dtype=np.uint8)
