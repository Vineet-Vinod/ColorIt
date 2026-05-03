from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.pipeline.ffmpeg_utils import ffprobe_media, open_rawvideo_reader
from src.pipeline.segments import (
    SegmentFrame,
    SegmentInstance,
    SegmentManifest,
    SegmentTrack,
    mask_bbox,
    write_segment_manifest,
)


@dataclass(frozen=True)
class PersonDetection:
    mask: np.ndarray
    bbox: list[int]
    score: float
    centroid: tuple[float, float]
    area: int


@dataclass
class ActiveActor:
    track_id: str
    bbox: list[int]
    centroid: tuple[float, float]
    last_frame_index: int
    missed_frames: int = 0


def run_person_maskrcnn_segmentation(
    *,
    input_path: Path,
    output_dir: Path,
    device: str,
    score_threshold: float,
    mask_threshold: float,
    min_area: int,
    iou_threshold: float,
    max_center_distance: float,
    max_missing_frames: int,
    overwrite: bool,
) -> SegmentManifest:
    torch, transforms, maskrcnn_resnet50_fpn_v2, weights_type = _load_torchvision_dependencies()

    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input clip not found: {input_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Segment output dir is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    media_info = ffprobe_media(input_path)
    width = int(media_info["width"])
    height = int(media_info["height"])
    fps = str(media_info["fps"])
    frame_bytes = width * height * 3

    selected_device = _select_device(torch, device)
    weights = weights_type.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=True)
    model.to(selected_device)
    model.eval()
    to_tensor = transforms.ToTensor()

    reader = open_rawvideo_reader(input_path=input_path)
    if reader.stdout is None or reader.stderr is None:
        raise RuntimeError("ffmpeg rawvideo reader failed to expose stdout/stderr pipes.")

    next_actor_number = 1
    active_actors: dict[str, ActiveActor] = {}
    tracks: dict[str, SegmentTrack] = {}
    frames: list[SegmentFrame] = []
    frame_index = 0
    try:
        while True:
            frame_data = reader.stdout.read(frame_bytes)
            if not frame_data:
                break
            if len(frame_data) != frame_bytes:
                raise RuntimeError(
                    f"Unexpected end of rawvideo stream; expected {frame_bytes} bytes, got {len(frame_data)}."
                )
            frame_rgb = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
            detections = _predict_people(
                torch=torch,
                model=model,
                to_tensor=to_tensor,
                frame_rgb=frame_rgb,
                device=selected_device,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
                min_area=min_area,
            )
            assignments = _assign_detections_to_actors(
                detections=detections,
                active_actors=active_actors,
                frame_index=frame_index,
                iou_threshold=iou_threshold,
                max_center_distance=max_center_distance,
                max_missing_frames=max_missing_frames,
            )

            instances: list[SegmentInstance] = []
            assigned_actor_ids: set[str] = set()
            for detection_index, detection in enumerate(detections):
                actor_id = assignments.get(detection_index)
                if actor_id is None:
                    actor_id = f"actor_{next_actor_number:03d}"
                    next_actor_number += 1
                assigned_actor_ids.add(actor_id)
                active_actors[actor_id] = ActiveActor(
                    track_id=actor_id,
                    bbox=detection.bbox,
                    centroid=detection.centroid,
                    last_frame_index=frame_index,
                    missed_frames=0,
                )
                tracks.setdefault(
                    actor_id,
                    SegmentTrack(
                        track_id=actor_id,
                        label="actor",
                        kind="person_maskrcnn",
                        metadata={"proposal_source": "torchvision_maskrcnn_resnet50_fpn_v2"},
                    ),
                )

                mask_path = output_dir / "masks" / f"frame_{frame_index:06d}_{actor_id}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(detection.mask).save(mask_path)
                instances.append(
                    SegmentInstance(
                        track_id=actor_id,
                        label="actor",
                        kind="person_maskrcnn",
                        mask_path=str(mask_path.relative_to(output_dir)),
                        bbox=detection.bbox,
                        confidence=detection.score,
                        metadata={"area": detection.area},
                    )
                )

            _age_unassigned_actors(
                active_actors=active_actors,
                assigned_actor_ids=assigned_actor_ids,
                max_missing_frames=max_missing_frames,
            )
            frames.append(SegmentFrame(frame_index=frame_index, instances=instances))
            frame_index += 1
        reader_returncode = reader.wait()
    finally:
        if reader.stdout is not None:
            reader.stdout.close()

    if reader_returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo reader failed: {reader.stderr.read().decode().strip()}")
    if reader.stderr is not None:
        reader.stderr.close()

    manifest = SegmentManifest(
        source_clip=str(input_path),
        backend="person-maskrcnn",
        frame_count=frame_index,
        width=width,
        height=height,
        fps=fps,
        tracks=tracks,
        frames=frames,
        metadata={
            "score_threshold": score_threshold,
            "mask_threshold": mask_threshold,
            "min_area": min_area,
            "iou_threshold": iou_threshold,
            "max_center_distance": max_center_distance,
            "max_missing_frames": max_missing_frames,
        },
    )
    write_segment_manifest(output_dir / "segment_manifest.json", manifest)
    return manifest


def _predict_people(
    *,
    torch,
    model,
    to_tensor,
    frame_rgb: np.ndarray,
    device,
    score_threshold: float,
    mask_threshold: float,
    min_area: int,
) -> list[PersonDetection]:
    tensor = to_tensor(Image.fromarray(frame_rgb)).to(device)
    with torch.no_grad():
        prediction = model([tensor])[0]

    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    boxes = prediction["boxes"].detach().cpu().numpy()
    masks = prediction["masks"].detach().cpu().numpy()[:, 0]

    detections: list[PersonDetection] = []
    for label, score, box, mask_prob in zip(labels, scores, boxes, masks, strict=False):
        if int(label) != 1 or float(score) < score_threshold:
            continue
        mask = (mask_prob >= mask_threshold).astype(np.uint8) * 255
        mask = _largest_component(mask)
        area = int(np.count_nonzero(mask))
        if area < min_area:
            continue
        bbox = mask_bbox(mask)
        if bbox == [0, 0, 0, 0]:
            x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
            bbox = [x1, y1, x2, y2]
        moments = cv2.moments(mask, binaryImage=True)
        if moments["m00"] == 0:
            centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        else:
            centroid = (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))
        detections.append(
            PersonDetection(
                mask=mask,
                bbox=bbox,
                score=float(score),
                centroid=centroid,
                area=area,
            )
        )
    detections.sort(key=lambda detection: detection.bbox[0])
    return detections


def _assign_detections_to_actors(
    *,
    detections: list[PersonDetection],
    active_actors: dict[str, ActiveActor],
    frame_index: int,
    iou_threshold: float,
    max_center_distance: float,
    max_missing_frames: int,
) -> dict[int, str]:
    candidates: list[tuple[float, int, str]] = []
    for detection_index, detection in enumerate(detections):
        for actor_id, actor in active_actors.items():
            if frame_index - actor.last_frame_index > max_missing_frames + 1:
                continue
            iou = _bbox_iou(detection.bbox, actor.bbox)
            distance = float(np.hypot(detection.centroid[0] - actor.centroid[0], detection.centroid[1] - actor.centroid[1]))
            if iou < iou_threshold and distance > max_center_distance:
                continue
            score = iou - (distance / max(max_center_distance, 1.0) * 0.05)
            candidates.append((score, detection_index, actor_id))

    assignments: dict[int, str] = {}
    used_actor_ids: set[str] = set()
    for _, detection_index, actor_id in sorted(candidates, reverse=True):
        if detection_index in assignments or actor_id in used_actor_ids:
            continue
        assignments[detection_index] = actor_id
        used_actor_ids.add(actor_id)
    return assignments


def _age_unassigned_actors(
    *,
    active_actors: dict[str, ActiveActor],
    assigned_actor_ids: set[str],
    max_missing_frames: int,
) -> None:
    stale_actor_ids: list[str] = []
    for actor_id, actor in active_actors.items():
        if actor_id in assigned_actor_ids:
            continue
        actor.missed_frames += 1
        if actor.missed_frames > max_missing_frames:
            stale_actor_ids.append(actor_id)
    for actor_id in stale_actor_ids:
        active_actors.pop(actor_id, None)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return mask
    largest_index = max(range(1, component_count), key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    output = np.zeros(mask.shape, dtype=np.uint8)
    output[labels == largest_index] = 255
    return output


def _bbox_iou(left: list[int], right: list[int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    return intersection / float(left_area + right_area - intersection)


def _select_device(torch, requested_device: str):
    requested_device = requested_device.lower()
    if requested_device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested_device == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if requested_device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported person-maskrcnn device: {requested_device}")


def _load_torchvision_dependencies():
    try:
        import torch
        from torchvision import transforms
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
    except ImportError as exc:
        raise RuntimeError("The person-maskrcnn backend requires torch and torchvision.") from exc
    return torch, transforms, maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
