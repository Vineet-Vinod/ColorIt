from __future__ import annotations

from pathlib import Path
import re
import threading
from typing import Any

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


DEFAULT_HUMAN_PARSER_MODEL_ID = "models/segformer_b2_clothes"
_MODEL_LOAD_LOCK = threading.Lock()
_DEPENDENCIES: tuple[Any, Any, Any] | None = None
_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, dict[int, str]]] = {}


def run_human_parser_segmentation(
    *,
    input_path: Path,
    output_dir: Path,
    model_id: str,
    device: str,
    frame_stride: int,
    include_labels: list[str] | None,
    overwrite: bool,
    skip_background: bool = True,
) -> SegmentManifest:
    torch, auto_image_processor, auto_model = _load_transformers_dependencies()

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
    model_source = _resolve_model_source(model_id)
    processor, model, id_to_label = _load_parser_model(
        auto_image_processor=auto_image_processor,
        auto_model=auto_model,
        model_source=model_source,
        device=selected_device,
    )
    include_label_set = {
        _normalize_label(label)
        for label in (include_labels or [])
        if label.strip()
    }

    reader = open_rawvideo_reader(input_path=input_path)
    if reader.stdout is None or reader.stderr is None:
        raise RuntimeError("ffmpeg rawvideo reader failed to expose stdout/stderr pipes.")

    tracks: dict[str, SegmentTrack] = {}
    frames: list[SegmentFrame] = []
    frame_index = 0
    frame_stride = max(1, int(frame_stride))
    class_map: np.ndarray | None = None
    cached_instances: list[SegmentInstance] = []
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
            if class_map is None or frame_index % frame_stride == 0:
                class_map = _predict_class_map(
                    torch=torch,
                    processor=processor,
                    model=model,
                    frame_rgb=frame_rgb,
                    device=selected_device,
                )
                cached_instances = _instances_from_class_map(
                    class_map=class_map,
                    frame_index=frame_index,
                    id_to_label=id_to_label,
                    include_label_set=include_label_set,
                    output_dir=output_dir,
                    tracks=tracks,
                    skip_background=skip_background,
                )
            frames.append(SegmentFrame(frame_index=frame_index, instances=cached_instances))
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
        backend="human-parser",
        frame_count=frame_index,
        width=width,
        height=height,
        fps=fps,
        tracks=tracks,
        frames=frames,
    )
    write_segment_manifest(output_dir / "segment_manifest.json", manifest)
    return manifest


def _instances_from_class_map(
    *,
    class_map: np.ndarray,
    frame_index: int,
    id_to_label: dict[int, str],
    include_label_set: set[str],
    output_dir: Path,
    tracks: dict[str, SegmentTrack],
    skip_background: bool,
) -> list[SegmentInstance]:
    instances: list[SegmentInstance] = []
    for class_id in sorted(int(value) for value in np.unique(class_map)):
        raw_label = id_to_label.get(class_id, str(class_id))
        label = _normalize_label(raw_label)
        if skip_background and label == "background":
            continue
        if include_label_set and label not in include_label_set:
            continue

        mask = (class_map == class_id).astype(np.uint8) * 255
        if not np.any(mask):
            continue

        track_id = f"semantic_{class_id:02d}_{label}"
        tracks.setdefault(
            track_id,
            SegmentTrack(
                track_id=track_id,
                label=label,
                kind="human_part",
            ),
        )

        mask_path = output_dir / "masks" / f"frame_{frame_index:06d}_{track_id}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(mask_path)
        instances.append(
            SegmentInstance(
                track_id=track_id,
                label=label,
                kind="human_part",
                mask_path=str(mask_path.relative_to(output_dir)),
                bbox=mask_bbox(mask),
                confidence=1.0,
            )
        )
    return instances


def _predict_class_map(
    *,
    torch,
    processor,
    model,
    frame_rgb: np.ndarray,
    device,
) -> np.ndarray:
    image = Image.fromarray(frame_rgb)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=frame_rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
    )
    return logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)


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
    raise ValueError(f"Unsupported human parser device: {requested_device}")


def _load_transformers_dependencies():
    global _DEPENDENCIES
    try:
        with _MODEL_LOAD_LOCK:
            if _DEPENDENCIES is None:
                import torch
                from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

                _DEPENDENCIES = (torch, AutoImageProcessor, AutoModelForSemanticSegmentation)
            return _DEPENDENCIES
    except ImportError as exc:
        raise RuntimeError(
            "The human-parser backend requires optional dependencies: "
            "torch, transformers, safetensors, and pillow. Install the segmentation extra first."
        ) from exc


def _load_parser_model(*, auto_image_processor, auto_model, model_source: str, device) -> tuple[Any, Any, dict[int, str]]:
    cache_key = (model_source, str(device))
    with _MODEL_LOAD_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        processor = auto_image_processor.from_pretrained(model_source, trust_remote_code=False)
        model = auto_model.from_pretrained(model_source, trust_remote_code=False)
        model.to(device)
        model.eval()
        id_to_label = {int(key): str(value) for key, value in model.config.id2label.items()}
        cached = (processor, model, id_to_label)
        _MODEL_CACHE[cache_key] = cached
        return cached


def _resolve_model_source(model_id: str) -> str:
    candidate = Path(model_id).expanduser()
    if _is_complete_local_model(candidate):
        return str(candidate.resolve())

    cwd_candidate = (Path.cwd() / model_id).resolve()
    if _is_complete_local_model(cwd_candidate):
        return str(cwd_candidate)

    home_project_candidate = (Path.home() / "ColorIt" / model_id).resolve()
    if _is_complete_local_model(home_project_candidate):
        return str(home_project_candidate)

    return model_id


def _is_complete_local_model(path: Path) -> bool:
    return (
        path.exists()
        and (path / "config.json").exists()
        and (path / "preprocessor_config.json").exists()
        and (path / "model.safetensors").exists()
    )


def _normalize_label(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return normalized or "segment"
