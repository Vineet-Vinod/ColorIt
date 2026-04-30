from __future__ import annotations

from pathlib import Path
import re
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


DEFAULT_HUMAN_PARSER_MODEL_ID = "mattmdjaga/segformer_b2_clothes"


def run_human_parser_segmentation(
    *,
    input_path: Path,
    output_dir: Path,
    model_id: str,
    device: str,
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
    processor = auto_image_processor.from_pretrained(model_id, trust_remote_code=False)
    model = auto_model.from_pretrained(model_id, trust_remote_code=False)
    model.to(selected_device)
    model.eval()
    id_to_label = {int(key): str(value) for key, value in model.config.id2label.items()}

    reader = open_rawvideo_reader(input_path=input_path)
    if reader.stdout is None or reader.stderr is None:
        raise RuntimeError("ffmpeg rawvideo reader failed to expose stdout/stderr pipes.")

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
            class_map = _predict_class_map(
                torch=torch,
                processor=processor,
                model=model,
                frame_rgb=frame_rgb,
                device=selected_device,
            )

            instances: list[SegmentInstance] = []
            for class_id in sorted(int(value) for value in np.unique(class_map)):
                raw_label = id_to_label.get(class_id, str(class_id))
                label = _normalize_label(raw_label)
                if skip_background and label == "background":
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
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    except ImportError as exc:
        raise RuntimeError(
            "The human-parser backend requires optional dependencies: "
            "torch, transformers, safetensors, and pillow. Install the segmentation extra first."
        ) from exc
    return torch, AutoImageProcessor, AutoModelForSemanticSegmentation


def _normalize_label(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return normalized or "segment"
