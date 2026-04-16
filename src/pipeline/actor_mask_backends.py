from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    LRASPP_MobileNet_V3_Large_Weights,
    deeplabv3_resnet50,
    lraspp_mobilenet_v3_large,
)


PERSON_CATEGORY_ID = 1
PERSON_CLASS_INDEX = 15


@dataclass(frozen=True)
class ActorMaskBackendSpec:
    name: str
    family: str
    person_prob_threshold: float | None = None
    detection_score_threshold: float | None = None
    mask_prob_threshold: float | None = None
    erode_scale: float = 0.0
    dilate_scale: float = 0.0
    min_component_fraction: float = 0.001


@dataclass
class ActorMaskBackend:
    spec: ActorMaskBackendSpec
    model: torch.nn.Module
    preprocess: object
    device: torch.device


@dataclass(frozen=True)
class ActorMaskResult:
    mask: np.ndarray
    metadata: dict[str, object]


BACKEND_SPECS = {
    "deeplabv3_union": ActorMaskBackendSpec(
        name="deeplabv3_union",
        family="semantic",
        person_prob_threshold=0.32,
        erode_scale=0.007,
        dilate_scale=0.0,
        min_component_fraction=0.003,
    ),
    "lraspp_union": ActorMaskBackendSpec(
        name="lraspp_union",
        family="semantic",
        person_prob_threshold=0.44,
        erode_scale=0.010,
        dilate_scale=0.0,
        min_component_fraction=0.003,
    ),
    "maskrcnn_v1_conservative": ActorMaskBackendSpec(
        name="maskrcnn_v1_conservative",
        family="detection",
        detection_score_threshold=0.82,
        mask_prob_threshold=0.72,
        erode_scale=0.005,
        dilate_scale=0.0,
        min_component_fraction=0.002,
    ),
    "maskrcnn_v2_balanced": ActorMaskBackendSpec(
        name="maskrcnn_v2_balanced",
        family="detection",
        detection_score_threshold=0.72,
        mask_prob_threshold=0.66,
        erode_scale=0.004,
        dilate_scale=0.0,
        min_component_fraction=0.002,
    ),
    "maskrcnn_v2_conservative": ActorMaskBackendSpec(
        name="maskrcnn_v2_conservative",
        family="detection",
        detection_score_threshold=0.84,
        mask_prob_threshold=0.74,
        erode_scale=0.005,
        dilate_scale=0.0,
        min_component_fraction=0.002,
    ),
    "maskrcnn_v2_tight": ActorMaskBackendSpec(
        name="maskrcnn_v2_tight",
        family="detection",
        detection_score_threshold=0.90,
        mask_prob_threshold=0.82,
        erode_scale=0.008,
        dilate_scale=0.0,
        min_component_fraction=0.002,
    ),
}


def list_actor_mask_backends() -> list[str]:
    return sorted(BACKEND_SPECS.keys())


def resolve_device(device_name: str) -> torch.device:
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_actor_mask_backend(name: str, device: torch.device) -> ActorMaskBackend:
    if name not in BACKEND_SPECS:
        raise KeyError(f"Unknown actor mask backend: {name}")
    spec = BACKEND_SPECS[name]

    if name == "deeplabv3_union":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights).to(device).eval()
        preprocess = weights.transforms()
    elif name == "lraspp_union":
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        model = lraspp_mobilenet_v3_large(weights=weights).to(device).eval()
        preprocess = weights.transforms()
    elif name == "maskrcnn_v1_conservative":
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()
        preprocess = weights.transforms()
    elif name in {"maskrcnn_v2_balanced", "maskrcnn_v2_conservative", "maskrcnn_v2_tight"}:
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
        preprocess = weights.transforms()
    else:
        raise AssertionError(f"Unhandled actor mask backend: {name}")

    return ActorMaskBackend(spec=spec, model=model, preprocess=preprocess, device=device)


def predict_actor_mask(backend: ActorMaskBackend, image: Image.Image) -> ActorMaskResult:
    image = image.convert("RGB")
    image_np = np.asarray(image)
    start_time = time.perf_counter()

    with torch.no_grad():
        if backend.spec.family == "semantic":
            mask, metadata = _predict_semantic_union_mask(backend, image_np, image)
        elif backend.spec.family == "detection":
            mask, metadata = _predict_detection_union_mask(backend, image_np, image)
        else:
            raise AssertionError(f"Unhandled backend family: {backend.spec.family}")

    inference_seconds = time.perf_counter() - start_time
    metadata["backend"] = backend.spec.name
    metadata["family"] = backend.spec.family
    metadata["inference_seconds"] = inference_seconds
    metadata["mask_fraction"] = float(mask.mean())
    metadata["mask_nonzero"] = int(mask.sum())
    return ActorMaskResult(mask=mask.astype(np.uint8), metadata=metadata)


def _predict_semantic_union_mask(
    backend: ActorMaskBackend,
    image_np: np.ndarray,
    image: Image.Image,
) -> tuple[np.ndarray, dict[str, object]]:
    assert backend.spec.person_prob_threshold is not None
    x = backend.preprocess(image).unsqueeze(0).to(backend.device)
    output = backend.model(x)["out"]
    probabilities = torch.softmax(output, dim=1)[0, PERSON_CLASS_INDEX : PERSON_CLASS_INDEX + 1]
    probabilities = F.interpolate(
        probabilities.unsqueeze(0),
        size=image_np.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    person_prob = probabilities.detach().cpu().numpy()
    raw_mask = (person_prob >= backend.spec.person_prob_threshold).astype(np.uint8)
    mask = finalize_actor_mask(
        raw_mask,
        erode_scale=backend.spec.erode_scale,
        dilate_scale=backend.spec.dilate_scale,
        min_component_fraction=backend.spec.min_component_fraction,
    )
    return mask, {
        "person_prob_threshold": backend.spec.person_prob_threshold,
        "person_prob_max": float(person_prob.max()),
        "person_prob_mean": float(person_prob.mean()),
    }


def _predict_detection_union_mask(
    backend: ActorMaskBackend,
    image_np: np.ndarray,
    image: Image.Image,
) -> tuple[np.ndarray, dict[str, object]]:
    assert backend.spec.detection_score_threshold is not None
    assert backend.spec.mask_prob_threshold is not None
    x = backend.preprocess(image).to(backend.device)
    output = backend.model([x])[0]
    scores = output["scores"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    masks = output["masks"].detach().cpu().numpy()

    accepted_count = 0
    raw_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    accepted_scores: list[float] = []
    for score, label, instance_mask in zip(scores, labels, masks, strict=True):
        if int(label) != PERSON_CATEGORY_ID:
            continue
        if float(score) < backend.spec.detection_score_threshold:
            continue
        binary = (instance_mask[0] >= backend.spec.mask_prob_threshold).astype(np.uint8)
        if binary.sum() == 0:
            continue
        raw_mask = np.maximum(raw_mask, binary)
        accepted_count += 1
        accepted_scores.append(float(score))

    mask = finalize_actor_mask(
        raw_mask,
        erode_scale=backend.spec.erode_scale,
        dilate_scale=backend.spec.dilate_scale,
        min_component_fraction=backend.spec.min_component_fraction,
    )
    return mask, {
        "detection_score_threshold": backend.spec.detection_score_threshold,
        "mask_prob_threshold": backend.spec.mask_prob_threshold,
        "accepted_instance_count": accepted_count,
        "accepted_scores": accepted_scores,
        "top_scores": [float(score) for score in scores[:5]],
    }


def finalize_actor_mask(
    raw_mask: np.ndarray,
    *,
    erode_scale: float,
    dilate_scale: float,
    min_component_fraction: float,
) -> np.ndarray:
    mask = raw_mask.astype(np.uint8).copy()
    if mask.sum() == 0:
        return mask

    close_kernel = make_kernel(mask.shape[0], mask.shape[1], scale=0.006)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = filter_small_components(mask, min_component_fraction=min_component_fraction)

    if erode_scale > 0.0:
        erode_kernel = make_kernel(mask.shape[0], mask.shape[1], scale=erode_scale)
        mask = cv2.erode(mask, erode_kernel, iterations=1)
    if dilate_scale > 0.0:
        dilate_kernel = make_kernel(mask.shape[0], mask.shape[1], scale=dilate_scale)
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    mask = filter_small_components(mask, min_component_fraction=min_component_fraction)
    return mask.astype(np.uint8)


def filter_small_components(mask: np.ndarray, *, min_component_fraction: float) -> np.ndarray:
    if mask.sum() == 0:
        return mask.astype(np.uint8)
    min_area = max(1, int(round(mask.size * min_component_fraction)))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(np.uint8)
    filtered = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            filtered[labels == label] = 1
    return filtered


def make_kernel(height: int, width: int, scale: float) -> np.ndarray:
    size = max(3, int(round(min(height, width) * scale)))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(path)


def write_actor_overlay(path: Path, image_rgb: np.ndarray, mask: np.ndarray, *, label: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = image_rgb.astype(np.float32).copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.45 + np.array([255, 90, 70], dtype=np.float32) * 0.55
    output = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
    if label:
        draw = ImageDraw.Draw(output)
        draw.text((12, 12), label, fill=(255, 255, 255))
    output.save(path)
