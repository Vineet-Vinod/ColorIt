from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import time

import cv2
import numpy as np
import torch
from PIL import Image

from src.pipeline.model_loader import IMAGENET_MEAN, IMAGENET_STD, ModelBundle


@dataclass(frozen=True)
class InferenceProfile:
    preprocess_seconds: float
    model_seconds: float
    postprocess_seconds: float


def colorize_image_file(
    *,
    model_bundle: ModelBundle,
    input_path: Path,
    output_path: Path,
    render_factor: int,
    postprocess_config: dict | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_image = Image.open(input_path).convert("RGB")
    result = colorize_pil_image(
        model_bundle=model_bundle,
        input_image=input_image,
        render_factor=render_factor,
        postprocess_config=postprocess_config,
    )
    result.save(output_path)


def colorize_rgb_frame(
    *,
    model_bundle: ModelBundle,
    input_rgb: np.ndarray,
    render_factor: int,
    postprocess_config: dict | None = None,
) -> np.ndarray:
    result, _ = colorize_rgb_frame_profiled(
        model_bundle=model_bundle,
        input_rgb=input_rgb,
        render_factor=render_factor,
        postprocess_config=postprocess_config,
    )
    return result


def colorize_pil_image(
    *,
    model_bundle: ModelBundle,
    input_image: Image.Image,
    render_factor: int,
    postprocess_config: dict | None = None,
) -> Image.Image:
    result, _ = colorize_pil_image_profiled(
        model_bundle=model_bundle,
        input_image=input_image,
        render_factor=render_factor,
        postprocess_config=postprocess_config,
    )
    return result


def colorize_pil_image_profiled(
    *,
    model_bundle: ModelBundle,
    input_image: Image.Image,
    render_factor: int,
    postprocess_config: dict | None = None,
) -> tuple[Image.Image, InferenceProfile]:
    preprocess_started = time.perf_counter()
    render_size = render_factor * 16
    result_np, profile = colorize_rgb_frame_profiled(
        model_bundle=model_bundle,
        input_rgb=np.asarray(input_image),
        render_factor=render_factor,
        postprocess_config=postprocess_config,
    )
    return Image.fromarray(result_np), profile


def colorize_rgb_frame_profiled(
    *,
    model_bundle: ModelBundle,
    input_rgb: np.ndarray,
    render_factor: int,
    postprocess_config: dict | None = None,
) -> tuple[np.ndarray, InferenceProfile]:
    preprocess_started = time.perf_counter()
    input_rgb = np.ascontiguousarray(input_rgb)
    input_height, input_width = input_rgb.shape[:2]
    render_size = render_factor * 16
    model_input = cv2.resize(input_rgb, (render_size, render_size), interpolation=cv2.INTER_LINEAR)
    gray_input = cv2.cvtColor(model_input, cv2.COLOR_RGB2GRAY)
    model_input = cv2.cvtColor(gray_input, cv2.COLOR_GRAY2RGB)

    tensor = torch.from_numpy(model_input).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    tensor = tensor.unsqueeze(0).to(model_bundle.device)
    preprocess_seconds = time.perf_counter() - preprocess_started

    model_started = time.perf_counter()
    with torch.no_grad():
        output = model_bundle.model(tensor)[0].cpu()
    model_seconds = time.perf_counter() - model_started

    postprocess_started = time.perf_counter()
    output = (output * IMAGENET_STD) + IMAGENET_MEAN
    output = output.clamp(0.0, 1.0)
    colorized_square = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    colorized = cv2.resize(colorized_square, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    result = _post_process_np(
        raw_color_np=colorized,
        orig_np=input_rgb,
        postprocess_config=postprocess_config,
    )
    postprocess_seconds = time.perf_counter() - postprocess_started
    return result, InferenceProfile(
        preprocess_seconds=preprocess_seconds,
        model_seconds=model_seconds,
        postprocess_seconds=postprocess_seconds,
    )


def _post_process(
    *,
    raw_color: Image.Image,
    orig: Image.Image,
    postprocess_config: dict | None,
) -> Image.Image:
    adjusted = _post_process_np(
        raw_color_np=np.asarray(raw_color),
        orig_np=np.asarray(orig),
        postprocess_config=postprocess_config,
    )
    return Image.fromarray(adjusted)


def _post_process_np(
    *,
    raw_color_np: np.ndarray,
    orig_np: np.ndarray,
    postprocess_config: dict | None,
) -> np.ndarray:
    color_yuv = cv2.cvtColor(raw_color_np, cv2.COLOR_RGB2YUV)
    orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)
    hires = np.copy(orig_yuv)
    hires[:, :, 1:3] = color_yuv[:, :, 1:3]
    final = cv2.cvtColor(hires, cv2.COLOR_YUV2RGB)
    return _apply_color_bias(final, postprocess_config or {})


def _apply_color_bias(image_rgb: np.ndarray, postprocess_config: dict) -> np.ndarray:
    warmth = float(postprocess_config.get("warmth", 0.0))
    shadow_warmth = float(postprocess_config.get("shadow_warmth", 0.0))
    blue_reduction = float(postprocess_config.get("blue_reduction", 0.0))

    if warmth == 0.0 and shadow_warmth == 0.0 and blue_reduction == 0.0:
        return image_rgb

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel, a_channel, b_channel = cv2.split(lab)

    luminance = l_channel / 255.0
    warm_weight = warmth + shadow_warmth * (1.0 - luminance)

    a_channel = a_channel + (8.0 * warm_weight)
    b_channel = b_channel + (16.0 * warm_weight)

    if blue_reduction > 0.0:
        blue_amount = np.clip(128.0 - b_channel, 0.0, None)
        b_channel = b_channel + blue_amount * blue_reduction

    adjusted = cv2.merge(
        [
            np.clip(l_channel, 0.0, 255.0),
            np.clip(a_channel, 0.0, 255.0),
            np.clip(b_channel, 0.0, 255.0),
        ]
    ).astype(np.uint8)
    return cv2.cvtColor(adjusted, cv2.COLOR_LAB2RGB)
