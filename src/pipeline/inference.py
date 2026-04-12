from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.pipeline.model_loader import IMAGENET_MEAN, IMAGENET_STD, ModelBundle


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


def colorize_pil_image(
    *,
    model_bundle: ModelBundle,
    input_image: Image.Image,
    render_factor: int,
    postprocess_config: dict | None = None,
) -> Image.Image:
    render_size = render_factor * 16
    model_input = input_image.resize((render_size, render_size), resample=Image.BILINEAR)
    model_input = model_input.convert("LA").convert("RGB")

    tensor = torch.from_numpy(np.array(model_input)).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    tensor = tensor.unsqueeze(0).to(model_bundle.device)

    with torch.no_grad():
        output = model_bundle.model(tensor)[0].cpu()

    output = (output * IMAGENET_STD) + IMAGENET_MEAN
    output = output.clamp(0.0, 1.0)
    colorized_square = Image.fromarray((output.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    colorized = colorized_square.resize(input_image.size, resample=Image.BILINEAR)
    return _post_process(
        raw_color=colorized,
        orig=input_image,
        postprocess_config=postprocess_config,
    )


def _post_process(
    *,
    raw_color: Image.Image,
    orig: Image.Image,
    postprocess_config: dict | None,
) -> Image.Image:
    color_np = np.asarray(raw_color)
    orig_np = np.asarray(orig)
    color_yuv = cv2.cvtColor(color_np, cv2.COLOR_RGB2YUV)
    orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)
    hires = np.copy(orig_yuv)
    hires[:, :, 1:3] = color_yuv[:, :, 1:3]
    final = cv2.cvtColor(hires, cv2.COLOR_YUV2RGB)
    adjusted = _apply_color_bias(final, postprocess_config or {})
    return Image.fromarray(adjusted)


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
