from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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
    result = colorize_rgb_batch(
        model_bundle=model_bundle,
        input_rgbs=[np.asarray(input_image)],
        render_factor=render_factor,
        postprocess_config=postprocess_config,
    )
    Image.fromarray(result[0]).save(output_path)


def colorize_rgb_batch(
    *,
    model_bundle: ModelBundle,
    input_rgbs: list[np.ndarray],
    render_factor: int,
    postprocess_config: dict | None = None,
) -> list[np.ndarray]:
    if not input_rgbs:
        return []

    render_size = render_factor * 16
    processed_inputs: list[np.ndarray] = []
    input_sizes: list[tuple[int, int]] = []

    for input_rgb in input_rgbs:
        input_rgb = np.ascontiguousarray(input_rgb)
        input_height, input_width = input_rgb.shape[:2]
        input_sizes.append((input_height, input_width))

        model_input = cv2.resize(input_rgb, (render_size, render_size), interpolation=cv2.INTER_LINEAR)
        gray_input = cv2.cvtColor(model_input, cv2.COLOR_RGB2GRAY)
        processed_inputs.append(cv2.cvtColor(gray_input, cv2.COLOR_GRAY2RGB))

    tensor = torch.from_numpy(np.stack(processed_inputs)).permute(0, 3, 1, 2).float() / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    tensor = tensor.to(model_bundle.device)

    with torch.no_grad():
        outputs = model_bundle.model(tensor)

    outputs = (outputs * IMAGENET_STD.to(outputs.device)) + IMAGENET_MEAN.to(outputs.device)
    outputs = outputs.clamp(0.0, 1.0)

    results: list[np.ndarray] = []
    for output, input_rgb, output_size in zip(outputs, input_rgbs, input_sizes, strict=True):
        results.append(
            _post_process_tensor(
                raw_color_tensor=output,
                orig_np=input_rgb,
                output_size=output_size,
                postprocess_config=postprocess_config or {},
            )
        )
    return results


def _post_process_tensor(
    *,
    raw_color_tensor: torch.Tensor,
    orig_np: np.ndarray,
    output_size: tuple[int, int],
    postprocess_config: dict,
) -> np.ndarray:
    upsampled = F.interpolate(
        raw_color_tensor.unsqueeze(0),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0]

    orig_tensor = (
        torch.from_numpy(np.array(orig_np, copy=True))
        .to(raw_color_tensor.device)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    final_tensor = _transfer_chroma_tensor(orig_tensor=orig_tensor, color_tensor=upsampled)
    final_tensor = final_tensor.clamp(0.0, 1.0).permute(1, 2, 0)
    final_np = (final_tensor.cpu().numpy() * 255.0).astype(np.uint8)
    return _apply_color_bias(final_np, postprocess_config)


def _transfer_chroma_tensor(
    *,
    orig_tensor: torch.Tensor,
    color_tensor: torch.Tensor,
) -> torch.Tensor:
    y = 0.299 * orig_tensor[0] + 0.587 * orig_tensor[1] + 0.114 * orig_tensor[2]
    u = -0.14713 * color_tensor[0] - 0.28886 * color_tensor[1] + 0.436 * color_tensor[2]
    v = 0.615 * color_tensor[0] - 0.51499 * color_tensor[1] - 0.10001 * color_tensor[2]

    rgb = torch.empty_like(orig_tensor)
    rgb[0] = y + 1.13983 * v
    rgb[1] = y - 0.39465 * u - 0.58060 * v
    rgb[2] = y + 2.03211 * u
    return rgb


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
