from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.pipeline.model_loader import IMAGENET_MEAN, IMAGENET_STD, ModelBundle


@dataclass(frozen=True)
class InferenceProfile:
    preprocess_seconds: float
    model_seconds: float
    postprocess_seconds: float
    preprocess_upload_seconds: float = 0.0
    postprocess_upload_seconds: float = 0.0
    postprocess_graph_seconds: float = 0.0
    postprocess_download_seconds: float = 0.0
    postprocess_cpu_seconds: float = 0.0


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


def colorize_rgb_batch(
    *,
    model_bundle: ModelBundle,
    input_rgbs: list[np.ndarray],
    render_factor: int,
    postprocess_config: dict | None = None,
) -> list[np.ndarray]:
    results, _ = colorize_rgb_batch_profiled(
        model_bundle=model_bundle,
        input_rgbs=input_rgbs,
        render_factor=render_factor,
        postprocess_config=postprocess_config,
    )
    return results


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
    preprocess_upload_seconds = 0.0
    input_rgb = np.ascontiguousarray(input_rgb)
    input_height, input_width = input_rgb.shape[:2]
    render_size = render_factor * 16
    model_input = cv2.resize(input_rgb, (render_size, render_size), interpolation=cv2.INTER_LINEAR)
    gray_input = cv2.cvtColor(model_input, cv2.COLOR_RGB2GRAY)
    model_input = cv2.cvtColor(gray_input, cv2.COLOR_GRAY2RGB)

    tensor = torch.from_numpy(model_input).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    upload_started = time.perf_counter()
    tensor = tensor.unsqueeze(0).to(model_bundle.device)
    _synchronize_for_timing(model_bundle.device)
    preprocess_upload_seconds = time.perf_counter() - upload_started
    preprocess_seconds = time.perf_counter() - preprocess_started

    model_started = time.perf_counter()
    _synchronize_for_timing(model_bundle.device)
    with torch.no_grad():
        output = model_bundle.model(tensor)[0]
    _synchronize_for_timing(model_bundle.device)
    model_seconds = time.perf_counter() - model_started

    postprocess_started = time.perf_counter()
    output = (output * IMAGENET_STD.to(output.device)) + IMAGENET_MEAN.to(output.device)
    output = output.clamp(0.0, 1.0)
    result, postprocess_upload_seconds, postprocess_graph_seconds, postprocess_download_seconds, postprocess_cpu_seconds = _post_process_tensor_profiled(
        raw_color_tensor=output,
        orig_np=input_rgb,
        output_size=(input_height, input_width),
        device=model_bundle.device,
        postprocess_config=postprocess_config or {},
    )
    postprocess_seconds = time.perf_counter() - postprocess_started
    return result, InferenceProfile(
        preprocess_seconds=preprocess_seconds,
        model_seconds=model_seconds,
        postprocess_seconds=postprocess_seconds,
        preprocess_upload_seconds=preprocess_upload_seconds,
        postprocess_upload_seconds=postprocess_upload_seconds,
        postprocess_graph_seconds=postprocess_graph_seconds,
        postprocess_download_seconds=postprocess_download_seconds,
        postprocess_cpu_seconds=postprocess_cpu_seconds,
    )


def colorize_rgb_batch_profiled(
    *,
    model_bundle: ModelBundle,
    input_rgbs: list[np.ndarray],
    render_factor: int,
    postprocess_config: dict | None = None,
) -> tuple[list[np.ndarray], InferenceProfile]:
    if not input_rgbs:
        return [], InferenceProfile(0.0, 0.0, 0.0)

    preprocess_started = time.perf_counter()
    preprocess_upload_seconds = 0.0
    render_size = render_factor * 16
    processed_inputs: list[np.ndarray] = []
    input_sizes: list[tuple[int, int]] = []
    for input_rgb in input_rgbs:
        input_rgb = np.ascontiguousarray(input_rgb)
        input_height, input_width = input_rgb.shape[:2]
        input_sizes.append((input_height, input_width))
        model_input = cv2.resize(input_rgb, (render_size, render_size), interpolation=cv2.INTER_LINEAR)
        gray_input = cv2.cvtColor(model_input, cv2.COLOR_RGB2GRAY)
        model_input = cv2.cvtColor(gray_input, cv2.COLOR_GRAY2RGB)
        processed_inputs.append(model_input)

    tensor = torch.from_numpy(np.stack(processed_inputs)).permute(0, 3, 1, 2).float() / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    upload_started = time.perf_counter()
    tensor = tensor.to(model_bundle.device)
    _synchronize_for_timing(model_bundle.device)
    preprocess_upload_seconds = time.perf_counter() - upload_started
    preprocess_seconds = time.perf_counter() - preprocess_started

    model_started = time.perf_counter()
    _synchronize_for_timing(model_bundle.device)
    with torch.no_grad():
        outputs = model_bundle.model(tensor)
    _synchronize_for_timing(model_bundle.device)
    model_seconds = time.perf_counter() - model_started

    postprocess_started = time.perf_counter()
    postprocess_upload_seconds = 0.0
    postprocess_graph_seconds = 0.0
    postprocess_download_seconds = 0.0
    postprocess_cpu_seconds = 0.0
    outputs = (outputs * IMAGENET_STD.to(outputs.device)) + IMAGENET_MEAN.to(outputs.device)
    outputs = outputs.clamp(0.0, 1.0)
    results: list[np.ndarray] = []
    for output, input_rgb, output_size in zip(outputs, input_rgbs, input_sizes, strict=True):
        result, upload_seconds, graph_seconds, download_seconds, cpu_seconds = _post_process_tensor_profiled(
            raw_color_tensor=output,
            orig_np=input_rgb,
            output_size=output_size,
            device=model_bundle.device,
            postprocess_config=postprocess_config or {},
        )
        results.append(result)
        postprocess_upload_seconds += upload_seconds
        postprocess_graph_seconds += graph_seconds
        postprocess_download_seconds += download_seconds
        postprocess_cpu_seconds += cpu_seconds
    postprocess_seconds = time.perf_counter() - postprocess_started
    return results, InferenceProfile(
        preprocess_seconds=preprocess_seconds,
        model_seconds=model_seconds,
        postprocess_seconds=postprocess_seconds,
        preprocess_upload_seconds=preprocess_upload_seconds,
        postprocess_upload_seconds=postprocess_upload_seconds,
        postprocess_graph_seconds=postprocess_graph_seconds,
        postprocess_download_seconds=postprocess_download_seconds,
        postprocess_cpu_seconds=postprocess_cpu_seconds,
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


def _post_process_tensor(
    *,
    raw_color_tensor: torch.Tensor,
    orig_np: np.ndarray,
    output_size: tuple[int, int],
    postprocess_config: dict,
) -> np.ndarray:
    result, _, _, _, _ = _post_process_tensor_profiled(
        raw_color_tensor=raw_color_tensor,
        orig_np=orig_np,
        output_size=output_size,
        device=raw_color_tensor.device,
        postprocess_config=postprocess_config,
    )
    return result


def _post_process_tensor_profiled(
    *,
    raw_color_tensor: torch.Tensor,
    orig_np: np.ndarray,
    output_size: tuple[int, int],
    device: torch.device,
    postprocess_config: dict,
) -> tuple[np.ndarray, float, float, float, float]:
    graph_started = time.perf_counter()
    upsampled = F.interpolate(
        raw_color_tensor.unsqueeze(0),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0]
    _synchronize_for_timing(device)
    graph_seconds = time.perf_counter() - graph_started

    upload_started = time.perf_counter()
    orig_tensor = (
        torch.from_numpy(np.array(orig_np, copy=True))
        .to(raw_color_tensor.device)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )
    _synchronize_for_timing(device)
    upload_seconds = time.perf_counter() - upload_started

    graph_started = time.perf_counter()
    final_tensor = _transfer_chroma_tensor(orig_tensor=orig_tensor, color_tensor=upsampled)
    final_tensor = final_tensor.clamp(0.0, 1.0).permute(1, 2, 0)
    _synchronize_for_timing(device)
    graph_seconds += time.perf_counter() - graph_started

    download_started = time.perf_counter()
    final_np = (final_tensor.cpu().numpy() * 255.0).astype(np.uint8)
    download_seconds = time.perf_counter() - download_started

    cpu_started = time.perf_counter()
    warmth = float(postprocess_config.get("warmth", 0.0))
    shadow_warmth = float(postprocess_config.get("shadow_warmth", 0.0))
    blue_reduction = float(postprocess_config.get("blue_reduction", 0.0))
    if warmth == 0.0 and shadow_warmth == 0.0 and blue_reduction == 0.0:
        return final_np, upload_seconds, graph_seconds, download_seconds, time.perf_counter() - cpu_started
    return (
        _apply_color_bias(final_np, postprocess_config),
        upload_seconds,
        graph_seconds,
        download_seconds,
        time.perf_counter() - cpu_started,
    )


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


def _synchronize_for_timing(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


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
