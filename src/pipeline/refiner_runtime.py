from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.pipeline.refiner_arch import CostumeRefinerUNet
from src.pipeline.refiner_color import apply_refiner_delta, compose_refiner_input


@dataclass(frozen=True)
class RefinerBundle:
    model: torch.nn.Module
    device: torch.device
    input_channels: int
    ab_delta_scale: float


def load_refiner_bundle(
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> RefinerBundle:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    with torch.serialization.safe_globals([slice]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    meta = checkpoint.get("meta", {})
    model_args = meta.get("model_args", {})
    input_channels = int(model_args.get("input_channels", 4))
    base_channels = int(model_args.get("base_channels", 32))
    ab_delta_scale = float(model_args.get("ab_delta_scale", 24.0))
    model = CostumeRefinerUNet(
        input_channels=input_channels,
        base_channels=base_channels,
        ab_delta_scale=ab_delta_scale,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return RefinerBundle(
        model=model,
        device=device,
        input_channels=input_channels,
        ab_delta_scale=ab_delta_scale,
    )


def refine_base_pil(
    *,
    bundle: RefinerBundle,
    base_image: Image.Image,
    gray_image: Image.Image,
    mask_image: Image.Image | None = None,
) -> Image.Image:
    base_tensor = _pil_to_tensor(base_image).unsqueeze(0).to(bundle.device)
    gray_tensor = _pil_to_tensor(gray_image).unsqueeze(0).to(bundle.device)
    mask_tensor = _mask_to_tensor(mask_image).unsqueeze(0).to(bundle.device) if mask_image is not None else None
    refined = refine_base_tensor(bundle=bundle, base_rgb=base_tensor, gray_rgb=gray_tensor, mask=mask_tensor)
    return _tensor_to_pil(refined[0])


def refine_base_tensor(
    *,
    bundle: RefinerBundle,
    base_rgb: torch.Tensor,
    gray_rgb: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        model_input = compose_refiner_input(base_rgb=base_rgb, gray_rgb=gray_rgb, mask=mask if bundle.input_channels > 4 else None)
        predicted_delta = bundle.model(model_input)
        return apply_refiner_delta(base_rgb=base_rgb, predicted_delta_ab=predicted_delta, mask=mask)


def refine_base_np(
    *,
    bundle: RefinerBundle,
    base_rgb: np.ndarray,
    gray_rgb: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    base_tensor = torch.from_numpy(np.ascontiguousarray(base_rgb)).permute(2, 0, 1).float() / 255.0
    gray_tensor = torch.from_numpy(np.ascontiguousarray(gray_rgb)).permute(2, 0, 1).float() / 255.0
    mask_tensor = None
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[:, :, None]
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).permute(2, 0, 1).float()
    refined = refine_base_tensor(
        bundle=bundle,
        base_rgb=base_tensor.unsqueeze(0).to(bundle.device),
        gray_rgb=gray_tensor.unsqueeze(0).to(bundle.device),
        mask=mask_tensor.unsqueeze(0).to(bundle.device) if mask_tensor is not None else None,
    )[0]
    return (refined.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _mask_to_tensor(image: Image.Image | None) -> torch.Tensor:
    if image is None:
        raise ValueError("Mask image is required for mask tensor conversion.")
    array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array[None, ...])


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array)
