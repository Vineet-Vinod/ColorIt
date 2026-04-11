from __future__ import annotations

from dataclasses import dataclass

import torch

from src.pipeline.config import AppConfig
from src.pipeline.model_arch import DeoldifyVideoModel
from src.pipeline.paths import resolve_project_paths


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    device: torch.device
    backend: str


def select_device(config: AppConfig) -> tuple[torch.device, str]:
    preferred = str(config.raw["runtime"]["backend_preference"]).lower()
    fallback = str(config.raw["runtime"]["fallback_backend"]).lower()

    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    if fallback == "cpu":
        return torch.device("cpu"), "cpu"

    raise RuntimeError(
        f"Unable to satisfy backend preference '{preferred}' with fallback '{fallback}'."
    )


def load_colorizer_bundle(config: AppConfig) -> ModelBundle:
    paths = resolve_project_paths(config)
    device, backend = select_device(config)

    model = DeoldifyVideoModel()
    checkpoint = _load_checkpoint(paths.weights_path)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    return ModelBundle(model=model, device=device, backend=backend)


def _load_checkpoint(weights_path):
    with torch.serialization.safe_globals([slice]):
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint does not contain a 'model' state dict.")
    return checkpoint
