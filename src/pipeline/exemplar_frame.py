from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

from src.pipeline.actor_masks import (
    derive_costume_mask,
    derive_skin_mask,
    select_primary_person_mask,
    write_debug_overlay,
    write_mask,
)
from src.pipeline.config import AppConfig
from src.pipeline.ffmpeg_utils import extract_single_frame
from src.pipeline.inference import colorize_pil_image
from src.pipeline.manifest import write_json_manifest
from src.pipeline.model_loader import load_colorizer_bundle, select_device
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths
from src.pipeline.probes import parse_timecode_to_seconds
from src.pipeline.refiner_color import apply_delta_ab_np, rgb_to_lab_np


EXEMPLAR_VGG_FEATURE_KEYS = ["r12", "r22", "r32", "r42", "r52"]
PERSON_CLASS_INDEX = 15
DEFAULT_TARGET_SHAPE = (432, 768)


@dataclass(frozen=True)
class CheckpointInspection:
    path: str
    unsafe_globals: list[str]
    loaded_type: str


@dataclass(frozen=True)
class ExemplarCheckpointPaths:
    nonlocal_path: Path
    colornet_path: Path
    vgg_path: Path


@dataclass(frozen=True)
class PadTransform:
    target_height: int
    target_width: int
    resized_height: int
    resized_width: int
    pad_top: int
    pad_left: int
    original_height: int
    original_width: int


def run_exemplar_frame_experiment(
    *,
    config: AppConfig,
    config_path: Path,
    target_movie_path: Path,
    target_time: str,
    reference_movie_path: Path,
    reference_time: str,
    output_root: Path | None,
    overwrite: bool,
    blend_alpha: float,
    person_threshold: float,
) -> int:
    paths = resolve_project_paths(config)
    ensure_runtime_directories(paths)

    target_movie_path = target_movie_path.expanduser().resolve()
    reference_movie_path = reference_movie_path.expanduser().resolve()
    if not target_movie_path.exists():
        raise FileNotFoundError(f"Target movie not found: {target_movie_path}")
    if not reference_movie_path.exists():
        raise FileNotFoundError(f"Reference movie not found: {reference_movie_path}")

    experiment_id = slugify(
        f"target_{target_movie_path.stem}_{target_time}_ref_{reference_movie_path.stem}_{reference_time}"
    )
    if output_root is None:
        output_root = (paths.root / "data/experiments/exemplar_frame" / experiment_id).resolve()
    else:
        output_root = output_root.expanduser().resolve()
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_root}. Use --overwrite to replace it.")
    output_root.mkdir(parents=True, exist_ok=True)

    frame_dir = output_root / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    target_frame_path = frame_dir / "target_frame.png"
    reference_frame_path = frame_dir / "reference_frame.png"
    baseline_path = output_root / "baseline_deoldify.png"
    exemplar_path = output_root / "exemplar_raw.png"
    replace_path = output_root / "replace_full.png"
    blend_path = output_root / "blend_full.png"
    mask_path = output_root / "mask_costume.png"
    mask_overlay_path = output_root / "mask_overlay.png"
    costume_mask_path = output_root / "costume_mask.png"
    manifest_path = output_root / "manifest.json"
    contact_sheet_path = output_root / "contact_sheet.png"

    extract_single_frame(
        input_path=target_movie_path,
        output_path=target_frame_path,
        time_seconds=parse_timecode_to_seconds(target_time),
    )
    extract_single_frame(
        input_path=reference_movie_path,
        output_path=reference_frame_path,
        time_seconds=parse_timecode_to_seconds(reference_time),
    )

    target_image = Image.open(target_frame_path).convert("RGB")
    reference_image = Image.open(reference_frame_path).convert("RGB")

    deoldify_bundle = load_colorizer_bundle(config)
    baseline_image = colorize_pil_image(
        model_bundle=deoldify_bundle,
        input_image=target_image,
        render_factor=int(config.model["render_factor"]),
        postprocess_config=config.raw["postprocess"],
    )
    baseline_image.save(baseline_path)

    checkpoint_paths = ExemplarCheckpointPaths(
        nonlocal_path=(paths.root / "models/exemplar/nonlocal_net_iter_76000.pth").resolve(),
        colornet_path=(paths.root / "models/exemplar/colornet_iter_76000.pth").resolve(),
        vgg_path=(paths.root / "models/exemplar/vgg19_conv.pth").resolve(),
    )
    runtime = ExemplarRuntime.from_checkpoints(config=config, checkpoints=checkpoint_paths)
    exemplar_image = runtime.colorize_frame(target_image=target_image, reference_image=reference_image)
    exemplar_image.save(exemplar_path)

    baseline_np = np.asarray(baseline_image)
    exemplar_np = np.asarray(exemplar_image)
    replace_np = exemplar_np.copy()
    blend_np = blend_images(base=baseline_np, overlay=exemplar_np, alpha=blend_alpha)

    costume_mask = build_costume_mask(
        image_rgb=baseline_np,
        device=deoldify_bundle.device,
        threshold=person_threshold,
    )
    delta_ab = rgb_to_lab_np(exemplar_np)[:, :, 1:3] - rgb_to_lab_np(baseline_np)[:, :, 1:3]
    masked_np = apply_delta_ab_np(base_rgb=baseline_np, delta_ab=delta_ab, mask=costume_mask.astype(np.float32)[..., None])

    Image.fromarray(replace_np).save(replace_path)
    Image.fromarray(blend_np).save(blend_path)
    Image.fromarray(masked_np).save(mask_path)
    write_mask(costume_mask_path, costume_mask.astype(np.uint8))
    write_debug_overlay(
        mask_overlay_path,
        baseline_np,
        person_mask=np.zeros_like(costume_mask, dtype=np.uint8),
        skin_mask=np.zeros_like(costume_mask, dtype=np.uint8),
        costume_mask=costume_mask.astype(np.uint8),
    )
    build_contact_sheet(
        items=[
            ("Target B/W", target_image),
            ("Reference", reference_image),
            ("Baseline", baseline_image),
            ("Exemplar", exemplar_image),
            ("Full Replace", Image.fromarray(replace_np)),
            ("Full Blend", Image.fromarray(blend_np)),
            ("Mask", Image.fromarray(masked_np)),
            ("Mask Overlay", Image.open(mask_overlay_path).convert("RGB")),
        ],
        output_path=contact_sheet_path,
    )

    manifest: dict[str, Any] = {
        "config_path": str(config_path.resolve()),
        "target_movie_path": str(target_movie_path),
        "target_time": target_time,
        "reference_movie_path": str(reference_movie_path),
        "reference_time": reference_time,
        "blend_alpha": blend_alpha,
        "person_threshold": person_threshold,
        "checkpoints": {
            "nonlocal": asdict(runtime.nonlocal_inspection),
            "colornet": asdict(runtime.colornet_inspection),
            "vgg": asdict(runtime.vgg_inspection),
        },
        "outputs": {
            "target_frame": str(target_frame_path),
            "reference_frame": str(reference_frame_path),
            "baseline": str(baseline_path),
            "exemplar": str(exemplar_path),
            "replace": str(replace_path),
            "blend": str(blend_path),
            "mask": str(mask_path),
            "costume_mask": str(costume_mask_path),
            "mask_overlay": str(mask_overlay_path),
            "contact_sheet": str(contact_sheet_path),
        },
    }
    write_json_manifest(manifest_path, manifest)
    print(f"Experiment output dir: {output_root}")
    print(f"Contact sheet: {contact_sheet_path}")
    print(f"Manifest: {manifest_path}")
    return 0


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_costume_mask(*, image_rgb: np.ndarray, device: torch.device, threshold: float) -> np.ndarray:
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    segmentation_device = device if device.type in {"mps", "cpu"} else torch.device("cpu")
    model = deeplabv3_resnet50(weights=weights).to(segmentation_device).eval()
    with torch.no_grad():
        batch = preprocess(Image.fromarray(image_rgb)).unsqueeze(0).to(segmentation_device)
        probabilities = torch.softmax(model(batch)["out"], dim=1)[0]
    person_prob = probabilities[PERSON_CLASS_INDEX : PERSON_CLASS_INDEX + 1]
    person_prob = F.interpolate(
        person_prob.unsqueeze(0),
        size=image_rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0, 0].detach().cpu().numpy()
    person_mask = select_primary_person_mask(person_prob, threshold)
    skin_mask = derive_skin_mask(image_rgb, person_mask)
    return derive_costume_mask(person_mask, skin_mask)


def blend_images(*, base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    blended = base.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def build_contact_sheet(
    *,
    items: list[tuple[str, Image.Image]],
    output_path: Path,
    tile_size: tuple[int, int] = (360, 220),
    columns: int = 2,
) -> None:
    margin = 20
    label_height = 36
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            columns * tile_size[0] + (columns + 1) * margin,
            rows * (tile_size[1] + label_height) + (rows + 1) * margin,
        ),
        color=(18, 18, 18),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(items):
        row = index // columns
        column = index % columns
        x = margin + column * (tile_size[0] + margin)
        y = margin + row * (tile_size[1] + label_height + margin)
        fitted = ImageOpsHelper.fit(image, tile_size)
        sheet.paste(fitted, (x, y))
        draw.text((x, y + tile_size[1] + 8), label, fill=(235, 235, 235))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


class ImageOpsHelper:
    @staticmethod
    def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
        image = image.convert("RGB")
        src_w, src_h = image.size
        dst_w, dst_h = size
        scale = min(dst_w / src_w, dst_h / src_h)
        resized = image.resize((max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, color=(0, 0, 0))
        offset = ((dst_w - resized.width) // 2, (dst_h - resized.height) // 2)
        canvas.paste(resized, offset)
        return canvas


def restricted_load_state_dict(path: Path) -> tuple[dict[str, torch.Tensor], CheckpointInspection]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    unsafe_globals: list[str] = []
    try:
        unsafe_globals = list(torch.serialization.get_unsafe_globals_in_checkpoint(str(path)))
    except ValueError:
        unsafe_globals = []
    if unsafe_globals:
        raise ValueError(f"Restricted static inspection failed for {path}: unsafe globals {json.dumps(unsafe_globals)}")

    with torch.serialization.safe_globals([]):
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a state-dict-like checkpoint at {path}, found {type(loaded).__name__}.")
    return loaded, CheckpointInspection(
        path=str(path),
        unsafe_globals=unsafe_globals,
        loaded_type=f"{type(loaded).__module__}.{type(loaded).__name__}",
    )


class ExemplarRuntime:
    def __init__(
        self,
        *,
        device: torch.device,
        nonlocal_net: nn.Module,
        colornet: nn.Module,
        vggnet: nn.Module,
        nonlocal_inspection: CheckpointInspection,
        colornet_inspection: CheckpointInspection,
        vgg_inspection: CheckpointInspection,
        target_shape: tuple[int, int] = DEFAULT_TARGET_SHAPE,
    ) -> None:
        self.device = device
        self.nonlocal_net = nonlocal_net
        self.colornet = colornet
        self.vggnet = vggnet
        self.nonlocal_inspection = nonlocal_inspection
        self.colornet_inspection = colornet_inspection
        self.vgg_inspection = vgg_inspection
        self.target_shape = target_shape

    @classmethod
    def from_checkpoints(cls, *, config: AppConfig, checkpoints: ExemplarCheckpointPaths) -> ExemplarRuntime:
        device, _ = select_device(config)
        nonlocal_state, nonlocal_inspection = restricted_load_state_dict(checkpoints.nonlocal_path)
        colornet_state, colornet_inspection = restricted_load_state_dict(checkpoints.colornet_path)
        vgg_state, vgg_inspection = restricted_load_state_dict(checkpoints.vgg_path)

        nonlocal_net = WarpNet(1)
        colornet = ColorVidNet(7)
        vggnet = VGG19Pytorch()
        nonlocal_net.load_state_dict(strip_module_prefix(nonlocal_state), strict=True)
        colornet.load_state_dict(strip_module_prefix(colornet_state), strict=True)
        vggnet.load_state_dict(strip_module_prefix(vgg_state), strict=True)

        nonlocal_net.to(device).eval()
        colornet.to(device).eval()
        vggnet.to(device).eval()
        for model in (nonlocal_net, colornet, vggnet):
            for param in model.parameters():
                param.requires_grad = False

        return cls(
            device=device,
            nonlocal_net=nonlocal_net,
            colornet=colornet,
            vggnet=vggnet,
            nonlocal_inspection=nonlocal_inspection,
            colornet_inspection=colornet_inspection,
            vgg_inspection=vgg_inspection,
        )

    def colorize_frame(self, *, target_image: Image.Image, reference_image: Image.Image) -> Image.Image:
        target_np = np.asarray(target_image.convert("RGB"))
        reference_np = np.asarray(reference_image.convert("RGB"))
        target_prepared, target_transform = resize_with_padding(target_np, self.target_shape)
        reference_prepared, _ = resize_with_padding(reference_np, self.target_shape)

        target_large = normalized_lab_tensor_from_rgb(target_prepared, self.device)
        reference_large = normalized_lab_tensor_from_rgb(reference_prepared, self.device)
        target_small = F.interpolate(target_large, scale_factor=0.5, mode="bilinear", align_corners=False)
        reference_small = F.interpolate(reference_large, scale_factor=0.5, mode="bilinear", align_corners=False)

        target_l = target_small[:, 0:1]
        with torch.no_grad():
            reference_rgb_tensor = tensor_lab2rgb(torch.cat((uncenter_l(reference_small[:, 0:1]), reference_small[:, 1:3]), dim=1))
            features_b = self.vggnet(reference_rgb_tensor, EXEMPLAR_VGG_FEATURE_KEYS, preprocess=True)
            initial_prev = torch.zeros_like(target_small, device=self.device)
            predicted_ab, _, _ = frame_colorization(
                target_small,
                reference_small,
                initial_prev,
                features_b,
                self.vggnet,
                self.nonlocal_net,
                self.colornet,
                feature_noise=0,
                temperature=1e-10,
                joint_training=False,
            )
            predicted_ab = F.interpolate(predicted_ab, scale_factor=2, mode="bilinear", align_corners=False) * 1.25
            output_lab = torch.cat((uncenter_l(target_large[:, 0:1]), predicted_ab), dim=1)
            output_rgb = tensor_lab2rgb(output_lab)[0].detach().cpu().permute(1, 2, 0).numpy()

        output_uint8 = np.clip(output_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        restored = remove_padding_and_resize(output_uint8, target_transform)
        return Image.fromarray(restored)


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def resize_with_padding(image_rgb: np.ndarray, target_shape: tuple[int, int]) -> tuple[np.ndarray, PadTransform]:
    target_height, target_width = target_shape
    original_height, original_width = image_rgb.shape[:2]
    scale = min(target_height / original_height, target_width / original_width)
    resized_height = max(1, int(round(original_height * scale)))
    resized_width = max(1, int(round(original_width * scale)))
    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    pad_top = (target_height - resized_height) // 2
    pad_left = (target_width - resized_width) // 2
    canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    return canvas, PadTransform(
        target_height=target_height,
        target_width=target_width,
        resized_height=resized_height,
        resized_width=resized_width,
        pad_top=pad_top,
        pad_left=pad_left,
        original_height=original_height,
        original_width=original_width,
    )


def remove_padding_and_resize(image_rgb: np.ndarray, transform: PadTransform) -> np.ndarray:
    cropped = image_rgb[
        transform.pad_top : transform.pad_top + transform.resized_height,
        transform.pad_left : transform.pad_left + transform.resized_width,
    ]
    return cv2.resize(
        cropped,
        (transform.original_width, transform.original_height),
        interpolation=cv2.INTER_CUBIC,
    )


def normalized_lab_tensor_from_rgb(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb_float = image_rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)
    lab_tensor = torch.from_numpy(lab).permute(2, 0, 1).unsqueeze(0).to(device)
    lab_tensor[:, 0:1] = lab_tensor[:, 0:1] - 50.0
    return lab_tensor


def uncenter_l(l_channel: torch.Tensor) -> torch.Tensor:
    return l_channel + 50.0


def gray2rgb_batch(l_channel: torch.Tensor) -> torch.Tensor:
    uncentered = uncenter_l(l_channel) / 100.0
    return torch.cat((uncentered, uncentered, uncentered), dim=1)


def feature_normalize(feature_in: torch.Tensor) -> torch.Tensor:
    feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + torch.finfo(feature_in.dtype).eps
    return torch.div(feature_in, feature_in_norm)


def vgg_preprocess(tensor: torch.Tensor) -> torch.Tensor:
    tensor_bgr = torch.cat((tensor[:, 2:3], tensor[:, 1:2], tensor[:, 0:1]), dim=1)
    mean = torch.tensor([0.40760392, 0.45795686, 0.48501961], device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor_bgr - mean) * 255.0


def tensor_lab2rgb(input_tensor: torch.Tensor) -> torch.Tensor:
    input_trans = input_tensor.transpose(1, 2).transpose(2, 3)
    l_channel = input_trans[:, :, :, 0:1]
    a_channel = input_trans[:, :, :, 1:2]
    b_channel = input_trans[:, :, :, 2:]

    y_channel = (l_channel + 16.0) / 116.0
    x_channel = a_channel / 500.0 + y_channel
    z_channel = y_channel - b_channel / 200.0
    z_channel = z_channel.clamp(min=0.0)
    xyz = torch.cat((x_channel, y_channel, z_channel), dim=3)

    mask = xyz > 0.2068966
    xyz = torch.where(mask, xyz.pow(3.0), (xyz - 16.0 / 116.0) / 7.787)
    xyz[:, :, :, 0] = xyz[:, :, :, 0] * 0.95047
    xyz[:, :, :, 2] = xyz[:, :, :, 2] * 1.08883

    rgb_from_xyz = torch.tensor(
        [
            [3.24048134, -0.96925495, 0.05564664],
            [-1.53715152, 1.87599, -0.20404134],
            [-0.49853633, 0.04155593, 1.05731107],
        ],
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    rgb_trans = torch.mm(xyz.reshape(-1, 3), rgb_from_xyz).view(input_tensor.size(0), input_tensor.size(2), input_tensor.size(3), 3)
    rgb = rgb_trans.transpose(2, 3).transpose(1, 2)

    mask = rgb > 0.0031308
    rgb = torch.where(mask, 1.055 * torch.pow(rgb.clamp(min=0.0), 1.0 / 2.4) - 0.055, rgb * 12.92)
    return rgb.clamp(0.0, 1.0)


def warp_color(
    ia_l: torch.Tensor,
    ib_lab: torch.Tensor,
    features_b: list[torch.Tensor],
    vggnet: nn.Module,
    nonlocal_net: nn.Module,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    ia_rgb_from_gray = gray2rgb_batch(ia_l)
    with torch.no_grad():
        a_relu1_1, a_relu2_1, a_relu3_1, a_relu4_1, a_relu5_1 = vggnet(ia_rgb_from_gray, EXEMPLAR_VGG_FEATURE_KEYS, preprocess=True)
        b_relu1_1, b_relu2_1, b_relu3_1, b_relu4_1, b_relu5_1 = features_b

    features_a = [a_relu1_1, a_relu2_1, a_relu3_1, a_relu4_1, a_relu5_1]
    nonlocal_ba_lab, similarity_map = nonlocal_net(
        ib_lab,
        feature_normalize(a_relu2_1),
        feature_normalize(a_relu3_1),
        feature_normalize(a_relu4_1),
        feature_normalize(a_relu5_1),
        feature_normalize(b_relu2_1),
        feature_normalize(b_relu3_1),
        feature_normalize(b_relu4_1),
        feature_normalize(b_relu5_1),
        temperature=temperature,
    )
    return nonlocal_ba_lab, similarity_map, features_a


def frame_colorization(
    ia_lab: torch.Tensor,
    ib_lab: torch.Tensor,
    ia_last_lab: torch.Tensor,
    features_b: list[torch.Tensor],
    vggnet: nn.Module,
    nonlocal_net: nn.Module,
    colornet: nn.Module,
    *,
    joint_training: bool,
    feature_noise: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    del feature_noise
    ia_l = ia_lab[:, 0:1]
    with torch.autograd.set_grad_enabled(joint_training):
        nonlocal_ba_lab, similarity_map, features_a_gray = warp_color(
            ia_l,
            ib_lab,
            features_b,
            vggnet,
            nonlocal_net,
            temperature,
        )
        nonlocal_ba_ab = nonlocal_ba_lab[:, 1:3]
        color_input = torch.cat((ia_l, nonlocal_ba_ab, similarity_map, ia_last_lab), dim=1)
        ia_ab_predict = colornet(color_input)
    return ia_ab_predict, nonlocal_ba_lab, features_a_gray


class ColorVidNet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1_2norm = nn.InstanceNorm2d(64)
        self.conv1_2norm_ss = nn.Conv2d(64, 64, 1, 2, bias=False, groups=64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2_2norm = nn.InstanceNorm2d(128)
        self.conv2_2norm_ss = nn.Conv2d(128, 128, 1, 2, bias=False, groups=128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3norm = nn.InstanceNorm2d(256)
        self.conv3_3norm_ss = nn.Conv2d(256, 256, 1, 2, bias=False, groups=256)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3norm = nn.InstanceNorm2d(512)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3norm = nn.InstanceNorm2d(512)
        self.conv6_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3norm = nn.InstanceNorm2d(512)
        self.conv7_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3norm = nn.InstanceNorm2d(512)
        self.conv8_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(512, 256, 3, 1, 1))
        self.conv3_3_short = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3norm = nn.InstanceNorm2d(256)
        self.conv9_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(256, 128, 3, 1, 1))
        self.conv2_2_short = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2norm = nn.InstanceNorm2d(128)
        self.conv10_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(128, 128, 3, 1, 1))
        self.conv1_2_short = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv10_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv10_ab = nn.Conv2d(128, 2, 1, 1)
        self.relu = nn.ReLU()
        self.final_relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        conv1_2norm = self.conv1_2norm(conv1_2)
        conv2_1 = self.relu(self.conv2_1(self.conv1_2norm_ss(conv1_2norm)))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        conv2_2norm = self.conv2_2norm(conv2_2)
        conv3_1 = self.relu(self.conv3_1(self.conv2_2norm_ss(conv2_2norm)))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        conv3_3norm = self.conv3_3norm(conv3_3)
        conv4_1 = self.relu(self.conv4_1(self.conv3_3norm_ss(conv3_3norm)))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))
        conv4_3norm = self.conv4_3norm(conv4_3)
        conv5_1 = self.relu(self.conv5_1(conv4_3norm))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))
        conv5_3norm = self.conv5_3norm(conv5_3)
        conv6_1 = self.relu(self.conv6_1(conv5_3norm))
        conv6_2 = self.relu(self.conv6_2(conv6_1))
        conv6_3 = self.relu(self.conv6_3(conv6_2))
        conv6_3norm = self.conv6_3norm(conv6_3)
        conv7_1 = self.relu(self.conv7_1(conv6_3norm))
        conv7_2 = self.relu(self.conv7_2(conv7_1))
        conv7_3 = self.relu(self.conv7_3(conv7_2))
        conv7_3norm = self.conv7_3norm(conv7_3)
        conv8_1 = self.conv8_1(conv7_3norm)
        conv8_1_comb = self.relu(conv8_1 + self.conv3_3_short(conv3_3norm))
        conv8_2 = self.relu(self.conv8_2(conv8_1_comb))
        conv8_3 = self.relu(self.conv8_3(conv8_2))
        conv8_3norm = self.conv8_3norm(conv8_3)
        conv9_1 = self.conv9_1(conv8_3norm)
        conv9_1_comb = self.relu(conv9_1 + self.conv2_2_short(conv2_2norm))
        conv9_2 = self.relu(self.conv9_2(conv9_1_comb))
        conv9_2norm = self.conv9_2norm(conv9_2)
        conv10_1 = self.conv10_1(conv9_2norm)
        conv10_1_comb = self.relu(conv10_1 + self.conv1_2_short(conv1_2norm))
        conv10_2 = self.final_relu(self.conv10_2(conv10_1_comb))
        return torch.tanh(self.conv10_ab(conv10_2)) * 128.0


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.padding1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=0, stride=1)
        self.bn1 = nn.InstanceNorm2d(channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=0, stride=1)
        self.bn2 = nn.InstanceNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.prelu(self.bn1(self.conv1(self.padding1(x))))
        out = self.bn2(self.conv2(self.padding2(out)))
        return self.prelu(out + residual)


class WarpNet(nn.Module):
    def __init__(self, batch_size: int):
        del batch_size
        super().__init__()
        self.feature_channel = 64
        self.in_channels = self.feature_channel * 4
        self.inter_channels = 256
        self.layer2_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, self.feature_channel, kernel_size=3, padding=0, stride=2),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
        )
        self.layer3_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
        )
        self.layer4_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
        )
        self.layer5_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
        )
        self.layer = nn.Sequential(
            ResidualBlock(self.feature_channel * 4),
            ResidualBlock(self.feature_channel * 4),
            ResidualBlock(self.feature_channel * 4),
        )
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.upsampling = nn.Upsample(scale_factor=4)

    def forward(
        self,
        b_lab_map: torch.Tensor,
        a_relu2_1: torch.Tensor,
        a_relu3_1: torch.Tensor,
        a_relu4_1: torch.Tensor,
        a_relu5_1: torch.Tensor,
        b_relu2_1: torch.Tensor,
        b_relu3_1: torch.Tensor,
        b_relu4_1: torch.Tensor,
        b_relu5_1: torch.Tensor,
        *,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channel, image_height, image_width = b_lab_map.shape
        feature_height = image_height // 4
        feature_width = image_width // 4

        a_feature2_1 = self.layer2_1(a_relu2_1)
        b_feature2_1 = self.layer2_1(b_relu2_1)
        a_feature3_1 = self.layer3_1(a_relu3_1)
        b_feature3_1 = self.layer3_1(b_relu3_1)
        a_feature4_1 = self.layer4_1(a_relu4_1)
        b_feature4_1 = self.layer4_1(b_relu4_1)
        a_feature5_1 = self.layer5_1(a_relu5_1)
        b_feature5_1 = self.layer5_1(b_relu5_1)
        if a_feature5_1.shape[2:] != a_feature2_1.shape[2:]:
            a_feature5_1 = F.pad(a_feature5_1, (0, 0, 1, 1), "replicate")
            b_feature5_1 = F.pad(b_feature5_1, (0, 0, 1, 1), "replicate")

        a_features = self.layer(torch.cat((a_feature2_1, a_feature3_1, a_feature4_1, a_feature5_1), dim=1))
        b_features = self.layer(torch.cat((b_feature2_1, b_feature3_1, b_feature4_1, b_feature5_1), dim=1))

        theta = self.theta(a_features).view(batch_size, self.inter_channels, -1)
        theta = theta - theta.mean(dim=-1, keepdim=True)
        theta = theta / (torch.norm(theta, 2, 1, keepdim=True) + torch.finfo(theta.dtype).eps)
        phi = self.phi(b_features).view(batch_size, self.inter_channels, -1)
        phi = phi - phi.mean(dim=-1, keepdim=True)
        phi = phi / (torch.norm(phi, 2, 1, keepdim=True) + torch.finfo(phi.dtype).eps)
        similarity = torch.matmul(theta.permute(0, 2, 1), phi)

        similarity_map = similarity.unsqueeze(1).amax(dim=-1, keepdim=True).view(batch_size, 1, feature_height, feature_width)
        weights = F.softmax(similarity / temperature, dim=-1)
        b_lab = F.avg_pool2d(b_lab_map, 4).view(batch_size, channel, -1).permute(0, 2, 1)
        propagated = torch.matmul(weights, b_lab).permute(0, 2, 1).contiguous().view(batch_size, channel, feature_height, feature_width)
        return self.upsampling(propagated), self.upsampling(similarity_map)


class VGG19Pytorch(nn.Module):
    def __init__(self, pool: str = "max"):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        pool_cls = nn.MaxPool2d if pool == "max" else nn.AvgPool2d
        self.pool1 = pool_cls(kernel_size=2, stride=2)
        self.pool2 = pool_cls(kernel_size=2, stride=2)
        self.pool3 = pool_cls(kernel_size=2, stride=2)
        self.pool4 = pool_cls(kernel_size=2, stride=2)
        self.pool5 = pool_cls(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, out_keys: list[str], preprocess: bool = True) -> list[torch.Tensor]:
        if preprocess:
            x = vgg_preprocess(x)
        out: dict[str, torch.Tensor] = {}
        out["r11"] = F.relu(self.conv1_1(x))
        out["r12"] = F.relu(self.conv1_2(out["r11"]))
        out["p1"] = self.pool1(out["r12"])
        out["r21"] = F.relu(self.conv2_1(out["p1"]))
        out["r22"] = F.relu(self.conv2_2(out["r21"]))
        out["p2"] = self.pool2(out["r22"])
        out["r31"] = F.relu(self.conv3_1(out["p2"]))
        out["r32"] = F.relu(self.conv3_2(out["r31"]))
        out["r33"] = F.relu(self.conv3_3(out["r32"]))
        out["r34"] = F.relu(self.conv3_4(out["r33"]))
        out["p3"] = self.pool3(out["r34"])
        out["r41"] = F.relu(self.conv4_1(out["p3"]))
        out["r42"] = F.relu(self.conv4_2(out["r41"]))
        out["r43"] = F.relu(self.conv4_3(out["r42"]))
        out["r44"] = F.relu(self.conv4_4(out["r43"]))
        out["p4"] = self.pool4(out["r44"])
        out["r51"] = F.relu(self.conv5_1(out["p4"]))
        out["r52"] = F.relu(self.conv5_2(out["r51"]))
        out["r53"] = F.relu(self.conv5_3(out["r52"]))
        out["r54"] = F.relu(self.conv5_4(out["r53"]))
        out["p5"] = self.pool5(out["r54"])
        return [out[key] for key in out_keys]
