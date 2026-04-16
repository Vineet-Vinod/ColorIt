from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.config import load_config
from src.pipeline.ffmpeg_utils import extract_single_frame, get_media_duration_seconds
from src.pipeline.inference import colorize_pil_image
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.probes import parse_timecode_to_seconds
from src.pipeline.exemplar_frame import (
    ColorVidNet,
    VGG19Pytorch,
    WarpNet,
    restricted_load_state_dict,
)
from scripts.exemplar_frame_official_cpu import OfficialTransform, build_contact_sheet, run_official_single_frame


TARGET_SIZE = (1280, 720)
TARGET_SARI_POLYGONS = {
    "left_sari": [
        (306, 266),
        (370, 250),
        (451, 277),
        (486, 403),
        (476, 719),
        (330, 719),
        (293, 500),
        (282, 344),
    ],
    "center_sari": [
        (576, 246),
        (650, 232),
        (736, 312),
        (741, 478),
        (730, 719),
        (604, 719),
        (552, 522),
        (542, 335),
    ],
}


@dataclass(frozen=True)
class ReferenceCandidate:
    movie_path: str
    time_seconds: float
    variant: str
    reference_score: float
    red_fraction: float
    dominant_red_blob_fraction: float
    dominant_red_bbox_aspect: float
    dominant_red_bbox_area_fraction: float
    orange_fraction: float
    non_red_saturation: float
    mean_saturation: float
    mean_value: float
    crop_box_xyxy: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class OutputScore:
    total_score: float
    roi_red_signal: float
    roi_red_fraction: float
    roi_orange_fraction: float
    outside_red_signal: float
    outside_red_fraction: float
    outside_orange_fraction: float


@dataclass(frozen=True)
class EvaluatedCandidate:
    movie_path: str
    time_seconds: float
    variant: str
    reference_score: float
    output_score: OutputScore
    reference_image_path: str
    output_image_path: str
    pair_image_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search colored reference frames that push target saris toward vivid red.")
    parser.add_argument("--config", default="configs/quality.yaml")
    parser.add_argument("--target-movie", default="../Movies/Kannada/emme thammanna.mp4")
    parser.add_argument("--target-time", default="00:58:28")
    parser.add_argument("--dataset-dir", default="../Movies/Kannada/finetune_dataset")
    parser.add_argument("--output-root", default="data/experiments/exemplar_vivid_red")
    parser.add_argument("--coarse-step-seconds", type=float, default=300.0)
    parser.add_argument("--coarse-top-k", type=int, default=18)
    parser.add_argument("--refine-top-seeds", type=int, default=6)
    parser.add_argument("--refine-window-seconds", type=float, default=120.0)
    parser.add_argument("--refine-step-seconds", type=float, default=15.0)
    parser.add_argument("--refine-top-k", type=int, default=18)
    parser.add_argument("--final-top-k", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config).expanduser().resolve())

    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    refs_dir = output_root / "references"
    outs_dir = output_root / "outputs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    outs_dir.mkdir(parents=True, exist_ok=True)

    target_movie = Path(args.target_movie).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not target_movie.exists():
        raise FileNotFoundError(f"Target movie not found: {target_movie}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    movies = sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi"}
    )
    if not movies:
        raise ValueError(f"No candidate movies found in {dataset_dir}")

    target_frame = output_root / "target_frame.png"
    extract_single_frame(
        input_path=target_movie,
        output_path=target_frame,
        time_seconds=parse_timecode_to_seconds(args.target_time),
    )
    target_image = Image.open(target_frame).convert("RGB")
    save_target_roi_overlay(target_image, output_root / "target_roi_overlay.png")

    baseline_bundle = load_colorizer_bundle(config)
    baseline_image = colorize_pil_image(
        model_bundle=baseline_bundle,
        input_image=target_image,
        render_factor=int(config.model["render_factor"]),
        postprocess_config=config.raw["postprocess"],
    )
    baseline_path = output_root / "baseline_deoldify.png"
    baseline_image.save(baseline_path)

    from skimage import color
    from skimage.transform import resize

    target_transform = OfficialTransform(image_size=(432, 768), color_module=color, resize_fn=resize)
    reference_transform = OfficialTransform(image_size=(432, 768), color_module=color, resize_fn=resize)

    nonlocal_state, nonlocal_inspection = restricted_load_state_dict((PROJECT_ROOT / "models/exemplar/nonlocal_net_iter_76000.pth").resolve())
    colornet_state, colornet_inspection = restricted_load_state_dict((PROJECT_ROOT / "models/exemplar/colornet_iter_76000.pth").resolve())
    vgg_state, vgg_inspection = restricted_load_state_dict((PROJECT_ROOT / "models/exemplar/vgg19_conv.pth").resolve())

    device = torch.device("cpu")
    nonlocal_net = WarpNet(1).to(device)
    colornet = ColorVidNet(7).to(device)
    vggnet = VGG19Pytorch().to(device)
    nonlocal_net.load_state_dict(nonlocal_state, strict=True)
    colornet.load_state_dict(colornet_state, strict=True)
    vggnet.load_state_dict(vgg_state, strict=True)
    for model in (nonlocal_net, colornet, vggnet):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    temp_ref_path = output_root / "_temp_reference.png"

    seen: set[tuple[str, int]] = set()
    coarse_candidates = score_reference_times(
        movies=movies,
        step_seconds=float(args.coarse_step_seconds),
        temp_ref_path=temp_ref_path,
        seen=seen,
    )
    top_coarse = sorted(coarse_candidates, key=lambda item: item.reference_score, reverse=True)[: int(args.coarse_top_k)]

    evaluated: list[EvaluatedCandidate] = []
    for candidate in top_coarse:
        evaluated.append(
            evaluate_candidate(
                candidate=candidate,
                target_image=target_image,
                target_transform=target_transform,
                reference_transform=reference_transform,
                nonlocal_net=nonlocal_net,
                colornet=colornet,
                vggnet=vggnet,
                refs_dir=refs_dir,
                outs_dir=outs_dir,
                temp_ref_path=temp_ref_path,
            )
        )

    refine_candidates = score_refine_windows(
        seeds=sorted(evaluated, key=lambda item: item.output_score.total_score, reverse=True)[: int(args.refine_top_seeds)],
        window_seconds=float(args.refine_window_seconds),
        step_seconds=float(args.refine_step_seconds),
        temp_ref_path=temp_ref_path,
        seen=seen,
    )
    top_refine = sorted(refine_candidates, key=lambda item: item.reference_score, reverse=True)[: int(args.refine_top_k)]
    for candidate in top_refine:
        evaluated.append(
            evaluate_candidate(
                candidate=candidate,
                target_image=target_image,
                target_transform=target_transform,
                reference_transform=reference_transform,
                nonlocal_net=nonlocal_net,
                colornet=colornet,
                vggnet=vggnet,
                refs_dir=refs_dir,
                outs_dir=outs_dir,
                temp_ref_path=temp_ref_path,
            )
        )

    ranked = sorted(evaluated, key=lambda item: item.output_score.total_score, reverse=True)
    top_final = ranked[: int(args.final_top_k)]
    summary_sheet = output_root / "summary_top_results.png"
    build_summary_sheet(top_final, summary_sheet)

    heuristics = derive_heuristics(ranked)
    manifest = {
        "target_movie": str(target_movie),
        "target_time": args.target_time,
        "dataset_dir": str(dataset_dir),
        "roi_polygons_xy": {name: [list(point) for point in points] for name, points in TARGET_SARI_POLYGONS.items()},
        "checkpoints": {
            "nonlocal": asdict(nonlocal_inspection),
            "colornet": asdict(colornet_inspection),
            "vgg": asdict(vgg_inspection),
        },
        "settings": {
            "coarse_step_seconds": float(args.coarse_step_seconds),
            "coarse_top_k": int(args.coarse_top_k),
            "refine_top_seeds": int(args.refine_top_seeds),
            "refine_window_seconds": float(args.refine_window_seconds),
            "refine_step_seconds": float(args.refine_step_seconds),
            "refine_top_k": int(args.refine_top_k),
            "final_top_k": int(args.final_top_k),
        },
        "top_results": [
            {
                "movie_path": item.movie_path,
                "time_seconds": item.time_seconds,
                "timecode": seconds_to_timecode(item.time_seconds),
                "variant": item.variant,
                "reference_score": item.reference_score,
                "output_score": asdict(item.output_score),
                "reference_image_path": item.reference_image_path,
                "output_image_path": item.output_image_path,
                "pair_image_path": item.pair_image_path,
            }
            for item in top_final
        ],
        "heuristics": heuristics,
        "artifacts": {
            "target_frame": str(target_frame),
            "target_roi_overlay": str(output_root / "target_roi_overlay.png"),
            "baseline": str(baseline_path),
            "summary_top_results": str(summary_sheet),
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Output root: {output_root}")
    print(f"Summary sheet: {summary_sheet}")
    if top_final:
        best = top_final[0]
        print(
            "Best reference: "
            f"{best.movie_path} @ {seconds_to_timecode(best.time_seconds)} [{best.variant}] "
            f"score={best.output_score.total_score:.2f}"
        )
    print(f"Manifest: {manifest_path}")
    return 0


def score_reference_times(
    *,
    movies: list[Path],
    step_seconds: float,
    temp_ref_path: Path,
    seen: set[tuple[str, int]],
) -> list[ReferenceCandidate]:
    candidates: list[ReferenceCandidate] = []
    for movie in movies:
        duration = get_media_duration_seconds(movie)
        seconds = 0.0
        while seconds < duration:
            key = (str(movie), int(round(seconds * 1000)))
            if key not in seen:
                seen.add(key)
                candidates.extend(score_reference_frame_variants(movie, seconds, temp_ref_path))
            seconds += step_seconds
    return candidates


def score_refine_windows(
    *,
    seeds: list[EvaluatedCandidate],
    window_seconds: float,
    step_seconds: float,
    temp_ref_path: Path,
    seen: set[tuple[str, int]],
) -> list[ReferenceCandidate]:
    candidates: list[ReferenceCandidate] = []
    for seed in seeds:
        movie = Path(seed.movie_path)
        duration = get_media_duration_seconds(movie)
        start = max(0.0, seed.time_seconds - window_seconds)
        end = min(duration, seed.time_seconds + window_seconds)
        seconds = start
        while seconds <= end:
            key = (str(movie), int(round(seconds * 1000)))
            if key not in seen:
                seen.add(key)
                candidates.extend(score_reference_frame_variants(movie, seconds, temp_ref_path))
            seconds += step_seconds
    return candidates


def score_reference_frame_variants(movie_path: Path, time_seconds: float, temp_ref_path: Path) -> list[ReferenceCandidate]:
    extract_single_frame(input_path=movie_path, output_path=temp_ref_path, time_seconds=time_seconds)
    image = np.asarray(Image.open(temp_ref_path).convert("RGB"))
    deep_red_mask, orange_mask, red_family_mask = build_reference_masks(image)
    if float(deep_red_mask.mean()) < 0.003:
        return []

    candidates = [
        build_reference_candidate(
            movie_path=movie_path,
            time_seconds=time_seconds,
            image=image,
            deep_red_mask=deep_red_mask,
            orange_mask=orange_mask,
            red_family_mask=red_family_mask,
            variant="full",
            crop_box_xyxy=None,
        )
    ]

    crop_variants = {
        "crop": derive_red_crop_box(deep_red_mask.astype(np.uint8), pad_x_scale=0.70, pad_y_scale=0.40, min_size=180),
        "tight": derive_red_crop_box(deep_red_mask.astype(np.uint8), pad_x_scale=0.18, pad_y_scale=0.18, min_size=120),
    }
    for variant_name, crop_box in crop_variants.items():
        if crop_box is None:
            continue
        x0, y0, x1, y1 = crop_box
        crop = image[y0 : y1 + 1, x0 : x1 + 1]
        crop_deep_red_mask, crop_orange_mask, crop_red_family_mask = build_reference_masks(crop)
        if float(crop_deep_red_mask.mean()) < 0.015:
            continue
        candidate = build_reference_candidate(
            movie_path=movie_path,
            time_seconds=time_seconds,
            image=crop,
            deep_red_mask=crop_deep_red_mask,
            orange_mask=crop_orange_mask,
            red_family_mask=crop_red_family_mask,
            variant=variant_name,
            crop_box_xyxy=crop_box,
        )
        if not (variant_name == "tight" and candidate.dominant_red_bbox_aspect < 0.42):
            candidates.append(candidate)

    return [candidate for candidate in candidates if candidate.reference_score > 0.0]


def build_reference_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1].astype(np.float32)
    v_channel = hsv[:, :, 2].astype(np.float32)
    rgb = image.astype(np.float32)
    r_channel = rgb[:, :, 0]
    g_channel = rgb[:, :, 1]
    b_channel = rgb[:, :, 2]
    red_signal = r_channel - 0.60 * g_channel - 0.60 * b_channel
    deep_red_mask = (
        ((h_channel <= 8) | (h_channel >= 172))
        & (s_channel >= 110)
        & (v_channel >= 40)
        & (red_signal >= 48.0)
        & (r_channel >= g_channel * 1.18)
        & (r_channel >= b_channel * 1.03)
    )
    orange_mask = (
        (h_channel >= 9)
        & (h_channel <= 24)
        & (s_channel >= 95)
        & (v_channel >= 55)
        & (r_channel >= g_channel * 1.02)
        & (g_channel >= b_channel * 1.02)
    )
    red_family_mask = (
        (((h_channel <= 12) | (h_channel >= 168)) & (s_channel >= 90) & (v_channel >= 40))
        & (red_signal >= 30.0)
        & (r_channel >= g_channel * 1.10)
    )
    return deep_red_mask, orange_mask, red_family_mask


def build_reference_candidate(
    *,
    movie_path: Path,
    time_seconds: float,
    image: np.ndarray,
    deep_red_mask: np.ndarray,
    orange_mask: np.ndarray,
    red_family_mask: np.ndarray,
    variant: str,
    crop_box_xyxy: tuple[int, int, int, int] | None,
) -> ReferenceCandidate:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1].astype(np.float32)
    v_channel = hsv[:, :, 2].astype(np.float32)
    red_fraction = float(deep_red_mask.mean())
    orange_fraction = float(orange_mask.mean())
    largest_blob_fraction, bbox_aspect, bbox_area_fraction = largest_component_stats(deep_red_mask.astype(np.uint8))
    mean_saturation = float(s_channel.mean())
    mean_value = float(v_channel.mean())
    non_red_pixels = s_channel[~red_family_mask]
    non_red_saturation = float(non_red_pixels.mean()) if non_red_pixels.size else 0.0
    rgb = image.astype(np.float32)
    red_signal = np.maximum(rgb[:, :, 0] - 0.60 * rgb[:, :, 1] - 0.60 * rgb[:, :, 2], 0.0)
    red_mean = float(red_signal[deep_red_mask].mean()) if deep_red_mask.any() else 0.0
    dominant_blob_ratio = largest_blob_fraction / max(red_fraction, 1e-6)
    overfill_penalty = max(red_fraction - 0.30, 0.0) * 200.0
    wide_shape_penalty = max(0.45 - bbox_aspect, 0.0) * 180.0
    oversize_component_penalty = max(bbox_area_fraction - 0.45, 0.0) * 140.0
    reference_score = (
        red_mean * 0.60
        + 255.0 * largest_blob_fraction * 0.95
        + 180.0 * dominant_blob_ratio * 0.35
        + 10.0 * min(bbox_aspect, 1.2)
        - 210.0 * orange_fraction
        - 0.42 * non_red_saturation
        - overfill_penalty
        - wide_shape_penalty
        - oversize_component_penalty
    )
    if variant == "crop":
        reference_score += 8.0
    elif variant == "tight":
        reference_score += 12.0
    return ReferenceCandidate(
        movie_path=str(movie_path),
        time_seconds=time_seconds,
        variant=variant,
        reference_score=float(reference_score),
        red_fraction=red_fraction,
        dominant_red_blob_fraction=float(largest_blob_fraction),
        dominant_red_bbox_aspect=bbox_aspect,
        dominant_red_bbox_area_fraction=bbox_area_fraction,
        orange_fraction=orange_fraction,
        non_red_saturation=non_red_saturation,
        mean_saturation=mean_saturation,
        mean_value=mean_value,
        crop_box_xyxy=crop_box_xyxy,
    )


def largest_component_fraction(mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return float(mask.mean())
    largest = max(float(stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels))
    return largest / mask.size


def largest_component_stats(mask: np.ndarray) -> tuple[float, float, float]:
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        fraction = float(mask.mean())
        return fraction, 1.0, fraction
    best_label = max(range(1, num_labels), key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
    area = float(stats[best_label, cv2.CC_STAT_AREA])
    width = float(max(1, stats[best_label, cv2.CC_STAT_WIDTH]))
    height = float(max(1, stats[best_label, cv2.CC_STAT_HEIGHT]))
    return area / mask.size, height / width, (width * height) / mask.size


def derive_red_crop_box(
    mask: np.ndarray,
    *,
    pad_x_scale: float,
    pad_y_scale: float,
    min_size: int,
) -> tuple[int, int, int, int] | None:
    if mask.sum() == 0:
        return None
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None
    best_label = max(range(1, num_labels), key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    w = int(stats[best_label, cv2.CC_STAT_WIDTH])
    h = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    image_h, image_w = mask.shape[:2]
    if (w * h) / float(image_h * image_w) < 0.012:
        return None
    pad_x = max(int(round(w * pad_x_scale)), max(20, min_size // 4))
    pad_y = max(int(round(h * pad_y_scale)), max(20, min_size // 4))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(image_w - 1, x + w + pad_x - 1)
    y1 = min(image_h - 1, y + h + pad_y - 1)
    if (x1 - x0 + 1) < min_size or (y1 - y0 + 1) < min_size:
        return None
    return (x0, y0, x1, y1)


def build_target_sari_mask(width: int, height: int) -> np.ndarray:
    mask_image = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(mask_image)
    for points in TARGET_SARI_POLYGONS.values():
        draw.polygon(points, fill=255)
    return np.asarray(mask_image, dtype=np.uint8) > 0


def evaluate_candidate(
    *,
    candidate: ReferenceCandidate,
    target_image: Image.Image,
    target_transform: OfficialTransform,
    reference_transform: OfficialTransform,
    nonlocal_net: WarpNet,
    colornet: ColorVidNet,
    vggnet: VGG19Pytorch,
    refs_dir: Path,
    outs_dir: Path,
    temp_ref_path: Path,
) -> EvaluatedCandidate:
    extract_single_frame(
        input_path=Path(candidate.movie_path),
        output_path=temp_ref_path,
        time_seconds=candidate.time_seconds,
    )
    reference_image = Image.open(temp_ref_path).convert("RGB")
    if candidate.crop_box_xyxy is not None:
        reference_image = reference_image.crop(candidate.crop_box_xyxy)
    official_image, _ = run_official_single_frame(
        target_image=target_image,
        reference_image=reference_image,
        target_transform=target_transform,
        reference_transform=reference_transform,
        vggnet=vggnet,
        nonlocal_net=nonlocal_net,
        colornet=colornet,
    )
    official_upscaled = official_image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    output_score = score_output_image(np.asarray(official_upscaled))

    stem = f"{Path(candidate.movie_path).stem}_{seconds_to_slug(candidate.time_seconds)}_{candidate.variant}"
    reference_image_path = refs_dir / f"{stem}_ref.png"
    output_image_path = outs_dir / f"{stem}_out.png"
    pair_image_path = outs_dir / f"{stem}_pair.png"
    reference_image.save(reference_image_path)
    official_upscaled.save(output_image_path)
    build_pair_sheet(reference_image, official_upscaled, pair_image_path, title_left="Reference", title_right="Output")

    return EvaluatedCandidate(
        movie_path=candidate.movie_path,
        time_seconds=candidate.time_seconds,
        variant=candidate.variant,
        reference_score=candidate.reference_score,
        output_score=output_score,
        reference_image_path=str(reference_image_path),
        output_image_path=str(output_image_path),
        pair_image_path=str(pair_image_path),
    )


def score_output_image(image_rgb: np.ndarray) -> OutputScore:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1].astype(np.float32)
    v_channel = hsv[:, :, 2].astype(np.float32)
    rgb = image_rgb.astype(np.float32)
    r_channel = rgb[:, :, 0]
    g_channel = rgb[:, :, 1]
    b_channel = rgb[:, :, 2]
    red_mask = (
        ((h_channel <= 9) | (h_channel >= 172))
        & (s_channel >= 105)
        & (v_channel >= 45)
        & ((r_channel - 0.58 * g_channel - 0.58 * b_channel) >= 24.0)
        & (r_channel >= g_channel * 1.10)
    )
    orange_mask = (h_channel >= 10) & (h_channel <= 24) & (s_channel >= 95) & (v_channel >= 55)
    red_signal = np.maximum(r_channel - 0.58 * g_channel - 0.58 * b_channel, 0.0)

    roi_mask = build_target_sari_mask(image_rgb.shape[1], image_rgb.shape[0])
    outside_mask = ~roi_mask

    roi_red_signal = float((red_signal[roi_mask] * (0.35 + s_channel[roi_mask] / 255.0)).mean())
    outside_red_signal = float((red_signal[outside_mask] * (0.35 + s_channel[outside_mask] / 255.0)).mean())
    roi_red_fraction = float(red_mask[roi_mask].mean())
    outside_red_fraction = float(red_mask[outside_mask].mean())
    roi_orange_fraction = float(orange_mask[roi_mask].mean())
    outside_orange_fraction = float(orange_mask[outside_mask].mean())
    roi_mean_saturation = float(s_channel[roi_mask].mean())
    outside_mean_saturation = float(s_channel[outside_mask].mean())

    total_score = (
        roi_red_signal
        + 210.0 * roi_red_fraction
        - 110.0 * roi_orange_fraction
        - 0.85 * outside_red_signal
        - 180.0 * outside_red_fraction
        - 85.0 * outside_orange_fraction
        + 0.18 * max(roi_mean_saturation - outside_mean_saturation, 0.0)
    )
    return OutputScore(
        total_score=float(total_score),
        roi_red_signal=roi_red_signal,
        roi_red_fraction=roi_red_fraction,
        roi_orange_fraction=roi_orange_fraction,
        outside_red_signal=outside_red_signal,
        outside_red_fraction=outside_red_fraction,
        outside_orange_fraction=outside_orange_fraction,
    )


def save_target_roi_overlay(target_image: Image.Image, output_path: Path) -> None:
    overlay = target_image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    colors = {"left_sari": (255, 70, 70), "center_sari": (70, 255, 120)}
    for name, points in TARGET_SARI_POLYGONS.items():
        draw.polygon(points, outline=colors[name], width=6)
        label_x = min(point[0] for point in points) + 8
        label_y = min(point[1] for point in points) + 8
        draw.text((label_x, label_y), name, fill=colors[name])
    overlay.save(output_path)


def build_pair_sheet(left: Image.Image, right: Image.Image, output_path: Path, *, title_left: str, title_right: str) -> None:
    build_contact_sheet(
        items=[
            (title_left, left),
            (title_right, right),
        ],
        output_path=output_path,
    )


def build_summary_sheet(top_results: list[EvaluatedCandidate], output_path: Path) -> None:
    items: list[tuple[str, Image.Image]] = []
    for index, item in enumerate(top_results, start=1):
        reference = Image.open(item.reference_image_path).convert("RGB")
        output = Image.open(item.output_image_path).convert("RGB")
        items.append((f"{index}. Ref {Path(item.movie_path).stem} {item.variant}", reference))
        items.append((f"{index}. Out {seconds_to_timecode(item.time_seconds)} {item.variant}", output))
    if not items:
        items = [("No Results", Image.new("RGB", TARGET_SIZE, color=(0, 0, 0)))]
    build_contact_sheet(items=items, output_path=output_path)


def derive_heuristics(ranked: list[EvaluatedCandidate]) -> dict[str, object]:
    if not ranked:
        return {"notes": ["No candidates were evaluated."]}

    top = ranked[: min(5, len(ranked))]
    bottom = ranked[-min(5, len(ranked)) :]
    top_scores = [item.output_score.total_score for item in top]
    bottom_scores = [item.output_score.total_score for item in bottom]
    crop_wins = sum(1 for item in top if item.variant == "crop")
    return {
        "notes": [
            "The strongest outputs are driven by a deep-red garment blob, not by globally warm or orange-heavy frames.",
            "Orange and gold-heavy references often warm the entire scene brown instead of turning the sari regions vivid red.",
            "Tighter crops help when they isolate a red garment or shirt, but broad red fields and title cards overfit into a global red wash.",
            "When crop variants win, the reference works better after removing surrounding palette clutter and letting the garment dominate.",
            "Neutral or darker surroundings in the reference reduce red spill into the roof, walls, and ground of the target frame.",
        ],
        "top_score_range": [float(min(top_scores)), float(max(top_scores))],
        "bottom_score_range": [float(min(bottom_scores)), float(max(bottom_scores))],
        "top_crop_count": crop_wins,
        "top_examples": [
            {
                "movie": Path(item.movie_path).name,
                "timecode": seconds_to_timecode(item.time_seconds),
                "variant": item.variant,
                "score": float(item.output_score.total_score),
            }
            for item in top
        ],
    }


def seconds_to_timecode(seconds: float) -> str:
    total_millis = int(round(seconds * 1000))
    millis = total_millis % 1000
    total_seconds = total_millis // 1000
    secs = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def seconds_to_slug(seconds: float) -> str:
    return seconds_to_timecode(seconds).replace(":", "_").replace(".", "_")


if __name__ == "__main__":
    raise SystemExit(main())
