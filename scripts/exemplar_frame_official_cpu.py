from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.transforms import CenterCrop

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.exemplar_frame import (
    ColorVidNet,
    VGG19Pytorch,
    WarpNet,
    frame_colorization,
    restricted_load_state_dict,
    tensor_lab2rgb,
    uncenter_l,
)
from src.pipeline.ffmpeg_utils import extract_single_frame
from src.pipeline.inference import colorize_pil_image
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.probes import parse_timecode_to_seconds


IMAGE_SIZE = (216 * 2, 384 * 2)
FEATURE_KEYS = ["r12", "r22", "r32", "r42", "r52"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official Deep Exemplar single-frame path on CPU.")
    parser.add_argument("--config", default="configs/quality.yaml")
    parser.add_argument("--movie", required=True, help="Target grayscale movie path.")
    parser.add_argument("--time", required=True, help="Target timestamp in HH:MM:SS[.mmm].")
    parser.add_argument("--reference-movie", required=True, help="Reference color movie path.")
    parser.add_argument("--reference-time", required=True, help="Reference timestamp in HH:MM:SS[.mmm].")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output directory. Defaults under data/experiments/exemplar_frame_official/.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from skimage import color
        from skimage.transform import resize
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This script needs scikit-image. Run it with `uv run --with scikit-image --with opencv-contrib-python-headless ...`."
        ) from exc

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    target_movie_path = Path(args.movie).expanduser().resolve()
    reference_movie_path = Path(args.reference_movie).expanduser().resolve()
    if not target_movie_path.exists():
        raise FileNotFoundError(f"Target movie not found: {target_movie_path}")
    if not reference_movie_path.exists():
        raise FileNotFoundError(f"Reference movie not found: {reference_movie_path}")

    experiment_id = slugify(
        f"official_target_{target_movie_path.stem}_{args.time}_ref_{reference_movie_path.stem}_{args.reference_time}"
    )
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (PROJECT_ROOT / "data/experiments/exemplar_frame_official" / experiment_id).resolve()
    )
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output_root}. Use --overwrite to replace it.")
    output_root.mkdir(parents=True, exist_ok=True)
    frames_dir = output_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    target_frame_path = frames_dir / "target_frame.png"
    reference_frame_path = frames_dir / "reference_frame.png"
    baseline_path = output_root / "baseline_deoldify.png"
    official_path = output_root / "official_exact_cpu.png"
    manifest_path = output_root / "manifest.json"
    contact_sheet_path = output_root / "contact_sheet.png"

    extract_single_frame(
        input_path=target_movie_path,
        output_path=target_frame_path,
        time_seconds=parse_timecode_to_seconds(args.time),
    )
    extract_single_frame(
        input_path=reference_movie_path,
        output_path=reference_frame_path,
        time_seconds=parse_timecode_to_seconds(args.reference_time),
    )

    target_image = Image.open(target_frame_path).convert("RGB")
    reference_image = Image.open(reference_frame_path).convert("RGB")

    target_transform = OfficialTransform(image_size=IMAGE_SIZE, color_module=color, resize_fn=resize)
    reference_transform = OfficialTransform(image_size=IMAGE_SIZE, color_module=color, resize_fn=resize)

    nonlocal_state, nonlocal_inspection = restricted_load_state_dict(
        (PROJECT_ROOT / "models/exemplar/nonlocal_net_iter_76000.pth").resolve()
    )
    colornet_state, colornet_inspection = restricted_load_state_dict(
        (PROJECT_ROOT / "models/exemplar/colornet_iter_76000.pth").resolve()
    )
    vgg_state, vgg_inspection = restricted_load_state_dict(
        (PROJECT_ROOT / "models/exemplar/vgg19_conv.pth").resolve()
    )

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

    official_image, used_wls = run_official_single_frame(
        target_image=target_image,
        reference_image=reference_image,
        target_transform=target_transform,
        reference_transform=reference_transform,
        vggnet=vggnet,
        nonlocal_net=nonlocal_net,
        colornet=colornet,
    )
    official_image.save(official_path)

    baseline_bundle = load_colorizer_bundle(config)
    baseline_image = colorize_pil_image(
        model_bundle=baseline_bundle,
        input_image=target_image,
        render_factor=int(config.model["render_factor"]),
        postprocess_config=config.raw["postprocess"],
    )
    baseline_image.save(baseline_path)

    build_contact_sheet(
        items=[
            ("Target B/W", target_image),
            ("Reference", reference_image),
            ("Baseline", baseline_image),
            ("Official CPU", official_image),
        ],
        output_path=contact_sheet_path,
    )

    manifest = {
        "target_movie_path": str(target_movie_path),
        "target_time": args.time,
        "reference_movie_path": str(reference_movie_path),
        "reference_time": args.reference_time,
        "device": str(device),
        "image_size": list(IMAGE_SIZE),
        "used_wls_filter": used_wls,
        "checkpoints": {
            "nonlocal": {
                "path": nonlocal_inspection.path,
                "unsafe_globals": nonlocal_inspection.unsafe_globals,
                "loaded_type": nonlocal_inspection.loaded_type,
            },
            "colornet": {
                "path": colornet_inspection.path,
                "unsafe_globals": colornet_inspection.unsafe_globals,
                "loaded_type": colornet_inspection.loaded_type,
            },
            "vgg": {
                "path": vgg_inspection.path,
                "unsafe_globals": vgg_inspection.unsafe_globals,
                "loaded_type": vgg_inspection.loaded_type,
            },
        },
        "outputs": {
            "target_frame": str(target_frame_path),
            "reference_frame": str(reference_frame_path),
            "baseline": str(baseline_path),
            "official_exact_cpu": str(official_path),
            "contact_sheet": str(contact_sheet_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Experiment output dir: {output_root}")
    print(f"Contact sheet: {contact_sheet_path}")
    print(f"Manifest: {manifest_path}")
    return 0


class OfficialTransform:
    def __init__(self, *, image_size: tuple[int, int], color_module, resize_fn) -> None:
        self.image_size = image_size
        self.center_crop = CenterCrop(image_size)
        self.color = color_module
        self.resize = resize_fn

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = center_pad(image, self.image_size, self.resize)
        image = self.center_crop(image)
        lab = self.color.rgb2lab(np.asarray(image))
        tensor = to_mytensor(lab)
        tensor[0:1, :, :] = normalize_channel(tensor[0:1, :, :], 50, 1)
        tensor[1:3, :, :] = normalize_channel(tensor[1:3, :, :], (0, 0), (1, 1))
        return tensor


def run_official_single_frame(
    *,
    target_image: Image.Image,
    reference_image: Image.Image,
    target_transform: OfficialTransform,
    reference_transform: OfficialTransform,
    vggnet: VGG19Pytorch,
    nonlocal_net: WarpNet,
    colornet: ColorVidNet,
) -> tuple[Image.Image, bool]:
    ia_lab_large = target_transform(target_image).unsqueeze(0)
    ib_lab_large = reference_transform(reference_image).unsqueeze(0)
    ia_lab = torch.nn.functional.interpolate(ia_lab_large, scale_factor=0.5, mode="bilinear")
    ib_lab = torch.nn.functional.interpolate(ib_lab_large, scale_factor=0.5, mode="bilinear")

    ia_l = ia_lab[:, 0:1, :, :]
    with torch.no_grad():
        i_reference_lab = ib_lab
        i_reference_l = i_reference_lab[:, 0:1, :, :]
        i_reference_ab = i_reference_lab[:, 1:3, :, :]
        i_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(i_reference_l), i_reference_ab), dim=1))
        features_b = vggnet(i_reference_rgb, FEATURE_KEYS, preprocess=True)
        i_last_lab_predict = torch.zeros_like(ia_lab)
        i_current_ab_predict, _, _ = frame_colorization(
            ia_lab,
            i_reference_lab,
            i_last_lab_predict,
            features_b,
            vggnet,
            nonlocal_net,
            colornet,
            joint_training=False,
            feature_noise=0,
            temperature=1e-10,
        )

    curr_bs_l = ia_lab_large[:, 0:1, :, :]
    curr_predict = torch.nn.functional.interpolate(i_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear") * 1.25
    used_wls = False
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createFastGlobalSmootherFilter"):
        used_wls = True
        guide_image = uncenter_l(curr_bs_l) * 255 / 100
        wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
            guide_image[0, 0, :, :].cpu().numpy().astype(np.uint8),
            500,
            4,
        )
        curr_predict_a = wls_filter.filter(curr_predict[0, 0, :, :].cpu().numpy())
        curr_predict_b = wls_filter.filter(curr_predict[0, 1, :, :].cpu().numpy())
        curr_predict_filter = torch.cat(
            (
                torch.from_numpy(curr_predict_a).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(curr_predict_b).unsqueeze(0).unsqueeze(0),
            ),
            dim=1,
        )
        output_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict_filter[:32, ...])
    else:
        output_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict[:32, ...])
    return Image.fromarray(output_rgb), used_wls


def center_pad(image: Image.Image, image_size: tuple[int, int], resize_fn) -> Image.Image:
    image_np = np.array(image)
    height_old = np.size(image_np, 0)
    width_old = np.size(image_np, 1)
    old_size = [height_old, width_old]
    height, width = image_size
    ratio = height / width
    if height_old / width_old == ratio:
        if height_old == height:
            return Image.fromarray(image_np.astype(np.uint8))
        new_size = [int(x * height / height_old) for x in old_size]
        resized = resize_fn(image_np, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
        return Image.fromarray(resized.astype(np.uint8))

    canvas = np.zeros((height, width, np.size(image_np, 2)))
    if height_old / width_old > ratio:
        new_size = [int(x * width / width_old) for x in old_size]
        resized = resize_fn(image_np, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
        height_resize = np.size(resized, 0)
        start_height = (height_resize - height) // 2
        canvas[:, :, :] = resized[start_height : (start_height + height), :, :]
    else:
        new_size = [int(x * height / height_old) for x in old_size]
        resized = resize_fn(image_np, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
        width_resize = np.size(resized, 1)
        start_width = (width_resize - width) // 2
        canvas[:, :, :] = resized[:, start_width : (start_width + width), :]
    return Image.fromarray(canvas.astype(np.uint8))


def to_mytensor(pic) -> torch.Tensor:
    pic_arr = np.array(pic)
    if pic_arr.ndim == 2:
        pic_arr = pic_arr[..., np.newaxis]
    return torch.from_numpy(pic_arr.transpose((2, 0, 1))).float()


def normalize_channel(tensor: torch.Tensor, mean, std) -> torch.Tensor:
    if tensor.size(0) == 1:
        tensor.sub_(mean).div_(std)
        return tensor
    for current, current_mean, current_std in zip(tensor, mean, std, strict=True):
        current.sub_(current_mean).div_(current_std)
    return tensor


def batch_lab2rgb_transpose_mc(img_l_mc: torch.Tensor, img_ab_mc: torch.Tensor) -> np.ndarray:
    from skimage import color

    img_l = img_l_mc + 50.0
    img_ab = img_ab_mc
    pred_lab = torch.cat((img_l, img_ab), dim=1)
    grid_lab = pred_lab[0].permute(1, 2, 0).numpy().astype("float64")
    return (np.clip(color.lab2rgb(grid_lab), 0, 1) * 255).astype("uint8")


def build_contact_sheet(*, items: list[tuple[str, Image.Image]], output_path: Path) -> None:
    tile_size = (360, 220)
    margin = 20
    label_height = 36
    columns = 2
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
        fitted = fit_image(image, tile_size)
        sheet.paste(fitted, (x, y))
        draw.text((x, y + tile_size[1] + 8), label, fill=(235, 235, 235))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    src_w, src_h = image.size
    dst_w, dst_h = size
    scale = min(dst_w / src_w, dst_h / src_h)
    resized = image.resize(
        (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, color=(0, 0, 0))
    offset = ((dst_w - resized.width) // 2, (dst_h - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_config(path: Path):
    from src.pipeline.config import load_config as _load_config

    return _load_config(path)


if __name__ == "__main__":
    raise SystemExit(main())
