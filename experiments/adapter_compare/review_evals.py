from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[2]
CASES = ("havc_automatic", "havc_assisted", "ltx_automatic", "ltx_assisted")


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Path = ROOT / "deep_rem.mp4"
    baseline: Path = ROOT / "tmp/cmnet2_flux_gpu/deep_rem_flux_cmnet2_gpu_180s.mp4"
    queue: Path = ROOT / "tmp/adapter_compare_90_120/unattended"
    start: int = Field(default=2250, ge=0)
    frames: int = Field(default=750, gt=0)
    budget_bytes: int = Field(default=16867622, gt=0)
    wait: bool = False


class Stream(BaseModel):
    model_config = ConfigDict(extra="ignore")
    codec_type: str
    codec_name: str
    width: int | None = None
    height: int | None = None
    nb_read_frames: str | None = None
    duration: str | None = None
    avg_frame_rate: str | None = None


class Probe(BaseModel):
    model_config = ConfigDict(extra="ignore")
    streams: list[Stream]


class QueueResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    case: str
    exit_code: int
    seconds: float


class QueueStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: str
    completed: list[QueueResult] = Field(default_factory=list)


class Measurements(BaseModel):
    frames: int
    mean_chroma: float
    luma_mae: float
    luma_ssim: float
    source_motion_compensated_ab_mae: float
    natural_cut_scores: dict[str, float]
    metrics_are_quality_scores: bool = False


def probe(path: Path) -> Probe:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return Probe.model_validate_json(result.stdout)


def validate(path: Path, count: int, audio: bool, budget: int | None) -> Probe:
    result = probe(path)
    video = [stream for stream in result.streams if stream.codec_type == "video"]
    if len(video) != 1 or video[0].nb_read_frames != str(count):
        raise RuntimeError(f"Wrong decoded frame count: {path}: {result}")
    if video[0].avg_frame_rate != "25/1" or float(video[0].duration or "0") != count / 25:
        raise RuntimeError(f"Wrong video timing: {path}: {video[0]}")
    if audio and not any(stream.codec_type == "audio" for stream in result.streams):
        raise RuntimeError(f"Missing audio: {path}")
    if budget is not None and path.stat().st_size > budget:
        raise RuntimeError(f"Size limit exceeded: {path}")
    return result


def frame_at(path: Path, index: int) -> Image.Image:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"Missing frame {index}: {path}")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()


def extract_clip(source: Path, output: Path, start: int, frames: int) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-ss", str(start / 25),
         "-t", str(frames / 25), "-c:v", "libx264", "-crf", "18", "-preset", "fast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(output)],
        check=True,
    )


def contact_sheet(paths: dict[str, Path], indices: list[int], output: Path) -> None:
    tile_width, tile_height = 320, 180
    label_height = 28
    canvas = Image.new("RGB", (len(paths)*tile_width, len(indices)*(tile_height+label_height)), "#09090b")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    for row, index in enumerate(indices):
        for column, (name, path) in enumerate(paths.items()):
            image = frame_at(path, index).resize((tile_width, tile_height), Image.Resampling.LANCZOS)
            origin = column*tile_width, row*(tile_height+label_height)
            canvas.paste(image, (origin[0], origin[1]+label_height))
            draw.text((origin[0]+6, origin[1]+6), f"{90+index/25:.2f}s · {name}", font=font, fill="#f4f4f5")
    canvas.save(output, quality=94)


def measurements(source: Path, result: Path, count: int, cuts: list[int]) -> Measurements:
    original = cv2.VideoCapture(str(source))
    colored = cv2.VideoCapture(str(result))
    size = (320, 180)
    chroma, luma_errors, luma_scores, motion_errors = [], [], [], []
    cut_scores: dict[str, float] = {}
    last_gray = None
    last_ab = None
    grid_x: NDArray[np.int64]
    grid_y: NDArray[np.int64]
    grid_x, grid_y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    try:
        for index in range(count):
            source_ok, source_bgr = original.read()
            color_ok, color_bgr = colored.read()
            if not source_ok or not color_ok:
                raise RuntimeError(f"Metric input ended at frame {index}: {result}")
            source_small = cv2.resize(source_bgr, size)
            color_small = cv2.resize(color_bgr, size)
            gray = cv2.cvtColor(source_small, cv2.COLOR_BGR2GRAY)
            source_lab = cv2.cvtColor(source_small.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            lab = cv2.cvtColor(color_small.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            ab = lab[:, :, 1:]
            chroma.append(float(np.linalg.norm(ab, axis=-1).mean()))
            luma_errors.append(float(np.abs(lab[:, :, 0]-source_lab[:, :, 0]).mean())/100)
            luma_scores.append(float(structural_similarity(source_lab[:, :, 0], lab[:, :, 0], data_range=100)))
            if last_ab is not None and last_gray is not None:
                if index in cuts:
                    cut_scores[f"{90+index/25:.2f}"] = float(np.abs(ab-last_ab).mean())
                else:
                    flow = cv2.calcOpticalFlowFarneback(gray, last_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    map_x = (grid_x+flow[:, :, 0]).astype(np.float32)
                    map_y = (grid_y+flow[:, :, 1]).astype(np.float32)
                    prior_gray = cv2.remap(last_gray, map_x, map_y, cv2.INTER_LINEAR)
                    prior_ab = cv2.remap(last_ab, map_x, map_y, cv2.INTER_LINEAR)
                    valid = np.abs(gray.astype(np.float32)-prior_gray.astype(np.float32)) < 12
                    valid &= (map_x >= 0) & (map_x < size[0]-1) & (map_y >= 0) & (map_y < size[1]-1)
                    if valid.any():
                        motion_errors.append(float(np.abs(ab-prior_ab)[valid].mean()))
            last_gray, last_ab = gray, ab
    finally:
        original.release()
        colored.release()
    return Measurements(
        frames=count, mean_chroma=float(np.mean(chroma)), luma_mae=float(np.mean(luma_errors)),
        luma_ssim=float(np.mean(luma_scores)), source_motion_compensated_ab_mae=float(np.mean(motion_errors)),
        natural_cut_scores=cut_scores,
    )


def review(options: Options) -> None:
    if options.wait:
        deadline = time.monotonic() + 6 * 60 * 60
        while time.monotonic() < deadline:
            status = QueueStatus.model_validate_json((options.queue / "status.json").read_text())
            if status.status not in ("running", "waiting_for_kernel_validation"):
                break
            time.sleep(15)
        else:
            raise TimeoutError("Evaluation queue did not finish within six hours")
    status = QueueStatus.model_validate_json((options.queue / "status.json").read_text())
    if status.status != "completed" or len(status.completed) != 4 or any(case.exit_code != 0 for case in status.completed):
        raise RuntimeError(f"Four successful cases are required before review: {status}")
    work = options.queue / "review"
    work.mkdir(exist_ok=True)
    source = work / "source.mp4"
    baseline = work / "baseline.mp4"
    extract_clip(options.source, source, options.start, options.frames)
    extract_clip(options.baseline, baseline, options.start, options.frames)
    validate(source, options.frames, True, None)
    validate(baseline, options.frames, True, None)
    paths = {"source": source, "baseline": baseline}
    validation = {}
    for case in CASES:
        folder = options.queue / "cases" / case
        raw = folder / "output.mp4"
        raw_probe = validate(raw, options.frames, case.startswith("havc"), None)
        delivery = raw
        if case.startswith("ltx"):
            delivery = folder / "delivery.mp4"
            subprocess.run(
                [sys.executable, "-m", "experiments.adapter_compare.delivery", "--source", str(options.source),
                 "--predictions", str(raw), "--output", str(delivery), "--start", str(options.start),
                 "--frames", str(options.frames), "--budget-bytes", str(options.budget_bytes),
                 "--restore-luminance" if case.endswith("assisted") else "--no-restore-luminance"],
                check=True, cwd=ROOT,
            )
        checked = validate(delivery, options.frames, True, options.budget_bytes)
        black_references = []
        if case.startswith("havc"):
            for reference in sorted((folder / "references").glob("ref_*.*")):
                if np.array(Image.open(reference).convert("RGB")).max() == 0:
                    black_references.append(reference.name)
        validation[case] = {"path": str(delivery), "bytes": delivery.stat().st_size,
                            "probe": checked.model_dump(), "raw_path": str(raw),
                            "raw_bytes": raw.stat().st_size, "raw_probe": raw_probe.model_dump(),
                            "all_black_reference_files": black_references,
                            "queue_seconds": next(item.seconds for item in status.completed if item.case == case)}
        paths[case] = delivery
    indices = [10, 85, 140, 155, 225, 320, 400, 475, 485, 550, 620, 635, 700, 720, 747]
    contact_sheet(paths, indices, work / "overview.jpg")
    native_paths = {"source": source, "baseline": baseline,
                    "ltx_auto_native": options.queue / "cases/ltx_automatic/output.mp4",
                    "ltx_assist_native": options.queue / "cases/ltx_assisted/output.mp4"}
    contact_sheet(native_paths, indices, work / "ltx_native.jpg")
    cuts = [141, 476, 625, 715, 744]
    contact_sheet(paths, [frame+offset for frame in cuts for offset in (-2, -1, 0, 1)], work / "cuts.jpg")
    metrics = {name: measurements(source, path, options.frames, cuts).model_dump() for name, path in paths.items() if name != "source"}
    (work / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    (work / "validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    inputs = []
    for path in paths.values():
        inputs += ["-threads", "2", "-i", str(path)]
    for name in paths:
        label = work / f"label_{name}.png"
        image = Image.new("RGB", (640, 28), "#09090b")
        ImageDraw.Draw(image).text((8, 4), name, font=ImageFont.load_default(size=18), fill="#f4f4f5")
        image.save(label)
        inputs += ["-loop", "1", "-i", str(label)]
    filters = []
    for index in range(len(paths)):
        filters.append(f"[{index}:v]scale=640:360,pad=640:388:0:28:color=black[p{index}]")
        filters.append(f"[p{index}][{index+len(paths)}:v]overlay=0:0:shortest=1[v{index}]")
    filters.append("".join(f"[v{index}]" for index in range(6))+"xstack=inputs=6:layout=0_0|640_0|1280_0|0_388|640_388|1280_388[v]")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex_threads", "4", "-filter_complex", ";".join(filters),
         "-map", "[v]", "-map", "0:a:0", "-frames:v", str(options.frames), "-c:v", "libx264",
         "-threads", "4", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
         "-movflags", "+faststart", str(work / "comparison.mp4")], check=True,
    )
    validate(work / "comparison.mp4", options.frames, True, 64_000_000)
    print("Review artifacts ready", work, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=Options.model_fields["queue"].default)
    parser.add_argument("--wait", action="store_true")
    review(Options.model_validate(vars(parser.parse_args())))
