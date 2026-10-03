from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]
TESTED = ROOT / "tmp/cmnet2_flux_gpu"


class Reference(BaseModel):
    model_config = ConfigDict(extra="ignore")
    frame: int = Field(ge=0)
    path: Path


class Manifest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    references: list[Reference] = Field(min_length=1)


class Scenes(BaseModel):
    model_config = ConfigDict(extra="ignore")
    scene_boundaries: list[int]


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Path
    references: Path
    manifest: Path
    output: Path
    mode: Literal["assisted", "automatic"]
    scenes: Path = ROOT / "experiments/adapter_compare/scene_manifest.json"
    start: int = Field(default=2250, ge=0)
    frames: int = Field(default=750, gt=0)
    prepare_resize: bool = False
    budget_bytes: int = Field(default=16867622, gt=0)

    @model_validator(mode="after")
    def check_inputs(self) -> Options:
        for path in (self.source, self.manifest):
            if not path.is_file():
                raise ValueError(f"Missing input: {path}")
        if not self.references.is_dir():
            raise ValueError(f"Missing reference directory: {self.references}")
        if not self.output.resolve().is_relative_to(ROOT / "tmp"):
            raise ValueError("Experiment output must be under ColorIt/tmp")
        return self


class Processor(Protocol):
    last_ti_key: object | None
    last_ti_value: object | None

    def clear_memory(self) -> None: ...
    def set_all_labels(self, labels: list[int]) -> None: ...


class Renderer(Protocol):
    network: object
    processor: Processor
    frame_count: int
    total_colored_frames: int
    first_mask_loaded: bool

    def preload_reference(self, image: Image.Image, frame_idx: int) -> None: ...
    def slide_permanent_memory(self, n_frames: int) -> None: ...
    def set_ref_frame(self, image: Image.Image | None) -> None: ...
    def colorize_frame(
        self, ti: int, frame_i: Image.Image, lab_mode: str
    ) -> Image.Image: ...


def reference_bank(options: Options) -> list[Reference]:
    content = json.loads(options.manifest.read_text())
    manifest = Manifest.model_validate(
        {"references": content} if isinstance(content, list) else content
    )
    bank = sorted(manifest.references, key=lambda reference: reference.frame)
    if len({reference.frame for reference in bank}) != len(bank):
        raise ValueError("Reference frame indices must be unique")
    for reference in bank:
        if not reference.path.is_absolute():
            reference.path = options.references / reference.path
        if not reference.path.is_file():
            raise ValueError(f"Missing reference: {reference.path}")
    if options.mode == "automatic" and len(bank) < 2:
        raise ValueError("Author automatic memory requires at least two references")
    return bank


def prepare_resize(options: Options, bank: list[Reference]) -> None:
    import vapoursynth as vs

    vs.core.num_threads = 4
    vs.core.max_cache_size = 256
    work = options.output.parent / "spline36"
    work.mkdir(parents=True, exist_ok=True)

    def resize(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        blank = vs.core.std.BlankClip(
            width=image.shape[1], height=image.shape[0], format=vs.RGB24, length=1
        )

        def pixels(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            result = f.copy()
            for plane in range(3):
                np.copyto(np.asarray(result[plane]), image[:, :, plane])
            return result

        clip = blank.std.ModifyFrame(clips=blank, selector=pixels)
        frame = clip.resize.Spline36(width=512, height=288).get_frame(0)
        return np.stack([np.asarray(frame[plane]) for plane in range(3)], axis=-1)

    capture = cv2.VideoCapture(str(options.source))
    capture.set(cv2.CAP_PROP_POS_FRAMES, options.start)
    output = np.lib.format.open_memmap(
        work / "source.npy", mode="w+", dtype=np.uint8,
        shape=(options.frames, 288, 512, 3),
    )
    try:
        for index in range(options.frames):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Missing source frame {options.start + index}")
            output[index] = resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output.flush()
    finally:
        capture.release()
    for reference in bank:
        with Image.open(reference.path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        Image.fromarray(resize(pixels)).save(work / f"ref_{reference.frame:06d}.png")


def propagate(options: Options) -> None:
    bank = reference_bank(options)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    if options.prepare_resize:
        prepare_resize(options, bank)
        return
    started = perf_counter()
    source_small: NDArray[np.uint8] | None = None
    if options.mode == "automatic":
        command = [str(ROOT / "tmp/havc_cpu/.venv/bin/python"), str(Path(__file__)),
                   *sys.argv[1:], "--prepare-resize"]
        subprocess.run(command, check=True)
        source_small = np.load(options.output.parent / "spline36/source.npy", mmap_mode="r")
    os.environ["CMNET_DEVICE"] = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for name, folder in [("HF_HOME", "huggingface"), ("TORCH_HOME", "torch"),
                         ("XDG_CACHE_HOME", "xdg")]:
        os.environ[name] = str(TESTED / "cache" / folder)
    sys.path[:0] = [str(TESTED), str(TESTED / "mps_source"),
                    str(ROOT / "tmp/cmnet2_dinov3_cpu")]
    import torch
    from colormnet.colormnet_render import ColorMNetRender
    from sparse_readout import patch_softmax

    if not torch.backends.mps.is_available():
        raise RuntimeError("CMNET2 requires the MPS GPU for this experiment")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.mps.set_per_process_memory_fraction(0.07)
    cv2.setNumThreads(1)
    patch_softmax(TESTED / "mps_source")
    model = cast(Renderer, ColorMNetRender(
        image_size=-1, vid_length=max(options.frames, 100), encode_mode=2,
        max_memory_frames=1000, reset_on_ref_update=False,
        project_dir=str(TESTED / "mps_source"), backbone="dinov3",
        enable_proximity_bias=False,
    ))
    network = cast(torch.nn.Module, model.network)
    devices = {parameter.device.type for parameter in network.parameters()}
    if devices != {"mps"}:
        raise RuntimeError(f"Unexpected CMNET2 parameter devices: {devices}")
    if options.mode == "assisted":
        importlib.import_module("mps_readout").install()
    dimensions = (683, 384) if options.mode == "assisted" else (512, 288)

    def image(reference: Reference) -> Image.Image:
        path = reference.path
        if options.mode == "automatic":
            path = options.output.parent / "spline36" / f"ref_{reference.frame:06d}.png"
        with Image.open(path) as original:
            rgb = original.convert("RGB")
        return rgb.resize(dimensions, Image.Resampling.BILINEAR)

    def preload(references: list[Reference]) -> None:
        for reference in references:
            model.preload_reference(image(reference), frame_idx=reference.frame-options.start)
            torch.mps.synchronize()
            torch.mps.empty_cache()

    boundaries = [options.start, options.start + options.frames]
    if options.mode == "assisted":
        scenes = Scenes.model_validate_json(options.scenes.read_text())
        boundaries = sorted(set(boundaries + [frame for frame in scenes.scene_boundaries
                                             if boundaries[0] < frame < boundaries[-1]]))
    window_size = min(20, len(bank) // 2 * 2)
    next_reference = window_size
    half_index = max(0, round(window_size * 0.5) - 1)
    if options.mode == "automatic":
        preload(bank[:window_size])
    source = cv2.VideoCapture(str(options.source))
    source.set(cv2.CAP_PROP_POS_FRAMES, options.start)
    predictions_path = options.output.parent / "predictions.npy"
    predictions = np.lib.format.open_memmap(
        predictions_path, mode="w+", dtype=np.uint8,
        shape=(options.frames, dimensions[1], dimensions[0], 3),
    )
    rendered = 0
    with torch.inference_mode():
        for begin, end in pairwise(boundaries):
            if options.mode == "assisted":
                model.processor.clear_memory()
                model.processor.last_ti_key = None
                model.processor.last_ti_value = None
                model.processor.set_all_labels([1, 2])
                model.frame_count = 0
                model.total_colored_frames = 0
                model.first_mask_loaded = False
                preload(bank)
            first = bank[0] if options.mode == "automatic" else min(
                bank, key=lambda reference: abs(reference.frame - begin))
            for frame in range(begin, end):
                index = frame - options.start
                if options.mode == "automatic" and frame > bank[half_index].frame and next_reference < len(bank):
                    model.slide_permanent_memory(2)
                    added = bank[next_reference:next_reference + 2]
                    preload(added)
                    next_reference += len(added)
                    half_index = min(half_index + len(added), len(bank) - 1)
                ok, bgr = source.read()
                if not ok:
                    raise RuntimeError(f"Source ended at frame {frame}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                input_image = Image.fromarray(source_small[index]) if source_small is not None else (
                    Image.fromarray(rgb).resize(dimensions, Image.Resampling.BILINEAR))
                model.set_ref_frame(image(first) if frame == begin else None)
                prediction = model.colorize_frame(frame - begin, input_image, lab_mode="cpu")
                predictions[index] = np.asarray(prediction)
                rendered += 1
                torch.mps.synchronize()
                if index % 5 == 0:
                    torch.mps.empty_cache()
                if index % 25 == 0 or rendered == options.frames:
                    progress = {"mode": options.mode, "frames": rendered,
                                "total_frames": options.frames, "seconds": perf_counter()-started,
                                "gpu_allocated_bytes": torch.mps.current_allocated_memory(),
                                "gpu_driver_bytes": torch.mps.driver_allocated_memory()}
                    options.output.with_suffix(".progress.json").write_text(json.dumps(progress) + "\n")
                    print(progress, flush=True)
    source.release()
    predictions.flush()
    propagation_seconds = perf_counter() - started
    subprocess.run([sys.executable, "-m", "experiments.adapter_compare.delivery",
                    "--source", str(options.source), "--predictions", str(predictions_path),
                    "--output", str(options.output), "--start", str(options.start),
                    "--frames", str(options.frames), "--budget-bytes", str(options.budget_bytes)], check=True)
    result = {"mode": options.mode, "frames": rendered, "start": options.start,
              "inference_size": dimensions, "references": len(bank),
              "shot_resets": len(boundaries)-1 if options.mode == "assisted" else 0,
              "permanent_window": len(bank) if options.mode == "assisted" else window_size,
              "propagation_seconds": propagation_seconds, "total_seconds": perf_counter()-started,
              "output": str(options.output), "output_bytes": options.output.stat().st_size,
              "adaptations": ["PyTorch MPS with tested HF DINOv3 encoder and correlation fallback",
                              "Source luminance restored in Lab instead of native VapourSynth YUV",
                              "Direct in-process execution instead of CMNET2 RPC"],
              "render_vivid": False, "retry_threshold": 0.0,
              "resize": "bilinear" if options.mode == "assisted" else "VapourSynth Spline36"}
    options.output.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")
    print(result, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("source", "references", "manifest", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--mode", choices=("assisted", "automatic"), required=True)
    parser.add_argument("--scenes", type=Path, default=Options.model_fields["scenes"].default)
    parser.add_argument("--start", type=int, default=2250)
    parser.add_argument("--frames", type=int, default=750)
    parser.add_argument("--budget-bytes", type=int, default=16867622)
    parser.add_argument("--prepare-resize", action="store_true")
    propagate(Options.model_validate(vars(parser.parse_args())))


if __name__ == "__main__":
    main()
