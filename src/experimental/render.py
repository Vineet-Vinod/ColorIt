from __future__ import annotations

import gc
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Protocol, cast

import cv2
import numpy as np
import torch
from PIL import Image

from src.experimental.experimental_dataclasses import (
    ClipInfo,
    ModelAssets,
    Reference,
    Shot,
)
from src.experimental.media import restore_luminance

PROMPT = (
    "Colorize this black-and-white archival film frame. Preserve its people, faces, "
    "costumes, objects, composition, lighting, and period photographic realism. "
    "Use believable material colors, vivid costume colors, and natural skin tones. "
    "Keep every color inside the exact source object boundary. Costume colors must "
    "not tint skin, hair, or the surroundings. Do not crop, redraw, add, remove, or restyle anything."
)


class ImageResult(Protocol):
    image: Image.Image


class FluxEditor(Protocol):
    def generate_image(
        self,
        *,
        seed: int,
        prompt: str,
        num_inference_steps: int,
        width: int,
        height: int,
        guidance: float,
        image_paths: list[Path],
        scheduler: str,
        use_kv_cache: bool,
    ) -> ImageResult: ...


class FluxFactory(Protocol):
    def __call__(self, *, model_path: str, quantize: int) -> FluxEditor: ...


class Processor(Protocol):
    def clear_memory(self) -> None: ...


class CMNETRenderer(Protocol):
    processor: Processor
    frame_count: int
    total_colored_frames: int
    first_mask_loaded: bool

    def preload_reference(self, ref_img: Image.Image, frame_idx: int) -> None: ...
    def set_ref_frame(self, frame_ref: Image.Image | None) -> None: ...
    def colorize_frame(
        self, ti: int, frame_i: Image.Image, lab_mode: str
    ) -> Image.Image: ...


class CMNETFactory(Protocol):
    def __call__(
        self,
        *,
        image_size: int,
        vid_length: int,
        encode_mode: int,
        max_memory_frames: int,
        reset_on_ref_update: bool,
        project_dir: str,
        backbone: str,
        enable_proximity_bias: bool,
    ) -> CMNETRenderer: ...


def generate_references(
    assets: ModelAssets, references: list[Reference], info: ClipInfo
) -> None:
    module = importlib.import_module(
        "mflux.models.flux2.variants.edit.flux2_klein_edit"
    )
    factory = cast(FluxFactory, module.Flux2KleinEdit)
    model = factory(model_path=str(assets.flux), quantize=8)
    scale = 1024 / max(info.width, info.height)
    width = max(64, round(info.width * scale / 16) * 16)
    height = max(64, round(info.height * scale / 16) * 16)
    for index, reference in enumerate(references):
        result = model.generate_image(
            seed=101,
            prompt=PROMPT,
            num_inference_steps=4,
            width=width,
            height=height,
            guidance=1.0,
            image_paths=[reference.source],
            scheduler="flow_match_euler_discrete",
            use_kv_cache=False,
        )
        source = np.asarray(Image.open(reference.source).convert("RGB"), dtype=np.uint8)
        Image.fromarray(restore_luminance(source, result.image)).save(reference.colored)
        print(f"FLUX references: {index + 1}/{len(references)}", flush=True)
    del model
    gc.collect()
    importlib.import_module("mlx.core").clear_cache()


def propagate(
    source: Path,
    output: Path,
    assets: ModelAssets,
    shots: list[Shot],
    references: list[Reference],
    info: ClipInfo,
) -> None:
    os.environ["CMNET_DEVICE"] = "mps"
    sys.path.insert(0, str(assets.cmnet))
    factory = cast(
        CMNETFactory,
        importlib.import_module("colormnet.colormnet_render").ColorMNetRender,
    )
    renderer = factory(
        image_size=-1,
        vid_length=max(info.frames, 100),
        encode_mode=2,
        max_memory_frames=1000,
        reset_on_ref_update=False,
        project_dir=str(assets.cmnet),
        backbone="dinov3",
        enable_proximity_bias=False,
    )
    scale = 384 / min(info.width, info.height)
    size = (round(info.width * scale), round(info.height * scale))
    capture = cv2.VideoCapture(str(source))
    by_frame = {reference.frame: reference for reference in references}
    log_path = output.with_suffix(".ffmpeg.log")
    with log_path.open("w") as log:
        encoder = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{info.width}x{info.height}",
                "-r",
                str(info.fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "ffv1",
                str(output),
            ],
            stdin=subprocess.PIPE,
            stderr=log,
        )
        try:
            assert encoder.stdin is not None
            with torch.inference_mode():
                for shot_index, shot in enumerate(shots):
                    renderer.processor.clear_memory()
                    renderer.frame_count = 0
                    renderer.total_colored_frames = 0
                    renderer.first_mask_loaded = False
                    images = []
                    for frame in shot.references:
                        with Image.open(by_frame[frame].colored) as image:
                            resized = image.convert("RGB").resize(
                                size, Image.Resampling.BILINEAR
                            )
                        images.append(resized)
                        renderer.preload_reference(
                            resized, frame_idx=frame - shot.start
                        )
                    for frame in range(shot.start, shot.end):
                        ok, bgr = capture.read()
                        if not ok:
                            raise RuntimeError(f"Failed to read frame {frame}")
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        renderer.set_ref_frame(
                            images[0] if frame == shot.start else None
                        )
                        prediction = renderer.colorize_frame(
                            frame - shot.start,
                            Image.fromarray(rgb).resize(
                                size, Image.Resampling.BILINEAR
                            ),
                            lab_mode="cpu",
                        )
                        encoder.stdin.write(
                            restore_luminance(rgb, prediction).tobytes()
                        )
                        if frame % 25 == 0:
                            print(
                                f"CMNET2 frames: {frame + 1}/{info.frames}", flush=True
                            )
                    print(f"CMNET2 shots: {shot_index + 1}/{len(shots)}", flush=True)
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError(f"Frame encoding failed. See {log_path}")
        finally:
            capture.release()
            if encoder.poll() is None:
                encoder.terminate()
                encoder.wait()
