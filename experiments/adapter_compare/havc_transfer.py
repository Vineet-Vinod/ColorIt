from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import cast

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from .havc_video import ROOT, TESTED, Renderer


class Transfer(BaseModel):
    frame: int
    target: Path
    reference: Path


class Options(BaseModel):
    manifest: Path
    output: Path
    width: int = Field(default=960, ge=384, le=1920)


def transfer(options: Options) -> None:
    records = [Transfer.model_validate(item) for item in json.loads(options.manifest.read_text())]
    os.environ["CMNET_DEVICE"] = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path[:0] = [str(TESTED), str(TESTED / "mps_source"), str(ROOT / "tmp/cmnet2_dinov3_cpu")]
    import torch
    from colormnet.colormnet_render import ColorMNetRender
    from sparse_readout import patch_softmax

    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(0.07)
    patch_softmax(TESTED / "mps_source")
    importlib.import_module("mps_readout").install()
    model = cast(Renderer, ColorMNetRender(
        image_size=-1, vid_length=100, encode_mode=2, max_memory_frames=1000,
        reset_on_ref_update=False, project_dir=str(TESTED / "mps_source"),
        backbone="dinov3", enable_proximity_bias=False,
    ))
    dimensions = (options.width, round(options.width * 9 / 16 / 2) * 2)
    options.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    results = []
    with torch.inference_mode():
        for record in records:
            model.processor.clear_memory()
            model.processor.last_ti_key = None
            model.processor.last_ti_value = None
            model.processor.set_all_labels([1, 2])
            model.frame_count = 0
            model.total_colored_frames = 0
            model.first_mask_loaded = False
            with Image.open(record.reference) as image:
                reference = image.convert("RGB").resize(dimensions, Image.Resampling.BILINEAR)
            with Image.open(record.target) as image:
                target = image.convert("RGB")
            model.set_ref_frame(reference)
            predicted = model.colorize_frame(0, target.resize(dimensions, Image.Resampling.BILINEAR), "cpu")
            lab = cv2.cvtColor(np.asarray(target).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
            predicted_lab = cv2.cvtColor(np.asarray(predicted).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
            lab[:, :, 1:] = cv2.resize(predicted_lab[:, :, 1:], target.size)
            rgb = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255, 0, 255).astype(np.uint8)
            output = options.output / f"ref_{record.frame:06d}.png"
            Image.fromarray(rgb).save(output)
            results.append({"frame": record.frame, "path": str(output.resolve()),
                            "source_path": str(record.target.resolve()),
                            "reused_reference": str(record.reference.resolve())})
            torch.mps.synchronize()
            torch.mps.empty_cache()
            print("Transferred", record.frame, "seconds", perf_counter() - started, flush=True)
    (options.output / "references.json").write_text(json.dumps({"references": results,
        "seconds": perf_counter() - started, "dimensions": dimensions,
        "method": "Author Fix Colors single-reference CMNET2 workflow, MPS port"}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    transfer(Options.model_validate(vars(parser.parse_args())))
