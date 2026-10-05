from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import types
from itertools import pairwise
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "tmp/adapter_compare_90_120"


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Path
    output: Path
    mode: Literal["assisted", "automatic"]
    start: int = Field(default=2250, ge=0)
    frames: int = Field(default=750, ge=2)
    scenes: Path = ROOT / "experiments/adapter_compare/scene_manifest.json"
    gui: bool = False
    duplicate_first: bool = False


class SceneManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    scene_boundaries: list[int]


def automatic_frames(options: Options) -> list[int]:
    import vapoursynth as vs

    wheel = WORK / "upstream/HAVCServerDiT/packages/vscmnet2-1.1.0-py3-none-any.whl"
    upstream = WORK / "upstream/vscmnet2-wheel"
    if not upstream.exists():
        with ZipFile(wheel) as archive:
            archive.extractall(upstream)
    package = types.ModuleType("vscmnet2")
    package.__path__ = [str(upstream / "vscmnet2")]
    sys.modules["vscmnet2"] = package
    vs.core.num_threads = 1
    vs.core.max_cache_size = 512
    for namespace, library in [
        ("misc", ROOT / "tmp/havc_cpu/plugins/libmiscfilters.dylib"),
        ("tcanny", WORK / "upstream/TCanny/libtcanny.dylib"),
    ]:
        if not hasattr(vs.core, namespace):
            vs.core.std.LoadPlugin(path=str(library))
    detector = importlib.import_module("vscmnet2.vsslib.vsscdetect_edge")
    capture = cv2.VideoCapture(str(options.source))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if capture.get(cv2.CAP_PROP_FPS) != 25:
        raise ValueError("This experiment requires a 25 fps source")
    capture.set(cv2.CAP_PROP_POS_FRAMES, options.start)
    source_path = options.output / "source_rgb.npy"
    decoded = np.lib.format.open_memmap(
        source_path, mode="w+", dtype=np.uint8,
        shape=(options.frames, height, width, 3),
    )
    try:
        for index in range(options.frames):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Source ended at frame {options.start + index}")
            decoded[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        decoded.flush()
    finally:
        capture.release()
    blank = vs.core.std.BlankClip(
        width=width, height=height, format=vs.RGB24,
        length=options.frames, fpsnum=25, fpsden=1,
    )

    def pixels(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        result = f.copy()
        for plane in range(3):
            np.copyto(np.asarray(result[plane]), decoded[n, :, :, plane])
        return result

    source = blank.std.ModifyFrame(clips=blank, selector=pixels)
    detected = detector.SceneDetectEdges(
        source, threshold=0.035, frequency=0, ssim_threshold=0.80,
        sc_diff_offset=2, sc_min_int=25, sc_mult_tht=15,
        tht_white=0.70, tht_black=0.09 if options.gui else 0.10,
    )
    model_source = source.resize.Spline36(width=512, height=288)
    model_inputs = options.output / "model_inputs"
    model_inputs.mkdir(exist_ok=True)
    references = []
    for index in range(options.frames):
        frame = detected.get_frame(index)
        if frame.props["_SceneChangePrev"] == 1:
            references.append(options.start + index)
            model_frame = model_source.get_frame(index)
            rgb = np.stack([np.asarray(model_frame[plane]) for plane in range(3)], axis=-1)
            Image.fromarray(rgb).save(model_inputs / f"ref_{options.start + index:06d}.png")
        if index % 100 == 0:
            print("Author scene detector", index, "anchors", len(references), flush=True)
    if len(references) < 2:
        raise RuntimeError("Author defaults produced fewer than two references; no automatic anchors added")
    return references


def extract(options: Options) -> None:
    if not options.output.resolve().is_relative_to(ROOT / "tmp"):
        raise ValueError("Source artifacts must remain under ColorIt/tmp")
    options.output.mkdir(parents=True, exist_ok=True)
    if options.mode == "automatic":
        frames = automatic_frames(options)
        method = "Author SceneDetectEdges; GUI thresholds" if options.gui else "Author SceneDetectEdges; native vs_cmnet2dit defaults"
    else:
        saved = SceneManifest.model_validate_json(options.scenes.read_text())
        end = options.start + options.frames
        boundaries = sorted(set([options.start, end] + [frame for frame in saved.scene_boundaries
                                                        if options.start < frame < end]))
        frames = []
        for begin, finish in pairwise(boundaries):
            if finish - begin < 25:
                frames.append((begin + finish) // 2)
            else:
                frames.extend(range(begin + min(10, (finish - begin) // 3), finish - 5, 50))
        method = "Visually checked shot cuts, approximately one anchor every two seconds; no color curation yet"
    capture = cv2.VideoCapture(str(options.source))
    records = []
    for frame in frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"Failed to extract reference frame {frame}")
        path = options.output / f"ref_{frame:06d}.png"
        Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path)
        model_path = options.output / "model_inputs" / path.name if options.mode == "automatic" and not options.gui else path
        records.append({"frame": frame, "source_path": str(path.resolve()), "path": str(model_path.resolve())})
    capture.release()
    if options.duplicate_first:
        if not options.gui or len(records) < 2:
            raise ValueError("First-reference duplication requires GUI extraction and two references")
        shutil.copyfile(options.output / f"ref_{frames[1]:06d}.png",
                        options.output / f"ref_{frames[0]:06d}.png")
    result = {"source": str(options.source.resolve()), "start_frame": options.start,
              "frame_count": options.frames, "mode": options.mode, "method": method,
              "first_reference_copied_from": frames[1] if options.duplicate_first else None,
              "references": records, "reference_count": len(records),
              "reference_input_resize": "VapourSynth Spline36 512×288 before DiT" if options.mode == "automatic" and not options.gui else "Original 1920×1080",
              "pairing": "Chosen by the image runner; extraction does not colorize references",
              "automatic_defaults": {"sc_thresh": 0.035, "sc_tht_ssim": 0.80,
                                     "sc_min_int": 25, "sc_tht_offset": 2,
                                     "sc_min_freq": 0} if options.mode == "automatic" else None}
    (options.output / "input_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print("References ready", options.mode, frames, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("assisted", "automatic"), required=True)
    parser.add_argument("--start", type=int, default=2250)
    parser.add_argument("--frames", type=int, default=750)
    parser.add_argument("--scenes", type=Path, default=ROOT / "experiments/adapter_compare/scene_manifest.json")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--duplicate-first", action="store_true")
    extract(Options.model_validate(vars(parser.parse_args())))


if __name__ == "__main__":
    main()
