from __future__ import annotations

import argparse
import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Path
    predictions: Path
    output: Path
    start: int = Field(default=2250, ge=0)
    frames: int = Field(default=750, ge=1)
    budget_bytes: int = Field(default=16867622, gt=0)
    restore_luminance: bool = True


def prediction_frames(options: Options) -> Iterator[NDArray[np.uint8]]:
    if options.predictions.suffix == ".npy":
        predictions = np.load(options.predictions, mmap_mode="r")
        if predictions.shape[0] != options.frames:
            raise ValueError("Prediction count differs from requested source interval")
        yield from predictions
        return
    capture = cv2.VideoCapture(str(options.predictions))
    if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != options.frames:
        capture.release()
        raise ValueError("Prediction count differs from requested source interval")
    try:
        for index in range(options.frames):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Prediction video ended at frame {index}")
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def deliver(options: Options) -> None:
    predictions = prediction_frames(options)
    capture = cv2.VideoCapture(str(options.source))
    capture.set(cv2.CAP_PROP_POS_FRAMES, options.start)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps != 25:
        raise ValueError("This experiment expects a 25 fps source")
    duration = options.frames / fps
    bitrate = int((options.budget_bytes * 0.90 / duration * 8 - 96000) / 1000)
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            "25",
            "-i",
            "-",
            "-ss",
            str(options.start / fps),
            "-t",
            str(duration),
            "-i",
            str(options.source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-frames:v",
            str(options.frames),
            "-c:v",
            "libx265",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-maxrate",
            f"{bitrate}k",
            "-bufsize",
            f"{bitrate * 2}k",
            "-x265-params",
            "pools=4:frame-threads=2:log-level=error",
            "-tag:v",
            "hvc1",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            str(options.output),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    cv2.setNumThreads(4)
    for index, prediction in enumerate(predictions):
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"Source ended at relative frame {index}")
        if options.restore_luminance:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            source_lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
            predicted_lab = cv2.cvtColor(
                prediction.astype(np.float32) / 255, cv2.COLOR_RGB2LAB
            )
            source_lab[:, :, 1:] = cv2.resize(predicted_lab[:, :, 1:], (width, height))
            colored = np.clip(
                cv2.cvtColor(source_lab, cv2.COLOR_LAB2RGB) * 255, 0, 255
            ).astype(np.uint8)
        else:
            colored = cv2.resize(prediction, (width, height))
        encoder.stdin.write(colored.tobytes())
        if index % 100 == 0:
            print("Encoded", index, flush=True)
    capture.release()
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("Delivery encoding failed")
    size = options.output.stat().st_size
    if size > options.budget_bytes:
        raise RuntimeError(
            f"Delivery exceeds size budget: {size} > {options.budget_bytes}"
        )
    print("Delivery ready", options.output, size, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "predictions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--start", type=int, default=2250)
    parser.add_argument("--frames", type=int, default=750)
    parser.add_argument("--budget-bytes", type=int, default=16867622)
    parser.add_argument(
        "--restore-luminance", action=argparse.BooleanOptionalAction, default=True
    )
    deliver(Options.model_validate(vars(parser.parse_args())))
