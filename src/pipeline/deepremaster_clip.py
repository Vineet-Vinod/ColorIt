from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import cv2
import mlx.core as mx
import numpy as np
from PIL import Image

from src.pipeline.deepremaster_mlx import (
    ReferenceFeatures,
    convert_official_checkpoint,
    load_deepremaster_mlx,
)
from src.pipeline.ffmpeg_utils import (
    ffprobe_media,
    open_rawvideo_reader,
    open_rawvideo_writer,
    playable_cfr_frame_count,
)
from src.pipeline.progress import ProgressBar


@dataclass(frozen=True)
class DeepRemasterClipRecord:
    input_path: str
    output_path: str
    reference_paths: list[str]
    frame_count: int
    inference_width: int
    inference_height: int
    temporal_block_size: int
    precision: str
    compiled: bool
    runtime_seconds: float
    inference_seconds: float
    frames_per_second: float
    peak_memory_bytes: int


class DeepRemasterRunner:
    """High-throughput MLX runner with reusable model weights."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        converted_weights_path: Path,
        precision: str = "float16",
        min_dimension: int = 192,
        temporal_block_size: int = 5,
        compile_model: bool = True,
        dense_max_scores: int = 32_000_000,
        source_tile_size: int = 1024,
        reference_tile_size: int = 2048,
    ) -> None:
        if precision not in {"float16", "float32"}:
            raise ValueError("DeepRemaster precision must be float16 or float32")
        if min_dimension < 32:
            raise ValueError("DeepRemaster minimum dimension must be at least 32")
        if temporal_block_size < 1:
            raise ValueError("DeepRemaster temporal block size must be positive")
        if not converted_weights_path.exists():
            convert_official_checkpoint(
                checkpoint_path,
                converted_weights_path,
                dtype=precision,
            )
        self.model = load_deepremaster_mlx(
            converted_weights_path,
            dense_max_scores=dense_max_scores,
            source_tile_size=source_tile_size,
            reference_tile_size=reference_tile_size,
        )
        self.precision = precision
        self.min_dimension = min_dimension
        self.temporal_block_size = temporal_block_size
        self.compile_model = compile_model
        self._compiled_forward = None
        if compile_model:
            self._compiled_forward = mx.compile(self._forward_with_features)

    def _forward_with_features(self, luma, level8, level16):
        return self.model(
            luma,
            reference_features=ReferenceFeatures(level8=level8, level16=level16),
        )

    def run_clip(
        self,
        *,
        input_path: Path,
        output_path: Path,
        reference_paths: list[Path],
        overwrite: bool,
        progress_label: str,
        output_crf: int = 16,
        output_preset: str = "ultrafast",
    ) -> DeepRemasterClipRecord:
        input_path = input_path.expanduser().resolve()
        output_path = output_path.expanduser().resolve()
        references = [path.expanduser().resolve() for path in reference_paths]
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        if not references or any(not path.exists() for path in references):
            raise FileNotFoundError("DeepRemaster requires at least one existing reference image")
        if output_path.exists() and not overwrite:
            raise FileExistsError(output_path)

        media = ffprobe_media(input_path)
        width = int(media["width"])
        height = int(media["height"])
        inference_width, inference_height = _inference_size(
            width,
            height,
            min_dimension=self.min_dimension,
        )
        reference_tensor = _prepare_references(references, precision=self.precision)
        reference_features = self.model.prepare_references(reference_tensor)
        mx.eval(reference_features.level8, reference_features.level16)

        reader = open_rawvideo_reader(input_path=input_path)
        writer = open_rawvideo_writer(
            output_path=output_path,
            width=width,
            height=height,
            fps=str(media["fps"]),
            video_codec="libx264",
            crf=output_crf,
            pixel_format="yuv420p",
            preset=output_preset,
            audio_input_path=None,
        )
        if reader.stdout is None or reader.stderr is None:
            raise RuntimeError("ffmpeg DeepRemaster reader failed to expose pipes")
        if writer.stdin is None or writer.stderr is None:
            raise RuntimeError("ffmpeg DeepRemaster writer failed to expose pipes")

        frame_bytes = width * height * 3
        frame_count = 0
        inference_seconds = 0.0
        started = time.perf_counter()
        mx.reset_peak_memory()
        with ProgressBar(
            progress_label,
            total=playable_cfr_frame_count(media),
            unit="frame",
        ) as progress:
            try:
                while True:
                    frames: list[np.ndarray] = []
                    for _ in range(self.temporal_block_size):
                        data = _read_exact(reader.stdout, frame_bytes)
                        if data is None:
                            break
                        frames.append(
                            np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                        )
                    if not frames:
                        break

                    luma = _prepare_luma(
                        frames,
                        inference_width=inference_width,
                        inference_height=inference_height,
                        precision=self.precision,
                    )
                    block_started = time.perf_counter()
                    if self._compiled_forward is None:
                        restored, ab = self.model(
                            luma,
                            reference_features=reference_features,
                        )
                    else:
                        restored, ab = self._compiled_forward(
                            luma,
                            reference_features.level8,
                            reference_features.level16,
                        )
                    mx.eval(restored, ab)
                    inference_seconds += time.perf_counter() - block_started
                    outputs = _lab_to_rgb_frames(
                        restored,
                        ab,
                        output_width=width,
                        output_height=height,
                    )
                    for output in outputs:
                        writer.stdin.write(np.ascontiguousarray(output).tobytes())
                        frame_count += 1
                        progress.update()

                writer.stdin.close()
                writer_returncode = writer.wait()
                reader_returncode = reader.wait()
            finally:
                if reader.stdout is not None:
                    reader.stdout.close()
                if writer.stdin is not None:
                    writer.stdin.close()

            if reader_returncode != 0:
                raise RuntimeError(
                    f"ffmpeg DeepRemaster reader failed: {reader.stderr.read().decode().strip()}"
                )
            if writer_returncode != 0:
                raise RuntimeError(
                    f"ffmpeg DeepRemaster writer failed: {writer.stderr.read().decode().strip()}"
                )
        reader.stderr.close()
        writer.stderr.close()

        runtime_seconds = time.perf_counter() - started
        return DeepRemasterClipRecord(
            input_path=str(input_path),
            output_path=str(output_path),
            reference_paths=[str(path) for path in references],
            frame_count=frame_count,
            inference_width=inference_width,
            inference_height=inference_height,
            temporal_block_size=self.temporal_block_size,
            precision=self.precision,
            compiled=self.compile_model,
            runtime_seconds=round(runtime_seconds, 6),
            inference_seconds=round(inference_seconds, 6),
            frames_per_second=round(frame_count / max(inference_seconds, 1e-9), 4),
            peak_memory_bytes=int(mx.get_peak_memory()),
        )


def clip_record_to_dict(record: DeepRemasterClipRecord) -> dict:
    return asdict(record)


def _inference_size(width: int, height: int, *, min_dimension: int) -> tuple[int, int]:
    scale = min_dimension / min(width, height)
    target_width = max(16, int(round(width * scale / 16.0)) * 16)
    target_height = max(16, int(round(height * scale / 16.0)) * 16)
    return target_width, target_height


def _prepare_luma(
    frames: list[np.ndarray],
    *,
    inference_width: int,
    inference_height: int,
    precision: str,
):
    lumas = []
    for frame in frames:
        resized = cv2.resize(
            frame,
            (inference_width, inference_height),
            interpolation=cv2.INTER_AREA,
        )
        lumas.append(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY))
    array = np.stack(lumas).astype(np.float32)[None, :, :, :, None] / 255.0
    result = mx.array(array)
    return result.astype(mx.float16 if precision == "float16" else mx.float32)


def _prepare_references(reference_paths: list[Path], *, precision: str):
    images = [Image.open(path).convert("RGB") for path in reference_paths]
    aspect = sum(image.width / image.height for image in images) / len(images)
    if aspect >= 1.0:
        target_width = max(16, int(round((256 * aspect) / 16.0)) * 16)
        target_height = 256
    else:
        target_width = 256
        target_height = max(16, int(round((256 / aspect) / 16.0)) * 16)

    prepared = []
    for image in images:
        scale = min(target_width / image.width, target_height / image.height)
        resized_width = max(1, min(target_width, int(round(image.width * scale))))
        resized_height = max(1, min(target_height, int(round(image.height * scale))))
        resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (target_width, target_height))
        canvas.paste(
            resized,
            ((target_width - resized_width) // 2, (target_height - resized_height) // 2),
        )
        prepared.append(np.asarray(canvas, dtype=np.float32) / 255.0)
    result = mx.array(np.stack(prepared)[None, ...])
    return result.astype(mx.float16 if precision == "float16" else mx.float32)


def _lab_to_rgb_frames(restored, ab, *, output_width: int, output_height: int) -> list[np.ndarray]:
    restored_np = np.asarray(restored, dtype=np.float32)[0]
    ab_np = np.asarray(ab, dtype=np.float32)[0]
    outputs = []
    for luma, chroma in zip(restored_np, ab_np, strict=True):
        lab = np.empty((*luma.shape[:2], 3), dtype=np.float32)
        lab[:, :, 0] = luma[:, :, 0] * 100.0
        lab[:, :, 1:3] = np.clip(chroma * 255.0 - 128.0, -100.0, 100.0)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        if rgb.shape[:2] != (output_height, output_width):
            rgb = cv2.resize(rgb, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
        outputs.append(rgb)
    return outputs


def _read_exact(stream, size: int) -> bytes | None:
    buffer = bytearray()
    while len(buffer) < size:
        chunk = stream.read(size - len(buffer))
        if not chunk:
            if not buffer:
                return None
            raise RuntimeError(
                f"Unexpected end of video stream: expected {size} bytes, got {len(buffer)}"
            )
        buffer.extend(chunk)
    return bytes(buffer)
