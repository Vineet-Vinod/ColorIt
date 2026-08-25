from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
import time
import warnings

import cv2
import mlx.core as mx
import numpy as np
from PIL import Image
from skimage import color

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
    reference_min_dimension: int
    restoration_strength: float
    chroma_gain: float
    target_chroma_ratio: float | None
    raw_calibration_ratio: float | None
    applied_chroma_gain: float
    calibration_inference_seconds: float
    calibration_block_count: int
    lab_backend: str
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
        reference_min_dimension: int = 256,
        restoration_strength: float = 1.0,
        chroma_gain: float = 1.0,
        target_chroma_ratio: float | None = None,
        max_chroma_gain: float = 2.5,
        lab_backend: str = "opencv",
    ) -> None:
        if precision not in {"float16", "float32"}:
            raise ValueError("DeepRemaster precision must be float16 or float32")
        if min_dimension < 32:
            raise ValueError("DeepRemaster minimum dimension must be at least 32")
        if temporal_block_size < 1:
            raise ValueError("DeepRemaster temporal block size must be positive")
        if reference_min_dimension < 16:
            raise ValueError("DeepRemaster reference minimum dimension must be at least 16")
        if not 0.0 <= restoration_strength <= 1.0:
            raise ValueError("DeepRemaster restoration strength must be between 0 and 1")
        if chroma_gain <= 0.0:
            raise ValueError("DeepRemaster chroma gain must be positive")
        if target_chroma_ratio is not None and target_chroma_ratio <= 0.0:
            raise ValueError("DeepRemaster target chroma ratio must be positive")
        if max_chroma_gain <= 0.0:
            raise ValueError("DeepRemaster maximum chroma gain must be positive")
        if lab_backend not in {"opencv", "skimage"}:
            raise ValueError("DeepRemaster Lab backend must be opencv or skimage")
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
        self.reference_min_dimension = reference_min_dimension
        self.restoration_strength = restoration_strength
        self.chroma_gain = chroma_gain
        self.target_chroma_ratio = target_chroma_ratio
        self.max_chroma_gain = max_chroma_gain
        self.lab_backend = lab_backend
        self.compile_model = compile_model
        self._compiled_forward = None
        if compile_model:
            self._compiled_forward = mx.compile(self._forward_with_features)

    def _forward_with_features(
        self,
        luma,
        level8,
        level16,
        level8_key,
        level8_value,
        level16_key,
        level16_value,
    ):
        return self.model(
            luma,
            reference_features=ReferenceFeatures(
                level8=level8,
                level16=level16,
                level8_key=level8_key,
                level8_value=level8_value,
                level16_key=level16_key,
                level16_value=level16_value,
            ),
        )

    def run_clip(
        self,
        *,
        input_path: Path,
        output_path: Path,
        reference_paths: list[Path],
        reference_times_seconds: list[float] | None = None,
        overwrite: bool,
        progress_label: str,
        output_crf: int = 16,
        output_preset: str = "ultrafast",
        output_pixel_format: str = "yuv420p",
    ) -> DeepRemasterClipRecord:
        input_path = input_path.expanduser().resolve()
        output_path = output_path.expanduser().resolve()
        references = [path.expanduser().resolve() for path in reference_paths]
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        if not references or any(not path.exists() for path in references):
            raise FileNotFoundError("DeepRemaster requires at least one existing reference image")
        if reference_times_seconds is not None and len(reference_times_seconds) != len(references):
            raise ValueError(
                "DeepRemaster reference times must have one entry for every reference image"
            )
        if self.target_chroma_ratio is not None and reference_times_seconds is None:
            raise ValueError(
                "DeepRemaster adaptive chroma calibration requires reference times"
            )
        if reference_times_seconds is not None and any(time_seconds < 0.0 for time_seconds in reference_times_seconds):
            raise ValueError("DeepRemaster reference times must be non-negative")
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
        reference_tensor = _prepare_references(
            references,
            precision=self.precision,
            min_dimension=self.reference_min_dimension,
        )
        reference_features = self.model.prepare_references(reference_tensor)
        mx.eval(
            reference_features.level8,
            reference_features.level16,
            reference_features.level8_key,
            reference_features.level8_value,
            reference_features.level16_key,
            reference_features.level16_value,
        )

        expected_frame_count = playable_cfr_frame_count(media)
        started = time.perf_counter()
        mx.reset_peak_memory()
        applied_chroma_gain = self.chroma_gain
        raw_calibration_ratio: float | None = None
        calibration_inference_seconds = 0.0
        calibration_block_count = 0
        if self.target_chroma_ratio is not None:
            raw_calibration_ratio, calibration_inference_seconds, calibration_block_count = (
                self._calibrate_chroma_gain(
                    input_path=input_path,
                    media=media,
                    references=references,
                    reference_times_seconds=reference_times_seconds or [],
                    expected_frame_count=expected_frame_count,
                    inference_width=inference_width,
                    inference_height=inference_height,
                    reference_features=reference_features,
                )
            )
            # ``raw_calibration_ratio`` is the model's unscaled chroma divided by
            # the colored reference chroma. The target therefore specifies the
            # desired final ratio, independently of the historical fixed gain.
            applied_chroma_gain = min(
                self.max_chroma_gain,
                self.target_chroma_ratio / max(raw_calibration_ratio, 1e-6),
            )

        reader = open_rawvideo_reader(input_path=input_path)
        writer = open_rawvideo_writer(
            output_path=output_path,
            width=width,
            height=height,
            fps=str(media["fps"]),
            video_codec="libx264",
            crf=output_crf,
            pixel_format=output_pixel_format,
            preset=output_preset,
            audio_input_path=None,
        )
        if reader.stdout is None or reader.stderr is None:
            raise RuntimeError("ffmpeg DeepRemaster reader failed to expose pipes")
        if writer.stdin is None or writer.stderr is None:
            raise RuntimeError("ffmpeg DeepRemaster writer failed to expose pipes")

        frame_bytes = width * height * 3
        frame_count = 0
        inference_seconds = calibration_inference_seconds
        with ProgressBar(
            progress_label,
            total=expected_frame_count,
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
                    restored, ab = self._run_model(luma, reference_features)
                    mx.eval(restored, ab)
                    inference_seconds += time.perf_counter() - block_started
                    outputs = _lab_to_rgb_frames(
                        restored,
                        ab,
                        source_frames=frames,
                        output_width=width,
                        output_height=height,
                        restoration_strength=self.restoration_strength,
                        chroma_gain=applied_chroma_gain,
                        lab_backend=self.lab_backend,
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
            reference_min_dimension=self.reference_min_dimension,
            restoration_strength=self.restoration_strength,
            chroma_gain=self.chroma_gain,
            target_chroma_ratio=self.target_chroma_ratio,
            raw_calibration_ratio=(
                round(raw_calibration_ratio, 6) if raw_calibration_ratio is not None else None
            ),
            applied_chroma_gain=round(applied_chroma_gain, 6),
            calibration_inference_seconds=round(calibration_inference_seconds, 6),
            calibration_block_count=calibration_block_count,
            lab_backend=self.lab_backend,
            compiled=self.compile_model,
            runtime_seconds=round(runtime_seconds, 6),
            inference_seconds=round(inference_seconds, 6),
            frames_per_second=round(frame_count / max(inference_seconds, 1e-9), 4),
            peak_memory_bytes=int(mx.get_peak_memory()),
        )

    def _calibrate_chroma_gain(
        self,
        *,
        input_path: Path,
        media: dict[str, str | int | float],
        references: list[Path],
        reference_times_seconds: list[float],
        expected_frame_count: int,
        inference_width: int,
        inference_height: int,
        reference_features: ReferenceFeatures,
    ) -> tuple[float, float, int]:
        """Measure unscaled model chroma only in blocks containing references.

        The prepass deliberately uses the same frame-zero block phase as the
        production pass. It decodes intervening raw frames to reach later
        keyframes, but performs MLX inference only for the unique selected
        blocks.
        """
        reference_indices = _reference_frame_indices(
            reference_times_seconds,
            fps=str(media["fps"]),
            frame_count=expected_frame_count,
        )
        blocks: dict[int, list[tuple[int, int]]] = {}
        for reference_index, frame_index in enumerate(reference_indices):
            block_index = frame_index // self.temporal_block_size
            blocks.setdefault(block_index, []).append((reference_index, frame_index))

        width = int(media["width"])
        height = int(media["height"])
        frame_bytes = width * height * 3
        reference_chroma = [
            _reference_chroma_magnitude(
                reference_path,
                width=inference_width,
                height=inference_height,
                lab_backend=self.lab_backend,
            )
            for reference_path in references
        ]
        predicted_total = 0.0
        reference_total = 0.0
        inference_seconds = 0.0
        block_count = 0
        reader = open_rawvideo_reader(input_path=input_path)
        if reader.stdout is None or reader.stderr is None:
            raise RuntimeError("ffmpeg calibration reader failed to expose pipes")
        try:
            block_start = 0
            while block_start < expected_frame_count:
                frames: list[np.ndarray] | None = None
                requested_references = blocks.get(block_start // self.temporal_block_size)
                exhausted = False
                for _ in range(self.temporal_block_size):
                    data = _read_exact(reader.stdout, frame_bytes)
                    if data is None:
                        exhausted = True
                        break
                    if requested_references is not None:
                        if frames is None:
                            frames = []
                        frames.append(
                            np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                        )
                if requested_references is not None:
                    if not frames:
                        raise RuntimeError("DeepRemaster calibration could not read a reference frame")
                    luma = _prepare_luma(
                        frames,
                        inference_width=inference_width,
                        inference_height=inference_height,
                        precision=self.precision,
                    )
                    block_started = time.perf_counter()
                    _, ab = self._run_model(luma, reference_features)
                    mx.eval(ab)
                    inference_seconds += time.perf_counter() - block_started
                    predicted_ab = np.asarray(ab, dtype=np.float32)[0] * 255.0 - 128.0
                    for reference_index, frame_index in requested_references:
                        local_index = frame_index - block_start
                        if local_index >= len(frames):
                            raise RuntimeError(
                                "DeepRemaster calibration reference falls beyond decoded clip frames"
                            )
                        predicted_total += _mean_chroma_magnitude(predicted_ab[local_index])
                        reference_total += reference_chroma[reference_index]
                    block_count += 1
                if exhausted:
                    break
                block_start += self.temporal_block_size
        finally:
            if reader.stdout is not None:
                reader.stdout.close()
            reader_returncode = reader.wait()
            stderr = reader.stderr.read().decode().strip() if reader.stderr is not None else ""
            if reader.stderr is not None:
                reader.stderr.close()
        if reader_returncode != 0:
            raise RuntimeError(f"ffmpeg DeepRemaster calibration reader failed: {stderr}")
        if reference_total <= 1e-6:
            raise ValueError("DeepRemaster reference chroma is too small to calibrate")
        return predicted_total / reference_total, inference_seconds, block_count

    def _run_model(self, luma, reference_features: ReferenceFeatures):
        if self._compiled_forward is None:
            return self.model(luma, reference_features=reference_features)
        return self._compiled_forward(
            luma,
            reference_features.level8,
            reference_features.level16,
            reference_features.level8_key,
            reference_features.level8_value,
            reference_features.level16_key,
            reference_features.level16_value,
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
            interpolation=cv2.INTER_LINEAR,
        )
        lumas.append(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY))
    array = np.stack(lumas).astype(np.float32)[None, :, :, :, None] / 255.0
    result = mx.array(array)
    return result.astype(mx.float16 if precision == "float16" else mx.float32)


def _reference_frame_indices(
    reference_times_seconds: list[float],
    *,
    fps: str,
    frame_count: int,
) -> list[int]:
    """Map extraction times to the first CFR frame at or after that timestamp."""
    if frame_count < 1:
        raise ValueError("DeepRemaster calibration requires at least one video frame")
    fps_value = float(Fraction(fps))
    if fps_value <= 0.0:
        raise ValueError("DeepRemaster calibration requires a positive frame rate")
    return [
        min(frame_count - 1, max(0, int(np.ceil(time_seconds * fps_value - 1e-9))))
        for time_seconds in reference_times_seconds
    ]


def _mean_chroma_magnitude(ab: np.ndarray) -> float:
    """Mean CIE Lab chroma in the model's physical a/b units."""
    return float(np.mean(np.hypot(ab[:, :, 0], ab[:, :, 1]), dtype=np.float64))


def _reference_chroma_magnitude(
    reference_path: Path,
    *,
    width: int,
    height: int,
    lab_backend: str,
) -> float:
    rgb = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.uint8)
    if rgb.shape[1] != width or rgb.shape[0] != height:
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)
    normalized = rgb.astype(np.float32) / 255.0
    if lab_backend == "opencv":
        ab = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)[:, :, 1:3]
    elif lab_backend == "skimage":
        ab = color.rgb2lab(normalized.astype(np.float64))[:, :, 1:3]
    else:
        raise ValueError("DeepRemaster Lab backend must be opencv or skimage")
    return _mean_chroma_magnitude(ab)


def _prepare_references(
    reference_paths: list[Path],
    *,
    precision: str,
    min_dimension: int = 256,
):
    images = [Image.open(path).convert("RGB") for path in reference_paths]
    aspect = sum(image.width / image.height for image in images) / len(images)
    if aspect >= 1.0:
        target_width = int(min_dimension * aspect)
        target_height = min_dimension
    else:
        target_width = min_dimension
        target_height = int(min_dimension / aspect)

    prepared = []
    for image in images:
        scale = max(target_width, target_height) / max(image.width, image.height)
        resized_width = max(16, int(image.width * scale / 16.0) * 16)
        resized_height = max(16, int(image.height * scale / 16.0) * 16)
        resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (target_width, target_height))
        canvas.paste(
            resized,
            ((target_width - resized_width) // 2, (target_height - resized_height) // 2),
        )
        prepared.append(np.asarray(canvas, dtype=np.float32) / 255.0)
    result = mx.array(np.stack(prepared)[None, ...])
    return result.astype(mx.float16 if precision == "float16" else mx.float32)


def _lab_to_rgb_frames(
    restored,
    ab,
    *,
    source_frames: list[np.ndarray],
    output_width: int,
    output_height: int,
    restoration_strength: float = 1.0,
    chroma_gain: float = 1.0,
    lab_backend: str = "opencv",
) -> list[np.ndarray]:
    restored_np = np.asarray(restored, dtype=np.float32)[0]
    ab_np = np.asarray(ab, dtype=np.float32)[0]
    outputs = []
    for luma, chroma, source in zip(restored_np, ab_np, source_frames, strict=True):
        restored_luma = cv2.resize(
            luma[:, :, 0],
            (output_width, output_height),
            interpolation=cv2.INTER_CUBIC,
        )
        predicted_ab = cv2.resize(
            chroma * 255.0 - 128.0,
            (output_width, output_height),
            interpolation=cv2.INTER_CUBIC,
        )
        lab = np.empty((output_height, output_width, 3), dtype=np.float64)
        if restoration_strength == 1.0:
            lab[:, :, 0] = restored_luma * 100.0
        else:
            if lab_backend == "opencv":
                source_luma = cv2.cvtColor(
                    source.astype(np.float32) / 255.0,
                    cv2.COLOR_RGB2LAB,
                )[:, :, 0]
            else:
                source_luma = color.rgb2lab(source.astype(np.float64) / 255.0)[:, :, 0]
            lab[:, :, 0] = (
                source_luma * (1.0 - restoration_strength)
                + restored_luma * 100.0 * restoration_strength
            )
        lab[:, :, 1:3] = np.clip(predicted_ab * chroma_gain, -100.0, 100.0)
        if lab_backend == "opencv":
            rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
        elif lab_backend == "skimage":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                rgb = color.lab2rgb(lab.astype(np.float64))
        else:
            raise ValueError("DeepRemaster Lab backend must be opencv or skimage")
        rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
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
