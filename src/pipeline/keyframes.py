from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from pathlib import Path
import time

import cv2
from PIL import Image

from src.pipeline.config import AppConfig
from src.pipeline.ddcolor_clip import _load_colorizer, _select_device
from src.pipeline.ffmpeg_utils import extract_single_frame, ffprobe_media
from src.pipeline.inference import colorize_rgb_batch
from src.pipeline.model_loader import (
    ModelBundle,
    load_artistic_colorizer_bundle,
    load_stable_colorizer_bundle,
)
from src.pipeline.weights import (
    DEFAULT_DDCOLOR_ARTISTIC_WEIGHTS_PATH,
    DEFAULT_DDCOLOR_WEIGHTS_PATH,
    DEFAULT_DEOLDIFY_ARTISTIC_WEIGHTS_PATH,
    DEFAULT_DEOLDIFY_STABLE_WEIGHTS_PATH,
)


KEYFRAME_COLORING_MODELS = (
    "deoldify",
    "deoldify_stable",
    "ddcolor",
    "ddcolor_artistic",
)


@dataclass(frozen=True)
class KeyframeRecord:
    scene_id: str
    model: str
    source_path: str
    colored_path: str
    time_seconds: float
    runtime_seconds: float
    scene_relative_position: float = 0.5
    reused: bool = False


def normalize_keyframe_positions(positions: object) -> list[float]:
    """Validate scene-relative reference positions and preserve their order."""
    if not isinstance(positions, (list, tuple)) or not positions:
        raise ValueError("deep_remaster.keyframe_positions must be a non-empty list")

    normalized: list[float] = []
    filenames: set[str] = set()
    for raw_position in positions:
        try:
            position = float(raw_position)
        except (TypeError, ValueError) as exc:
            raise ValueError("DeepRemaster keyframe positions must be numeric") from exc
        if not math.isfinite(position) or not 0.0 <= position <= 1.0:
            raise ValueError("DeepRemaster keyframe positions must be between 0.0 and 1.0")
        token = _position_token(position)
        if token in filenames:
            raise ValueError(
                "DeepRemaster keyframe positions must remain distinct after filename rounding"
            )
        filenames.add(token)
        normalized.append(position)
    return normalized


def keyframe_colored_paths(
    *,
    scene_id: str,
    colored_dir: Path,
    positions: object,
) -> list[Path]:
    """Return deterministic reference paths without touching image artifacts."""
    normalized = normalize_keyframe_positions(positions)
    return [
        colored_dir / f"{_keyframe_stem(scene_id, position, len(normalized))}.png"
        for position in normalized
    ]


def _position_token(position: float) -> str:
    return f"p{int(round(position * 10_000)):04d}"


def _keyframe_stem(scene_id: str, position: float, reference_count: int) -> str:
    # Keep the original one-reference artifact name so existing midpoint runs resume.
    if reference_count == 1:
        return scene_id
    return f"{scene_id}__{_position_token(position)}"


class KeyframeColorizer:
    """Reuse one image model while producing scene references."""

    def __init__(self, *, config: AppConfig, model: str, root: Path):
        if model not in KEYFRAME_COLORING_MODELS:
            raise ValueError(f"Unsupported keyframe colorizer: {model}")
        self.config = config
        self.model_name = model
        self.root = root
        settings = config.raw.get("deep_remaster", {})
        render_factor_key = (
            "deoldify_stable_render_factor"
            if model == "deoldify_stable"
            else "deoldify_render_factor"
        )
        self.deoldify_render_factor = int(settings.get(render_factor_key, 35))
        self.ddcolor_input_size = int(settings.get("ddcolor_input_size", 256))
        if self.deoldify_render_factor < 1:
            raise ValueError("DeepRemaster DeOldify render factor must be positive")
        if self.ddcolor_input_size < 64 or self.ddcolor_input_size % 16:
            raise ValueError(
                "DeepRemaster DDColor input size must be at least 64 and divisible by 16"
            )
        self._deoldify: ModelBundle | None = None
        self._ddcolor = None

    def colorize_scene_midpoint(
        self,
        *,
        scene_id: str,
        clip_path: Path,
        source_dir: Path,
        colored_dir: Path,
    ) -> KeyframeRecord:
        """Backward-compatible one-reference midpoint helper."""
        return self.colorize_scene_references(
            scene_id=scene_id,
            clip_path=clip_path,
            source_dir=source_dir,
            colored_dir=colored_dir,
            positions=[0.5],
            reuse_existing=False,
        )[0]

    def colorize_scene_references(
        self,
        *,
        scene_id: str,
        clip_path: Path,
        source_dir: Path,
        colored_dir: Path,
        positions: object,
        reuse_existing: bool,
    ) -> list[KeyframeRecord]:
        """Color one or more scene-relative references with deterministic names.

        With exactly one position this deliberately retains the historical
        ``<scene_id>.png`` path. Multiple positions append a fixed four-decimal
        scene-relative token, for example ``scene_0001__p2000.png``.
        """
        normalized = normalize_keyframe_positions(positions)
        media = ffprobe_media(clip_path)
        duration_seconds = max(0.0, float(media["video_duration_seconds"]))
        fps = float(Fraction(str(media["fps"])))
        last_frame_time = max(0.0, duration_seconds - 1.0 / max(fps, 1.0))
        source_dir.mkdir(parents=True, exist_ok=True)
        colored_dir.mkdir(parents=True, exist_ok=True)

        records: list[KeyframeRecord | None] = [None] * len(normalized)
        pending: list[tuple[int, float, float, Path, Path, float]] = []
        for index, position in enumerate(normalized):
            stem = _keyframe_stem(scene_id, position, len(normalized))
            source_path = source_dir / f"{stem}.png"
            colored_path = colored_dir / f"{stem}.png"
            time_seconds = min(duration_seconds * position, last_frame_time)
            if reuse_existing and _is_usable_image(source_path) and _is_usable_image(colored_path):
                records[index] = KeyframeRecord(
                    scene_id=scene_id,
                    model=self.model_name,
                    source_path=str(source_path),
                    colored_path=str(colored_path),
                    time_seconds=round(time_seconds, 6),
                    runtime_seconds=0.0,
                    scene_relative_position=position,
                    reused=True,
                )
                continue

            started = time.perf_counter()
            extract_single_frame(
                input_path=clip_path,
                output_path=source_path,
                time_seconds=time_seconds,
            )
            pending.append(
                (
                    index,
                    position,
                    time_seconds,
                    source_path,
                    colored_path,
                    time.perf_counter() - started,
                )
            )

        if pending:
            color_started = time.perf_counter()
            self._colorize_batch(
                source_paths=[item[3] for item in pending],
                colored_paths=[item[4] for item in pending],
            )
            color_share = (time.perf_counter() - color_started) / len(pending)
            for (
                index,
                position,
                time_seconds,
                source_path,
                colored_path,
                extraction_seconds,
            ) in pending:
                records[index] = KeyframeRecord(
                    scene_id=scene_id,
                    model=self.model_name,
                    source_path=str(source_path),
                    colored_path=str(colored_path),
                    time_seconds=round(time_seconds, 6),
                    runtime_seconds=round(extraction_seconds + color_share, 6),
                    scene_relative_position=position,
                )

        return [record for record in records if record is not None]

    def _colorize_batch(self, *, source_paths: list[Path], colored_paths: list[Path]) -> None:
        if self.model_name in {"deoldify", "deoldify_stable"}:
            if self._deoldify is None:
                if self.model_name == "deoldify_stable":
                    self._deoldify = load_stable_colorizer_bundle(
                        self.config,
                        weights_path=self.root / DEFAULT_DEOLDIFY_STABLE_WEIGHTS_PATH,
                    )
                else:
                    self._deoldify = load_artistic_colorizer_bundle(
                        self.config,
                        weights_path=self.root / DEFAULT_DEOLDIFY_ARTISTIC_WEIGHTS_PATH,
                    )
            source_rgbs = []
            for source_path in source_paths:
                source_bgr = _read_image(source_path)
                source_rgbs.append(cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB))
            outputs = colorize_rgb_batch(
                model_bundle=self._deoldify,
                input_rgbs=source_rgbs,
                render_factor=self.deoldify_render_factor,
            )
            for colored_path, output_rgb in zip(colored_paths, outputs, strict=True):
                colored_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(output_rgb).save(colored_path)
            return

        if self._ddcolor is None:
            device = _select_device("auto")
            weights_path = (
                DEFAULT_DDCOLOR_ARTISTIC_WEIGHTS_PATH
                if self.model_name == "ddcolor_artistic"
                else DEFAULT_DDCOLOR_WEIGHTS_PATH
            )
            self._ddcolor = _load_colorizer(
                weights_path=self.root / weights_path,
                input_size=self.ddcolor_input_size,
                device=device,
            )
        outputs = self._ddcolor.process_batch([_read_image(path) for path in source_paths])
        for colored_path, output_bgr in zip(colored_paths, outputs, strict=True):
            colored_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)).save(colored_path)


def keyframe_record_to_dict(record: KeyframeRecord) -> dict[str, str | float | bool]:
    return asdict(record)


def _is_usable_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image is not None and image.size > 0


def _read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read extracted keyframe: {path}")
    return image
