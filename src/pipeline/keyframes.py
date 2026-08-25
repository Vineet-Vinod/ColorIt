from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import cv2
from PIL import Image

from src.pipeline.config import AppConfig
from src.pipeline.ddcolor_clip import _load_colorizer, _select_device
from src.pipeline.ffmpeg_utils import extract_single_frame, ffprobe_media
from src.pipeline.inference import colorize_image_file
from src.pipeline.model_loader import ModelBundle, load_artistic_colorizer_bundle
from src.pipeline.weights import (
    DEFAULT_DDCOLOR_WEIGHTS_PATH,
    DEFAULT_DEOLDIFY_ARTISTIC_WEIGHTS_PATH,
)


@dataclass(frozen=True)
class KeyframeRecord:
    scene_id: str
    model: str
    source_path: str
    colored_path: str
    time_seconds: float
    runtime_seconds: float


class KeyframeColorizer:
    """Reuse one image model while producing scene references."""

    def __init__(self, *, config: AppConfig, model: str, root: Path):
        if model not in {"deoldify", "ddcolor"}:
            raise ValueError(f"Unsupported keyframe colorizer: {model}")
        self.config = config
        self.model_name = model
        self.root = root
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
        started = time.perf_counter()
        media = ffprobe_media(clip_path)
        time_seconds = max(0.0, float(media["video_duration_seconds"]) * 0.5)
        source_path = source_dir / f"{scene_id}.png"
        colored_path = colored_dir / f"{scene_id}.png"
        extract_single_frame(
            input_path=clip_path,
            output_path=source_path,
            time_seconds=time_seconds,
        )
        if self.model_name == "deoldify":
            self._colorize_deoldify(source_path=source_path, colored_path=colored_path)
        else:
            self._colorize_ddcolor(source_path=source_path, colored_path=colored_path)
        return KeyframeRecord(
            scene_id=scene_id,
            model=self.model_name,
            source_path=str(source_path),
            colored_path=str(colored_path),
            time_seconds=round(time_seconds, 6),
            runtime_seconds=round(time.perf_counter() - started, 6),
        )

    def _colorize_deoldify(self, *, source_path: Path, colored_path: Path) -> None:
        if self._deoldify is None:
            self._deoldify = load_artistic_colorizer_bundle(
                self.config,
                weights_path=self.root / DEFAULT_DEOLDIFY_ARTISTIC_WEIGHTS_PATH,
            )
        colorize_image_file(
            model_bundle=self._deoldify,
            input_path=source_path,
            output_path=colored_path,
            render_factor=35,
        )

    def _colorize_ddcolor(self, *, source_path: Path, colored_path: Path) -> None:
        if self._ddcolor is None:
            device = _select_device("auto")
            self._ddcolor = _load_colorizer(
                weights_path=self.root / DEFAULT_DDCOLOR_WEIGHTS_PATH,
                input_size=256,
                device=device,
            )
        source_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source_bgr is None:
            raise ValueError(f"Failed to read extracted keyframe: {source_path}")
        colored_bgr = self._ddcolor.process(source_bgr)
        colored_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)).save(colored_path)


def keyframe_record_to_dict(record: KeyframeRecord) -> dict[str, str | float]:
    return asdict(record)
