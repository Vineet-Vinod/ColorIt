from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


@dataclass(frozen=True)
class SampleRecord:
    split: str
    movie: str
    source_movie_path: str
    image_path: str
    sample_index: int
    timestamp_seconds: float
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract color training frames for DeOldify fine-tuning. "
            "Training inputs should be generated from these color targets as synthetic grayscale, "
            "not from separate B/W masters."
        )
    )
    parser.add_argument(
        "--movie",
        action="append",
        required=True,
        help="Path to a color movie master. Repeat for multiple titles.",
    )
    parser.add_argument(
        "--output-root",
        default="data/finetune",
        help="Directory that will contain extracted images and manifests.",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=2.0,
        help="Extract one frame every N seconds.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=512,
        help="Resize extracted frames so the longest side matches this value.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Fraction of time blocks reserved for validation.",
    )
    parser.add_argument(
        "--val-block-seconds",
        type=float,
        default=60.0,
        help="Contiguous timeline block size for train/val assignment.",
    )
    parser.add_argument(
        "--min-mean-diff",
        type=float,
        default=2.0,
        help="Skip a sampled frame if mean grayscale delta to the previous accepted frame is below this threshold.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output root before extracting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    images_root = output_root / "images"
    manifest_path = output_root / "manifest.jsonl"
    summary_path = output_root / "summary.json"

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --overwrite.")
        shutil.rmtree(output_root)

    (images_root / "train").mkdir(parents=True, exist_ok=True)
    (images_root / "val").mkdir(parents=True, exist_ok=True)

    all_records: list[SampleRecord] = []
    per_movie_summary: list[dict[str, object]] = []
    for movie_arg in args.movie:
        movie_path = Path(movie_arg).expanduser().resolve()
        if not movie_path.exists():
            raise FileNotFoundError(f"Movie not found: {movie_path}")
        movie_slug = slugify(movie_path.stem)
        kept_records = extract_movie_samples(
            movie_path=movie_path,
            movie_slug=movie_slug,
            images_root=images_root,
            sample_seconds=args.sample_seconds,
            max_side=args.max_side,
            val_ratio=args.val_ratio,
            val_block_seconds=args.val_block_seconds,
            min_mean_diff=args.min_mean_diff,
        )
        all_records.extend(kept_records)
        per_movie_summary.append(
            {
                "movie": movie_slug,
                "source_movie_path": str(movie_path),
                "samples_total": len(kept_records),
                "samples_train": sum(1 for record in kept_records if record.split == "train"),
                "samples_val": sum(1 for record in kept_records if record.split == "val"),
            }
        )
        print(
            f"{movie_slug}: kept {len(kept_records)} samples "
            f"({per_movie_summary[-1]['samples_train']} train / {per_movie_summary[-1]['samples_val']} val)"
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")

    summary = {
        "output_root": str(output_root),
        "sample_seconds": args.sample_seconds,
        "max_side": args.max_side,
        "val_ratio": args.val_ratio,
        "val_block_seconds": args.val_block_seconds,
        "min_mean_diff": args.min_mean_diff,
        "movies": per_movie_summary,
        "samples_total": len(all_records),
        "samples_train": sum(1 for record in all_records if record.split == "train"),
        "samples_val": sum(1 for record in all_records if record.split == "val"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    return 0


def extract_movie_samples(
    *,
    movie_path: Path,
    movie_slug: str,
    images_root: Path,
    sample_seconds: float,
    max_side: int,
    val_ratio: float,
    val_block_seconds: float,
    min_mean_diff: float,
) -> list[SampleRecord]:
    duration_seconds = probe_duration_seconds(movie_path)
    kept_records: list[SampleRecord] = []
    previous_signature: np.ndarray | None = None

    with tempfile.TemporaryDirectory(prefix=f"{movie_slug}_extract_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        extract_with_ffmpeg(
            movie_path=movie_path,
            output_dir=temp_dir,
            sample_seconds=sample_seconds,
            max_side=max_side,
        )
        extracted_paths = sorted(temp_dir.glob("*.png"))
        if not extracted_paths:
            raise RuntimeError(f"No frames were extracted from {movie_path}")

        for sample_index, frame_path in enumerate(extracted_paths):
            timestamp_seconds = min(sample_index * sample_seconds, duration_seconds)
            image_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            grayscale_signature = build_signature(image_bgr)
            if previous_signature is not None:
                mean_diff = float(np.mean(np.abs(grayscale_signature - previous_signature)))
                if mean_diff < min_mean_diff:
                    continue
            previous_signature = grayscale_signature

            split = select_split(
                movie_slug=movie_slug,
                timestamp_seconds=timestamp_seconds,
                val_ratio=val_ratio,
                val_block_seconds=val_block_seconds,
            )
            output_name = f"{movie_slug}_{sample_index:06d}.png"
            output_path = images_root / split / output_name
            shutil.move(str(frame_path), output_path)
            height, width = image_bgr.shape[:2]
            kept_records.append(
                SampleRecord(
                    split=split,
                    movie=movie_slug,
                    source_movie_path=str(movie_path),
                    image_path=str(output_path),
                    sample_index=sample_index,
                    timestamp_seconds=round(timestamp_seconds, 3),
                    width=width,
                    height=height,
                )
            )
    return kept_records


def probe_duration_seconds(movie_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(movie_path),
    ]
    payload = json.loads(subprocess.check_output(command))
    return float(payload["format"]["duration"])


def extract_with_ffmpeg(
    *,
    movie_path: Path,
    output_dir: Path,
    sample_seconds: float,
    max_side: int,
) -> None:
    fps_filter = f"fps=1/{sample_seconds:.8f}"
    scale_filter = (
        f"scale='if(gt(iw,ih),{max_side},-2)':'if(gt(iw,ih),-2,{max_side})':flags=lanczos"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(movie_path),
        "-vf",
        f"{fps_filter},{scale_filter}",
        "-vsync",
        "vfr",
        str(output_dir / "%06d.png"),
    ]
    subprocess.run(command, check=True)


def build_signature(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def select_split(
    *,
    movie_slug: str,
    timestamp_seconds: float,
    val_ratio: float,
    val_block_seconds: float,
) -> str:
    if val_ratio <= 0.0:
        return "train"
    block_index = int(timestamp_seconds // max(val_block_seconds, 1.0))
    cycle = max(int(round(1.0 / val_ratio)), 2)
    offset_digest = hashlib.sha256(movie_slug.encode("utf-8")).hexdigest()
    offset = int(offset_digest[:8], 16) % cycle
    return "val" if (block_index + offset) % cycle == 0 else "train"


def slugify(value: str) -> str:
    chars = [character.lower() if character.isalnum() else "_" for character in value.strip()]
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
