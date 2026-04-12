from __future__ import annotations

from pathlib import Path

from src.pipeline.config import AppConfig
from src.pipeline.ffmpeg_utils import compress_video
from src.pipeline.manifest import write_json_manifest
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths


def run_compress_final(
    *,
    config: AppConfig,
    input_path: Path,
    output_path: Path | None,
) -> int:
    paths = resolve_project_paths(config)
    ensure_runtime_directories(paths)

    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input movie not found: {input_path}")

    compression = config.compression
    if output_path is None:
        suffix = (
            f"_crf{int(compression['review_crf'])}"
            f"_{str(compression['review_preset']).replace(' ', '_')}"
        )
        output_path = input_path.with_stem(f"{input_path.stem}{suffix}")
    else:
        output_path = output_path.expanduser().resolve()

    compress_video(
        input_path=input_path,
        output_path=output_path,
        video_codec=str(compression["review_video_codec"]),
        preset=str(compression["review_preset"]),
        crf=int(compression["review_crf"]),
        audio_codec=str(compression["review_audio_codec"]),
        audio_bitrate=str(compression["review_audio_bitrate"]),
        faststart=bool(compression.get("review_faststart", True)),
    )

    manifest_path = paths.manifest_dir / f"compression_{input_path.stem}.json"
    write_json_manifest(
        manifest_path,
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "profile": {
                "video_codec": str(compression["review_video_codec"]),
                "preset": str(compression["review_preset"]),
                "crf": int(compression["review_crf"]),
                "audio_codec": str(compression["review_audio_codec"]),
                "audio_bitrate": str(compression["review_audio_bitrate"]),
                "faststart": bool(compression.get("review_faststart", True)),
            },
        },
    )
    print(f"Compressed review copy written to {output_path}")
    print(f"Compression manifest written to {manifest_path}")
    return 0
