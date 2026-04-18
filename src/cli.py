from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.bootstrap import run_verify_env
from src.pipeline.colorize_clip import run_colorize_clip
from src.pipeline.config import load_config
from src.pipeline.inference import colorize_image_file
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.movie import run_colorize_movie
from src.pipeline.weights import run_download_weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="colorit", description="DeOldify movie colorization CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify-env",
        help="Validate local prerequisites and optional single-frame inference.",
    )
    verify_parser.add_argument("--config", default="configs/default.yaml")
    verify_parser.add_argument("--test-image", default=None)
    verify_parser.add_argument("--test-frame-time-seconds", type=float, default=60.0)
    verify_parser.add_argument("--skip-inference", action="store_true")
    verify_parser.set_defaults(handler=handle_verify_env)

    download_parser = subparsers.add_parser(
        "download-weights",
        help="Download the base DeOldify video checkpoint.",
    )
    download_parser.add_argument("--config", default="configs/default.yaml")
    download_parser.add_argument("--url", default=None)
    download_parser.add_argument("--force", action="store_true")
    download_parser.set_defaults(handler=handle_download_weights)

    frame_parser = subparsers.add_parser(
        "colorize-frame",
        help="Colorize a single image.",
    )
    frame_parser.add_argument("--config", default="configs/default.yaml")
    frame_parser.add_argument("--input", required=True)
    frame_parser.add_argument("--output", required=True)
    frame_parser.add_argument("--render-factor", type=int, default=None)
    frame_parser.set_defaults(handler=handle_colorize_frame)

    clip_parser = subparsers.add_parser(
        "colorize-clip",
        help="Colorize a single clip and preserve audio.",
    )
    clip_parser.add_argument("--config", default="configs/default.yaml")
    clip_parser.add_argument("--input", required=True)
    clip_parser.add_argument("--output", required=True)
    clip_parser.add_argument("--overwrite", action="store_true")
    clip_parser.set_defaults(handler=handle_colorize_clip)

    movie_parser = subparsers.add_parser(
        "colorize-movie",
        help="Split a movie into scenes, colorize each scene, and reassemble a final movie.",
    )
    movie_parser.add_argument("--config", default="configs/full_movie.yaml")
    movie_parser.add_argument("--input", required=True, help="Input movie path.")
    movie_parser.add_argument(
        "--output",
        default=None,
        help="Optional final output path. Defaults to the input path with '_color' appended to the stem.",
    )
    movie_parser.add_argument(
        "--scene-threshold",
        type=float,
        default=None,
        help="Optional ffmpeg scene-detection threshold override.",
    )
    movie_parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep scene manifests and intermediate scene clips after a successful run.",
    )
    movie_parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed scene outputs if a previous run left intermediates behind.",
    )
    movie_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for processing only the first N scene units.",
    )
    movie_parser.add_argument("--overwrite", action="store_true")
    movie_parser.set_defaults(handler=handle_colorize_movie)

    return parser


def handle_verify_env(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_verify_env(
        config=config,
        config_path=Path(args.config),
        test_image=Path(args.test_image) if args.test_image else None,
        test_frame_time_seconds=float(args.test_frame_time_seconds),
        skip_inference=bool(args.skip_inference),
    )


def handle_download_weights(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_download_weights(
        config=config,
        config_path=Path(args.config),
        url_override=args.url,
        force=bool(args.force),
    )


def handle_colorize_frame(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    bundle = load_colorizer_bundle(config)
    render_factor = int(args.render_factor or config.model["render_factor"])
    colorize_image_file(
        model_bundle=bundle,
        input_path=Path(args.input),
        output_path=Path(args.output),
        render_factor=render_factor,
        postprocess_config=config.raw["postprocess"],
    )
    print(f"Colorized frame written to {Path(args.output).expanduser().resolve()}")
    return 0


def handle_colorize_clip(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_colorize_clip(
        config=config,
        config_path=Path(args.config),
        input_path=Path(args.input),
        output_path=Path(args.output),
        manifest_path=None,
        overwrite=bool(args.overwrite),
    )


def handle_colorize_movie(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_colorize_movie(
        config=config,
        config_path=Path(args.config),
        movie_path=Path(args.input),
        output_path=Path(args.output) if args.output else None,
        scene_threshold=args.scene_threshold,
        keep_intermediates=bool(args.keep_intermediates),
        resume=bool(args.resume),
        limit=args.limit,
        overwrite=bool(args.overwrite),
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
