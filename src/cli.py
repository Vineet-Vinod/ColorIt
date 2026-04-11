from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.bootstrap import run_verify_env
from src.pipeline.config import load_config
from src.pipeline.inference import colorize_image_file
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.weights import run_download_weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="colorit", description="ColorIt CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify-env",
        help="Validate local prerequisites without loading model weights.",
    )
    verify_parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML config file.",
    )
    verify_parser.add_argument(
        "--test-image",
        default=None,
        help="Optional still image to use instead of extracting a frame from the source movie.",
    )
    verify_parser.add_argument(
        "--test-frame-time-seconds",
        default=60.0,
        type=float,
        help="Timestamp used when extracting a verification frame from the source movie.",
    )
    verify_parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Validate imports and model loading without running a frame through inference.",
    )
    verify_parser.set_defaults(handler=handle_verify_env)

    download_parser = subparsers.add_parser(
        "download-weights",
        help="Download model weights and record metadata without loading them.",
    )
    download_parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML config file.",
    )
    download_parser.add_argument(
        "--url",
        default=None,
        help="Optional direct download URL override.",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination file already exists.",
    )
    download_parser.set_defaults(handler=handle_download_weights)

    frame_parser = subparsers.add_parser(
        "colorize-frame",
        help="Colorize a single still image using the video model.",
    )
    frame_parser.add_argument("--config", default="configs/default.yaml")
    frame_parser.add_argument("--input", required=True, help="Input image path.")
    frame_parser.add_argument("--output", required=True, help="Output image path.")
    frame_parser.add_argument(
        "--render-factor",
        type=int,
        default=None,
        help="Optional render factor override.",
    )
    frame_parser.set_defaults(handler=handle_colorize_frame)

    return parser


def handle_verify_env(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_verify_env(
        config=config,
        config_path=Path(args.config),
        test_image=Path(args.test_image) if args.test_image else None,
        test_frame_time_seconds=args.test_frame_time_seconds,
        skip_inference=args.skip_inference,
    )


def handle_download_weights(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_download_weights(
        config=config,
        config_path=Path(args.config),
        url_override=args.url,
        force=args.force,
    )


def handle_colorize_frame(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    bundle = load_colorizer_bundle(config)
    render_factor = args.render_factor or int(config.model["render_factor"])
    colorize_image_file(
        model_bundle=bundle,
        input_path=Path(args.input),
        output_path=Path(args.output),
        render_factor=render_factor,
    )
    print(f"Colorized frame written to {Path(args.output).resolve()}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
