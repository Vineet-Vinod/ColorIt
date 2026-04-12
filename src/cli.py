from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.bootstrap import run_verify_env
from src.pipeline.colorize_clip import run_colorize_clip
from src.pipeline.config import load_config
from src.pipeline.inference import colorize_image_file
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.probes import run_extract_probes
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

    clip_parser = subparsers.add_parser(
        "colorize-clip",
        help="Colorize a single video clip and preserve its audio track.",
    )
    clip_parser.add_argument("--config", default="configs/quality.yaml")
    clip_parser.add_argument("--input", required=True, help="Input clip path.")
    clip_parser.add_argument("--output", required=True, help="Output clip path.")
    clip_parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional run manifest override.",
    )
    clip_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    clip_parser.set_defaults(handler=handle_colorize_clip)

    probes_parser = subparsers.add_parser(
        "extract-probes",
        help="Extract reproducible probe clips from the source movie.",
    )
    probes_parser.add_argument("--config", default="configs/default.yaml")
    probes_parser.add_argument(
        "--movie",
        default="~/Movies/Kannada/emme thammanna.mp4",
        help="Source movie path.",
    )
    probes_parser.add_argument(
        "--clip",
        action="append",
        default=[],
        help="Clip spec in the form clip_id=HH:MM:SS-HH:MM:SS or clip_id=...|notes",
    )
    probes_parser.add_argument(
        "--clip-file",
        default=None,
        help="Optional YAML or JSON file containing clip definitions.",
    )
    probes_parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional output manifest override.",
    )
    probes_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing extracted clips.",
    )
    probes_parser.set_defaults(handler=handle_extract_probes)

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
        postprocess_config=config.raw["postprocess"],
    )
    print(f"Colorized frame written to {Path(args.output).resolve()}")
    return 0


def handle_colorize_clip(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_colorize_clip(
        config=config,
        config_path=Path(args.config),
        input_path=Path(args.input),
        output_path=Path(args.output),
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
        overwrite=args.overwrite,
    )


def handle_extract_probes(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_extract_probes(
        config=config,
        config_path=Path(args.config),
        movie_path=Path(args.movie),
        clip_specs=list(args.clip),
        clip_file=Path(args.clip_file) if args.clip_file else None,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
        force=args.force,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
