from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from src.pipeline.colorize_clip import run_colorize_clip
from src.pipeline.config import load_config
from src.pipeline.ddcolor_clip import DEFAULT_DDCOLOR_WEIGHTS_PATH, run_ddcolor_clip
from src.pipeline.fast_semantic_movie import run_fast_semantic_movie
from src.pipeline.inference import colorize_image_file
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.movie import run_colorize_movie
from src.pipeline.weights import run_download_weights


COMMANDS = {
    "download-weights",
    "colorize-movie",
    "colorize-frame",
    "colorize-clip",
    "ddcolor-clip",
    "fast-semantic-movie",
}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colorit",
        description="Automatic movie colorization. Pass a movie path for the default pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="{download-weights,colorize-movie}")

    download_parser = subparsers.add_parser("download-weights", help="Download the DeOldify checkpoint.")
    download_parser.add_argument("--config", default="configs/default.yaml")
    download_parser.add_argument("--url", default=None)
    download_parser.add_argument("--force", action="store_true")
    download_parser.set_defaults(handler=handle_download_weights)

    default_movie_parser = subparsers.add_parser("__movie", prog="colorit", help=argparse.SUPPRESS)
    add_movie_args(default_movie_parser, positional=True)
    default_movie_parser.set_defaults(handler=handle_default_movie)

    movie_parser = subparsers.add_parser("colorize-movie", help="Run the full movie pipeline.")
    add_movie_args(movie_parser, positional=False)
    movie_parser.add_argument("--config", default="configs/full_movie.yaml")
    movie_parser.add_argument("--scene-threshold", type=float, default=None)
    movie_parser.add_argument("--limit", type=int, default=None)
    movie_parser.set_defaults(handler=handle_colorize_movie)

    frame_parser = subparsers.add_parser("colorize-frame", help=argparse.SUPPRESS)
    frame_parser.add_argument("--config", default="configs/default.yaml")
    frame_parser.add_argument("--input", required=True)
    frame_parser.add_argument("--output", required=True)
    frame_parser.add_argument("--render-factor", type=int, default=None)
    frame_parser.set_defaults(handler=handle_colorize_frame)

    clip_parser = subparsers.add_parser("colorize-clip", help=argparse.SUPPRESS)
    clip_parser.add_argument("--config", default="configs/default.yaml")
    clip_parser.add_argument("--input", required=True)
    clip_parser.add_argument("--output", required=True)
    clip_parser.add_argument("--overwrite", action="store_true")
    clip_parser.set_defaults(handler=handle_colorize_clip)

    ddcolor_parser = subparsers.add_parser("ddcolor-clip", help=argparse.SUPPRESS)
    ddcolor_parser.add_argument("--input", required=True)
    ddcolor_parser.add_argument("--output", required=True)
    ddcolor_parser.add_argument("--weights", default=str(DEFAULT_DDCOLOR_WEIGHTS_PATH))
    ddcolor_parser.add_argument("--input-size", type=int, default=512)
    ddcolor_parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    ddcolor_parser.add_argument("--output-preset", default="medium")
    ddcolor_parser.add_argument("--overwrite", action="store_true")
    ddcolor_parser.set_defaults(handler=handle_ddcolor_clip)

    fast_parser = subparsers.add_parser("fast-semantic-movie", help=argparse.SUPPRESS)
    fast_parser.add_argument("--input", required=True)
    fast_parser.add_argument("--output", default=None)
    fast_parser.add_argument("--weights", default=str(DEFAULT_DDCOLOR_WEIGHTS_PATH))
    fast_parser.add_argument("--work-dir", default=None)
    fast_parser.add_argument("--chunk-seconds", type=float, default=60.0)
    fast_parser.add_argument("--limit-chunks", type=int, default=None)
    fast_parser.add_argument("--overwrite", action="store_true")
    fast_parser.set_defaults(handler=handle_fast_semantic_movie)

    subparsers._choices_actions = [
        action for action in subparsers._choices_actions if action.dest in {"download-weights", "colorize-movie"}
    ]
    return parser


def add_movie_args(parser: argparse.ArgumentParser, *, positional: bool) -> None:
    if positional:
        parser.add_argument("movie", help="Input movie path.")
    else:
        parser.add_argument("--input", required=True, help="Input movie path.")
    parser.add_argument("--output", default=None, help="Defaults to the input path with '_color' appended.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def handle_download_weights(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    return run_download_weights(
        config=load_config(config_path),
        config_path=config_path,
        url_override=args.url,
        force=bool(args.force),
    )


def handle_colorize_frame(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    render_factor = int(args.render_factor or config.model["render_factor"])
    colorize_image_file(
        model_bundle=load_colorizer_bundle(config),
        input_path=Path(args.input),
        output_path=Path(args.output),
        render_factor=render_factor,
    )
    print(f"Colorized frame written to {Path(args.output).expanduser().resolve()}")
    return 0


def handle_colorize_clip(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    return run_colorize_clip(
        config=load_config(config_path),
        config_path=config_path,
        input_path=Path(args.input),
        output_path=Path(args.output),
        manifest_path=None,
        overwrite=bool(args.overwrite),
    )


def handle_ddcolor_clip(args: argparse.Namespace) -> int:
    return run_ddcolor_clip(
        input_path=Path(args.input),
        output_path=Path(args.output),
        weights_path=Path(args.weights),
        input_size=int(args.input_size),
        device=str(args.device),
        output_preset=str(args.output_preset),
        overwrite=bool(args.overwrite),
    )


def handle_fast_semantic_movie(args: argparse.Namespace) -> int:
    return run_fast_semantic_movie(
        input_path=Path(args.input),
        output_path=Path(args.output) if args.output else None,
        weights_path=Path(args.weights),
        work_dir=Path(args.work_dir) if args.work_dir else None,
        chunk_seconds=float(args.chunk_seconds),
        limit_chunks=args.limit_chunks,
        overwrite=bool(args.overwrite),
    )


def handle_default_movie(args: argparse.Namespace) -> int:
    return run_movie(
        config_path=Path("configs/full_movie.yaml"),
        movie_path=Path(args.movie),
        output_path=Path(args.output) if args.output else None,
        scene_threshold=None,
        keep_intermediates=bool(args.keep_intermediates),
        resume=bool(args.resume),
        limit=None,
        overwrite=bool(args.overwrite),
    )


def handle_colorize_movie(args: argparse.Namespace) -> int:
    return run_movie(
        config_path=Path(args.config),
        movie_path=Path(args.input),
        output_path=Path(args.output) if args.output else None,
        scene_threshold=args.scene_threshold,
        keep_intermediates=bool(args.keep_intermediates),
        resume=bool(args.resume),
        limit=args.limit,
        overwrite=bool(args.overwrite),
    )


def run_movie(
    *,
    config_path: Path,
    movie_path: Path,
    output_path: Path | None,
    scene_threshold: float | None,
    keep_intermediates: bool,
    resume: bool,
    limit: int | None,
    overwrite: bool,
) -> int:
    return run_colorize_movie(
        config=load_config(config_path),
        config_path=config_path,
        movie_path=movie_path,
        output_path=output_path,
        scene_threshold=scene_threshold,
        keep_intermediates=keep_intermediates,
        resume=resume,
        limit=limit,
        overwrite=overwrite,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "run":
        parser.error("invalid command: 'run'. Use `colorit <movie>` for the default pipeline.")
    if args_list and _looks_like_movie_path(args_list[0]):
        args_list.insert(0, "__movie")
    args = parser.parse_args(args_list)
    return int(args.handler(args))


def _looks_like_movie_path(value: str) -> bool:
    if value.startswith("-") or value in COMMANDS:
        return False
    expanded = Path(value).expanduser()
    return expanded.exists() or expanded.suffix.lower() in VIDEO_SUFFIXES or "/" in value


if __name__ == "__main__":
    raise SystemExit(main())
