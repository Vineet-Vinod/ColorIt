from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from src.pipeline.config import load_config
from src.pipeline.movie import run_colorize_movie
from src.pipeline.weights import run_download_weights


DEFAULT_MOVIE_CONFIG = Path("configs/full_movie.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colorit",
        description="Automatic movie colorization.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="{download-weights,colorize-movie}")

    download_parser = subparsers.add_parser("download-weights", help="Download required model weights.")
    download_parser.set_defaults(handler=handle_download_weights)

    movie_parser = subparsers.add_parser("colorize-movie", help="Run the full movie pipeline.")
    add_movie_args(movie_parser)
    movie_parser.set_defaults(handler=handle_colorize_movie)

    return parser


def add_movie_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Input movie path.")
    parser.add_argument("--output", default=None, help="Defaults to the input path with '_color' appended.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def handle_download_weights(args: argparse.Namespace) -> int:
    config_path = DEFAULT_MOVIE_CONFIG
    return run_download_weights(
        config=load_config(config_path),
        config_path=config_path,
        url_override=None,
        force=False,
    )


def handle_colorize_movie(args: argparse.Namespace) -> int:
    return run_movie(
        config_path=DEFAULT_MOVIE_CONFIG,
        movie_path=Path(args.input),
        output_path=Path(args.output) if args.output else None,
        scene_threshold=None,
        keep_intermediates=False,
        resume=bool(args.resume),
        limit=None,
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
        parser.error("invalid command: 'run'. Use `colorit colorize-movie --input <movie>`.")
    args = parser.parse_args(args_list)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
