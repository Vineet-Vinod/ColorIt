from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.bootstrap import run_verify_env
from src.pipeline.config import load_config
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

    return parser


def handle_verify_env(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_verify_env(config=config, config_path=Path(args.config))


def handle_download_weights(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_download_weights(
        config=config,
        config_path=Path(args.config),
        url_override=args.url,
        force=args.force,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
