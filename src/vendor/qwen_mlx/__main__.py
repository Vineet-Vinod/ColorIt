"""Command line entry point for the isolated Qwen MLX adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapter import (
    DEFAULT_MODEL_DIR,
    QwenEditConfig,
    QwenImageEditRunner,
    download_official_weights,
    smoke_check,
    validate_official_weights,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pinned Qwen Image Edit 2511 through MLX/mflux.")
    parser.add_argument("--download", action="store_true", help="Download and hash-verify official weights.")
    parser.add_argument("--validate", action="store_true", help="Hash-verify an existing weight directory.")
    parser.add_argument("--smoke", action="store_true", help="Check the pinned mflux CLI without loading weights.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--quantize", type=int, choices=(3, 4, 5, 6, 8), default=8)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--mlx-cache-limit-gb", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.download:
        print(json.dumps(download_official_weights(args.model_dir), indent=2))
        return
    if args.validate:
        print(json.dumps(validate_official_weights(args.model_dir), indent=2))
        return
    if args.smoke:
        print(json.dumps(smoke_check(), indent=2))
        return
    if not (args.image and args.prompt and args.output):
        raise SystemExit("provide --image, --prompt, and --output, or use --download/--validate/--smoke")
    config = QwenEditConfig(
        model_dir=args.model_dir,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        quantize=args.quantize,
        width=args.width,
        height=args.height,
        mlx_cache_limit_gb=args.mlx_cache_limit_gb,
    )
    runner = QwenImageEditRunner(config)
    output = runner.edit(args.image, args.prompt, args.output, negative_prompt=args.negative_prompt)
    print(json.dumps({"output": str(output), "config": config.metadata()}, indent=2))


if __name__ == "__main__":
    main()
