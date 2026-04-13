from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.assemble import run_assemble_final
from src.pipeline.benchmark import run_benchmark_clips
from src.pipeline.bootstrap import run_verify_env
from src.pipeline.batch import run_colorize_batch
from src.pipeline.colorize_clip import run_colorize_clip
from src.pipeline.compress import run_compress_final
from src.pipeline.config import load_config
from src.pipeline.conv_opt import run_benchmark_convs, run_extract_conv_shapes
from src.pipeline.inference import colorize_image_file
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.probes import run_extract_probes
from src.pipeline.scenes import run_detect_scenes
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

    scenes_parser = subparsers.add_parser(
        "detect-scenes",
        help="Detect scene boundaries and write a scene manifest.",
    )
    scenes_parser.add_argument("--config", default="configs/full_movie.yaml")
    scenes_parser.add_argument(
        "--movie",
        default="~/Movies/Kannada/emme thammanna.mp4",
        help="Source movie path.",
    )
    scenes_parser.add_argument(
        "--output",
        default=None,
        help="Optional scene manifest output path.",
    )
    scenes_parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="ffmpeg scene-change threshold. Lower values produce more cuts.",
    )
    scenes_parser.set_defaults(handler=handle_detect_scenes)

    batch_parser = subparsers.add_parser(
        "colorize-batch",
        help="Colorize scenes from a scene manifest with resume support.",
    )
    batch_parser.add_argument("--config", default="configs/full_movie.yaml")
    batch_parser.add_argument(
        "--movie",
        default="~/Movies/Kannada/emme thammanna.mp4",
        help="Source movie path.",
    )
    batch_parser.add_argument(
        "--scene-manifest",
        required=True,
        help="Path to a scene manifest generated by detect-scenes.",
    )
    batch_parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already completed scene outputs if present in the batch manifest.",
    )
    batch_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for validating only the first N scene units.",
    )
    batch_parser.set_defaults(handler=handle_colorize_batch)

    assemble_parser = subparsers.add_parser(
        "assemble-final",
        help="Concatenate colorized scene clips into a final movie output.",
    )
    assemble_parser.add_argument("--config", default="configs/full_movie.yaml")
    assemble_parser.add_argument(
        "--scene-manifest",
        required=True,
        help="Path to a scene manifest generated by detect-scenes.",
    )
    assemble_parser.add_argument(
        "--output",
        default=None,
        help="Optional output movie path.",
    )
    assemble_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for validating only the first N scene units.",
    )
    assemble_parser.set_defaults(handler=handle_assemble_final)

    compress_parser = subparsers.add_parser(
        "compress-final",
        help="Generate a smaller review copy from a final assembled movie.",
    )
    compress_parser.add_argument("--config", default="configs/full_movie.yaml")
    compress_parser.add_argument(
        "--input",
        required=True,
        help="Input assembled movie path.",
    )
    compress_parser.add_argument(
        "--output",
        default=None,
        help="Optional compressed output path.",
    )
    compress_parser.set_defaults(handler=handle_compress_final)

    benchmark_parser = subparsers.add_parser(
        "benchmark-clips",
        help="Profile representative clip runs and record stage timings plus resource samples.",
    )
    benchmark_parser.add_argument("--config", default="configs/quality.yaml")
    benchmark_parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input clip path. Repeat for multiple clips.",
    )
    benchmark_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for benchmark renders.",
    )
    benchmark_parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional benchmark manifest path override.",
    )
    benchmark_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing benchmark outputs.",
    )
    benchmark_parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=1.0,
        help="Sampling interval for CPU and GPU utilization.",
    )
    benchmark_parser.set_defaults(handler=handle_benchmark_clips)

    conv_shapes_parser = subparsers.add_parser(
        "extract-conv-shapes",
        help="Extract real convolution layer shapes from the current model on a representative clip.",
    )
    conv_shapes_parser.add_argument("--config", default="configs/quality.yaml")
    conv_shapes_parser.add_argument(
        "--input",
        default="data/probe_clips/clip_10.mp4",
        help="Representative input clip path.",
    )
    conv_shapes_parser.add_argument(
        "--output",
        default="optimize/artifacts/conv_shapes.json",
        help="Output JSON manifest path.",
    )
    conv_shapes_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on sampled frames.",
    )
    conv_shapes_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on sampled batches.",
    )
    conv_shapes_parser.set_defaults(handler=handle_extract_conv_shapes)

    conv_bench_parser = subparsers.add_parser(
        "benchmark-convs",
        help="Benchmark real convolution shapes against the PyTorch MPS baseline and an optional candidate.",
    )
    conv_bench_parser.add_argument("--config", default="configs/full_movie.yaml")
    conv_bench_parser.add_argument(
        "--opt-config",
        default="optimize/configs/default.yaml",
        help="Optimize workspace config for tolerances and loop defaults.",
    )
    conv_bench_parser.add_argument(
        "--shapes-manifest",
        default="optimize/artifacts/conv_shapes.json",
        help="Convolution shape manifest path from extract-conv-shapes.",
    )
    conv_bench_parser.add_argument(
        "--candidate",
        default=None,
        help="Optional candidate Python module path implementing run_case(...).",
    )
    conv_bench_parser.add_argument(
        "--output",
        default="optimize/artifacts/conv_benchmark_results.json",
        help="Output JSON results path.",
    )
    conv_bench_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Benchmark the top K cases by total MACs.",
    )
    conv_bench_parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Specific case id to benchmark. Repeat to select multiple cases.",
    )
    conv_bench_parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Optional warmup iteration override.",
    )
    conv_bench_parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Optional timed iteration override.",
    )
    conv_bench_parser.add_argument(
        "--correctness-trials",
        type=int,
        default=None,
        help="Optional correctness trial override.",
    )
    conv_bench_parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Optional absolute tolerance override.",
    )
    conv_bench_parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Optional relative tolerance override.",
    )
    conv_bench_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed override.",
    )
    conv_bench_parser.set_defaults(handler=handle_benchmark_convs)

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


def handle_detect_scenes(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_detect_scenes(
        config=config,
        config_path=Path(args.config),
        movie_path=Path(args.movie),
        output_path=Path(args.output) if args.output else None,
        threshold=args.threshold,
    )


def handle_colorize_batch(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_colorize_batch(
        config=config,
        config_path=Path(args.config),
        movie_path=Path(args.movie),
        scene_manifest_path=Path(args.scene_manifest),
        resume=args.resume,
        limit=args.limit,
    )


def handle_assemble_final(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_assemble_final(
        config=config,
        scene_manifest_path=Path(args.scene_manifest),
        output_path=Path(args.output) if args.output else None,
        limit=args.limit,
    )


def handle_compress_final(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_compress_final(
        config=config,
        input_path=Path(args.input),
        output_path=Path(args.output) if args.output else None,
    )


def handle_benchmark_clips(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_benchmark_clips(
        config=config,
        config_path=Path(args.config),
        input_paths=[Path(value) for value in args.input],
        output_dir=Path(args.output_dir) if args.output_dir else None,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
        overwrite=args.overwrite,
        sample_interval_seconds=args.sample_interval_seconds,
    )


def handle_extract_conv_shapes(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_extract_conv_shapes(
        config=config,
        config_path=Path(args.config),
        input_path=Path(args.input),
        output_path=Path(args.output) if args.output else None,
        max_frames=args.max_frames,
        max_batches=args.max_batches,
    )


def handle_benchmark_convs(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    return run_benchmark_convs(
        config=config,
        config_path=Path(args.config),
        optimize_config_path=Path(args.opt_config) if args.opt_config else None,
        shapes_manifest_path=Path(args.shapes_manifest),
        candidate_path=Path(args.candidate) if args.candidate else None,
        output_path=Path(args.output) if args.output else None,
        top_k=args.top_k,
        case_ids=list(args.case_id),
        warmup_iterations=args.warmup,
        timed_iterations=args.iterations,
        correctness_trials=args.correctness_trials,
        atol=args.atol,
        rtol=args.rtol,
        seed=args.seed,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
