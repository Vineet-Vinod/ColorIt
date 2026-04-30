from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.colorize_clip import run_colorize_clip
from src.pipeline.config import load_config
from src.pipeline.ffmpeg_utils import compress_video
from src.pipeline.inference import colorize_image_file
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.movie import run_colorize_movie
from src.pipeline.segment_clip import run_segment_clip
from src.pipeline.segment_debug import run_render_segment_debug
from src.pipeline.segment_filter import run_filter_segments
from src.pipeline.segment_recolor import run_recolor_segments
from src.pipeline.segment_track import run_track_segments
from src.pipeline.weights import run_download_weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="colorit", description="DeOldify movie colorization CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    compress_parser = subparsers.add_parser(
        "compress-video",
        help="Create a compressed derivative of an existing video.",
    )
    compress_parser.add_argument("--input", required=True)
    compress_parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to the input path with '_compressed' appended to the stem.",
    )
    compress_parser.add_argument("--video-codec", default="libx264")
    compress_parser.add_argument("--preset", default="medium")
    compress_parser.add_argument("--crf", type=int, default=23)
    compress_parser.add_argument("--audio-codec", default="aac")
    compress_parser.add_argument("--audio-bitrate", default="128k")
    compress_parser.add_argument("--no-faststart", action="store_true")
    compress_parser.add_argument("--overwrite", action="store_true")
    compress_parser.set_defaults(handler=handle_compress_video)

    segment_parser = subparsers.add_parser(
        "segment-clip",
        help="Generate a reusable segment manifest and masks for a clip.",
    )
    segment_parser.add_argument("--input", required=True, help="Input clip path.")
    segment_parser.add_argument(
        "--backend",
        default="polygon",
        choices=("polygon", "human-parser"),
        help="Segmentation backend to run.",
    )
    segment_parser.add_argument("--output-dir", required=True, help="Segment artifact directory.")
    segment_parser.add_argument(
        "--tracks",
        default=None,
        help="Backend-specific polygon track JSON file.",
    )
    segment_parser.add_argument(
        "--model-id",
        default=None,
        help="Backend-specific model id. Defaults to the vetted human parser model.",
    )
    segment_parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps"),
        help="Model device for model-backed segmentation backends.",
    )
    segment_parser.add_argument("--overwrite", action="store_true")
    segment_parser.set_defaults(handler=handle_segment_clip)

    segment_debug_parser = subparsers.add_parser(
        "render-segment-debug",
        help="Render a video overlay for a segment manifest.",
    )
    segment_debug_parser.add_argument("--input", required=True, help="Input clip path.")
    segment_debug_parser.add_argument(
        "--segment-manifest",
        required=True,
        help="Segment manifest JSON path.",
    )
    segment_debug_parser.add_argument("--output", required=True, help="Output overlay video path.")
    segment_debug_parser.add_argument(
        "--include-label",
        action="append",
        default=[],
        help="Only render this label. Can be passed multiple times.",
    )
    segment_debug_parser.add_argument(
        "--include-track",
        action="append",
        default=[],
        help="Only render this track id. Can be passed multiple times.",
    )
    segment_debug_parser.add_argument(
        "--exclude-label",
        action="append",
        default=[],
        help="Skip this label. Can be passed multiple times.",
    )
    segment_debug_parser.add_argument("--alpha", type=float, default=0.45)
    segment_debug_parser.add_argument("--overwrite", action="store_true")
    segment_debug_parser.set_defaults(handler=handle_render_segment_debug)

    segment_filter_parser = subparsers.add_parser(
        "filter-segments",
        help="Build a cleaned segment manifest from include and veto labels.",
    )
    segment_filter_parser.add_argument(
        "--segment-manifest",
        required=True,
        help="Source segment manifest JSON path.",
    )
    segment_filter_parser.add_argument("--output-dir", required=True, help="Filtered segment directory.")
    segment_filter_parser.add_argument(
        "--include-label",
        action="append",
        default=[],
        help="Include this source label. Can be passed multiple times.",
    )
    segment_filter_parser.add_argument(
        "--veto-label",
        action="append",
        default=[],
        help="Subtract this source label. Can be passed multiple times.",
    )
    segment_filter_parser.add_argument(
        "--split-guide-label",
        action="append",
        default=[],
        help="Use connected components from this label as actor anchors for splitting the output mask.",
    )
    segment_filter_parser.add_argument("--output-label", default="costume_candidate")
    segment_filter_parser.add_argument("--min-area", type=int, default=500)
    segment_filter_parser.add_argument("--guide-min-area", type=int, default=120)
    segment_filter_parser.add_argument("--guide-merge-distance", type=float, default=95.0)
    segment_filter_parser.add_argument("--close-px", type=int, default=0)
    segment_filter_parser.add_argument("--erode-px", type=int, default=0)
    segment_filter_parser.add_argument("--dilate-px", type=int, default=0)
    segment_filter_parser.add_argument("--veto-dilate-px", type=int, default=0)
    segment_filter_parser.add_argument("--overwrite", action="store_true")
    segment_filter_parser.set_defaults(handler=handle_filter_segments)

    segment_recolor_parser = subparsers.add_parser(
        "recolor-segments",
        help="Apply a fixed chroma color inside segment masks of an existing colorized clip.",
    )
    segment_recolor_parser.add_argument("--input", required=True, help="Input colorized clip path.")
    segment_recolor_parser.add_argument(
        "--segment-manifest",
        required=True,
        help="Segment manifest JSON path aligned to the input clip.",
    )
    segment_recolor_parser.add_argument("--output", required=True, help="Output recolored clip path.")
    segment_recolor_parser.add_argument(
        "--color",
        default=None,
        help="Fallback target #RRGGBB color. Required unless --palette-manifest covers every target track.",
    )
    segment_recolor_parser.add_argument(
        "--palette-manifest",
        default=None,
        help="Optional JSON palette with {'tracks': {'track_id': '#RRGGBB'}}.",
    )
    segment_recolor_parser.add_argument(
        "--include-label",
        action="append",
        default=[],
        help="Only recolor this label. Defaults to all labels in the manifest.",
    )
    segment_recolor_parser.add_argument(
        "--recolor-mode",
        choices=("lab-chroma", "color-filter"),
        default="lab-chroma",
        help="Compositing method for adding target colors.",
    )
    segment_recolor_parser.add_argument("--chroma-blend", type=float, default=0.70)
    segment_recolor_parser.add_argument("--mask-erode-px", type=int, default=1)
    segment_recolor_parser.add_argument("--mask-feather-px", type=int, default=3)
    segment_recolor_parser.add_argument("--temporal-mask-blend", type=float, default=0.20)
    segment_recolor_parser.add_argument("--overwrite", action="store_true")
    segment_recolor_parser.set_defaults(handler=handle_recolor_segments)

    segment_track_parser = subparsers.add_parser(
        "track-segments",
        help="Split segment masks into connected components and track them over time.",
    )
    segment_track_parser.add_argument(
        "--segment-manifest",
        required=True,
        help="Source segment manifest JSON path.",
    )
    segment_track_parser.add_argument("--output-dir", required=True, help="Tracked segment directory.")
    segment_track_parser.add_argument(
        "--include-label",
        action="append",
        default=[],
        help="Only track this label. Defaults to all labels in the manifest.",
    )
    segment_track_parser.add_argument("--output-label", default="costume_track")
    segment_track_parser.add_argument("--min-area", type=int, default=1000)
    segment_track_parser.add_argument("--iou-threshold", type=float, default=0.10)
    segment_track_parser.add_argument("--max-center-distance", type=float, default=180.0)
    segment_track_parser.add_argument("--max-missing-frames", type=int, default=3)
    segment_track_parser.add_argument(
        "--split-wide-components",
        action="store_true",
        help="Split oversized connected components at vertical mask-density valleys before tracking.",
    )
    segment_track_parser.add_argument(
        "--preserve-source-instances",
        action="store_true",
        help="Track source instances separately instead of merging all included labels before component analysis.",
    )
    segment_track_parser.add_argument(
        "--max-component-width-ratio",
        type=float,
        default=0.42,
        help="Frame-width ratio above which --split-wide-components may split a component.",
    )
    segment_track_parser.add_argument("--min-split-valley-ratio", type=float, default=0.45)
    segment_track_parser.add_argument("--overwrite", action="store_true")
    segment_track_parser.set_defaults(handler=handle_track_segments)

    return parser


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


def handle_compress_video(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_compressed{input_path.suffix}")
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    compress_video(
        input_path=input_path,
        output_path=output_path,
        video_codec=str(args.video_codec),
        preset=str(args.preset),
        crf=int(args.crf),
        audio_codec=str(args.audio_codec),
        audio_bitrate=str(args.audio_bitrate),
        faststart=not bool(args.no_faststart),
    )
    print(f"Compressed video written to {output_path}")
    return 0


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


def handle_segment_clip(args: argparse.Namespace) -> int:
    return run_segment_clip(
        input_path=Path(args.input),
        backend=str(args.backend),
        output_dir=Path(args.output_dir),
        tracks_path=Path(args.tracks) if args.tracks else None,
        model_id=args.model_id,
        device=str(args.device),
        overwrite=bool(args.overwrite),
    )


def handle_render_segment_debug(args: argparse.Namespace) -> int:
    return run_render_segment_debug(
        input_path=Path(args.input),
        segment_manifest_path=Path(args.segment_manifest),
        output_path=Path(args.output),
        include_labels=list(args.include_label),
        include_tracks=list(args.include_track),
        exclude_labels=list(args.exclude_label),
        alpha=float(args.alpha),
        overwrite=bool(args.overwrite),
    )


def handle_filter_segments(args: argparse.Namespace) -> int:
    return run_filter_segments(
        segment_manifest_path=Path(args.segment_manifest),
        output_dir=Path(args.output_dir),
        include_labels=list(args.include_label),
        veto_labels=list(args.veto_label),
        split_guide_labels=list(args.split_guide_label),
        output_label=str(args.output_label),
        min_area=int(args.min_area),
        guide_min_area=int(args.guide_min_area),
        guide_merge_distance=float(args.guide_merge_distance),
        close_px=int(args.close_px),
        erode_px=int(args.erode_px),
        dilate_px=int(args.dilate_px),
        veto_dilate_px=int(args.veto_dilate_px),
        overwrite=bool(args.overwrite),
    )


def handle_recolor_segments(args: argparse.Namespace) -> int:
    return run_recolor_segments(
        input_path=Path(args.input),
        segment_manifest_path=Path(args.segment_manifest),
        output_path=Path(args.output),
        color_hex=str(args.color) if args.color else None,
        palette_manifest_path=Path(args.palette_manifest) if args.palette_manifest else None,
        include_labels=list(args.include_label),
        recolor_mode=str(args.recolor_mode),
        chroma_blend=float(args.chroma_blend),
        mask_erode_px=int(args.mask_erode_px),
        mask_feather_px=int(args.mask_feather_px),
        temporal_mask_blend=float(args.temporal_mask_blend),
        overwrite=bool(args.overwrite),
    )


def handle_track_segments(args: argparse.Namespace) -> int:
    return run_track_segments(
        segment_manifest_path=Path(args.segment_manifest),
        output_dir=Path(args.output_dir),
        include_labels=list(args.include_label),
        output_label=str(args.output_label),
        min_area=int(args.min_area),
        iou_threshold=float(args.iou_threshold),
        max_center_distance=float(args.max_center_distance),
        max_missing_frames=int(args.max_missing_frames),
        split_wide_components=bool(args.split_wide_components),
        preserve_source_instances=bool(args.preserve_source_instances),
        max_component_width_ratio=float(args.max_component_width_ratio),
        min_split_valley_ratio=float(args.min_split_valley_ratio),
        overwrite=bool(args.overwrite),
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
