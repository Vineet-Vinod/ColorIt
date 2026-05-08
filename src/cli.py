from __future__ import annotations

import argparse
from pathlib import Path
import sys
from collections.abc import Sequence

from src.pipeline.actor_stitch import run_stitch_actors
from src.pipeline.auto_costume_track import run_auto_costume_track
from src.pipeline.colorize_clip import run_colorize_clip
from src.pipeline.config import load_config
from src.pipeline.ddcolor_clip import run_ddcolor_clip
from src.pipeline.ffmpeg_utils import compress_video
from src.pipeline.inference import colorize_image_file
from src.pipeline.manifest_stats import run_manifest_stats
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.model_chroma_propagate import run_model_chroma_propagate
from src.pipeline.movie import run_colorize_movie
from src.pipeline.segment_clip import run_segment_clip
from src.pipeline.segment_debug import run_render_segment_debug
from src.pipeline.segment_filter import run_filter_segments
from src.pipeline.segment_recolor import run_recolor_segments
from src.pipeline.segment_track import run_track_segments
from src.pipeline.weights import run_download_weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colorit",
        description="Automatic movie colorization. Pass a movie path and ColorIt handles scenes, colorization, assembly, compression, and cleanup.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{download-weights,colorize-movie}",
    )

    download_parser = subparsers.add_parser(
        "download-weights",
        help="Download the base DeOldify video checkpoint.",
    )
    download_parser.add_argument("--config", default="configs/default.yaml")
    download_parser.add_argument("--url", default=None)
    download_parser.add_argument("--force", action="store_true")
    download_parser.set_defaults(handler=handle_download_weights)

    default_movie_parser = subparsers.add_parser(
        "__movie",
        prog="colorit",
        help=argparse.SUPPRESS,
    )
    default_movie_parser.add_argument("movie", help="Input movie path.")
    default_movie_parser.add_argument(
        "--output",
        default=None,
        help="Optional final output path. Defaults to the input path with '_color' appended.",
    )
    default_movie_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run if intermediate manifests are still present.",
    )
    default_movie_parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep scene clips and manifests for inspection after a successful run.",
    )
    default_movie_parser.add_argument("--overwrite", action="store_true")
    default_movie_parser.set_defaults(handler=handle_default_movie)

    frame_parser = subparsers.add_parser(
        "colorize-frame",
        help=argparse.SUPPRESS,
    )
    frame_parser.add_argument("--config", default="configs/default.yaml")
    frame_parser.add_argument("--input", required=True)
    frame_parser.add_argument("--output", required=True)
    frame_parser.add_argument("--render-factor", type=int, default=None)
    frame_parser.set_defaults(handler=handle_colorize_frame)

    clip_parser = subparsers.add_parser(
        "colorize-clip",
        help=argparse.SUPPRESS,
    )
    clip_parser.add_argument("--config", default="configs/default.yaml")
    clip_parser.add_argument("--input", required=True)
    clip_parser.add_argument("--output", required=True)
    clip_parser.add_argument("--overwrite", action="store_true")
    clip_parser.set_defaults(handler=handle_colorize_clip)

    ddcolor_parser = subparsers.add_parser(
        "ddcolor-clip",
        help=argparse.SUPPRESS,
    )
    ddcolor_parser.add_argument("--input", required=True)
    ddcolor_parser.add_argument("--output", required=True)
    ddcolor_parser.add_argument("--ddcolor-repo", required=True)
    ddcolor_parser.add_argument("--weights", required=True)
    ddcolor_parser.add_argument("--input-size", type=int, default=512)
    ddcolor_parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    ddcolor_parser.add_argument("--output-preset", default="medium")
    ddcolor_parser.add_argument("--overwrite", action="store_true")
    ddcolor_parser.set_defaults(handler=handle_ddcolor_clip)

    chroma_propagate_parser = subparsers.add_parser(
        "model-chroma-propagate",
        help=argparse.SUPPRESS,
    )
    chroma_propagate_parser.add_argument("--source", required=True, help="Base colorized or grayscale source clip.")
    chroma_propagate_parser.add_argument("--model-color", required=True, help="Model-colorized clip to sample keyframes from.")
    chroma_propagate_parser.add_argument("--output", required=True)
    chroma_propagate_parser.add_argument("--keyframe-stride", type=int, default=35)
    chroma_propagate_parser.add_argument(
        "--propagation-mode",
        choices=("flow", "model"),
        default="flow",
        help="Use optical-flow keyframe propagation or the model-color clip chroma directly.",
    )
    chroma_propagate_parser.add_argument("--output-preset", default="medium")
    chroma_propagate_parser.add_argument("--chroma-blend", type=float, default=0.80)
    chroma_propagate_parser.add_argument(
        "--fallback-color",
        default=None,
        help="Optional #RRGGBB chroma to use where forward/backward propagation disagrees.",
    )
    chroma_propagate_parser.add_argument("--fallback-strength", type=float, default=0.85)
    chroma_propagate_parser.add_argument(
        "--fallback-uncertainty",
        choices=("hue", "ab-delta"),
        default="hue",
        help="Uncertainty signal used for fallback-color blending.",
    )
    chroma_propagate_parser.add_argument("--disagreement-start", type=float, default=20.0)
    chroma_propagate_parser.add_argument("--disagreement-end", type=float, default=70.0)
    chroma_propagate_parser.add_argument(
        "--scene-cut-threshold",
        type=float,
        default=0.0,
        help="Mean grayscale frame-delta threshold for adding cut-local model keyframes. Disabled by default.",
    )
    chroma_propagate_parser.add_argument(
        "--scene-keyframe-window",
        type=int,
        default=2,
        help="Number of neighboring frames around each detected cut to force as model keyframes.",
    )
    chroma_propagate_parser.add_argument(
        "--chroma-smooth-diameter",
        type=int,
        default=0,
        help="Bilateral filter diameter for propagated Lab chroma. Disabled by default.",
    )
    chroma_propagate_parser.add_argument("--chroma-smooth-sigma-color", type=float, default=16.0)
    chroma_propagate_parser.add_argument("--chroma-smooth-sigma-space", type=float, default=7.0)
    chroma_propagate_parser.add_argument(
        "--dark-fill-strength",
        type=float,
        default=0.0,
        help="Borrow nearby propagated chroma into dark low-chroma regions. Disabled by default.",
    )
    chroma_propagate_parser.add_argument("--dark-fill-luma-end", type=float, default=92.0)
    chroma_propagate_parser.add_argument("--dark-fill-chroma-end", type=float, default=22.0)
    chroma_propagate_parser.add_argument("--dark-fill-sigma", type=float, default=8.0)
    chroma_propagate_parser.add_argument(
        "--model-fill-strength",
        type=float,
        default=0.0,
        help="Use current-frame model chroma to repair weak or uncertain propagated chroma. Disabled by default.",
    )
    chroma_propagate_parser.add_argument("--model-fill-chroma-end", type=float, default=28.0)
    chroma_propagate_parser.add_argument("--model-fill-disagreement-start", type=float, default=25.0)
    chroma_propagate_parser.add_argument("--model-fill-disagreement-end", type=float, default=80.0)
    chroma_propagate_parser.add_argument("--model-fill-blur-sigma", type=float, default=1.5)
    chroma_propagate_parser.add_argument(
        "--model-fill-chroma-floor",
        type=float,
        default=0.0,
        help="Minimum Lab chroma magnitude for model-fill repair. Disabled by default.",
    )
    chroma_propagate_parser.add_argument(
        "--component-fill-strength",
        type=float,
        default=0.0,
        help="Force connected dark regions toward their median current-frame model chroma. Disabled by default.",
    )
    chroma_propagate_parser.add_argument("--component-fill-luma-end", type=float, default=100.0)
    chroma_propagate_parser.add_argument("--component-fill-min-area", type=int, default=1800)
    chroma_propagate_parser.add_argument("--component-fill-model-chroma-min", type=float, default=14.0)
    chroma_propagate_parser.add_argument(
        "--blue-suppress-strength",
        type=float,
        default=0.0,
        help="Blend blue/cyan-biased chroma toward fallback-color. Requires --fallback-color.",
    )
    chroma_propagate_parser.add_argument("--blue-suppress-hue-start", type=float, default=85.0)
    chroma_propagate_parser.add_argument("--blue-suppress-hue-end", type=float, default=132.0)
    chroma_propagate_parser.add_argument(
        "--semantic-consensus-manifest",
        default=None,
        help="Optional human-parser segment manifest used for garment-region chroma consensus.",
    )
    chroma_propagate_parser.add_argument(
        "--semantic-consensus-label",
        action="append",
        default=[],
        help="Semantic garment label to use for consensus. Can be passed multiple times.",
    )
    chroma_propagate_parser.add_argument(
        "--semantic-protect-label",
        action="append",
        default=[],
        help="Semantic label to subtract from garment consensus masks. Can be passed multiple times.",
    )
    chroma_propagate_parser.add_argument("--semantic-protect-dilate", type=int, default=2)
    chroma_propagate_parser.add_argument(
        "--semantic-split-label",
        action="append",
        default=[],
        help="Semantic label whose component centroids split merged garment masks. Can be passed multiple times.",
    )
    chroma_propagate_parser.add_argument("--semantic-consensus-strength", type=float, default=0.0)
    chroma_propagate_parser.add_argument("--semantic-consensus-min-area", type=int, default=700)
    chroma_propagate_parser.add_argument("--semantic-consensus-model-chroma-min", type=float, default=10.0)
    chroma_propagate_parser.add_argument("--semantic-consensus-feather-sigma", type=float, default=1.2)
    chroma_propagate_parser.add_argument("--semantic-consensus-diversify-strength", type=float, default=0.0)
    chroma_propagate_parser.add_argument("--semantic-consensus-diversify-threshold", type=float, default=12.0)
    chroma_propagate_parser.add_argument("--overwrite", action="store_true")
    chroma_propagate_parser.set_defaults(handler=handle_model_chroma_propagate)

    movie_parser = subparsers.add_parser(
        "colorize-movie",
        help="Advanced full-movie command with tuning flags.",
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
        help=argparse.SUPPRESS,
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
        help=argparse.SUPPRESS,
    )
    segment_parser.add_argument("--input", required=True, help="Input clip path.")
    segment_parser.add_argument(
        "--backend",
        default="polygon",
        choices=("polygon", "human-parser", "person-maskrcnn"),
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
    segment_parser.add_argument("--score-threshold", type=float, default=0.70)
    segment_parser.add_argument("--mask-threshold", type=float, default=0.50)
    segment_parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="For human-parser, run model inference every N frames and reuse the last mask between samples.",
    )
    segment_parser.add_argument("--min-area", type=int, default=3000)
    segment_parser.add_argument("--iou-threshold", type=float, default=0.10)
    segment_parser.add_argument("--max-center-distance", type=float, default=260.0)
    segment_parser.add_argument("--max-missing-frames", type=int, default=8)
    segment_parser.add_argument("--overwrite", action="store_true")
    segment_parser.set_defaults(handler=handle_segment_clip)

    segment_debug_parser = subparsers.add_parser(
        "render-segment-debug",
        help=argparse.SUPPRESS,
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
        help=argparse.SUPPRESS,
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
    segment_filter_parser.add_argument(
        "--track-split-guides",
        action="store_true",
        help="Track split guide anchors over time and use those stable ids in output track ids.",
    )
    segment_filter_parser.add_argument("--guide-track-max-distance", type=float, default=140.0)
    segment_filter_parser.add_argument("--guide-track-max-missing", type=int, default=6)
    segment_filter_parser.add_argument("--close-px", type=int, default=0)
    segment_filter_parser.add_argument("--erode-px", type=int, default=0)
    segment_filter_parser.add_argument("--dilate-px", type=int, default=0)
    segment_filter_parser.add_argument("--veto-dilate-px", type=int, default=0)
    segment_filter_parser.add_argument("--overwrite", action="store_true")
    segment_filter_parser.set_defaults(handler=handle_filter_segments)

    segment_recolor_parser = subparsers.add_parser(
        "recolor-segments",
        help=argparse.SUPPRESS,
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
        "--protect-segment-manifest",
        default=None,
        help="Optional segment manifest containing labels to subtract from recolor masks.",
    )
    segment_recolor_parser.add_argument(
        "--protect-label",
        action="append",
        default=[],
        help="Label from --protect-segment-manifest to avoid. Can be passed multiple times.",
    )
    segment_recolor_parser.add_argument("--protect-dilate-px", type=int, default=0)
    segment_recolor_parser.add_argument("--protect-feather-px", type=int, default=0)
    segment_recolor_parser.add_argument(
        "--protect-skin-tones",
        action="store_true",
        help="Suppress recolor on skin-like chroma regions in the input colorized frame.",
    )
    segment_recolor_parser.add_argument(
        "--protect-skin-track",
        action="append",
        default=[],
        help="Apply --protect-skin-tones only to this track id. Defaults to all recolored tracks.",
    )
    segment_recolor_parser.add_argument("--skin-protect-dilate-px", type=int, default=0)
    segment_recolor_parser.add_argument("--skin-protect-feather-px", type=int, default=0)
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
    segment_recolor_parser.add_argument(
        "--temporal-flow-blend",
        type=float,
        default=0.0,
        help="Blend current masks with previous masks warped by optical flow.",
    )
    segment_recolor_parser.add_argument(
        "--temporal-flow-fill",
        action="store_true",
        help="Use warped previous masks to fill holes instead of averaging them into current masks.",
    )
    segment_recolor_parser.add_argument(
        "--temporal-carry-frames",
        type=int,
        default=0,
        help="Carry missing tracks forward for this many frames using optical-flow-warped alpha.",
    )
    segment_recolor_parser.add_argument("--temporal-carry-decay", type=float, default=0.72)
    segment_recolor_parser.add_argument("--overwrite", action="store_true")
    segment_recolor_parser.set_defaults(handler=handle_recolor_segments)

    segment_track_parser = subparsers.add_parser(
        "track-segments",
        help=argparse.SUPPRESS,
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

    auto_costume_parser = subparsers.add_parser(
        "auto-costume-track",
        help=argparse.SUPPRESS,
    )
    auto_costume_parser.add_argument(
        "--human-parser-manifest",
        required=True,
        help="Human parser segment manifest JSON path.",
    )
    auto_costume_parser.add_argument(
        "--actor-manifest",
        default=None,
        help="Optional actor segment manifest. If set, costume proposals are intersected with actor masks.",
    )
    auto_costume_parser.add_argument("--output-dir", required=True, help="Output segment artifact directory.")
    auto_costume_parser.add_argument(
        "--actor-guide-label",
        action="append",
        default=[],
        help="Label used to auto-detect actor anchors. Defaults to face and hair.",
    )
    auto_costume_parser.add_argument(
        "--clothing-label",
        action="append",
        default=[],
        help="Clothing label to turn into actor-scoped costume proposals.",
    )
    auto_costume_parser.add_argument(
        "--skin-label",
        action="append",
        default=[],
        help="Label to subtract from costume proposals.",
    )
    auto_costume_parser.add_argument("--min-mask-area", type=int, default=800)
    auto_costume_parser.add_argument("--min-confidence", type=float, default=0.18)
    auto_costume_parser.add_argument("--min-track-frames", type=int, default=2)
    auto_costume_parser.add_argument("--guide-min-area", type=int, default=120)
    auto_costume_parser.add_argument("--guide-merge-distance", type=float, default=95.0)
    auto_costume_parser.add_argument("--actor-max-distance", type=float, default=150.0)
    auto_costume_parser.add_argument("--actor-max-missing", type=int, default=6)
    auto_costume_parser.add_argument("--skin-dilate-px", type=int, default=5)
    auto_costume_parser.add_argument("--actor-prior-dilate-px", type=int, default=18)
    auto_costume_parser.add_argument("--close-px", type=int, default=3)
    auto_costume_parser.add_argument("--erode-px", type=int, default=1)
    auto_costume_parser.add_argument("--dilate-px", type=int, default=0)
    auto_costume_parser.add_argument("--overwrite", action="store_true")
    auto_costume_parser.set_defaults(handler=handle_auto_costume_track)

    stitch_actor_parser = subparsers.add_parser(
        "stitch-actors",
        help=argparse.SUPPRESS,
    )
    stitch_actor_parser.add_argument("--actor-manifest", required=True, help="Source actor segment manifest.")
    stitch_actor_parser.add_argument("--output-dir", required=True, help="Output stitched actor directory.")
    stitch_actor_parser.add_argument("--min-source-frames", type=int, default=5)
    stitch_actor_parser.add_argument("--max-gap-frames", type=int, default=35)
    stitch_actor_parser.add_argument("--max-centroid-distance", type=float, default=360.0)
    stitch_actor_parser.add_argument("--min-iou", type=float, default=0.02)
    stitch_actor_parser.add_argument("--overlap-merge-iou", type=float, default=0.16)
    stitch_actor_parser.add_argument("--allow-overlap-frames", type=int, default=2)
    stitch_actor_parser.add_argument("--overwrite", action="store_true")
    stitch_actor_parser.set_defaults(handler=handle_stitch_actors)

    manifest_stats_parser = subparsers.add_parser(
        "manifest-stats",
        help=argparse.SUPPRESS,
    )
    manifest_stats_parser.add_argument("--segment-manifest", required=True)
    manifest_stats_parser.add_argument("--palette-manifest", default=None)
    manifest_stats_parser.add_argument("--output", default=None)
    manifest_stats_parser.set_defaults(handler=handle_manifest_stats)

    visible_commands = {"download-weights", "colorize-movie"}
    subparsers._choices_actions = [
        action for action in subparsers._choices_actions if action.dest in visible_commands
    ]

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


def handle_ddcolor_clip(args: argparse.Namespace) -> int:
    return run_ddcolor_clip(
        input_path=Path(args.input),
        output_path=Path(args.output),
        ddcolor_repo_path=Path(args.ddcolor_repo),
        weights_path=Path(args.weights),
        input_size=int(args.input_size),
        device=str(args.device),
        output_preset=str(args.output_preset),
        overwrite=bool(args.overwrite),
    )


def handle_model_chroma_propagate(args: argparse.Namespace) -> int:
    return run_model_chroma_propagate(
        source_path=Path(args.source),
        model_color_path=Path(args.model_color),
        output_path=Path(args.output),
        keyframe_stride=int(args.keyframe_stride),
        propagation_mode=str(args.propagation_mode),
        output_preset=str(args.output_preset),
        chroma_blend=float(args.chroma_blend),
        fallback_color_hex=str(args.fallback_color) if args.fallback_color else None,
        fallback_strength=float(args.fallback_strength),
        fallback_uncertainty=str(args.fallback_uncertainty),
        disagreement_start=float(args.disagreement_start),
        disagreement_end=float(args.disagreement_end),
        scene_cut_threshold=float(args.scene_cut_threshold),
        scene_keyframe_window=int(args.scene_keyframe_window),
        chroma_smooth_diameter=int(args.chroma_smooth_diameter),
        chroma_smooth_sigma_color=float(args.chroma_smooth_sigma_color),
        chroma_smooth_sigma_space=float(args.chroma_smooth_sigma_space),
        dark_fill_strength=float(args.dark_fill_strength),
        dark_fill_luma_end=float(args.dark_fill_luma_end),
        dark_fill_chroma_end=float(args.dark_fill_chroma_end),
        dark_fill_sigma=float(args.dark_fill_sigma),
        model_fill_strength=float(args.model_fill_strength),
        model_fill_chroma_end=float(args.model_fill_chroma_end),
        model_fill_disagreement_start=float(args.model_fill_disagreement_start),
        model_fill_disagreement_end=float(args.model_fill_disagreement_end),
        model_fill_blur_sigma=float(args.model_fill_blur_sigma),
        model_fill_chroma_floor=float(args.model_fill_chroma_floor),
        component_fill_strength=float(args.component_fill_strength),
        component_fill_luma_end=float(args.component_fill_luma_end),
        component_fill_min_area=int(args.component_fill_min_area),
        component_fill_model_chroma_min=float(args.component_fill_model_chroma_min),
        blue_suppress_strength=float(args.blue_suppress_strength),
        blue_suppress_hue_start=float(args.blue_suppress_hue_start),
        blue_suppress_hue_end=float(args.blue_suppress_hue_end),
        semantic_consensus_manifest_path=Path(args.semantic_consensus_manifest)
        if args.semantic_consensus_manifest
        else None,
        semantic_consensus_labels=[str(label) for label in args.semantic_consensus_label],
        semantic_protect_labels=[str(label) for label in args.semantic_protect_label],
        semantic_protect_dilate=int(args.semantic_protect_dilate),
        semantic_split_labels=[str(label) for label in args.semantic_split_label],
        semantic_consensus_strength=float(args.semantic_consensus_strength),
        semantic_consensus_min_area=int(args.semantic_consensus_min_area),
        semantic_consensus_model_chroma_min=float(args.semantic_consensus_model_chroma_min),
        semantic_consensus_feather_sigma=float(args.semantic_consensus_feather_sigma),
        semantic_consensus_diversify_strength=float(args.semantic_consensus_diversify_strength),
        semantic_consensus_diversify_threshold=float(args.semantic_consensus_diversify_threshold),
        overwrite=bool(args.overwrite),
    )


def handle_default_movie(args: argparse.Namespace) -> int:
    config_path = Path("configs/full_movie.yaml")
    config = load_config(config_path)
    return run_colorize_movie(
        config=config,
        config_path=config_path,
        movie_path=Path(args.movie),
        output_path=Path(args.output) if args.output else None,
        scene_threshold=None,
        keep_intermediates=bool(args.keep_intermediates),
        resume=bool(args.resume),
        limit=None,
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
        frame_stride=int(args.frame_stride),
        score_threshold=float(args.score_threshold),
        mask_threshold=float(args.mask_threshold),
        min_area=int(args.min_area),
        iou_threshold=float(args.iou_threshold),
        max_center_distance=float(args.max_center_distance),
        max_missing_frames=int(args.max_missing_frames),
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
        track_split_guides=bool(args.track_split_guides),
        guide_track_max_distance=float(args.guide_track_max_distance),
        guide_track_max_missing=int(args.guide_track_max_missing),
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
        protect_segment_manifest_path=(
            Path(args.protect_segment_manifest) if args.protect_segment_manifest else None
        ),
        protect_labels=list(args.protect_label),
        protect_dilate_px=int(args.protect_dilate_px),
        protect_feather_px=int(args.protect_feather_px),
        protect_skin_tones=bool(args.protect_skin_tones),
        protect_skin_tracks=list(args.protect_skin_track),
        skin_protect_dilate_px=int(args.skin_protect_dilate_px),
        skin_protect_feather_px=int(args.skin_protect_feather_px),
        include_labels=list(args.include_label),
        recolor_mode=str(args.recolor_mode),
        chroma_blend=float(args.chroma_blend),
        mask_erode_px=int(args.mask_erode_px),
        mask_feather_px=int(args.mask_feather_px),
        temporal_mask_blend=float(args.temporal_mask_blend),
        temporal_flow_blend=float(args.temporal_flow_blend),
        temporal_flow_fill=bool(args.temporal_flow_fill),
        temporal_carry_frames=int(args.temporal_carry_frames),
        temporal_carry_decay=float(args.temporal_carry_decay),
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


def handle_auto_costume_track(args: argparse.Namespace) -> int:
    return run_auto_costume_track(
        human_parser_manifest_path=Path(args.human_parser_manifest),
        actor_manifest_path=Path(args.actor_manifest) if args.actor_manifest else None,
        output_dir=Path(args.output_dir),
        actor_guide_labels=list(args.actor_guide_label),
        clothing_labels=list(args.clothing_label),
        skin_labels=list(args.skin_label),
        min_mask_area=int(args.min_mask_area),
        min_confidence=float(args.min_confidence),
        min_track_frames=int(args.min_track_frames),
        guide_min_area=int(args.guide_min_area),
        guide_merge_distance=float(args.guide_merge_distance),
        actor_max_distance=float(args.actor_max_distance),
        actor_max_missing=int(args.actor_max_missing),
        skin_dilate_px=int(args.skin_dilate_px),
        actor_prior_dilate_px=int(args.actor_prior_dilate_px),
        close_px=int(args.close_px),
        erode_px=int(args.erode_px),
        dilate_px=int(args.dilate_px),
        overwrite=bool(args.overwrite),
    )


def handle_stitch_actors(args: argparse.Namespace) -> int:
    return run_stitch_actors(
        actor_manifest_path=Path(args.actor_manifest),
        output_dir=Path(args.output_dir),
        min_source_frames=int(args.min_source_frames),
        max_gap_frames=int(args.max_gap_frames),
        max_centroid_distance=float(args.max_centroid_distance),
        min_iou=float(args.min_iou),
        overlap_merge_iou=float(args.overlap_merge_iou),
        allow_overlap_frames=int(args.allow_overlap_frames),
        overwrite=bool(args.overwrite),
    )


def handle_manifest_stats(args: argparse.Namespace) -> int:
    return run_manifest_stats(
        segment_manifest_path=Path(args.segment_manifest),
        palette_manifest_path=Path(args.palette_manifest) if args.palette_manifest else None,
        output_path=Path(args.output) if args.output else None,
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
    if value.startswith("-"):
        return False
    if value in {
        "download-weights",
        "colorize-frame",
        "colorize-clip",
        "ddcolor-clip",
        "model-chroma-propagate",
        "colorize-movie",
        "compress-video",
        "segment-clip",
        "render-segment-debug",
        "filter-segments",
        "recolor-segments",
        "track-segments",
        "auto-costume-track",
        "stitch-actors",
        "manifest-stats",
    }:
        return False

    expanded = Path(value).expanduser()
    video_suffixes = {
        ".avi",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".webm",
    }
    return expanded.exists() or expanded.suffix.lower() in video_suffixes or "/" in value


if __name__ == "__main__":
    raise SystemExit(main())
