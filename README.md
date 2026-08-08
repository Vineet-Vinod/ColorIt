# ColorIt

ColorIt is a local, automatic full-movie colorization pipeline for black-and-white films.

It is built for the practical workflow: give it a movie file, let it split and colorize the film scene by scene, and get back a compressed color movie with the original audio preserved.

## What It Does

- detects scene boundaries
- applies contrast-aware preprocessing
- colorizes each scene with DeOldify
- colorizes each scene with DDColor
- propagates DDColor chroma over the DeOldify base for temporal stability
- reassembles the full movie
- compresses the final output with size-aware defaults
- reports live progress and ETA for every long-running stage
- cleans intermediate scene clips and manifests after successful runs

ColorIt is a pipeline, not a single new model. The value is in making strong open-source colorizers work on full-length movies with sane defaults and no manual per-scene work.

## Requirements

- Python `>=3.11,<3.12`
- `uv`
- `ffmpeg` and `ffprobe` on `PATH`
- PyTorch-supported hardware

Apple Silicon MPS is used when available, with CPU fallback. CUDA is also supported by the underlying dependencies when available.

A GPU or Desktop machine with sizeable compute is needed to color a movie in a reasonable time. For example, the entire pipeline takes about 6 hours to color a 2.5 hour movie on an M3 ultra Mac Studio.

## Install

```bash
uv sync
```

Fetch the required model weights:

```bash
uv run colorit download-weights
```

This downloads DeOldify to `models/deoldify/ColorizeVideo_gen.pth` and DDColor to `models/ddcolor/pytorch_model.bin`.

## Usage

Colorize a movie:

```bash
uv run colorit colorize-movie --input /path/to/movie.mp4 --overwrite
```

Write to a specific output path:

```bash
uv run colorit colorize-movie \
  --input /path/to/movie.mp4 \
  --output /path/to/movie_color.mp4 \
  --overwrite
```

Resume a failed or interrupted run:

```bash
uv run colorit colorize-movie \
  --input /path/to/movie.mp4 \
  --output /path/to/movie_color.mp4 \
  --resume \
  --overwrite
```

By default, output is written next to the input with `_color` appended to the filename.

Progress is reported automatically. Frame-processing and FFmpeg stages count actual
frames, while the batch ETA is weighted by scene duration. On resumed runs, reusable
scenes begin as completed work and do not distort the remaining-time estimate.

## CLI

The public CLI intentionally exposes only two commands:

```text
colorit download-weights
colorit colorize-movie
```

`colorize-movie` accepts:

```text
--input
--output
--resume
--overwrite
```

Internal frame, clip, model-debug, and configuration flags are not part of the launch CLI.

## Output Size

The final movie is compressed by default. The pipeline retries with higher CRF values if needed so the colorized output stays within the configured size target.

Current launch defaults target at most `1.5x` the source movie size.

## Defaults

The main pipeline uses `configs/full_movie.yaml`.

Important defaults:

- DeOldify `render_factor: 17`
- MPS first, CPU fallback
- rawvideo ffmpeg piping instead of PNG frame round trips
- scene threshold `0.60`
- scene clips encoded with H.264 CRF `16`
- DDColor correction with full-frame chroma propagation at blend `1.0`
- final compression with H.264 CRF `20`, retrying `23`, `26`, and `28`
- successful runs clean intermediate scene clips and manifests

## Current Limits

ColorIt is fully automatic, so it does not ask for reference frames, prompts, masks, or manual actor labels. That is the point, but it also means some hard cases remain:

- costume colors may be conservative rather than vivid
- the same costume can still shift across difficult cuts
- heavy occlusion, fast motion, dances, and fights remain challenging
- source films with poor contrast or damaged transfers can still produce weak color

The pipeline is optimized for full-movie usefulness over perfect frame-by-frame artistic control.

## License

ColorIt is released under the [ColorIt Attribution License 1.0](LICENSE).

You can use, modify, distribute, and sell the software, including for commercial work. If you publicly share or distribute movies, clips, or other audiovisual outputs colorized with ColorIt or a derivative pipeline, include reasonable attribution such as:

```text
Colorized with ColorIt
```

This license only covers ColorIt. You are responsible for having the necessary rights to any movies or media you process.
