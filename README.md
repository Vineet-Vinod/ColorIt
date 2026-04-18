# ColorIt

Movie-first DeOldify pipeline for black-and-white film colorization.

The repo is centered on one practical path:

- split a movie into scenes
- colorize each scene with the optimized DeOldify clip pipeline
- reassemble a final movie next to the source with `_color` appended to the filename

## Commands

- `uv run colorit download-weights`
- `uv run colorit verify-env`
- `uv run colorit colorize-frame`
- `uv run colorit colorize-clip`
- `uv run colorit colorize-movie`

## Quick Start

Install dependencies and fetch weights:

```bash
uv sync
uv run colorit download-weights
```

Verify the environment:

```bash
uv run colorit verify-env --skip-inference
uv run colorit verify-env --test-image path/to/frame.png
```

Colorize an entire movie:

```bash
uv run colorit colorize-movie \
  --input /path/to/movie.mp4 \
  --overwrite
```

By default this writes:

- input: `/path/to/movie.mp4`
- output: `/path/to/movie_color.mp4`

To keep temporary scene clips and manifests for inspection:

```bash
uv run colorit colorize-movie \
  --input /path/to/movie.mp4 \
  --keep-intermediates \
  --overwrite
```

## Configs

- `configs/default.yaml`: base config for single-frame and single-clip work
- `configs/full_movie.yaml`: optimized full-movie pipeline

The optimized path uses:

- ffmpeg rawvideo piping instead of PNG frame round-trips
- batched inference on MPS when available, with CPU fallback
- one model load reused across the whole batch pass
- optional scene artifact cleanup after successful assembly
