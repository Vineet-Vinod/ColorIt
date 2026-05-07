# ColorIt

Automatic movie-first DeOldify pipeline for black-and-white film colorization.

The production path is intentionally small:

- detect scene boundaries
- colorize each scene with one reused DeOldify model load
- reassemble the full movie
- compress the final output with size-aware defaults
- clean intermediate scene clips and manifests after a successful run

## Quick Start

Install dependencies and fetch weights once:

```bash
uv sync
uv run colorit download-weights
```

Colorize a movie:

```bash
uv run colorit /path/to/movie.mp4 --overwrite
```

The explicit form is equivalent:

```bash
uv run colorit run /path/to/movie.mp4 --overwrite
```

By default this writes `/path/to/movie_color.mp4`. The output is compressed as
part of the run, with retries at higher CRF values if the result exceeds the
configured `2x` source-size target.

Useful production flags:

```bash
uv run colorit run /path/to/movie.mp4 \
  --output /path/to/movie_color.mp4 \
  --resume \
  --overwrite
```

Use `--keep-intermediates` only when you need to inspect a failed or suspicious
run. Normal successful runs delete generated scene clips, colorized scene clips,
and run manifests automatically.

## Defaults

The main pipeline uses `configs/full_movie.yaml`.

Important defaults:

- `render_factor: 17`
- MPS first, CPU fallback
- rawvideo ffmpeg piping instead of PNG frame round trips
- scene threshold `0.60`
- scene clips encoded with H.264 CRF `16`
- final compression enabled with H.264 CRF `20`, retrying `23`, `26`, and `28`
- final size target `<= 2x` the source movie

Most older segmentation and costume-track commands are still available for
research, but they are intentionally hidden from the default help output. The
default user-facing path should stay `colorit run <movie>`.
