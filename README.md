# ColorIt

Automatic movie-first colorization for black-and-white film.

The production path is intentionally small:

- detect scene boundaries
- colorize each scene with one reused DeOldify model load
- colorize the same scene with DDColor
- propagate DDColor chroma over the DeOldify base with optical flow
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
uv run colorit colorize-movie --input /path/to/movie.mp4 --overwrite
```

By default this writes `/path/to/movie_color.mp4`. The output is compressed as
part of the run, with retries at higher CRF values if the result exceeds the
configured `2x` source-size target.

Useful production flags:

```bash
uv run colorit colorize-movie \
  --input /path/to/movie.mp4 \
  --output /path/to/movie_color.mp4 \
  --resume \
  --overwrite
```

Normal successful runs delete generated scene clips, colorized scene clips, and
run manifests automatically.

## Defaults

The main pipeline uses `configs/full_movie.yaml`.

Important defaults:

- `render_factor: 17`
- MPS first, CPU fallback
- rawvideo ffmpeg piping instead of PNG frame round trips
- scene threshold `0.60`
- scene clips encoded with H.264 CRF `16`
- DDColor correction with full-frame chroma propagation at blend `1.0`
- final compression enabled with H.264 CRF `20`, retrying `23`, `26`, and `28`
- final size target `<= 2x` the source movie
