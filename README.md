# ColorIt

Movie-first DeOldify pipeline for black-and-white film colorization.

The repo is centered on one practical path:

- split a movie into scenes
- colorize each scene with the optimized DeOldify clip pipeline
- reassemble a final movie next to the source with `_color` appended to the filename

## Commands

- `uv run colorit download-weights`
- `uv run colorit colorize-frame`
- `uv run colorit colorize-clip`
- `uv run colorit compress-video`
- `uv run colorit colorize-movie`

## Quick Start

Install dependencies and fetch weights:

```bash
uv sync
uv run colorit download-weights
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

Create a compressed derivative when you need one:

```bash
uv run colorit compress-video \
  --input /path/to/movie_color.mp4 \
  --overwrite
```

By default this writes `/path/to/movie_color_compressed.mp4`.

To keep temporary scene clips and manifests for inspection:

```bash
uv run colorit colorize-movie \
  --input /path/to/movie.mp4 \
  --keep-intermediates \
  --overwrite
```

To resume an interrupted movie run from the saved scene and batch manifests:

```bash
uv run colorit colorize-movie \
  --input /path/to/movie.mp4 \
  --keep-intermediates \
  --resume \
  --overwrite
```

## Configs

- `configs/default.yaml`: base config for single-frame and single-clip work
- `configs/full_movie.yaml`: optimized full-movie pipeline

The optimized path uses:

- ffmpeg rawvideo piping instead of PNG frame round-trips
- batched inference on MPS when available, with CPU fallback
- one model load reused across the whole batch pass
- scene, batch, and top-level movie manifests for deterministic resume
- optional scene artifact cleanup after successful assembly

## Segmentation Manifests

The `segment-clip` command creates reusable per-frame mask manifests for
downstream experiments such as costume recoloring, actor-only postprocesses, and
region-specific temporal smoothing.

```bash
uv run colorit segment-clip \
  --input data/eval/clips/multiple.mp4 \
  --backend polygon \
  --tracks data/eval/tracks/multiple_tracks.json \
  --output-dir data/segments/multiple \
  --overwrite
```

The first backend is `polygon`, a model-free debug backend that interpolates
track keyframes. It is useful for validating downstream consumers before adding
model-backed human parsing or video segmentation.

Segment manifests are written as:

```text
data/segments/<clip>/
  segment_manifest.json
  masks/
    frame_000000_track_0001.png
```

Each manifest records clip metadata, backend name, track metadata, frame-local
instances, mask paths, bounding boxes, labels, and confidence values.
