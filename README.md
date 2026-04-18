# ColorIt

CLI-first pipeline scaffold for colorizing black-and-white Kannada films with DeOldify.

The current repository state is intentionally conservative:

- Python tooling is pinned to `3.11`
- the project layout matches the execution plan in [plan.md](/Users/darksca/ColorIt/plan.md)
- model weights can be downloaded and checksummed
- local inference uses a restricted checkpoint load path
- the checkpoint is not loaded via `weights_only=False`

## Environment setup

This phase avoids running untrusted model code. The only supported bootstrap tasks right now are:

- validating local prerequisites
- creating the expected working layout
- downloading `ColorizeVideo_gen.pth`
- recording file metadata and SHA-256
- verifying restricted model loading and single-frame inference

### Prerequisites

- `uv`
- Python `3.11`
- `ffmpeg`
- enough disk space for the model weights and generated media

### Bootstrap commands

```bash
uv run colorit verify-env
uv run colorit download-weights
uv run colorit colorize-frame --input input.png --output output.png
uv run colorit extract-probes --movie ~/Movies/Kannada/'emme thammanna.mp4' --clip clip_01=00:10:00-00:10:15
uv run colorit colorize-clip --input data/probe_clips/clip_01.mp4 --output data/colorized/probes/clip_01_quality.mp4 --config configs/quality.yaml
uv run colorit colorize-clip --input data/probe_clips/clip_01.mp4 --output data/colorized/probes/clip_01_quality_warm.mp4 --config configs/quality_warm.yaml
uv run colorit detect-scenes --movie ~/Movies/Kannada/'emme thammanna.mp4' --output data/manifests/scenes_t060.json --threshold 0.60
uv run colorit colorize-batch --movie ~/Movies/Kannada/'emme thammanna.mp4' --scene-manifest data/manifests/scenes_t060.json --config configs/full_movie.yaml --resume
uv run colorit assemble-final --scene-manifest data/manifests/scenes_t060.json --config configs/full_movie.yaml
uv run colorit compress-final --config configs/full_movie.yaml --input data/final/emme_thammanna_colorized_v1.mp4
uv run colorit benchmark-clips --config configs/quality.yaml --input data/probe_clips/clip_02.mp4 --input data/probe_clips/clip_04.mp4
```

By default, `download-weights` fetches:

- source: `spensercai/DeOldify`
- file: `ColorizeVideo_gen.pth`
- destination: `models/deoldify/ColorizeVideo_gen.pth`

It also writes a metadata manifest to `data/manifests/weights.json`.

### Probe extraction

Phase 1 extraction is available now.

Example with inline clip specs:

```bash
uv run colorit extract-probes \
  --movie ~/Movies/Kannada/'emme thammanna.mp4' \
  --clip clip_01=00:10:00-00:10:15 \
  --clip 'clip_02=00:15:00-00:15:12|dialogue_scene'
```

You can also provide a YAML or JSON file:

```yaml
- clip_id: clip_01
  start_time: 00:10:00
  end_time: 00:10:15
  notes: close_up_face
- clip_id: clip_02
  start_time: 00:15:00
  end_time: 00:15:12
  notes: dialogue_scene
```

Run it with:

```bash
uv run colorit extract-probes \
  --movie ~/Movies/Kannada/'emme thammanna.mp4' \
  --clip-file clips.yaml
```

Output clips are written to `data/probe_clips/` and the manifest is written to `data/manifests/probe_clips.json`.

### Baseline clip colorization

Phase 2 baseline colorization is available for individual clips.

```bash
uv run colorit colorize-clip \
  --input data/probe_clips/clip_01.mp4 \
  --output data/colorized/probes/clip_01_quality.mp4 \
  --config configs/quality.yaml \
  --overwrite
```

Each run appends metadata to `data/manifests/probe_runs.json` unless you override the manifest path.

The repository also includes warmed comparison presets:

- `configs/quality_warm.yaml`
- `configs/speed_warm.yaml`
- `configs/quality_warm_smooth.yaml`
- `configs/quality_aggressive.yaml`
- `configs/quality_aggressive_smooth.yaml`

These apply warm-bias correction, and `quality_warm_smooth.yaml` also adds lightweight temporal chroma smoothing for flicker reduction.

### Costume Palette Guidance

There is now an optional post-colorization stage for the specific failure mode where costumes stay faded, green/cyan, or washed out even when the rest of the frame is acceptable.

The stage is intentionally narrow:

- it builds a conservative actor/costume mask
- it extracts a small vivid palette from reference images or explicit colors
- it remaps only costume chroma while preserving scene luminance and cloth shading

This is configured under `postprocess.costume_palette`:

```yaml
postprocess:
  temporal_smoothing: false
  smoothing_strength: 0.0
  smoothing_chroma_threshold: 24.0
  adaptive_smoothing_boost: 0.0
  warmth: 0.0
  shadow_warmth: 0.0
  blue_reduction: 0.0
  costume_palette:
    enabled: true
    mask_backend: maskrcnn_v2_conservative
    reference_images:
      - data/references/era_palette_01.png
      - data/references/era_palette_02.png
    palette_colors:
      - "#7a1028"
      - "#8a1e4b"
      - "#a56b16"
    palette_size: 5
    strength: 0.72
    neutral_boost: 0.18
    min_chroma: 26.0
    warm_bias: 0.10
```

Use `reference_images` when you have similar-era color films. Use `palette_colors` when you know a costume family should be pushed toward specific hues.

### Costume Hint Recolor

For frame-level costume correction, the repo also includes a localized hint-propagation experiment:

```bash
uv run python scripts/render_costume_hint_stills.py \
  --movie ~/Movies/Kannada/'emme thammanna.mp4' \
  --timestamp 00:58:28 \
  --config configs/quality.yaml \
  --output-root data/experiments/costume_hint_stills \
  --palette-color '#7a1028' \
  --palette-color '#8a1e4b' \
  --palette-color '#a56b16' \
  --device mps \
  --overwrite
```

This path keeps the DeOldify base frame, derives a conservative costume mask, places a small number of vivid seed colors inside the costume, and propagates them with edge-aware luminance guidance so the recolor stays localized to the garment.

## Current Baseline Decision

The current v1 baseline remains `configs/quality.yaml`.

Probe review findings:

- the base quality preset is the best watchable tradeoff so far
- warm/aggressive variants can reduce some cool bias, but they do not reliably fix model-level blue patches
- fighting scenes still show color inconsistency in high motion
- outdoor scenes flicker more than indoor scenes

Known v1 limitations:

- cool/blue patches can appear on skin and clothing
- fast motion can cause color flips between adjacent frames
- outdoor shots are less temporally stable than indoor shots

## Overnight Movie Pass

The current production path is:

```bash
uv run colorit detect-scenes \
  --movie ~/Movies/Kannada/'emme thammanna.mp4' \
  --output data/manifests/scenes_t060.json \
  --threshold 0.60

uv run colorit colorize-batch \
  --movie ~/Movies/Kannada/'emme thammanna.mp4' \
  --scene-manifest data/manifests/scenes_t060.json \
  --config configs/full_movie.yaml \
  --resume
```

Validated state:

- `threshold 0.60` produced `412` scene units on the film
- scene outputs are written to `data/colorized/scenes/scenes_t060/`
- run manifests are written to:
  - `data/manifests/full_run_scenes_t060.json`
  - `data/manifests/scene_runs_scenes_t060.json`
- `--resume` skips completed scene outputs cleanly
- temporary extracted frame directories are cleaned up after successful clip renders

After the batch finishes, assemble the final movie:

```bash
uv run colorit assemble-final \
  --scene-manifest data/manifests/scenes_t060.json \
  --config configs/full_movie.yaml \
  --output data/final/emme_thammanna_colorized_v1.mp4
```

Then generate the locked review copy:

```bash
uv run colorit compress-final \
  --config configs/full_movie.yaml \
  --input data/final/emme_thammanna_colorized_v1.mp4
```

The current locked review profile is:

- video codec: `libx264`
- preset: `slow`
- `CRF 22`
- audio: `AAC 128k`
- `+faststart`

By default this writes:

- input: `data/final/emme_thammanna_colorized_v1.mp4`
- output: `data/final/emme_thammanna_colorized_v1_crf22_slow.mp4`
- manifest: `data/manifests/compression_emme_thammanna_colorized_v1.json`

## Performance Benchmarking

Benchmark the current clip pipeline with:

```bash
uv run colorit benchmark-clips \
  --config configs/quality.yaml \
  --input data/probe_clips/clip_02.mp4 \
  --input data/probe_clips/clip_04.mp4 \
  --input data/probe_clips/clip_06.mp4 \
  --input data/probe_clips/clip_10.mp4 \
  --overwrite
```

The benchmark manifest is written to `data/manifests/benchmark_runs.json`.

Current baseline findings on representative probe clips:

- the old PNG-frame path delivered about `8 fps` on `mps`
- the streamed `pipe` path delivers about `16 fps` on the same representative clips
- the old PNG path spent about `35-39%` of runtime saving frames and `7-9%` decoding them back
- the streamed path reduces frame transport overhead to about `1-2s` total per clip
- sampled GPU device utilization improved from about `79-82%` to about `84-86%`

Implication:

- the first major optimization was pipeline I/O reduction, not custom shaders
- the default runtime transport is now `pipe`
- shader work is still not the first justified optimization target

### Safety boundary

The checkpoint was first inspected in an isolated Lima VM. Local loading now uses:

- `torch.load(..., weights_only=True)`
- `torch.serialization.safe_globals([slice])`

This checkpoint format requires allowlisting Python's built-in `slice`, but does not require `weights_only=False`.
