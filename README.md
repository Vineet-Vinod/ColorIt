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

### Safety boundary

The checkpoint was first inspected in an isolated Lima VM. Local loading now uses:

- `torch.load(..., weights_only=True)`
- `torch.serialization.safe_globals([slice])`

This checkpoint format requires allowlisting Python's built-in `slice`, but does not require `weights_only=False`.
