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

### Safety boundary

The checkpoint was first inspected in an isolated Lima VM. Local loading now uses:

- `torch.load(..., weights_only=True)`
- `torch.serialization.safe_globals([slice])`

This checkpoint format requires allowlisting Python's built-in `slice`, but does not require `weights_only=False`.
