# ColorIt

CLI-first pipeline scaffold for colorizing black-and-white Kannada films with DeOldify.

The current repository state is intentionally conservative:

- Python tooling is pinned to `3.11`
- the project layout matches the execution plan in [plan.md](/Users/darksca/ColorIt/plan.md)
- model weights can be downloaded and checksummed
- no DeOldify code is executed yet
- no `.pth` file is loaded or unpickled yet

## Environment setup

This phase avoids running untrusted model code. The only supported bootstrap tasks right now are:

- validating local prerequisites
- creating the expected working layout
- downloading `ColorizeVideo_gen.pth`
- recording file metadata and SHA-256 without loading the file

### Prerequisites

- `uv`
- Python `3.11`
- `ffmpeg`
- enough disk space for the model weights and generated media

### Bootstrap commands

```bash
uv run colorit verify-env
uv run colorit download-weights
```

By default, `download-weights` fetches:

- source: `spensercai/DeOldify`
- file: `ColorizeVideo_gen.pth`
- destination: `models/deoldify/ColorizeVideo_gen.pth`

It also writes a metadata manifest to `data/manifests/weights.json`.

### Safety boundary

`verify-env` currently does not load DeOldify, import PyTorch, or unpickle model weights. That work is intentionally deferred until we move model execution into a sandboxed environment.
