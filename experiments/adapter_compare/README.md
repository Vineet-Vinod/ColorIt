# Colorization comparison, 90–120 seconds

Experiments on branch `deep_rem`, on the M3 Ultra with 256 GiB unified memory.
No new branch or worktree. Generated media and logs are in
`tmp/adapter_compare_90_120`; weights are in `data/adapter_compare_90_120`.

## Current state

- HAVC code audit is in `havc_audit.json`. The learned CMNET2 model matches our
  tested family, but default resolution, reference memory and shot resets differ.
- FCTCVC completed 750 frames at 25 fps using its released 15-frame configuration.
  Both window endpoints receive aligned colors from the approved CMNET2 output.
  This is a propagation test with dense supplied colored frames. The original
  scene manifest missed three transitions; the final run uses verified boundaries
  in `scene_manifest.json`.
- HAVC Qwen2.1 + CMNET2 and LTX-2.5 completed automatic and assisted runs on all
  750 frames. The fresh runtime was frozen at `d1e6278` after debugging the large
  Qwen VAE on MPS. The corrected HAVC path uses the author's VAE on CPU FP32.
- The [video comparison](http://darkmac:1516/colorit-adapter-90-120.html)
  includes the existing curated CMNET2 + FLUX baseline, all four first complete
  fresh outputs, raw LTX videos, selected frames and delivery validation.
  Exact measurements and limitations are in `unattended_results.json`.
- The curated baseline retains the most consistent costume palette in the
  inspected frames. HAVC automatic changes outfit colors; assisted HAVC still
  confuses costume roles in the close-up. Assisted LTX follows the requested
  palette better than automatic LTX, but fades during the long close-up and
  changes saturation across cuts. Keep the existing baseline for this clip.

| Fresh case | Queue wall time | Working size | 1080p delivery bytes |
| --- | ---: | --- | ---: |
| HAVC automatic | 1,023.69 s | 512×288 propagation | 15,393,932 |
| HAVC assisted | 1,051.32 s | 683×384 propagation | 15,395,209 |
| LTX automatic | 3,165.64 s | 960×544 | 15,373,683 |
| LTX assisted | 3,179.43 s | 960×544 | 15,402,248 |

Each delivery independently decodes 750 frames at 25 fps, with 30 seconds of
video and audio, at 1920×1080. All are below the 16,867,622-byte interval cap.
Queue times include HAVC delivery; LTX's subsequent CPU 1080p packaging is
excluded. The baseline was reused, so it has no comparable fresh run time.
Both HAVC deliveries and assisted LTX restore original source Lab lightness.
Automatic LTX keeps generated RGB. Raw 960×544 LTX outputs remain available.
Local type checking, lint and all 70 tests passed after the full queue.

## Pinned upstream code

| Repository | Revision |
| --- | --- |
| `dan64/vs-havc` | `6a8063143f3ec89fa159c836557851e5e4e57bad` |
| `Lightricks/LTX-2` | `9ec55f9f22798a3198d9c923856824821bc3317e` |
| `Zhaiyan1996/FCTCVC` | `94bb32982918f8d6cd0b74992fadba1b72f070d1` |

Clones live under `tmp/adapter_compare_90_120/upstream`. `fctcvc.patch` limits
legacy package imports to the tested network; it changes no learned operators.
Apply it with `git -C tmp/adapter_compare_90_120/upstream/FCTCVC apply` and the
absolute path to that patch.

## Isolated environment

The existing environment is `tmp/adapter_compare_90_120/env`, Python 3.11.
Install the two editable official LTX packages from the pinned clone. The directly
used versions are torch 2.14.1, torchaudio 2.11.0, torchvision 0.29.1, transformers
5.14.1, MLX 0.32.2, Pydantic 2.13.5, mmcv 1.6.0, setuptools 80.10.2 and
opencv-python 5.0.0.93. `uv pip check` passed. Additional packages are timm,
imageio, sqlalchemy, gdown, pytest, mypy, ruff and taskipy.

Run local checks from the repository root:

```sh
PATH="$PWD/tmp/adapter_compare_90_120/env/bin:$PATH" task experiment-ci
```

## FCTCVC run

The released checkpoint is the public Google Drive file
`1yK88yjcYo0AHMZPLq9qTsGuigL0CfxFb`. Save it as
`data/adapter_compare_90_120/fctcvc-checkpoint` using gdown. It includes GMA
weights. The runner strictly loads all `generator.*` checkpoint tensors.
Checkpoint SHA-256 is
`05cfb68b830643ce684255358982a35e2fb88efe51538c3ac417a47fdbb661a5`.

Launch in tmux with the experiment interpreter:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH="$PWD" \
  tmp/adapter_compare_90_120/env/bin/python -u -m experiments.adapter_compare.fctcvc \
  --source deep_rem.mp4 \
  --anchors tmp/cmnet2_flux_gpu/deep_rem_flux_cmnet2_gpu_180s.mp4 \
  --checkpoint data/adapter_compare_90_120/fctcvc-checkpoint \
  --upstream tmp/adapter_compare_90_120/upstream/FCTCVC \
  --scene-manifest experiments/adapter_compare/scene_manifest.json \
  --output tmp/adapter_compare_90_120/fctcvc_cut_aware_90_120.mp4
```

Defaults select frames 2250–2999, 683×384 internal size, 15-frame windows with
two frames of overlap, and reset windows at five visually checked shot boundaries.
Raw RGB predictions, timings and configuration are saved beside the video.
`delivery.py` upsamples predicted Lab chroma and restores the original 1080p
source luminance, using the same reconstruction as the approved CMNET2 render.
It encodes HEVC with original audio and checks the interval's 2× file-size budget.

```sh
PYTHONPATH="$PWD" tmp/adapter_compare_90_120/env/bin/python -u \
  -m experiments.adapter_compare.delivery --source deep_rem.mp4 \
  --predictions tmp/adapter_compare_90_120/fctcvc_cut_aware_90_120.npy \
  --output tmp/adapter_compare_90_120/fctcvc_1080p.mp4
```

## LTX setup

The saved Hugging Face account must have access to both `Lightricks/LTX-2.5` and
`Lightricks/LTX-2.5-22b-IC-LoRA-Colorization`. `download.py` pins both revisions
and downloads the adapter, distilled transformer, embedded Gemma4 text encoder,
audio VAE and optional convolutional video VAE. Total download is about 71 GB.

```sh
PYTHONPATH="$PWD" tmp/adapter_compare_90_120/env/bin/python -u \
  -m experiments.adapter_compare.download \
  --destination data/adapter_compare_90_120/ltx25
```

`source_960x544.mp4` already contains the exact 750 source frames. `prompts/`
describes the approved baseline palette for each shot. The prepared runner uses
the official distilled stage-1 colorization recipe, strength 1, seed 42, no CFG,
121-frame chunks, 17-frame latent carry and decoded overlap within each shot.
The output remains at source 25 fps. The official convolutional VAE is selected
for Mac compatibility; this differs from the foundation model's default DiffVAE.

```sh
PYTHONPATH="$PWD" tmp/adapter_compare_90_120/env/bin/python -u \
  -m experiments.adapter_compare.ltx \
  --source tmp/adapter_compare_90_120/source_960x544.mp4 \
  --weights data/adapter_compare_90_120/ltx25 \
  --unused-upsampler tmp/reference_video_models/ltx_colorization_weights/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --scene-manifest experiments/adapter_compare/scene_manifest.json \
  --prompts experiments/adapter_compare/prompts \
  --output tmp/adapter_compare_90_120/ltx25_90_120.mp4
```

Run coloring commands in tmux and serialize GPU workers with the memory guard.
`--no-mlx` selects the official PyTorch MPS transformer for that comparison.
The upsampler argument satisfies the upstream constructor; stage 2 is skipped
and those weights never participate. Current upstream removed the model card's
`tile_reference_encode` flag, so the prepared native recipe encodes references
without tiling. Both full trained-weight evaluation modes completed. This is
stage 1 only, with no learned spatial upscaler. Short numerical checks and full
outputs do not establish end-to-end equivalence to the author's CUDA pipeline.

## MLX scope

`ltx_mlx.py` runs the transformer block stack on MLX. Text encoding, video/audio
VAE and transformer input/output layers remain official PyTorch MPS code. It uses
MLX Metal SDPA, RMS normalization and matrix products, with compiled modulation
and residual fusion. It rejects guidance perturbations outside the tested
distilled path. Fresh builders avoid reusing shells whose blocks were replaced.

Numerical tests compare video and joint audio/video blocks, RoPE, attention masks,
and the installed block stack against official PyTorch code in FP32 and BF16.
`benchmark_mlx.py` measures random-weight blocks at official default widths,
using the official `MPSSdpaAttention` implementation. Historical timings with
plain Torch attention do not establish performance against that implementation
or the trained pipeline. Use the full-case measurements above for local runtime.
