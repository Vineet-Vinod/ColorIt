# DeOldify Fine-Tuning

This repo now includes a pragmatic fine-tuning path for the existing DeOldify generator checkpoint.

It is intentionally not full GAN re-training. The goal is narrower and matches the current failure mode:

- reduce unnecessary blue on skin and costumes
- learn warmer and bolder clothing colors from same-era reference films
- keep checkpoint output compatible with the current inference pipeline

## Important Data Rule

For supervised fine-tuning, use the color masters as the target and generate the input B/W frames from those exact same frames.

Do not train on a separate B/W copy of the same title unless you have verified frame-accurate alignment. Different masters often differ in:

- frame rate
- edits
- restoration choices
- titles and leader footage

Misaligned supervision is worse than having less data.

## 1. Pick The Color Masters

From `../Movies/Kannada/finetune_dataset`, only pass the files that are true color references.

If a movie exists as both color and B/W, use the color file for dataset prep. The B/W copy is useful later for qualitative evaluation, not for paired training.

## 2. Extract Training Samples

Recommended starting point:

- sample every `2.0s`
- resize longest side to `512`
- hold out `10%` of time blocks for validation
- skip near-duplicates with `--min-mean-diff 2.0`

Example:

```bash
uv run python scripts/prepare_finetune_dataset.py \
  --movie ../Movies/Kannada/finetune_dataset/'Bangaarada Manushya.mp4' \
  --movie ../Movies/Kannada/finetune_dataset/'Bidugade.mp4' \
  --movie ../Movies/Kannada/finetune_dataset/'Kasthuri Nivasa Color.mp4' \
  --movie ../Movies/Kannada/finetune_dataset/'Krishnadevaraya.mp4' \
  --output-root data/finetune/kannada_period \
  --sample-seconds 2.0 \
  --max-side 512 \
  --overwrite
```

Outputs:

- images: `data/finetune/kannada_period/images/train/` and `.../val/`
- manifest: `data/finetune/kannada_period/manifest.jsonl`
- summary: `data/finetune/kannada_period/summary.json`

The prep script stores only color targets. The training script converts each target frame to synthetic grayscale on the fly, which guarantees exact alignment.

## 3. Fine-Tune The Generator

Stage 1 recommendation:

- `256x256`
- batch size `4`
- `8` epochs
- encoder frozen for epoch `1`
- decoder LR `1e-4`
- encoder LR scaled down to `0.25x`

Example:

```bash
uv run python scripts/finetune_deoldify.py \
  --manifest data/finetune/kannada_period/manifest.jsonl \
  --checkpoint models/deoldify/ColorizeVideo_gen.pth \
  --output-dir models/deoldify/finetune/kannada_period_stage1 \
  --image-size 256 \
  --batch-size 4 \
  --epochs 8 \
  --learning-rate 1e-4 \
  --freeze-encoder-epochs 1
```

What this script optimizes:

- RGB reconstruction
- chroma reconstruction
- saturation matching
- extra penalty when the prediction goes bluer than the target where the target is not blue-dominant

That last term is the deliberate bias against blue skin patches and “everything became navy” costume outputs.

Outputs:

- best checkpoint: `best.pth`
- last checkpoint: `last.pth`
- resumable trainer state: `training_state.pth`
- epoch metrics: `metrics.jsonl`
- validation previews: `previews/epoch_XX.png`

The saved checkpoint includes a top-level `model` state dict, so the current inference loader can use it directly.

## Overnight Launch

Use the helper script so the machine stays awake, output is logged, and resume is enabled by default:

```bash
bash scripts/run_finetune_overnight.sh \
  kannada_period_stage1 \
  data/finetune/kannada_period/manifest.jsonl \
  --checkpoint models/deoldify/ColorizeVideo_gen.pth \
  --image-size 256 \
  --batch-size 4 \
  --epochs 8 \
  --learning-rate 1e-4 \
  --freeze-encoder-epochs 1
```

This writes:

- checkpoints: `models/deoldify/finetune/kannada_period_stage1/`
- logs: `data/logs/finetune/`

If the process stops after a completed epoch, rerun the same command and it resumes from `training_state.pth`.

## 4. Optional Stage 2

If stage 1 clearly improves palette choice, do one short higher-resolution pass:

```bash
uv run python scripts/finetune_deoldify.py \
  --manifest data/finetune/kannada_period/manifest.jsonl \
  --checkpoint models/deoldify/finetune/kannada_period_stage1/best.pth \
  --output-dir models/deoldify/finetune/kannada_period_stage2 \
  --image-size 320 \
  --batch-size 2 \
  --epochs 3 \
  --learning-rate 5e-5 \
  --freeze-encoder-epochs 0
```

Do not jump straight to a long high-resolution run. On this architecture it is slower and easier to overfit.

## 5. Use The Fine-Tuned Checkpoint

Point your config at the new checkpoint, for example:

```yaml
model:
  weights_path: models/deoldify/finetune/kannada_period_stage1/best.pth
```

Then run the existing clip or batch pipeline as usual.

## Expected Time On This Mac Studio

Local checks on this machine:

- Mac Studio, Apple `M3 Ultra`
- `256 GB` unified memory
- `torch 2.11.0`
- `mps` available
- forward/backward pass for this model works on `mps`

Practical expectation:

- dataset extraction at `2s` sampling for about `11-12h` of unique color movies: about `1-3h`, depending on decode speed and how many titles you include
- stage 1 fine-tune at `256`, batch `4`, roughly `8-12k` usable samples after duplicate filtering: about `6-12h`
- stage 2 fine-tune at `320`, batch `2`, `3` epochs: about `2-5h`

So the realistic first full pass on this Mac Studio is:

- prep + stage 1 overnight
- stage 2 the next morning if stage 1 looks good

## What To Check In Validation Previews

Focus on the exact failure modes you described:

- faces no longer pick up blue shadows on cheeks and foreheads
- costumes stop collapsing into dark blue or slate
- gold, cream, maroon, saffron, and green fabrics separate more clearly
- background quality does not regress badly while fixing foreground colors

If the model gets warmer but starts to wash everything into orange/brown, lower:

- `--blue-penalty-weight`
- `--saturation-loss-weight`

If costumes are still timid and grayish, raise:

- `--chroma-loss-weight`
- `--saturation-loss-weight`

If faces still show blue contamination, raise:

- `--blue-penalty-weight`

in small steps, not big jumps.
