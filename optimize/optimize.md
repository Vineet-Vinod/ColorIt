# Convolution Optimization Challenge

## Objective

Beat the current Apple MPS convolution path on the convolution shapes that matter for this project.

This is not a full-pipeline optimization task.
This is a kernel and testbench task.

The agent should focus on:

- extracting the real convolution shapes from the current model
- benchmarking those shapes against the PyTorch MPS baseline
- implementing and testing candidate convolution paths
- keeping only candidates that are both correct and faster

If no candidate can beat Apple’s current path on the important shapes, the correct outcome is to stop and report that clearly.

## Why This Is The Target

The main inference pipeline has already removed the big host-side bottlenecks.

Current profiling says:

- convolution dominates steady-state model-forward time
- `bmm` from self-attention is a distant second
- everything else is too small to matter much under Amdahl’s law

So convolution is the only optimization target that is still worth serious effort.

## Scope

In scope:

- direct benchmarking of the model’s real convolution shapes
- custom Metal or other custom convolution implementations
- correctness and speed testbenches
- automated experiment loops

Out of scope:

- changing model architecture
- retraining
- changing weights
- optimizing ffmpeg, postprocess, or unrelated Python glue
- integrating anything into the full pipeline before it wins in the standalone testbench

## Workspace Layout

- brief: `optimize/optimize.md`
- default loop config: `optimize/configs/default.yaml`
- candidate modules: `optimize/candidates/`
- generated artifacts: `optimize/artifacts/`

## CLI Entry Points

Extract the real shapes first:

```bash
uv run colorit extract-conv-shapes \
  --config configs/quality.yaml \
  --input data/probe_clips/clip_10.mp4 \
  --output optimize/artifacts/conv_shapes.json
```

Benchmark the top shapes against the baseline only:

```bash
uv run colorit benchmark-convs \
  --config configs/full_movie.yaml \
  --opt-config optimize/configs/default.yaml \
  --shapes-manifest optimize/artifacts/conv_shapes.json \
  --output optimize/artifacts/conv_benchmark_results.json
```

Benchmark a custom candidate:

```bash
uv run colorit benchmark-convs \
  --config configs/full_movie.yaml \
  --opt-config optimize/configs/default.yaml \
  --shapes-manifest optimize/artifacts/conv_shapes.json \
  --candidate optimize/candidates/template_candidate.py \
  --output optimize/artifacts/conv_benchmark_results.json
```

## Candidate Interface

A candidate module must define:

```python
def run_case(case: dict, x, weight, bias, *, prepared_state=None):
    ...
```

Optional hooks:

```python
def is_supported(case: dict) -> bool:
    ...

def prepare_case(case: dict, device):
    ...
```

`prepare_case(...)` is for one-time setup and is not included in the timed loop, but it is recorded.

The benchmark harness always compares the candidate against the built-in PyTorch baseline:

- `torch.nn.functional.conv2d`
- `torch.nn.functional.conv_transpose2d`

## Correctness Rules

The candidate must be compared against the current baseline on the same inputs and weights.

For each case the harness checks:

- output shape equality
- `torch.allclose(...)`
- `max_abs_error`
- `mean_abs_error`
- `max_rel_error`
- `mean_rel_error`

Default tolerances live in `optimize/configs/default.yaml`.

These are intentionally tight but reasonable for floating-point inference.

## Performance Rules

Each candidate benchmark must:

- warm up first
- synchronize before and after timing
- run repeated timed iterations
- report median, mean, min, and max

The baseline to beat is PyTorch on MPS using the current optimized full-movie config.

## Success Criteria

A candidate is interesting only if it is:

- correct enough to match baseline behavior closely
- faster on the important shapes
- maintainable enough to justify integration

Practical target:

- aim for `>= 5%` speedup on important shapes
- smaller wins only matter if they apply across many hot shapes

## Validation Loop

For each experiment:

1. Extract or reuse the real convolution shape manifest.
2. Pick the top shapes or explicit `case_id`s.
3. Run the baseline.
4. Run the candidate.
5. Check correctness.
6. Check performance.
7. Keep going only if the candidate wins.

Do not stack speculative changes.
One implementation idea at a time.

## Stop Condition

Stop if:

- the candidate does not beat MPS on the important shapes
- wins are too small to matter
- correctness is too fragile
- implementation complexity outweighs the gain

That is a valid result.

The goal is not to force a custom kernel into the codebase.
The goal is to determine, with evidence, whether Apple’s convolution path can realistically be beaten for this workload.
