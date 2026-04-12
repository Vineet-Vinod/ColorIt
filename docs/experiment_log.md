# Experiment Log

## Baseline Decision

The v1 reference baseline is `configs/quality.yaml`.

Reason:

- it delivered the best first-pass watchability on the probe set
- warmer and more aggressive postprocess variants changed tone, but did not reliably solve the remaining model-level blue patches and motion flicker
- aggressive warming degraded neutrality before it fixed the core temporal problems

## Probe Findings

Observed repeatedly across the probe set:

- blue is overused across skin, clothes, and shadows
- outdoor shots flicker more than indoor shots
- high-motion scenes can flip clothing colors frame to frame
- translucent patch artifacts appear intermittently in some sequences

Implication:

- the baseline is good enough for a v1 movie pass
- remaining issues should be treated as known model limitations, not blockers for orchestration work

This document will track:

- probe clip runs
- preset comparisons
- runtime notes
- visual QC observations

## Production Prep

The current overnight path is validated with `configs/full_movie.yaml`.

- scene detection threshold: `0.60`
- resulting scene units on the full movie: `412`
- batch runner: resumable and failure-tolerant
- temp frame cleanup: enabled after successful clip renders
- source scene clip cleanup: enabled after successful batch items

Validation completed:

- `detect-scenes` on the full movie
- `colorize-batch --resume --limit 2`
- `assemble-final --limit 2`

## Compression

The review-copy compression profile is now locked into the pipeline.

- video codec: `libx264`
- preset: `slow`
- `CRF 22`
- audio: `AAC 128k`
- `+faststart`

Validation completed:

- `compress-final` on `scenes_t060_first2.mp4`
- full-movie review copy generated at `emme_thammanna_colorized_v1_crf22_slow.mp4`

## Performance

Baseline benchmark command:

- `benchmark-clips` on `clip_02`, `clip_04`, `clip_06`, and `clip_10` with `configs/quality.yaml`

Observed baseline:

- throughput: about `7.9-8.2 fps`
- GPU utilization average: about `79-82%`
- process CPU average: about `81-82%`

Dominant stage costs:

- model inference: about `37-38%`
- PNG frame save: about `35-39%`
- PNG frame decode: about `7-9%`
- postprocess: about `7-8%`
- ffmpeg encode: about `2-3%`

Implication:

- the current v1 pipeline is not purely model-bound
- shader work is not the first justified optimization
- the next perf experiment should remove the PNG round-trip and re-benchmark

## Streaming Frame Transport

The PNG round-trip has now been replaced with streamed rawvideo frame transport through ffmpeg pipes.

Observed on the same representative clip set:

- throughput: about `15.9-16.1 fps`
- GPU utilization average: about `84-86%`
- process CPU average: about `74-78%`

Dominant stage costs after the transport change:

- model inference remains the clear runtime leader
- frame transport overhead dropped from roughly `43-47%` combined to roughly `1-2s` total per clip
- this is roughly a `2x` throughput improvement versus the PNG path

Implication:

- pipeline I/O was the main v1 performance tax
- moving to custom shaders is still premature
- the next perf work should focus on reducing remaining Python/image conversion overhead and evaluating whether decode/encode can overlap more cleanly with inference
