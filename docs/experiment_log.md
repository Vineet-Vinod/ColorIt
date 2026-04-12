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
