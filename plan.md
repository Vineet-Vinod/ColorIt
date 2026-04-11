# DeOldify Pipeline Plan for Colorizing *Emme Thammanna* on a Mac Studio M3 Ultra

## Purpose

Build a local, reproducible pipeline to colorize the black-and-white Kannada film *Emme Thammanna* using **pretrained DeOldify weights only**, with no training, on a **Mac Studio M3 Ultra**. The pipeline should prioritize:

- a **good viewing experience** over perfect historical accuracy,
- **stable and believable** colors,
- the ability to process a **full feature film** by chunking and reassembly,
- a design that starts with a reference implementation and later allows targeted optimization for Apple Silicon.

This plan is written so an implementation agent can execute it in phases with minimal ambiguity.

---

## Non-goals

Do **not** do any of the following in the initial pipeline:

- train or fine-tune any model,
- switch to a different model family,
- build a full custom Metal inference engine before the reference pipeline works,
- attempt historically authoritative color assignment,
- optimize for real-time playback.

---

## Success criteria

The project is successful when all of the following are true:

1. A short clip from *Emme Thammanna* can be colorized locally using pretrained DeOldify weights.
2. The pipeline can process multiple scenes from the film with deterministic settings and reproducible outputs.
3. The system can process the full movie in chunks and reassemble a playable final file with original audio preserved.
4. Output is judged visually acceptable on representative scenes:
   - faces,
   - indoor scenes,
   - outdoor daylight,
   - movement-heavy scenes,
   - costume-heavy scenes.
5. The codebase is structured so that:
   - the reference implementation remains usable,
   - optimized backends can be added later without changing the orchestration layer.

---

## High-level strategy

Build in this order:

1. **Reference pipeline**
   - PyTorch + MPS if supported and stable.
   - CPU fallback if MPS is unstable.
   - Use existing DeOldify pretrained video weights.
   - Prove output quality on short clips.

2. **Film-oriented orchestration**
   - Extract clips/scenes from the movie.
   - Colorize scene chunks.
   - Reassemble with original audio.
   - Add logging, manifests, and resumability.

3. **Quality improvements**
   - Temporal smoothing.
   - Per-scene consistency controls.
   - Shot-boundary-aware processing.
   - Triage and manual override support.

4. **Performance improvements**
   - Profile.
   - Optimize decode, batching, tensor movement, encode, and post-process.
   - Only then evaluate deeper Apple-native acceleration.

---

## Assumptions

- The movie file already exists locally on disk (it is in `~/Movies/Kannada/emme thammanna.mp4`).
- The operator has shell access on macOS.
- The operator is willing to install Python, ffmpeg, and system dependencies.
- The operator can tolerate long runtimes for full-film processing.
- The operator will visually inspect outputs before running the full film.

---

## Required outputs

The implementation must ultimately produce:

- a command-line tool or script entry point,
- a reproducible environment file,
- a directory layout for source media, intermediates, logs, and outputs,
- a manifest for each run,
- test clips and benchmark results,
- a final colorized movie file,
- a markdown report summarizing settings used and observed quality issues.

---

## Canonical project structure

Use this layout unless there is a strong reason not to:

```text
ColorIt/
├── README.md
├── pyproject.toml
├── Makefile
├── .python-version
├── .gitignore
├── configs/
│   ├── default.yaml
│   ├── quality.yaml
│   ├── speed.yaml
│   └── full_movie.yaml
├── scripts/
│   ├── verify_env.py
│   ├── extract_probe_clips.py
│   ├── detect_scenes.py
│   ├── colorize_clip.py
│   ├── colorize_batch.py
│   ├── temporal_smooth.py
│   ├── reassemble_movie.py
│   ├── make_manifest.py
│   ├── benchmark_backend.py
│   └── qc_contact_sheet.py
├── src/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── paths.py
│   │   ├── logging_utils.py
│   │   ├── ffmpeg_utils.py
│   │   ├── scene_utils.py
│   │   ├── model_loader.py
│   │   ├── inference.py
│   │   ├── smoothing.py
│   │   ├── assembly.py
│   │   └── manifest.py
│   └── cli.py
├── models/
│   └── deoldify/
├── data/
│   ├── source/
│   ├── probe_clips/
│   ├── scenes/
│   ├── frames/
│   ├── colorized/
│   ├── manifests/
│   ├── logs/
│   └── final/
└── docs/
    ├── experiment_log.md
    ├── qc_rubric.md
    └── optimization_notes.md
```

---

## Phase 0: environment and feasibility

### Goal

Establish a stable local environment and confirm that DeOldify can load and run on the Mac.

### Tasks

1. Install or verify:
   - ffmpeg,
   Note: We will use `uv` for everythin Python related
2. Clone the DeOldify reference repo or vendor only the necessary inference code.
3. Download pretrained **video colorization** weights into `models/deoldify/`. Look at [Hugging Face 1](https://huggingface.co/thookham/DeOldify/tree/main) and [Hugging Face 2](https://huggingface.co/spensercai/DeOldify/tree/main) and be careful coz they are large repos and are untrusted. Download from one source.
4. Create a small verification script that:
   - prints Python version,
   - prints PyTorch version,
   - checks whether MPS is available,
   - loads the model and runs a single inference pass on a test frame.

### Deliverables

- `pyproject.toml`
- `scripts/verify_env.py`
- a `README.md` section called `Environment setup`
- a text log showing:
  - MPS available or not,
  - model weights found,
  - single-frame inference succeeded

### Acceptance criteria

- The model loads without manual notebook interaction.
- A single frame is colorized and written to disk.
- Failures are reported with actionable error messages.

### Notes for implementation

- Prefer a normal Python script over notebook code.
- Avoid hardcoding absolute paths.
- The environment verification must exit non-zero on failure.

---

## Phase 1: probe clips from the source movie

### Goal

Extract representative short clips from *Emme Thammanna* to evaluate quality before full-film runs.

### Tasks

1. Create a script that accepts the source movie path and outputs **probe clips**.
2. Extract at least 8 clips, each 10 to 20 seconds long, covering:
   - close-up faces,
   - two-person dialogue,
   - outdoor daylight,
   - indoor low light,
   - medium motion,
   - fast motion,
   - costume-rich scenes,
   - a song or dance scene if present.
   Ask me if needed and I will give timestamps.
3. Store metadata for each probe clip:
   - source time range,
   - duration,
   - frame rate,
   - resolution,
   - notes field.

### Deliverables

- `scripts/extract_probe_clips.py`
- `data/probe_clips/`
- `data/manifests/probe_clips.json`

### Acceptance criteria

- Probe clips are reproducibly generated from the movie file.
- A manifest exists and includes exact timestamps and output filenames.

### Notes for implementation

- Use ffmpeg for clip extraction.
- Prefer stream copy where possible for speed, but re-encode if necessary for clean cut points.
- Include an option to override clip timestamps manually.

---

## Phase 2: baseline colorization on short clips

### Goal

Build a repeatable clip-level colorization command and establish a baseline quality preset.

### Tasks

1. Create `scripts/colorize_clip.py` that:
   - accepts an input clip,
   - loads the DeOldify video model,
   - runs colorization,
   - writes an output clip,
   - records runtime and configuration.
2. Create two presets:
   - `quality`
   - `speed`
3. Run all probe clips through the baseline pipeline.
4. For each run, record:
   - clip id,
   - preset,
   - runtime,
   - backend used (`mps` or `cpu`),
   - failures,
   - output path.

### Deliverables

- `scripts/colorize_clip.py`
- `configs/quality.yaml`
- `configs/speed.yaml`
- `data/colorized/probes/`
- `data/manifests/probe_runs.json`

### Acceptance criteria

- Every probe clip can be processed from the command line.
- Output files are playable.
- Logs clearly identify which backend was used.
- No interactive notebook steps are required.

### Required CLI shape

Support a command with this structure or very close to it:

```bash
uv run src.cli colorize-clip \
  --input data/probe_clips/clip_01.mp4 \
  --output data/colorized/probes/clip_01_quality.mp4 \
  --config configs/quality.yaml
```

---

## Phase 3: visual QC and parameter selection

### Goal

Choose the baseline settings that produce the best viewing experience on representative material.

### Tasks

1. Define a QC rubric in markdown with numeric scoring for:
   - face realism,
   - skin tone plausibility,
   - costume plausibility,
   - background plausibility,
   - temporal stability,
   - flicker,
   - artifact severity,
   - overall watchability.
2. Generate a side-by-side inspection artifact for each probe:
   - original clip,
   - colorized clip,
   - optional frame contact sheet.
3. Score all probe outputs.
4. Select a default preset for full-movie processing.

### Deliverables

- `docs/qc_rubric.md`
- `scripts/qc_contact_sheet.py`
- `docs/experiment_log.md`
- `configs/full_movie.yaml`

### Acceptance criteria

- There is a documented reason for the chosen full-movie preset.
- Probe results can be reviewed without rerunning inference.

### Notes for implementation

- Keep the QC process lightweight.
- A markdown table is enough.
- Do not over-automate aesthetic judgment.

---

## Phase 4: scene-aware chunking for full-film processing

### Goal

Make the movie processable in chunks that align with scene boundaries as much as possible.

### Tasks

1. Create `scripts/detect_scenes.py`.
2. Detect scene boundaries using one of:
   - ffmpeg-based heuristics,
   - PySceneDetect,
   - a simple content-difference method.
3. Output a scene manifest with:
   - scene id,
   - start time,
   - end time,
   - duration,
   - source file,
   - chosen processing preset.
4. Add support for:
   - minimum scene duration,
   - maximum scene duration,
   - splitting very long scenes into subchunks with overlap.
5. Preserve ordering deterministically.

### Deliverables

- `scripts/detect_scenes.py`
- `data/manifests/scenes.json`

### Acceptance criteria

- The full film can be represented as an ordered list of processable units.
- Extremely short false-positive scenes are filtered out.
- Very long scenes are split safely for memory/runtime control.

### Notes for implementation

- Use overlap on subchunks, for example 0.5 to 2 seconds.
- Keep a stable naming scheme like `scene_0001_part_00.mp4`.

---

## Phase 5: batch pipeline for full movie

### Goal

Run the entire movie through an orchestrated, resumable batch pipeline.

### Tasks

1. Create `scripts/colorize_batch.py` that:
   - reads the scene manifest,
   - processes each scene or subchunk in order,
   - writes outputs into deterministic folders,
   - supports resume,
   - supports skipping already completed outputs,
   - writes structured logs.
2. Create a manifest for each run with:
   - run id,
   - source file hash,
   - config hash,
   - model weight filename,
   - backend,
   - per-scene status,
   - timing data.
3. Add retry logic for transient failures.
4. Ensure partial runs can continue without reprocessing completed units.

### Deliverables

- `scripts/colorize_batch.py`
- `scripts/make_manifest.py`
- `data/manifests/full_run_*.json`
- `data/logs/`

### Acceptance criteria

- A failed run can be resumed safely.
- Completed chunks are not rerun unless explicitly requested.
- Output location and manifest format are documented.

### Required CLI shape

Support a command with this structure or very close to it:

```bash
uv run src.cli colorize-batch \
  --movie data/source/emme_thammanna.mp4 \
  --scene-manifest data/manifests/scenes.json \
  --config configs/full_movie.yaml \
  --resume
```

---

## Phase 6: reassembly and audio preservation

### Goal

Produce a single playable colorized film file with original audio preserved.

### Tasks

1. Create `scripts/reassemble_movie.py`.
2. Concatenate processed chunks in the correct order.
3. Preserve original audio exactly where possible.
4. Handle overlap trimming cleanly if subchunks were used.
5. Output final files:
   - mezzanine/master encode,
   - optionally a smaller review encode.

### Deliverables

- `scripts/reassemble_movie.py`
- `data/final/emme_thammanna_colorized_master.mp4`
- optional `data/final/emme_thammanna_colorized_review.mp4`

### Acceptance criteria

- Final file plays from start to finish.
- Audio is in sync.
- No missing or duplicated chunk boundaries are visible during playback.

### Notes for implementation

- Keep a concat manifest for ffmpeg.
- Prefer a high-quality mezzanine output first.
- Review encode can be lower bitrate for quick human inspection.

---

## Phase 7: temporal stabilization and consistency improvements

### Goal

Reduce flicker and improve consistency between adjacent frames and chunks.

### Tasks

1. Add an optional post-process smoothing step in `scripts/temporal_smooth.py`.
2. Start simple:
   - temporal blending,
   - rolling average in color space,
   - optional luminance/chrominance separation.
3. Make smoothing configurable and disable by default until validated.
4. Compare probe clips before and after smoothing.
5. If smoothing helps, integrate it into the full-movie preset.

### Deliverables

- `scripts/temporal_smooth.py`
- before/after comparisons on probe clips
- updated `configs/full_movie.yaml` if adopted

### Acceptance criteria

- Smoothing reduces visible flicker on at least some problematic clips.
- Smoothing does not introduce severe ghosting.
- The decision to enable or disable smoothing is documented.

### Notes for implementation

- This phase is important because model-level color may be acceptable even when temporal stability is weak.
- Keep this as a post-process first; do not modify model internals yet.

---

## Phase 8: performance profiling on Apple Silicon

### Goal

Measure where time is spent before attempting deeper optimization.

### Tasks

1. Create `scripts/benchmark_backend.py`.
2. Benchmark at least:
   - frame decode,
   - preprocessing,
   - model inference,
   - post-process,
   - encode,
   - disk I/O.
3. Benchmark on:
   - 10-second clip,
   - 60-second clip,
   - one long scene.
4. Record backend:
   - MPS,
   - CPU fallback.
5. Produce a simple markdown report ranking bottlenecks.

### Deliverables

- `scripts/benchmark_backend.py`
- `docs/optimization_notes.md`

### Acceptance criteria

- There is data showing the top bottlenecks.
- Optimization work is prioritized by measured impact, not guesswork.

### Notes for implementation

- Do not write custom Metal kernels before this phase is complete.
- Distinguish model time from video pipeline time.

---

## Phase 9: Apple-specific optimization targets

### Goal

Improve throughput on the Mac without destabilizing the reference pipeline.

### Optimization order

Implement in this order:

1. **Pipeline-level optimizations**
   - larger but safe frame batches,
   - asynchronous decode/encode,
   - reduced tensor copies,
   - pinned directory layout and cache reuse,
   - parallel scene scheduling if memory allows.

2. **Backend-level improvements**
   - validate PyTorch MPS behavior and memory use,
   - identify unsupported ops causing fallback,
   - replace obvious Python bottlenecks in preprocessing/post-processing.

3. **Selective native acceleration**
   - move non-model hot paths to Swift or Metal only if profiling justifies it,
   - examples: color smoothing, colorspace conversion, frame packing, overlap trimming.

4. **Deeper model-port exploration**
   - only if the reference model is good enough and a major speed bottleneck remains,
   - investigate export feasibility and correctness before rewriting.

### Deliverables

- updated benchmark report,
- implementation notes on which optimizations were worth keeping,
- stable optimized mode behind a config flag.

### Acceptance criteria

- Optimized mode produces output visually equivalent to the reference mode.
- Any speedup is documented with measured comparisons.

---

## Phase 10: QC for full-film release candidate

### Goal

Evaluate the full-film result and identify scenes that need alternative handling.

### Tasks

1. Review the final movie and log:
   - severe flicker,
   - implausible skin tones,
   - costume failures,
   - scene-boundary discontinuities,
   - low-light failures.
2. Create a scene-level issue list.
3. Add support for per-scene overrides:
   - alternate preset,
   - stronger smoothing,
   - re-run only selected scenes.
4. Reassemble a release candidate after targeted fixes.

### Deliverables

- `docs/full_movie_qc.md`
- `data/manifests/scene_overrides.json`
- release candidate output

### Acceptance criteria

- It is possible to rerun only problematic scenes.
- The release candidate is clearly better than the first full-film pass.

---

## Required implementation principles

### 1. Determinism where possible
- Use stable naming.
- Store configs and manifests.
- Log hashes for source file and model weights.

### 2. Fail loudly and usefully
- Every script must validate inputs.
- Missing tools or files should produce actionable messages.

### 3. Resumability
- Long jobs must be restartable.
- Never assume a full-movie run completes in one shot.

### 4. Separation of concerns
- Keep model inference separate from orchestration.
- Keep ffmpeg utilities separate from model code.
- Keep config parsing centralized.

### 5. Command-line first
- No notebook-only workflow.
- Everything needed for production runs must exist as scripts or CLI commands.

---

## Configuration requirements

Use YAML configs with at least these fields:

```yaml
model:
  weights_path: models/deoldify/ColorizeVideo_gen.pth
  render_factor: 21

runtime:
  backend_preference: mps
  fallback_backend: cpu
  num_workers: 1
  deterministic: true

video:
  preserve_fps: true
  output_codec: libx264
  crf: 16
  pixel_format: yuv420p

scenes:
  detect: true
  min_scene_seconds: 2.0
  max_scene_seconds: 90.0
  overlap_seconds: 1.0

postprocess:
  temporal_smoothing: false
  smoothing_strength: 0.0

paths:
  source_dir: data/source
  probe_dir: data/probe_clips
  scene_dir: data/scenes
  colorized_dir: data/colorized
  manifest_dir: data/manifests
  final_dir: data/final
  log_dir: data/logs
```

The exact schema may differ, but it must support the same intent.

---

## Minimum commands that must exist

The implementation must expose commands equivalent to:

```bash
uv run src.cli verify-env
uv run src.cli extract-probes --movie /path/to/Emme_Thammanna.mp4
uv run src.cli colorize-clip --input clip.mp4 --output out.mp4 --config configs/quality.yaml
uv run src.cli detect-scenes --movie /path/to/Emme_Thammanna.mp4 --output data/manifests/scenes.json
uv run src.cli colorize-batch --movie /path/to/Emme_Thammanna.mp4 --scene-manifest data/manifests/scenes.json --config configs/full_movie.yaml --resume
uv run src.cli reassemble --scene-manifest data/manifests/scenes.json --config configs/full_movie.yaml
uv run src.cli benchmark --input clip.mp4 --config configs/quality.yaml
```

---

## Experiment sequence to run in order

This sequence is mandatory.

### Step 1
Get environment verification passing.

### Step 2
Download weights and run one-frame inference.

### Step 3
Extract probe clips from *Emme Thammanna*.

### Step 4
Run baseline colorization on all probes.

### Step 5
Score outputs and choose the default preset.

### Step 6
Detect scenes across the full movie.

### Step 7
Run a partial batch on 3 to 5 representative scenes.

### Step 8
Enable resumable full-movie batch processing.

### Step 9
Reassemble first full pass with original audio.

### Step 10
Add temporal smoothing if probe comparisons justify it.

### Step 11
Profile bottlenecks.

### Step 12
Optimize only the measured bottlenecks.

### Step 13
Create a release candidate and scene override workflow.

---

## Explicit decisions Codex should follow

1. **Do not replace DeOldify** in the first implementation.
2. **Do not introduce training**.
3. **Do not build UI first**.
4. **Do not optimize before the baseline works**.
5. **Do not process the full movie before probe validation**.
6. **Do not rely on notebook code paths**.
7. **Do not assume MPS is fully compatible**; implement backend fallback cleanly.
8. **Do not lose provenance**; every output must be traceable to config, weights, and source.

---

## Risks and mitigations

### Risk: MPS incompatibility or instability
**Mitigation:** provide clean CPU fallback and isolate backend selection.

### Risk: Full movie takes too long
**Mitigation:** process by scene, support resume, and benchmark before optimization.

### Risk: Flicker or temporal inconsistency
**Mitigation:** add optional temporal smoothing and scene-aware chunking.

### Risk: Scene boundaries show color discontinuity
**Mitigation:** use overlap, per-scene statistics, and selective reruns.

### Risk: Some culturally important colors are implausible
**Mitigation:** accept that the model is plausible, not authoritative; add manual review and scene overrides.

---

## Final expected milestone states

### Milestone A: baseline proven
- one-frame inference works,
- one clip colorizes successfully,
- environment is reproducible.

### Milestone B: probe evaluation complete
- all probe clips processed,
- QC rubric filled out,
- full-movie preset selected.

### Milestone C: full pipeline operational
- scenes detected,
- batch processing works,
- resume works,
- movie reassembles with audio.

### Milestone D: release candidate
- full movie processed,
- biggest quality issues documented,
- selective rerun workflow exists,
- optimized mode tested against reference mode.

---

## What to hand back after each phase

Codex should return after each phase with:

1. files added or changed,
2. commands to run,
3. expected outputs,
4. known issues,
5. whether acceptance criteria were met.

---

## Suggested first concrete task for Codex

Start with **Phase 0** only.

Implement:

- project skeleton,
- environment files,
- `verify-env`,
- model weight path handling,
- one-frame inference smoke test.

Do not proceed to clip extraction or batch orchestration until the smoke test works.

---

## End state

When the full plan is complete, the operator should be able to point the pipeline at the local *Emme Thammanna* movie file, run a deterministic sequence of commands, review probe outputs, process the full film in resumable chunks, and produce a playable colorized final movie with preserved audio.
