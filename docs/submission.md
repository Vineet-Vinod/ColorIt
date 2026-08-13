# ColorIt — Submission

## Concept and Impact

### Inspiration

I love watching old Kannada movies. Seeing the restored and colorized version of *Kasturi Nivasa* made a familiar film feel more alive and real. I was impressed by the result, and struck by how intensive, expensive, and slow that kind of restoration can be.

I wanted a faster alternative: an AI-assisted pipeline that an individual could run without a restoration studio, while still preserving the timing, audio, and character of the original film.

### Problem Statement

Old films carry language, performance, fashion, music, and cultural memory, but professional colorization can demand weeks of work from a specialist team. Individual viewers, small archives, and many rights holders cannot afford that process. Meanwhile, frame-by-frame AI colorizers often flicker, mishandle cuts, lose audio or timing, and produce files that are impractical to store.

**How can one person give an old film a credible second life in hours, on their own computer, without manually grading every scene?**

### Solution Overview

ColorIt is a one-command, local movie-colorization pipeline. It detects scenes, restores weak luminance contrast conservatively, runs DeOldify and DDColor as complementary AI colorizers, smooths predicted chroma over time, reassembles the film with its original audio, verifies its timing, and compresses the output to a practical size.

### Target Users

- Individual users who want to color films or clips themselves.
- Film restoration experts who want a fast first pass before artistic finishing.
- Movie studios and rights holders restoring back-catalog titles.

### Potential Impact

ColorIt reduces a process that can take several weeks of team effort to a few hours of unattended processing on powerful consumer hardware. A 2.5-hour movie takes about six hours on an M3 Ultra Mac Studio, while CPU fallback keeps the pipeline accessible to ordinary laptops at slower speeds.

Skies, water, terrain, vegetation, and many actor shots already colorize convincingly. Costume palettes, high motion, and cross-cut identity are still imperfect, but those are focused research problems rather than reasons to keep restoration inaccessible. As consistency improves, films that currently live mainly in archives can gain a new life with modern audiences.

The same local AI viewing layer can eventually combine colorization, detail reconstruction, multilingual subtitles, translation, voice, and lip synchronization, making historical and personal media more immersive across devices and languages.

## Technical Architecture

See [`architecture.svg`](architecture.svg) for the complete system diagram.

The system has three stages:

1. **Prepare:** inspect the movie, detect cuts, create frame-exact scenes, and improve luminance contrast with conservative CLAHE.
2. **Infer and fuse:** run DeOldify and DDColor, smooth DDColor chroma over time in Lab space, and apply it to the stable DeOldify base.
3. **Deliver:** normalize and concatenate every scene, restore source audio, compress against a size target, and validate frame count, frame rate, and duration.

Resumable manifests, live progress, raw-video streaming, and automatic cleanup support the whole pipeline.

## Technology Stack

- Python 3.11 and `uv`
- PyTorch and TorchVision
- DeOldify Video and DDColor
- OpenCV, NumPy, Pillow, and PyYAML
- FFmpeg and `ffprobe`
- Apple Metal Performance Shaders with CPU fallback
- JSON run manifests; no external database or API

## MVP Scope

The MVP is an automatic, local pipeline that handles varied scenes without masks, prompts, actor labels, or manual per-shot grading. It produces a plausible color pass, gently improves weak contrast without inventing detail, preserves source audio and timing, and keeps output size practical.

Temporal consistency is improved but not solved. Fast action, occlusion, difficult cuts, recurring actor identity, and costume continuity remain the most important limitations.

## Features Implemented

- One-command full-movie colorization
- Cut-aware, frame-exact scene processing
- Conservative luma-only CLAHE enhancement
- DeOldify and DDColor dual-model inference
- Temporal Lab-chroma smoothing and fusion
- Raw-video FFmpeg pipes and batched inference
- MPS acceleration and CPU fallback
- Resumable stage and scene manifests
- Live frame progress and duration-weighted ETA
- Original audio preservation
- Frame-rate, frame-count, and duration validation
- Size-aware H.264 compression retries
- Automatic cleanup after successful runs

## Future Roadmap

1. **Train temporally conditioned colorization models from scratch.** Use multi-frame, cross-cut, actor-identity, and costume-identity context to keep skin tones and costume palettes stable through motion and editing.
2. **Reconstruct missing detail for 1080p and 4K.** Build temporally aware generative super-resolution that synthesizes plausible high-frequency information absent from the source rather than simply stretching pixels. Confidence gating and temporal constraints will limit hallucination and flicker.
3. **Build a multilingual viewing layer.** Combine ASR, time-aligned subtitles, translation, voice generation, and speech-aware lip synchronization to make older cinema accessible across languages.
4. **Add optional expert controls.** Support reference palettes and review checkpoints without compromising the one-command default.

## AI Development Workflow

OpenAI Codex was the primary AI development collaborator. The human-led loop was: define a visual problem, use Codex to explore research and prototype promising ideas, inspect actual frames and video, integrate or discard the result, and repeat. Codex reduced the search space and made rapid validation possible; visual judgment and product decisions remained human-controlled.

At runtime, DeOldify and DDColor perform movie colorization locally. Codex is not part of inference.
