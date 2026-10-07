# Experimental clip colorization

This work in progress generates reference images with FLUX.2 Klein 4B and
propagates their colors with Dan64's CMNET2/DINOv3. It accepts clips strictly
shorter than 60 seconds and currently runs on Apple Silicon macOS.

```sh
uv sync --extra experimental
uv run --extra experimental colorit colorize-movie --input clip.mp4 --experimental
```

The first run downloads pinned, checksum-verified assets, about 17 GB in total.
To download them separately:

```sh
uv run --extra experimental colorit download-weights --experimental
```

The output defaults to `clip_experimental_color.mp4`. Use `--output` to choose
another MP4 path, or `--overwrite` to replace an existing output. The command
rejects using the input file as the output. Experimental runs do not support
`--resume` yet.

The regular command continues to use the existing DeOldify/DDColor pipeline.
Its model downloads are separate from the experimental assets.

## Behavior and limitations

FFmpeg normalizes the clip to its reported average frame rate and detects cuts
with scene threshold 0.1. Candidates within half a second are reduced to the
strongest cut; candidates in the initial half second are ignored. Each shot supplies up to three reference frames, at
20%, 50%, and 80% of its duration. Every reference receives the same generic
colorization prompt and seed, with four FLUX steps and 8-bit model quantization.
There are no curated palettes, manual repairs, or actor labels.

CMNET2 preloads each shot's references and starts fresh temporal memory at every
detected cut. Propagation runs in FP32 at a 384-pixel short edge. The delivery
keeps the original dimensions and luminance, restores audio when present, and
encodes HEVC. Frame count, frame rate, duration, audio availability, and the
twice-input-size limit are checked before the output is copied to its destination.
If CRF 18, 23, and 28 cannot meet the size limit, the command fails and retains
the candidate under the run artifacts.

Independent FLUX references can disagree about costume colors or change object
boundaries. CMNET2 can propagate those mistakes or blend conflicting palettes.
This automatic path does not reproduce the quality of the earlier manually
curated FLUX references, and it does not establish costume consistency across cuts.

Variable-frame-rate input is converted to constant frame rate. Even source
dimensions and a readable frame rate and duration are required. The 60-second
limit bounds the experiment; it is not a runtime or memory guarantee.

Assets are stored under `data/experimental/models/`, and each run keeps source
references, generated references, lossless intermediates, logs, and a result
manifest under `tmp/experimental/`. Both use the existing runtime root, including
`COLORIT_RUNTIME_ROOT` and the canonical `~/ColorIt` location when using worktrees.

## Attribution and source pins

CMNET2 is by **Dan64**. The runner uses
[dan64/cmnet2](https://github.com/dan64/cmnet2) at
`e0d51432d224476769babfbff5e90f531a454939`, with the released DINOv3
`p374099` checkpoint. CMNET2 credits ColorMNet, XMem, XMem++, DINOv2, and DINOv3;
its upstream README is retained alongside the downloaded source. The tracked
adapter patch changes CUDA device assumptions for MPS, disables unused correlation
operators, fixes the attention reshape, and requires complete checkpoint loading.

FLUX.2 Klein 4B is by Black Forest Labs, at Hugging Face revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`. MFLUX 0.19.1 supplies its MLX image
editing implementation. See [third-party notices](../THIRD_PARTY_NOTICES.md).

The old `deep_rem` experiments remain recoverable in Git history. The baseline
restore commit makes the tracked files match main before introducing this path.

## Local validation, October 7, 2026

Ruff, strict mypy for the experimental module and tests, and all 14 tests pass:

```sh
uv run --extra experimental task ci
```

The checks cover the duration limit, CLI dispatch, output protection, checksum
rejection, the missed-cut regression, luminance restoration, and real FFmpeg
delivery with fractional frame rates both with and without audio. A real
60-second clip was also rejected before model setup. The regular CLI was imported
successfully while access to MLX, MFLUX, and Pydantic was blocked.

Two dance excerpts were rendered in tmux on the M3 Ultra. Each contained two
detected shots and six automatically generated references. The eight-second
delivery has 200 frames and is 1.016 times its input size. The corrected
twelve-second delivery has 300 frames and is 1.038 times its input size. Both
retain 25 fps, 1080p, their exact duration, and audio, and decode completely.

The first twelve-second run exposed a missed cut at the original 0.25 threshold.
The final detector finds it at frame 141 using threshold 0.1; nearby candidates
are consolidated so the transition does not create a one-frame shot.

Inspected sequence samples show sky, vegetation, and costume color, but the
woman's costume changes from pink in the wider shot to green in the close-up,
with further color variation during motion. The second excerpt's distant
references also disagree about pink versus blue. These are expected WIP quality
failures, rather than evidence of solved temporal or costume consistency.

Videos, review sheets, full decoder checks, logs, and `validation.json` remain
under the canonical runtime root at `tmp/deep_rem_pr/`. Per-run references and
lossless intermediates remain under `tmp/experimental/`.
