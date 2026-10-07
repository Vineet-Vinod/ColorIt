# ColorIt

ColorIt automatically colors black-and-white movies on your computer. It preserves
the original audio and timing and compresses the result for practical storage.
Model downloads happen once; movie processing runs locally.

The default pipeline combines DeOldify and DDColor, processes movies scene by
scene, and smooths colors over time. Apple Silicon acceleration is used when
available, with CPU fallback.

## Install

You need Python 3.11, [uv](https://docs.astral.sh/uv/), and FFmpeg. Allow about
1.7 GB for the default model weights, plus space for temporary video files.

On macOS with Homebrew:

```bash
brew install uv ffmpeg
```

From the project directory, install dependencies and download the models:

```bash
uv sync
uv run colorit download-weights
```

Models are stored in `models/deoldify/` and `models/ddcolor/`. Repeating the
download command reuses existing files.

## Color a movie

```bash
uv run colorit colorize-movie --input /path/to/movie.mp4
```

The output is saved beside the input as `movie_color.mp4`. Processing can take
hours, depending on the movie length and your hardware. Compression targets a
file size of at most 1.5 times the input by default.

Choose an output path:

```bash
uv run colorit colorize-movie --input movie.mp4 --output colored.mp4
```

Resume an interrupted run with the same input and output paths:

```bash
uv run colorit colorize-movie --input movie.mp4 --output colored.mp4 --resume
```

Add `--overwrite` to replace an existing output file.

## Experimental clip colorization

An experimental pipeline uses FLUX and CMNET2 for clips shorter than 60 seconds.
It requires Apple Silicon macOS and does not yet support `--resume`.

```bash
uv sync --extra experimental
uv run colorit download-weights --experimental
uv run colorit colorize-movie --input /path/to/clip.mp4 --experimental
```

Models are stored in `models/flux/` and `models/cmnet2/`. The default output name
is `clip_experimental_color.mp4`.

## Limitations

Colors are inferred and may differ from the original scene. Costume colors can
shift between cuts, and fast motion or occlusion can cause inconsistent results.
Poor source quality can also limit the result.

## License

ColorIt uses the [ColorIt Attribution License 1.0](LICENSE). Include a credit such
as `Colorized with ColorIt` when sharing output publicly. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for model and dependency licenses.
