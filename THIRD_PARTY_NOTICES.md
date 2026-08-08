# Third-Party Notices

ColorIt builds on open-source software and pretrained AI models. Each component remains subject to its own license; the ColorIt license does not replace or restrict those terms.

This file is a practical attribution and license index, not a substitute for the license text distributed by each project.

## AI Models and Adapted Code

| Component | Role in ColorIt | Source | License |
|---|---|---|---|
| DeOldify Video | Video-oriented colorization base and pretrained `ColorizeVideo_gen.pth` weights | [jantic/DeOldify](https://github.com/jantic/DeOldify), weights mirrored by [spensercai/DeOldify](https://huggingface.co/spensercai/DeOldify) | MIT; the upstream project states that its listed pretrained weights are also MIT-licensed |
| DDColor | Semantic color prediction; a minimal inference subset is vendored in `src/vendor/ddcolor` | [piddnad/DDColor](https://github.com/piddnad/DDColor), weights from [piddnad/ddcolor_modelscope](https://huggingface.co/piddnad/ddcolor_modelscope) | Apache License 2.0 |

The Apache 2.0 license accompanying the vendored DDColor inference code is retained at [`src/vendor/ddcolor/LICENSE`](src/vendor/ddcolor/LICENSE).

DDColor credits adapted research/code from BasicSR, ColorFormer, BigColor, ConvNeXt, Mask2Former, and DETR in its upstream repository. ColorIt vendors only the subset documented in [`src/vendor/ddcolor/README.md`](src/vendor/ddcolor/README.md).

## Direct Runtime Dependencies

| Component | Purpose | Project license/reference |
|---|---|---|
| PyTorch | Tensor execution and model inference | [BSD-style license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| TorchVision | Vision model utilities | [BSD 3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) |
| NumPy | Array operations | [BSD 3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| OpenCV / `opencv-python-headless` | Color spaces, CLAHE, and image processing | [Apache License 2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE) and the wheel project's bundled notices |
| Pillow | Image handling | [HPND License](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| PyYAML | Configuration loading | [MIT License](https://github.com/yaml/pyyaml/blob/main/LICENSE) |
| tqdm | Progress support | [MPL-2.0 and MIT licenses](https://github.com/tqdm/tqdm/blob/master/LICENCE) |
| FFmpeg / `ffprobe` | Video inspection, decoding, encoding, audio, scene detection, and compression | [LGPL/GPL depending on the installed build](https://ffmpeg.org/legal.html) |
| uv | Python environment and package management | [Apache-2.0 or MIT](https://github.com/astral-sh/uv) |

Transitive packages installed from `uv.lock` retain their respective licenses. Consult the installed distribution metadata and upstream repositories for the exact notices applicable to a particular platform build.

## Input Media

Third-party software and model licenses do not grant rights to movies processed by ColorIt. Users are responsible for ensuring that they may copy, modify, colorize, demonstrate, or distribute their input and output media.

The included hackathon demonstration is a one-minute excerpt from *Ramanjaneya Yuddha*, provided by SRS Movies, covering approximately `00:58:13–00:59:13`. It is identified for attribution and evaluation; any use beyond the submitted demonstration remains subject to the rights holder's terms.
