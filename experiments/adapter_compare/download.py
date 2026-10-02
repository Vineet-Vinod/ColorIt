from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

FILES = (
    (
        "Lightricks/LTX-2.5-22b-IC-LoRA-Colorization",
        "a2b3d526bfe3b251ba61cdfbf93deb51eb607198",
        "ltx-2.5-22b-ic-lora-colorization-0.9.safetensors",
    ),
    (
        "Lightricks/LTX-2.5",
        "5e6e71018ee1756ed329b697a7b4aedc934dfce9",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    ),
    (
        "Lightricks/LTX-2.5",
        "5e6e71018ee1756ed329b697a7b4aedc934dfce9",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
    ),
    (
        "Lightricks/LTX-2.5",
        "5e6e71018ee1756ed329b697a7b4aedc934dfce9",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    ),
    (
        "Lightricks/LTX-2.5",
        "5e6e71018ee1756ed329b697a7b4aedc934dfce9",
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    ),
)


def download(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for repo, revision, filename in FILES:
        print("Downloading", repo, filename, flush=True)
        path = hf_hub_download(repo, filename, revision=revision, local_dir=destination)
        print("Ready", path, Path(path).stat().st_size, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    download(parser.parse_args().destination)
