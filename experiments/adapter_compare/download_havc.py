from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from requests import RequestException


class WeightRecord(BaseModel):
    repo: str
    revision: str
    path: str
    bytes: int
    sha256: str
    optional: bool
    url: str


def destination(record: WeightRecord, directory: Path) -> Path:
    name = Path(record.path).name
    if name.endswith(".gguf"):
        if name == "mmproj-BF16.gguf":
            name = "Qwen3-VL-8B-Instruct-mmproj-BF16.gguf"
        return directory / "text_encoders" / name
    if "viggle-turbo" in name:
        return directory / "loras" / name
    if record.path.startswith("vae/"):
        return directory / "vae" / name
    return directory / "diffusion_models" / name


def verify(path: Path, record: WeightRecord) -> None:
    if path.stat().st_size != record.bytes:
        raise RuntimeError(f"Incorrect size: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != record.sha256:
        raise RuntimeError(f"Incorrect SHA256: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download pinned HAVC Qwen2.1 weights and verify hashes."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tmp/adapter_compare_90_120/havc_qwen21_manifest.json"),
    )
    parser.add_argument(
        "--directory", type=Path, default=Path("data/adapter_compare_90_120/havcdit")
    )
    parser.add_argument("--precision", choices=["bf16", "int8"], default="bf16")
    args = parser.parse_args()
    records = [
        WeightRecord.model_validate(item)
        for item in json.loads(args.manifest.read_text())
    ]
    completed = []
    for record in records:
        if "diffusion_models/" in record.path and (
            (args.precision == "bf16") != record.optional
        ):
            continue
        target = destination(record, args.directory).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            print(f"Downloading {record.repo}/{record.path} -> {target}", flush=True)
            for attempt in range(5):
                try:
                    downloaded = Path(
                        hf_hub_download(
                            repo_id=record.repo,
                            filename=record.path,
                            revision=record.revision,
                            local_dir=args.directory
                            / "downloads"
                            / record.repo.replace("/", "--"),
                        )
                    )
                    break
                except RequestException:
                    if attempt == 4:
                        raise
                    print(
                        f"Network interruption, resuming attempt {attempt + 2}",
                        flush=True,
                    )
                    time.sleep(2**attempt)
            os.link(downloaded, target)
        print(f"Verifying {target}", flush=True)
        verify(target, record)
        completed.append({**record.model_dump(), "local_path": str(target)})
        (args.directory / "verified_weights.json").write_text(
            json.dumps(completed, indent=2) + "\n"
        )
    print("All requested weights downloaded and SHA256 verified.", flush=True)


if __name__ == "__main__":
    main()
