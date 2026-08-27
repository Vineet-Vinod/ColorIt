from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from src.pipeline.config import AppConfig
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths


DEFAULT_SPENSERCAI_URL = (
    "https://huggingface.co/spensercai/DeOldify/resolve/"
    "448956a51080c3e5fb29a4e0f3e09457afd94eb0/ColorizeVideo_gen.pth"
)
DEFAULT_DDCOLOR_URL = (
    "https://huggingface.co/piddnad/ddcolor_modelscope/resolve/"
    "060f67494e31883a4b13cb27f889f3154847ada4/pytorch_model.bin"
)
DEFAULT_DDCOLOR_ARTISTIC_URL = (
    "https://huggingface.co/piddnad/ddcolor_artistic/resolve/"
    "aa10f72fffc89a6658e37b48556050b4d9a26f63/pytorch_model.bin"
)
DEFAULT_DDCOLOR_WEIGHTS_PATH = Path("models/ddcolor/pytorch_model.bin")
DEFAULT_DDCOLOR_ARTISTIC_WEIGHTS_PATH = Path("models/ddcolor/ddcolor_artistic.bin")
DEFAULT_DEOLDIFY_ARTISTIC_URL = (
    "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth"
)
DEFAULT_DEOLDIFY_ARTISTIC_WEIGHTS_PATH = Path("models/deoldify/ColorizeArtistic_gen.pth")
DEFAULT_DEOLDIFY_STABLE_URL = (
    "https://www.dropbox.com/s/axsd2g85uyixaho/ColorizeStable_gen.pth?dl=1"
)
DEFAULT_DEOLDIFY_STABLE_WEIGHTS_PATH = Path("models/deoldify/ColorizeStable_gen.pth")
DEFAULT_DEEPREMASTER_URL = (
    "https://iizuka.cs.tsukuba.ac.jp/data/remasternet.pth.tar"
)
DEFAULT_DEEPREMASTER_WEIGHTS_PATH = Path("models/deepremaster/remasternet.pth.tar")

# DeOldify does not publish a checksum. This SHA-256 was cross-checked against
# independent mirrors of the canonical DeepAI file before adding it here.
DEOLDIFY_ARTISTIC_SHA256 = "3f750246fa220529323b85a8905f9b49c0e5d427099185334d048fb5b5e22477"
DEOLDIFY_ARTISTIC_SIZE = 255_144_681
DEOLDIFY_STABLE_SHA256 = (
    "ca9cd7f43fb8b222c9a70f7b292e305a000694b0ff9d2ae4a6747b1a2e1ee5af"
)
DEOLDIFY_STABLE_SIZE = 874_066_230
DEOLDIFY_VIDEO_SHA256 = "e3d98bb6a222354c79f95485c2f078a89dc724e9183662506d9e0f5aafac83ad"
DEOLDIFY_VIDEO_SIZE = 874_066_230

# Hugging Face exposes the SHA-256 and size of LFS objects in its model API.
# These values are pinned to the immutable revisions used by the URLs above.
DDCOLOR_MODELSCOPE_SHA256 = (
    "d81711971ec59200da26d5e8a1afae8dd3778d495ea8ad7a7dadc769f403f7e7"
)
DDCOLOR_ARTISTIC_SHA256 = (
    "5a076e84efce59848394f572a105f7b78f7f6e917f615d1ee88537241b96dca3"
)
DDCOLOR_LARGE_SIZE = 911_914_869

# DeepRemaster's official script publishes MD5
# 1219f5830e4a7208b1c7ba2f089a16c8. SHA-256 is used here after verifying both.
DEEPREMASTER_SHA256 = "ce006df28fada47d73c5ef18f7e8ad48a35b4e7e660fdec192f8aeb4ceb79295"
DEEPREMASTER_SIZE = 256_796_033


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_weights_manifest(manifest_path: Path, payload: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def download_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> None:
    if not url.lower().startswith("https://"):
        raise ValueError(f"Model downloads require HTTPS: {url}")

    temp_path: Path | None = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ColorIt/0.1"})
        with urllib.request.urlopen(request) as response:
            final_url = response.geturl()
            if not final_url.lower().startswith("https://"):
                raise ValueError(f"Model download redirected outside HTTPS: {final_url}")
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                delete=False,
                prefix=destination.name + ".",
                suffix=".part",
            ) as temp_handle:
                temp_path = Path(temp_handle.name)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temp_handle.write(chunk)
                temp_handle.flush()
                os.fsync(temp_handle.fileno())

        verify_file(
            temp_path,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
        temp_path.replace(destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def verify_file(
    path: Path,
    *,
    expected_sha256: str | None,
    expected_size: int | None,
) -> None:
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(
            f"Unexpected model size for {path.name}: "
            f"expected {expected_size}, got {path.stat().st_size}"
        )
    if expected_sha256 is not None:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {path.name}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )


def run_download_weights(
    *,
    config: AppConfig,
    config_path: Path,
    url_override: str | None,
    force: bool,
) -> int:
    paths = resolve_project_paths(config)
    ensure_runtime_directories(paths)

    deoldify_url = url_override or DEFAULT_SPENSERCAI_URL
    manifest_path = paths.manifest_dir / "weights.json"
    targets = [
        {
            "name": "deoldify",
            "repo_id": config.model["repo_id"],
            "url": deoldify_url,
            "destination": paths.weights_path,
            "expected_sha256": DEOLDIFY_VIDEO_SHA256,
            "expected_size": DEOLDIFY_VIDEO_SIZE,
        },
        {
            "name": "ddcolor",
            "repo_id": "piddnad/ddcolor_modelscope",
            "url": DEFAULT_DDCOLOR_URL,
            "destination": (paths.root / DEFAULT_DDCOLOR_WEIGHTS_PATH).resolve(),
            "expected_sha256": DDCOLOR_MODELSCOPE_SHA256,
            "expected_size": DDCOLOR_LARGE_SIZE,
        },
        {
            "name": "ddcolor_artistic",
            "repo_id": "piddnad/ddcolor_artistic",
            "url": DEFAULT_DDCOLOR_ARTISTIC_URL,
            "destination": (
                paths.root / DEFAULT_DDCOLOR_ARTISTIC_WEIGHTS_PATH
            ).resolve(),
            "expected_sha256": DDCOLOR_ARTISTIC_SHA256,
            "expected_size": DDCOLOR_LARGE_SIZE,
        },
        {
            "name": "deoldify_artistic",
            "repo_id": "jantic/DeOldify",
            "url": DEFAULT_DEOLDIFY_ARTISTIC_URL,
            "destination": (paths.root / DEFAULT_DEOLDIFY_ARTISTIC_WEIGHTS_PATH).resolve(),
            "expected_sha256": DEOLDIFY_ARTISTIC_SHA256,
            "expected_size": DEOLDIFY_ARTISTIC_SIZE,
        },
        {
            "name": "deoldify_stable",
            "repo_id": "jantic/DeOldify",
            "url": DEFAULT_DEOLDIFY_STABLE_URL,
            "destination": (
                paths.root / DEFAULT_DEOLDIFY_STABLE_WEIGHTS_PATH
            ).resolve(),
            "expected_sha256": DEOLDIFY_STABLE_SHA256,
            "expected_size": DEOLDIFY_STABLE_SIZE,
        },
        {
            "name": "deepremaster",
            "repo_id": "satoshiiizuka/siggraphasia2019_remastering",
            "url": DEFAULT_DEEPREMASTER_URL,
            "destination": (paths.root / DEFAULT_DEEPREMASTER_WEIGHTS_PATH).resolve(),
            "expected_sha256": DEEPREMASTER_SHA256,
            "expected_size": DEEPREMASTER_SIZE,
        },
    ]

    print(f"Config: {config_path.resolve()}")
    print("Model loading: intentionally disabled in this bootstrap phase")

    weights = []
    for target in targets:
        destination = target["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"{target['name']} source: {target['url']}")
        print(f"{target['name']} destination: {destination}")
        if destination.exists() and not force:
            verify_file(
                destination,
                expected_sha256=target.get("expected_sha256"),
                expected_size=target.get("expected_size"),
            )
            print(f"{target['name']} weights already exist and passed verification.")
        else:
            download_file(
                str(target["url"]),
                destination,
                expected_sha256=target.get("expected_sha256"),
                expected_size=target.get("expected_size"),
            )
            print(f"{target['name']} weights downloaded successfully.")

        weights.append(
            {
                "name": target["name"],
                "filename": destination.name,
                "repo_id": target["repo_id"],
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
                "source_url": target["url"],
                "weights_path": str(destination),
            }
        )

    payload: dict[str, Any] = {"weights": weights}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if isinstance(existing, dict) and isinstance(existing.get("image_edit_models"), dict):
            payload["schema_version"] = existing.get("schema_version", 1)
            payload["image_edit_models"] = existing["image_edit_models"]
    write_weights_manifest(manifest_path, payload)
    print(f"Manifest written to {manifest_path}")
    return 0
