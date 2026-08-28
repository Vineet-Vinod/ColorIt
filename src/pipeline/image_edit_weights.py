"""Pinned, verified downloads for locally runnable image-edit models.

This module deliberately does not use an arbitrary Hugging Face repository or
revision supplied at runtime.  Each entry below names a public author-owned
repository, an immutable commit, and the exact files required by Diffusers.
Large LFS objects also carry the size and SHA-256 published by the Hub.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable


HUGGING_FACE_ORIGIN = "https://huggingface.co"
MANIFEST_VERSION = 1
_CHUNK_SIZE = 8 * 1024 * 1024
_TRUSTED_DOWNLOAD_HOST_SUFFIXES = ("huggingface.co", "hf.co")


@dataclass(frozen=True)
class SnapshotFile:
    """One allowlisted file within a pinned model repository."""

    path: str
    size_bytes: int | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_relative_path(self.path)
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError(f"Negative size for {self.path}")
        if self.sha256 is not None:
            if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
                raise ValueError(f"Invalid SHA-256 for {self.path}")


@dataclass(frozen=True)
class ImageEditModel:
    """A vetted, immutable source snapshot used by the image-edit runners."""

    name: str
    repo_id: str
    revision: str
    license: str
    destination_name: str
    files: tuple[SnapshotFile, ...]
    source_url: str
    automatic_download: bool = True
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.name.replace("_", "").isalnum():
            raise ValueError(f"Unsafe model name: {self.name}")
        if "/" not in self.repo_id or self.repo_id.startswith("/"):
            raise ValueError(f"Invalid Hugging Face repository: {self.repo_id}")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError(f"Revision must be a full Git commit: {self.revision}")
        _validate_relative_path(self.destination_name)
        names = [item.path for item in self.files]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate allowlist entry for {self.name}")
        if self.automatic_download and not self.files:
            raise ValueError(f"Automatic model {self.name} has no files")


def _validate_relative_path(value: str) -> None:
    candidate = Path(value)
    if not value or candidate.is_absolute() or "\\" in value or ".." in candidate.parts:
        raise ValueError(f"Unsafe relative path: {value!r}")


def _snapshot_file(path: str, size_bytes: int, sha256: str) -> SnapshotFile:
    return SnapshotFile(path=path, size_bytes=size_bytes, sha256=sha256)


def _config_files(paths: Iterable[str]) -> tuple[SnapshotFile, ...]:
    return tuple(SnapshotFile(path) for path in paths)


_QWEN_CONFIG_FILES = _config_files(
    (
        "model_index.json",
        "processor/added_tokens.json",
        "processor/chat_template.jinja",
        "processor/merges.txt",
        "processor/preprocessor_config.json",
        "processor/special_tokens_map.json",
        "processor/tokenizer_config.json",
        "processor/video_preprocessor_config.json",
        "processor/vocab.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/generation_config.json",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/added_tokens.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/config.json",
    )
)

_QWEN_TEXT_ENCODER_FILES = (
    _snapshot_file(
        "text_encoder/model-00001-of-00004.safetensors",
        4_968_243_304,
        "d725335e4ea2399be706469e4b8807716a8fa64bd03468252e9f7acf2415fee4",
    ),
    _snapshot_file(
        "text_encoder/model-00002-of-00004.safetensors",
        4_991_495_816,
        "b1830db6908dcc76df3a71492acbcf2b8cac130114cf1f3c2d9edae8de8c6de3",
    ),
    _snapshot_file(
        "text_encoder/model-00003-of-00004.safetensors",
        4_932_751_040,
        "09c1807c6d00d7cab94f7db39d4c02ebb8537225ccde383861ac48db97945aa6",
    ),
    _snapshot_file(
        "text_encoder/model-00004-of-00004.safetensors",
        1_691_924_384,
        "5dd068336d14d45ffb43cef374d286cc6ba9d8741b028f90a7d040d847961f4a",
    ),
    _snapshot_file(
        "processor/tokenizer.json",
        11_421_896,
        "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa",
    ),
)

_QWEN_VAE = _snapshot_file(
    "vae/diffusion_pytorch_model.safetensors",
    253_806_966,
    "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344",
)


IMAGE_EDIT_MODELS: dict[str, ImageEditModel] = {
    "firered_image_edit_1_1": ImageEditModel(
        name="firered_image_edit_1_1",
        repo_id="FireRedTeam/FireRed-Image-Edit-1.1",
        revision="3bc3f2a12722fd9883eb6357500de191d56baaf5",
        license="apache-2.0",
        destination_name="firered-image-edit-1.1",
        source_url="https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1",
        files=(
            *_QWEN_CONFIG_FILES,
            *_QWEN_TEXT_ENCODER_FILES,
            _snapshot_file(
                "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
                9_973_578_592,
                "cd6f0d78a3a8c21792538d0abae604bd7abbca1508a2e8c778ea359f5fabd180",
            ),
            _snapshot_file(
                "transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
                9_987_326_072,
                "bb6b283ea5954aa16e8df94fbbd37368c48c07ff3cfcf3a514117333c3753463",
            ),
            _snapshot_file(
                "transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
                9_987_307_440,
                "e0602f9a002cf2807080bfb6d055cbcd6991887be7e89c82fa9411f938195923",
            ),
            _snapshot_file(
                "transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
                9_930_685_712,
                "ae1d1ec1a35f5f59b086c1947dbf62d67d972d25cf7c771640921c9aa97ee492",
            ),
            _snapshot_file(
                "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
                982_130_472,
                "9d304d6539e7dad0647346cecf68d1fd5dc0efc7319626cdde2fdfc3e1533417",
            ),
            _QWEN_VAE,
        ),
    ),
    "qwen_image_edit_2511": ImageEditModel(
        name="qwen_image_edit_2511",
        repo_id="Qwen/Qwen-Image-Edit-2511",
        revision="6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
        license="apache-2.0",
        destination_name="qwen-image-edit-2511",
        source_url="https://huggingface.co/Qwen/Qwen-Image-Edit-2511",
        files=(
            *_QWEN_CONFIG_FILES,
            *_QWEN_TEXT_ENCODER_FILES,
            _snapshot_file(
                "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
                9_973_578_592,
                "2a0c30c9ba44a5f11c21ca139e37951430bbde814ff4e0b5b1a68b80530e7a1a",
            ),
            _snapshot_file(
                "transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
                9_987_326_072,
                "54ec249b07b4376e19cf16b764054f03ca03ae2cfbd9939453e2085f4e9bd259",
            ),
            _snapshot_file(
                "transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
                9_987_307_440,
                "c55157843525653161e8f6af5acc670ba3aceff04284f7cf657199d24d065e16",
            ),
            _snapshot_file(
                "transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
                9_930_685_712,
                "ffcfb5a4895702635890a67bad183591e0ae515d794bdcb26e217b27a7f6d12d",
            ),
            _snapshot_file(
                "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
                982_130_472,
                "2b2556b736629e10a5a0dfa14606f2057f4f81c2ba53f94103682c7ac42d4940",
            ),
            _QWEN_VAE,
        ),
    ),
    "flux2_klein_4b": ImageEditModel(
        name="flux2_klein_4b",
        repo_id="black-forest-labs/FLUX.2-klein-4B",
        revision="e7b7dc27f91deacad38e78976d1f2b499d76a294",
        license="apache-2.0",
        destination_name="flux2-klein-4b",
        source_url="https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
        files=(
            *_config_files(
                (
                    "model_index.json",
                    "scheduler/scheduler_config.json",
                    "text_encoder/config.json",
                    "text_encoder/generation_config.json",
                    "text_encoder/model.safetensors.index.json",
                    "tokenizer/added_tokens.json",
                    "tokenizer/chat_template.jinja",
                    "tokenizer/merges.txt",
                    "tokenizer/special_tokens_map.json",
                    "tokenizer/tokenizer_config.json",
                    "tokenizer/vocab.json",
                    "transformer/config.json",
                    "vae/config.json",
                )
            ),
            _snapshot_file(
                "text_encoder/model-00001-of-00002.safetensors",
                4_967_215_360,
                "8c0506e7f4936fa7e26183a4fd8da4e2bdbc5990ba64ae441f965d51228f36ea",
            ),
            _snapshot_file(
                "text_encoder/model-00002-of-00002.safetensors",
                3_077_766_632,
                "82f2bd839378541b0557bfabaf37c7d3d637071fdcb73302dedd7cf61162ce07",
            ),
            _snapshot_file(
                "tokenizer/tokenizer.json",
                11_422_654,
                "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
            ),
            _snapshot_file(
                "transformer/diffusion_pytorch_model.safetensors",
                7_751_109_744,
                "9f29f9edcfdae452a653ffb51a534ca4decd389952c225724ff3b94042612a6e",
            ),
            _snapshot_file(
                "vae/diffusion_pytorch_model.safetensors",
                168_120_878,
                "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04",
            ),
        ),
    ),
    "control_color": ImageEditModel(
        name="control_color",
        repo_id="ZhexinLiang/Control-Color",
        revision="f21054af54f524591f7a3c0862fa90392f7d33c0",
        license="unpublished",
        destination_name="control-color",
        source_url="https://github.com/ZhexinLiang/Control-Color",
        files=(),
        automatic_download=False,
        skip_reason=(
            "The authors publish weights only through a Google Drive folder and "
            "do not publish file hashes or a model-weight license. The downloader "
            "refuses an unverifiable automatic download."
        ),
    ),
}


def sha256_file(path: Path, chunk_size: int = _CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_resolve_url(model: ImageEditModel, file: SnapshotFile) -> str:
    """Return the pinned Hugging Face resolver URL for one allowlisted file."""

    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in model.repo_id.split("/"))
    quoted_path = urllib.parse.quote(file.path, safe="/")
    return f"{HUGGING_FACE_ORIGIN}/{quoted_repo}/resolve/{model.revision}/{quoted_path}"


def verify_snapshot_file(path: Path, file: SnapshotFile) -> None:
    if not path.is_file():
        raise ValueError(f"Missing model file: {path}")
    actual_size = path.stat().st_size
    if file.size_bytes is not None and actual_size != file.size_bytes:
        raise ValueError(
            f"Unexpected size for {path.name}: expected {file.size_bytes}, got {actual_size}"
        )
    if file.sha256 is not None:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != file.sha256:
            raise ValueError(
                f"Checksum mismatch for {path.name}: expected {file.sha256}, got {actual_sha256}"
            )


def _is_trusted_redirect(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and any(
        host == suffix or host.endswith("." + suffix)
        for suffix in _TRUSTED_DOWNLOAD_HOST_SUFFIXES
    )


def _part_path(path: Path) -> Path:
    return path.with_name(path.name + ".part")


def download_snapshot_file(
    model: ImageEditModel,
    file: SnapshotFile,
    destination: Path,
    *,
    force: bool = False,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
    max_attempts: int = 12,
) -> bool:
    """Download one file with Range-resume and an atomic final rename.

    Returns ``True`` when bytes were fetched and ``False`` when an existing,
    verified file was reused. A failed transfer leaves only ``.part`` data.
    """

    _validate_relative_path(file.path)
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        verify_snapshot_file(destination, file)
        return False
    if force:
        destination.unlink(missing_ok=True)

    part = _part_path(destination)
    if part.exists() and not part.is_file():
        raise ValueError(f"Download part path is not a file: {part}")
    if file.size_bytes is not None and part.exists() and part.stat().st_size > file.size_bytes:
        part.unlink()
    if file.size_bytes is not None and part.exists() and part.stat().st_size == file.size_bytes:
        verify_snapshot_file(part, file)
        part.replace(destination)
        return True

    start = part.stat().st_size if part.exists() else 0
    url = build_resolve_url(model, file)
    headers = {"User-Agent": "ColorIt/0.1"}
    if start:
        headers["Range"] = f"bytes={start}-"

    request = urllib.request.Request(url, headers=headers)
    with urlopen(request) as response:
        final_url = response.geturl()
        if not _is_trusted_redirect(final_url):
            raise ValueError(f"Model download redirected outside trusted HTTPS hosts: {final_url}")
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()
        if start and status != 206:
            # The Hub normally honors Range requests. A mirror that does not is
            # safe to use only after discarding the incomplete prefix.
            part.unlink(missing_ok=True)
            if max_attempts == 1:
                raise ValueError(f"Server refused Range retries for {file.path}")
            return download_snapshot_file(
                model,
                file,
                destination,
                force=False,
                urlopen=urlopen,
                max_attempts=max_attempts - 1,
            )
        mode = "ab" if start else "wb"
        with part.open(mode) as handle:
            while chunk := response.read(_CHUNK_SIZE):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())

    if file.size_bytes is not None and part.stat().st_size < file.size_bytes:
        if max_attempts == 1:
            raise ValueError(
                f"Download remained incomplete after retries for {file.path}: "
                f"expected {file.size_bytes}, got {part.stat().st_size}"
            )
        return download_snapshot_file(
            model,
            file,
            destination,
            force=False,
            urlopen=urlopen,
            max_attempts=max_attempts - 1,
        )
    verify_snapshot_file(part, file)
    part.replace(destination)
    return True


def _safe_destination(root: Path, relative_path: str) -> Path:
    _validate_relative_path(relative_path)
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Model destination escapes root: {relative_path}")
    return candidate


def snapshot_manifest(model: ImageEditModel, destination: Path) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_VERSION,
        "model": model.name,
        "repo_id": model.repo_id,
        "revision": model.revision,
        "license": model.license,
        "source_url": model.source_url,
        "destination": str(destination),
        "files": [
            {
                "path": item.path,
                "size_bytes": destination.joinpath(item.path).stat().st_size,
                "sha256": sha256_file(destination / item.path),
                "source_url": build_resolve_url(model, item),
            }
            for item in model.files
        ],
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False, prefix=path.name + ".", suffix=".part"
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_model_snapshot(model: ImageEditModel, destination: Path) -> None:
    """Verify every allowlisted file without accepting extras as model inputs."""

    if not model.automatic_download:
        raise ValueError(model.skip_reason or f"{model.name} is not automatically downloadable")
    for file in model.files:
        verify_snapshot_file(_safe_destination(destination, file.path), file)


def download_model_snapshot(
    model_name: str,
    models_dir: Path,
    *,
    manifest_path: Path | None = None,
    force: bool = False,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Fetch and record one complete image-edit model snapshot.

    ``models_dir`` is the containing directory, usually ``models/image-edit``.
    The per-model completion manifest is written only after every allowlisted
    file verifies. Interrupted downloads remain resumable ``.part`` files.
    """

    try:
        model = IMAGE_EDIT_MODELS[model_name]
    except KeyError as exc:
        choices = ", ".join(sorted(IMAGE_EDIT_MODELS))
        raise ValueError(f"Unknown image-edit model {model_name!r}. Choices: {choices}") from exc
    if not model.automatic_download:
        raise ValueError(model.skip_reason or f"{model.name} is not automatically downloadable")

    destination = _safe_destination(models_dir, model.destination_name)
    destination.mkdir(parents=True, exist_ok=True)
    for file in model.files:
        target = _safe_destination(destination, file.path)
        download_snapshot_file(model, file, target, force=force, urlopen=urlopen)

    verify_model_snapshot(model, destination)
    manifest = snapshot_manifest(model, destination)
    _atomic_write_json(destination / "snapshot.manifest.json", manifest)
    if manifest_path is not None:
        _record_snapshot_in_manifest(manifest_path, manifest)
    return manifest


def _record_snapshot_in_manifest(manifest_path: Path, snapshot: dict[str, Any]) -> None:
    payload: dict[str, Any] = {"schema_version": MANIFEST_VERSION, "image_edit_models": {}}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if not isinstance(existing, dict):
            raise ValueError(f"Invalid weights manifest: {manifest_path}")
        payload.update(existing)
        payload.setdefault("schema_version", MANIFEST_VERSION)
        payload.setdefault("image_edit_models", {})
    entries = payload["image_edit_models"]
    if not isinstance(entries, dict):
        raise ValueError(f"Invalid image_edit_models in {manifest_path}")
    entries[snapshot["model"]] = {
        **snapshot,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    _atomic_write_json(manifest_path, payload)


def discard_partial_downloads(models_dir: Path, model_name: str) -> int:
    """Remove only unverified ``.part`` files for one known model snapshot."""

    model = IMAGE_EDIT_MODELS.get(model_name)
    if model is None:
        raise ValueError(f"Unknown image-edit model {model_name!r}")
    destination = _safe_destination(models_dir, model.destination_name)
    removed = 0
    for file in model.files:
        part = _part_path(_safe_destination(destination, file.path))
        if part.exists():
            if not part.is_file():
                raise ValueError(f"Refusing to remove non-file download part: {part}")
            part.unlink()
            removed += 1
    return removed
