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


_BONSAI_COMMON_FILES = (
    _snapshot_file("LICENSE", 10_174, "69849221bfb90053de2134ef5e6d540287b4b98062326492f1f96f5da685524b"),
    _snapshot_file("NOTICE.md", 623, "bbefa4a26b836efc040c1a0f155a425d1d833eae1b2534ffc414b1eada3cd922"),
    _snapshot_file("model_index.json", 81, "ecb4735e37691a8733f62957fd6c548f841af40624905bb24fa639756153c8a1"),
    _snapshot_file("scheduler/scheduler_config.json", 486, "067afb012cef64553a763447d1efd93daeffcc0123ca7e25b09f8de20b90762e"),
    _snapshot_file("text_encoder-mlx-4bit/added_tokens.json", 707, "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680"),
    _snapshot_file("text_encoder-mlx-4bit/config.json", 937, "b5efdcf3b0035a3638e7228dad4d85f5c4a23f156eb7cdb0b44c8366a5d34d9b"),
    _snapshot_file("text_encoder-mlx-4bit/merges.txt", 1_671_853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    _snapshot_file("text_encoder-mlx-4bit/model.safetensors", 2_263_022_529, "e240c0bdc0ebb0681bf0da0f98d9719fd6ebe269a3633f81542c13e81345651d"),
    _snapshot_file("text_encoder-mlx-4bit/model.safetensors.index.json", 63_924, "f7825defe5865d179c3b593173d37056be5f202dcb7153985cf74e75ecf1628b"),
    _snapshot_file("text_encoder-mlx-4bit/special_tokens_map.json", 613, "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd"),
    _snapshot_file("text_encoder-mlx-4bit/tokenizer.json", 11_422_654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    _snapshot_file("text_encoder-mlx-4bit/tokenizer_config.json", 9_706, "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0"),
    _snapshot_file("text_encoder-mlx-4bit/vocab.json", 2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    _snapshot_file("tokenizer/added_tokens.json", 707, "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680"),
    _snapshot_file("tokenizer/chat_template.jinja", 4_168, "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"),
    _snapshot_file("tokenizer/merges.txt", 1_671_853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    _snapshot_file("tokenizer/special_tokens_map.json", 613, "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd"),
    _snapshot_file("tokenizer/tokenizer.json", 11_422_654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    _snapshot_file("tokenizer/tokenizer_config.json", 5_404, "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0"),
    _snapshot_file("tokenizer/vocab.json", 2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    _snapshot_file("transformer-packed-mflux/config.json", 619, "14c6d8314d28cc027ce636d52dfb98cecc11b65c1455bd51b394a971f4b7b49e"),
    _snapshot_file("vae/config.json", 821, "0d6dfb69ae95a5e2ac9836284bbb63d8b38ce67b25ba2dff380752b2a10ab948"),
    _snapshot_file("vae/diffusion_pytorch_model.safetensors", 168_120_878, "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04"),
)


IMAGE_EDIT_MODELS: dict[str, ImageEditModel] = {
    "bonsai_image_binary_4b_mlx_1bit": ImageEditModel(
        name="bonsai_image_binary_4b_mlx_1bit",
        repo_id="prism-ml/bonsai-image-binary-4B-mlx-1bit",
        revision="d1b3ac11a7f1ba61d84b277339daeeed4a98e0e2",
        license="apache-2.0",
        destination_name="bonsai-image-binary-4b-mlx-1bit",
        source_url="https://huggingface.co/prism-ml/bonsai-image-binary-4B-mlx-1bit",
        files=(
            *_BONSAI_COMMON_FILES,
            _snapshot_file("manifest.json", 4_617, "898f6743600667071a354bd4fc98004afb5f20703e59e267b86e86d62937d061"),
            _snapshot_file("README.md", 11_996, "1527258bbdd58161a3241245985c78ad5503635fd32bc2d0d37634c192a574e5"),
            _snapshot_file("transformer-packed-mflux/diffusion_pytorch_model.safetensors", 965_208_136, "1792b31d857d95fcbe32df8e6d2fc96b30e800a195e295565d033deccea2dd75"),
            _snapshot_file("transformer-packed-mflux/quantization_config.json", 5_054, "ff8e78812e547f25868eff7b9a86cbcf7a91bee95f81f2bf0e039f198dbbabf0"),
        ),
    ),
    "bonsai_image_ternary_4b_mlx_2bit": ImageEditModel(
        name="bonsai_image_ternary_4b_mlx_2bit",
        repo_id="prism-ml/bonsai-image-ternary-4B-mlx-2bit",
        revision="2c24c81b934a658ba5590cf39088ba929985b4a8",
        license="apache-2.0",
        destination_name="bonsai-image-ternary-4b-mlx-2bit",
        source_url="https://huggingface.co/prism-ml/bonsai-image-ternary-4B-mlx-2bit",
        files=(
            *_BONSAI_COMMON_FILES,
            _snapshot_file("manifest.json", 4_619, "a82ee88186754b17e7796d3d0130a6ead7e51198f0a61d3f936fab71a3eba178"),
            _snapshot_file("README.md", 12_443, "4017f9c74fc1f89212a8b736a29bcc742fe1fe4679fbeb1a5818bf395f80d55e"),
            _snapshot_file("transformer-packed-mflux/diffusion_pytorch_model.safetensors", 1_425_271_472, "b21737bdf02690b7d662907781c4dc8b8bf22a2c98b823b1ca3336f48371a84f"),
            _snapshot_file("transformer-packed-mflux/quantization_config.json", 5_054, "6a792a07051e534b177aefaac5222796ec13bbdd1a597a2b08695b4c6c75fec7"),
        ),
    ),
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
) -> bool:
    """Download one file with Range-resume and an atomic final rename.

    Returns ``True`` when bytes were fetched and ``False`` when an existing,
    verified file was reused. A failed transfer leaves only ``.part`` data.
    """

    _validate_relative_path(file.path)
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
            return download_snapshot_file(model, file, destination, force=False, urlopen=urlopen)
        mode = "ab" if start else "wb"
        with part.open(mode) as handle:
            while chunk := response.read(_CHUNK_SIZE):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())

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
