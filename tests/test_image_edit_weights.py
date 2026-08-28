from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.pipeline import image_edit_weights as weights


class _Response:
    def __init__(self, payload: bytes, *, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.position = 0

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def geturl(self) -> str:
        return "https://us.aws.cdn.hf.co/xet-bridge-us/verified-object"

    def getcode(self) -> int:
        return self.status

    def read(self, size: int) -> bytes:
        chunk = self.payload[self.position : self.position + size]
        self.position += len(chunk)
        return chunk


def _model_with_file(payload: bytes) -> tuple[weights.ImageEditModel, weights.SnapshotFile]:
    file = weights.SnapshotFile(
        path="subdir/model.safetensors",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    return (
        weights.ImageEditModel(
            name="test_model",
            repo_id="owner/test-model",
            revision="a" * 40,
            license="apache-2.0",
            destination_name="test-model",
            files=(file,),
            source_url="https://huggingface.co/owner/test-model",
        ),
        file,
    )


class ImageEditWeightsTest(unittest.TestCase):
    def test_registry_uses_immutable_official_revisions_and_lfs_checksums(self) -> None:
        automatic = [model for model in weights.IMAGE_EDIT_MODELS.values() if model.automatic_download]
        self.assertEqual(
            {model.name for model in automatic},
            {
                "bonsai_image_binary_4b_mlx_1bit",
                "bonsai_image_ternary_4b_mlx_2bit",
                "firered_image_edit_1_1",
                "flux2_klein_4b",
                "qwen_image_edit_2511",
            },
        )
        for model in automatic:
            self.assertTrue(model.source_url.startswith("https://huggingface.co/"))
            self.assertEqual(len(model.revision), 40)
            self.assertEqual(len({file.path for file in model.files}), len(model.files))
            self.assertTrue(any(file.size_bytes and file.size_bytes > 1_000_000_000 for file in model.files))
            for file in model.files:
                if file.size_bytes is not None:
                    self.assertTrue(file.sha256 and len(file.sha256) == 64)
                    self.assertIn(model.revision, weights.build_resolve_url(model, file))

    def test_download_is_verified_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = b"verified model bytes"
            model, file = _model_with_file(payload)
            root = Path(directory)
            target = root / file.path
            fetched = weights.download_snapshot_file(
                model, file, target, urlopen=lambda _request: _Response(payload)
            )
            self.assertTrue(fetched)
            self.assertEqual(target.read_bytes(), payload)
            self.assertFalse((root / "subdir/model.safetensors.part").exists())

    def test_download_resumes_a_verified_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = b"verified model bytes"
            model, file = _model_with_file(payload)
            target = Path(directory) / file.path
            target.parent.mkdir(parents=True)
            partial = target.with_name(target.name + ".part")
            partial.write_bytes(payload[:8])
            requests = []

            def opener(request):
                requests.append(request)
                return _Response(payload[8:], status=206)

            weights.download_snapshot_file(model, file, target, urlopen=opener)
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(requests[0].get_header("Range"), "bytes=8-")

    def test_download_retries_a_truncated_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = b"verified model bytes"
            model, file = _model_with_file(payload)
            target = Path(directory) / file.path
            requests = []

            def opener(request):
                requests.append(request)
                if len(requests) == 1:
                    return _Response(payload[:8])
                return _Response(payload[8:], status=206)

            weights.download_snapshot_file(model, file, target, urlopen=opener)
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(requests[1].get_header("Range"), "bytes=8-")

    def test_complete_part_is_verified_without_an_invalid_range_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = b"verified model bytes"
            model, file = _model_with_file(payload)
            target = Path(directory) / file.path
            target.parent.mkdir(parents=True)
            target.with_name(target.name + ".part").write_bytes(payload)

            weights.download_snapshot_file(
                model,
                file,
                target,
                urlopen=lambda _request: self.fail("complete part must not request the network"),
            )
            self.assertEqual(target.read_bytes(), payload)

    def test_checksum_failure_leaves_only_resumable_part(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = b"verified model bytes"
            model, file = _model_with_file(payload)
            target = Path(directory) / file.path
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                weights.download_snapshot_file(
                    model, file, target, urlopen=lambda _request: _Response(b"x" * len(payload))
                )
            self.assertFalse(target.exists())
            self.assertTrue(target.with_name(target.name + ".part").exists())

    def test_complete_snapshot_writes_both_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = b"verified model bytes"
            model, _file = _model_with_file(payload)
            registry = dict(weights.IMAGE_EDIT_MODELS)
            registry[model.name] = model
            root = Path(directory)
            global_manifest = root / "weights.json"
            with patch.dict(weights.IMAGE_EDIT_MODELS, registry, clear=True):
                snapshot = weights.download_model_snapshot(
                    model.name,
                    root / "models",
                    manifest_path=global_manifest,
                    urlopen=lambda _request: _Response(payload),
                )
            model_manifest = root / "models/test-model/snapshot.manifest.json"
            self.assertEqual(json.loads(model_manifest.read_text())["revision"], model.revision)
            self.assertEqual(
                json.loads(global_manifest.read_text())["image_edit_models"][model.name]["files"],
                snapshot["files"],
            )


if __name__ == "__main__":
    unittest.main()
