from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .havc_image import InputReference


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_manifest: Path
    output: Path
    upstream: Path = Path("tmp/adapter_compare_90_120/steelman/upstream/wheel")
    threshold: float = Field(default=0.95, gt=0, le=1)


def select(options: Options) -> None:
    import torch

    torch.set_num_threads(4)
    for name, relative in [
        ("vscmnet2", "vscmnet2"),
        ("vscmnet2.colormnet2", "vscmnet2/colormnet2"),
        ("vscmnet2.colormnet2.model", "vscmnet2/colormnet2/model"),
    ]:
        package = types.ModuleType(name)
        package.__path__ = [str(options.upstream.resolve() / relative)]
        sys.modules[name] = package
    selector = importlib.import_module("vscmnet2.cmnet2_refselect")
    content = json.loads(options.input_manifest.read_text())
    records = [InputReference.model_validate(item) for item in content["references"]]
    candidates = options.input_manifest.parent
    if any(item.path.parent != candidates.resolve() for item in records):
        raise ValueError("Selection needs original-resolution candidates in the manifest directory")
    result = selector.select_reference_frames(
        ref_framedir=str(candidates), out_framedir=str(options.output),
        similarity_threshold=options.threshold, select_window=50,
        input_size=224, batch_size=4, device="cpu", debug_html=True,
    )
    selected = {int(path.stem.split("_")[1]) for path in options.output.glob("ref_*.*")}
    content["references"] = [item.model_dump(mode="json") for item in records if item.frame in selected]
    content["reference_count"] = len(selected)
    content["selection"] = result
    (options.output / "input_manifest.json").write_text(json.dumps(content, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Options.model_fields["upstream"].default)
    parser.add_argument("--threshold", type=float, default=0.95)
    select(Options.model_validate(vars(parser.parse_args())))
