from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from .gpu_guard import supervise


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Path
    output: Path
    width: int = Field(default=640, ge=384, le=1920)
    enhance_prompt: bool = False
    cache_text_weights: bool = False


def run(options: Options) -> int:
    folder = options.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    protocol = folder / "protocol.json"
    if protocol.exists():
        raise FileExistsError(f"Existing frozen run: {protocol}")
    candidates = folder / "candidates"
    selected = folder / "selected"
    references = folder / "references"
    commands = [
        ["tmp/havc_cpu/.venv/bin/python", "-u", "-m", "experiments.adapter_compare.havc_sources",
         "--source", str(options.source.resolve()), "--output", str(candidates), "--mode", "automatic", "--gui"],
        ["tmp/havc_cpu/.venv/bin/python", "-u", "-m", "experiments.adapter_compare.havc_selection",
         "--input-manifest", str(candidates / "input_manifest.json"), "--output", str(selected)],
        ["tmp/adapter_compare_90_120/havc_env/bin/python", "-u", "-m", "experiments.adapter_compare.havc_image",
         "--input-manifest", str(selected / "input_manifest.json"), "--output-dir", str(references),
         "--reference-manifest", str(folder / "references.json"), "--backend", "mlx", "--steps", "6"],
        [".venv/bin/python", "-u", "-m", "experiments.adapter_compare.havc_video",
         "--source", str(options.source.resolve()), "--references", str(references),
         "--manifest", str(folder / "references.json"), "--output", str(folder / "output.mp4"),
         "--mode", "automatic", "--width", str(options.width)],
    ]
    if options.enhance_prompt:
        commands[2].append("--enhance-prompt")
    if options.cache_text_weights:
        commands[2].append("--cache-text-weights")
    protocol.write_text(json.dumps({**options.model_dump(mode="json"), "commands": commands,
        "automatic_policy": "Frozen whole-clip GUI extraction, native .95/50 DINOv3 dedup, one fixed generic prompt and seed42, single-image native six-step Qwen2.1/Viggle, continuous CMNET2 memory. No reference or palette curation after launch.",
        "upstream": "HAVC 6accc5d image backend + 666b7d4 bundled vscmnet2 1.2.1 selection",
        "manual_intervention_during_run": False}, indent=2) + "\n")
    started = perf_counter()
    for index, command in enumerate(commands):
        if index < 2:
            subprocess.run(command, check=True)
        else:
            code = supervise(command, folder / f"stage_{index}_guard.json")
            if code:
                return code
    (folder / "result.json").write_text(json.dumps({"seconds": perf_counter() - started,
        "output": str(folder / "output.mp4")}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--enhance-prompt", action="store_true")
    parser.add_argument("--cache-text-weights", action="store_true")
    raise SystemExit(run(Options.model_validate(vars(parser.parse_args()))))
