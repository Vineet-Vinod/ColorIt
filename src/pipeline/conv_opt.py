from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from src.pipeline.config import AppConfig
from src.pipeline.manifest import write_json_manifest
from src.pipeline.model_loader import IMAGENET_MEAN, IMAGENET_STD, load_colorizer_bundle, select_device


DEFAULT_OPTIMIZE_CONFIG_PATH = Path("optimize/configs/default.yaml")


@dataclass(frozen=True)
class ConvCase:
    case_id: str
    module_name: str
    op_type: str
    stage: str
    input_shape: list[int]
    output_shape: list[int]
    weight_shape: list[int]
    has_bias: bool
    stride: list[int]
    padding: list[int]
    dilation: list[int]
    groups: int
    output_padding: list[int]
    calls: int
    macs_per_call: int
    total_macs: int


def run_extract_conv_shapes(
    *,
    config: AppConfig,
    config_path: Path,
    input_path: Path,
    output_path: Path | None,
    max_frames: int | None,
    max_batches: int | None,
) -> int:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input clip not found: {input_path}")

    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else (config.path.parent.parent / "optimize/artifacts/conv_shapes.json").resolve()
    )

    batch_size = max(1, int(config.raw.get("runtime", {}).get("inference_batch_size", 1)))
    render_factor = int(config.model["render_factor"])
    bundle = load_colorizer_bundle(config)

    collector = ConvShapeCollector()
    hooks = []
    for module_name, module in bundle.model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(module.register_forward_hook(collector.make_hook(module_name, module)))

    try:
        for batch_tensor in _iter_preprocessed_batches(
            input_path=input_path,
            batch_size=batch_size,
            render_factor=render_factor,
            max_frames=max_frames,
            max_batches=max_batches,
            device=bundle.device,
        ):
            with torch.no_grad():
                _ = bundle.model(batch_tensor)
    finally:
        for hook in hooks:
            hook.remove()

    cases = collector.to_cases()
    payload = {
        "config_path": str(config_path.resolve()),
        "input_path": str(input_path),
        "backend": bundle.backend,
        "render_factor": render_factor,
        "batch_size": batch_size,
        "max_frames": max_frames,
        "max_batches": max_batches,
        "case_count": len(cases),
        "cases": [asdict(case) for case in cases],
    }
    write_json_manifest(output_path, payload)

    print(f"Convolution shape manifest written: {output_path}")
    print(f"Cases captured: {len(cases)}")
    for case in cases[:10]:
        print(
            f"{case.case_id}: {case.op_type} {tuple(case.input_shape)} -> {tuple(case.output_shape)}"
            f" weight={tuple(case.weight_shape)} calls={case.calls} total_macs={case.total_macs}"
        )
    return 0


def run_benchmark_convs(
    *,
    config: AppConfig,
    config_path: Path,
    optimize_config_path: Path | None,
    shapes_manifest_path: Path,
    candidate_path: Path | None,
    output_path: Path | None,
    top_k: int | None,
    case_ids: list[str],
    warmup_iterations: int | None,
    timed_iterations: int | None,
    correctness_trials: int | None,
    atol: float | None,
    rtol: float | None,
    seed: int | None,
) -> int:
    shapes_manifest_path = shapes_manifest_path.expanduser().resolve()
    if not shapes_manifest_path.exists():
        raise FileNotFoundError(f"Convolution shape manifest not found: {shapes_manifest_path}")

    optimize_config = _load_optimize_config(
        optimize_config_path.expanduser().resolve() if optimize_config_path is not None else DEFAULT_OPTIMIZE_CONFIG_PATH.resolve()
    )
    bench_config = optimize_config.get("benchmark", {})

    warmup = int(warmup_iterations if warmup_iterations is not None else bench_config.get("warmup_iterations", 25))
    iterations = int(timed_iterations if timed_iterations is not None else bench_config.get("timed_iterations", 100))
    trials = int(correctness_trials if correctness_trials is not None else bench_config.get("correctness_trials", 5))
    atol_value = float(atol if atol is not None else bench_config.get("atol", 5e-4))
    rtol_value = float(rtol if rtol is not None else bench_config.get("rtol", 5e-3))
    seed_value = int(seed if seed is not None else bench_config.get("seed", 1234))
    top_k_value = int(top_k if top_k is not None else bench_config.get("top_k", 10))

    payload = json.loads(shapes_manifest_path.read_text())
    all_cases = [ConvCase(**case) for case in payload["cases"]]
    selected_cases = _select_cases(all_cases, top_k=top_k_value, case_ids=case_ids)
    if not selected_cases:
        raise ValueError("No convolution cases selected for benchmarking.")

    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else (config.path.parent.parent / "optimize/artifacts/conv_benchmark_results.json").resolve()
    )

    device, backend = select_device(config)
    candidate = _load_candidate(candidate_path.expanduser().resolve()) if candidate_path is not None else None
    results = []
    for case in selected_cases:
        result = _benchmark_case(
            case=case,
            device=device,
            backend=backend,
            candidate=candidate,
            warmup_iterations=warmup,
            timed_iterations=iterations,
            correctness_trials=trials,
            atol=atol_value,
            rtol=rtol_value,
            seed=seed_value,
        )
        results.append(result)
        candidate_fragment = (
            f" candidate_ms={result['candidate']['median_ms']:.4f} speedup={result['candidate']['speedup_vs_baseline']:.4f}"
            if result.get("candidate") and result["candidate"].get("timed")
            else ""
        )
        correctness_status = result["correctness"]["status"] if result.get("correctness") else "n/a"
        print(
            f"{case.case_id}: baseline_ms={result['baseline']['median_ms']:.4f}"
            f"{candidate_fragment} correctness={correctness_status}"
        )

    final_payload = {
        "config_path": str(config_path.resolve()),
        "optimize_config_path": str(
            (
                optimize_config_path.expanduser().resolve()
                if optimize_config_path is not None
                else DEFAULT_OPTIMIZE_CONFIG_PATH.resolve()
            )
        ),
        "backend": backend,
        "device": str(device),
        "shapes_manifest_path": str(shapes_manifest_path),
        "candidate_path": str(candidate_path.expanduser().resolve()) if candidate_path is not None else None,
        "candidate_name": candidate.name if candidate is not None else None,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "correctness_trials": trials,
        "atol": atol_value,
        "rtol": rtol_value,
        "seed": seed_value,
        "result_count": len(results),
        "results": results,
    }
    write_json_manifest(output_path, final_payload)
    print(f"Convolution benchmark results written: {output_path}")
    return 0


class ConvShapeCollector:
    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def make_hook(self, module_name: str, module: nn.Module):
        def hook(_, inputs, output):
            if not inputs:
                return
            input_tensor = inputs[0]
            if not isinstance(input_tensor, torch.Tensor) or not isinstance(output, torch.Tensor):
                return

            op_type = "conv_transpose2d" if isinstance(module, nn.ConvTranspose2d) else "conv2d"
            input_shape = list(input_tensor.shape)
            output_shape = list(output.shape)
            weight = module.weight.detach()
            weight_shape = list(weight.shape)
            key_payload = {
                "module_name": module_name,
                "op_type": op_type,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "weight_shape": weight_shape,
                "has_bias": module.bias is not None,
                "stride": list(module.stride),
                "padding": list(module.padding),
                "dilation": list(module.dilation),
                "groups": int(module.groups),
                "output_padding": list(getattr(module, "output_padding", (0, 0))),
            }
            key = json.dumps(key_payload, sort_keys=True)
            record = self._records.setdefault(
                key,
                {
                    **key_payload,
                    "calls": 0,
                    "macs_per_call": _estimate_conv_macs(
                        op_type=op_type,
                        input_shape=input_shape,
                        output_shape=output_shape,
                        weight_shape=weight_shape,
                        groups=int(module.groups),
                    ),
                    "stage": _classify_stage(module_name),
                },
            )
            record["calls"] += 1

        return hook

    def to_cases(self) -> list[ConvCase]:
        ranked_records = sorted(
            self._records.values(),
            key=lambda value: (value["macs_per_call"] * value["calls"], value["calls"]),
            reverse=True,
        )
        cases: list[ConvCase] = []
        for index, record in enumerate(ranked_records, start=1):
            total_macs = int(record["macs_per_call"] * record["calls"])
            cases.append(
                ConvCase(
                    case_id=f"conv_case_{index:03d}",
                    module_name=record["module_name"],
                    op_type=record["op_type"],
                    stage=record["stage"],
                    input_shape=list(record["input_shape"]),
                    output_shape=list(record["output_shape"]),
                    weight_shape=list(record["weight_shape"]),
                    has_bias=bool(record["has_bias"]),
                    stride=list(record["stride"]),
                    padding=list(record["padding"]),
                    dilation=list(record["dilation"]),
                    groups=int(record["groups"]),
                    output_padding=list(record["output_padding"]),
                    calls=int(record["calls"]),
                    macs_per_call=int(record["macs_per_call"]),
                    total_macs=total_macs,
                )
            )
        return cases


@dataclass(frozen=True)
class Candidate:
    name: str
    module: ModuleType
    path: Path


def _load_candidate(path: Path) -> Candidate:
    if not path.exists():
        raise FileNotFoundError(f"Candidate module not found: {path}")
    spec = importlib.util.spec_from_file_location(f"optimize_candidate_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import candidate module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_case"):
        raise ValueError(f"Candidate module must define run_case(...): {path}")
    name = getattr(module, "CANDIDATE_NAME", path.stem)
    return Candidate(name=str(name), module=module, path=path)


def _load_optimize_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Optimize config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def _select_cases(all_cases: list[ConvCase], *, top_k: int, case_ids: list[str]) -> list[ConvCase]:
    if case_ids:
        selected = [case for case in all_cases if case.case_id in set(case_ids)]
        selected.sort(key=lambda case: case.total_macs, reverse=True)
        return selected
    return sorted(all_cases, key=lambda case: case.total_macs, reverse=True)[:top_k]


def _benchmark_case(
    *,
    case: ConvCase,
    device: torch.device,
    backend: str,
    candidate: Candidate | None,
    warmup_iterations: int,
    timed_iterations: int,
    correctness_trials: int,
    atol: float,
    rtol: float,
    seed: int,
) -> dict[str, Any]:
    baseline_inputs = _make_case_tensors(case=case, seed=seed, device=device)
    baseline_setup_seconds = 0.0
    baseline_stats = _time_runner(
        runner=lambda: _run_baseline_case(case, *baseline_inputs),
        device=device,
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
    )

    correctness_result: dict[str, Any] | None = None
    candidate_payload: dict[str, Any] | None = None

    if candidate is not None:
        supported = bool(getattr(candidate.module, "is_supported", lambda *_args, **_kwargs: True)(asdict(case)))
        if supported:
            prepared_state, setup_seconds = _prepare_candidate(candidate, case, device)
            correctness_result = _run_correctness_trials(
                case=case,
                device=device,
                candidate=candidate,
                prepared_state=prepared_state,
                correctness_trials=correctness_trials,
                atol=atol,
                rtol=rtol,
                seed=seed,
            )
            candidate_inputs = _make_case_tensors(case=case, seed=seed, device=device)
            candidate_stats = _time_runner(
                runner=lambda: _run_candidate_case(candidate, case, prepared_state, *candidate_inputs),
                device=device,
                warmup_iterations=warmup_iterations,
                timed_iterations=timed_iterations,
            )
            candidate_payload = {
                "name": candidate.name,
                "path": str(candidate.path),
                "supported": True,
                "setup_seconds": round(setup_seconds, 6),
                "timed": True,
                **candidate_stats,
                "speedup_vs_baseline": round(baseline_stats["median_ms"] / candidate_stats["median_ms"], 6)
                if candidate_stats["median_ms"] > 0
                else None,
            }
        else:
            candidate_payload = {
                "name": candidate.name,
                "path": str(candidate.path),
                "supported": False,
                "timed": False,
            }

    return {
        "case": asdict(case),
        "backend": backend,
        "baseline": {
            "name": "torch_mps_reference" if device.type == "mps" else f"torch_{device.type}_reference",
            "setup_seconds": baseline_setup_seconds,
            **baseline_stats,
        },
        "correctness": correctness_result,
        "candidate": candidate_payload,
    }


def _prepare_candidate(candidate: Candidate, case: ConvCase, device: torch.device) -> tuple[Any, float]:
    prepare_fn = getattr(candidate.module, "prepare_case", None)
    if prepare_fn is None:
        return None, 0.0
    started = time.perf_counter()
    prepared = prepare_fn(asdict(case), device)
    return prepared, time.perf_counter() - started


def _run_correctness_trials(
    *,
    case: ConvCase,
    device: torch.device,
    candidate: Candidate,
    prepared_state: Any,
    correctness_trials: int,
    atol: float,
    rtol: float,
    seed: int,
) -> dict[str, Any]:
    max_abs_errors: list[float] = []
    mean_abs_errors: list[float] = []
    max_rel_errors: list[float] = []
    mean_rel_errors: list[float] = []
    shape_match = True
    allclose = True

    for trial in range(correctness_trials):
        inputs = _make_case_tensors(case=case, seed=seed + trial, device=device)
        baseline = _run_baseline_case(case, *inputs)
        candidate_out = _run_candidate_case(candidate, case, prepared_state, *inputs)
        if list(candidate_out.shape) != list(baseline.shape):
            shape_match = False
            allclose = False
            break
        diff = (candidate_out.float() - baseline.float()).abs()
        baseline_abs = baseline.float().abs().clamp_min(1e-8)
        rel = diff / baseline_abs
        max_abs_errors.append(float(diff.max().item()))
        mean_abs_errors.append(float(diff.mean().item()))
        max_rel_errors.append(float(rel.max().item()))
        mean_rel_errors.append(float(rel.mean().item()))
        if not torch.allclose(candidate_out.float(), baseline.float(), atol=atol, rtol=rtol):
            allclose = False

    return {
        "status": "passed" if shape_match and allclose else "failed",
        "shape_match": shape_match,
        "allclose": allclose,
        "atol": atol,
        "rtol": rtol,
        "trials": correctness_trials,
        "max_abs_error": _safe_max(max_abs_errors),
        "mean_abs_error": _safe_mean(mean_abs_errors),
        "max_rel_error": _safe_max(max_rel_errors),
        "mean_rel_error": _safe_mean(mean_rel_errors),
    }


def _time_runner(
    *,
    runner,
    device: torch.device,
    warmup_iterations: int,
    timed_iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup_iterations):
        runner()
    _synchronize(device)

    samples_ms: list[float] = []
    for _ in range(timed_iterations):
        _synchronize(device)
        started = time.perf_counter()
        runner()
        _synchronize(device)
        samples_ms.append((time.perf_counter() - started) * 1000.0)

    return {
        "warmup_iterations": warmup_iterations,
        "timed_iterations": timed_iterations,
        "median_ms": round(statistics.median(samples_ms), 6),
        "mean_ms": round(statistics.fmean(samples_ms), 6),
        "min_ms": round(min(samples_ms), 6),
        "max_ms": round(max(samples_ms), 6),
        "samples_ms": [round(value, 6) for value in samples_ms],
    }


def _run_baseline_case(
    case: ConvCase,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if case.op_type == "conv2d":
        return F.conv2d(
            x,
            weight,
            bias,
            stride=tuple(case.stride),
            padding=tuple(case.padding),
            dilation=tuple(case.dilation),
            groups=case.groups,
        )
    if case.op_type == "conv_transpose2d":
        return F.conv_transpose2d(
            x,
            weight,
            bias,
            stride=tuple(case.stride),
            padding=tuple(case.padding),
            output_padding=tuple(case.output_padding),
            dilation=tuple(case.dilation),
            groups=case.groups,
        )
    raise ValueError(f"Unsupported op type: {case.op_type}")


def _run_candidate_case(
    candidate: Candidate,
    case: ConvCase,
    prepared_state: Any,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return candidate.module.run_case(
        asdict(case),
        x,
        weight,
        bias,
        prepared_state=prepared_state,
    )


def _make_case_tensors(*, case: ConvCase, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(tuple(case.input_shape), generator=generator, dtype=torch.float32)
    weight = torch.randn(tuple(case.weight_shape), generator=generator, dtype=torch.float32)
    bias = None
    if case.has_bias:
        bias_channels = case.output_shape[1]
        bias = torch.randn((bias_channels,), generator=generator, dtype=torch.float32)
    return x.to(device), weight.to(device), bias.to(device) if bias is not None else None


def _iter_preprocessed_batches(
    *,
    input_path: Path,
    batch_size: int,
    render_factor: int,
    max_frames: int | None,
    max_batches: int | None,
    device: torch.device,
):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open clip for convolution shape extraction: {input_path}")

    render_size = render_factor * 16
    frames_processed = 0
    batches_processed = 0
    try:
        while True:
            if max_batches is not None and batches_processed >= max_batches:
                break
            batch_frames: list[np.ndarray] = []
            while len(batch_frames) < batch_size:
                if max_frames is not None and frames_processed >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (render_size, render_size), interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                batch_frames.append(rgb)
                frames_processed += 1
            if not batch_frames:
                break

            batch_np = np.stack(batch_frames).astype(np.float32) / 255.0
            batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2)
            batch_tensor = (batch_tensor - IMAGENET_MEAN) / IMAGENET_STD
            yield batch_tensor.to(device)
            batches_processed += 1
            if max_frames is not None and frames_processed >= max_frames:
                break
    finally:
        cap.release()


def _estimate_conv_macs(
    *,
    op_type: str,
    input_shape: list[int],
    output_shape: list[int],
    weight_shape: list[int],
    groups: int,
) -> int:
    batch = int(input_shape[0])
    out_channels = int(output_shape[1])
    out_h = int(output_shape[2])
    out_w = int(output_shape[3])
    if op_type == "conv2d":
        in_channels_per_group = int(weight_shape[1])
    else:
        in_channels_per_group = int(weight_shape[0] // groups)
    kernel_h = int(weight_shape[2])
    kernel_w = int(weight_shape[3])
    return int(batch * out_channels * out_h * out_w * in_channels_per_group * kernel_h * kernel_w)


def _classify_stage(module_name: str) -> str:
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
        layer_idx = int(parts[1])
        if layer_idx == 0:
            return "encoder"
        if layer_idx <= 3:
            return "middle"
        if layer_idx <= 8:
            return "decoder"
        return "head"
    return "unknown"


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.fmean(values), 8)


def _safe_max(values: list[float]) -> float | None:
    if not values:
        return None
    return round(max(values), 8)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
