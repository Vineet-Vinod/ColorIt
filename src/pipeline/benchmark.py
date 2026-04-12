from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any

from src.pipeline.colorize_clip import run_colorize_clip_profiled
from src.pipeline.config import AppConfig
from src.pipeline.manifest import write_json_manifest
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths


@dataclass(frozen=True)
class ResourceSampleSummary:
    sample_count: int
    process_cpu_percent_avg: float | None
    process_cpu_percent_peak: float | None
    gpu_device_util_percent_avg: float | None
    gpu_device_util_percent_peak: float | None
    gpu_matched_sample_count: int


class ResourceSampler:
    def __init__(self, *, pid: int, interval_seconds: float) -> None:
        self.pid = pid
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.cpu_percent_samples: list[float] = []
        self.gpu_util_samples: list[float] = []

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> ResourceSampleSummary:
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 2.0))
        return ResourceSampleSummary(
            sample_count=max(len(self.cpu_percent_samples), len(self.gpu_util_samples)),
            process_cpu_percent_avg=_avg(self.cpu_percent_samples),
            process_cpu_percent_peak=_peak(self.cpu_percent_samples),
            gpu_device_util_percent_avg=_avg(self.gpu_util_samples),
            gpu_device_util_percent_peak=_peak(self.gpu_util_samples),
            gpu_matched_sample_count=len(self.gpu_util_samples),
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            cpu = _sample_process_cpu_percent(self.pid)
            if cpu is not None:
                self.cpu_percent_samples.append(cpu)

            gpu = _sample_gpu_device_util_percent(self.pid)
            if gpu is not None:
                self.gpu_util_samples.append(gpu)

            self._stop_event.wait(self.interval_seconds)


def run_benchmark_clips(
    *,
    config: AppConfig,
    config_path: Path,
    input_paths: list[Path],
    output_dir: Path | None,
    manifest_path: Path | None,
    overwrite: bool,
    sample_interval_seconds: float,
) -> int:
    paths = resolve_project_paths(config)
    ensure_runtime_directories(paths)

    if not input_paths:
        raise ValueError("At least one input clip is required for benchmarking.")

    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else (paths.colorized_dir / "benchmarks").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else (paths.manifest_dir / "benchmark_runs.json").resolve()
    )

    benchmark_payload: dict[str, Any]
    if manifest_path.exists():
        benchmark_payload = json.loads(manifest_path.read_text())
    else:
        benchmark_payload = {"runs": []}

    for input_path in input_paths:
        input_path = input_path.expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Benchmark input clip not found: {input_path}")

        output_path = output_dir / f"{input_path.stem}_benchmark.mp4"
        print(f"Benchmark input: {input_path}")
        sampler = ResourceSampler(pid=os.getpid(), interval_seconds=sample_interval_seconds)
        sampler.start()
        started = time.perf_counter()
        result = run_colorize_clip_profiled(
            config=config,
            config_path=config_path,
            input_path=input_path,
            output_path=output_path,
            manifest_path=paths.manifest_dir / "benchmark_clip_runs.json",
            overwrite=overwrite,
            collect_profile=True,
        )
        total_wall_seconds = time.perf_counter() - started
        resource_summary = sampler.stop()

        if result.stage_profile is None:
            raise RuntimeError("Expected stage profiling to be enabled during benchmark run.")

        entry = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "config_path": str(config_path.resolve()),
            "render_factor": int(config.model["render_factor"]),
            "backend": result.run_record.backend,
            "run_record": asdict(result.run_record),
            "stage_profile": asdict(result.stage_profile),
            "resource_summary": asdict(resource_summary),
            "wall_seconds_with_sampling": round(total_wall_seconds, 6),
        }
        benchmark_payload.setdefault("runs", []).append(entry)
        write_json_manifest(manifest_path, benchmark_payload)

        print(
            "Benchmark summary:"
            f" fps={result.stage_profile.effective_fps:.2f}"
            f" model={result.stage_profile.inference_model_seconds:.2f}s"
            f" decode={result.stage_profile.frame_decode_seconds:.2f}s"
            f" save={result.stage_profile.frame_save_seconds:.2f}s"
            f" encode={result.stage_profile.encode_seconds:.2f}s"
            f" gpu_avg={_fmt_optional(resource_summary.gpu_device_util_percent_avg)}"
            f" cpu_avg={_fmt_optional(resource_summary.process_cpu_percent_avg)}"
        )

    print(f"Benchmark manifest written to {manifest_path}")
    return 0


def _sample_process_cpu_percent(pid: int) -> float | None:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "%cpu="],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _sample_gpu_device_util_percent(pid: int) -> float | None:
    result = subprocess.run(
        ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    pid_match = re.search(r'"fLastSubmissionPID"=(\d+)', result.stdout)
    util_match = re.search(r'"Device Utilization %"=(\d+)', result.stdout)
    if pid_match is None or util_match is None:
        return None
    if int(pid_match.group(1)) != pid:
        return None
    return float(util_match.group(1))


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _peak(values: list[float]) -> float | None:
    if not values:
        return None
    return round(max(values), 3)


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"
