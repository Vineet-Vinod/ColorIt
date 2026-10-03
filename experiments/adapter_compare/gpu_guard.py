from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import psutil


def configure_gpu_limits() -> None:
    import mlx.core as mx
    import torch

    mx.set_memory_limit(80 * 1024**3)
    mx.set_cache_limit(1024**3)
    torch.mps.set_per_process_memory_fraction(0.35)
    torch.set_num_threads(8)


def supervise(command: list[str], record_path: Path) -> int:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path("tmp/adapter_compare_90_120/gpu.lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        baseline_swap = psutil.swap_memory().used
        if psutil.virtual_memory().available < 96 * 1024**3:
            raise RuntimeError(
                "Insufficient memory headroom to start the GPU experiment"
            )
        process = subprocess.Popen(command, start_new_session=True)
        started = time.monotonic()
        peak_rss = 0
        minimum_available = psutil.virtual_memory().available
        reason = None
        samples = []
        while process.poll() is None:
            try:
                worker = psutil.Process(process.pid)
                family = [worker, *worker.children(recursive=True)]
                rss = sum(
                    child.memory_info().rss for child in family if child.is_running()
                )
            except psutil.NoSuchProcess:
                continue
            available = psutil.virtual_memory().available
            swap_growth = psutil.swap_memory().used - baseline_swap
            peak_rss = max(peak_rss, rss)
            minimum_available = min(minimum_available, available)
            samples.append(
                {
                    "seconds": time.monotonic() - started,
                    "rss": rss,
                    "available": available,
                }
            )
            if rss > 160 * 1024**3:
                reason = "Worker family exceeded 160 GiB resident memory"
            elif available < 48 * 1024**3:
                reason = "System memory headroom fell below 48 GiB"
            elif swap_growth > 1024**3:
                reason = "System swap grew by more than 1 GiB"
            if reason is not None:
                print("Stopping experiment:", reason, flush=True)
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(2)
        exit_code = process.wait()
        record = {
            "command": command,
            "exit_code": exit_code,
            "stop_reason": reason,
            "seconds": time.monotonic() - started,
            "peak_family_rss": peak_rss,
            "minimum_system_available": minimum_available,
            "samples": samples,
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    command = options.command[1:] if options.command[:1] == ["--"] else options.command
    if not command:
        parser.error("A worker command is required")
    raise SystemExit(supervise(command, options.record))
