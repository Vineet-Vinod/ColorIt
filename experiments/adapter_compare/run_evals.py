from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "tmp/adapter_compare_90_120/unattended"


def stop(signum: int, _frame: object) -> None:
    children = psutil.Process().children(recursive=True)
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        child.kill()
    raise SystemExit(128 + signum)


def record(state: dict[str, object]) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = WORK / "status.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(WORK / "status.json")
    print(json.dumps(state), flush=True)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    ready = WORK / "kernel_ready.json"
    record({"status": "waiting_for_kernel_validation"})
    deadline = time.monotonic() + 45 * 60
    while not ready.exists():
        if time.monotonic() > deadline:
            record(
                {"status": "blocked", "reason": "Kernel validation exceeded 45 minutes"}
            )
            return 1
        time.sleep(5)
    settings = json.loads(ready.read_text())
    if settings.get("status") != "ready":
        record(
            {
                "status": "blocked",
                "reason": settings.get("reason", "Kernel validation failed"),
            }
        )
        return 1
    completed: list[dict[str, object]] = []
    for model in ("havc", "ltx"):
        for case in ("automatic", "assisted"):
            name = f"{model}_{case}"
            backend = settings.get(f"{model}_backend", "mlx")
            if backend not in ("mlx", "mps"):
                raise ValueError(f"Invalid backend: {backend}")
            record({"status": "running", "case": name, "completed": completed})
            started = time.monotonic()
            with (WORK / f"{name}.log").open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "experiments.adapter_compare.run_case",
                        model,
                        "--case",
                        case,
                        "--backend",
                        backend,
                        "--destination",
                        str(WORK / "cases"),
                    ],
                    cwd=ROOT,
                    check=False,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            completed.append(
                {
                    "case": name,
                    "exit_code": result.returncode,
                    "seconds": time.monotonic() - started,
                }
            )
            folder = WORK / "cases" / name
            for guard in folder.glob("*guard.json"):
                reason = json.loads(guard.read_text()).get("stop_reason")
                if reason:
                    record(
                        {
                            "status": "safety_stop",
                            "reason": reason,
                            "completed": completed,
                        }
                    )
                    return 1
    success = all(item["exit_code"] == 0 for item in completed)
    record(
        {
            "status": "completed" if success else "completed_with_errors",
            "completed": completed,
        }
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
