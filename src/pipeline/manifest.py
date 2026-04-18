from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def load_json_manifest(path: Path, default: dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json_manifest(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
