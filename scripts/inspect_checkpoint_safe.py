from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a PyTorch checkpoint using the restricted weights_only path."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to the checkpoint file.")
    parser.add_argument(
        "--allow-slice",
        action="store_true",
        help="Allowlist Python's built-in slice type during restricted loading.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    import torch

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    print(f"torch_version={torch.__version__}")
    print(f"checkpoint={checkpoint}")

    try:
        unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(str(checkpoint))
        print("unsafe_globals=" + json.dumps(unsafe_globals))
        if unsafe_globals:
            print(
                "ERROR: restricted static inspection found non-allowlisted globals; not attempting load.",
                file=sys.stderr,
            )
            return 2
    except ValueError as exc:
        print(f"unsafe_globals_error={exc}")
        print("unsafe_globals=" + json.dumps([]))

    safe_types = [slice] if args.allow_slice else []
    with torch.serialization.safe_globals(safe_types):
        loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    print(f"loaded_type={type(loaded).__module__}.{type(loaded).__name__}")

    if isinstance(loaded, dict):
        keys = list(loaded.keys())
        print(f"top_level_key_count={len(keys)}")
        print("top_level_keys_sample=" + json.dumps([str(key) for key in keys[:20]]))

        tensor_entries = []
        for key, value in loaded.items():
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                tensor_entries.append(
                    {
                        "key": str(key),
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                )
            if len(tensor_entries) >= 10:
                break
        print("tensor_entries_sample=" + json.dumps(tensor_entries))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
