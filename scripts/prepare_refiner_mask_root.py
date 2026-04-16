from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.finetune_deoldify import load_manifest
from src.pipeline.actor_masks import derive_costume_mask, make_kernel, write_debug_overlay, write_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an actor-aware refiner mask root from conservative actor masks.")
    parser.add_argument("--manifest", required=True, help="Path to the fine-tune manifest.jsonl.")
    parser.add_argument("--base-mask-root", required=True, help="Existing person/skin/costume mask root.")
    parser.add_argument("--actor-mask-root", required=True, help="Actor mask root containing person/<split>/ masks.")
    parser.add_argument("--output-root", required=True, help="Output root for prepared person/skin/costume masks.")
    parser.add_argument("--debug-count", type=int, default=24, help="Number of debug overlays to save.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    base_mask_root = Path(args.base_mask_root).expanduser().resolve()
    actor_mask_root = Path(args.actor_mask_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    samples = load_manifest(manifest_path)
    summary: dict[str, object] = {
        "samples": 0,
        "with_person": 0,
        "with_skin": 0,
        "with_costume": 0,
        "used_actor_person": 0,
        "used_costume_hint": 0,
        "used_costume_fallback": 0,
        "frames": [],
    }
    debug_saved = 0

    for sample in samples:
        image_path = Path(sample.image_path)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        old_person = read_mask(base_mask_root / "person" / sample.split / image_path.name)
        old_skin = read_mask(base_mask_root / "skin" / sample.split / image_path.name)
        old_costume = read_mask(base_mask_root / "costume" / sample.split / image_path.name)
        actor_person = read_mask(actor_mask_root / "person" / sample.split / image_path.name)

        person_mask, used_actor_person = choose_person_mask(actor_person=actor_person, old_person=old_person)
        skin_mask = derive_actor_aligned_skin_mask(person_mask=person_mask, old_skin=old_skin)
        costume_mask, used_costume_hint = derive_actor_aligned_costume_mask(
            person_mask=person_mask,
            skin_mask=skin_mask,
            old_costume=old_costume,
        )

        write_mask(output_root / "person" / sample.split / image_path.name, person_mask)
        write_mask(output_root / "skin" / sample.split / image_path.name, skin_mask)
        write_mask(output_root / "costume" / sample.split / image_path.name, costume_mask)
        if debug_saved < args.debug_count:
            write_debug_overlay(
                output_root / "debug" / sample.split / f"{image_path.stem}_overlay.png",
                image,
                person_mask,
                skin_mask,
                costume_mask,
            )
            debug_saved += 1

        summary["samples"] += 1
        summary["with_person"] += int(person_mask.any())
        summary["with_skin"] += int(skin_mask.any())
        summary["with_costume"] += int(costume_mask.any())
        summary["used_actor_person"] += int(used_actor_person)
        summary["used_costume_hint"] += int(used_costume_hint)
        summary["used_costume_fallback"] += int(person_mask.any() and not used_costume_hint)
        summary["frames"].append(
            {
                "image_path": str(image_path),
                "split": sample.split,
                "person_fraction": float(person_mask.mean()),
                "skin_fraction": float(skin_mask.mean()),
                "costume_fraction": float(costume_mask.mean()),
                "used_actor_person": bool(used_actor_person),
                "used_costume_hint": bool(used_costume_hint),
            }
        )

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"output_root={output_root}")
    print(f"summary={summary_path}")
    return 0


def read_mask(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((1, 1), dtype=np.uint8)
    return (np.asarray(Image.open(path).convert("L")) >= 128).astype(np.uint8)


def choose_person_mask(*, actor_person: np.ndarray, old_person: np.ndarray) -> tuple[np.ndarray, bool]:
    if actor_person.shape != old_person.shape:
        if actor_person.size == 1:
            return old_person.astype(np.uint8), False
        if old_person.size == 1:
            return actor_person.astype(np.uint8), True
        raise ValueError("Actor and old person masks have mismatched shapes.")
    if actor_person.any():
        return actor_person.astype(np.uint8), True
    return old_person.astype(np.uint8), False


def derive_actor_aligned_skin_mask(*, person_mask: np.ndarray, old_skin: np.ndarray) -> np.ndarray:
    if old_skin.shape != person_mask.shape:
        old_skin = np.zeros_like(person_mask, dtype=np.uint8)
    if not person_mask.any():
        return np.zeros_like(person_mask, dtype=np.uint8)
    kernel = make_kernel(person_mask.shape[0], person_mask.shape[1], scale=0.008)
    expanded_person = cv2.dilate(person_mask, kernel, iterations=1)
    skin_mask = (old_skin.astype(np.uint8) * expanded_person).astype(np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = (skin_mask * person_mask).astype(np.uint8)
    return skin_mask


def derive_actor_aligned_costume_mask(
    *,
    person_mask: np.ndarray,
    skin_mask: np.ndarray,
    old_costume: np.ndarray,
) -> tuple[np.ndarray, bool]:
    if old_costume.shape != person_mask.shape:
        old_costume = np.zeros_like(person_mask, dtype=np.uint8)
    if not person_mask.any():
        return np.zeros_like(person_mask, dtype=np.uint8), False

    kernel = make_kernel(person_mask.shape[0], person_mask.shape[1], scale=0.01)
    costume_hint = (old_costume.astype(np.uint8) * person_mask).astype(np.uint8)
    costume_hint = cv2.morphologyEx(costume_hint, cv2.MORPH_OPEN, kernel)
    costume_hint = cv2.morphologyEx(costume_hint, cv2.MORPH_CLOSE, kernel)
    fallback = derive_costume_mask(person_mask, skin_mask)

    min_area = max(64, int(round(person_mask.size * 0.0015)))
    if int(costume_hint.sum()) >= min_area:
        costume_mask = costume_hint
        used_costume_hint = True
    else:
        costume_mask = fallback
        used_costume_hint = False

    costume_mask = (costume_mask * person_mask).astype(np.uint8)
    costume_mask[skin_mask > 0] = 0
    costume_mask = cv2.morphologyEx(costume_mask, cv2.MORPH_OPEN, kernel)
    costume_mask = cv2.morphologyEx(costume_mask, cv2.MORPH_CLOSE, kernel)
    return costume_mask.astype(np.uint8), used_costume_hint


if __name__ == "__main__":
    raise SystemExit(main())
