from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import cv2
from PIL import Image

from src.experimental.assets import prepare_assets, require_apple_silicon
from src.experimental.experimental_dataclasses import ClipRequest, Reference
from src.experimental.media import deliver_clip, detect_shots, probe_clip


def run_experimental_clip(request: ClipRequest, root: Path) -> int:
    source = request.source.expanduser().resolve()
    info = probe_clip(source)
    if info.duration >= 60:
        raise ValueError(
            "Experimental colorization accepts clips shorter than 60 seconds only."
        )
    if request.resume:
        raise ValueError(
            "Experimental clip colorization does not yet support --resume."
        )
    output = request.output or source.with_name(source.stem + "_experimental_color.mp4")
    output = output.expanduser().resolve()
    if output == source:
        raise ValueError("Output must differ from the input clip.")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Experimental output must use the .mp4 extension.")
    if output.exists() and not request.overwrite:
        raise FileExistsError(
            f"Output exists: {output}. Use --overwrite to replace it."
        )
    require_apple_silicon()
    if importlib.util.find_spec("mflux") is None:
        raise RuntimeError(
            "Install experimental dependencies with `uv sync --extra experimental`."
        )
    assets = prepare_assets(root)
    from src.experimental.render import PROMPT, generate_references, propagate

    directory = root / "tmp/experimental"
    directory.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="clip-", dir=directory))
    print(f"Experimental FLUX + CMNET2. Artifacts: {run}", flush=True)
    normalized = run / "source.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            f"fps={info.fps}",
            "-an",
            "-c:v",
            "ffv1",
            str(normalized),
        ],
        check=True,
    )
    normalized_info = probe_clip(normalized, count_frames=True)
    shots = detect_shots(normalized, normalized_info)
    references = []
    capture = cv2.VideoCapture(str(normalized))
    try:
        for shot in shots:
            for frame in shot.references:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"Failed to extract reference frame {frame}")
                reference = Reference(
                    frame,
                    run / f"source_{frame:06d}.png",
                    run / f"colored_{frame:06d}.png",
                )
                Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(
                    reference.source
                )
                references.append(reference)
    finally:
        capture.release()
    generate_references(assets, references, info)
    raw = run / "colored.mkv"
    propagate(normalized, raw, assets, shots, references, normalized_info)
    delivery = run / "delivery.mp4"
    delivered_info = replace(normalized_info, fps=info.fps, audio=info.audio)
    deliver_clip(raw, source, delivery, delivered_info)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        delivery.open("rb") as incoming,
        output.open("wb" if request.overwrite else "xb") as outgoing,
    ):
        shutil.copyfileobj(incoming, outgoing)
    manifest = {
        "source": str(source),
        "output": str(output),
        "prompt": PROMPT,
        "frames": delivered_info.frames,
        "fps": str(info.fps),
        "references": len(references),
        "shots": len(shots),
        "output_bytes": output.stat().st_size,
        "cmnet2_credit": "Dan64, CMNET2",
        "status": "complete",
    }
    (run / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Experimental colorization complete: {output}", flush=True)
    return 0
