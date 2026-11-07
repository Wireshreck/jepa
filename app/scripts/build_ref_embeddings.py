"""Builds the demo's frozen artifacts with the repo's own evaluation pipeline (`evaluate.py`):

  runs/ref/flickr_dinov2       DINOv2 embeddings of the 2,014 val+test photos (benchmark + rotation side)
  runs/ref/flickr_T            our text JEPA's embeddings of the same pairs' captions
  runs/ref/coco5k_dinov2       DINOv2 embeddings of the 5,000 COCO test photos (the demo's photo gallery)

Each step is an `evaluate.py embed` invocation, exactly as the repo's quick start prescribes;
our side is redone whenever the checkpoint is newer. The rotation itself is fitted on the
val rows of the flickr embeds (-fit_demo_artifacts.py, the same protocol as `align`).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "repo"
STATE = ROOT / "app" / "state"
sys.path.insert(0, str(REPO))
NAME = "local109m"


def latest_checkpoint() -> Path | None:
    ckpts = sorted(STATE.glob("ckpts/runs/*/*/*/checkpoints/last.ckpt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def sh(*cmd) -> None:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(REPO), check=True)


def embeds_stamp(directory: str) -> float:
    return max((p.stat().st_mtime for p in (REPO / directory).rglob("*.pt")), default=0)


def embed(model: str, out: str, pairs: str, fresh_after: float = 0) -> None:
    """`evaluate.py embed <model> <out> --pairs <pairs>`, run only when its outputs are stale."""
    if embeds_stamp(out) >= max(fresh_after, 1):
        print(f"# {out} exists; skipping", flush=True)
        return
    sh(ROOT / ".venv" / "Scripts" / "python.exe", "evaluate.py", "embed", model, out, "--pairs", pairs)


def main() -> None:
    ckpt = latest_checkpoint()
    if ckpt is None:
        raise SystemExit("no trained checkpoint yet (`jepa train` first)")
    if not (REPO / "data/flickr30k/flickr30k-images").exists():
        raise SystemExit("photos not downloaded yet (`jepa pull` first)")
    print(f"# checkpoint: {ckpt}", flush=True)
    born = ckpt.stat().st_mtime
    embed("dinov2", "runs/ref/flickr_dinov2", "flickr")              # the benchmark's frozen image side
    embed(str(ckpt), "runs/ref/flickr_T", "flickr", fresh_after=born)  # our side on the same pairs
    embed("dinov2", "runs/ref/coco5k_dinov2", "coco5k")              # the demo's photo gallery
    print("# embeds done", flush=True)
    print("# embeds done", flush=True)


if __name__ == "__main__":
    main()
