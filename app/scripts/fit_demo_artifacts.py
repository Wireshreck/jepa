"""Fits the one rotation (the paper's showcase) and packs the demo's artifacts:

  app/state/rotation.pt   the map image-space -> text-JEPA-space (Procrustes on the Flickr val pairs)
  app/state/gallery.pt    the 5,000 COCO test photos through that rotation (search photos by description)
  app/state/captions.pt   the 25,000 COCO test captions embedded in the text JEPA's own space

The Flickr-side fit uses the val rows of runs/ref/flickr_{dinov2,T} — the same
fit-on-validation protocol that `evaluate.py align` uses for the benchmark, never the test pairs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "repo"
STATE = ROOT / "app" / "state"
STATE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))

from prh import data, features       # noqa: E402  (the repo's own code)
from prh.align import Rotation      # noqa: E402


def latest_checkpoint() -> Path | None:
    ckpts = sorted(STATE.glob("ckpts/runs/*/*/*/checkpoints/last.ckpt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def fit_rotation() -> Rotation:
    """Procrustes from the frozen DINOv2 space to our text-JEPA space, on the validation pairs
    (the repo's own alignment protocol: means/rotation/whitening fit on valid, scored on test)."""
    xi, pi, ti = features.last_embeddings(REPO / "runs/ref/flickr_dinov2", "embedding")
    xt, pt, tt = features.last_embeddings(REPO / "runs/ref/flickr_T", "embedding")
    assert torch.equal(pi, pt), "the two embed outputs must run through the same pair order"
    rows = ti == False  # noqa: E712  the val rows (`--last` split of `evaluate.py align`); test rows untouched
    a, b = xi[rows].double(), xt[rows].double()
    print(f"# fitting the rotation on {int(rows.sum().item())} Flickr val pairs (test rows untouched)", flush=True)
    return Rotation.fit(a, b)


def build_gallery(rot: Rotation) -> None:
    """The 5,000 COCO test photos, rotated into the text JEPA's space, with their files and first captions."""
    pairs = data.coco5k_pairs().sort_values("pair_id").reset_index(drop=True)
    zi, pi, _ = features.last_embeddings(REPO / "runs/ref/coco5k_dinov2", "embedding")
    order = torch.argsort(pi)                       # the embed output may be unordered: match the pairs table
    zi, pi = zi[order], pi[order]
    assert len(pi) == len(pairs) and torch.equal(pi, torch.as_tensor(pairs.pair_id.values)), \
        "the ref embeddings and the pairs table must run through the same pair order"
    zg = rot(zi)
    files = [str(p) for p in pairs.path]
    caps = [caps_list[0] for caps_list in pairs.captions]
    torch.save({"z": zg.half().cpu(), "files": files, "captions": caps}, STATE / "gallery.pt")
    print(f"# gallery: {len(zg)} photos, dim {zg.shape[1]}", flush=True)


def build_caption_bank() -> None:
    """All five captions of each of the 5,000 photos, in the text JEPA's own space (for `similar captions`)."""
    ckpt = latest_checkpoint()
    pairs = data.coco5k_pairs()
    captions = [c for caps_list in pairs.captions for c in caps_list]
    print(f"# embedding {len(captions)} captions with our text JEPA (CPU, about 1-2 h)", flush=True)
    z = features.embed_texts(str(ckpt), captions, batch_size=128)
    torch.save({"z": z.half().cpu(), "captions": captions}, STATE / "captions.pt")
    print(f"# captions bank ready: {len(captions)} texts, dim {z.shape[1]}", flush=True)


def main() -> None:
    if latest_checkpoint() is None:
        raise SystemExit("no checkpoint yet")
    rot = fit_rotation()
    rot.save(STATE / "rotation.pt")
    print("# rotation saved: app/state/rotation.pt", flush=True)
    build_gallery(rot)
    build_caption_bank()
    print("# demo artifacts done", flush=True)


if __name__ == "__main__":
    main()
