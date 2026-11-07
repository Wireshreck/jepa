"""Shared runtime: loads the trained 109M text JEPA, the rotation and the photos' embeddings.

The model always comes from the repo's own code (``prh.models.TextEncoder.from_checkpoint``);
the rotation and photo gallery are the artifacts that ``app/scripts/*.py`` build with the
repo's own evaluation pipeline. Nothing here re-implements the paper's maths.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parent
REPO = ROOT / "repo"
STATE = APP / "state"

CONFIG_PATH = APP / "config.json"


def config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def latest_checkpoint() -> Path | None:
    run_dir = STATE / "ckpts"
    ckpts = sorted(run_dir.glob("runs/*/*/*/checkpoints/last.ckpt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def ckpt_info(path: Path) -> dict:
    """step/epoch of a Lightning checkpoint without loading its tensors (mmap reads scalars only)."""
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    steps = ck.get("global_step", 0) if isinstance(ck, dict) else 0
    return {
        "step": int(steps or 0),
        "epoch": int(ck.get("epoch") or 0) if isinstance(ck, dict) else 0,
        "modified": path.stat().st_mtime,
    }


class Runtime:
    """Everything the web page and `jepa chat` need, ready or clearly absent."""

    def __init__(self):
        self.cfg = config()
        self.ready = threading.Event()
        self.status = "loading"
        self.encoder = None
        self.pad_id = 0
        self.embed_dim = 0
        self.params = 0
        self.ckpt_path: Path | None = None
        self.ckpt_step = 0
        self.ckpt_mtime_seen = 0.0
        self._reload_thread: threading.Thread | None = None
        self.rotation = None
        self.gallery: dict | None = None
        self.caption_bank: dict | None = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._load, daemon=True)
        self.thread.start()

    def maybe_reload(self) -> None:
        """`jepa train` refreshes last.ckpt every 250 steps; hot-swap it into the running server."""
        if self._reload_thread is not None and self._reload_thread.is_alive():
            return
        if self.status not in ("model", "ready"):
            return
        try:
            ck = latest_checkpoint()
            mtime = ck.stat().st_mtime if ck else 0.0
        except OSError:
            return
        if not mtime or mtime <= self.ckpt_mtime_seen:
            return
        if time.time() - mtime < 30:  # still being written by the trainer
            return
        self.ckpt_mtime_seen = mtime
        self._reload_thread = threading.Thread(target=self._load, daemon=True)
        self._reload_thread.start()
        self.status = "reloading"

    def _load(self) -> None:  # runs in the background so the page is served immediately
        import sys

        sys.path.insert(0, str(REPO))
        import torch
        from prh.models import TextEncoder
        from transformers import AutoTokenizer

        try:
            ck = self._pick_checkpoint()
            if ck is None:
                self.status = "no-model"
                self._load_artifacts()
                return
            self.ckpt_path = ck
            self.ckpt_mtime_seen = ck.stat().st_mtime
            info = ckpt_info(ck)
            self.ckpt_step, self.ckpt_epoch = info["step"], info["epoch"]
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.enc = TextEncoder.from_checkpoint(ck, tok.pad_token_id)
            self.enc = self.enc.float()
            self.embed_dim = int(self.enc.embed_dim)
            self.pad_id = int(tok.pad_token_id)
            self.params = sum(p.numel() for p in self.enc.parameters())
            self.status = "model"
            self.device = str(next(self.enc.parameters()).device)
        except Exception as e:  # a corrupt mid-training checkpoint must not kill the page
            self.error = str(e)
            self.status = "error"
            self.ready.set()
            return
        self._load_artifacts()
        self.status = "ready"
        self.ready.set()

    def _load_artifacts(self) -> None:
        """Rotation and banks may exist before the first checkpoint; report them either way."""
        from prh.align import Rotation

        rot_path = STATE / "rotation.pt"
        if rot_path.exists():
            try:
                self.rotation = Rotation.load(rot_path)
            except Exception as e:
                self.error = f"rotation: {e}"
        for name, attr in (("gallery.pt", "gallery"), ("captions.pt", "caption_bank")):
            p = STATE / name
            if p.exists():
                try:
                    import torch

                    data = torch.load(p, map_location="cpu", weights_only=True)
                    data["z"] = data["z"].float()
                    setattr(self, attr, data)
                except Exception as e:
                    self.error = f"{name}: {e}"

    def _pick_checkpoint(self) -> Path | None:
        cfg = self.cfg.get("checkpoint")
        if cfg:  # pinned in config.json
            p = Path(cfg)
            return p if p.exists() else None
        return latest_checkpoint()

    # ---------------- queries ----------------

    def _q(self, texts: list[str]):
        import torch

        with self.lock:
            return self.enc.encode(texts, max_len=64).float().cpu()

    def similar_captions(self, query: str, k: int = 6, mode: str = "captions"):
        """The ``k`` nearest captions of the COCO-5k test set (text space, no pairs involved)."""
        if self.status not in ("model", "ready") or self.caption_bank is None:
            return []
        import torch

        q = self._q([query])
        bank = self.caption_bank
        z = torch.nn.functional.normalize(bank["z"], dim=1)
        sims = (z @ torch.nn.functional.normalize(q, dim=1).T).squeeze(1)
        # hide near-self matches when the query is a bank caption verbatim
        top = sims.argsort(descending=True)[:k]
        return [(bank["captions"][int(i)], float(sims[int(i)])) for i in top]

    def similar_images(self, query: str, k: int = 8):
        """The ``k`` nearest photos (rotated into the model's text space by the paper's one rotation)."""
        if self.gallery is None or self.rotation is None:
            return []
        import torch

        q = self._q([query])
        rot = self.rotation
        zq = rot(q.to(dtype=torch.float32))
        bank = self.gallery
        z = torch.nn.functional.normalize(bank["z"], dim=1)
        sims = (z @ torch.nn.functional.normalize(zq, dim=1).T).squeeze(1)
        top = sims.argsort(descending=True)[: int(k)]
        return [(str(bank["files"][int(i)]), float(sims[int(i)]) if float(sims[int(i)]) <= 1 else 1.0) for i in top]

    def embed(self, texts: list[str]):
        return self._q(texts)

    def close(self) -> None:
        pass
