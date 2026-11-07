"""Measures this install's alignment numbers with the repo's own protocol and writes app/state/metrics.json.

    align  : Flickr val pairs fitted (means, rotation, whitening), Flickr test pairs scored -> mknn, R@1, R@5
    wit    : the PRH paper's WIT-1024 benchmark against DINOv2 (mutual 10-NN, per-layer max)
    sts    : STS-B test Spearman (text-only quality)
    unimodal : Flickr caption-to-caption R@1 (text-only quality)

The paper's reference numbers are quoted along the local ones so the page never conflates the two.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "repo"
STATE = ROOT / "app" / "state"
sys.path.insert(0, str(REPO))
PY = str(ROOT / ".venv" / "Scripts" / "python.exe") if sys.platform == "win32" else str(ROOT / ".venv" / "bin" / "python")
NAME = "local109m"

#: the headline numbers of the paper (its README and Tables), quoted for comparison
REFERENCE = {
    "wit1024_dinov2_b": {"ours109M_flagship": 0.184, "Qwen2.5-72B": 0.154, "Llama-3-70B": 0.151,
                         "CLIP_text": 0.124, "NTP_flagship": None, "note": "mutual 10-NN, per-layer max; the flagship 109M JEPA was trained on the texts of 940M web images (~900M samples, 16 GPUs)"},
    "coco5k_dinov2_b": {"ours109M_flagship": 0.161, "all-mpnet-base-v2": 0.156, "CLIP_text": 0.132,
                        "Llama-3-70B": 0.113, "Qwen2.5-72B": 0.108},
    "flickr_quickstart": {"good_view_JEPA_R@1": 0.25, "bad_view_JEPA_R@1": 0.13, "next_token_R@1": 0.12,
                          "chance": 0.001, "note": "the repo's quick-start expectation (200 epochs on a GPU)"},
}
PAPER = "arXiv:2610.39302 (Balestriero & LeCun, 2026)"


def latest_checkpoint() -> Path | None:
    ckpts = sorted(STATE.glob("ckpts/runs/*/*/*/checkpoints/last.ckpt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def run_eval(*args: str) -> str:
    cmd = [PY, str(REPO / "evaluate.py"), *args]
    print("+ evaluate.py " + " ".join(args), flush=True)
    p = subprocess.run(cmd, cwd=str(REPO), text=True, capture_output=True)
    out = (p.stdout or "") + (p.stderr or "")
    print(out.strip()[-4000:], flush=True)
    return out


def read_alignment_csv() -> dict | None:
    import pandas as pd

    csv = STATE / "align_flickr.csv"
    rows = run_eval("align", "runs/ref/flickr_dinov2", "runs/ref/flickr_T", "--last", "--csv", str(csv))
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    row = df.iloc[-1].to_dict()
    return {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}


def wit_scores(ckpt: Path) -> dict:
    run_eval("wit-embed", "dinov2")                            # one-time cached per-layer features
    run_eval("wit-embed", str(ckpt), "--name", NAME)           # our per-layer features
    out = run_eval("wit", f"dinov2:{NAME}")
    scores = {}
    for line in out.splitlines():
        m = re.match(r"PRH WIT-1024 (mutual-kNN|unbiased-CKA) (\S+) x (\S+): ([0-9.]+)", line)
        if m:
            scores[f"{m.group(3)} x {m.group(2)} ({m.group(1)})"] = float(m.group(4))
    return scores


def sts_score(ckpt: Path) -> float | None:
    out = run_eval("sts", str(ckpt))
    m = re.search(r"STS-B Spearman ([0-9.]+)", out)
    return float(m.group(1)) if m else None


def caption_r1(ckpt: Path) -> float | None:
    out = run_eval("unimodal", str(ckpt))
    m = re.search(r"Flickr caption-to-caption R@1 ([0-9.]+)", out)
    return float(m.group(1)) if m else None


def main() -> None:
    ckpt = latest_checkpoint()
    if ckpt is None:
        raise SystemExit("no checkpoint yet")
    out = json.loads((STATE / "metrics.json").read_text(encoding="utf-8")) if (STATE / "metrics.json").exists() else {}
    out.update({
        "updated_at": time.time(),
        "checkpoint": str(ckpt.relative_to(ROOT)),
        "paper": PAPER,
        "reference": REFERENCE,
    })
    align = read_alignment_csv()
    if align:
        out["flickr_alignment"] = align
    try:
        out["wit1024"] = wit_scores(ckpt)
    except Exception as e:
        out["wit1024_error"] = str(e)
    sts = sts_score(ckpt)
    if sts is not None:
        out["sts_b_spearman"] = sts
    r1 = caption_r1(ckpt)
    if r1 is not None:
        out["flickr_caption_r1"] = r1
    out["local_compute_note"] = ("This CPU-only install trains the paper's exact 109M architecture "
                                 "(`--arch base`, good views) on a small caption corpus; expect partial convergence "
                                 "against the paper's 900M-sample, 16-GPU flagship run.")
    (STATE / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("# metrics.json written", flush=True)


if __name__ == "__main__":
    main()
