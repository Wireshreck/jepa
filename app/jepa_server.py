"""The jepa local server (FastAPI): the web page, the model queries, and training control.

Run by `jepa` / `jepa serve` (app/jepa.py). It serves app/static/index.html at / and:
    GET  /api/health    is the model loaded (server starts before the weights are, like ollama)
    GET  /api/model     the model card: architecture, params, paper numbers, local config
    POST /api/embed     {texts, rotate?} -> embeddings
    POST /api/search    {text, k, mode: images|captions} -> nearest entries
    GET  /api/metrics   app/state/metrics.json (measured vs the paper's table)
    GET  /api/photos/<i> the gallery photo files
    GET  /api/training  live progress of the 109M model's local training
    POST /api/train     start or resume that training in the background
    POST /api/stop-train
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import jepa as cli
from jepa_runtime import APP, REPO, ROOT, STATE, Runtime, config, latest_checkpoint

app = FastAPI(title="jepa", version="1.0.0", docs_url=None, redoc_url=None)
RT = Runtime()
STARTED = time.time()
app.mount("/static", StaticFiles(directory=str(APP / "static")), name="static")


@app.middleware("http")
async def static_revalidate(request, call_next):
    resp = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static"):
        resp.headers["Cache-Control"] = "no-cache"  # edits to the page land on the next reload
    return resp
_coco5k = ROOT / "repo" / "data" / "coco5k" / "images_mscoco_2014_5k_test"
if _coco5k.exists():
    app.mount("/photos", StaticFiles(directory=str(_coco5k)), name="photos")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(APP / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    RT.ready.wait(0.1)
    RT.maybe_reload()  # the trainer commits last.ckpt every 250 steps: serve whatever is newest
    st = state_summary()
    return {"ok": True, "uptime": round(time.time() - STARTED, 1), **st}


def _rel(p: Path | None) -> str | None:
    if p is None:
        return None
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def state_summary() -> dict:
    from jepa_runtime import ckpt_info

    ck = RT.ckpt_path or latest_checkpoint()
    return {
        "runtime_status": RT.status,
        "checkpoint": _rel(ck),
        "step": RT.ckpt_step or (ckpt_info(ck)["step"] if ck else 0),
        "params": RT.params or None,
        "embed_dim": RT.embed_dim or None,
        "rotation_ready": RT.rotation is not None,
        "gallery_images": len(RT.gallery["files"]) if RT.gallery else 0,
        "caption_bank": len(RT.caption_bank["captions"]) if RT.caption_bank else 0,
        "config": config(),
        "device": "cpu",
    }


@app.get("/api/model")
def model() -> dict:
    cfg = config()
    return {
        "paper": {"title": "Proving and Improving the Platonic Representation Hypothesis with JEPAs",
                  "authors": "Randall Balestriero, Yann LeCun", "arxiv": "2610.39302"},
        "architecture": {"backbone": "BERT (bidirectional pad_masked)", "arch": cfg["arch"], "width": 768,
                         "layers": 12, "heads": 12, "vocab": 30522, "params": cfg["params"] or "109,183,234", "pooling": "[CLS]"},
        "objective": {"name": "LeJEPA (good views)", "views": f"{cfg['views'][0]} global + {cfg['views'][1]} local",
                      "sigreg_lamb": cfg["lamb"],
                      "good_views": "two captions of the same image = two observations of one scene"},
        "headline": "109M text JEPA aligns with DINOv2/CLIP image encoders better than every LLM we evaluated, up to 72B",
        "local": {**state_summary(), "runtime_status": RT.status},
        "note": ("official checkpoints are restricted; this install trained the exact architecture locally on this machine "
                 "with the paper's own pipeline (good views + SIGReg) - see the Training tab."),
    }


@app.post("/api/embed")
def embed(body: dict) -> dict:
    _require_model()
    texts = list(body.get("texts") or [])
    if not texts or len(texts) > 64:
        raise HTTPException(400, "texts: a list of 1..64 strings")
    z = RT.embed(texts)
    out = [[round(x, 5) for x in row] for row in z.tolist()]
    if body.get("rotate") and RT.rotation is not None:
        import torch

        rz = RT.rotation(z)
        out = [[round(x, 5) for x in row] for row in rz.tolist()]
    return {"dim": z.shape[1], "step": RT.ckpt_step, "embeddings": out}


def _require_model() -> None:
    RT.ready.wait(30)
    if RT.status not in ("model", "ready"):
        raise HTTPException(503, {"error": "the model is not loaded yet", "status": RT.status,
                                  "hint": "start training: POST /api/train or `jepa train`"})


@app.post("/api/search")
def search(body: dict) -> dict:
    _require_model()
    text = (body.get("text") or "").strip()
    k = int(body.get("k") or 8)
    mode = body.get("mode") or "captions"
    if not text:
        raise HTTPException(400, "text required")
    if mode == "images":
        hits = RT.similar_images(text, k)
        if not RT.rotation:
            raise HTTPException(503, {"error": "no rotation yet",
                                      "hint": "the rotation is fitted after the first checkpoint: `jepa evaluate`"})
        return {"mode": "images", "query": text, "results":
                [{"url": f"/photos/{Path(p).name}", "src": str(p), "score": round(s, 4)} for p, s in hits]}
    hits = RT.similar_captions(text, k)
    if not RT.caption_bank:
        raise HTTPException(503, {"error": "the caption bank is not built yet",
                                  "hint": "`jepa evaluate` builds it from the COCO test captions after the first checkpoint"})
    return {"mode": "captions", "query": text, "results": [{"text": t, "score": round(s, 4)} for t, s in hits]}


@app.get("/api/metrics")
def metrics() -> JSONResponse:
    p = STATE / "metrics.json"
    if not p.exists():
        raise HTTPException(404, {"error": "no metrics yet", "hint": "`jepa evaluate` will produce app/state/metrics.json"})
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/training")
def training() -> dict:
    from jepa_runtime import ckpt_info

    pid = cli.train_live()
    ck = RT.ckpt_path or latest_checkpoint()
    step = epoch = 0
    if ck:
        info = ckpt_info(ck)
        step, epoch = info["step"], info["epoch"]
    log = newest(cli.LOGS.glob("train_*.log"))
    tail = last_log_lines(log, 6) if log else []
    return {"running": bool(pid), "pid": pid, "step": step, "epoch": epoch,
            "checkpoint": _rel(ck),
            "target_epoch": config()["epochs"], "target_steps": config().get("target_steps"),
            "log_tail": tail, "log": str(log) if log else None,
            "note": "CPU-only training local render. Paper's flagship run: 16 GPUs x 2 nodes."}


def newest(paths: "list[Path]") -> Path | None:
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def last_log_lines(log: Path, n: int = 6) -> "list[str]":
    try:
        lines = [l.strip() for l in log.read_text(encoding="utf-8", errors="ignore").splitlines()[-2000:] if l.strip()]
        keep = [l for l in lines if "Epoch" in l or "loss" in l or l.startswith("Epoch")][-n:]
        return keep if keep else lines[-n:]
    except OSError:
        return []


@app.post("/api/train")
def train(body: dict | None = None) -> dict:
    body = body or {}
    pid = cli.train_running()
    if pid:
        return {"running": True, "pid": pid, "note": "training already running on this machine"}
    cli.start_train(body.get("extra") or [])
    return {"started": True, "port": config().get("port", 7860)}


@app.post("/api/stop-train")
def stop_train() -> dict:
    cli.stop_train()
    return {"stopped": True}


@app.get("/api/photos/{name}")
def photo(name: str) -> FileResponse:
    p = _coco5k / Path(name).name
    if not p.exists():
        raise HTTPException(404, "photo not here")
    return FileResponse(p)


if __name__ == "__main__":
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port or config().get("port", 7860), access_log=False)
