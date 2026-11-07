"""jepa — a one-command interface to the 109M-parameter text JEPA of Balestriero & LeCun (2026).

    jepa            start the server in the background and open the web interface (like ollama)
    jepa serve      run the server in the foreground (Ctrl+C to stop)
    jepa stop       stop the background server
    jepa status     what is installed, what is ready, what is training
    jepa train      start (or resume) local training of the 109M text JEPA, in the background
    jepa pull       download every dataset this install needs (captions, photos, benchmark)
    jepa evaluate   measure alignment and write app/state/metrics.json for the Benchmarks tab
    jepa chat       terminal REPL: embed a sentence, list its nearest captions / photos
    jepa version    print versions of the stack

This file is the only entry the user needs: app/jepa_server.py serves the page, app/scripts/*.py
are the pipelines behind `pull` and `evaluate`. Everything runs with this folder's venv.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import colorama

    colorama.just_fix_windows_console()
    GREEN, YELLOW, RED, BOLD, DIM = "\033[92m", "\033[93m", "\033[91m", "\033[1m", "\033[2m"
except ImportError:  # pragma: no cover
    GREEN = YELLOW = RED = BOLD = DIM = ""

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "repo"
APP = ROOT / "app"
STATE = APP / "state"
LOGS = APP / "logs"
STATE.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe" if os.name == "nt" else ROOT / ".venv" / "bin" / "python"
SERVER = APP / "jepa_server.py"
STATE_FILE = STATE / "server.json"
DEVICES = {}
DEFAULT_PORT = 7860


# ---------------- helpers ----------------

def config() -> dict:
    return json.loads((APP / "config.json").read_text(encoding="utf-8"))


def read_server_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_server_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps({**read_server_state(), **state}, indent=2), encoding="utf-8")


def clear_server_state() -> None:
    STATE_FILE.unlink(missing_ok=True)


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"], capture_output=True, text=True).stdout
        return str(int(pid)) in out
    except Exception:
        return False


def http_ok(port: int, timeout: float = 1.5) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def open_browser(port: int) -> None:
    url = f"http://localhost:{port}"
    print(f"\n  {BOLD}{url}\033[0m\n")
    threading.Thread(target=lambda: (time.sleep(0.5), _open(url)), daemon=True).start()


def _open(url: str) -> None:
    try:
        if os.name == "nt":
            os.startfile(url)  # noqa: S606
        else:
            import webbrowser

            webbrowser.open(url)
    except Exception as e:  # headless hosts
        print(f"(could not open a browser: {e}; visit {url} yourself)")


def detached(args: list[str], log_path: Path) -> int:
    flags = 0
    kw = {}
    if os.name == "nt":
        flags = 0x00000008 | 0x00000200  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}  # the framework logs unicode
    p = subprocess.Popen(args, stdout=open(log_path, "ab"), stderr=subprocess.STDOUT,
                         creationflags=flags or 0, cwd=str(ROOT), env=env, **kw)
    return p.pid


def train_live() -> int | None:
    live = [int(p.read_text().strip()) for p in sorted(LOGS.glob("train_*.pid")) if alive(int(p.read_text().strip() or 0))]
    return live[0] if live else None


def latest_checkpoint() -> Path | None:
    return max((STATE).glob("ckpts/runs/*/*/*/checkpoints/last.ckpt"), key=lambda p: p.stat().st_mtime, default=None)


def datasets_ready() -> tuple[bool, list[str]]:
    cfg = config()
    missing = [name for name, rel in cfg["data_dirs"].items() if not (REPO / rel).exists()]
    return not missing, missing


# ---------------- server ----------------

def start_server(foreground: bool, port: int, do_open: bool) -> None:
    st = read_server_state()
    if alive(st.get("pid")) and http_ok(port):
        if do_open:
            open_browser(port)
        return
    if foreground:
        print(f"  jepa serving on port {port} — Ctrl+C to stop")
        if do_open:
            open_browser(port)  # opens just before uvicorn binds; the page waits while the model loads
        import uvicorn

        sys.path.insert(0, str(APP))
        from jepa_server import app

        uvicorn.run(app, host="127.0.0.1", port=port, access_log=False)
        clear_server_state()
        return
    pid = detached([str(VENV_PY), "-u", str(SERVER), "--port", str(port)], LOGS / "server.log")
    write_server_state({"pid": pid, "port": port, "started_at": time.time()})
    print(f"reading the model and the saved artifacts (background, pid {pid})…", flush=True)
    deadline = time.time() + 120
    while time.time() < deadline:
        time.sleep(0.5)
        if http_ok(port):
            break
    else:
        print(f"server did not answer on {port} yet; log: {LOGS / 'server.log'}", file=sys.stderr)
    if do_open:
        open_browser(port)


def stop_server() -> None:
    st = read_server_state()
    pid = st.get("pid")
    if alive(pid):
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        clear_server_state()
        print(f"  stopped the server (pid {pid})")
    else:
        print("  server is not running")


# ---------------- training ----------------

def train_cmd(extra: list[str]) -> list[str]:
    cfg = config()
    entry = APP / "win_train.py" if os.name == "nt" else REPO / "train.py"  # thin adapter; train.py unchanged
    return [
        str(VENV_PY), "-u", str(entry), "--modality", "text",
        "--caption-views",                    # good views: the two captions of one image
        "--arch", cfg["arch"], "--lr", str(cfg["lr"]), "--clip", "1.0",
        "--data", cfg["train_data"], "--views", *[str(v) for v in cfg["views"]],
        "--batch-size", str(cfg["batch_size"]), "--lamb", str(cfg["lamb"]),
        "--epochs", str(cfg["epochs"]), "--epoch-size", str(cfg["epoch_size"]),
        "--val-every", str(cfg["val_every"]), "--max-len", str(cfg["max_len"]),
        "--workers", str(cfg["workers"]),
        "--cache-dir", str(STATE / "ckpts"), "--seed", str(cfg["seed"]),
        *extra,
    ]


def start_train(extra: list[str]) -> None:
    if live := train_live():
        print(f"  training is already running (pid {live})")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOGS / f"train_{stamp}.log"
    ckpt = latest_checkpoint()
    forwardable = [a for a in extra if a != "--fresh"]  # `--fresh` is a jepa-level switch, not train.py's
    resume = ckpt if not ({"--fresh", "--steps", "--resume"} & set(extra)) else None
    if resume:
        forwardable = ["--resume", str(resume), *forwardable]
    pid = detached(train_cmd(forwardable), log_path)
    (LOGS / f"train_{stamp}.pid").write_text(str(pid), encoding="utf-8")
    print(f"  training started (pid {pid}); log: app/logs/train_{stamp}.log")
    if resume:
        print(f"  resuming from {resume.relative_to(STATE / 'ckpts')}")


def stop_train() -> None:
    if pid := train_live():
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        print(f"  stopped training (pid {pid}); the last checkpoint is safe — `jepa train` resumes it")
    else:
        print("  training is not running")


# ---------------- status ----------------

def status() -> None:
    cfg = config()
    pid = read_server_state().get("pid")
    port = read_server_state().get("port", DEFAULT_PORT)
    ok_ds, missing = datasets_ready()
    ckpt = latest_checkpoint()
    up = alive(pid)
    rows = [
        ("model", f"BERT-{cfg['arch']} * {cfg['params']:,} params * {cfg['width']}d/{cfg['layers']}L/{cfg['heads']}H"),
        ("paper", "Balestriero & LeCun 2026 (arXiv:2610.39302): their 109M text JEPA beats Qwen2.5-72B / Llama-3-70B on cross-modal alignment"),
        ("server", f"http://localhost:{port} (pid {pid}, up)" if up and http_ok(port) else "stopped — run `jepa`"),
        ("training", f"running (pid {p})" if (p := train_live()) else "stopped — `jepa train` starts/resumes it"),
        ("checkpoint", str(ckpt.relative_to(STATE / "ckpts")) if ckpt else "none yet — train first"),
        ("datasets", "ready" if ok_ds else "missing: " + ", ".join(missing) + " — run `jepa pull`"),
        ("rotation", "fitted (app/state/rotation.pt)" if (STATE / "rotation.pt").exists() else "not fitted yet — `jepa evaluate`"),
        ("gallery", f"{int(STATE.joinpath('gallery.pt').stat().st_size / 2**20)} MiB on disk" if (STATE / "gallery.pt").exists() else "not built yet — `jepa evaluate`"),
        ("metrics", f"{int(STATE.joinpath('metrics.json').stat().st_size / 1024)} KiB" if (STATE / "metrics.json").exists() else "not computed yet — `jepa evaluate`"),
    ]
    w = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {BOLD}{k:<{w}}\033[0m  {v}")


# ---------------- chat ----------------

def chat() -> None:
    sys.path.insert(0, str(APP))
    from jepa_runtime import Runtime

    rt = Runtime()
    if not rt.ready.wait(180) or rt.status not in ("model", "ready"):
        print(f"  the model is not ready ({rt.status}); run `jepa train` (no model yet) or check `jepa status`")
        return
    print(f"  jepa chat — the {rt.params:,}-parameter text JEPA, step {rt.ckpt_step:,}.")
    print("  type text to see its nearest captions (demo corpus: 25k COCO captions) or `img <text>` for photos. empty line quits.\n")
    mode = "captions"
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        if line.startswith("img "):
            line, mode = line[4:], "images"
        if line == "txt":
            mode = "captions"
        if not line:
            continue
        try:
            hits = rt.similar_captions(line, 6) if mode == "captions" else rt.similar_images(line, 6)
            if not hits:
                what = "caption bank" if mode == "captions" else "photo gallery (needs a fitted rotation)"
                print(f"  (the {what} is not built yet; shapes up after the first `jepa evaluate`)")
            for text, score in hits:
                shown = Path(str(text)).name if mode == "images" else text
                print(f"  {score:.3f}  {shown}")
        except Exception as e:
            print(f"  ! {e}")
    print("bye")


# ---------------- evaluate / pull ----------------

def evaluate() -> None:
    for script in ("build_ref_embeddings.py", "fit_demo_artifacts.py", "metrics_snapshot.py"):
        print(f"\n== app/scripts/{script} ==")
        r = subprocess.run([str(VENV_PY), str(APP / "scripts" / script)], cwd=str(ROOT))
        if r.returncode:
            sys.exit(f"  {script} failed — see its output above")


def pull() -> None:
    r = subprocess.run([str(VENV_PY), str(APP / "scripts" / "prepare_data.py")], cwd=str(ROOT))
    if r.returncode:
        sys.exit(r.returncode)


# ---------------- entry ----------------

def main() -> None:
    p = argparse.ArgumentParser(prog="jepa", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="run the server in the foreground, with the web page opened")
    s.add_argument("--port", type=int)
    sub.add_parser("stop").set_defaults(fn=lambda a: stop_server())
    sub.add_parser("status").set_defaults(fn=lambda a: status())
    t = sub.add_parser("train", help="start/resume background training (options passed through to train.py)")
    t.add_argument("args", nargs="*")
    t.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint and start from scratch")
    t.set_defaults(fn=lambda a: start_train((["--fresh"] if a.fresh else []) + a.args))
    sub.add_parser("pull", help="download datasets").set_defaults(fn=lambda a: pull())
    sub.add_parser("evaluate", help="fit the rotation, build the gallery, measure the benchmarks").set_defaults(fn=lambda a: evaluate())
    sub.add_parser("chat", help="terminal REPL").set_defaults(fn=lambda a: chat())
    sub.add_parser("version").set_defaults(fn=lambda a: version())
    args = p.parse_args()
    if not args.cmd:  # bare `jepa`
        start_server(False, DEFAULT_PORT, do_open=True)
        return
    if args.cmd == "serve":
        start_server(True, args.port or DEFAULT_PORT, do_open=True)
        return
    args.fn(args)


def version() -> None:
    import torch

    stack = f"torch {torch.__version__}"
    try:
        import stable_pretraining

        stack += f" · stable-pretraining {getattr(stable_pretraining, '__version__', 'ab836bf')}"
    except Exception:
        pass
    print(f"  jepa 1.0 — {stack} — CPU (this box); paper: arXiv:2610.39302")


if __name__ == "__main__":
    main()
