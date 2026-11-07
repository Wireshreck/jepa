"""Windows-only launcher for repo/train.py — a thin adapter; the repo script runs unchanged.

stable-pretraining's Manager is the training lifecycle (per AGENTS.md) and it reports the
four Unix signals it manages; Windows has no SIGUSR1/SIGUSR2/SIGCONT, so we install
harmless stand-in constants before train.py executes, and force utf-8 streams (the default
cp1252 console breaks the framework's logging on any non-latin1 glyph).
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

if os.name == "nt":
    import enum
    import signal

    missing = {name: num for name, num in (("SIGUSR1", 10), ("SIGUSR2", 12), ("SIGCONT", 18))
               if not hasattr(signal, name)}
    if missing:
        standins = enum.IntEnum("PosixSignalStandins", missing)  # .name works; getsignal reads the slot
        for name in missing:
            setattr(signal, name, getattr(standins, name))

train = Path(__file__).resolve().parents[1] / "repo" / "train.py"
for stream in (sys.stdout, sys.stderr):  # foreground runs: stdio encoding was fixed at process start
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
sys.argv[0] = str(train)
runpy.run_path(str(train), run_name="__main__")
