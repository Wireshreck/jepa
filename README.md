# jepa — a local install of the 109M-parameter text JEPA

This folder installs and serves the **109M text JEPA** from *“Proving and Improving the Platonic
Representation Hypothesis with JEPAs”* (Randall Balestriero & Yann LeCun, 2026,
[arXiv:2610.39302](https://arxiv.org/abs/2610.39302)) — the model whose headline result is beating a
72B-parameter LLM at cross-modal reasoning (WIT-1024 alignment with DINOv2-B: **.184**, vs Qwen2.5-72B
**.154**, Llama-3-70B **.151**, CLIP's text encoder **.124**).

**Why train it here?** The lab's official checkpoints are restricted. This install therefore trains
the *exact architecture* (`--arch base` = 108,891,648 parameters) with the **paper's own repository**
(`repo/`, galilai-group/prh-proof, MIT) and its training framework (`stable-pretraining` at the paper's
revision). Nothing here re-implements the paper's maths or training loops.

## Quick start

| command | what it does |
|---|---|
| `jepa` | start the web interface in the background and open it in your browser (like ollama) |
| `jepa status` | what is ready: server, training, checkpoint, datasets, rotation, gallery, metrics |
| `jepa serve` | the same web interface, in the foreground (Ctrl+C to stop) |
| `jepa stop` | stop the background server |
| `jepa train` | start (or resume) local training in the background; `jepa train --fresh` starts over |
| `jepa evaluate` | measure alignment + fit the rotation + build the photo gallery (after a checkpoint exists) |
| `jepa chat` | tiny terminal REPL: type text → nearest captions / photos |
| `jepa pull` | (re)download every dataset this install uses (~16 GB; captions first, photos last) |
| `jepa version` | versions of the stack |

The `jepa` command was installed to `%LOCALAPPDATA%\Programs\jepa\jepa.bat` and put on your PATH
(user scope, no admin). In a terminal opened *before* that PATH change, call it by full path:
`%LOCALAPPDATA%\Programs\jepa\jepa.bat`.

### The web interface

`http://localhost:7860` (also opened automatically by `jepa`):

- **Home** — one query box; search the photo gallery or the caption world by description.
- **Find photos** — your description goes through the text JEPA, then through *one fitted rotation*, and
  lands next to photos the frozen DINOv2 (no captions in its training) embedded. No member of this
  pipeline was ever trained on an image–caption pair; the pairs only fit the rotation.
- **Neighbours** — the model's text space: nearest captions of the COCO-5k test bank.
- **Benchmarks** — this install's numbers (measured with `repo/evaluate.py`, the paper's protocol) next
  to the paper's reported numbers.
- **Training** — live step/epoch counters, the log tail, start/resume/stop.
- **About** — the model card.

## What's installed

```
C:\Users\user\JEPA\
├── repo\          the paper's own code (prh-proof, with docs/, tests/, evaluate.py, train.py)
├── data\          its datasets: repo\data\coco_train.parquet, stsb_test.parquet,
│                  repo\data\flickr30k\{flickr_annotations_30k.csv, flickr30k-images\},
│                  repo\data\coco5k\{test_5k_mscoco_2014.csv, images_mscoco_2014_5k_test\}
├── app\
│   ├── config.json          the model/serving/training configuration (single source of every CLI flag)
│   ├── jepa.py              the CLI (`jepa ...`)
│   ├── jepa_server.py       the FastAPI web app (port 7860)
│   ├── jepa_runtime.py      loads the newest checkpoint + rotation + galleries; hot-swaps fresh saves
│   ├── win_train.py         thin Windows adapter around repo\train.py (signal stand-ins + UTF-8 logging)
│   ├── scripts\             the pipelines behind `jepa pull` and `jepa evaluate`
│   ├── state\               artifacts: state\ckpts\runs\...\last.ckpt, rotation.pt, gallery.pt,
│   │                        captions.pt, metrics.json, server.json
│   ├── logs\                server.log, train_*.log
│   └── static\              the web page (plain HTML/JS, no build step)
└── .venv\         python 3.13 venv: torch 2.15.0+cpu, stable-pretraining (revision ab836bf),
                   lightning, timm 1.0.31 (DINOv2), transformers, datasets, fastapi, uvicorn
```

## Training on this CPU: what to expect, honestly

The paper's flagship 109M run used ~900M caption samples on 16 GPUs. This machine has an i5-11400F
(6 cores / 12 threads, no usable GPU), so `app/config.json` shaped the same objective down:

| setting | here | why |
|---|---|---|
| architecture | `--arch base` (the 109M) | the paper's model, unchanged |
| objective | LeJEPA **good views** (`--caption-views`) | the paper's core trick: two captions of one image |
| views | `--views 2 1` | 2 global + 1 local caption view (the repo's LeJEPA needs ≥1 local view) |
| tokens | `--max-len 32` | captions are short; half the transformer compute |
| batch / λ | 16 / 0.16 | SIGReg's effective weight scales with batch: paper 0.02×128 → here 0.16×16 |
| schedule | `epoch-size 4096` × 50 epochs ≈ 12,800 steps ≈ 4 s/step ≈ **~15 h** | checkpoints commit every 250 steps (~20 min) |
| precision | fp32 (CPU) | `precision: "32"`, gradient clip 1.0 |

Data: COCO Karpathy captions (113k images × 5) + Flickr30k captions (31k × 5), good-view pairs only.
Validation uses the Flickr30k val+test caption pairs (text-only, no images needed).

- The server serves whatever checkpoint exists **and hot-swaps newer ones** as training progresses;
  early checkpoints (steps 1–15k) read as random until the first few thousand samples.
- `jepa evaluate` measures Kapathy-split Flickr retrieval through one rotation (fit on train+val pairs,
  scored on test pairs) and the paper's WIT-1024 protocol against DINOv2-B. Early numbers will be far
  below the quick-start reference (good-view JEPAs reach R@1 ≈ .25 there after 200 epochs on a GPU) —
  the honest local expectation is "improving with every epoch", visible on the Benchmarks tab by
  re-running `jepa evaluate` later.

## Notes

- Everything is user-scoped (no admin was needed; no UAC prompts).
- The first ask of `jepa` may take a minute: the server starts before the 1.4 GB checkpoint finishes
  loading (status shows `loading`, then `ready`).
- `HF_TOKEN` is optional; without it the Hub is polite but slower.
- Datasets and pretrained weights (DINOv2 ViT-B/14 via timm, WIT-1024, BERT tokenizer) download from
  the Hugging Face Hub on first use and live in `%USERPROFILE%\.cache\huggingface` + `repo\data`.
- The photo gallery and the rotation are built once checkpoints + photos exist: `jepa evaluate`.
- Paper: <https://arxiv.org/abs/2610.39302> · code: <https://github.com/galilai-group/prh-proof>
  · live demo of the *official* model: <https://huggingface.co/spaces/galilai-group/prh_demo>
