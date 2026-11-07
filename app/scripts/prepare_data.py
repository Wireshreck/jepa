"""Downloads everything the local install needs (jepa pull). Order matters: the tiny
text inputs that drive training come first so training can start while photos download.

 1. repo data tables: COCO Karpathy captions + STS-B               (python -m prh.data)
 2. Flickr30k annotations (Karpathy split CSV)                     (HF: nlphuji/flickr30k)
 3. COCO Karpathy 5k test captions                                 (HF: nlphuji/mscoco_2014_5k_test_image_text_retrieval)
 4. frozen-image weights: DINOv2 ViT-B/14 for the benchmark        (timm -> HF)
 5. photos: Flickr30k images zip (~8 GB) and COCO-5k images zip     (both zips, extracted)
 6. the WIT-1024 pairs of the PRH paper                            (HF: minhuh/prh, used by jepa evaluate)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "repo"
DATA = REPO / "data"
sys.path.insert(0, str(REPO))

FLICKR = DATA / "flickr30k"
COCO5K = DATA / "coco5k"


def sh(*cmd: str) -> int:
    print("+ " + " ".join(c for c in map(str, cmd)), flush=True)
    r = subprocess.run(list(map(str, cmd)), cwd=str(REPO))
    return r.returncode


def hf_repo_dir(repo_id: str, dirname: Path, allow: list[str], repo_type: str = "dataset") -> None:
    from huggingface_hub import snapshot_download

    print(f"# downloading {repo_id} -> {dirname}", flush=True)
    dirname.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id, repo_type=repo_type, local_dir=str(dirname),
                      allow_patterns=allow, token=os.environ.get("HF_TOKEN"))
    for f in allow:
        pass


def unzip(zip_path: Path, dest: Path, want_dir: str, expect: Path) -> None:
    """Extract only the members of ``want_dir`` (or every member when absent) and make sure ``expect`` exists."""
    print(f"# extracting {zip_path.name} ({zip_path.stat().st_size / 2**30:.1f} GiB) -> {dest}", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        inside = [n for n in names if n.startswith(want_dir)]
        use = inside if inside else names
        t0 = time.time()
        for i, n in enumerate(use):
            i and (i % 2000 == 0) and print(f"    {i}/{len(use)} files ({time.time() - t0:.0f} s)", flush=True)
            z.extract(n, dest)
    if not expect.exists():
        raise SystemExit(f"expected {expect} after extraction; layout differs from the paper's repo")
    print(f"# done: {expect} ready", flush=True)


def main() -> None:
    venv_py = str(ROOT / ".venv" / "Scripts" / "python.exe") if os.name == "nt" else str(ROOT / ".venv" / "bin" / "python")

    # 1. COCO captions + STS-B (both drive training and the text-only checks)
    if not (DATA / "coco_train.parquet").exists() or not (DATA / "stsb_test.parquet").exists():
        sh(venv_py, "-m", "prh.data")

    # 2. Flickr30k annotations (the training captions and the val/test pairs)
    FLICKR.mkdir(parents=True, exist_ok=True)
    if not (FLICKR / "flickr_annotations_30k.csv").exists():
        from huggingface_hub import hf_hub_download

        p = hf_hub_download("nlphuji/flickr30k", "flickr_annotations_30k.csv", repo_type="dataset")
        (FLICKR / "flickr_annotations_30k.csv").write_bytes(Path(p).read_bytes())
    print("# Flickr30k annotations ready", flush=True)

    # 3. COCO-5k test captions
    COCO5K.mkdir(parents=True, exist_ok=True)
    if not (COCO5K / "test_5k_mscoco_2014.csv").exists():
        from huggingface_hub import hf_hub_download

        p = hf_hub_download("nlphuji/mscoco_2014_5k_test_image_text_retrieval", "test_5k_mscoco_2014.csv", repo_type="dataset")
        (COCO5K / "test_5k_mscoco_2014.csv").write_bytes(Path(p).read_bytes())

    # 4. DINOv2 (ViT-B/14) weights for the benchmark + the demo's photo embeddings
    import timm
    import torch

    print("# fetching DINOv2 ViT-B/14 weights (timm)", flush=True)
    timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
    print("# DINOv2 ready", flush=True)

    # 5. photos (large; continue to the benchmark data even if these are slow)
    try:
        if not (FLICKR / "flickr30k-images" / ".done").exists():
            if not (FLICKR / "flickr30k-images.zip").exists():
                url = "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip"
                core_download(url, FLICKR / "flickr30k-images.zip")
            unzip(FLICKR / "flickr30k-images.zip", FLICKR, "flickr30k-images/", FLICKR / "flickr30k-images")
            (FLICKR / "flickr30k-images" / ".done").write_text("", encoding="utf-8")
        if not (COCO5K / "images_mscoco_2014_5k_test" / ".done").exists():
            if not (COCO5K / "images_mscoco_2014_5k_test.zip").exists():
                url = "https://huggingface.co/datasets/nlphuji/mscoco_2014_5k_test_image_text_retrieval/resolve/main/images_mscoco_2014_5k_test.zip"
                core_download(url, COCO5K / "images_mscoco_2014_5k_test.zip")
            unzip(COCO5K / "images_mscoco_2014_5k_test.zip", COCO5K, "images_mscoco_2014_5k_test/",
                  COCO5K / "images_mscoco_2014_5k_test")
            (COCO5K / "images_mscoco_2014_5k_test" / ".done").write_text("", encoding="utf-8")
    except Exception as e:
        print(f"! photos download failed ({e}); evaluation can start once they arrive", flush=True)

    # 6. WIT-1024 (the benchmark of the PRH paper; ~1024 photos with their captions)
    try:
        import datasets

        print("# fetching minhuh/prh wit_1024", flush=True)
        ds = datasets.load_dataset("minhuh/prh", revision="wit_1024", split="train")
        print(f"# WIT-1024 ready ({len(ds)} pairs)", flush=True)
    except Exception as e:
        print(f"! WIT-1024 not ready ({e}); jepa evaluate will fetch it later", flush=True)

    print("# pull complete", flush=True)


def core_download(url: str, to: Path) -> None:
    t0 = time.time()
    print(f"# downloading {url}", flush=True)

    def hook(n: int, block: int, total: int) -> None:
        done = (n * block) / 2**30
        now = time.time()
        if now - getattr(core_download, "_last", 0) > 30 or (total and n * block >= total):
            core_download._last = now  # type: ignore[attr-defined]
            print(f"    {done:.1f} GiB{f' / {total / 2**30:.1f}' if total else ''} in {now - t0:.0f} s", flush=True)

    urllib.request.urlretrieve(url, to, hook)
    print(f"# downloaded {to} ({to.stat().st_size / 2**30:.1f} GiB)", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"# pull took {time.time() - t0:.0f} s", flush=True)
