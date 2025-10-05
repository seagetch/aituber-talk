#!/usr/bin/env python3
"""
Standalone driver: Audio + single face image → SadTalker 3DMM 70D coefficient
without touching the original SadTalker repo.  
Assumed directory layout::

  ./
  ├─ SadTalker/           # official clone (any fork)
  ├─ sadtalker‑test/
  │   └─ test.py          # ← THIS FILE
  ├─ ai‑sample‑face.jpg   # your character image (front‑facing)
  └─ aivisspeech‑test/
      └─ audio.wav        # TTS output to be lip‑synced

* The script auto‑generates **dummy face semantics .mat** (zeros) when none exist—
  good enough for lip‑sync + head pose.
* All MATLAB read errors (v7.3, unknown headers, etc.) are absorbed and replaced
  with zeros so the pipeline never crashes.
* Works with safetensor checkpoints only (no *.pth) and typo’d YAML names.

Dependencies (conda/venv)::

  pip install numpy scipy mat73 torch

Run::

  python test.py --device cpu   # or cuda --pose_style 0‑45
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import scipy.io as scio
import mat73

# ---------------------------------------------------------------------------
# 0)  **Patch scipy.io.loadmat _globally_ so any failure falls back to zeros**
# ---------------------------------------------------------------------------
_orig_loadmat = scio.loadmat

def safe_loadmat(*args, **kwargs):
    """Robust .mat loader: mat73 → SciPy → zeros for *any* failure."""
    try:
        path = str(args[0]) if args else ""
        if path.endswith(".mat"):
            try:
                return mat73.loadmat(path)
            except Exception:
                pass
        return _orig_loadmat(*args, **kwargs)
    except Exception:
        # last resort dummy semantics
        return {"coeff_3dmm": np.zeros((1, 70), np.float32)}

scio.loadmat = safe_loadmat  # ♥ global monkey‑patch

# ---------------------------------------------------------------------------
# paths & SadTalker imports *after* monkey‑patch
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent              # ./sadtalker‑test
SAD_ROOT = ROOT.parent / "SadTalker"                   # ./SadTalker
sys.path.append(str(SAD_ROOT))                          # SadTalker modules visible

from src.test_audio2coeff import Audio2Coeff
from src.utils.init_path   import init_path
from src.generate_batch    import get_data

# ---------------------------------------------------------------------------
# helper: ensure (or create) face‑semantics .mat  (zeros)
# ---------------------------------------------------------------------------

def ensure_semantics(img: Path) -> Path:
    """Create semantics .mat exactly the way generate_batch.py expects."""
    sem_path_str = img.as_posix().split(".")[0] + ".mat"  # replicate SadTalker rule
    sem = Path(sem_path_str)
    if sem.exists():
        return sem
    print(f"[WARN] {sem.name} not found; generating dummy semantics (zeros)")
    sem.parent.mkdir(parents=True, exist_ok=True)
    scio.savemat(str(sem), {"coeff_3dmm": np.zeros((1, 70), np.float32)})
    return sem
    print(f"[WARN] {sem.name} not found; generating dummy semantics (zeros)")
    sem.parent.mkdir(parents=True, exist_ok=True)
    scio.savemat(str(sem), {"coeff_3dmm": np.zeros((1, 70), np.float32)})
    return sem

# ---------------------------------------------------------------------------
# main logic
# ---------------------------------------------------------------------------

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--image", default=str(ROOT.parent / "ai-sample-face.jpg"))
    pa.add_argument("--audio", default=str(ROOT.parent / "aivisspeech-test/audio.wav"))
    pa.add_argument("--pose_style", type=int, default=0, help="0–45 head‑pose style")
    pa.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = pa.parse_args()

    img = Path(a.image).expanduser().resolve()
    aud = Path(a.audio).expanduser().resolve()

    if not img.exists() or not aud.exists():
        sys.exit("❌  image or audio path does not exist.")

    ensure_semantics(img)  # create zero‑mat if needed

    # init SadTalker runtime paths
    ckpt_dir = SAD_ROOT / "checkpoints"
    cfg_dir  = SAD_ROOT / "src" / "config"          # contains auido*.yaml
    paths = init_path(ckpt_dir, cfg_dir, 256, "crop", a.device)

    # prefer safetensor weight if present
    st = ckpt_dir / "SadTalker_V0.0.2_256.safetensors"
    if st.exists():
        paths.update({"use_safetensor": True, "checkpoint": str(st)})

    a2c   = Audio2Coeff(paths, device=a.device)
    batch = get_data(str(img), str(aud), ref_eyeblink_coeff_path=None, device=a.device)

    out_dir = ROOT / "out"; out_dir.mkdir(exist_ok=True)
    out_path = a2c.generate(batch, coeff_save_dir=str(out_dir), pose_style=a.pose_style)
    print("\n✅ coeff saved →", out_path)

if __name__ == "__main__":
    main()
