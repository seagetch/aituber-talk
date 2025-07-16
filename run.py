#!/usr/bin/env python3
"""
run.py –– text → (aivisspeech) → wav → (SadTalker) → coeff
        → SDL 再生 + OSC/VMC 送信   *2025‑07‑18 stable*

■ 特徴
- Python 3.8 互換
- Producer / Consumer 2 thread + queue（バックプレッシャ付き）
- /audio_query は POST、text & speaker は query‑string
- *.mat 安全ロード・瞬き乱数・NaN/Inf・旧/新版 generate() すべて対策済
"""

from __future__ import annotations
import argparse, os, re, signal, sys, time, tempfile, threading, queue, math
from pathlib import Path
from typing import Tuple
import requests, numpy as np, librosa

# ---------------------------------------------------------------------------
# 0) SciPy .mat → 安全ロード (v7.3/HDF5 & 例外時ゼロ埋め)
# ---------------------------------------------------------------------------
import scipy.io as scio
try:
    import mat73
except ImportError:
    mat73 = None
_orig_loadmat = scio.loadmat
def _safe_loadmat(path, *args, **kwargs):
    try:
        if isinstance(path, (str, bytes)) and str(path).endswith(".mat"):
            if mat73 is not None:
                try:
                    return mat73.loadmat(path)
                except Exception:
                    pass
        return _orig_loadmat(path, *args, **kwargs)
    except Exception:
        return {"coeff_3dmm": np.zeros((1, 70), np.float32)}
scio.loadmat = _safe_loadmat

from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_message_builder import OscMessageBuilder
from scipy.spatial.transform import Rotation as R
import pygame

# SadTalker を PYTHONPATH に
SAD_ROOT = Path(__file__).resolve().parent / "SadTalker"
sys.path.append(str(SAD_ROOT))

# ---------------------------------------------------------------------------
# 1) まばたき乱数 – 短フレーム安全化
# ---------------------------------------------------------------------------
from src import generate_batch as _gb
_orig_blink_fn = _gb.generate_blink_seq_randomly
def _safe_blink(num_frames: int):
    return [0] * num_frames if num_frames < 10 else _orig_blink_fn(num_frames)
_gb.generate_blink_seq_randomly = _safe_blink

from src.test_audio2coeff import Audio2Coeff
from src.utils.init_path   import init_path
from src.generate_batch    import get_data

# ---------------------------------------------------------------------------
# 2) 定数
# ---------------------------------------------------------------------------
DEFAULT_STYLE_ID   = 888753760
DEFAULT_IMAGE_PATH = Path(__file__).resolve().parent / "ai-sample-face.jpg"
SENT_SPLIT_RE = re.compile(r'(?<=[。！？\?！])\s*')

def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]

# ---------------------------------------------------------------------------
# 3) aivisspeech – text → wav bytes
# ---------------------------------------------------------------------------
def tts_generate(text: str, style_id: int,
                 host="127.0.0.1", port=10101) -> bytes:
    rq = requests.post(f"http://{host}:{port}/audio_query",
                       params={"speaker": style_id, "text": text})
    rq.raise_for_status()
    query = rq.content
    rs = requests.post(f"http://{host}:{port}/synthesis",
                       params={"speaker": style_id},
                       data=query,
                       headers={"Content-Type": "application/json"})
    rs.raise_for_status()
    return rs.content

# ---------------------------------------------------------------------------
# 4) SadTalker – wav + image → coeff ndarray
# ---------------------------------------------------------------------------
def sadtalker_coeff(image: Path, wav_path: Path,
                    device: str, pose_style: int) -> np.ndarray:
    ckpt_dir = SAD_ROOT / "checkpoints"
    cfg_dir  = SAD_ROOT / "src" / "config"
    paths = init_path(ckpt_dir, cfg_dir, 256, "crop", device)
    st_ckpt = ckpt_dir / "SadTalker_V0.0.2_256.safetensors"
    if st_ckpt.exists():
        paths.update({"use_safetensor": True, "checkpoint": str(st_ckpt)})

    a2c   = Audio2Coeff(paths, device=device)
    batch = get_data(str(image), str(wav_path),
                     ref_eyeblink_coeff_path=None,
                     device=device)

    try:
        # 新版 generate は ndarray 返し
        return a2c.generate(batch,
                            coeff_save_dir=None,
                            pose_style=pose_style,
                            return_path=False).astype(np.float32)
    except TypeError:
        # 旧版 generate は .mat パス返し
        with tempfile.TemporaryDirectory() as tmpdir:
            mat_path = a2c.generate(batch,
                                    coeff_save_dir=tmpdir,
                                    pose_style=pose_style)
            coeff = scio.loadmat(mat_path)["coeff_3dmm"].astype(np.float32)
        return coeff

# ---------------------------------------------------------------------------
# 5) OSC / VMC 送信
# ---------------------------------------------------------------------------
AR52 = [
    "browDown_L","browDown_R","browInnerUp","browOuterUp_L","browOuterUp_R",
    "cheekPuff","cheekSquint_L","cheekSquint_R",
    "eyeBlink_L","eyeBlink_R","eyeLookDown_L","eyeLookDown_R",
    "eyeLookIn_L","eyeLookIn_R","eyeLookOut_L","eyeLookOut_R",
    "eyeLookUp_L","eyeLookUp_R","eyeSquint_L","eyeSquint_R",
    "eyeWide_L","eyeWide_R",
    "jawForward","jawLeft","jawOpen","jawRight",
    "mouthClose","mouthDimple_L","mouthDimple_R","mouthFrown_L","mouthFrown_R",
    "mouthFunnel","mouthLeft","mouthLowerDown_L","mouthLowerDown_R",
    "mouthPress_L","mouthPress_R","mouthPucker","mouthRight",
    "mouthRollLower","mouthRollUpper","mouthShrugLower","mouthShrugUpper",
    "mouthSmile_L","mouthSmile_R","mouthStretch_L","mouthStretch_R",
    "mouthUpperUp_L","mouthUpperUp_R",
    "noseSneer_L","noseSneer_R","tongueOut"
]
EXP_NAMES = AR52 + [f"exp{idx:02d}" for idx in range(52, 64)]
OSC_BONE = "/VMC/Ext/Bone/Pos"
OSC_BVAL = "/VMC/Ext/Blend/Val"
OSC_APPL = "/VMC/Ext/Blend/Apply"
HEAD     = "Head"

def _pose_msg(vals):
    b = OscMessageBuilder(address=OSC_BONE)
    b.add_arg(HEAD)
    for v in vals:
        b.add_arg(float(v))
    return b.build().dgram

def map_exp(exp_row: np.ndarray) -> np.ndarray:
    # ReLU + NaN/Inf サニタイズ
    out = np.clip(exp_row, 0.0, 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

def send_frame(sock: SimpleUDPClient, row: np.ndarray):
    yaw, pitch, roll = row[64:67]
    tx, ty, tz       = row[67:70]
    qx,qy,qz,qw      = R.from_euler("yxz", [yaw, pitch, roll]).as_quat()
    sock._sock.sendto(_pose_msg([tx,ty,tz,qx,qy,qz,qw]),
                      (sock._address, sock._port))

    safe_vals = (
        0.0 if (math.isnan(x) or math.isinf(x)) else float(x)
        for x in map_exp(row[:64]).astype(np.float32)
    )
    for name, v in zip(EXP_NAMES, safe_vals):
        sock.send_message(OSC_BVAL, [name, v])
    sock.send_message(OSC_APPL, [])

# ---------------------------------------------------------------------------
# 6) Producer / Consumer スレッド
# ---------------------------------------------------------------------------
Task = Tuple[bytes, np.ndarray]

def producer(sentences, image, q, style_id, device, pose_style):
    for i, sent in enumerate(sentences, 1):
        print(f"⏳ [Producer] {i}/{len(sentences)} TTS...")
        wav_bytes = tts_generate(sent, style_id)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
            tf.write(wav_bytes); tf.flush()
            print("   ⏳ SadTalker...")
            coeff = sadtalker_coeff(image, Path(tf.name), device, pose_style)
        q.put((wav_bytes, coeff))
    q.put((None, None))
    print("[Producer] done.")

def consumer(q, sock):
    pygame.init(); pygame.mixer.init()
    while True:
        wav_bytes, coeff = q.get()
        if wav_bytes is None:
            break
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(wav_bytes); wav_path = tf.name
        sound  = pygame.mixer.Sound(wav_path)
        length = sound.get_length()
        frames = len(coeff)
        dt     = length / frames if frames else 1/25
        ch     = sound.play()
        start  = time.perf_counter()
        next_t = 0.0
        idx    = 0
        while ch.get_busy() and idx < frames:
            now = time.perf_counter() - start
            while idx < frames and now >= next_t:
                send_frame(sock, coeff[idx])
                idx += 1; next_t += dt
            time.sleep(0.001)
        while idx < frames:
            send_frame(sock, coeff[idx]); idx += 1
        os.unlink(wav_path)
        print(f"[Consumer] segment done ({length:.2f}s)")
    pygame.mixer.quit(); pygame.quit()
    print("[Consumer] finished.")

# ---------------------------------------------------------------------------
# 7) main
# ---------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--text", required=True)
    pa.add_argument("--style_id", type=int, default=DEFAULT_STYLE_ID)
    pa.add_argument("--image", default=str(DEFAULT_IMAGE_PATH))
    pa.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    pa.add_argument("--pose_style", type=int, default=0)
    pa.add_argument("--osc_host", default="127.0.0.1")
    pa.add_argument("--osc_port", type=int, default=39540)
    args = pa.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        sys.exit("❌ image not found")

    sentences = split_sentences(args.text)
    if not sentences:
        sys.exit("❌ empty text")

    q: "queue.Queue[Task]" = queue.Queue(maxsize=3)
    sock = SimpleUDPClient(args.osc_host, args.osc_port)

    th_prod = threading.Thread(target=producer,
                               args=(sentences, image_path, q,
                                     args.style_id, args.device, args.pose_style),
                               daemon=True)
    th_cons = threading.Thread(target=consumer, args=(q, sock), daemon=True)

    th_cons.start(); th_prod.start()
    try:
        th_prod.join(); th_cons.join()
    except KeyboardInterrupt:
        print("Interrupted")
    print("✅ All done.")

if __name__ == "__main__":
    main()
