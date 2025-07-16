#!/usr/bin/env python3
"""
run.py – stable base + Producer multiprocess  *2025‑07‑18 mp‑fixed‑full*
"""

from __future__ import annotations
import argparse, os, re, sys, time, tempfile, math, signal, threading
from pathlib import Path
from typing import Tuple
from multiprocessing import Process, Queue as MPQueue
import requests, numpy as np

# ---------------------------------------------------------------------------
# 0) SciPy .mat → 安全ロード
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

# ---------------------------------------------------------------------------
# SadTalker patches
# ---------------------------------------------------------------------------
SAD_ROOT = Path(__file__).resolve().parent / "SadTalker"
sys.path.append(str(SAD_ROOT))

from src import generate_batch as _gb
_orig_blink = _gb.generate_blink_seq_randomly
def _safe_blink(n): return [0]*n if n < 10 else _orig_blink(n)
_gb.generate_blink_seq_randomly = _safe_blink

from src.test_audio2coeff import Audio2Coeff
from src.utils.init_path   import init_path
from src.generate_batch    import get_data

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
DEFAULT_STYLE_ID   = 888753760
DEFAULT_IMAGE_PATH = Path(__file__).resolve().parent / "ai-sample-face.jpg"
SENT_RE = re.compile(r'(?<=[。！？\?！])\s*')
split_sentences = lambda t: [s.strip() for s in SENT_RE.split(t) if s.strip()]

# ---------------------------------------------------------------------------
# 1) aivisspeech
# ---------------------------------------------------------------------------
def tts_generate(text: str, style: int,
                 host="127.0.0.1", port=10101) -> bytes:
    rq = requests.post(f"http://{host}:{port}/audio_query",
                       params={"speaker": style, "text": text})
    rq.raise_for_status()
    rs = requests.post(f"http://{host}:{port}/synthesis",
                       params={"speaker": style},
                       data=rq.content,
                       headers={"Content-Type": "application/json"})
    rs.raise_for_status()
    return rs.content

# ---------------------------------------------------------------------------
# 2) SadTalker coeff
# ---------------------------------------------------------------------------
def sadtalker_coeff(img: Path, wav_path: Path,
                    device: str, pose_style: int) -> np.ndarray:
    ckpt_dir = SAD_ROOT / "checkpoints"
    cfg_dir  = SAD_ROOT / "src" / "config"
    paths = init_path(ckpt_dir, cfg_dir, 256, "crop", device)
    st = ckpt_dir / "SadTalker_V0.0.2_256.safetensors"
    if st.exists():
        paths.update({"use_safetensor": True, "checkpoint": str(st)})

    a2c = Audio2Coeff(paths, device=device)
    batch = get_data(
        str(img),                   # image_path (positional)
        str(wav_path),              # audio_path (positional)
        ref_eyeblink_coeff_path=None,
        device=device)              # ← 重複なし

    try:
        return a2c.generate(batch,
                            coeff_save_dir=None,
                            pose_style=pose_style,
                            return_path=False).astype(np.float32)
    except TypeError:
        # 旧版
        with tempfile.TemporaryDirectory() as td:
            mat_path = a2c.generate(batch,
                                    coeff_save_dir=td,
                                    pose_style=pose_style)
            return scio.loadmat(mat_path)["coeff_3dmm"].astype(np.float32)

# ---------------------------------------------------------------------------
# 3) OSC / VMC ヘルパ
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
    for v in vals: b.add_arg(float(v))
    return b.build().dgram

def map_exp(e: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.clip(e, 0, 1), nan=0.0, posinf=1.0, neginf=0.0)

def send_frame(sock: SimpleUDPClient, row: np.ndarray):
    yaw,pit,rol = row[64:67]; tx,ty,tz = row[67:70]
    qx,qy,qz,qw = R.from_euler("yxz", [yaw,pit,rol]).as_quat()
    sock._sock.sendto(_pose_msg([tx,ty,tz,qx,qy,qz,qw]),
                      (sock._address, sock._port))
    for name, val in zip(EXP_NAMES, map_exp(row[:64])):
        sock.send_message(OSC_BVAL, [name, float(val)])
    sock.send_message(OSC_APPL, [])

# ---------------------------------------------------------------------------
# 4) Producer (multiprocessing)
# ---------------------------------------------------------------------------
Task = Tuple[bytes, np.ndarray]

def producer_proc(sents, img_path, q: MPQueue,
                  style_id, device, pose_style):
    for i, sent in enumerate(sents, 1):
        print(f"⏳ [Producer] {i}/{len(sents)} TTS...")
        wav = tts_generate(sent, style_id)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
            tf.write(wav); tf.flush()
            print("   ⏳ SadTalker...")
            coeff = sadtalker_coeff(img_path, Path(tf.name), device, pose_style)
        q.put((wav, coeff))
    q.put((None, None))
    print("[Producer] done.")

# ---------------------------------------------------------------------------
# 5) Consumer (thread)
# ---------------------------------------------------------------------------
def consumer_thread(q: MPQueue, sock: SimpleUDPClient):
    import pygame, pygame.mixer
    pygame.init(); pygame.mixer.init()
    while True:
        wav, coeff = q.get()
        if wav is None: break
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(wav); wav_path = tf.name
        sound  = pygame.mixer.Sound(wav_path)
        length = sound.get_length()
        frames = len(coeff)
        dt = length/frames if frames else 1/25
        ch = sound.play()
        start = time.perf_counter(); idx = 0; next_t = 0.0
        while ch.get_busy() and idx < frames:
            now = time.perf_counter() - start
            while idx < frames and now >= next_t:
                send_frame(sock, coeff[idx]); idx+=1; next_t+=dt
            time.sleep(0.001)
        while idx < frames:
            send_frame(sock, coeff[idx]); idx+=1
        os.unlink(wav_path)
        print(f"[Consumer] segment done ({length:.2f}s)")
    pygame.mixer.quit(); pygame.quit()
    print("[Consumer] finished.")

# ---------------------------------------------------------------------------
# 6) main
# ---------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--text", required=True)
    pa.add_argument("--style_id", type=int, default=DEFAULT_STYLE_ID)
    pa.add_argument("--image", default=str(DEFAULT_IMAGE_PATH))
    pa.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    pa.add_argument("--pose_style", type=int, default=0)
    pa.add_argument("--osc_host", default="127.0.0.1")
    pa.add_argument("--osc_port", type=int, default=39540)
    args = pa.parse_args()

    img_path = Path(args.image).expanduser().resolve()
    if not img_path.exists(): sys.exit("❌ image not found")

    sentences = split_sentences(args.text)
    if not sentences: sys.exit("❌ empty text")

    q: MPQueue[Task] = MPQueue(maxsize=10)
    sock = SimpleUDPClient(args.osc_host, args.osc_port)

    prod = Process(target=producer_proc,
                   args=(sentences, img_path, q,
                         args.style_id, args.device, args.pose_style),
                   daemon=True)
    cons = threading.Thread(target=consumer_thread,
                            args=(q, sock), daemon=True)

    cons.start(); prod.start()
    try:
        prod.join(); cons.join()
    except KeyboardInterrupt:
        print("Interrupted")
    print("✅ All done.")

if __name__ == "__main__":
    main()
