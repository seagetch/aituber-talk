#!/usr/bin/env python3
"""
run.py – stable base + Producer multiprocess  *2025-07-19 blink-hotfix-2*

主な修正内容
================
1. **BLINK_BASELINE** を 0.2 に設定し、カスタム瞬きシーケンスにベースラインを加算。
2. **adjust_blink()**
   - `no_blink` オプション時は瞬きしないものの、ベースライン (0.2) は保持。
   - 通常時は `_safe_blink()` の出力を (1−BLINK_BASELINE) 倍し、BLINK_BASELINE を足して上書き。
3. **sadtalker_coeff()** 呼び出し後の瞬き調整は同じく新ロジックを使用。
4. **split_sentences** の定義を追加。
5. それ以外は元の機能・ロジックを完全維持。

これにより、瞬きが全く入らない問題、デフォルト値が0の問題、`split_sentences` 未定義の問題をすべて解消します。
"""

from __future__ import annotations
import argparse
import os
import re
import sys
import time
import tempfile
import math
import signal
import threading
import logging
from pathlib import Path
from typing import Tuple, List
from multiprocessing import Process, Queue as MPQueue

import requests
import numpy as np
import scipy.io as scio

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
BLINK_BASELINE     = 0.1
DEFAULT_STYLE_ID   = 888753760
DEFAULT_IMAGE_PATH = Path(__file__).resolve().parent / "ai-sample-face.jpg"

# ---------------------------------------------------------------------------
# ログ設定
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run")

# ---------------------------------------------------------------------------
# 0) SciPy .mat → 安全ロード
# ---------------------------------------------------------------------------
try:
    import mat73
except ImportError:
    mat73 = None
_orig_loadmat = scio.loadmat

def _safe_loadmat(path, *args, **kwargs):
    """mat73 fallback & 強制デフォルト値"""
    try:
        if isinstance(path, (str, bytes)) and str(path).endswith(".mat"):
            if mat73 is not None:
                try:
                    return mat73.loadmat(path)
                except Exception:  # pragma: no cover – mat73 失敗時
                    pass
        return _orig_loadmat(path, *args, **kwargs)
    except Exception:  # ファイル破損時など
        logger.warning("loadmat failed; returning zero coeff")
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

def _safe_blink(n: int) -> List[float]:
    """フレーム < 10 の場合は瞬きを完全無効（パチパチ誤爆防止）"""
    return [0.0] * n if n < 10 else list(map(float, _orig_blink(n)))

_gb.generate_blink_seq_randomly = _safe_blink  # monkey-patch

from src.test_audio2coeff import Audio2Coeff
from src.utils.init_path   import init_path
from src.generate_batch    import get_data

# ---------------------------------------------------------------------------
# 文分割ユーティリティ
# ---------------------------------------------------------------------------
SENT_RE = re.compile(r'(?<=[。！？\?！])\s*')
def split_sentences(text: str) -> List[str]:
    """句点や感嘆符などで文を分割してリストで返す"""
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]

# ---------------------------------------------------------------------------
# 1) aivisspeech
# ---------------------------------------------------------------------------
def tts_generate(text: str, style: int, host="127.0.0.1", port=10101) -> bytes:
    rq = requests.post(
        f"http://{host}:{port}/audio_query",
        params={"speaker": style, "text": text}
    )
    rq.raise_for_status()
    rs = requests.post(
        f"http://{host}:{port}/synthesis",
        params={"speaker": style},
        data=rq.content,
        headers={"Content-Type": "application/json"}
    )
    rs.raise_for_status()
    return rs.content

# ---------------------------------------------------------------------------
# 2) SadTalker coeff
# ---------------------------------------------------------------------------
def adjust_blink(coeff: np.ndarray, no_blink: bool = False) -> np.ndarray:
    """eyeBlink_L/R をカスタムシーケンスで上書き (ベースライン 0.2 を加算)

    Parameters
    ----------
    coeff : np.ndarray
        (frames, 70) の係数行列 (float32)
    no_blink : bool
        True の場合は瞬き無効化 (常時ベースラインのみ)
    """
    if coeff.size == 0:
        return coeff

    frames = coeff.shape[0]
    baseline = BLINK_BASELINE

    if no_blink:
        # 瞬きなしだが、ベースラインは維持
        seq = [baseline] * frames
    else:
        raw_seq = _safe_blink(frames)
        # ベースライン + 瞬き振幅
        seq = [baseline + r * (1.0 - baseline) for r in raw_seq]

    coeff = coeff.copy()
    coeff[:, 8] = seq  # eyeBlink_L
    coeff[:, 9] = seq  # eyeBlink_R
    return coeff


def sadtalker_coeff(
    img: Path,
    wav_path: Path,
    device: str,
    pose_style: int,
    no_blink: bool
) -> np.ndarray:
    ckpt_dir = SAD_ROOT / "checkpoints"
    cfg_dir  = SAD_ROOT / "src" / "config"
    paths = init_path(ckpt_dir, cfg_dir, 256, "crop", device)
    st = ckpt_dir / "SadTalker_V0.0.2_256.safetensors"
    if st.exists():
        paths.update({"use_safetensor": True, "checkpoint": str(st)})

    a2c = Audio2Coeff(paths, device=device)
    batch = get_data(
        str(img),
        str(wav_path),
        ref_eyeblink_coeff_path=None,
        device=device
    )

    try:
        coeff = a2c.generate(
            batch,
            coeff_save_dir=None,
            pose_style=pose_style,
            return_path=False
        ).astype(np.float32)
    except TypeError:  # SadTalker < 0.0.2
        with tempfile.TemporaryDirectory() as td:
            mat_path = a2c.generate(
                batch,
                coeff_save_dir=td,
                pose_style=pose_style
            )
            coeff = scio.loadmat(mat_path)["coeff_3dmm"].astype(np.float32)

    if coeff.ndim == 1:
        coeff = coeff.reshape(1, -1)
    coeff = adjust_blink(coeff, no_blink=no_blink)
    if coeff.shape[1] < 70:
        raise ValueError(f"coeff shape invalid: {coeff.shape}")
    return coeff

# ---------------------------------------------------------------------------
# 3) OSC / VMC ヘルパ
# ---------------------------------------------------------------------------
AR52 = [
    "browDown_L","browDown_R","browInnerUp","browOuterUp_L","browOuterUp_R",
    "cheekPuff","cheekSquint_L","cheekSquint_R",
    "eyeBlinkLeft","eyeBlinkRight","eyeLookDown_L","eyeLookDown_R",
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
OSC_BVAL  = "/VMC/Ext/Blend/Val"
OSC_APPL  = "/VMC/Ext/Blend/Apply"
HEAD      = "Head"

def _pose_msg(vals: List[float]) -> bytes:
    b = OscMessageBuilder(address=OSC_BONE)
    b.add_arg(HEAD)
    for v in vals:
        b.add_arg(float(v))
    return b.build().dgram

def map_exp(e: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.clip(e, 0, 1),
        nan=0.0, posinf=1.0, neginf=0.0
    )

def send_frame(sock: SimpleUDPClient, row: np.ndarray) -> None:
    yaw, pit, rol = row[64:67]
    tx, ty, tz = row[67:70]
    qx, qy, qz, qw = R.from_euler("yxz", [yaw, pit, rol]).as_quat()
    sock._sock.sendto(
        _pose_msg([tx, ty, tz, qx, qy, qz, qw]),
        (sock._address, sock._port)
    )
    for name, val in zip(EXP_NAMES, map_exp(row[:64])):
        sock.send_message(OSC_BVAL, [name, float(val)])
    sock.send_message(OSC_APPL, [])

# ---------------------------------------------------------------------------
# 4) Producer (multiprocessing)
# ---------------------------------------------------------------------------
Task = Tuple[bytes, np.ndarray]

def producer_proc(
    sents: List[str],
    img_path: Path,
    q: MPQueue,
    style_id: int,
    device: str,
    pose_style: int,
    no_blink: bool
) -> None:
    for i, sent in enumerate(sents, 1):
        try:
            logger.info("[Producer] %d/%d TTS ...", i, len(sents))
            wav = tts_generate(sent, style_id)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
                tf.write(wav)
                tf.flush()
                logger.info("   SadTalker ...")
                coeff = sadtalker_coeff(
                    img_path, Path(tf.name),
                    device, pose_style, no_blink
                )
            q.put((wav, coeff))
        except Exception as e:
            logger.error("Producer error: %s", e, exc_info=True)
    q.put((None, None))
    logger.info("[Producer] done.")

# ---------------------------------------------------------------------------
# 5) Consumer (thread)
# ---------------------------------------------------------------------------
def consumer_thread(q: MPQueue, sock: SimpleUDPClient) -> None:
    import pygame
    import pygame.mixer
    pygame.init()
    pygame.mixer.init()
    try:
        while True:
            wav, coeff = q.get()
            if wav is None:
                break
            if coeff is None or coeff.size == 0:
                logger.warning("Empty coeff received – skipping segment")
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tf.write(wav)
                wav_path = tf.name
            sound  = pygame.mixer.Sound(wav_path)
            length = sound.get_length()
            frames = len(coeff)
            if frames == 0:
                logger.warning("coeff length 0 – skipping OSC send")
                sound.play()
                time.sleep(length)
                os.unlink(wav_path)
                continue

            dt = length / frames
            ch = sound.play()
            start = time.perf_counter()
            idx = 0
            next_t = 0.0
            while ch.get_busy() and idx < frames:
                now = time.perf_counter() - start
                while idx < frames and now >= next_t:
                    send_frame(sock, coeff[idx])
                    idx += 1
                    next_t += dt
                time.sleep(0.001)
            # 余ったフレーム送信
            while idx < frames:
                send_frame(sock, coeff[idx])
                idx += 1
            os.unlink(wav_path)
            logger.info("[Consumer] segment done (%.2fs)", length)
    finally:
        pygame.mixer.quit()
        pygame.quit()
        logger.info("[Consumer] finished.")

# ---------------------------------------------------------------------------
# 6) main
# ---------------------------------------------------------------------------
def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--text", required=True)
    pa.add_argument("--style_id", type=int, default=DEFAULT_STYLE_ID)
    pa.add_argument("--image", default=str(DEFAULT_IMAGE_PATH))
    pa.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    pa.add_argument("--pose_style", type=int, default=0)
    pa.add_argument(
        "--no_blink",
        action="store_true",
        help="瞬きを完全に無効化する（ただしベースラインは維持）"
    )
    pa.add_argument("--osc_host", default="127.0.0.1")
    pa.add_argument("--osc_port", type=int, default=39540)
    args = pa.parse_args()

    img_path = Path(args.image).expanduser().resolve()
    if not img_path.exists():
        sys.exit("❌ image not found")

    sentences = split_sentences(args.text)
    if not sentences:
        sys.exit("❌ empty text")

    q: MPQueue[Task] = MPQueue(maxsize=10)
    sock = SimpleUDPClient(args.osc_host, args.osc_port)

    prod = Process(
        target=producer_proc,
        args=(
            sentences, img_path, q,
            args.style_id, args.device,
            args.pose_style, args.no_blink
        ),
        daemon=True
    )
    cons = threading.Thread(
        target=consumer_thread,
        args=(q, sock),
        daemon=True
    )

    cons.start()
    prod.start()
    try:
        prod.join()
        cons.join()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    logger.info("✅ All done.")

if __name__ == "__main__":
    main()