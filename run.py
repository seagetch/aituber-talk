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
import select
from pathlib import Path
from typing import Tuple, List
from multiprocessing import Queue as MPQueue

import logging
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue as MPQueue

# --- set up parent logging handler ---
_log_queue: MPQueue = MPQueue()
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(
    logging.Formatter(
        fmt="[%(processName)s %(levelname)s] %(asctime)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
)
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
# clear any existing handlers
for _h in _root_logger.handlers[:]:
    _root_logger.removeHandler(_h)
_root_logger.addHandler(_console_handler)
# start listener that will take log records from queue and handle them
_listener = QueueListener(_log_queue, _console_handler)
_listener.start()

from queue import Queue
import queue  # for Empty exception
from queue import Queue as ThreadQueue

# FastAPI for text input via API
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


app = FastAPI()
# グローバルテキストキュー
text_queue: Queue[str] = Queue()

class TextRequest(BaseModel):
    text: str

@app.post("/talk")
async def talk_text(request: TextRequest):
    text_queue.put(request.text)
    return {"status": "queued", "text": request.text}

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
    # child process logging via queue
    proc_logger = logging.getLogger("run.producer")
    proc_logger.setLevel(logging.INFO)
    proc_logger.handlers.clear()
    proc_logger.addHandler(QueueHandler(_log_queue))
    proc_logger.propagate = False
    proc_logger.info("[Producer] process started")
    for i, sent in enumerate(sents, 1):
        try:
            proc_logger.info("[Producer] %d/%d TTS '%s'...", i, len(sents), sent)
            wav = tts_generate(sent, style_id)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
                tf.write(wav)
                tf.flush()
                proc_logger.info("   SadTalker ...")
                coeff = sadtalker_coeff(
                    img_path, Path(tf.name),
                    device, pose_style, no_blink
                )
            q.put((wav, coeff))
        except Exception as e:
            proc_logger.error("Producer error: %s", e, exc_info=True)
    q.put((None, None))
    proc_logger.info("[Producer] done.")

# ---------------------------------------------------------------------------
# 5) Consumer (thread)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# idle-motion generator thread
# ---------------------------------------------------------------------------
def idle_generator_thread(
    idle_q: ThreadQueue,
    motion_q: MPQueue,
    img_path: Path,
    device: str,
    pose_style: int,
    no_blink: bool
) -> None:
    """先読みでアイドルモーションを生成してキューに詰める"""
    import tempfile
    import wave
    while True:
        # テキスト処理中は待機モーション生成を待機
        while not motion_q.empty():
            time.sleep(0.1)
        # 3秒の無音WAVを作成
        duration = 3.0
        sample_rate = 16000
        n_samples = int(duration * sample_rate)
        wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with wave.open(wav_file, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"\x00\x00" * n_samples)
        coeff_idle = sadtalker_coeff(img_path, Path(wav_file), device, pose_style, no_blink)
        os.unlink(wav_file)
        if coeff_idle.ndim == 1:
            coeff_idle = coeff_idle.reshape(1, -1)
        # 2つ分までキューに保持
        idle_q.put((coeff_idle, duration))


def consumer_thread(
    q: MPQueue,
    idle_q: ThreadQueue,
    sock: SimpleUDPClient,
    img_path: Path,
    device: str,
    pose_style: int,
    no_blink: bool
) -> None:
    import pygame, pygame.mixer
    pygame.init(); pygame.mixer.init()
    try:
        while True:
            # ノンブロッキングで音声モーションを取得
            try:
                wav, coeff = q.get_nowait()
                is_speech = True
            except queue.Empty:
                is_speech = False
            if not is_speech:
                # アイドルバッファから取得
                coeff_idle, duration = idle_q.get()
                frames = coeff_idle.shape[0]
                dt = duration / frames
                for idx in range(frames):
                    # フレーム送信前に音声モーションが来ていれば即時切替
                    if not q.empty():
                        break
                    send_frame(sock, coeff_idle[idx])
                    time.sleep(dt)
                continue

            # Speech segment processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tf.write(wav); wav_path = tf.name
            sound = pygame.mixer.Sound(wav_path)
            length = sound.get_length()
            if coeff is None or coeff.size == 0:
                logger.warning("Empty coeff received – regenerating via SadTalker model")
                coeff = sadtalker_coeff(img_path, Path(wav_path),
                                        device, pose_style, no_blink)
            if coeff.ndim == 1:
                coeff = coeff.reshape(1, -1)
            frames = coeff.shape[0]
            dt = length / frames
            ch = sound.play()
            start = time.perf_counter()
            idx = 0
            next_t = 0.0
            while ch.get_busy() and idx < frames:
                now = time.perf_counter() - start
                while idx < frames and now >= next_t:
                    send_frame(sock, coeff[idx]); idx += 1; next_t += dt
                time.sleep(0.001)
            while idx < frames:
                send_frame(sock, coeff[idx]); idx += 1
            os.unlink(wav_path)
            logger.info("[Consumer] segment done (%.2fs)", length)
    finally:
        pygame.mixer.quit(); pygame.quit()
        logger.info("[Consumer] finished.")

# ---------------------------------------------------------------------------
# 6) main
# ---------------------------------------------------------------------------

# デフォルト設定
DEFAULT_DEVICE = "cpu"
DEFAULT_POSE_STYLE = 0
DEFAULT_NO_BLINK = False




# ---------------------------------------------------------------------------
# text producer thread
# ---------------------------------------------------------------------------
def text_producer_thread(
    text_q: Queue,
    motion_q: MPQueue,
    img_path: Path,
    device: str,
    pose_style: int,
    no_blink: bool,
    style_id: int
) -> None:
    """標準入力テキストを受け取り、音声とモーションを生成して motion_q に渡す"""
    tp_logger = logging.getLogger("run.text_producer")
    tp_logger.setLevel(logging.INFO)
    tp_logger.handlers.clear()
    tp_logger.addHandler(QueueHandler(_log_queue))
    tp_logger.propagate = False
    while True:
        try:
            text = text_q.get()
            tp_logger.info("[TextProducer] received text: %r", text)
            sentences = split_sentences(text)
            for sent in sentences:
                tp_logger.info("[TextProducer] processing sentence: %r", sent)
                wav = tts_generate(sent, style_id)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
                    tf.write(wav)
                    tf.flush()
                    coeff = sadtalker_coeff(
                        img_path,
                        Path(tf.name),
                        device,
                        pose_style,
                        no_blink
                    )
                motion_q.put((wav, coeff))
        except Exception as e:
            tp_logger.error("[TextProducer] error: %s", e, exc_info=True)
            continue


# ---------------------------------------------------------------------------
# stdin-driven producer main loop
# ---------------------------------------------------------------------------
def main() -> None:
    # CLI引数パース
    pa = argparse.ArgumentParser()
    pa.add_argument("--style_id", type=int, default=DEFAULT_STYLE_ID)
    pa.add_argument("--image", default=str(DEFAULT_IMAGE_PATH))
    pa.add_argument("--device", choices=["cpu", "cuda"], default=DEFAULT_DEVICE)
    pa.add_argument("--pose_style", type=int, default=DEFAULT_POSE_STYLE)
    pa.add_argument("--no_blink", action="store_true",
                    help="瞬きを完全に無効化する（ただしベースラインは維持）")
    pa.add_argument("--osc_host", default="127.0.0.1")
    pa.add_argument("--osc_port", type=int, default=39540)
    args = pa.parse_args()

    # キューとOSCクライアントを作成
    q: MPQueue[Task] = MPQueue(maxsize=10)
    sock = SimpleUDPClient(args.osc_host, args.osc_port)
    img_path = Path(args.image).expanduser().resolve()
    if not img_path.exists():
        sys.exit("❌ image not found")
    device = args.device
    pose_style = args.pose_style
    no_blink = args.no_blink

    # グローバルテキストキューを使ってProducerスレッドを起動
    text_thr = threading.Thread(
        target=text_producer_thread,
        args=(text_queue, q, img_path, device, pose_style, no_blink, args.style_id),
        daemon=True
    )
    text_thr.start()

    # アイドルモーションバッファキュー
    idle_queue: ThreadQueue = ThreadQueue(maxsize=2)
    # アイドル生成スレッド開始
    idle_gen_thr = threading.Thread(
        target=idle_generator_thread,
        args=(idle_queue, q, img_path, device, pose_style, no_blink),
        daemon=True
    )
    idle_gen_thr.start()

    # コンシューマスレッド起動（アイドルキューを渡す）
    cons_thr = threading.Thread(
        target=consumer_thread,
        args=(q, idle_queue, sock, img_path, device, pose_style, no_blink),
        daemon=True
    )
    cons_thr.start()
    # 標準入力ループを廃止し、API経由でテキスト受付
    return


# --- Ensure child processes inherit stdout by using 'fork' start method ---
import multiprocessing
try:
    multiprocessing.set_start_method('fork')
except RuntimeError:
    # Start method may have been set already
    pass

if __name__ == "__main__":
    main()
    uvicorn.run(app, host="0.0.0.0", port=34512)