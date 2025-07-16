#!/usr/bin/env python3
"""
play_coeff_osc.py –– Stream SadTalker 70‑D coeffs (.mat) as OSC for VMC.

改訂 2025‑07‑17
────────────────────────────────────────────────────────
* 既定ポートを **39540**。
* `--fps` を残しつつ、`--audio` を渡すと **音声長 / フレーム数** から
  自動でフレーム間隔 (dt) を算出。
  – これで 75 f しか無い .mat でも 3 秒音声と同期。
* 各フレーム送信が終わった **直後に** ループ実測時間 (ms) を表示。
* ARKit 52 ブレンドシェイプへの符号・スケール変換を最小例で維持。
"""
from __future__ import annotations
import argparse, signal, sys, time, pathlib
from typing import List, Tuple
import numpy as np
import scipy.io as scio
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_message_builder import OscMessageBuilder
from scipy.spatial.transform import Rotation as R
import warnings

try:
    import librosa  # オーディオ長算出用 (optional)
except ImportError:
    librosa = None

# ---------------------------------------------------------------------------
# ARKit 52 キー + 余剰 12 枠
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
EXP_NAMES: List[str] = AR52 + [f"exp{idx:02d}" for idx in range(52, 64)]
assert len(EXP_NAMES) == 64

OSC_BONE = "/VMC/Ext/Bone/Pos"
OSC_BVAL = "/VMC/Ext/Blend/Val"
OSC_APPL = "/VMC/Ext/Blend/Apply"
HEAD     = "Head"

# --------------------------- 簡易マッピング例 -------------------------------
# index → (dst idx/list, sign, scale)
MAP: dict[int, Tuple[List[int], float, float]] = {
    25: ([24], +1, 1.2),      # jawOpen → jawOpen
    24: ([23, 22], +1, 1.0),  # jawLeft(+), jawRight(-) を分離
    8:  ([8],  +1, 1.0),      # eyeBlink_L
    9:  ([9],  +1, 1.0),
}

# ---------------------------------------------------------------------------
# Expression 正規化
# ---------------------------------------------------------------------------

def map_exp_to_arkit(exp: np.ndarray) -> np.ndarray:
    out = np.zeros(64, np.float32)
    for src, (dsts, sgn, scl) in MAP.items():
        v = sgn * exp[src] / scl
        if len(dsts) == 1:
            out[dsts[0]] = max(0.0, v)
        else:
            out[dsts[0]] = max(0.0,  v)
            out[dsts[1]] = max(0.0, -v)
    # 残りは ReLU で直接
    for i in range(64):
        if out[i] == 0:
            out[i] = max(0.0, exp[i])
    return np.clip(out, 0.0, 1.0)

# ---------------------------------------------------------------------------
# OSC helpers
# ---------------------------------------------------------------------------

def _build_pose_msg(pose_vals: List[float]):
    b = OscMessageBuilder(address=OSC_BONE)
    b.add_arg(HEAD)
    for v in pose_vals:
        b.add_arg(float(v))
    return b.build().dgram


def send_frame(sock: SimpleUDPClient, row: np.ndarray):
    yaw, pitch, roll = row[64:67]
    tx, ty, tz       = row[67:70]
    qx,qy,qz,qw      = R.from_euler("yxz", [yaw, pitch, roll]).as_quat()
    sock._sock.sendto(_build_pose_msg([tx,ty,tz,qx,qy,qz,qw]), (sock._address, sock._port))

    bs = map_exp_to_arkit(row[:64])
    for name, v in zip(EXP_NAMES, bs):
        sock.send_message(OSC_BVAL, [name, float(v)])
    sock.send_message(OSC_APPL, [])

# ---------------------------------------------------------------------------
# Streaming loop
# ---------------------------------------------------------------------------

def stream(coeff: np.ndarray, sock: SimpleUDPClient, dt: float, loop: bool):
    frames = len(coeff)
    idx = 0
    next_t = time.perf_counter()
    tic = time.perf_counter()
    while True:
        send_frame(sock, coeff[idx])

        idx += 1
        if idx >= frames:
            if not loop:
                break
            idx = 0
            toc = time.perf_counter()
            print(f"frame {idx or frames}/{frames}  {(toc - tic)*1000:.2f} ms")
            tic = toc
        # 時間調整
        next_t += dt
        time.sleep(max(0, next_t - time.perf_counter()))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True)
    p.add_argument("--audio", help="元音声 wav を指定すると自動 dt 計算 (librosa 必須)")
    p.add_argument("--fps", type=float, default=25.0, help="fps 指定 (audio が無い場合)")
    p.add_argument("--osc-host", default="127.0.0.1")
    p.add_argument("--osc-port", type=int, default=39540)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()

    coeff = scio.loadmat(a.mat)["coeff_3dmm"].astype(np.float32)
    if coeff.shape[1] != 70:
        sys.exit("❌ coeff_3dmm must be (T,70)")

    # dt 計算
    # --- dt (frame interval) -----------------------------------------
    dt = 1.0 / a.fps  # SadTalker は generate_batch 内部で fps=25 を想定
    if abs(dt*len(coeff) - 1.0) < 0.1:
        warnings.warn("⚠ fps とフレーム数の整合性に注意 (duration≈1s)")

    sock = SimpleUDPClient(a.osc_host, a.osc_port)
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    stream(coeff, sock, dt=dt, loop=not a.once)

if __name__ == "__main__":
    main()
