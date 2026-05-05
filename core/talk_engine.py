"""Talk engine core services for aituber-talk.

This module refactors the legacy `run.py` runtime into a reusable
`TalkEngine` class so controller plugins can share a single speech &
motion pipeline.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import sys
import tempfile
import threading
import time
import wave
from io import BytesIO
from dataclasses import dataclass, field
from multiprocessing import Queue as MPQueue
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
import scipy.io as scio
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.udp_client import SimpleUDPClient
from scipy.spatial.transform import Rotation as R

# =============================================================================
# Global constants
# =============================================================================
BLINK_BASELINE = 0.1
DEFAULT_STYLE_ID = 888753760
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "ai-sample-face.jpg"
SAD_ROOT = PROJECT_ROOT / "SadTalker"

if str(SAD_ROOT) not in sys.path:
    sys.path.append(str(SAD_ROOT))

from src import generate_batch as _gb  # type: ignore  # noqa: E402
from src.generate_batch import get_data  # type: ignore  # noqa: E402
from src.test_audio2coeff import Audio2Coeff  # type: ignore  # noqa: E402
from src.utils.init_path import init_path  # type: ignore  # noqa: E402

# =============================================================================
# Logging setup shared across processes
# =============================================================================
from logging.handlers import QueueHandler, QueueListener
import atexit

_log_queue: MPQueue = MPQueue()
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(
    logging.Formatter(
        fmt="[%(processName)s %(levelname)s] %(asctime)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
for handler in _root_logger.handlers[:]:
    _root_logger.removeHandler(handler)
_root_logger.addHandler(_console_handler)
_listener = QueueListener(_log_queue, _console_handler)
if not _listener._thread:  # Listener may already run when re-imported
    _listener.start()

def _shutdown_log_listener() -> None:
    _listener.stop()

atexit.register(_shutdown_log_listener)


logger = logging.getLogger("talk_engine")

# =============================================================================
# SadTalker compatibility helpers
# =============================================================================
_orig_blink = _gb.generate_blink_seq_randomly


def _safe_blink(frame_count: int) -> List[float]:
    if frame_count < 10:
        return [0.0] * frame_count
    values = np.asarray(_orig_blink(frame_count), dtype=np.float32).reshape(-1)
    if values.size < frame_count:
        values = np.pad(values, (0, frame_count - values.size), mode="constant")
    return [float(value) for value in values[:frame_count]]


_gb.generate_blink_seq_randomly = _safe_blink

try:
    import mat73  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    mat73 = None

_orig_loadmat = scio.loadmat


def _safe_loadmat(path, *args, **kwargs):
    try:
        if isinstance(path, (str, bytes)) and str(path).endswith(".mat"):
            if mat73 is not None:
                try:
                    return mat73.loadmat(path)
                except Exception:  # pragma: no cover
                    pass
        return _orig_loadmat(path, *args, **kwargs)
    except Exception:
        logger.warning("loadmat failed; returning zero coeff")
        return {"coeff_3dmm": np.zeros((1, 70), np.float32)}


scio.loadmat = _safe_loadmat

# =============================================================================
# Text utilities
# =============================================================================
SENT_RE = re.compile(r"(?<=[.!?。！？])\s*")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]


# =============================================================================
# External service helpers
# =============================================================================

def tts_generate(text: str, style: int, host: str, port: int) -> bytes:
    query = requests.post(
        f"http://{host}:{port}/audio_query",
        params={"speaker": style, "text": text},
        timeout=30,
    )
    query.raise_for_status()
    synthesis = requests.post(
        f"http://{host}:{port}/synthesis",
        params={"speaker": style},
        data=query.content,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    synthesis.raise_for_status()
    return synthesis.content


def _wav_duration_seconds(wav: bytes) -> float:
    try:
        with wave.open(BytesIO(wav), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate if rate else 0.0
    except Exception:
        return 0.0


def adjust_blink(coeff: np.ndarray, no_blink: bool = False) -> np.ndarray:
    if coeff.size == 0:
        return coeff
    frames = coeff.shape[0]
    baseline = BLINK_BASELINE
    if no_blink:
        seq = [baseline] * frames
    else:
        raw_seq = _safe_blink(frames)
        seq = [baseline + r * (1.0 - baseline) for r in raw_seq]
    coeff = coeff.copy()
    coeff[:, 8] = seq
    coeff[:, 9] = seq
    return coeff


def sadtalker_coeff(
    img: Path,
    wav_path: Path,
    device: str,
    pose_style: int,
    no_blink: bool,
) -> np.ndarray:
    ckpt_dir = SAD_ROOT / "checkpoints"
    cfg_dir = SAD_ROOT / "src" / "config"
    paths = init_path(ckpt_dir, cfg_dir, 256, "crop", device)
    st = ckpt_dir / "SadTalker_V0.0.2_256.safetensors"
    if st.exists():
        paths.update({"use_safetensor": True, "checkpoint": str(st)})

    a2c = Audio2Coeff(paths, device=device)
    batch = get_data(str(img), str(wav_path), ref_eyeblink_coeff_path=None, device=device)

    try:
        coeff = a2c.generate(
            batch,
            coeff_save_dir=None,
            pose_style=pose_style,
            return_path=False,
        ).astype(np.float32)
    except TypeError:  # Legacy SadTalker fallback
        with tempfile.TemporaryDirectory() as tmp_dir:
            mat_path = a2c.generate(
                batch,
                coeff_save_dir=tmp_dir,
                pose_style=pose_style,
            )
            coeff = scio.loadmat(mat_path)["coeff_3dmm"].astype(np.float32)

    if coeff.ndim == 1:
        coeff = coeff.reshape(1, -1)
    coeff = adjust_blink(coeff, no_blink=no_blink)
    if coeff.shape[1] < 70:
        raise ValueError(f"coeff shape invalid: {coeff.shape}")
    return coeff


AR52 = [
    "browDown_L",
    "browDown_R",
    "browInnerUp",
    "browOuterUp_L",
    "browOuterUp_R",
    "cheekPuff",
    "cheekSquint_L",
    "cheekSquint_R",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDown_L",
    "eyeLookDown_R",
    "eyeLookIn_L",
    "eyeLookIn_R",
    "eyeLookOut_L",
    "eyeLookOut_R",
    "eyeLookUp_L",
    "eyeLookUp_R",
    "eyeSquint_L",
    "eyeSquint_R",
    "eyeWide_L",
    "eyeWide_R",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimple_L",
    "mouthDimple_R",
    "mouthFrown_L",
    "mouthFrown_R",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDown_L",
    "mouthLowerDown_R",
    "mouthPress_L",
    "mouthPress_R",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmile_L",
    "mouthSmile_R",
    "mouthStretch_L",
    "mouthStretch_R",
    "mouthUpperUp_L",
    "mouthUpperUp_R",
    "noseSneer_L",
    "noseSneer_R",
    "tongueOut",
]
EXP_NAMES = AR52 + [f"exp{idx:02d}" for idx in range(52, 64)]
JAW_OPEN_INDEX = EXP_NAMES.index("jawOpen")
DEFAULT_JAW_OPEN_SOURCE_INDEX = 35
DEFAULT_JAW_OPEN_ZERO_VALUE = 0.0
DEFAULT_JAW_OPEN_ONE_VALUE = 0.15
OSC_BONE = "/VMC/Ext/Bone/Pos"
OSC_BVAL = "/VMC/Ext/Blend/Val"
OSC_APPL = "/VMC/Ext/Blend/Apply"
HEAD = "Head"


def _pose_msg(vals: Iterable[float]) -> bytes:
    msg = OscMessageBuilder(address=OSC_BONE)
    msg.add_arg(HEAD)
    for val in vals:
        msg.add_arg(float(val))
    return msg.build().dgram


def map_exp(
    values: np.ndarray,
    *,
    jaw_open_scale: float = 1.0,
    jaw_open_source_index: int = DEFAULT_JAW_OPEN_SOURCE_INDEX,
    jaw_open_zero_value: float = DEFAULT_JAW_OPEN_ZERO_VALUE,
    jaw_open_one_value: float = DEFAULT_JAW_OPEN_ONE_VALUE,
) -> np.ndarray:
    source = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    mapped = np.zeros(64, np.float32)
    limit = min(source.shape[0], mapped.shape[0])
    for idx in range(limit):
        mapped[idx] = max(0.0, float(source[idx]))
    if 0 <= jaw_open_source_index < source.shape[0]:
        src = float(source[jaw_open_source_index])
        denom = jaw_open_one_value - jaw_open_zero_value
        if denom <= 1e-6:
            mapped[JAW_OPEN_INDEX] = 0.0
        else:
            mapped[JAW_OPEN_INDEX] = (src - jaw_open_zero_value) / denom
    if mapped.shape[0] > JAW_OPEN_INDEX:
        mapped[JAW_OPEN_INDEX] = mapped[JAW_OPEN_INDEX] * jaw_open_scale
    return np.clip(mapped, 0.0, 1.0)


def map_jaw_open(
    values: np.ndarray,
    *,
    jaw_open_scale: float = 1.0,
    jaw_open_source_index: int = DEFAULT_JAW_OPEN_SOURCE_INDEX,
    jaw_open_zero_value: float = DEFAULT_JAW_OPEN_ZERO_VALUE,
    jaw_open_one_value: float = DEFAULT_JAW_OPEN_ONE_VALUE,
) -> Tuple[Optional[float], float]:
    source = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    if not 0 <= jaw_open_source_index < source.shape[0]:
        return None, 0.0
    src = float(source[jaw_open_source_index])
    denom = jaw_open_one_value - jaw_open_zero_value
    if denom <= 1e-6:
        return src, 0.0
    value = ((src - jaw_open_zero_value) / denom) * jaw_open_scale
    return src, float(np.clip(value, 0.0, 1.0))


def send_frame(
    sock: SimpleUDPClient,
    row: np.ndarray,
    *,
    jaw_open_scale: float = 1.0,
    jaw_open_source_index: int = DEFAULT_JAW_OPEN_SOURCE_INDEX,
    jaw_open_zero_value: float = DEFAULT_JAW_OPEN_ZERO_VALUE,
    jaw_open_one_value: float = DEFAULT_JAW_OPEN_ONE_VALUE,
) -> Tuple[Optional[float], float]:
    yaw, pitch, roll = row[64:67]
    tx, ty, tz = row[67:70]
    qx, qy, qz, qw = R.from_euler("yxz", [yaw, pitch, roll]).as_quat()
    sock._sock.sendto(_pose_msg([tx, ty, tz, qx, qy, qz, qw]), (sock._address, sock._port))
    mapped = map_exp(
        row[:64],
        jaw_open_scale=jaw_open_scale,
        jaw_open_source_index=jaw_open_source_index,
        jaw_open_zero_value=jaw_open_zero_value,
        jaw_open_one_value=jaw_open_one_value,
    )
    for name, val in zip(EXP_NAMES, mapped):
        sock.send_message(OSC_BVAL, [name, float(val)])
    sock.send_message(OSC_APPL, [])
    return map_jaw_open(
        row[:64],
        jaw_open_scale=jaw_open_scale,
        jaw_open_source_index=jaw_open_source_index,
        jaw_open_zero_value=jaw_open_zero_value,
        jaw_open_one_value=jaw_open_one_value,
    )


# =============================================================================
# Configuration & runtime classes
# =============================================================================

@dataclass
class BufferedSpeechSegment:
    request_id: str
    text: str
    wav: bytes
    coeff: np.ndarray
    duration: float


@dataclass
class TalkEngineConfig:
    image_path: Path = DEFAULT_IMAGE_PATH
    default_style_id: int = DEFAULT_STYLE_ID
    device: str = "cpu"
    pose_style: int = 0
    no_blink: bool = False
    osc_host: str = "127.0.0.1"
    osc_port: int = 39540
    jaw_open_scale: float = 1.0
    jaw_open_source_index: int = DEFAULT_JAW_OPEN_SOURCE_INDEX
    jaw_open_zero_value: float = DEFAULT_JAW_OPEN_ZERO_VALUE
    jaw_open_one_value: float = DEFAULT_JAW_OPEN_ONE_VALUE
    speaker_host: str = "127.0.0.1"
    speaker_port: int = 10101
    motion_queue_size: int = 10
    idle_queue_size: int = 2


class TalkEngine:
    """Threaded orchestrator for speech synthesis and motion playback."""

    def __init__(self, config: TalkEngineConfig | None = None) -> None:
        self.config = config or TalkEngineConfig()
        self._style_id = self.config.default_style_id
        self._image_path = self.config.image_path
        self._text_queue: "queue.Queue[Tuple[str, Optional[int], Optional[str]]]" = queue.Queue()
        self._motion_queue: Optional[MPQueue] = None
        self._idle_queue: Optional["queue.Queue[Tuple[np.ndarray, float]]"] = None
        self._sock: Optional[SimpleUDPClient] = None
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._idle_enabled = threading.Event()
        self._current_session_id: Optional[str] = None
        self._state_lock = threading.Lock()
        self._speech_timing_lock = threading.Lock()
        self._current_speech_ends_at: Optional[float] = None
        self._queued_speech_seconds: float = 0.0
        self._jaw_open_lock = threading.Lock()
        self._last_jaw_open_source: Optional[float] = None
        self._last_jaw_open_value: Optional[float] = None
        self._threads: Dict[str, threading.Thread] = {}
        self._playback_events: Dict[str, threading.Event] = {}
        self._playback_counts: Dict[str, int] = {}
        self._playback_lock = threading.Lock()
        self._stopped_sessions: set[str] = set()
        self._started = False
        self._logger = logging.getLogger("talk_engine.runtime")
        self._logger.addHandler(QueueHandler(_log_queue))
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        image_path = self.config.image_path.expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        self._image_path = image_path
        self._motion_queue = MPQueue(maxsize=self.config.motion_queue_size)
        self._idle_queue = queue.Queue(maxsize=self.config.idle_queue_size)
        self._sock = SimpleUDPClient(self.config.osc_host, self.config.osc_port)
        self._stop_event.clear()
        self._pause_event.clear()
        self._idle_enabled.set()

        self._threads = {
            "producer": threading.Thread(target=self._text_producer_loop, name="TalkProducer", daemon=True),
            "idle": threading.Thread(target=self._idle_generator_loop, name="TalkIdleGen", daemon=True),
            "consumer": threading.Thread(target=self._consumer_loop, name="TalkConsumer", daemon=True),
        }
        for thread in self._threads.values():
            thread.start()
        self._started = True
        self._logger.info("TalkEngine started (device=%s, pose_style=%s)", self.config.device, self.config.pose_style)

    def shutdown(self, wait: bool = False) -> None:
        if not self._started:
            return
        self._stop_event.set()
        self._text_queue.put(("", None, None))
        if self._idle_queue is not None:
            try:
                self._idle_queue.put_nowait((np.zeros((1, 70), np.float32), 0.0))
            except queue.Full:
                pass
        if wait:
            for thread in self._threads.values():
                thread.join(timeout=5)
        self._started = False
        self._logger.info("TalkEngine shutdown complete")

    # ------------------------------------------------------------------
    # Public API used by controller plugins
    # ------------------------------------------------------------------
    def _generate_speech_segment(
        self,
        sentence: str,
        *,
        style_id: int,
        request_id: str,
        on_progress: Optional[Callable[[str, str], None]] = None,
    ) -> BufferedSpeechSegment:
        if on_progress:
            on_progress("tts_started", sentence)
        wav = tts_generate(
            sentence,
            style=style_id,
            host=self.config.speaker_host,
            port=self.config.speaker_port,
        )
        if on_progress:
            on_progress("tts_completed", sentence)
        duration = _wav_duration_seconds(wav)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp.write(wav)
            tmp.flush()
            if on_progress:
                on_progress("sadtalker_started", sentence)
            coeff = sadtalker_coeff(
                self._image_path,
                Path(tmp.name),
                self.config.device,
                self.config.pose_style,
                self.config.no_blink,
            )
            if on_progress:
                on_progress("sadtalker_completed", sentence)
        finally:
            tmp.close()
            os.unlink(tmp.name)
        return BufferedSpeechSegment(
            request_id=request_id,
            text=sentence,
            wav=wav,
            coeff=coeff,
            duration=duration,
        )

    def submit_text(self, text: str, *, style_id: Optional[int] = None, request_id: Optional[str] = None) -> str:
        if not self._started:
            self.start()
        req_id = request_id or os.urandom(8).hex()
        with self._playback_lock:
            if req_id not in self._playback_events:
                self._playback_events[req_id] = threading.Event()
                self._playback_counts[req_id] = 0
        self._text_queue.put((text, style_id, req_id))
        return req_id

    def prepare_text_segments(
        self,
        text: str,
        *,
        style_id: Optional[int] = None,
        request_id: Optional[str] = None,
        on_segment: Optional[Callable[[BufferedSpeechSegment], None]] = None,
        on_progress: Optional[Callable[[int, str, str], None]] = None,
    ) -> List[BufferedSpeechSegment]:
        """Generate speech audio and SadTalker coefficients without starting playback."""
        if not self._started:
            self.start()
        current_style = self._style_id if style_id is None else style_id
        req_id = request_id or os.urandom(8).hex()
        prepared: List[BufferedSpeechSegment] = []
        for idx, sentence in enumerate(split_sentences(text), 1):
            segment = self._generate_speech_segment(
                sentence,
                style_id=current_style,
                request_id=f"{req_id}-{idx}",
                on_progress=(lambda stage, text, index=idx: on_progress(index, stage, text)) if on_progress else None,
            )
            prepared.append(segment)
            if on_segment:
                on_segment(segment)
        return prepared

    def prepare_text(self, text: str, *, style_id: Optional[int] = None) -> List[Tuple[bytes, np.ndarray]]:
        return [(segment.wav, segment.coeff) for segment in self.prepare_text_segments(text, style_id=style_id)]

    def play_segments(
        self,
        segments: List[BufferedSpeechSegment],
        *,
        request_id: Optional[str] = None,
    ) -> str:
        if not self._started:
            self.start()
        req_id = request_id or os.urandom(8).hex()
        with self._playback_lock:
            if req_id not in self._playback_events:
                self._playback_events[req_id] = threading.Event()
            self._playback_counts[req_id] = len(segments)
            if not segments:
                self._playback_events[req_id].set()
        if self._motion_queue is not None:
            for segment in segments:
                self._motion_queue.put((segment.wav, segment.coeff, req_id))
        return req_id

    def submit_prepared(
        self,
        items: List[Tuple[bytes, np.ndarray]],
        *,
        request_id: Optional[str] = None,
    ) -> str:
        segments = [
            BufferedSpeechSegment(
                request_id=f"{request_id or 'prepared'}-{idx}",
                text="",
                wav=wav,
                coeff=coeff,
                duration=0.0,
            )
            for idx, (wav, coeff) in enumerate(items, 1)
        ]
        return self.play_segments(segments, request_id=request_id)

    def wait_for(self, request_id: str, timeout: float = 300.0) -> str:
        with self._playback_lock:
            event = self._playback_events.get(request_id)
        if event is None:
            return "unknown"
        finished = event.wait(timeout=timeout)
        if not finished:
            return "timeout"
        with self._playback_lock:
            self._playback_events.pop(request_id, None)
            self._playback_counts.pop(request_id, None)
        if request_id in self._stopped_sessions:
            self._stopped_sessions.discard(request_id)
            return "stopped"
        return "completed"

    def pause(self) -> Optional[str]:
        with self._state_lock:
            if not self._current_session_id:
                return None
            self._pause_event.set()
            return self._current_session_id

    def resume(self) -> Optional[str]:
        if not self._pause_event.is_set():
            return None
        self._pause_event.clear()
        with self._state_lock:
            return self._current_session_id

    def stop(self) -> Optional[str]:
        with self._state_lock:
            session_id = self._current_session_id
        if session_id:
            self._stopped_sessions.add(session_id)
            with self._playback_lock:
                event = self._playback_events.get(session_id)
                if event:
                    event.set()
                self._playback_counts.pop(session_id, None)
        self._pause_event.clear()
        if self._motion_queue is not None:
            while not self._motion_queue.empty():
                try:
                    self._motion_queue.get_nowait()
                except queue.Empty:
                    break
        if self._idle_queue is not None:
            while not self._idle_queue.empty():
                try:
                    self._idle_queue.get_nowait()
                except queue.Empty:
                    break
        return session_id

    def status(self) -> Dict[str, Any]:
        with self._state_lock:
            session = self._current_session_id
        with self._speech_timing_lock:
            now = time.monotonic()
            current_remaining = max(0.0, (self._current_speech_ends_at or 0.0) - now)
            queued_speech_seconds = max(0.0, self._queued_speech_seconds)
            speech_available_in = current_remaining + queued_speech_seconds
        with self._jaw_open_lock:
            jaw_open_source = self._last_jaw_open_source
            jaw_open_value = self._last_jaw_open_value
        queued: Optional[int]
        if self._motion_queue is None:
            queued = 0
        else:
            try:
                queued = self._motion_queue.qsize()
            except (NotImplementedError, OSError):
                queued = None
        return {
            "session": session,
            "paused": self._pause_event.is_set(),
            "queued": queued,
            "current_speech_remaining_sec": current_remaining,
            "queued_speech_seconds": queued_speech_seconds,
            "speech_available_in_sec": speech_available_in,
            "jaw_open_scale": self.config.jaw_open_scale,
            "jaw_open_source_index": self.config.jaw_open_source_index,
            "jaw_open_zero_value": self.config.jaw_open_zero_value,
            "jaw_open_one_value": self.config.jaw_open_one_value,
            "jaw_open_current_source": jaw_open_source,
            "jaw_open_current_value": jaw_open_value,
        }

    def set_style(self, style_id: int) -> None:
        self._style_id = style_id

    def set_idle(self, enabled: bool) -> None:
        """Enable or disable the idle motion generator."""
        if enabled:
            self._idle_enabled.set()
        else:
            self._idle_enabled.clear()
            # Also clear any pending idle motions from the queue
            if self._idle_queue is not None:
                while not self._idle_queue.empty():
                    try:
                        self._idle_queue.get_nowait()
                    except queue.Empty:
                        break

    def set_jaw_open_scale(self, scale: float) -> None:
        self.config.jaw_open_scale = max(float(scale), 0.0)

    def set_jaw_open_source_index(self, index: int) -> None:
        if not 0 <= int(index) < 64:
            raise ValueError("jaw_open_source_index must be between 0 and 63")
        self.config.jaw_open_source_index = int(index)

    def set_jaw_open_range(self, zero_value: float, one_value: float) -> None:
        if float(one_value) <= float(zero_value):
            raise ValueError("jaw_open_one_value must be greater than jaw_open_zero_value")
        self.config.jaw_open_zero_value = float(zero_value)
        self.config.jaw_open_one_value = float(one_value)

    def _record_jaw_open(self, source: Optional[float], value: float) -> None:
        with self._jaw_open_lock:
            self._last_jaw_open_source = source
            self._last_jaw_open_value = value

    # ------------------------------------------------------------------
    # Worker loops
    # ------------------------------------------------------------------
    def _text_producer_loop(self) -> None:
        thread_logger = logging.getLogger("talk_engine.text_producer")
        thread_logger.addHandler(QueueHandler(_log_queue))
        thread_logger.setLevel(logging.INFO)
        thread_logger.propagate = False
        current_style = self._style_id
        while not self._stop_event.is_set():
            try:
                text, style_override, req_id = self._text_queue.get()
                if self._stop_event.is_set():
                    break
                if style_override is not None:
                    thread_logger.info("style override: %s -> %s", current_style, style_override)
                    current_style = style_override
                sentences = split_sentences(text)
                if req_id is not None:
                    with self._playback_lock:
                        if sentences:
                            self._playback_counts[req_id] = self._playback_counts.get(req_id, 0) + len(sentences)
                        else:
                            event = self._playback_events.get(req_id)
                            if event:
                                event.set()
                for sentence in sentences:
                    wav = tts_generate(
                        sentence,
                        style=current_style,
                        host=self.config.speaker_host,
                        port=self.config.speaker_port,
                    )
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    try:
                        tmp.write(wav)
                        tmp.flush()
                        coeff = sadtalker_coeff(
                            self._image_path,
                            Path(tmp.name),
                            self.config.device,
                            self.config.pose_style,
                            self.config.no_blink,
                        )
                    finally:
                        tmp.close()
                        os.unlink(tmp.name)
                    if self._motion_queue is not None:
                        duration = _wav_duration_seconds(wav)
                        with self._speech_timing_lock:
                            self._queued_speech_seconds += duration
                        self._motion_queue.put((wav, coeff, req_id))
            except Exception as exc:  # pragma: no cover - defensive
                thread_logger.error("text producer failure: %s", exc, exc_info=True)

    def _idle_generator_loop(self) -> None:
        if self._idle_queue is None:
            return
        while not self._stop_event.is_set():
            self._idle_enabled.wait()
            if self._motion_queue is not None and not self._motion_queue.empty():
                time.sleep(0.1)
                continue
            duration = 3.0
            sample_rate = 16000
            n_samples = int(duration * sample_rate)
            tmp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            with wave.open(tmp_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(b"\x00\x00" * n_samples)
            coeff_idle = sadtalker_coeff(
                self._image_path,
                Path(tmp_path),
                self.config.device,
                self.config.pose_style,
                self.config.no_blink,
            )
            os.unlink(tmp_path)
            if coeff_idle.ndim == 1:
                coeff_idle = coeff_idle.reshape(1, -1)
            try:
                self._idle_queue.put((coeff_idle, duration), timeout=1)
            except queue.Full:
                time.sleep(0.1)

    def _consumer_loop(self) -> None:
        import pygame
        import pygame.mixer

        if self._sock is None or self._motion_queue is None or self._idle_queue is None:
            raise RuntimeError("TalkEngine not initialised correctly")

        pygame.init()
        pygame.mixer.init()
        try:
            while not self._stop_event.is_set():
                try:
                    wav, coeff, req_id = self._motion_queue.get_nowait()
                    is_speech = True
                except queue.Empty:
                    is_speech = False
                except ValueError:
                    wav, coeff = self._motion_queue.get_nowait()
                    req_id = None
                    is_speech = True

                if not is_speech:
                    try:
                        coeff_idle, duration = self._idle_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    # Before starting to play, check if idle has been disabled
                    if not self._idle_enabled.is_set():
                        continue

                    frames = coeff_idle.shape[0]
                    dt = duration / frames if frames else 0.033
                    for idx in range(frames):
                        # Check on each frame if idle has been disabled
                        if not self._idle_enabled.is_set():
                            break
                        if not self._motion_queue.empty():
                            break
                        while self._pause_event.is_set() and not self._stop_event.is_set():
                            time.sleep(0.1)
                        source, value = send_frame(
                            self._sock,
                            coeff_idle[idx],
                            jaw_open_scale=self.config.jaw_open_scale,
                            jaw_open_source_index=self.config.jaw_open_source_index,
                            jaw_open_zero_value=self.config.jaw_open_zero_value,
                            jaw_open_one_value=self.config.jaw_open_one_value,
                        )
                        self._record_jaw_open(source, value)
                        time.sleep(dt)
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(wav)
                    wav_path = Path(tmp.name)
                sound = pygame.mixer.Sound(str(wav_path))
                length = sound.get_length()
                with self._speech_timing_lock:
                    self._queued_speech_seconds = max(0.0, self._queued_speech_seconds - length)
                if coeff is None or coeff.size == 0:
                    coeff = sadtalker_coeff(
                        self._image_path,
                        wav_path,
                        self.config.device,
                        self.config.pose_style,
                        self.config.no_blink,
                    )
                if coeff.ndim == 1:
                    coeff = coeff.reshape(1, -1)
                frames = coeff.shape[0]
                dt = length / frames if frames else 0.033

                with self._state_lock:
                    self._current_session_id = req_id
                with self._speech_timing_lock:
                    self._current_speech_ends_at = time.monotonic() + length

                channel = sound.play()
                start = time.perf_counter()
                index = 0
                next_tick = 0.0
                try:
                    while channel.get_busy() and index < frames and not self._stop_event.is_set():
                        now = time.perf_counter() - start
                        while index < frames and now >= next_tick:
                            while self._pause_event.is_set() and not self._stop_event.is_set():
                                time.sleep(0.1)
                            source, value = send_frame(
                                self._sock,
                                coeff[index],
                                jaw_open_scale=self.config.jaw_open_scale,
                                jaw_open_source_index=self.config.jaw_open_source_index,
                                jaw_open_zero_value=self.config.jaw_open_zero_value,
                                jaw_open_one_value=self.config.jaw_open_one_value,
                            )
                            self._record_jaw_open(source, value)
                            index += 1
                            next_tick += dt
                        time.sleep(0.001)
                    while index < frames and not self._stop_event.is_set():
                        while self._pause_event.is_set() and not self._stop_event.is_set():
                            time.sleep(0.1)
                        source, value = send_frame(
                            self._sock,
                            coeff[index],
                            jaw_open_scale=self.config.jaw_open_scale,
                            jaw_open_source_index=self.config.jaw_open_source_index,
                            jaw_open_zero_value=self.config.jaw_open_zero_value,
                            jaw_open_one_value=self.config.jaw_open_one_value,
                        )
                        self._record_jaw_open(source, value)
                        index += 1
                finally:
                    os.unlink(str(wav_path))
                    with self._state_lock:
                        self._current_session_id = None
                    with self._speech_timing_lock:
                        self._current_speech_ends_at = None

                if req_id is not None:
                    with self._playback_lock:
                        remaining = self._playback_counts.get(req_id, 0) - 1
                        if remaining <= 0:
                            event = self._playback_events.get(req_id)
                            if event:
                                event.set()
                            self._playback_counts.pop(req_id, None)
                        else:
                            self._playback_counts[req_id] = remaining
        finally:
            pygame.mixer.quit()
            pygame.quit()


__all__ = ["BufferedSpeechSegment", "TalkEngine", "TalkEngineConfig", "split_sentences"]
