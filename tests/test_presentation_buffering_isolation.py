import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from controller.events import EventBus
from controller.session import SessionManager
from core.talk_engine import BufferedSpeechSegment, split_sentences
from plugins.present import PresentMode


class FakeEngine:
    class Config:
        default_style_id = 7

    config = Config()

    def __init__(self):
        self.submitted = []
        self.prepared = []
        self.played = []

    def submit_text(self, text, *, style_id=None, request_id=None):
        self.submitted.append((text, style_id, request_id))
        return request_id or "submitted"

    def prepare_text_segments(self, text, *, style_id=None, request_id=None, on_segment=None, on_progress=None):
        self.prepared.append((text, style_id, request_id))
        segments = []
        for idx, sentence in enumerate(split_sentences(text), 1):
            for stage in ("tts_started", "tts_completed", "sadtalker_started", "sadtalker_completed"):
                if on_progress:
                    on_progress(idx, stage, sentence)
            segment = BufferedSpeechSegment(
                request_id=f"{request_id}-{idx}",
                text=sentence,
                wav=b"RIFF",
                coeff=np.zeros((1, 70), np.float32),
                duration=1.0,
            )
            segments.append(segment)
            if on_segment:
                on_segment(segment)
        return segments

    def play_segments(self, segments, *, request_id=None):
        self.played.append((request_id, list(segments)))
        return request_id or "played"

    def wait_for(self, request_id, timeout=300.0):
        return "completed"

    def status(self):
        return {"paused": False, "queued": 0}

    def stop(self):
        return "stopped"

    def pause(self):
        return "paused"

    def resume(self):
        return "resumed"


class PresentationBufferingIsolationTest(unittest.TestCase):
    def make_mode(self):
        sessions = SessionManager()
        engine = FakeEngine()
        mode = PresentMode(engine, EventBus(), sessions)
        session = sessions.create("present", metadata={})
        return mode, engine, sessions, session

    def wait_for_status(self, sessions, session_id, status):
        deadline = time.time() + 3
        while sessions.get(session_id).status != status and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(sessions.get(session_id).status, status)

    def test_streaming_presentation_uses_submit_text_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "slides.md"
            script.write_text("# One\nHello.\n# Two\n![x](img.png)\nWorld.\n", encoding="utf-8")
            mode, engine, sessions, session = self.make_mode()

            result = mode.start(
                session,
                {"script_path": str(script), "all_buffered": False, "image_viewer": "off", "wait": 0},
            )
            self.assertEqual(result["status"], "running")
            self.wait_for_status(sessions, session.id, "completed")

            self.assertEqual(engine.prepared, [])
            self.assertEqual(engine.played, [])
            self.assertEqual([item[0] for item in engine.submitted], ["Hello.", "World."])

    def test_all_buffered_uses_prepare_then_play_segments_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "slides.md"
            script.write_text("# One\nHello. Again.\n# Two\nWorld.\n", encoding="utf-8")
            mode, engine, sessions, session = self.make_mode()

            result = mode.start(
                session,
                {"script_path": str(script), "all_buffered": True, "image_viewer": "off", "wait": 0},
            )
            self.assertEqual(result["status"], "buffering")
            self.wait_for_status(sessions, session.id, "ready")

            self.assertEqual(engine.submitted, [])
            self.assertEqual([item[0] for item in engine.prepared], ["Hello. Again.", "World."])

            play_result = mode.play(session)
            self.assertEqual(play_result["status"], "running")
            self.wait_for_status(sessions, session.id, "ready")

            self.assertEqual(engine.submitted, [])
            self.assertEqual(len(engine.played), 2)


if __name__ == "__main__":
    unittest.main()
