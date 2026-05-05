import json
import tempfile
import time
import unittest
from pathlib import Path

from controller.events import EventBus
from controller.session import SessionManager
from core.codex_watcher import CodexMessageEvent
from plugins.codex_watch import CodexWatchMode


class FakeEngine:
    class Config:
        default_style_id = 7

    config = Config()

    def __init__(self):
        self.submitted = []

    def submit_text(self, text, *, style_id=None, request_id=None):
        self.submitted.append((text, style_id, request_id))
        return request_id or "request"

    def status(self):
        return {"session": None, "queued": 0}


class CodexWatchModeTest(unittest.TestCase):
    def test_rewrite_true_submits_text(self):
        calls = []

        def rewriter(events, recent_codex, recent_utterances, profile, model):
            calls.append(([event.text for event in events], list(recent_codex), profile, model))
            return {"should_speak": True, "utterance": "実装の方針を確認しているよ。"}

        engine = FakeEngine()
        sessions = SessionManager()
        mode = CodexWatchMode(engine, EventBus(), sessions, rewriter=rewriter)
        session = sessions.create("codex_watch", metadata={})
        with tempfile.TemporaryDirectory() as tmp:
            rollout = Path(tmp) / "rollout-test.jsonl"
            rollout.write_text(
                json.dumps({"type": "session_meta", "payload": {"id": "s1", "cwd": "/tmp/project"}}) + "\n",
                encoding="utf-8",
            )
            result = mode.start(
                session,
                {
                    "rollout_path": str(rollout),
                    "model": "test-model",
                    "style_id": 3,
                    "batch_window_sec": 0.05,
                },
            )
            self.assertEqual(result["status"], "running")

            event = CodexMessageEvent("s1", rollout, None, "event_msg.agent_message", "assistant", "実装を始めます。", "/tmp/project", {})
            mode._enqueue_message(session.id, event)
            deadline = time.time() + 2
            while not engine.submitted and time.time() < deadline:
                time.sleep(0.01)

            mode.stop(session)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], ["実装を始めます。"])
        self.assertEqual(engine.submitted[0][0], "実装の方針を確認しているよ。")
        self.assertEqual(engine.submitted[0][1], 3)

    def test_completed_events_are_batched_before_rewrite(self):
        calls = []

        def rewriter(events, recent_codex, recent_utterances, profile, model):
            calls.append([event.text for event in events])
            return {"should_speak": True, "utterance": "まとめて進捗を伝えるね。"}

        engine = FakeEngine()
        sessions = SessionManager()
        mode = CodexWatchMode(engine, EventBus(), sessions, rewriter=rewriter)
        session = sessions.create("codex_watch", metadata={})
        with tempfile.TemporaryDirectory() as tmp:
            rollout = Path(tmp) / "rollout-test.jsonl"
            rollout.write_text("{}", encoding="utf-8")
            mode.start(
                session,
                {
                    "rollout_path": str(rollout),
                    "batch_window_sec": 0.1,
                    "max_batch_events": 4,
                },
            )
            mode._enqueue_message(
                session.id,
                CodexMessageEvent("s1", rollout, None, "event_msg.agent_message", "assistant", "確認します。", None, {}),
            )
            mode._enqueue_message(
                session.id,
                CodexMessageEvent("s1", rollout, None, "response_item.message", "assistant", "実装します。", None, {}),
            )
            deadline = time.time() + 2
            while not engine.submitted and time.time() < deadline:
                time.sleep(0.01)
            mode.stop(session)

        self.assertEqual(calls, [["確認します。", "実装します。"]])
        self.assertEqual(engine.submitted[0][0], "まとめて進捗を伝えるね。")

    def test_duplicate_assistant_text_from_different_sources_is_ignored(self):
        calls = []

        def rewriter(events, recent_codex, recent_utterances, profile, model):
            calls.append([event.text for event in events])
            return {"should_speak": True, "utterance": "重複は一回だけ読むね。"}

        engine = FakeEngine()
        sessions = SessionManager()
        mode = CodexWatchMode(engine, EventBus(), sessions, rewriter=rewriter)
        session = sessions.create("codex_watch", metadata={})
        with tempfile.TemporaryDirectory() as tmp:
            rollout = Path(tmp) / "rollout-test.jsonl"
            rollout.write_text("{}", encoding="utf-8")
            mode.start(session, {"rollout_path": str(rollout), "batch_window_sec": 0.05})
            text = "まず調査方針を確認します。"
            mode._enqueue_message(
                session.id,
                CodexMessageEvent("s1", rollout, None, "event_msg.agent_message", "assistant", text, None, {}),
            )
            mode._enqueue_message(
                session.id,
                CodexMessageEvent("s1", rollout, None, "response_item.message", "assistant", text, None, {}),
            )
            deadline = time.time() + 2
            while not engine.submitted and time.time() < deadline:
                time.sleep(0.01)
            mode.stop(session)

        self.assertEqual(calls, [["まず調査方針を確認します。"]])

    def test_rewrite_false_does_not_submit_text(self):
        engine = FakeEngine()
        sessions = SessionManager()
        mode = CodexWatchMode(
            engine,
            EventBus(),
            sessions,
            rewriter=lambda *args: {"should_speak": False, "utterance": ""},
        )
        session = sessions.create("codex_watch", metadata={})
        with tempfile.TemporaryDirectory() as tmp:
            rollout = Path(tmp) / "rollout-test.jsonl"
            rollout.write_text("{}", encoding="utf-8")
            mode.start(session, {"rollout_path": str(rollout)})
            event = CodexMessageEvent("s1", rollout, None, "response_item.message", "assistant", "tool output only", None, {})
            mode._enqueue_message(session.id, event)
            time.sleep(0.1)
            mode.stop(session)

        self.assertEqual(engine.submitted, [])

    def test_delivery_policy_catches_up_when_speech_backlog_is_large(self):
        engine = FakeEngine()
        sessions = SessionManager()
        mode = CodexWatchMode(engine, EventBus(), sessions, rewriter=lambda *args: {"should_speak": False, "utterance": ""})
        session = sessions.create("codex_watch", metadata={})
        with tempfile.TemporaryDirectory() as tmp:
            rollout = Path(tmp) / "rollout-test.jsonl"
            rollout.write_text("{}", encoding="utf-8")
            mode.start(session, {"rollout_path": str(rollout)})
            state = mode._state(session.id)
            state.speech_backlog_until = time.monotonic() + 20

            context = mode._delivery_context(state)
            mode.stop(session)

        self.assertEqual(context["policy"], "catch_up")
        self.assertLess(context["target_retention"], 0.5)

    def test_catch_up_drains_queued_events_without_dropping_input(self):
        engine = FakeEngine()
        sessions = SessionManager()
        mode = CodexWatchMode(engine, EventBus(), sessions, rewriter=lambda *args: {"should_speak": False, "utterance": ""})
        session = sessions.create("codex_watch", metadata={})
        with tempfile.TemporaryDirectory() as tmp:
            rollout = Path(tmp) / "rollout-test.jsonl"
            rollout.write_text("{}", encoding="utf-8")
            mode.start(
                session,
                {
                    "rollout_path": str(rollout),
                    "catch_up_drain_limit": 20,
                },
            )
            state = mode._state(session.id)
            events = [
                CodexMessageEvent("s1", rollout, None, "event_msg.agent_message", "assistant", f"event {idx}", None, {})
                for idx in range(12)
            ]
            for event in events[2:]:
                state.message_queue.put(event)

            merged, info = mode._drain_catch_up_events(events[:2], state)
            mode.stop(session)

        self.assertEqual(info["merged_event_count"], 12)
        self.assertEqual(info["drained_event_count"], 10)
        self.assertEqual([event.text for event in merged], [f"event {idx}" for idx in range(12)])


if __name__ == "__main__":
    unittest.main()
