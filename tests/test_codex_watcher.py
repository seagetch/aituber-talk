import json
import tempfile
import time
import unittest
from pathlib import Path

from core.codex_watcher import CodexJsonlWatcher


class CodexWatcherTest(unittest.TestCase):
    def test_reads_complete_lines_and_buffers_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            rollout = Path(tmp) / "rollout-test.jsonl"
            rollout.write_text("", encoding="utf-8")
            watcher = CodexJsonlWatcher(rollout, start_at_end=False)

            with rollout.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "type": "session_meta",
                            "payload": {"id": "s1", "cwd": "/tmp/project"},
                        }
                    )
                    + "\n"
                )
                fh.write('{"type":"event_msg","payload":{"type":"agent_message","message":"hel')
            self.assertEqual(watcher.read_available(), [])

            with rollout.open("a", encoding="utf-8") as fh:
                fh.write('lo"}}\n')
            events = watcher.read_available()

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].session_id, "s1")
            self.assertEqual(events[0].source_type, "event_msg.agent_message")
            self.assertEqual(events[0].role, "assistant")
            self.assertEqual(events[0].text, "hello")
            self.assertEqual(events[0].cwd, "/tmp/project")

    def test_truncate_resets_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            rollout = Path(tmp) / "rollout-test.jsonl"
            rollout.write_text('{"type":"event_msg","payload":{"type":"agent_message","message":"old message is longer"}}\n', encoding="utf-8")
            watcher = CodexJsonlWatcher(rollout, start_at_end=True)
            rollout.write_text('{"type":"event_msg","payload":{"type":"agent_message","message":"new"}}\n', encoding="utf-8")

            events = watcher.read_available()

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].text, "new")

    def test_start_polls_and_reports_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            rollout = Path(tmp) / "rollout-test.jsonl"
            rollout.write_text("", encoding="utf-8")
            watcher = CodexJsonlWatcher(rollout, start_at_end=True, poll_interval=0.02)
            events = []
            activities = []
            watcher.start(events.append, activity_callback=activities.append)
            try:
                with rollout.open("a", encoding="utf-8") as fh:
                    fh.write('{"type":"event_msg","payload":{"type":"agent_message","message":"poll ok"}}\n')
                deadline = time.time() + 1
                while not events and time.time() < deadline:
                    time.sleep(0.01)
            finally:
                watcher.stop()

            self.assertEqual([event.text for event in events], ["poll ok"])
            self.assertTrue(activities)
            self.assertGreaterEqual(activities[-1].lines_read, 1)


if __name__ == "__main__":
    unittest.main()
