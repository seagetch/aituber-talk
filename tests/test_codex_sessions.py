import json
import tempfile
import unittest
from pathlib import Path

from core.codex_sessions import list_codex_sessions


class CodexSessionsTest(unittest.TestCase):
    def test_list_codex_sessions_merges_index_and_rollout_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_id = "session-1"
            index = root / "session_index.jsonl"
            index.write_text(
                json.dumps(
                    {
                        "id": session_id,
                        "thread_name": "設計実装",
                        "updated_at": "2026-05-06T01:00:00Z",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            rollout = root / "sessions" / "2026" / "05" / "06" / "rollout-test.jsonl"
            rollout.parent.mkdir(parents=True)
            rollout.write_text(
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": session_id,
                            "cwd": "/tmp/project",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            sessions = list_codex_sessions(root)

            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].id, session_id)
            self.assertEqual(sessions[0].thread_name, "設計実装")
            self.assertEqual(sessions[0].cwd, "/tmp/project")
            self.assertEqual(sessions[0].rollout_path, rollout.resolve())


if __name__ == "__main__":
    unittest.main()
