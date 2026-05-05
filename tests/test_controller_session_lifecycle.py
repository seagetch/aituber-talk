import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from controller.api import attach_routes
from controller.events import EventBus
from controller.registry import ModeRegistry
from controller.session import SessionManager
from plugins.base import Mode


class FakeMode(Mode):
    def __init__(self, name, stops):
        super().__init__()
        self.name = name
        self.stops = stops

    def start(self, session, payload):
        return {"status": "running", "session": session.id}

    def stop(self, session):
        self.stops.append((self.name, session.id))
        return {"status": "stopped", "session": session.id}


class FakeEngine:
    class Config:
        jaw_open_scale = 1.0
        jaw_open_source_index = 35
        jaw_open_zero_value = 0.0
        jaw_open_one_value = 0.3

    config = Config()

    def status(self):
        return {}


class FakeController:
    def __init__(self, upload_dir):
        self.sessions = SessionManager()
        self.registry = ModeRegistry()
        self.event_bus = EventBus()
        self.upload_dir = upload_dir
        self.engine = FakeEngine()
        self.stops = []
        self._modes = {}
        for name in ("codex_watch", "talk"):
            self.registry.register(name, lambda mode_name=name, **deps: FakeMode(mode_name, self.stops))

    def start_engine(self):
        pass

    def get_mode(self, name):
        if name not in self._modes:
            self._modes[name] = self.registry.create(name)
        return self._modes[name]


class ControllerSessionLifecycleTest(unittest.TestCase):
    def test_talk_session_does_not_stop_active_codex_watch(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = FastAPI()
            controller = FakeController(Path(tmp))
            attach_routes(app, controller)
            watcher = controller.sessions.create("codex_watch", metadata={})
            controller.sessions.update_status(watcher.id, "running")

            response = TestClient(app).post("/sessions", json={"mode": "talk", "payload": {"text": "hello"}})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(controller.sessions.get(watcher.id).status, "running")
            self.assertEqual(controller.stops, [])


if __name__ == "__main__":
    unittest.main()
