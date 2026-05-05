import unittest
import wave
from io import BytesIO
from unittest import mock

import numpy as np

from core import talk_engine


class TalkEngineBlinkTest(unittest.TestCase):
    def test_safe_blink_flattens_column_vector(self):
        values = np.arange(21, dtype=np.float32).reshape(21, 1) / 10.0
        with mock.patch.object(talk_engine, "_orig_blink", return_value=values):
            self.assertEqual(len(talk_engine._safe_blink(21)), 21)
            np.testing.assert_allclose(talk_engine._safe_blink(21)[:3], [0.0, 0.1, 0.2])

    def test_wav_duration_seconds(self):
        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\0\0" * 8000)
        self.assertAlmostEqual(talk_engine._wav_duration_seconds(buf.getvalue()), 0.5)


if __name__ == "__main__":
    unittest.main()
