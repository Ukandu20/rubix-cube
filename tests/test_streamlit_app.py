from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


@unittest.skipUnless(
    importlib.util.find_spec("streamlit") is not None,
    "streamlit is not installed",
)
class StreamlitSmokeTests(unittest.TestCase):
    def test_app_starts_without_uncaught_exception(self):
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        self.assertEqual(app.exception, [])
        self.assertTrue(any("PPO Rubik" in title.value for title in app.title))


if __name__ == "__main__":
    unittest.main()
