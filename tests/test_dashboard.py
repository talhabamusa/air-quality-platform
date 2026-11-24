from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_runs_without_errors():
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.metric) == 5      # one forecast card per city