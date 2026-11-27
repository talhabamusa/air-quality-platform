import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "orchestration"))

import run_pipeline  # noqa: E402


def test_stage_names_are_unique():
    names = [stage[0] for stage in run_pipeline.STAGES]
    assert len(names) == len(set(names))


def test_every_stage_points_at_a_real_file():
    for name, cmd, _group in run_pipeline.STAGES:
        program = Path(str(cmd[0])).name
        if program == "python.exe":
            assert (ROOT / cmd[1]).exists(), f"{name}: {cmd[1]} not found"
        elif program == "dbt.exe":
            assert (ROOT / "dbt_project" / "dbt_project.yml").exists()