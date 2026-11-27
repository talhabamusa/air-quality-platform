r"""Prefect version of the pipeline. It runs the same stages as run_pipeline.py.

  .venv-orch\Scripts\python.exe orchestration\prefect_flow.py --from load_staging
  .venv-orch\Scripts\python.exe orchestration\prefect_flow.py
  .venv-orch\Scripts\python.exe orchestration\prefect_flow.py --serve
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from prefect import flow, get_run_logger, task

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pipeline as rp  # noqa: E402  (reuses the list of stages)

ROOT = rp.ROOT


@task
def run_stage(name: str, cmd: list[str]) -> str:
    """Run one stage as a subprocess. A non-zero exit code fails the task."""
    logger = get_run_logger()
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    lines = (result.stdout + result.stderr).splitlines()
    tail = "\n".join(lines[-15:])
    if result.returncode != 0:
        logger.error("Stage %s failed (exit code %s). Last lines:\n%s", name, result.returncode, tail)
        raise RuntimeError(f"stage {name} failed with exit code {result.returncode}")
    logger.info("Stage %s finished. Last lines:\n%s", name, tail)
    return next((line for line in reversed(lines) if line.strip()), "")


@flow(name="air-quality-pipeline", log_prints=True)
def air_quality_pipeline(include_discovery: bool = False, start_from: str | None = None):
    load_dotenv(ROOT / ".env")
    stages = [s for s in rp.STAGES if include_discovery or s[2] != "discovery"]
    names = [s[0] for s in stages]
    if start_from:
        if start_from not in names:
            raise ValueError(f"unknown stage '{start_from}'. Choose from {names}")
        stages = stages[names.index(start_from):]

    for name, cmd, group in stages:
        step = run_stage.with_options(
            name=name,
            retries=2 if group == "ingest" else 0,     # downloads can fail on a bad connection
            retry_delay_seconds=60,
        )
        last_line = step(name, [str(c) for c in cmd])
        print(f"{name}: {last_line}")
        if name == "start_database":
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start_from")
    parser.add_argument("--include-discovery", action="store_true")
    parser.add_argument("--serve", action="store_true",
                        help="keep running and start the flow every Sunday at 03:00 India time")
    args = parser.parse_args()
    if args.serve:
        air_quality_pipeline.serve(name="weekly", cron="0 3 * * 0", timezone="Asia/Kolkata")
    else:
        air_quality_pipeline(include_discovery=args.include_discovery, start_from=args.start_from)