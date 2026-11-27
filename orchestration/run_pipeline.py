r"""Run the pipeline stage by stage, with a log file and a stop at the first failure.

Run it with the pipeline environment, from the project root:
  .venv\Scripts\python.exe orchestration\run_pipeline.py --list
  .venv\Scripts\python.exe orchestration\run_pipeline.py
  .venv\Scripts\python.exe orchestration\run_pipeline.py --from build_silver_hours
  .venv\Scripts\python.exe orchestration\run_pipeline.py --only quality_rules,impute_silver
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PY_ML = ROOT / ".venv-ml" / "Scripts" / "python.exe"
DBT = ROOT / ".venv-dbt" / "Scripts" / "dbt.exe"

# (name, command, group). The discovery group depends on today's date, so it is off by default.
STAGES = [
    ("start_database", ["docker", "compose", "up", "-d"], "setup"),
    ("discover_locations", [PY, "ingestion/discover_locations.py"], "discovery"),
    ("build_sensor_manifest", [PY, "ingestion/build_sensor_manifest.py"], "discovery"),
    ("download_hours_long", [PY, "ingestion/download_hours_long.py"], "ingest"),
    ("download_weather_long", [PY, "ingestion/download_weather_long.py"], "ingest"),
    ("build_silver_weather", [PY, "transform/build_silver_weather.py"], "transform"),
    ("build_silver_hours", [PY, "transform/build_silver_hours.py"], "transform"),
    ("quality_rules", [PY, "transform/quality_rules.py"], "transform"),
    ("impute_silver", [PY, "transform/impute_silver.py"], "transform"),
    ("load_staging", [PY, "transform/load_staging.py"], "load"),
    ("dbt_build", [DBT, "build", "--project-dir", "dbt_project", "--profiles-dir", "dbt_project"], "warehouse"),
    ("train_daily", [PY_ML, "ml/train_daily.py"], "model"),
    ("train_intervals", [PY_ML, "ml/train_intervals.py"], "model"),
    ("export_dashboard_data", [PY_ML, "dashboard/export_data.py"], "serve"),
]


def last_lines(path, start, n):
    data = path.read_bytes()[start:]
    return data.decode("utf-8", errors="replace").splitlines()[-n:]


def main():
    parser = argparse.ArgumentParser(description="Run the air quality pipeline.")
    parser.add_argument("--list", action="store_true", help="show the stages and exit")
    parser.add_argument("--from", dest="start", help="first stage to run")
    parser.add_argument("--to", dest="end", help="last stage to run")
    parser.add_argument("--only", help="comma-separated stages to run")
    parser.add_argument("--include-discovery", action="store_true",
                        help="also run the two discovery stages (they depend on today's date)")
    args = parser.parse_args()

    stages = [s for s in STAGES if args.include_discovery or s[2] != "discovery"]
    names = [s[0] for s in stages]

    if args.list:
        for name, cmd, group in stages:
            print(f"{name:24} {group:10} {' '.join(Path(str(c)).name if i == 0 else str(c) for i, c in enumerate(cmd))}")
        return 0

    if args.only:
        wanted = [w.strip() for w in args.only.split(",")]
        unknown = [w for w in wanted if w not in names]
        if unknown:
            parser.error(f"unknown stage(s) {unknown}. Use --list (and --include-discovery for the discovery stages).")
        stages = [s for s in stages if s[0] in wanted]
    else:
        for label, value in (("--from", args.start), ("--to", args.end)):
            if value and value not in names:
                parser.error(f"{label}: unknown stage '{value}'. Use --list (and --include-discovery for the discovery stages).")
        first = names.index(args.start) if args.start else 0
        last = names.index(args.end) if args.end else len(names) - 1
        stages = stages[first:last + 1]
    if not stages:
        parser.error("no stages selected")

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"

    summary = []
    with open(log_path, "ab") as log:
        for k, (name, cmd, _group) in enumerate(stages, start=1):
            print(f"[{k}/{len(stages)}] {name} ...", flush=True)
            log.write(f"\n===== {name} | {datetime.now():%Y-%m-%d %H:%M:%S} =====\n".encode())
            log.flush()
            pos = log.tell()
            started = time.time()
            try:
                code = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env,
                                      stdout=log, stderr=subprocess.STDOUT).returncode
            except FileNotFoundError as exc:
                code = 127
                log.write(f"{exc}\n".encode())
            log.flush()
            minutes = (time.time() - started) / 60
            tail = last_lines(log_path, pos, 12)
            last = next((line for line in reversed(tail) if line.strip()), "")
            status = "ok" if code == 0 else "FAILED"
            summary.append((name, status, minutes, last))
            print(f"    {status} in {minutes:.1f} min. Last line: {last}", flush=True)

            if code != 0:
                print("\nLast lines of the output:")
                print("\n".join(tail))
                print(f"\nStopped. Fix the problem, then continue with:  --from {name}")
                print(f"Full log: {log_path}")
                return 1
            if name == "start_database":
                time.sleep(5)

    print("\nSummary:")
    for name, status, minutes, last in summary:
        print(f"  {name:24} {status:7} {minutes:6.1f} min   {last}")
    print(f"\nLog: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())