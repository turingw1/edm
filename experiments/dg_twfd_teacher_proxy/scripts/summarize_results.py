#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


EDM_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDM_ROOT))
sys.path.insert(0, str(EXP_ROOT))

from utils.summary import load_json, write_summary_tables  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a DG-TWFD EDM teacher-proxy run.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing metrics.json.")
    parser.add_argument("--results-dir", default=str(EXP_ROOT / "results"))
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metrics = load_json(run_dir / "metrics.json")
    md_path, csv_path = write_summary_tables(metrics=metrics, run_dir=run_dir, results_dir=Path(args.results_dir))
    print(f"Wrote markdown summary: {md_path}")
    print(f"Wrote CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
