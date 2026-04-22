from __future__ import annotations

import json
import math
from pathlib import Path


TARGET_ORDER = ("velocity", "residual", "endpoint")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary_tables(*, metrics: dict, run_dir: Path, results_dir: Path) -> tuple[Path, Path]:
    dataset = str(metrics["dataset"])
    dataset_key = dataset.lower().replace("-", "").replace(" ", "")
    if "cifar" in dataset_key:
        dataset_key = "cifar10"
    elif "imagenet" in dataset_key:
        dataset_key = "imagenet64"
    csv_path = results_dir / f"summary_{dataset_key}.csv"
    md_path = results_dir / f"summary_{dataset_key}.md"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = metrics["targets"]
    fieldnames = ["dataset", "target", "fid4", "defect", "match_mse", "checkpoint", "sampler_name", "num_samples", "seed"]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for target_name in TARGET_ORDER:
            if target_name not in rows:
                continue
            row = rows[target_name]
            values = [
                dataset,
                target_name,
                _format_value(row.get("fid4")),
                _format_value(row.get("defect")),
                _format_value(row.get("match_mse")),
                metrics["checkpoint"],
                metrics["sampler_name"],
                str(metrics["num_samples"]),
                str(metrics["seed"]),
            ]
            f.write(",".join(values) + "\n")

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write("Eval: same pretrained EDM checkpoint, same 4-step protocol\n\n")
        f.write("| target | FID@4↓ | defect↓ | match MSE↓ |\n")
        f.write("|---|---:|---:|---:|\n")
        for target_name in TARGET_ORDER:
            if target_name not in rows:
                continue
            row = rows[target_name]
            f.write(
                f"| {target_name} | {_format_value(row.get('fid4'))} | "
                f"{_format_value(row.get('defect'))} | {_format_value(row.get('match_mse'))} |\n"
            )
        f.write(f"\nSource run: `{run_dir}`\n")
    return md_path, csv_path


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    value = float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"
