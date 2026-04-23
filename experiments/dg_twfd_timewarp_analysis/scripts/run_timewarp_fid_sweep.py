#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


EDM_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDM_ROOT))
sys.path.insert(0, str(EXP_ROOT))

from utils.timewarp_core import (  # noqa: E402
    TIME_PARAMS,
    edm_root_from_file,
    generate_final_samples,
    load_edm_network,
    load_json,
    resolve_path,
    run_fid,
    save_schedule_csv,
    schedule_for_time_param,
    write_fid_sweep_tables,
    write_json,
)


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                raise ValueError(f"Invalid range: {item}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    return values


def parse_time_params(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    for value in values:
        if value not in TIME_PARAMS:
            raise ValueError(f"Unsupported time parameterization {value}; expected one of {TIME_PARAMS}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FID sweep for EDM identity vs DG-TWFD warped time schedules.")
    parser.add_argument("--config", required=True, help="Path to DG_TWFD timewarp analysis config.")
    parser.add_argument("--outdir", default=None, help="Output directory for samples, logs, and schedules.")
    parser.add_argument("--time-params", default="identity,dg_twfd_warp", help="Comma-separated time parameterizations.")
    parser.add_argument("--steps", default="16,32,48,64", help="Comma/range list of step counts.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples per setting for FID.")
    parser.add_argument("--seed", type=int, default=None, help="First seed.")
    parser.add_argument("--batch", type=int, default=None, help="Generation batch size.")
    parser.add_argument("--fid-batch", type=int, default=None, help="FID batch size.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 network execution. Default is FP32.")
    parser.add_argument("--skip-generate", action="store_true", help="Reuse existing samples.")
    parser.add_argument("--skip-fid", action="store_true", help="Generate samples and schedules without FID.")
    args = parser.parse_args()

    import torch

    edm_root = edm_root_from_file(__file__)
    cfg = load_json(args.config)
    if "fid_ref" not in cfg:
        raise ValueError("Config must define fid_ref for FID evaluation")
    steps = parse_int_list(args.steps)
    time_params = parse_time_params(args.time_params)
    num_samples = int(args.num_samples if args.num_samples is not None else cfg.get("fid_num_samples", cfg.get("num_samples", 50000)))
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    batch = int(args.batch if args.batch is not None else cfg.get("fid_generation_batch", cfg.get("batch", 64)))
    fid_batch = int(args.fid_batch if args.fid_batch is not None else cfg.get("fid_batch", 64))
    subdirs = bool(cfg.get("subdirs", True))

    outdir = resolve_path(args.outdir, root=edm_root) if args.outdir else resolve_path(cfg["outdir"], root=edm_root) / "fid_sweep"
    samples_dir = outdir / "samples"
    schedules_dir = outdir / "schedules"
    logs_dir = outdir / "logs"
    result_dir = EXP_ROOT / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    schedules_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    net = load_edm_network(cfg["checkpoint"], device=device, use_fp16=bool(args.fp16))
    seeds = list(range(seed, seed + num_samples))
    rows: list[dict] = []

    for time_param in time_params:
        for num_steps in steps:
            print(f"\n=== DG_TWFD time schedule FID: {time_param}, {num_steps} steps ===")
            sigmas, param = schedule_for_time_param(
                net,
                time_param=time_param,
                num_steps=int(num_steps),
                sigma_min=float(cfg["sigma_min"]),
                sigma_max=float(cfg["sigma_max"]),
                rho=float(cfg["rho"]),
                device=device,
                strength=float(cfg.get("default_warp_strength", 3.0)),
                power=float(cfg.get("default_warp_power", 1.5)),
            )
            sample_dir = samples_dir / f"DG_TWFD_{time_param}_steps{num_steps}"
            schedule_csv = schedules_dir / f"DG_TWFD_{time_param}_steps{num_steps}_schedule.csv"
            save_schedule_csv(schedule_csv, sigmas=sigmas.detach().cpu().numpy(), param=param)

            row = {
                "time_param": time_param,
                "num_steps": int(num_steps),
                "fid": None,
                "num_samples": num_samples,
                "seed": seed,
                "sample_dir": str(sample_dir),
                "schedule_csv": str(schedule_csv),
                "elapsed_sec": None,
            }
            start = time.time()
            if not args.skip_generate:
                generate_final_samples(
                    net,
                    sigmas=sigmas,
                    outdir=sample_dir,
                    seeds=seeds,
                    batch_size=batch,
                    class_idx=cfg.get("class_idx"),
                    subdirs=subdirs,
                    device=device,
                )
            if not args.skip_fid:
                row["fid"] = run_fid(
                    edm_root=edm_root,
                    images=sample_dir,
                    ref=cfg["fid_ref"],
                    num_samples=num_samples,
                    batch_size=fid_batch,
                    log_path=logs_dir / f"DG_TWFD_fid_{time_param}_steps{num_steps}.log",
                )
            row["elapsed_sec"] = time.time() - start
            rows.append(row)
            write_json(outdir / "fid_sweep_metrics.json", {"config": cfg, "rows": rows})

    csv_path = result_dir / "fid_schedule_comparison.csv"
    md_path = result_dir / "fid_schedule_comparison.md"
    write_fid_sweep_tables(rows=rows, md_path=md_path, csv_path=csv_path)
    write_json(outdir / "fid_sweep_metrics.json", {"config": cfg, "rows": rows})
    print(f"\nWrote FID CSV: {csv_path}")
    print(f"Wrote FID summary: {md_path}")
    print(f"Wrote run metrics: {outdir / 'fid_sweep_metrics.json'}")


if __name__ == "__main__":
    main()
