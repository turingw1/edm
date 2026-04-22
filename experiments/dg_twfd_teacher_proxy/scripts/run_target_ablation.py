#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path


EDM_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDM_ROOT))
sys.path.insert(0, str(EXP_ROOT))

from utils.summary import load_json, write_json, write_summary_tables  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DG-TWFD target-space teacher-proxy ablation on EDM.")
    parser.add_argument("--config", required=True, help="Path to a DG_TWFD target-ablation JSON config.")
    parser.add_argument("--outdir", default=None, help="Override config outdir.")
    parser.add_argument("--device", default="cuda", help="Torch device. Use cuda on the server.")
    parser.add_argument("--targets", default=None, help="Optional comma-separated target subset.")
    parser.add_argument("--num-samples", type=int, default=None, help="Override FID/sample count.")
    parser.add_argument("--eval-num-triplets", type=int, default=None, help="Override match/defect triplet count.")
    parser.add_argument("--skip-fid", action="store_true", help="Generate samples and metrics without FID.")
    parser.add_argument("--skip-generate", action="store_true", help="Skip sample generation and reuse existing images.")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip match_mse and defect.")
    parser.add_argument("--dry-run", action="store_true", help="Print FID command without executing it.")
    args = parser.parse_args()

    import torch
    from utils.edm_proxy import (
        TARGETS,
        edm_root_from_file,
        evaluate_match_and_defect,
        generate_target_samples,
        load_edm_network,
        resolve_path,
        run_fid,
    )

    edm_root = edm_root_from_file(__file__)
    cfg_path = Path(args.config).resolve()
    cfg = load_json(cfg_path)
    if args.outdir is not None:
        cfg["outdir"] = args.outdir
    if args.num_samples is not None:
        cfg["num_samples"] = int(args.num_samples)
    if args.eval_num_triplets is not None:
        cfg["eval_num_triplets"] = int(args.eval_num_triplets)

    requested_targets = cfg.get("targets", list(TARGETS))
    if args.targets:
        requested_targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    for target in requested_targets:
        if target not in TARGETS:
            raise ValueError(f"Unsupported target {target}; expected one of {TARGETS}")

    run_dir = resolve_path(cfg["outdir"], root=edm_root)
    samples_dir = run_dir / "samples"
    logs_dir = run_dir / "logs"
    result_dir = EXP_ROOT / "results"
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg_path, run_dir / "run_config.json")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    net = load_edm_network(cfg["checkpoint"], device=device)
    approx_cfg = cfg["approximation"]

    metrics = {
        "experiment_name": cfg["experiment_name"],
        "dataset": cfg["dataset"],
        "checkpoint": cfg["checkpoint"],
        "sampler_name": cfg["sampler_name"],
        "num_samples": int(cfg["num_samples"]),
        "seed": int(cfg["seed"]),
        "generation_steps": int(cfg["generation_steps"]),
        "approximation": approx_cfg,
        "targets": {},
    }

    for target in requested_targets:
        print(f"\n=== DG_TWFD target ablation: {target} ===")
        target_sample_dir = samples_dir / f"DG_TWFD_{target}_steps4"
        target_metrics = {
            "target": target,
            "fid4": None,
            "defect": None,
            "match_mse": None,
            "sample_dir": str(target_sample_dir),
        }

        start = time.time()
        if not args.skip_generate:
            generate_target_samples(
                net,
                target=target,
                outdir=target_sample_dir,
                num_samples=int(cfg["num_samples"]),
                seed=int(cfg["seed"]),
                batch_size=int(cfg["batch"]),
                num_steps=int(cfg["generation_steps"]),
                sigma_min=float(cfg["sigma_min"]),
                sigma_max=float(cfg["sigma_max"]),
                rho=float(cfg["rho"]),
                approx_cfg=approx_cfg,
                class_idx=cfg.get("class_idx"),
                subdirs=bool(cfg.get("subdirs", True)),
                device=device,
            )

        if not args.skip_metrics:
            metric_values = evaluate_match_and_defect(
                net,
                target=target,
                num_triplets=int(cfg["eval_num_triplets"]),
                seed=int(cfg["seed"]),
                batch_size=int(cfg["eval_batch"]),
                transition_grid_steps=int(cfg["transition_grid_steps"]),
                sigma_min=float(cfg["sigma_min"]),
                sigma_max=float(cfg["sigma_max"]),
                rho=float(cfg["rho"]),
                approx_cfg=approx_cfg,
                class_idx=cfg.get("class_idx"),
                defect_eps=float(cfg["defect_eps"]),
                device=device,
            )
            target_metrics.update(metric_values)

        if not args.skip_fid:
            fid = run_fid(
                edm_root=edm_root,
                images=target_sample_dir,
                ref=cfg["fid_ref"],
                num_samples=int(cfg["num_samples"]),
                batch_size=int(cfg["fid_batch"]),
                log_path=logs_dir / f"DG_TWFD_fid_{target}_steps4.log",
                dry_run=args.dry_run,
            )
            target_metrics["fid4"] = fid

        target_metrics["elapsed_sec"] = time.time() - start
        metrics["targets"][target] = target_metrics
        write_json(run_dir / "metrics.json", metrics)

    md_path, csv_path = write_summary_tables(metrics=metrics, run_dir=run_dir, results_dir=result_dir)
    write_json(run_dir / "metrics.json", metrics)
    print(f"\nWrote markdown summary: {md_path}")
    print(f"Wrote CSV summary: {csv_path}")
    print(f"Wrote run metrics: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
