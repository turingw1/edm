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
    parser.add_argument("--batch", type=int, default=None, help="Override generation batch size.")
    parser.add_argument("--eval-batch", type=int, default=None, help="Override match/defect batch size.")
    parser.add_argument("--fid-batch", type=int, default=None, help="Override FID batch size.")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 network execution.")
    parser.add_argument("--skip-edm-baseline", action="store_true", help="Skip official EDM 4-step baseline FID.")
    parser.add_argument("--skip-fid", action="store_true", help="Generate samples and metrics without FID.")
    parser.add_argument("--skip-generate", action="store_true", help="Skip sample generation and reuse existing images.")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip match_mse and defect.")
    parser.add_argument("--dry-run", action="store_true", help="Print FID command without executing it.")
    args = parser.parse_args()

    import torch
    from utils.edm_proxy import (
        TARGETS,
        edm_root_from_file,
        evaluate_targets_match_and_defect,
        generate_edm_baseline_samples,
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
    if args.batch is not None:
        cfg["batch"] = int(args.batch)
    if args.eval_batch is not None:
        cfg["eval_batch"] = int(args.eval_batch)
    if args.fid_batch is not None:
        cfg["fid_batch"] = int(args.fid_batch)

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
    use_fp16 = bool(cfg.get("use_fp16", False)) and not args.fp32
    net = load_edm_network(cfg["checkpoint"], device=device, use_fp16=use_fp16)
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
        "batch": int(cfg["batch"]),
        "eval_batch": int(cfg["eval_batch"]),
        "fid_batch": int(cfg["fid_batch"]),
        "use_fp16": use_fp16,
        "targets": {},
    }

    if not args.skip_edm_baseline:
        print("\n=== DG_TWFD official EDM 4-step baseline ===")
        edm_sample_dir = samples_dir / "DG_TWFD_edm_steps4"
        edm_metrics = {
            "target": "edm",
            "fid4": None,
            "defect": None,
            "match_mse": None,
            "sample_dir": str(edm_sample_dir),
        }
        start = time.time()
        if not args.skip_generate:
            generate_edm_baseline_samples(
                edm_root=edm_root,
                network=cfg["checkpoint"],
                outdir=edm_sample_dir,
                num_samples=int(cfg["num_samples"]),
                seed=int(cfg["seed"]),
                batch_size=int(cfg["batch"]),
                num_steps=int(cfg["generation_steps"]),
                subdirs=bool(cfg.get("subdirs", True)),
                class_idx=cfg.get("class_idx"),
                log_path=logs_dir / "DG_TWFD_generate_edm_steps4.log",
                dry_run=args.dry_run,
            )
        if not args.skip_fid:
            edm_metrics["fid4"] = run_fid(
                edm_root=edm_root,
                images=edm_sample_dir,
                ref=cfg["fid_ref"],
                num_samples=int(cfg["num_samples"]),
                batch_size=int(cfg["fid_batch"]),
                log_path=logs_dir / "DG_TWFD_fid_edm_steps4.log",
                dry_run=args.dry_run,
            )
        edm_metrics["elapsed_sec"] = time.time() - start
        metrics["targets"]["edm"] = edm_metrics
        write_json(run_dir / "metrics.json", metrics)

    shared_metric_values = {}
    if not args.skip_metrics:
        shared_metric_values = evaluate_targets_match_and_defect(
            net,
            targets=list(requested_targets),
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
            target_metrics.update(shared_metric_values[target])

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
