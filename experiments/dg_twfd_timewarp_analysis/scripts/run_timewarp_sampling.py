#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


EDM_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDM_ROOT))
sys.path.insert(0, str(EXP_ROOT))

from utils.timewarp_core import (  # noqa: E402
    TIME_PARAMS,
    edm_root_from_file,
    load_edm_network,
    load_json,
    resolve_path,
    sample_trajectories,
    save_schedule_csv,
    save_trajectory,
    schedule_for_time_param,
    write_json,
)


def parse_float_list(text: str | None) -> list[float] | None:
    if not text:
        return None
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample EDM trajectories with a selected time parameterization.")
    parser.add_argument("--config", required=True, help="Path to DG_TWFD timewarp analysis config.")
    parser.add_argument("--time-param", choices=TIME_PARAMS, required=True, help="Time parameterization to use.")
    parser.add_argument("--outdir", default=None, help="Output directory. Defaults to config outdir/time_param.")
    parser.add_argument("--num-steps", type=int, default=None, help="Number of sampling intervals.")
    parser.add_argument("--num-trajectories", type=int, default=None, help="Number of trajectories to save.")
    parser.add_argument("--seed", type=int, default=None, help="First seed.")
    parser.add_argument("--batch", type=int, default=None, help="Trajectory batch size.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 network execution.")
    parser.add_argument("--warp-weights", default=None, help="Comma-separated interval weights for dg_twfd_warp.")
    parser.add_argument("--warp-json", default=None, help="JSON file with a 'weights' array for dg_twfd_warp.")
    args = parser.parse_args()

    import torch

    edm_root = edm_root_from_file(__file__)
    cfg = load_json(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    num_steps = int(args.num_steps if args.num_steps is not None else cfg["num_steps"])
    num_trajectories = int(args.num_trajectories if args.num_trajectories is not None else cfg["num_trajectories"])
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    batch = int(args.batch if args.batch is not None else cfg["batch"])

    if args.outdir:
        outdir = resolve_path(args.outdir, root=edm_root)
    else:
        outdir = resolve_path(cfg["outdir"], root=edm_root) / args.time_param
    outdir.mkdir(parents=True, exist_ok=True)

    weights = parse_float_list(args.warp_weights)
    if args.warp_json:
        weights_payload = load_json(args.warp_json)
        weights = [float(item) for item in weights_payload["weights"]]

    net = load_edm_network(cfg["checkpoint"], device=device, use_fp16=bool(args.fp16))
    sigmas, time_param = schedule_for_time_param(
        net,
        time_param=args.time_param,
        num_steps=num_steps,
        sigma_min=float(cfg["sigma_min"]),
        sigma_max=float(cfg["sigma_max"]),
        rho=float(cfg["rho"]),
        device=device,
        weights=weights,
        strength=float(cfg.get("default_warp_strength", 3.0)),
        power=float(cfg.get("default_warp_power", 1.5)),
    )

    seeds = list(range(seed, seed + num_trajectories))
    states, labels = sample_trajectories(
        net,
        sigmas=sigmas,
        seeds=seeds,
        batch_size=batch,
        class_idx=cfg.get("class_idx"),
        device=device,
    )
    metadata = {
        "dataset": cfg["dataset"],
        "checkpoint": cfg["checkpoint"],
        "time_param": args.time_param,
        "num_steps": num_steps,
        "num_trajectories": num_trajectories,
        "seed": seed,
        "sigma_min": float(cfg["sigma_min"]),
        "sigma_max": float(cfg["sigma_max"]),
        "rho": float(cfg["rho"]),
        "tau": time_param.tau.tolist(),
        "warp_weights": time_param.weights.tolist(),
    }
    save_trajectory(outdir / "trajectory.pt", states=states, sigmas=sigmas, labels=labels, seeds=seeds, metadata=metadata)
    save_schedule_csv(outdir / "schedule.csv", sigmas=sigmas.detach().cpu().numpy(), param=time_param)
    write_json(outdir / "metadata.json", metadata)
    print(f"Wrote trajectory: {outdir / 'trajectory.pt'}")
    print(f"Wrote schedule: {outdir / 'schedule.csv'}")


if __name__ == "__main__":
    main()
