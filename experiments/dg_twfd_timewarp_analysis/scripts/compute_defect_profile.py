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
    compute_defect_rows,
    edm_root_from_file,
    load_edm_network,
    load_json,
    load_trajectory,
    resolve_path,
    save_defect_matrix,
    write_defect_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute interval-wise composition defect for an EDM trajectory.")
    parser.add_argument("--config", required=True, help="Path to DG_TWFD timewarp analysis config.")
    parser.add_argument("--trajectory", required=True, help="Path to trajectory.pt from run_timewarp_sampling.py.")
    parser.add_argument("--out-csv", default=None, help="Output per-interval defect CSV.")
    parser.add_argument("--out-summary", default=None, help="Output summary JSON.")
    parser.add_argument("--out-matrix", default=None, help="Optional per-trajectory defect matrix (.npz).")
    parser.add_argument("--batch", type=int, default=None, help="Defect evaluation batch size.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 network execution.")
    args = parser.parse_args()

    import torch

    edm_root = edm_root_from_file(__file__)
    cfg = load_json(args.config)
    trajectory = load_trajectory(args.trajectory)
    time_param = trajectory.get("metadata", {}).get("time_param", "unknown")
    out_csv = resolve_path(args.out_csv, root=edm_root) if args.out_csv else EXP_ROOT / "results" / f"defect_{time_param}.csv"
    out_summary = (
        resolve_path(args.out_summary, root=edm_root)
        if args.out_summary
        else EXP_ROOT / "results" / f"defect_{time_param}_summary.json"
    )
    out_matrix = (
        resolve_path(args.out_matrix, root=edm_root)
        if args.out_matrix
        else EXP_ROOT / "results" / f"defect_{time_param}_matrix.npz"
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    net = load_edm_network(cfg["checkpoint"], device=device, use_fp16=bool(args.fp16))
    rows, summary, defect_matrix = compute_defect_rows(
        net,
        trajectory=trajectory,
        batch_size=int(args.batch if args.batch is not None else cfg["defect_batch"]),
        defect_eps=float(cfg["defect_eps"]),
        device=device,
        return_matrix=True,
    )
    write_defect_csv(out_csv, rows)
    write_json(out_summary, summary)
    save_defect_matrix(
        out_matrix,
        defect_matrix=defect_matrix,
        seeds=trajectory.get("seeds", list(range(defect_matrix.shape[0]))),
        time_param=time_param,
    )
    print(f"Wrote per-interval defect CSV: {out_csv}")
    print(f"Wrote defect summary: {out_summary}")
    print(f"Wrote per-trajectory defect matrix: {out_matrix}")


if __name__ == "__main__":
    main()
