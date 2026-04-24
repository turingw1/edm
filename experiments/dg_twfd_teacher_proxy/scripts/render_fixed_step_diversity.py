#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EDM_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDM_ROOT))
sys.path.insert(0, str(EXP_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a fixed-step diversity grid for DG-TWFD qualitative analysis.")
    parser.add_argument("--config", required=True, help="Teacher-proxy config JSON.")
    parser.add_argument("--dataset", required=True, help="Dataset name for sanity checks.")
    parser.add_argument("--figure-id", required=True, help="Stable figure id.")
    parser.add_argument("--method", required=True, choices=["dg_twfd", "identity_clock"], help="Sampling clock.")
    parser.add_argument("--steps", required=True, type=int, help="Single step count.")
    parser.add_argument("--seeds", required=True, help="Comma/range list of seeds.")
    parser.add_argument("--grid-cols", type=int, default=8, help="Number of columns in the diversity grid.")
    parser.add_argument("--class-idx", type=int, default=None, help="Optional fixed class id for conditional models.")
    parser.add_argument("--output-root", required=True, help="Root directory for raw sample images.")
    parser.add_argument("--figure-path", required=True, help="Final PDF path.")
    parser.add_argument("--manifest-path", required=True, help="Figure manifest JSON path.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--batch", type=int, default=None, help="Override generation batch size.")
    parser.add_argument("--cell-size", type=int, default=None, help="Override rendered cell size.")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 network execution.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate raw samples even if files already exist.")
    args = parser.parse_args()

    import torch
    from utils.edm_proxy import load_edm_network  # noqa: E402
    from utils.qualitative import (  # noqa: E402
        build_diversity_canvas,
        dataset_key,
        load_json,
        parse_int_list,
        render_samples_for_rows,
        resolve_path,
        save_figure_bundle,
    )

    cfg = load_json(args.config)  # type: ignore[name-defined]
    if dataset_key(args.dataset) != dataset_key(cfg["dataset"]):
        raise ValueError(f"Dataset mismatch: command uses {args.dataset}, config uses {cfg['dataset']}")
    rows = [{"seed": int(seed), **({"class_idx": int(args.class_idx)} if args.class_idx is not None else {})} for seed in parse_int_list(args.seeds)]

    batch_size = int(args.batch or cfg["batch"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    use_fp16 = bool(cfg.get("use_fp16", False)) and not args.fp32
    net = load_edm_network(cfg["checkpoint"], device=device, use_fp16=use_fp16)

    output_root = resolve_path(args.output_root, root=EDM_ROOT)
    figure_path = resolve_path(args.figure_path, root=EDM_ROOT)
    manifest_path = resolve_path(args.manifest_path, root=EDM_ROOT)
    subdirs = bool(cfg.get("subdirs", True))
    cell_size = int(args.cell_size or (112 if dataset_key(args.dataset) == "cifar10" else 128))
    sample_dir = output_root / args.figure_id / args.method / f"steps{int(args.steps)}"

    stats = render_samples_for_rows(
        net,
        rows=rows,
        method=args.method,
        num_steps=int(args.steps),
        cfg=cfg,
        outdir=sample_dir,
        batch_size=batch_size,
        device=device,
        subdirs=subdirs,
        overwrite=bool(args.overwrite),
    )
    resolved_rows = list(stats.get("resolved_rows", rows))
    canvas = build_diversity_canvas(
        dataset=args.dataset,
        method=args.method,
        rows=resolved_rows,
        sample_dir=sample_dir,
        subdirs=subdirs,
        cell_size=cell_size,
        grid_cols=int(args.grid_cols),
        steps=int(args.steps),
    )
    payload = {
        "figure_id": args.figure_id,
        "dataset": cfg["dataset"],
        "method": args.method,
        "checkpoint": cfg["checkpoint"],
        "seeds": [int(row["seed"]) for row in resolved_rows],
        "class_labels": [row.get("class_idx") for row in resolved_rows if "class_idx" in row] or None,
        "steps": int(args.steps),
        "nfe": 2 * int(args.steps) - 1 if int(args.steps) > 1 else 1,
        "sampler_settings": {
            "clock": args.method,
            "sigma_min": float(cfg["sigma_min"]),
            "sigma_max": float(cfg["sigma_max"]),
            "rho": float(cfg["rho"]),
            "official_edm_sampler_kwargs": dict(cfg.get("official_edm_sampler_kwargs", {})),
            "use_fp16": use_fp16,
        },
        "image_grid_layout": {"rows": int((len(rows) + args.grid_cols - 1) / args.grid_cols), "cols": int(args.grid_cols), "cell_size": cell_size},
        "raw_output_root": str(sample_dir),
        "generation": {key: value for key, value in stats.items() if key != "resolved_rows"},
        "note": "Fixed-step diversity grid with many seeds under one method.",
    }
    outputs = save_figure_bundle(canvas=canvas, figure_path=figure_path, manifest_path=manifest_path, manifest_payload=payload)
    print(f"Wrote {outputs['figure_pdf']}")
    print(f"Wrote {outputs['figure_png']}")
    print(f"Wrote {outputs['manifest']}")


if __name__ == "__main__":
    main()
