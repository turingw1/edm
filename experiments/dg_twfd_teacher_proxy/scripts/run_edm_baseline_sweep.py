#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

EDM_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDM_ROOT))
sys.path.insert(0, str(EXP_ROOT))

from utils.summary import load_json, write_json  # noqa: E402


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


def dataset_key(name: str) -> str:
    key = name.lower().replace("-", "").replace(" ", "")
    if "cifar" in key:
        return "cifar10"
    if "imagenet" in key:
        return "imagenet64"
    return key


def image_path_for_seed(root: Path, seed: int, *, subdirs: bool) -> Path:
    if subdirs:
        return root / f"{seed - seed % 1000:06d}" / f"{seed:06d}.png"
    return root / f"{seed:06d}.png"


def format_float(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6g}"


def write_sweep_summaries(*, metrics: dict, outdir: Path, results_dir: Path) -> tuple[Path, Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    key = dataset_key(metrics["dataset"])
    csv_path = results_dir / f"edm_baseline_sweep_{key}.csv"
    md_path = results_dir / f"edm_baseline_sweep_{key}.md"
    log_path = outdir / "DG_TWFD_edm_baseline_sweep.log"

    rows = sorted(metrics["steps"].values(), key=lambda row: int(row["steps"]))
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("dataset,steps,fid,checkpoint,num_samples,seed,sample_dir,elapsed_sec\n")
        for row in rows:
            f.write(
                ",".join(
                    [
                        metrics["dataset"],
                        str(row["steps"]),
                        format_float(row.get("fid")),
                        metrics["checkpoint"],
                        str(metrics["num_samples"]),
                        str(metrics["seed"]),
                        row["sample_dir"],
                        format_float(row.get("elapsed_sec")),
                    ]
                )
                + "\n"
            )

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"Dataset: {metrics['dataset']}\n")
        f.write("Eval: official EDM sampler configuration, same seeds across step counts\n\n")
        f.write("| steps | FID↓ | elapsed sec |\n")
        f.write("|---:|---:|---:|\n")
        for row in rows:
            f.write(f"| {row['steps']} | {format_float(row.get('fid'))} | {format_float(row.get('elapsed_sec'))} |\n")
        f.write(f"\nGrid: `{metrics.get('grid_path', '')}`\n")
        f.write(f"Source run: `{outdir}`\n")

    with log_path.open("w", encoding="utf-8") as f:
        f.write("DG_TWFD EDM baseline step sweep\n")
        f.write(f"dataset: {metrics['dataset']}\n")
        f.write(f"checkpoint: {metrics['checkpoint']}\n")
        f.write(f"num_samples: {metrics['num_samples']}\n")
        f.write(f"seed: {metrics['seed']}\n")
        f.write(f"steps: {','.join(str(row['steps']) for row in rows)}\n")
        f.write(f"sampler_kwargs: {json.dumps(metrics['sampler_kwargs'], sort_keys=True)}\n\n")
        f.write("steps\tfid\telapsed_sec\tsample_dir\n")
        for row in rows:
            f.write(
                f"{row['steps']}\t{format_float(row.get('fid'))}\t"
                f"{format_float(row.get('elapsed_sec'))}\t{row['sample_dir']}\n"
            )
        f.write(f"\ngrid_path: {metrics.get('grid_path', '')}\n")
        f.write(f"csv_path: {csv_path}\n")
        f.write(f"markdown_path: {md_path}\n")
    return md_path, csv_path, log_path


def build_grid(
    *,
    step_dirs: dict[int, Path],
    steps: list[int],
    seeds: list[int],
    out_path: Path,
    subdirs: bool,
    cell_size: int,
) -> None:
    import PIL.Image
    import PIL.ImageDraw

    if not seeds:
        return
    label_w = max(72, len(str(max(seeds))) * 10 + 24)
    label_h = 28
    width = label_w + cell_size * len(steps)
    height = label_h + cell_size * len(seeds)
    canvas = PIL.Image.new("RGB", (width, height), "white")
    draw = PIL.ImageDraw.Draw(canvas)

    for col, step in enumerate(steps):
        x = label_w + col * cell_size
        draw.text((x + 6, 7), f"{step} step", fill=(0, 0, 0))
    for row, seed in enumerate(seeds):
        y = label_h + row * cell_size
        draw.text((8, y + cell_size // 2 - 6), f"seed {seed}", fill=(0, 0, 0))
        for col, step in enumerate(steps):
            img_path = image_path_for_seed(step_dirs[step], seed, subdirs=subdirs)
            x = label_w + col * cell_size
            if img_path.is_file():
                nearest = getattr(getattr(PIL.Image, "Resampling", PIL.Image), "NEAREST")
                image = PIL.Image.open(img_path).convert("RGB").resize((cell_size, cell_size), nearest)
                canvas.paste(image, (x, y))
            else:
                draw.rectangle((x, y, x + cell_size - 1, y + cell_size - 1), outline=(200, 0, 0), width=2)
                draw.text((x + 6, y + 6), "missing", fill=(200, 0, 0))

    for col in range(len(steps) + 1):
        x = label_w + col * cell_size
        draw.line((x, 0, x, height), fill=(220, 220, 220))
    for row in range(len(seeds) + 1):
        y = label_h + row * cell_size
        draw.line((0, y, width, y), fill=(220, 220, 220))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def edm_sampler_compatible(
    net,
    latents,
    class_labels=None,
    randn_like=None,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    import numpy as np
    import torch

    if randn_like is None:
        randn_like = torch.randn_like
    sigma_min = max(float(sigma_min), float(net.sigma_min))
    sigma_max = min(float(sigma_max), float(net.sigma_max))

    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if num_steps == 1:
        t_steps = torch.as_tensor([sigma_max], dtype=torch.float64, device=latents.device)
    else:
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(float(S_churn) / num_steps, np.sqrt(2) - 1) if float(S_min) <= t_cur <= float(S_max) else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * float(S_noise) * randn_like(x_cur)

        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def generate_samples(
    *,
    net,
    outdir: Path,
    seeds: list[int],
    batch_size: int,
    num_steps: int,
    sampler_kwargs: dict,
    class_idx: int | None,
    subdirs: bool,
    device,
) -> None:
    from utils.edm_proxy import StackedRandomGenerator, make_class_labels, save_image_batch

    outdir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(seeds), batch_size):
        batch_seeds = seeds[start : start + batch_size]
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([len(batch_seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = make_class_labels(net, rnd, len(batch_seeds), device, class_idx)
        images = edm_sampler_compatible(
            net,
            latents,
            class_labels=class_labels,
            randn_like=rnd.randn_like,
            num_steps=num_steps,
            **sampler_kwargs,
        )
        save_image_batch(images, batch_seeds=batch_seeds, outdir=outdir, subdirs=subdirs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official EDM baseline FID sweep over multiple step counts.")
    parser.add_argument("--config", required=True, help="Path to a DG_TWFD target-ablation JSON config.")
    parser.add_argument("--outdir", default=None, help="Output directory for sweep samples, logs, metrics, and grid.")
    parser.add_argument("--steps", default="1,2,4,8,16", help="Comma/range list of step counts, e.g. 1,2,4,8,16.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of generated samples per step for FID.")
    parser.add_argument("--seed", type=int, default=None, help="First seed for generation.")
    parser.add_argument("--batch", type=int, default=None, help="Generation batch size.")
    parser.add_argument("--fid-batch", type=int, default=None, help="FID batch size.")
    parser.add_argument("--grid-seeds", default=None, help="Comma/range list of seeds to show in the grid.")
    parser.add_argument("--grid-rows", type=int, default=8, help="Number of grid rows when --grid-seeds is omitted.")
    parser.add_argument("--grid-cell-size", type=int, default=128, help="Rendered grid cell size in pixels.")
    parser.add_argument("--device", default="cuda", help="Torch device. Use cuda on the server.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 network execution. Default is FP32 to match EDM generate.py.")
    parser.add_argument("--fp32", action="store_true", help="Deprecated compatibility flag; FP32 is already the default.")
    parser.add_argument("--skip-generate", action="store_true", help="Reuse existing samples.")
    parser.add_argument("--skip-fid", action="store_true", help="Generate samples and grid without calculating FID.")
    args = parser.parse_args()

    import torch
    from utils.edm_proxy import edm_root_from_file, load_edm_network, resolve_path, run_fid

    edm_root = edm_root_from_file(__file__)
    cfg_path = Path(args.config).resolve()
    cfg = load_json(cfg_path)
    steps = parse_int_list(args.steps)
    if not steps:
        raise ValueError("At least one step count is required")

    num_samples = int(args.num_samples if args.num_samples is not None else cfg["num_samples"])
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    batch = int(args.batch if args.batch is not None else cfg["batch"])
    fid_batch = int(args.fid_batch if args.fid_batch is not None else cfg["fid_batch"])
    subdirs = bool(cfg.get("subdirs", True))
    class_idx = cfg.get("class_idx")
    sampler_kwargs = {
        "sigma_min": float(cfg.get("sigma_min", 0.002)),
        "sigma_max": float(cfg.get("sigma_max", 80.0)),
        "rho": float(cfg.get("rho", 7.0)),
    }
    sampler_kwargs.update(dict(cfg.get("official_edm_sampler_kwargs", {})))

    if args.outdir is None:
        outdir = EDM_ROOT / "experiments" / "dg_twfd_teacher_proxy" / "outputs" / f"DG_TWFD_{dataset_key(cfg['dataset'])}_edm_baseline_sweep"
    else:
        outdir = resolve_path(args.outdir, root=edm_root)
    samples_root = outdir / "samples"
    logs_dir = outdir / "logs"
    grids_dir = outdir / "grids"
    results_dir = EXP_ROOT / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    samples_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    grids_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    use_fp16 = bool(args.fp16) and not args.fp32
    net = load_edm_network(cfg["checkpoint"], device=device, use_fp16=use_fp16)

    seeds = list(range(seed, seed + num_samples))
    if args.grid_seeds:
        grid_seeds = parse_int_list(args.grid_seeds)
    else:
        grid_seeds = seeds[: max(int(args.grid_rows), 0)]

    metrics = {
        "experiment_name": f"DG_TWFD_{dataset_key(cfg['dataset'])}_edm_baseline_sweep",
        "dataset": cfg["dataset"],
        "checkpoint": cfg["checkpoint"],
        "fid_ref": cfg["fid_ref"],
        "num_samples": num_samples,
        "seed": seed,
        "steps_requested": steps,
        "sampler_kwargs": sampler_kwargs,
        "batch": batch,
        "fid_batch": fid_batch,
        "use_fp16": use_fp16,
        "grid_seeds": grid_seeds,
        "steps": {},
    }
    step_dirs: dict[int, Path] = {}

    for step in steps:
        print(f"\n=== DG_TWFD EDM baseline sweep: {step} steps ===")
        step_dir = samples_root / f"DG_TWFD_edm_steps{step}"
        step_dirs[step] = step_dir
        row = {
            "steps": step,
            "fid": None,
            "sample_dir": str(step_dir),
            "elapsed_sec": None,
        }
        start_time = time.time()
        if not args.skip_generate:
            generate_samples(
                net=net,
                outdir=step_dir,
                seeds=seeds,
                batch_size=batch,
                num_steps=step,
                sampler_kwargs=sampler_kwargs,
                class_idx=class_idx,
                subdirs=subdirs,
                device=device,
            )
        if not args.skip_fid:
            row["fid"] = run_fid(
                edm_root=edm_root,
                images=step_dir,
                ref=cfg["fid_ref"],
                num_samples=num_samples,
                batch_size=fid_batch,
                log_path=logs_dir / f"DG_TWFD_fid_edm_steps{step}.log",
                dry_run=False,
            )
        row["elapsed_sec"] = time.time() - start_time
        metrics["steps"][str(step)] = row
        write_json(outdir / "metrics_edm_baseline_sweep.json", metrics)

    grid_path = grids_dir / f"DG_TWFD_edm_baseline_grid_steps{'_'.join(str(step) for step in steps)}.png"
    build_grid(
        step_dirs=step_dirs,
        steps=steps,
        seeds=grid_seeds,
        out_path=grid_path,
        subdirs=subdirs,
        cell_size=int(args.grid_cell_size),
    )
    metrics["grid_path"] = str(grid_path)
    write_json(outdir / "metrics_edm_baseline_sweep.json", metrics)

    md_path, csv_path, log_path = write_sweep_summaries(metrics=metrics, outdir=outdir, results_dir=results_dir)
    print(f"\nWrote sweep log: {log_path}")
    print(f"Wrote markdown summary: {md_path}")
    print(f"Wrote CSV summary: {csv_path}")
    print(f"Wrote grid: {grid_path}")
    print(f"Wrote run metrics: {outdir / 'metrics_edm_baseline_sweep.json'}")


if __name__ == "__main__":
    main()
