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
    derive_warp_weights_from_defect,
    edm_root_from_file,
    load_edm_network,
    load_json,
    plot_defect_comparison,
    plot_defect_profile,
    plot_trajectory_2d,
    resolve_path,
    sample_trajectories,
    save_schedule_csv,
    save_trajectory,
    schedule_for_time_param,
    select_trajectory_index,
    write_defect_csv,
    write_json,
    write_summary_csv,
)


def format_value(value) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return ""


def write_summary_md(
    path: Path,
    *,
    summaries: list[dict],
    figure_dir: Path,
    result_dir: Path,
    config_snapshot: dict,
    selected_seed: int | None = None,
    selected_index: int | None = None,
    selection_mode: str | None = None,
    label_sections: int = 8,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_name = {row["time_param"]: row for row in summaries}
    identity = by_name.get("identity", {})
    warp = by_name.get("dg_twfd_warp", {})
    ratio_delta = None
    if identity and warp:
        ratio_delta = float(warp["defect_uniformity_ratio"]) - float(identity["defect_uniformity_ratio"])

    with path.open("w", encoding="utf-8") as f:
        f.write("# DG-TWFD Timewarp Defect Analysis\n\n")
        f.write("This is a teacher-side EDM trajectory analysis; no student model is trained.\n\n")
        f.write("## Paper-Ready Setup Summary\n\n")
        f.write(
            f"- Dataset: `{config_snapshot['dataset']}`; checkpoint: `{config_snapshot['checkpoint']}`.\n"
        )
        f.write(
            f"- Trajectories: `{config_snapshot['num_trajectories']}` seeds starting from `{config_snapshot['first_seed']}`; "
            f"default analysis uses `{config_snapshot['num_steps']}` intervals over sigma range "
            f"`[{config_snapshot['sigma_max']}, {config_snapshot['sigma_min']}]`.\n"
        )
        f.write(
            f"- `identity`: linear sigma spacing; `dg_twfd_warp`: EDM rho schedule with `rho={config_snapshot['rho']}` "
            "followed by defect-derived monotone reallocation.\n"
        )
        f.write(
            "- Defect is the midpoint composition defect under deterministic EDM Heun transitions, normalized by local update magnitude.\n\n"
        )
        f.write("| time_param | mean defect | std defect | max defect | min defect | std/mean |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summaries:
            f.write(
                f"| {row['time_param']} | {format_value(row['mean_defect'])} | "
                f"{format_value(row['std_defect'])} | {format_value(row['max_defect'])} | "
                f"{format_value(row['min_defect'])} | {format_value(row['defect_uniformity_ratio'])} |\n"
            )
        f.write("\n")
        if ratio_delta is not None:
            direction = "more uniform" if ratio_delta < 0 else "less uniform"
            f.write(
                f"`dg_twfd_warp` is {direction} by std/mean: "
                f"{format_value(warp.get('defect_uniformity_ratio'))} vs "
                f"{format_value(identity.get('defect_uniformity_ratio'))} for identity.\n\n"
            )
        if selected_index is not None and selected_seed is not None:
            f.write(
                "The trajectory heatmaps use the same fixed seed under both schedules. "
                f"Selected trajectory index: `{selected_index}` (seed `{selected_seed}`), "
                f"selection mode: `{selection_mode}`.\n\n"
            )
        f.write(
            f"Time-point labels on the trajectory are plotted at the endpoints of `{label_sections}` coarse sections. "
            "For the default 64-bin analysis this yields labels at `0, 8, 16, ..., 64`, which matches 8 equal sections "
            "over 64 interval bins rather than the previous 7-gap sparse heuristic.\n\n"
        )
        f.write("## Figures\n\n")
        for name in [
            "trajectory_identity.png",
            "trajectory_dg_twfd_warp.png",
            "defect_profile_identity.png",
            "defect_profile_dg_twfd_warp.png",
            "defect_profile_comparison.png",
        ]:
            f.write(f"- `{figure_dir / name}`\n")
        f.write("\n## Data\n\n")
        for name in ["defect_identity.csv", "defect_dg_twfd_warp.csv", "defect_summary.csv"]:
            f.write(f"- `{result_dir / name}`\n")
        f.write(f"- `{result_dir / 'trajectory_figure_manifest.json'}`\n")


def build_config_snapshot(cfg: dict, *, num_steps: int, num_trajectories: int, seed: int) -> dict:
    return {
        "dataset": cfg["dataset"],
        "checkpoint": cfg["checkpoint"],
        "num_steps": num_steps,
        "num_trajectories": num_trajectories,
        "first_seed": seed,
        "sigma_min": float(cfg["sigma_min"]),
        "sigma_max": float(cfg["sigma_max"]),
        "rho": float(cfg["rho"]),
        "defect_eps": float(cfg["defect_eps"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare identity and DG-TWFD warped EDM trajectory defect profiles.")
    parser.add_argument("--config", required=True, help="Path to DG_TWFD timewarp analysis config.")
    parser.add_argument("--outdir", default=None, help="Output directory for trajectories and intermediate files.")
    parser.add_argument("--num-steps", type=int, default=None, help="Number of sampling intervals.")
    parser.add_argument("--num-trajectories", type=int, default=None, help="Number of trajectories used for defect stats.")
    parser.add_argument("--seed", type=int, default=None, help="First trajectory seed.")
    parser.add_argument("--batch", type=int, default=None, help="Trajectory sampling batch size.")
    parser.add_argument("--defect-batch", type=int, default=None, help="Defect evaluation batch size.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 network execution.")
    parser.add_argument("--warp-power", type=float, default=None, help="Power applied to identity defect when deriving warp weights.")
    parser.add_argument("--warp-floor", type=float, default=None, help="Minimum weight floor when deriving warp weights.")
    parser.add_argument(
        "--trajectory-select",
        choices=["first", "max_schedule_gap"],
        default="max_schedule_gap",
        help="Rule used to choose the visualized trajectory from the fixed seed pool.",
    )
    parser.add_argument(
        "--label-sections",
        type=int,
        default=8,
        help="Number of coarse sections used when labeling trajectory time points.",
    )
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
    defect_batch = int(args.defect_batch if args.defect_batch is not None else cfg["defect_batch"])
    outdir = resolve_path(args.outdir, root=edm_root) if args.outdir else resolve_path(cfg["outdir"], root=edm_root)
    result_dir = EXP_ROOT / "results"
    figure_dir = EXP_ROOT / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    net = load_edm_network(cfg["checkpoint"], device=device, use_fp16=bool(args.fp16))
    seeds = list(range(seed, seed + num_trajectories))
    config_snapshot = build_config_snapshot(cfg, num_steps=num_steps, num_trajectories=num_trajectories, seed=seed)
    summaries: list[dict] = []
    trajectories: dict[str, Path] = {}
    defect_csvs: dict[str, Path] = {}

    identity_dir = outdir / "identity"
    identity_dir.mkdir(parents=True, exist_ok=True)
    identity_sigmas, identity_param = schedule_for_time_param(
        net,
        time_param="identity",
        num_steps=num_steps,
        sigma_min=float(cfg["sigma_min"]),
        sigma_max=float(cfg["sigma_max"]),
        rho=float(cfg["rho"]),
        device=device,
    )
    identity_states, identity_labels = sample_trajectories(
        net,
        sigmas=identity_sigmas,
        seeds=seeds,
        batch_size=batch,
        class_idx=cfg.get("class_idx"),
        device=device,
    )
    identity_meta = {
        "dataset": cfg["dataset"],
        "checkpoint": cfg["checkpoint"],
        "time_param": "identity",
        "num_steps": num_steps,
        "num_trajectories": num_trajectories,
        "seed": seed,
        "tau": identity_param.tau.tolist(),
        "warp_weights": identity_param.weights.tolist(),
    }
    identity_traj = identity_dir / "trajectory.pt"
    save_trajectory(
        identity_traj,
        states=identity_states,
        sigmas=identity_sigmas,
        labels=identity_labels,
        seeds=seeds,
        metadata=identity_meta,
    )
    save_schedule_csv(identity_dir / "schedule.csv", sigmas=identity_sigmas.detach().cpu().numpy(), param=identity_param)
    identity_rows, identity_summary = compute_defect_rows(
        net,
        trajectory={"states": identity_states, "sigmas": identity_sigmas.cpu(), "class_labels": identity_labels, "metadata": identity_meta},
        batch_size=defect_batch,
        defect_eps=float(cfg["defect_eps"]),
        device=device,
    )
    identity_csv = result_dir / "defect_identity.csv"
    write_defect_csv(identity_csv, identity_rows)
    summaries.append(identity_summary)
    trajectories["identity"] = identity_traj
    defect_csvs["identity"] = identity_csv

    warp_weights = derive_warp_weights_from_defect(
        identity_rows,
        floor=float(args.warp_floor if args.warp_floor is not None else cfg.get("warp_floor", 0.05)),
        power=float(args.warp_power if args.warp_power is not None else cfg.get("warp_from_defect_power", 1.0)),
    )
    write_json(outdir / "dg_twfd_warp" / "warp_weights.json", {"weights": warp_weights, "source": "identity_defect"})

    warp_dir = outdir / "dg_twfd_warp"
    warp_dir.mkdir(parents=True, exist_ok=True)
    warp_sigmas, warp_param = schedule_for_time_param(
        net,
        time_param="dg_twfd_warp",
        num_steps=num_steps,
        sigma_min=float(cfg["sigma_min"]),
        sigma_max=float(cfg["sigma_max"]),
        rho=float(cfg["rho"]),
        device=device,
        weights=warp_weights,
    )
    warp_states, warp_labels = sample_trajectories(
        net,
        sigmas=warp_sigmas,
        seeds=seeds,
        batch_size=batch,
        class_idx=cfg.get("class_idx"),
        device=device,
    )
    warp_meta = {
        "dataset": cfg["dataset"],
        "checkpoint": cfg["checkpoint"],
        "time_param": "dg_twfd_warp",
        "num_steps": num_steps,
        "num_trajectories": num_trajectories,
        "seed": seed,
        "tau": warp_param.tau.tolist(),
        "warp_weights": warp_param.weights.tolist(),
    }
    warp_traj = warp_dir / "trajectory.pt"
    save_trajectory(warp_traj, states=warp_states, sigmas=warp_sigmas, labels=warp_labels, seeds=seeds, metadata=warp_meta)
    save_schedule_csv(warp_dir / "schedule.csv", sigmas=warp_sigmas.detach().cpu().numpy(), param=warp_param)
    warp_rows, warp_summary = compute_defect_rows(
        net,
        trajectory={"states": warp_states, "sigmas": warp_sigmas.cpu(), "class_labels": warp_labels, "metadata": warp_meta},
        batch_size=defect_batch,
        defect_eps=float(cfg["defect_eps"]),
        device=device,
    )
    warp_csv = result_dir / "defect_dg_twfd_warp.csv"
    write_defect_csv(warp_csv, warp_rows)
    summaries.append(warp_summary)
    trajectories["dg_twfd_warp"] = warp_traj
    defect_csvs["dg_twfd_warp"] = warp_csv

    write_summary_csv(result_dir / "defect_summary.csv", summaries)
    selected_index, selection_meta = select_trajectory_index(
        identity_states=identity_states,
        warp_states=warp_states,
        mode=args.trajectory_select,
    )
    selected_seed = seeds[selected_index]
    write_json(
        result_dir / "defect_summary.json",
        {
            "summaries": summaries,
            "selected_trajectory_index": selected_index,
            "selected_seed": selected_seed,
            "label_sections": args.label_sections,
            "config_snapshot": config_snapshot,
            **selection_meta,
        },
    )
    write_json(
        result_dir / "trajectory_figure_manifest.json",
        {
            "dataset": cfg["dataset"],
            "checkpoint": cfg["checkpoint"],
            "num_steps": num_steps,
            "num_trajectories": num_trajectories,
            "seed_range": [seeds[0], seeds[-1]],
            "selected_trajectory_index": selected_index,
            "selected_seed": selected_seed,
            "selection_mode": args.trajectory_select,
            "label_sections": args.label_sections,
            "sigma_min": float(cfg["sigma_min"]),
            "sigma_max": float(cfg["sigma_max"]),
            "rho": float(cfg["rho"]),
            "defect_eps": float(cfg["defect_eps"]),
            "figures": [
                {
                    "figure_id": "trajectory_identity",
                    "time_param": "identity",
                    "path": str(figure_dir / "trajectory_identity.png"),
                    "schedule_description": "linear sigma spacing from sigma_max to sigma_min, followed by the final jump to 0",
                    "paper_note": (
                        "Teacher-side EDM trajectory under linear sigma spacing. "
                        "The selected seed is shared with the warped trajectory so that differences reflect only time allocation."
                    ),
                },
                {
                    "figure_id": "trajectory_dg_twfd_warp",
                    "time_param": "dg_twfd_warp",
                    "path": str(figure_dir / "trajectory_dg_twfd_warp.png"),
                    "schedule_description": "EDM rho schedule followed by a monotone defect-derived warp",
                    "paper_note": (
                        "Teacher-side EDM trajectory under rho-based DG-TWFD-style time reparameterization. "
                        "Higher-defect intervals receive more resolution while the seed and checkpoint stay fixed."
                    ),
                },
            ],
        },
    )

    plot_trajectory_2d(
        trajectory_path=trajectories["identity"],
        defect_csv=defect_csvs["identity"],
        out_png=figure_dir / "trajectory_identity.png",
        out_pdf=figure_dir / "trajectory_identity.pdf",
        trajectory_index=selected_index,
        label_sections=args.label_sections,
        title="identity trajectory with defect heat",
    )
    plot_trajectory_2d(
        trajectory_path=trajectories["dg_twfd_warp"],
        defect_csv=defect_csvs["dg_twfd_warp"],
        out_png=figure_dir / "trajectory_dg_twfd_warp.png",
        out_pdf=figure_dir / "trajectory_dg_twfd_warp.pdf",
        trajectory_index=selected_index,
        label_sections=args.label_sections,
        title="DG-TWFD warped trajectory with defect heat",
    )
    plot_defect_profile(
        defect_csv=defect_csvs["identity"],
        out_png=figure_dir / "defect_profile_identity.png",
        out_pdf=figure_dir / "defect_profile_identity.pdf",
        title="identity interval defect",
    )
    plot_defect_profile(
        defect_csv=defect_csvs["dg_twfd_warp"],
        out_png=figure_dir / "defect_profile_dg_twfd_warp.png",
        out_pdf=figure_dir / "defect_profile_dg_twfd_warp.pdf",
        title="DG-TWFD warped interval defect",
    )
    plot_defect_comparison(
        identity_csv=defect_csvs["identity"],
        warp_csv=defect_csvs["dg_twfd_warp"],
        out_png=figure_dir / "defect_profile_comparison.png",
        out_pdf=figure_dir / "defect_profile_comparison.pdf",
    )
    write_summary_md(
        result_dir / "summary.md",
        summaries=summaries,
        figure_dir=figure_dir,
        result_dir=result_dir,
        config_snapshot=config_snapshot,
        selected_seed=selected_seed,
        selected_index=selected_index,
        selection_mode=args.trajectory_select,
        label_sections=args.label_sections,
    )

    print(f"Wrote identity trajectory: {trajectories['identity']}")
    print(f"Wrote warped trajectory: {trajectories['dg_twfd_warp']}")
    print(
        "Selected visualization trajectory: "
        f"index={selected_index}, seed={selected_seed}, mode={args.trajectory_select}"
    )
    print(f"Wrote summary: {result_dir / 'summary.md'}")
    print(f"Wrote figures under: {figure_dir}")


if __name__ == "__main__":
    main()
