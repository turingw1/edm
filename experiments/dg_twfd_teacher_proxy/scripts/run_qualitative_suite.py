#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EDM_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> None:
    print("\n>>>", " ".join(command))
    subprocess.run(command, check=True, cwd=str(EDM_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full DG-TWFD qualitative figure suite in a fixed order.")
    parser.add_argument("--device", default="cuda", help="Torch device passed to child scripts.")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 execution in child scripts.")
    parser.add_argument("--include-diversity", action="store_true", help="Also render the appendix diversity grid.")
    parser.add_argument("--include-galleries", action="store_true", help="Also render dense high-quality galleries.")
    parser.add_argument(
        "--suite",
        default="all",
        choices=["all", "main_only", "imagenet_only", "cifar_only"],
        help="Subset of the figure suite to run.",
    )
    args = parser.parse_args()

    python = sys.executable
    base = [python]
    common = ["--device", args.device]
    if args.fp32:
        common.append("--fp32")

    manifests_dir = "experiments/dg_twfd_teacher_proxy/manifests"
    outputs_cifar = "experiments/dg_twfd_teacher_proxy/outputs/cifar10"
    outputs_im64 = "experiments/dg_twfd_teacher_proxy/outputs/imagenet64"
    figures_main = "experiments/dg_twfd_teacher_proxy/figures/main"
    figures_appendix = "experiments/dg_twfd_teacher_proxy/figures/appendix"

    run_command(
        base
        + [
            "experiments/dg_twfd_teacher_proxy/scripts/prepare_qualitative_manifests.py",
            "--outdir",
            manifests_dir,
        ]
    )

    jobs: list[list[str]] = []

    if args.suite in {"all", "main_only", "imagenet_only"}:
        jobs.extend(
            [
                [
                    "experiments/dg_twfd_teacher_proxy/scripts/render_identity_vs_dgtwfd.py",
                    "--config",
                    "experiments/dg_twfd_teacher_proxy/configs/DG_TWFD_imagenet64_target_ablation.json",
                    "--dataset",
                    "imagenet64",
                    "--figure-id",
                    "DG_TWFD_imagenet64_identity_vs_full",
                    "--steps",
                    "32,64,128,256",
                    "--display-labels",
                    "1,2,4,8",
                    "--sampler-mode",
                    "config",
                    "--manifest",
                    f"{manifests_dir}/DG_TWFD_imagenet64_identity_rows.json",
                    "--output-root",
                    outputs_im64,
                    "--figure-path",
                    f"{figures_main}/DG_TWFD_imagenet64_identity_vs_full.pdf",
                    "--manifest-path",
                    f"{manifests_dir}/DG_TWFD_imagenet64_identity_vs_full.json",
                ],
                [
                    "experiments/dg_twfd_teacher_proxy/scripts/render_identity_vs_dgtwfd.py",
                    "--config",
                    "experiments/dg_twfd_teacher_proxy/configs/DG_TWFD_imagenet64_target_ablation.json",
                    "--dataset",
                    "imagenet64",
                    "--figure-id",
                    "DG_TWFD_imagenet64_identity_vs_full_deterministic",
                    "--steps",
                    "32,64,128,256",
                    "--display-labels",
                    "1,2,4,8",
                    "--sampler-mode",
                    "deterministic",
                    "--manifest",
                    f"{manifests_dir}/DG_TWFD_imagenet64_identity_rows.json",
                    "--output-root",
                    outputs_im64,
                    "--figure-path",
                    f"{figures_main}/DG_TWFD_imagenet64_identity_vs_full_deterministic.pdf",
                    "--manifest-path",
                    f"{manifests_dir}/DG_TWFD_imagenet64_identity_vs_full_deterministic.json",
                ],
                [
                    "experiments/dg_twfd_teacher_proxy/scripts/render_qualitative_grid.py",
                    "--config",
                    "experiments/dg_twfd_teacher_proxy/configs/DG_TWFD_imagenet64_target_ablation.json",
                    "--dataset",
                    "imagenet64",
                    "--figure-id",
                    "DG_TWFD_imagenet64_step_progression",
                    "--method",
                    "dg_twfd",
                    "--steps",
                    "32,64,128,256",
                    "--display-labels",
                    "1,2,4,8",
                    "--sampler-mode",
                    "config",
                    "--manifest",
                    f"{manifests_dir}/DG_TWFD_imagenet64_step_rows.json",
                    "--output-root",
                    outputs_im64,
                    "--figure-path",
                    f"{figures_main}/DG_TWFD_imagenet64_step_progression.pdf",
                    "--manifest-path",
                    f"{manifests_dir}/DG_TWFD_imagenet64_step_progression.json",
                ],
                [
                    "experiments/dg_twfd_teacher_proxy/scripts/render_qualitative_grid.py",
                    "--config",
                    "experiments/dg_twfd_teacher_proxy/configs/DG_TWFD_imagenet64_target_ablation.json",
                    "--dataset",
                    "imagenet64",
                    "--figure-id",
                    "DG_TWFD_imagenet64_step_progression_deterministic",
                    "--method",
                    "dg_twfd",
                    "--steps",
                    "32,64,128,256",
                    "--display-labels",
                    "1,2,4,8",
                    "--sampler-mode",
                    "deterministic",
                    "--manifest",
                    f"{manifests_dir}/DG_TWFD_imagenet64_step_rows.json",
                    "--output-root",
                    outputs_im64,
                    "--figure-path",
                    f"{figures_main}/DG_TWFD_imagenet64_step_progression_deterministic.pdf",
                    "--manifest-path",
                    f"{manifests_dir}/DG_TWFD_imagenet64_step_progression_deterministic.json",
                ],
            ]
        )

    if args.suite in {"all", "main_only", "cifar_only"}:
        jobs.extend(
            [
                [
                    "experiments/dg_twfd_teacher_proxy/scripts/render_identity_vs_dgtwfd.py",
                    "--config",
                    "experiments/dg_twfd_teacher_proxy/configs/DG_TWFD_cifar10_target_ablation.json",
                    "--dataset",
                    "cifar10",
                    "--figure-id",
                    "DG_TWFD_cifar10_identity_vs_full",
                    "--steps",
                    "24,48,72,96",
                    "--display-labels",
                    "1,2,4,8",
                    "--manifest",
                    f"{manifests_dir}/DG_TWFD_cifar10_identity_rows.json",
                    "--output-root",
                    outputs_cifar,
                    "--figure-path",
                    f"{figures_main}/DG_TWFD_cifar10_identity_vs_full.pdf",
                    "--manifest-path",
                    f"{manifests_dir}/DG_TWFD_cifar10_identity_vs_full.json",
                ],
                [
                    "experiments/dg_twfd_teacher_proxy/scripts/render_qualitative_grid.py",
                    "--config",
                    "experiments/dg_twfd_teacher_proxy/configs/DG_TWFD_cifar10_target_ablation.json",
                    "--dataset",
                    "cifar10",
                    "--figure-id",
                    "DG_TWFD_cifar10_step_progression",
                    "--method",
                    "dg_twfd",
                    "--steps",
                    "16,32,48,64",
                    "--display-labels",
                    "1,2,4,8",
                    "--manifest",
                    f"{manifests_dir}/DG_TWFD_cifar10_step_rows.json",
                    "--output-root",
                    outputs_cifar,
                    "--figure-path",
                    f"{figures_main}/DG_TWFD_cifar10_step_progression.pdf",
                    "--manifest-path",
                    f"{manifests_dir}/DG_TWFD_cifar10_step_progression.json",
                ],
            ]
        )

    if args.include_diversity and args.suite in {"all", "cifar_only"}:
        jobs.append(
            [
                "experiments/dg_twfd_teacher_proxy/scripts/render_fixed_step_diversity.py",
                "--config",
                "experiments/dg_twfd_teacher_proxy/configs/DG_TWFD_cifar10_target_ablation.json",
                "--dataset",
                "cifar10",
                "--figure-id",
                "DG_TWFD_fixed_step_diversity",
                "--method",
                "dg_twfd",
                "--steps",
                "64",
                "--display-label",
                "8",
                "--seeds",
                "0-63",
                "--grid-cols",
                "8",
                "--output-root",
                outputs_cifar,
                "--figure-path",
                f"{figures_appendix}/DG_TWFD_fixed_step_diversity.pdf",
                "--manifest-path",
                f"{manifests_dir}/DG_TWFD_fixed_step_diversity.json",
            ]
        )

    if args.include_galleries and args.suite in {"all", "cifar_only"}:
        jobs.append(
            [
                "experiments/dg_twfd_teacher_proxy/scripts/render_dense_gallery.py",
                "--config",
                "experiments/dg_twfd_teacher_proxy/configs/DG_TWFD_cifar10_target_ablation.json",
                "--dataset",
                "cifar10",
                "--figure-id",
                "DG_TWFD_cifar10_dense_gallery",
                "--method",
                "dg_twfd",
                "--steps",
                "96",
                "--display-label",
                "8",
                "--sampler-mode",
                "config",
                "--seeds",
                "2000-2063",
                "--grid-cols",
                "8",
                "--output-root",
                outputs_cifar,
                "--figure-path",
                f"{figures_appendix}/DG_TWFD_cifar10_dense_gallery.pdf",
                "--manifest-path",
                f"{manifests_dir}/DG_TWFD_cifar10_dense_gallery.json",
            ]
        )

    for job in jobs:
        run_command(base + job + common)

    print("\nCompleted qualitative suite.")


if __name__ == "__main__":
    main()
