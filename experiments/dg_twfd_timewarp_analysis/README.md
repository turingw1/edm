# DG-TWFD Time-Warp Analysis on EDM

This experiment is a teacher-side analysis on pretrained EDM trajectories. It
does not train a student model. It is designed to produce two paper-facing
results:

1. A trajectory heatmap showing that time reparameterization redistributes
   sampling points toward high-defect regions and makes interval defect more
   uniform.
2. A FID sweep comparing final sample quality under two schedules at
   `16/32/48/64` steps.

## What The Two Schedules Mean

- `identity`: pure linear sigma spacing from `sigma_max` to `sigma_min`, plus
  the final jump to `0`. This intentionally does not use EDM's `rho` schedule.
- `dg_twfd_warp`: EDM's `rho=7` schedule by default. In the one-click defect
  comparison, this schedule is further warped by weights derived from the
  identity defect profile, so high-defect intervals receive more resolution.

The implementation lives in `utils/timewarp_core.py`. The `TimeParameterization`
interface exposes:

```text
forward(sigma_original) -> tau
inverse(tau) -> sigma_original
sample_steps(num_steps)
map_schedule(original_t_steps)
```

The default config uses `num_steps=64`, so the defect analysis produces exactly
64 interval bins.

## Defect Definition

For each adjacent interval `[t_i, t_{i+1}]`, the analysis sets the midpoint
`t_j = (t_i + t_{i+1}) / 2` and computes:

```text
defect(i,j,k) =
  || Phi(t_i -> t_k, x_i) - Phi(t_j -> t_k, Phi(t_i -> t_j, x_i)) ||_mse^2
  / (eps + || Phi(t_i -> t_k, x_i) - x_i ||_mse^2)
```

Here `Phi` is an online deterministic EDM Heun rollout using the pretrained
checkpoint. The summary metric is:

```text
defect_uniformity_ratio = std(defect_bins) / mean(defect_bins)
```

Lower is more uniform.

## Visualization

The trajectory figure flattens saved states and projects them to 2D with PCA.
The path is colored by local defect heat. The plotting code now labels the
endpoints of `8` coarse sections over the default `64` bins, so the marked
time-point indices are `0, 8, 16, ..., 64`. This matches `8` equal sections of
the defect profile rather than the older sparse `8`-point heuristic that
visually looked like `7` gaps.

The intended visual story is:

- `identity`: linear sigma points allocate steps poorly; defect heat is uneven.
- `dg_twfd_warp`: rho schedule plus defect-derived warp places more points near
  difficult intervals; defect heat should look more balanced.

The comparison script does not simply visualize trajectory `0`. It selects one
trajectory deterministically from the fixed seed pool by maximizing the average
state-space gap between the identity and warped schedules for the same seed.
This keeps the selection reproducible while making the schedule difference more
visible in the paper figure.

## Setup

Run from the EDM root:

```bash
cd /data2/yl7622/Zhengwei/DG-TWFD/refs/edm
conda activate dg
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

The first run downloads the public CIFAR-10 EDM checkpoint and FID reference.
Plotting requires `matplotlib`:

```bash
python -m pip install matplotlib
```

## Command 1: Generate Identity Trajectory

Purpose: save a trajectory under pure linear sigma spacing. This is the control
showing how defect is distributed before EDM/DG-TWFD time allocation.

```bash
python experiments/dg_twfd_timewarp_analysis/scripts/run_timewarp_sampling.py \
  --config experiments/dg_twfd_timewarp_analysis/configs/DG_TWFD_cifar10_timewarp_analysis.json \
  --time-param identity
```

Outputs:

```text
outputs/DG_TWFD_cifar10_timewarp_analysis/identity/trajectory.pt
outputs/DG_TWFD_cifar10_timewarp_analysis/identity/schedule.csv
```

## Command 2: Compute Identity Defect

Purpose: compute the 64-bin interval defect profile on the identity trajectory.
This file is also used to derive DG-TWFD warp weights.

```bash
python experiments/dg_twfd_timewarp_analysis/scripts/compute_defect_profile.py \
  --config experiments/dg_twfd_timewarp_analysis/configs/DG_TWFD_cifar10_timewarp_analysis.json \
  --trajectory experiments/dg_twfd_timewarp_analysis/outputs/DG_TWFD_cifar10_timewarp_analysis/identity/trajectory.pt
```

Output:

```text
results/defect_identity.csv
results/defect_identity_summary.json
```

## Command 3: Generate Warped Trajectory

Purpose: save a trajectory under the DG-TWFD schedule. Standalone mode uses the
configurable monotone/rho schedule. For the full defect-derived warp, prefer the
one-click comparison command below.

```bash
python experiments/dg_twfd_timewarp_analysis/scripts/run_timewarp_sampling.py \
  --config experiments/dg_twfd_timewarp_analysis/configs/DG_TWFD_cifar10_timewarp_analysis.json \
  --time-param dg_twfd_warp
```

Outputs:

```text
outputs/DG_TWFD_cifar10_timewarp_analysis/dg_twfd_warp/trajectory.pt
outputs/DG_TWFD_cifar10_timewarp_analysis/dg_twfd_warp/schedule.csv
```

## Command 4: One-Click Defect And Heatmap Comparison

Purpose: this is the main command for the first paper-facing result. It runs
identity, computes identity defect, derives DG-TWFD warp weights, runs the
warped trajectory, computes warped defect, and writes all figures/tables.

```bash
python experiments/dg_twfd_timewarp_analysis/scripts/compare_identity_vs_warp.py \
  --config experiments/dg_twfd_timewarp_analysis/configs/DG_TWFD_cifar10_timewarp_analysis.json
```

Smoke run:

```bash
python experiments/dg_twfd_timewarp_analysis/scripts/compare_identity_vs_warp.py \
  --config experiments/dg_twfd_timewarp_analysis/configs/DG_TWFD_cifar10_timewarp_analysis.json \
  --num-steps 16 \
  --num-trajectories 4 \
  --batch 4 \
  --defect-batch 4
```

Paper-facing outputs:

```text
figures/trajectory_identity.png
figures/trajectory_dg_twfd_warp.png
figures/defect_profile_identity.png
figures/defect_profile_dg_twfd_warp.png
figures/defect_profile_comparison.png
results/defect_identity.csv
results/defect_dg_twfd_warp.csv
results/defect_summary.csv
results/summary.md
results/trajectory_figure_manifest.json
```

Use this set to argue: after time parameterization, defect is more uniform and
sampling points are allocated more reasonably along the trajectory.

`trajectory_figure_manifest.json` stores the figure-facing configuration details
for the selected trajectory, including the selected seed, sigma range, `rho`,
label-section setting, and short paper-ready notes for the identity and warped
trajectory figures.

## Command 5: FID Sweep For Final Samples

Purpose: this is the second paper-facing result. It compares final sample
quality under the two schedule families at `16/32/48/64` steps.

```bash
python experiments/dg_twfd_timewarp_analysis/scripts/run_timewarp_fid_sweep.py \
  --config experiments/dg_twfd_timewarp_analysis/configs/DG_TWFD_cifar10_timewarp_analysis.json \
  --time-params identity,dg_twfd_warp \
  --steps 16,32,48,64 \
  --num-samples 5000 \
  --batch 512 \
  --fid-batch 512
```

For paper-scale FID:

```bash
python experiments/dg_twfd_timewarp_analysis/scripts/run_timewarp_fid_sweep.py \
  --config experiments/dg_twfd_timewarp_analysis/configs/DG_TWFD_cifar10_timewarp_analysis.json \
  --time-params identity,dg_twfd_warp \
  --steps 16,32,48,64 \
  --num-samples 50000 \
  --batch 512 \
  --fid-batch 512
```

Outputs:

```text
results/fid_schedule_comparison.csv
results/fid_schedule_comparison.md
outputs/DG_TWFD_cifar10_timewarp_analysis/fid_sweep/samples/
outputs/DG_TWFD_cifar10_timewarp_analysis/fid_sweep/schedules/
outputs/DG_TWFD_cifar10_timewarp_analysis/fid_sweep/logs/
```

In this FID table, `identity` means linear sigma spacing and `dg_twfd_warp`
means EDM rho spacing unless you explicitly pass custom warp weights in code.

## Expected Paper Use

Use `results/summary.md` and `results/defect_summary.csv` for the mechanism
claim:

```text
time parameterization -> more uniform defect -> more reasonable sampling ratio
```

Use `results/fid_schedule_comparison.csv` for the sample-quality comparison:

```text
identity vs dg_twfd_warp at 16/32/48/64 steps
```
