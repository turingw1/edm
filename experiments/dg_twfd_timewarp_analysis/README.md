# DG-TWFD Time-Warp Analysis on EDM

This experiment analyzes how time parameterization changes the distribution of
local composition defect along pretrained EDM trajectories. It does not train a
student model. The goal is to produce teacher-side evidence for the DG-TWFD
claim that high-defect time regions should receive more sampling resolution.

## Time Parameterizations

The scripts support:

- `identity`: the original EDM sigma schedule.
- `dg_twfd_warp`: a monotone warp over the same sigma range. In the one-click
  comparison script, the warp is derived from the identity defect profile: high
  defect intervals get larger CDF mass and therefore more sampling resolution.

The interface is implemented in `utils/timewarp_core.py` through
`TimeParameterization`:

```text
forward(sigma_original) -> tau
inverse(tau) -> sigma_original
sample_steps(num_steps)
map_schedule(original_t_steps)
```

`dg_twfd_warp` is intentionally simple and non-trained. It is a configurable
monotone proxy for the learned time redistribution used by DG-TWFD.

## Defect

For three trajectory times `t_i > t_j > t_k`, the interval-wise proxy defect is:

```text
defect(i,j,k) =
  || Phi(t_i -> t_k, x_i) - Phi(t_j -> t_k, Phi(t_i -> t_j, x_i)) ||_mse^2
  / (eps + || Phi(t_i -> t_k, x_i) - x_i ||_mse^2)
```

`Phi` is EDM online deterministic Heun rollout using the pretrained checkpoint.
The default profile uses adjacent triples `(i, i+1, i+2)`, assigning each value
to an interval bin. The summary reports `std / mean` as
`defect_uniformity_ratio`; lower values mean the defect profile is more uniform.

## 2D Trajectory Visualization

The visualization flattens saved trajectory states, runs PCA to two dimensions,
and plots the trajectory curve. Edges are colored by local defect heat, while
points are colored by trajectory index. This gives a compact view of where the
ODE path is locally less reusable.

## Setup

Run from the EDM repository root:

```bash
cd /data2/yl7622/Zhengwei/DG-TWFD/refs/edm
conda activate dg
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

The first run needs access to the public EDM checkpoint:

```text
https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
```

Plotting requires `matplotlib`. If needed:

```bash
python -m pip install matplotlib
```

## Commands

### 1. Identity Sampling And Analysis

```bash
python experiments/dg_twfd_timewarp_analysis/scripts/run_timewarp_sampling.py \
  --config experiments/dg_twfd_timewarp_analysis/configs/DG_TWFD_cifar10_timewarp_analysis.json \
  --time-param identity

python experiments/dg_twfd_timewarp_analysis/scripts/compute_defect_profile.py \
  --config experiments/dg_twfd_timewarp_analysis/configs/DG_TWFD_cifar10_timewarp_analysis.json \
  --trajectory experiments/dg_twfd_timewarp_analysis/outputs/DG_TWFD_cifar10_timewarp_analysis/identity/trajectory.pt

python experiments/dg_twfd_timewarp_analysis/scripts/plot_trajectory_2d.py \
  --trajectory experiments/dg_twfd_timewarp_analysis/outputs/DG_TWFD_cifar10_timewarp_analysis/identity/trajectory.pt \
  --defect-csv experiments/dg_twfd_timewarp_analysis/results/defect_identity.csv
```

### 2. DG-TWFD Warp Sampling And Analysis

Standalone warped sampling uses the configurable monotone default warp:

```bash
python experiments/dg_twfd_timewarp_analysis/scripts/run_timewarp_sampling.py \
  --config experiments/dg_twfd_timewarp_analysis/configs/DG_TWFD_cifar10_timewarp_analysis.json \
  --time-param dg_twfd_warp

python experiments/dg_twfd_timewarp_analysis/scripts/compute_defect_profile.py \
  --config experiments/dg_twfd_timewarp_analysis/configs/DG_TWFD_cifar10_timewarp_analysis.json \
  --trajectory experiments/dg_twfd_timewarp_analysis/outputs/DG_TWFD_cifar10_timewarp_analysis/dg_twfd_warp/trajectory.pt

python experiments/dg_twfd_timewarp_analysis/scripts/plot_trajectory_2d.py \
  --trajectory experiments/dg_twfd_timewarp_analysis/outputs/DG_TWFD_cifar10_timewarp_analysis/dg_twfd_warp/trajectory.pt \
  --defect-csv experiments/dg_twfd_timewarp_analysis/results/defect_dg_twfd_warp.csv
```

### 3. One-Click Identity Vs Warp Comparison

This is the recommended command. It samples identity trajectories, computes the
identity defect profile, derives a DG-TWFD style warp from that profile, then
resamples and summarizes both settings.

```bash
python experiments/dg_twfd_timewarp_analysis/scripts/compare_identity_vs_warp.py \
  --config experiments/dg_twfd_timewarp_analysis/configs/DG_TWFD_cifar10_timewarp_analysis.json
```

### 4. FID Sweep For Different Schedules

To compare sample quality under the two time parameterizations at
`16/32/48/64` steps:

```bash
python experiments/dg_twfd_timewarp_analysis/scripts/run_timewarp_fid_sweep.py \
  --config experiments/dg_twfd_timewarp_analysis/configs/DG_TWFD_cifar10_timewarp_analysis.json \
  --time-params identity,dg_twfd_warp \
  --steps 16,32,48,64 \
  --num-samples 5000 \
  --batch 512 \
  --fid-batch 512
```

For paper-scale FID, increase `--num-samples` to `50000`. The script writes one
sample folder and one schedule CSV per `(time_param, steps)` setting. It runs
FP32 by default; use `--fp16` only for explicit precision sensitivity checks.

For a faster smoke run:

```bash
python experiments/dg_twfd_timewarp_analysis/scripts/compare_identity_vs_warp.py \
  --config experiments/dg_twfd_timewarp_analysis/configs/DG_TWFD_cifar10_timewarp_analysis.json \
  --num-steps 8 \
  --num-trajectories 4 \
  --batch 4 \
  --defect-batch 4
```

## Outputs

Figures:

```text
experiments/dg_twfd_timewarp_analysis/figures/trajectory_identity.png
experiments/dg_twfd_timewarp_analysis/figures/trajectory_dg_twfd_warp.png
experiments/dg_twfd_timewarp_analysis/figures/defect_profile_identity.png
experiments/dg_twfd_timewarp_analysis/figures/defect_profile_dg_twfd_warp.png
experiments/dg_twfd_timewarp_analysis/figures/defect_profile_comparison.png
```

Tables and summaries:

```text
experiments/dg_twfd_timewarp_analysis/results/defect_identity.csv
experiments/dg_twfd_timewarp_analysis/results/defect_dg_twfd_warp.csv
experiments/dg_twfd_timewarp_analysis/results/defect_summary.csv
experiments/dg_twfd_timewarp_analysis/results/summary.md
experiments/dg_twfd_timewarp_analysis/results/fid_schedule_comparison.csv
experiments/dg_twfd_timewarp_analysis/results/fid_schedule_comparison.md
```

Raw trajectories and schedules:

```text
experiments/dg_twfd_timewarp_analysis/outputs/DG_TWFD_cifar10_timewarp_analysis/
experiments/dg_twfd_timewarp_analysis/outputs/DG_TWFD_cifar10_timewarp_analysis/fid_sweep/
```

The paper-facing number is `defect_uniformity_ratio = std / mean` from
`defect_summary.csv`. The intended qualitative result is:

- `identity`: defect is concentrated in fewer intervals.
- `dg_twfd_warp`: high-defect regions are expanded, making the interval profile
  flatter and the `std / mean` ratio smaller.
