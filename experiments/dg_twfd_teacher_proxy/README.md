# DG-TWFD Teacher-Proxy Experiments on EDM

This folder contains DG-TWFD-side experiments that use the official EDM codebase
as the teacher/proxy backbone. The goal is to simulate the quality and
low-step behavior that a future DG-TWFD student should approach, before the
student implementation is strong enough to provide reliable numbers.

## Code Map

- `generate.py`: official EDM sampling CLI. It loads `pickle["ema"]`, samples
  Gaussian latents, optionally samples class labels, and writes PNG images.
- `fid.py`: official FID CLI. It computes Inception statistics for generated
  images and compares them to a reference `.npz`.
- `training/networks.py`: EDM/VP/VE preconditioning wrappers and U-Net
  backbones. CIFAR-10 uses the DDPM++/SongUNet-style backbone in the public VP
  checkpoint; ImageNet64 uses the ADM/Dhariwal-style conditional backbone in
  the public EDM checkpoint.
- `training/loss.py`: original VP/VE/EDM denoising losses. These are useful as
  references but are not modified for the teacher-proxy experiments.

## Current Experimental Entry

Use `run_edm_teacher_proxy.py` for repeatable step-count sweeps:

```bash
python experiments/dg_twfd_teacher_proxy/run_edm_teacher_proxy.py \
  --outdir runs/dg_twfd_teacher_proxy/cifar10_smoke \
  --num-samples 128 \
  --steps 1,2,4,8,16 \
  --batch 64 \
  --fid-batch 64
```

The script writes:

- `samples/steps_<N>/`: generated images for each step count.
- `logs/generate_steps_<N>.log`: sampling logs.
- `logs/fid_steps_<N>.log`: FID logs.
- `metrics.json`: parsed FID values when FID is enabled.
- `run_config.json`: exact command configuration.

For full CIFAR-10 teacher-proxy numbers, set `--num-samples 50000`. Small FID
with fewer samples is only a debugging proxy and must not be reported as final
FID.

## Default CIFAR-10 Public Assets

- Network:
  `https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl`
- FID reference:
  `https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz`

The EDM paper example reports FID 1.79 for this CIFAR-10 checkpoint using
18 EDM steps, which corresponds to 35 denoiser evaluations under Heun sampling.

## Extension Plan

Add one command block per future DG-TWFD experiment here, keeping the original
EDM code paths intact unless the experiment explicitly requires a new sampler.
If a new sampler is needed, add it as a separate script in this folder and
import EDM utilities instead of editing `generate.py` directly.
