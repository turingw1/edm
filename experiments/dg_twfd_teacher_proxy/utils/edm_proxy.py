from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import PIL.Image
import torch
import torch.nn.functional as F


TARGETS = ("velocity", "residual", "endpoint")


class StackedRandomGenerator:
    """Per-sample random generator copied from EDM's generation semantics."""

    def __init__(self, device: torch.device, seeds: Iterable[int]) -> None:
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, x: torch.Tensor) -> torch.Tensor:
        return self.randn(x.shape, dtype=x.dtype, layout=x.layout, device=x.device)

    def randint(self, high: int, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(high, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


def edm_root_from_file(file_path: str | Path) -> Path:
    path = Path(file_path).resolve()
    for parent in path.parents:
        if (parent / "generate.py").is_file() and (parent / "fid.py").is_file():
            return parent
    raise RuntimeError(f"Could not locate EDM root from {file_path}")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_path(path: str | Path, *, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def load_edm_network(network_pkl: str, *, device: torch.device, use_fp16: bool = False):
    import dnnlib

    print(f'Loading EDM network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)["ema"].to(device)
    if hasattr(net, "use_fp16") and device.type == "cuda":
        net.use_fp16 = bool(use_fp16)
    net.eval().requires_grad_(False)
    return net


def edm_sigma_schedule(
    net,
    *,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    sigma_min = max(float(sigma_min), float(net.sigma_min))
    sigma_max = min(float(sigma_max), float(net.sigma_max))
    if num_steps == 1:
        t_steps = torch.as_tensor([sigma_max], dtype=torch.float64, device=device)
    else:
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (
            sigma_max ** (1.0 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
        ) ** rho
    return torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])


def make_class_labels(net, rnd: StackedRandomGenerator, batch_size: int, device: torch.device, class_idx: int | None):
    if not getattr(net, "label_dim", 0):
        return None
    class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
    if class_idx is not None:
        class_labels[:, :] = 0
        class_labels[:, int(class_idx)] = 1
    return class_labels


def _append_dims(value: torch.Tensor, ndim: int) -> torch.Tensor:
    return value.reshape(-1, *([1] * (ndim - 1)))


@torch.no_grad()
def reference_transition(net, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor, class_labels=None) -> torch.Tensor:
    """Deterministic EDM Heun transition from sigma t to sigma s."""

    x_t = x_t.to(torch.float32)
    t = torch.as_tensor(t, dtype=x_t.dtype, device=x_t.device).reshape(-1)
    s = torch.as_tensor(s, dtype=x_t.dtype, device=x_t.device).reshape(-1)
    if t.numel() == 1:
        t = t.expand(x_t.shape[0])
    if s.numel() == 1:
        s = s.expand(x_t.shape[0])
    if (t <= 0).any():
        raise ValueError("reference_transition requires positive source sigma")
    if (s < 0).any() or (s > t).any():
        raise ValueError("reference_transition expects 0 <= s <= t in EDM sigma time")

    denoised = net(x_t, t, class_labels).to(x_t.dtype)
    d_cur = (x_t - denoised) / _append_dims(t, x_t.ndim)
    x_euler = x_t + _append_dims(s - t, x_t.ndim) * d_cur

    mask = s > 0
    if not mask.any():
        return x_euler

    x_next = x_euler.clone()
    labels_subset = class_labels[mask] if class_labels is not None else None
    denoised_next = net(x_euler[mask], s[mask], labels_subset).to(x_t.dtype)
    d_next = (x_euler[mask] - denoised_next) / _append_dims(s[mask], x_t.ndim)
    x_next[mask] = x_t[mask] + _append_dims(s[mask] - t[mask], x_t.ndim) * (0.5 * d_cur[mask] + 0.5 * d_next)
    return x_next


def approximate_target(z: torch.Tensor, approx_cfg: dict) -> torch.Tensor:
    """Apply the same restricted approximation rule to any target tensor."""

    block_size = int(approx_cfg.get("block_size", 1))
    quant_bits = int(approx_cfg.get("quant_bits", 0))
    clip = float(approx_cfg.get("clip", 0.0))
    y = z
    if block_size > 1:
        if y.shape[-1] % block_size != 0 or y.shape[-2] % block_size != 0:
            raise ValueError(f"Spatial size {tuple(y.shape[-2:])} must be divisible by block_size={block_size}")
        y = F.avg_pool2d(y, kernel_size=block_size, stride=block_size)
        y = F.interpolate(y, size=z.shape[-2:], mode="nearest")
    if quant_bits > 0:
        if clip <= 0:
            raise ValueError("approximation.clip must be positive when quant_bits > 0")
        levels = float(2**quant_bits - 1)
        y = torch.clamp(y, -clip, clip)
        y = torch.round((y + clip) / (2.0 * clip) * levels) / levels * (2.0 * clip) - clip
    return y


def induced_transition_from_reference(
    *,
    target: str,
    x_t: torch.Tensor,
    x_s_ref: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    approx_cfg: dict,
) -> torch.Tensor:
    if target not in TARGETS:
        raise ValueError(f"Unsupported target: {target}")
    t = torch.as_tensor(t, dtype=x_t.dtype, device=x_t.device).reshape(-1)
    s = torch.as_tensor(s, dtype=x_t.dtype, device=x_t.device).reshape(-1)
    if t.numel() == 1:
        t = t.expand(x_t.shape[0])
    if s.numel() == 1:
        s = s.expand(x_t.shape[0])
    dt = torch.clamp(t - s, min=1.0e-12)
    dt_img = _append_dims(dt, x_t.ndim)

    if target == "endpoint":
        return approximate_target(x_s_ref, approx_cfg)
    if target == "residual":
        return x_t + approximate_target(x_s_ref - x_t, approx_cfg)
    velocity = (x_t - x_s_ref) / dt_img
    return x_t - dt_img * approximate_target(velocity, approx_cfg)


@torch.no_grad()
def apply_induced_map(
    net,
    *,
    target: str,
    x_t: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    approx_cfg: dict,
    class_labels=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_s_ref = reference_transition(net, x_t, t, s, class_labels)
    x_s = induced_transition_from_reference(target=target, x_t=x_t, x_s_ref=x_s_ref, t=t, s=s, approx_cfg=approx_cfg)
    return x_s, x_s_ref


def mse_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).square().flatten(1).mean(dim=1)


@torch.no_grad()
def generate_target_samples(
    net,
    *,
    target: str,
    outdir: Path,
    num_samples: int,
    seed: int,
    batch_size: int,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    approx_cfg: dict,
    class_idx: int | None,
    subdirs: bool,
    device: torch.device,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    t_steps = edm_sigma_schedule(
        net,
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
    )
    for start in range(0, num_samples, batch_size):
        cur_batch = min(batch_size, num_samples - start)
        batch_seeds = list(range(seed + start, seed + start + cur_batch))
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([cur_batch, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = make_class_labels(net, rnd, cur_batch, device, class_idx)
        x = latents * t_steps[0].to(latents.dtype)
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            x, _ = apply_induced_map(
                net,
                target=target,
                x_t=x,
                t=t_cur,
                s=t_next,
                approx_cfg=approx_cfg,
                class_labels=class_labels,
            )
        save_image_batch(x, batch_seeds=batch_seeds, outdir=outdir, subdirs=subdirs)


def save_image_batch(images: torch.Tensor, *, batch_seeds: list[int], outdir: Path, subdirs: bool) -> None:
    images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    for seed, image_np in zip(batch_seeds, images_np):
        image_dir = outdir / f"{seed - seed % 1000:06d}" if subdirs else outdir
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{seed:06d}.png"
        if image_np.shape[2] == 1:
            PIL.Image.fromarray(image_np[:, :, 0], "L").save(image_path)
        else:
            PIL.Image.fromarray(image_np, "RGB").save(image_path)


@torch.no_grad()
def evaluate_match_and_defect(
    net,
    *,
    target: str,
    num_triplets: int,
    seed: int,
    batch_size: int,
    transition_grid_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    approx_cfg: dict,
    class_idx: int | None,
    defect_eps: float,
    device: torch.device,
) -> dict[str, float]:
    t_grid = edm_sigma_schedule(
        net,
        num_steps=transition_grid_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
    )
    num_grid = int(t_grid.numel())
    if num_grid < 4:
        raise ValueError("transition_grid_steps must produce at least 4 grid points")

    total_match = 0.0
    total_defect = 0.0
    total_count = 0
    rng = torch.Generator(device).manual_seed(int(seed) + 919)

    for start in range(0, num_triplets, batch_size):
        cur_batch = min(batch_size, num_triplets - start)
        batch_seeds = list(range(seed + 100000 + start, seed + 100000 + start + cur_batch))
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([cur_batch, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = make_class_labels(net, rnd, cur_batch, device, class_idx)

        states = [latents.to(torch.float64) * t_grid[0]]
        for t_cur, t_next in zip(t_grid[:-1], t_grid[1:]):
            states.append(reference_transition(net, states[-1], t_cur, t_next, class_labels))
        state_grid = torch.stack(states, dim=1)

        i_idx = torch.randint(0, num_grid - 2, [cur_batch], generator=rng, device=device)
        j_idx = torch.empty_like(i_idx)
        k_idx = torch.empty_like(i_idx)
        for row in range(cur_batch):
            j_idx[row] = torch.randint(int(i_idx[row].item()) + 1, num_grid - 1, [1], generator=rng, device=device)
            k_idx[row] = torch.randint(int(j_idx[row].item()) + 1, num_grid, [1], generator=rng, device=device)

        batch_rows = torch.arange(cur_batch, device=device)
        x_t = state_grid[batch_rows, i_idx]
        t = t_grid[i_idx]
        s = t_grid[j_idx]
        r = t_grid[k_idx]

        x_s_ref = reference_transition(net, x_t, t, s, class_labels)
        x_s_map = induced_transition_from_reference(
            target=target,
            x_t=x_t,
            x_s_ref=x_s_ref,
            t=t,
            s=s,
            approx_cfg=approx_cfg,
        )
        match = mse_per_sample(x_s_map, x_s_ref)

        x_r_ref = reference_transition(net, x_t, t, r, class_labels)
        direct = induced_transition_from_reference(
            target=target,
            x_t=x_t,
            x_s_ref=x_r_ref,
            t=t,
            s=r,
            approx_cfg=approx_cfg,
        )
        x_r_from_s_ref = reference_transition(net, x_s_map, s, r, class_labels)
        composed = induced_transition_from_reference(
            target=target,
            x_t=x_s_map,
            x_s_ref=x_r_from_s_ref,
            t=s,
            s=r,
            approx_cfg=approx_cfg,
        )
        numerator = mse_per_sample(direct, composed)
        denominator = float(defect_eps) + mse_per_sample(x_r_ref, x_t)
        defect = numerator / denominator

        total_match += float(match.sum().item())
        total_defect += float(defect.sum().item())
        total_count += cur_batch

    return {
        "match_mse": total_match / max(total_count, 1),
        "defect": total_defect / max(total_count, 1),
    }


@torch.no_grad()
def evaluate_targets_match_and_defect(
    net,
    *,
    targets: list[str],
    num_triplets: int,
    seed: int,
    batch_size: int,
    transition_grid_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    approx_cfg: dict,
    class_idx: int | None,
    defect_eps: float,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Evaluate all target spaces using one shared set of EDM reference rollouts."""

    for target in targets:
        if target not in TARGETS:
            raise ValueError(f"Unsupported target: {target}")

    t_grid = edm_sigma_schedule(
        net,
        num_steps=transition_grid_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
    )
    num_grid = int(t_grid.numel())
    if num_grid < 4:
        raise ValueError("transition_grid_steps must produce at least 4 grid points")

    totals = {target: {"match_mse": 0.0, "defect": 0.0} for target in targets}
    total_count = 0
    rng = torch.Generator(device).manual_seed(int(seed) + 919)

    for start in range(0, num_triplets, batch_size):
        cur_batch = min(batch_size, num_triplets - start)
        batch_seeds = list(range(seed + 100000 + start, seed + 100000 + start + cur_batch))
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([cur_batch, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = make_class_labels(net, rnd, cur_batch, device, class_idx)

        states = [latents * t_grid[0].to(latents.dtype)]
        for t_cur, t_next in zip(t_grid[:-1], t_grid[1:]):
            states.append(reference_transition(net, states[-1], t_cur, t_next, class_labels))
        state_grid = torch.stack(states, dim=1)

        i_idx = torch.randint(0, num_grid - 2, [cur_batch], generator=rng, device=device)
        j_idx = torch.empty_like(i_idx)
        k_idx = torch.empty_like(i_idx)
        for row in range(cur_batch):
            j_idx[row] = torch.randint(int(i_idx[row].item()) + 1, num_grid - 1, [1], generator=rng, device=device)
            k_idx[row] = torch.randint(int(j_idx[row].item()) + 1, num_grid, [1], generator=rng, device=device)

        batch_rows = torch.arange(cur_batch, device=device)
        x_t = state_grid[batch_rows, i_idx]
        t = t_grid[i_idx]
        s = t_grid[j_idx]
        r = t_grid[k_idx]

        x_s_ref = reference_transition(net, x_t, t, s, class_labels)
        x_r_ref = reference_transition(net, x_t, t, r, class_labels)
        denominator = float(defect_eps) + mse_per_sample(x_r_ref, x_t)

        for target in targets:
            x_s_map = induced_transition_from_reference(
                target=target,
                x_t=x_t,
                x_s_ref=x_s_ref,
                t=t,
                s=s,
                approx_cfg=approx_cfg,
            )
            direct = induced_transition_from_reference(
                target=target,
                x_t=x_t,
                x_s_ref=x_r_ref,
                t=t,
                s=r,
                approx_cfg=approx_cfg,
            )
            x_r_from_s_ref = reference_transition(net, x_s_map, s, r, class_labels)
            composed = induced_transition_from_reference(
                target=target,
                x_t=x_s_map,
                x_s_ref=x_r_from_s_ref,
                t=s,
                s=r,
                approx_cfg=approx_cfg,
            )
            totals[target]["match_mse"] += float(mse_per_sample(x_s_map, x_s_ref).sum().item())
            totals[target]["defect"] += float((mse_per_sample(direct, composed) / denominator).sum().item())

        total_count += cur_batch

    return {
        target: {
            "match_mse": values["match_mse"] / max(total_count, 1),
            "defect": values["defect"] / max(total_count, 1),
        }
        for target, values in totals.items()
    }


def run_fid(
    *,
    edm_root: Path,
    images: Path,
    ref: str,
    num_samples: int,
    batch_size: int,
    log_path: Path,
    dry_run: bool,
) -> float | None:
    command = [
        sys.executable,
        "fid.py",
        "calc",
        f"--images={images}",
        f"--ref={ref}",
        f"--num={num_samples}",
        f"--batch={batch_size}",
    ]
    lines = run_command(command, cwd=edm_root, log_path=log_path, dry_run=dry_run)
    return parse_fid(lines)


def run_command(command: list[str], *, cwd: Path, log_path: Path, dry_run: bool) -> list[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(command)
    if dry_run:
        print(printable)
        log_path.write_text(printable + "\n", encoding="utf-8")
        return [printable]

    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(printable + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            lines.append(line.rstrip("\n"))
        returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
    return lines


def parse_fid(lines: list[str]) -> float | None:
    for line in reversed(lines):
        text = line.strip()
        if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
            return float(text)
    return None
