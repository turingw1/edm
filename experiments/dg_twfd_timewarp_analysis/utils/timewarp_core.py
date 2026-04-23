from __future__ import annotations

import csv
import json
import os
import pickle
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


TIME_PARAMS = ("identity", "dg_twfd_warp")


class StackedRandomGenerator:
    """Per-sample random generator copied from EDM generation semantics."""

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


@dataclass
class TimeParameterization:
    name: str
    sigmas: np.ndarray
    tau: np.ndarray
    weights: np.ndarray

    def forward(self, sigma: np.ndarray | float) -> np.ndarray:
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        return np.interp(sigma_arr, self.sigmas[::-1], self.tau[::-1])

    def inverse(self, tau: np.ndarray | float) -> np.ndarray:
        tau_arr = np.asarray(tau, dtype=np.float64)
        return np.interp(tau_arr, self.tau, self.sigmas)

    def sample_steps(self, num_steps: int) -> np.ndarray:
        tau_steps = np.linspace(0.0, 1.0, int(num_steps) + 1, dtype=np.float64)
        return self.inverse(tau_steps)

    def map_schedule(self, original_t_steps: np.ndarray) -> np.ndarray:
        return self.inverse(np.linspace(0.0, 1.0, len(original_t_steps), dtype=np.float64))


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


def linear_sigma_schedule(
    net,
    *,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
) -> torch.Tensor:
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    sigma_min = max(float(sigma_min), float(net.sigma_min))
    sigma_max = min(float(sigma_max), float(net.sigma_max))
    if num_steps == 1:
        t_steps = torch.as_tensor([sigma_max], dtype=torch.float64, device=device)
    else:
        t_steps = torch.linspace(sigma_max, sigma_min, steps=num_steps, dtype=torch.float64, device=device)
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
def local_heun_transition(net, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor, class_labels=None) -> torch.Tensor:
    """One deterministic EDM Heun transition from sigma t to sigma s."""

    x_t = x_t.to(torch.float32)
    t = torch.as_tensor(t, dtype=x_t.dtype, device=x_t.device).reshape(-1)
    s = torch.as_tensor(s, dtype=x_t.dtype, device=x_t.device).reshape(-1)
    if t.numel() == 1:
        t = t.expand(x_t.shape[0])
    if s.numel() == 1:
        s = s.expand(x_t.shape[0])
    if (t <= 0).any():
        raise ValueError("local_heun_transition requires positive source sigma")
    if (s < 0).any() or (s > t).any():
        raise ValueError("local_heun_transition expects 0 <= s <= t in EDM sigma time")

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


def default_warp_weights(num_intervals: int, *, strength: float = 3.0, power: float = 1.5) -> list[float]:
    if num_intervals < 1:
        raise ValueError("num_intervals must be positive")
    q = np.linspace(0.0, 1.0, num_intervals, dtype=np.float64)
    weights = 1.0 + float(strength) * np.power(q, float(power))
    return weights.tolist()


def build_time_parameterization(
    *,
    name: str,
    base_sigmas: np.ndarray,
    weights: list[float] | None = None,
    strength: float = 3.0,
    power: float = 1.5,
) -> TimeParameterization:
    if name not in TIME_PARAMS:
        raise ValueError(f"Unsupported time parameterization {name}; expected {TIME_PARAMS}")
    sigmas = np.asarray(base_sigmas, dtype=np.float64)
    if sigmas.ndim != 1 or len(sigmas) < 2:
        raise ValueError("base_sigmas must be a 1D schedule with at least two points")

    num_intervals = len(sigmas) - 1
    if name == "identity":
        interval_weights = np.ones(num_intervals, dtype=np.float64)
    else:
        if weights is None:
            interval_weights = np.asarray(default_warp_weights(num_intervals, strength=strength, power=power), dtype=np.float64)
        else:
            interval_weights = np.asarray(weights, dtype=np.float64)
            if len(interval_weights) == num_intervals - 1:
                interval_weights = np.concatenate([interval_weights, interval_weights[-1:]])
            if len(interval_weights) != num_intervals:
                raise ValueError(f"Expected {num_intervals} warp weights, got {len(interval_weights)}")
            interval_weights = np.maximum(interval_weights, 1.0e-8)

    tau = np.concatenate([[0.0], np.cumsum(interval_weights)])
    tau = tau / tau[-1]
    return TimeParameterization(name=name, sigmas=sigmas, tau=tau, weights=interval_weights)


def schedule_for_time_param(
    net,
    *,
    time_param: str,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
    weights: list[float] | None = None,
    strength: float = 3.0,
    power: float = 1.5,
) -> tuple[torch.Tensor, TimeParameterization]:
    if time_param == "identity":
        base = linear_sigma_schedule(
            net,
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            device=device,
        )
    else:
        base = edm_sigma_schedule(
            net,
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            device=device,
        )
    base_np = base.detach().cpu().numpy()
    param = build_time_parameterization(
        name=time_param,
        base_sigmas=base_np,
        weights=weights,
        strength=strength,
        power=power,
    )
    mapped = base if time_param == "identity" else torch.as_tensor(param.sample_steps(num_steps), dtype=torch.float64, device=device)
    mapped = torch.clamp(mapped, min=0.0)
    mapped[0] = base[0]
    mapped[-1] = 0.0
    return mapped, param


@torch.no_grad()
def sample_trajectories(
    net,
    *,
    sigmas: torch.Tensor,
    seeds: list[int],
    batch_size: int,
    class_idx: int | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    all_states: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for start in range(0, len(seeds), batch_size):
        batch_seeds = seeds[start : start + batch_size]
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([len(batch_seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = make_class_labels(net, rnd, len(batch_seeds), device, class_idx)
        x = latents * sigmas[0].to(latents.dtype)
        states = [x.detach().cpu().to(torch.float32)]
        for t_cur, t_next in zip(sigmas[:-1], sigmas[1:]):
            x = local_heun_transition(net, x, t_cur, t_next, class_labels)
            states.append(x.detach().cpu().to(torch.float32))
        all_states.append(torch.stack(states, dim=1))
        if class_labels is not None:
            all_labels.append(class_labels.detach().cpu().to(torch.float32))
    labels = torch.cat(all_labels, dim=0) if all_labels else None
    return torch.cat(all_states, dim=0), labels


def save_image_batch(images: torch.Tensor, *, batch_seeds: list[int], outdir: Path, subdirs: bool) -> None:
    import PIL.Image

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
def generate_final_samples(
    net,
    *,
    sigmas: torch.Tensor,
    outdir: Path,
    seeds: list[int],
    batch_size: int,
    class_idx: int | None,
    subdirs: bool,
    device: torch.device,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(seeds), batch_size):
        batch_seeds = seeds[start : start + batch_size]
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([len(batch_seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = make_class_labels(net, rnd, len(batch_seeds), device, class_idx)
        x = latents * sigmas[0].to(latents.dtype)
        for t_cur, t_next in zip(sigmas[:-1], sigmas[1:]):
            x = local_heun_transition(net, x, t_cur, t_next, class_labels)
        save_image_batch(x, batch_seeds=batch_seeds, outdir=outdir, subdirs=subdirs)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_command(command: list[str], *, cwd: Path, log_path: Path, env_overrides: dict[str, str] | None = None) -> list[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(command)
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(printable + "\n\n")
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
            log_file.write("Environment overrides:\n")
            for key in sorted(env_overrides):
                log_file.write(f"{key}={env_overrides[key]}\n")
            log_file.write("\n")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
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


def run_fid(
    *,
    edm_root: Path,
    images: Path,
    ref: str,
    num_samples: int,
    batch_size: int,
    log_path: Path,
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
    env_overrides = {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(find_free_port()),
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
    }
    return parse_fid(run_command(command, cwd=edm_root, log_path=log_path, env_overrides=env_overrides))


def write_fid_sweep_tables(*, rows: list[dict], md_path: Path, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_param",
        "num_steps",
        "fid",
        "num_samples",
        "seed",
        "sample_dir",
        "schedule_csv",
        "elapsed_sec",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["time_param"]), []).append(row)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# EDM Time-Parameterization FID Sweep\n\n")
        f.write("| time_param | steps | FID↓ | num samples | elapsed sec |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for time_param in sorted(grouped):
            for row in sorted(grouped[time_param], key=lambda item: int(item["num_steps"])):
                fid = "" if row.get("fid") is None else f"{float(row['fid']):.6g}"
                elapsed = "" if row.get("elapsed_sec") is None else f"{float(row['elapsed_sec']):.6g}"
                f.write(
                    f"| {time_param} | {row['num_steps']} | {fid} | "
                    f"{row['num_samples']} | {elapsed} |\n"
                )
        f.write("\nSource samples and schedules are listed in the CSV file.\n")


def save_schedule_csv(path: str | Path, *, sigmas: np.ndarray, param: TimeParameterization) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "sigma", "tau"])
        writer.writeheader()
        for index, sigma in enumerate(sigmas):
            writer.writerow({"index": index, "sigma": f"{float(sigma):.10g}", "tau": f"{float(param.forward(sigma)):.10g}"})


def save_trajectory(
    path: str | Path,
    *,
    states: torch.Tensor,
    sigmas: torch.Tensor,
    labels: torch.Tensor | None,
    seeds: list[int],
    metadata: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "states": states,
        "sigmas": sigmas.detach().cpu(),
        "class_labels": labels,
        "seeds": seeds,
        "metadata": metadata,
    }
    torch.save(payload, path)


def load_trajectory(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def mse_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).square().flatten(1).mean(dim=1)


@torch.no_grad()
def compute_defect_rows(
    net,
    *,
    trajectory: dict,
    batch_size: int,
    defect_eps: float,
    device: torch.device,
) -> tuple[list[dict], dict]:
    states = trajectory["states"]
    sigmas = trajectory["sigmas"].to(device=device, dtype=torch.float32)
    labels = trajectory.get("class_labels")
    labels = labels.to(device) if labels is not None else None
    metadata = trajectory.get("metadata", {})
    time_param = metadata.get("time_param", "unknown")

    if states.ndim != 5:
        raise ValueError("trajectory states must have shape [N, T, C, H, W]")
    num_samples, num_points = int(states.shape[0]), int(states.shape[1])
    if num_points < 2:
        raise ValueError("at least two trajectory points are required")

    tau = np.asarray(metadata.get("tau", np.linspace(0.0, 1.0, num_points)), dtype=np.float64)
    rows: list[dict] = []
    all_defects: list[float] = []

    for interval_index in range(num_points - 1):
        per_interval: list[float] = []
        denom_values: list[float] = []
        t_i = sigmas[interval_index]
        t_k = sigmas[interval_index + 1]
        t_j = 0.5 * (t_i + t_k)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            x_i = states[start:end, interval_index].to(device)
            batch_labels = labels[start:end] if labels is not None else None
            direct = local_heun_transition(net, x_i, t_i, t_k, batch_labels)
            mid = local_heun_transition(net, x_i, t_i, t_j, batch_labels)
            composed = local_heun_transition(net, mid, t_j, t_k, batch_labels)
            numerator = mse_per_sample(direct, composed)
            denominator = (float(defect_eps) + mse_per_sample(direct, x_i)).clamp_min(float(defect_eps))
            defect = numerator / denominator
            finite = torch.isfinite(defect)
            if finite.any():
                per_interval.extend(defect[finite].detach().cpu().numpy().astype(np.float64).tolist())
                denom_values.extend(denominator[finite].detach().cpu().numpy().astype(np.float64).tolist())

        if per_interval:
            defect_arr = np.asarray(per_interval, dtype=np.float64)
            denom_arr = np.asarray(denom_values, dtype=np.float64)
            defect_mean = float(defect_arr.mean())
            defect_std = float(defect_arr.std())
            denom_mean = float(denom_arr.mean())
        else:
            defect_mean = float("nan")
            defect_std = float("nan")
            denom_mean = float("nan")
        all_defects.append(defect_mean)
        rows.append(
            {
                "time_param": time_param,
                "interval_index": interval_index,
                "source_index": interval_index,
                "mid_index": -1,
                "target_index": interval_index + 1,
                "sigma_start": float(sigmas[interval_index].detach().cpu()),
                "sigma_mid": float(t_j.detach().cpu()),
                "sigma_end": float(sigmas[interval_index + 1].detach().cpu()),
                "tau_start": float(tau[interval_index]),
                "tau_mid": float(0.5 * (tau[interval_index] + tau[interval_index + 1])),
                "tau_end": float(tau[interval_index + 1]),
                "defect_mean": defect_mean,
                "defect_std": defect_std,
                "denominator_mean": denom_mean,
                "num_samples": len(per_interval),
            }
        )

    finite_defects = np.asarray([x for x in all_defects if np.isfinite(x)], dtype=np.float64)
    if finite_defects.size:
        mean_defect = float(finite_defects.mean())
        std_defect = float(finite_defects.std())
        max_defect = float(finite_defects.max())
        min_defect = float(finite_defects.min())
        uniformity = float(std_defect / max(mean_defect, 1.0e-12))
    else:
        mean_defect = std_defect = max_defect = min_defect = uniformity = float("nan")
    summary = {
        "time_param": time_param,
        "num_steps": num_points - 1,
        "num_bins": len(rows),
        "mean_defect": mean_defect,
        "std_defect": std_defect,
        "max_defect": max_defect,
        "min_defect": min_defect,
        "defect_uniformity_ratio": uniformity,
    }
    return rows, summary


def write_defect_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_param",
        "interval_index",
        "source_index",
        "mid_index",
        "target_index",
        "sigma_start",
        "sigma_mid",
        "sigma_end",
        "tau_start",
        "tau_mid",
        "tau_end",
        "defect_mean",
        "defect_std",
        "denominator_mean",
        "num_samples",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: str | Path, summaries: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_param",
        "num_steps",
        "num_bins",
        "mean_defect",
        "std_defect",
        "max_defect",
        "min_defect",
        "defect_uniformity_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key, "") for key in fieldnames})


def derive_warp_weights_from_defect(rows: list[dict], *, floor: float = 0.05, power: float = 1.0) -> list[float]:
    values = np.asarray([float(row["defect_mean"]) for row in rows], dtype=np.float64)
    values = np.where(np.isfinite(values), values, np.nanmedian(values[np.isfinite(values)]) if np.isfinite(values).any() else 1.0)
    values = values - values.min()
    if values.max() > 0:
        values = values / values.max()
    weights = float(floor) + np.power(values + 1.0e-8, float(power))
    if len(weights) > 0:
        weights = np.concatenate([weights, weights[-1:]])
    return np.maximum(weights, 1.0e-8).tolist()


def pca_2d(states: torch.Tensor, *, trajectory_index: int = 0) -> np.ndarray:
    x = states[int(trajectory_index)].flatten(1).detach().cpu().numpy().astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def read_defect_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in list(row.keys()):
            if key not in {"time_param"}:
                try:
                    row[key] = float(row[key])
                except (TypeError, ValueError):
                    pass
    return rows


def plot_trajectory_2d(
    *,
    trajectory_path: str | Path,
    defect_csv: str | Path,
    out_png: str | Path,
    out_pdf: str | Path | None = None,
    trajectory_index: int = 0,
    title: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    trajectory = load_trajectory(trajectory_path)
    states = trajectory["states"]
    metadata = trajectory.get("metadata", {})
    rows = read_defect_csv(defect_csv)
    coords = pca_2d(states, trajectory_index=trajectory_index)
    defects = np.asarray([float(row["defect_mean"]) for row in rows], dtype=np.float64)
    if defects.size == 0:
        defects = np.zeros(max(len(coords) - 1, 1), dtype=np.float64)
    defects = np.where(np.isfinite(defects), defects, np.nanmedian(defects[np.isfinite(defects)]) if np.isfinite(defects).any() else 0.0)
    segment_values = np.asarray([defects[min(i, len(defects) - 1)] for i in range(len(coords) - 1)], dtype=np.float64)
    segments = np.stack([coords[:-1], coords[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(4.4, 3.45), dpi=180, constrained_layout=True)
    finite_values = segment_values[np.isfinite(segment_values)]
    if finite_values.size:
        vmin, vmax = np.percentile(finite_values, [5, 95])
        if vmax <= vmin:
            vmax = vmin + 1.0e-12
    else:
        vmin, vmax = 0.0, 1.0
    collection = LineCollection(segments, cmap="inferno", linewidths=2.0)
    collection.set_array(segment_values)
    collection.set_clim(vmin, vmax)
    ax.add_collection(collection)
    ax.scatter(coords[:, 0], coords[:, 1], color="white", edgecolor="black", linewidth=0.25, s=10, zorder=3)
    ax.annotate("start", xy=coords[0], xytext=(5, 5), textcoords="offset points", fontsize=7)
    ax.annotate("end", xy=coords[-1], xytext=(5, -10), textcoords="offset points", fontsize=7)
    ax.autoscale()
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title or f"{metadata.get('time_param', 'trajectory')} trajectory", fontsize=9)
    cbar = fig.colorbar(collection, ax=ax)
    cbar.set_label("local defect")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    if out_pdf:
        fig.savefig(out_pdf)
    plt.close(fig)

def plot_defect_profile(
    *,
    defect_csv: str | Path,
    out_png: str | Path,
    out_pdf: str | Path | None = None,
    title: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    rows = read_defect_csv(defect_csv)
    x = np.asarray([int(row["interval_index"]) for row in rows], dtype=np.int64)
    y = np.asarray([float(row["defect_mean"]) for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(4.8, 2.8), dpi=180, constrained_layout=True)
    ax.plot(x, y, marker="o", markersize=2.4, linewidth=1.25, color="#2f6fbb")
    ax.set_xlabel("interval index")
    ax.set_ylabel("normalized defect")
    ax.set_title(title or "Defect profile", fontsize=9)
    if len(x) > 10:
        tick_step = max(1, int(np.ceil(len(x) / 8)))
        ax.set_xticks(x[::tick_step])
    ax.grid(True, alpha=0.25)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    if out_pdf:
        fig.savefig(out_pdf)
    plt.close(fig)


def plot_defect_comparison(
    *,
    identity_csv: str | Path,
    warp_csv: str | Path,
    out_png: str | Path,
    out_pdf: str | Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    id_rows = read_defect_csv(identity_csv)
    warp_rows = read_defect_csv(warp_csv)
    fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=180, constrained_layout=True)
    for label, rows, color in [("identity", id_rows, "#4c78a8"), ("dg_twfd_warp", warp_rows, "#f58518")]:
        x = np.asarray([int(row["interval_index"]) for row in rows], dtype=np.int64)
        y = np.asarray([float(row["defect_mean"]) for row in rows], dtype=np.float64)
        ax.plot(x, y, marker="o", markersize=2.4, linewidth=1.25, label=label, color=color)
    ax.set_xlabel("interval index")
    ax.set_ylabel("normalized defect")
    ax.set_title("Defect profile comparison", fontsize=9)
    if id_rows:
        id_x = np.asarray([int(row["interval_index"]) for row in id_rows], dtype=np.int64)
        if len(id_x) > 10:
            tick_step = max(1, int(np.ceil(len(id_x) / 8)))
            ax.set_xticks(id_x[::tick_step])
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    if out_pdf:
        fig.savefig(out_pdf)
    plt.close(fig)
