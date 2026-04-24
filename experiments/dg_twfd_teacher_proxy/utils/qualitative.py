from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from .edm_proxy import StackedRandomGenerator, edm_sigma_schedule, save_image_batch


METHODS = ("dg_twfd", "identity_clock")


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for item in str(text).split(","):
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


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_path(path: str | Path, *, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def dataset_key(name: str) -> str:
    key = str(name).lower().replace("-", "").replace("_", "").replace(" ", "")
    if "cifar" in key:
        return "cifar10"
    if "imagenet" in key:
        return "imagenet64"
    return key


def pretty_method_label(method: str) -> str:
    if method == "dg_twfd":
        return "DG-TWFD"
    if method == "identity_clock":
        return "identity clock"
    return method


def row_label(row: dict[str, Any], *, dataset: str) -> str:
    if dataset_key(dataset) == "imagenet64":
        return f"seed {int(row['seed'])} / cls {int(row['class_idx'])}"
    return f"seed {int(row['seed'])}"


def nfe_from_steps(steps: int) -> int:
    steps = int(steps)
    if steps <= 1:
        return 1
    return 2 * steps - 1


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


def schedule_for_method(
    net,
    *,
    method: str,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    if method not in METHODS:
        raise ValueError(f"Unsupported method {method}; expected {METHODS}")
    if method == "identity_clock":
        return linear_sigma_schedule(
            net,
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            device=device,
        )
    return edm_sigma_schedule(
        net,
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
    )


def class_labels_from_ids(net, *, class_ids: list[int | None], device: torch.device) -> torch.Tensor | None:
    if not getattr(net, "label_dim", 0):
        return None
    labels = torch.zeros((len(class_ids), int(net.label_dim)), dtype=torch.float32, device=device)
    for row_index, class_id in enumerate(class_ids):
        if class_id is None:
            raise ValueError("Conditional model requires explicit class ids for all rows")
        labels[row_index, int(class_id)] = 1.0
    return labels


@torch.no_grad()
def sample_with_sigmas(
    net,
    *,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    class_labels: torch.Tensor | None,
    randn_like,
    sampler_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    sampler_kwargs = dict(sampler_kwargs or {})
    S_churn = float(sampler_kwargs.get("S_churn", 0.0))
    S_min = float(sampler_kwargs.get("S_min", 0.0))
    S_max = float(sampler_kwargs.get("S_max", float("inf")))
    S_noise = float(sampler_kwargs.get("S_noise", 1.0))
    num_steps = int(sigmas.numel() - 1)
    x_next = latents.to(torch.float64) * sigmas[0]

    for step_index, (t_cur, t_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        x_cur = x_next
        gamma = min(S_churn / max(num_steps, 1), math.sqrt(2.0) - 1.0) if S_min <= float(t_cur) <= S_max else 0.0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        noise_scale = (t_hat.square() - t_cur.square()).clamp_min(0.0).sqrt()
        x_hat = x_cur + noise_scale * S_noise * randn_like(x_cur)

        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if step_index < num_steps - 1 and float(t_next) > 0.0:
            denoised_next = net(x_next, t_next, class_labels).to(torch.float64)
            d_next = (x_next - denoised_next) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_next)

    return x_next.to(torch.float32)


def image_path_for_seed(root: Path, seed: int, *, subdirs: bool) -> Path:
    if subdirs:
        return root / f"{seed - seed % 1000:06d}" / f"{seed:06d}.png"
    return root / f"{seed:06d}.png"


def ensure_rows(rows: list[dict[str, Any]], *, dataset: str) -> list[dict[str, Any]]:
    fixed: list[dict[str, Any]] = []
    for row in rows:
        item = {"seed": int(row["seed"])}
        if dataset_key(dataset) == "imagenet64":
            if "class_idx" not in row:
                raise ValueError("ImageNet64 qualitative rows require class_idx")
            item["class_idx"] = int(row["class_idx"])
        fixed.append(item)
    return fixed


def render_samples_for_rows(
    net,
    *,
    rows: list[dict[str, Any]],
    method: str,
    num_steps: int,
    cfg: dict[str, Any],
    outdir: Path,
    batch_size: int,
    device: torch.device,
    subdirs: bool,
    overwrite: bool,
) -> dict[str, Any]:
    rows = ensure_rows(rows, dataset=str(cfg["dataset"]))
    seeds = [int(row["seed"]) for row in rows]
    expected = [image_path_for_seed(outdir, seed, subdirs=subdirs) for seed in seeds]
    if not overwrite and expected and all(path.is_file() for path in expected):
        return {"elapsed_sec": 0.0, "num_images": len(seeds), "skipped": True}

    sigmas = schedule_for_method(
        net,
        method=method,
        num_steps=int(num_steps),
        sigma_min=float(cfg["sigma_min"]),
        sigma_max=float(cfg["sigma_max"]),
        rho=float(cfg["rho"]),
        device=device,
    )
    sampler_kwargs = dict(cfg.get("official_edm_sampler_kwargs", {}))
    start_time = time.time()
    outdir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_seeds = [int(row["seed"]) for row in batch_rows]
        batch_classes = [row.get("class_idx") for row in batch_rows]
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([len(batch_rows), net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = class_labels_from_ids(net, class_ids=batch_classes, device=device)
        images = sample_with_sigmas(
            net,
            latents=latents,
            sigmas=sigmas,
            class_labels=class_labels,
            randn_like=rnd.randn_like,
            sampler_kwargs=sampler_kwargs,
        )
        save_image_batch(images, batch_seeds=batch_seeds, outdir=outdir, subdirs=subdirs)

    return {
        "elapsed_sec": float(time.time() - start_time),
        "num_images": len(seeds),
        "skipped": False,
    }


def _resize_image(path: Path, *, cell_size: int) -> Any:
    import PIL.Image

    nearest = getattr(getattr(PIL.Image, "Resampling", PIL.Image), "NEAREST")
    return PIL.Image.open(path).convert("RGB").resize((cell_size, cell_size), nearest)


def _save_canvas(canvas: Any, figure_path: Path) -> tuple[Path, Path]:
    png_path = figure_path.with_suffix(".png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path)
    canvas.save(figure_path, "PDF", resolution=200.0)
    return png_path, figure_path


def build_progression_canvas(
    *,
    dataset: str,
    method: str,
    rows: list[dict[str, Any]],
    steps: list[int],
    sample_dirs: dict[int, Path],
    subdirs: bool,
    cell_size: int,
) -> PIL.Image.Image:
    import PIL.Image
    import PIL.ImageDraw

    rows = ensure_rows(rows, dataset=dataset)
    label_w = 170 if dataset_key(dataset) == "imagenet64" else 120
    header_h = 54
    title_h = 34
    width = label_w + cell_size * len(steps)
    height = title_h + header_h + cell_size * len(rows)
    canvas = PIL.Image.new("RGB", (width, height), "white")
    draw = PIL.ImageDraw.Draw(canvas)

    draw.text((10, 8), f"{pretty_method_label(method)} | {dataset}", fill=(0, 0, 0))
    for col, step in enumerate(steps):
        x = label_w + col * cell_size
        draw.text((x + 8, title_h + 8), f"{step} st", fill=(0, 0, 0))
        draw.text((x + 8, title_h + 25), f"{nfe_from_steps(step)} NFE", fill=(85, 85, 85))

    for row_index, row in enumerate(rows):
        y = title_h + header_h + row_index * cell_size
        draw.text((8, y + cell_size // 2 - 7), row_label(row, dataset=dataset), fill=(0, 0, 0))
        for col, step in enumerate(steps):
            x = label_w + col * cell_size
            img = _resize_image(image_path_for_seed(sample_dirs[step], int(row["seed"]), subdirs=subdirs), cell_size=cell_size)
            canvas.paste(img, (x, y))

    for col in range(len(steps) + 1):
        x = label_w + col * cell_size
        draw.line((x, title_h, x, height), fill=(225, 225, 225))
    for row in range(len(rows) + 1):
        y = title_h + header_h + row * cell_size
        draw.line((0, y, width, y), fill=(225, 225, 225))
    draw.line((0, title_h, width, title_h), fill=(225, 225, 225))
    draw.line((0, title_h + header_h, width, title_h + header_h), fill=(225, 225, 225))
    return canvas


def build_identity_vs_canvas(
    *,
    dataset: str,
    rows: list[dict[str, Any]],
    steps: list[int],
    sample_dirs: dict[str, dict[int, Path]],
    subdirs: bool,
    cell_size: int,
) -> PIL.Image.Image:
    import PIL.Image
    import PIL.ImageDraw

    rows = ensure_rows(rows, dataset=dataset)
    label_w = 170 if dataset_key(dataset) == "imagenet64" else 120
    title_h = 34
    header_h = 54
    section_h = 26
    per_block_h = section_h + cell_size * len(rows)
    width = label_w + cell_size * len(steps)
    height = title_h + header_h + per_block_h * 2
    canvas = PIL.Image.new("RGB", (width, height), "white")
    draw = PIL.ImageDraw.Draw(canvas)

    draw.text((10, 8), f"identity clock vs DG-TWFD | {dataset}", fill=(0, 0, 0))
    for col, step in enumerate(steps):
        x = label_w + col * cell_size
        draw.text((x + 8, title_h + 8), f"{step} st", fill=(0, 0, 0))
        draw.text((x + 8, title_h + 25), f"{nfe_from_steps(step)} NFE", fill=(85, 85, 85))

    methods = ["identity_clock", "dg_twfd"]
    for method_index, method in enumerate(methods):
        block_y = title_h + header_h + method_index * per_block_h
        draw.rectangle((0, block_y, width, block_y + section_h), fill=(245, 245, 245))
        draw.text((8, block_y + 6), pretty_method_label(method), fill=(0, 0, 0))
        for row_index, row in enumerate(rows):
            y = block_y + section_h + row_index * cell_size
            draw.text((8, y + cell_size // 2 - 7), row_label(row, dataset=dataset), fill=(0, 0, 0))
            for col, step in enumerate(steps):
                x = label_w + col * cell_size
                img = _resize_image(
                    image_path_for_seed(sample_dirs[method][step], int(row["seed"]), subdirs=subdirs),
                    cell_size=cell_size,
                )
                canvas.paste(img, (x, y))

    for col in range(len(steps) + 1):
        x = label_w + col * cell_size
        draw.line((x, title_h, x, height), fill=(225, 225, 225))
    draw.line((0, title_h, width, title_h), fill=(225, 225, 225))
    draw.line((0, title_h + header_h, width, title_h + header_h), fill=(225, 225, 225))
    for method_index in range(2):
        block_y = title_h + header_h + method_index * per_block_h
        draw.line((0, block_y, width, block_y), fill=(225, 225, 225))
        draw.line((0, block_y + section_h, width, block_y + section_h), fill=(225, 225, 225))
        for row in range(len(rows) + 1):
            y = block_y + section_h + row * cell_size
            draw.line((0, y, width, y), fill=(225, 225, 225))
    return canvas


def build_diversity_canvas(
    *,
    dataset: str,
    method: str,
    rows: list[dict[str, Any]],
    sample_dir: Path,
    subdirs: bool,
    cell_size: int,
    grid_cols: int,
    steps: int,
) -> PIL.Image.Image:
    import PIL.Image
    import PIL.ImageDraw

    rows = ensure_rows(rows, dataset=dataset)
    grid_cols = max(1, int(grid_cols))
    grid_rows = int(math.ceil(len(rows) / grid_cols))
    title_h = 34
    width = cell_size * grid_cols
    height = title_h + cell_size * grid_rows
    canvas = PIL.Image.new("RGB", (width, height), "white")
    draw = PIL.ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{pretty_method_label(method)} | {dataset} | {steps} st / {nfe_from_steps(steps)} NFE", fill=(0, 0, 0))
    for index, row in enumerate(rows):
        col = index % grid_cols
        row_idx = index // grid_cols
        x = col * cell_size
        y = title_h + row_idx * cell_size
        img = _resize_image(image_path_for_seed(sample_dir, int(row["seed"]), subdirs=subdirs), cell_size=cell_size)
        canvas.paste(img, (x, y))
    return canvas


def save_figure_bundle(
    *,
    canvas: Any,
    figure_path: str | Path,
    manifest_path: str | Path,
    manifest_payload: dict[str, Any],
) -> dict[str, str]:
    figure_path = Path(figure_path)
    png_path, pdf_path = _save_canvas(canvas, figure_path)
    payload = dict(manifest_payload)
    payload["figure_pdf"] = str(pdf_path)
    payload["figure_png"] = str(png_path)
    write_json(manifest_path, payload)
    return {"figure_pdf": str(pdf_path), "figure_png": str(png_path), "manifest": str(Path(manifest_path))}


def cell_records(
    *,
    rows: list[dict[str, Any]],
    steps: list[int],
    sample_dirs: dict[int, Path],
    subdirs: bool,
    method: str,
    dataset: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(ensure_rows(rows, dataset=dataset)):
        for col_index, step in enumerate(steps):
            records.append(
                {
                    "row_index": row_index,
                    "col_index": col_index,
                    "method": method,
                    "seed": int(row["seed"]),
                    "class_idx": row.get("class_idx"),
                    "steps": int(step),
                    "nfe": nfe_from_steps(int(step)),
                    "image_path": str(image_path_for_seed(sample_dirs[int(step)], int(row["seed"]), subdirs=subdirs)),
                }
            )
    return records
