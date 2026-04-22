#!/usr/bin/env python3
"""Run EDM teacher-proxy sampling and FID sweeps for DG-TWFD.

This is intentionally a thin wrapper over the official EDM `generate.py` and
`fid.py` entry points so that experiments stay reproducible and easy to audit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_CIFAR10_NETWORK = (
    "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/"
    "edm-cifar10-32x32-cond-vp.pkl"
)
DEFAULT_CIFAR10_REF = "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz"


def _parse_steps(value: str) -> list[int]:
    steps: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        step = int(item)
        if step < 1:
            raise argparse.ArgumentTypeError("steps must be positive integers")
        steps.append(step)
    if not steps:
        raise argparse.ArgumentTypeError("at least one step count is required")
    return steps


def _seed_range(start: int, count: int) -> str:
    if count < 2:
        raise argparse.ArgumentTypeError("num-samples must be at least 2")
    return f"{start}-{start + count - 1}"


def _run_and_log(command: list[str], *, cwd: Path, log_path: Path, dry_run: bool) -> list[str]:
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


def _parse_fid(lines: list[str]) -> float | None:
    for line in reversed(lines):
        text = line.strip()
        if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
            return float(text)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default=DEFAULT_CIFAR10_NETWORK)
    parser.add_argument("--ref", default=DEFAULT_CIFAR10_REF)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--steps", type=_parse_steps, default=_parse_steps("1,2,4,8,16"))
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--fid-batch", type=int, default=64)
    parser.add_argument("--class-idx", type=int, default=None)
    parser.add_argument("--subdirs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    edm_root = Path(__file__).resolve().parents[2]
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = edm_root / outdir
    samples_root = outdir / "samples"
    logs_root = outdir / "logs"
    outdir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["edm_root"] = str(edm_root)
    config["outdir"] = str(outdir)
    (outdir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    metrics: dict[str, dict[str, float | int | str | None]] = {}
    seeds = _seed_range(args.seed_start, args.num_samples)

    for step_count in args.steps:
        step_key = f"steps_{step_count}"
        sample_dir = samples_root / step_key

        if not args.skip_generate:
            command = [
                sys.executable,
                "generate.py",
                f"--network={args.network}",
                f"--outdir={sample_dir}",
                f"--seeds={seeds}",
                f"--batch={args.batch}",
                f"--steps={step_count}",
            ]
            if args.subdirs:
                command.append("--subdirs")
            if args.class_idx is not None:
                command.append(f"--class={args.class_idx}")
            _run_and_log(
                command,
                cwd=edm_root,
                log_path=logs_root / f"generate_{step_key}.log",
                dry_run=args.dry_run,
            )

        fid_value = None
        if not args.skip_fid:
            command = [
                sys.executable,
                "fid.py",
                "calc",
                f"--images={sample_dir}",
                f"--ref={args.ref}",
                f"--num={args.num_samples}",
                f"--batch={args.fid_batch}",
            ]
            fid_lines = _run_and_log(
                command,
                cwd=edm_root,
                log_path=logs_root / f"fid_{step_key}.log",
                dry_run=args.dry_run,
            )
            fid_value = _parse_fid(fid_lines)

        metrics[step_key] = {
            "steps": step_count,
            "nfe_heun": 2 * step_count - 1,
            "num_samples": args.num_samples,
            "sample_dir": str(sample_dir),
            "fid": fid_value,
        }
        (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote results to {outdir}")


if __name__ == "__main__":
    main()
