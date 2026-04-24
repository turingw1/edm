#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EDM_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDM_ROOT))
sys.path.insert(0, str(EXP_ROOT))


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


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed row manifests for DG-TWFD qualitative figures.")
    parser.add_argument("--cifar-seeds", required=True, help="Comma/range list of CIFAR-10 seeds.")
    parser.add_argument("--imagenet-seeds", required=True, help="Comma/range list of ImageNet64 seeds.")
    parser.add_argument("--imagenet-classes", required=True, help="Comma/range list of ImageNet64 class ids.")
    parser.add_argument("--outdir", default=str(EXP_ROOT / "manifests"), help="Manifest output directory.")
    args = parser.parse_args()

    cifar_seeds = parse_int_list(args.cifar_seeds)
    imagenet_seeds = parse_int_list(args.imagenet_seeds)
    imagenet_classes = parse_int_list(args.imagenet_classes)
    if len(imagenet_seeds) != len(imagenet_classes):
        raise ValueError("ImageNet64 seeds and classes must have the same length")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cifar_payload = {
        "dataset": "CIFAR-10",
        "note": "Fixed seeds for qualitative figures. Reuse exactly across all CIFAR-10 panels.",
        "rows": [{"seed": int(seed)} for seed in cifar_seeds],
    }
    imagenet_payload = {
        "dataset": "ImageNet64",
        "note": "Fixed seed-class pairs for qualitative figures. Reuse exactly across all ImageNet64 panels.",
        "rows": [
            {"seed": int(seed), "class_idx": int(class_idx)}
            for seed, class_idx in zip(imagenet_seeds, imagenet_classes)
        ],
    }

    cifar_path = outdir / "DG_TWFD_cifar10_fixed_rows.json"
    imagenet_path = outdir / "DG_TWFD_imagenet64_fixed_rows.json"
    write_json(cifar_path, cifar_payload)
    write_json(imagenet_path, imagenet_payload)
    print(f"Wrote {cifar_path}")
    print(f"Wrote {imagenet_path}")


if __name__ == "__main__":
    main()
