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
    parser.add_argument("--cifar-seeds", default="42,123,314,512,777,1024", help="Generic CIFAR-10 seed list.")
    parser.add_argument("--cifar-step-seeds", default="42,123,314,512,777,1024", help="CIFAR-10 seeds for progression figures.")
    parser.add_argument("--cifar-identity-seeds", default="13,71,207,409,701,1201", help="CIFAR-10 seeds for identity-vs-DG-TWFD figures.")
    parser.add_argument("--imagenet-seeds", default="7,19,43,87,131,211", help="Generic ImageNet64 seed list.")
    parser.add_argument("--imagenet-classes", default="207,250,281,409,530,751", help="Generic ImageNet64 class list.")
    parser.add_argument("--imagenet-step-seeds", default="7,19,43,87,131,211", help="ImageNet64 seeds for progression figures.")
    parser.add_argument("--imagenet-step-classes", default="207,250,281,409,530,751", help="ImageNet64 classes for progression figures.")
    parser.add_argument("--imagenet-identity-seeds", default="17,41,79,113,167,227", help="ImageNet64 seeds for identity-vs-DG-TWFD figures.")
    parser.add_argument("--imagenet-identity-classes", default="281,530,207,751,409,250", help="ImageNet64 classes for identity-vs-DG-TWFD figures.")
    parser.add_argument("--outdir", default=str(EXP_ROOT / "manifests"), help="Manifest output directory.")
    args = parser.parse_args()

    cifar_seeds = parse_int_list(args.cifar_seeds)
    cifar_step_seeds = parse_int_list(args.cifar_step_seeds)
    cifar_identity_seeds = parse_int_list(args.cifar_identity_seeds)
    imagenet_seeds = parse_int_list(args.imagenet_seeds)
    imagenet_classes = parse_int_list(args.imagenet_classes)
    imagenet_step_seeds = parse_int_list(args.imagenet_step_seeds)
    imagenet_step_classes = parse_int_list(args.imagenet_step_classes)
    imagenet_identity_seeds = parse_int_list(args.imagenet_identity_seeds)
    imagenet_identity_classes = parse_int_list(args.imagenet_identity_classes)
    if len(imagenet_seeds) != len(imagenet_classes):
        raise ValueError("ImageNet64 seeds and classes must have the same length")
    if len(imagenet_step_seeds) != len(imagenet_step_classes):
        raise ValueError("ImageNet64 step seeds and classes must have the same length")
    if len(imagenet_identity_seeds) != len(imagenet_identity_classes):
        raise ValueError("ImageNet64 identity seeds and classes must have the same length")

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
    cifar_step_payload = {
        "dataset": "CIFAR-10",
        "note": "CIFAR-10 progression rows. Kept distinct from the identity-vs rows to broaden sample coverage.",
        "rows": [{"seed": int(seed)} for seed in cifar_step_seeds],
    }
    cifar_identity_payload = {
        "dataset": "CIFAR-10",
        "note": "CIFAR-10 identity-vs-DG-TWFD rows. Kept distinct from progression rows to broaden sample coverage.",
        "rows": [{"seed": int(seed)} for seed in cifar_identity_seeds],
    }
    imagenet_step_payload = {
        "dataset": "ImageNet64",
        "note": "ImageNet64 progression rows. Fixed seed-class pairs for DG-TWFD step progression figures.",
        "rows": [
            {"seed": int(seed), "class_idx": int(class_idx)}
            for seed, class_idx in zip(imagenet_step_seeds, imagenet_step_classes)
        ],
    }
    imagenet_identity_payload = {
        "dataset": "ImageNet64",
        "note": "ImageNet64 identity-vs-DG-TWFD rows. Kept distinct from progression rows to broaden sample coverage.",
        "rows": [
            {"seed": int(seed), "class_idx": int(class_idx)}
            for seed, class_idx in zip(imagenet_identity_seeds, imagenet_identity_classes)
        ],
    }

    cifar_path = outdir / "DG_TWFD_cifar10_fixed_rows.json"
    imagenet_path = outdir / "DG_TWFD_imagenet64_fixed_rows.json"
    cifar_step_path = outdir / "DG_TWFD_cifar10_step_rows.json"
    cifar_identity_path = outdir / "DG_TWFD_cifar10_identity_rows.json"
    imagenet_step_path = outdir / "DG_TWFD_imagenet64_step_rows.json"
    imagenet_identity_path = outdir / "DG_TWFD_imagenet64_identity_rows.json"
    write_json(cifar_path, cifar_payload)
    write_json(imagenet_path, imagenet_payload)
    write_json(cifar_step_path, cifar_step_payload)
    write_json(cifar_identity_path, cifar_identity_payload)
    write_json(imagenet_step_path, imagenet_step_payload)
    write_json(imagenet_identity_path, imagenet_identity_payload)
    print(f"Wrote {cifar_path}")
    print(f"Wrote {imagenet_path}")
    print(f"Wrote {cifar_step_path}")
    print(f"Wrote {cifar_identity_path}")
    print(f"Wrote {imagenet_step_path}")
    print(f"Wrote {imagenet_identity_path}")


if __name__ == "__main__":
    main()
