#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


EDM_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDM_ROOT))
sys.path.insert(0, str(EXP_ROOT))

from utils.timewarp_core import load_trajectory, plot_defect_profile, plot_trajectory_2d, plot_trajectory_gallery, resolve_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 2D PCA trajectory and defect profile for EDM timewarp analysis.")
    parser.add_argument("--trajectory", required=True, help="Path to trajectory.pt.")
    parser.add_argument("--defect-csv", required=True, help="Path to per-interval defect CSV.")
    parser.add_argument("--defect-matrix", default=None, help="Optional per-trajectory defect matrix (.npz).")
    parser.add_argument("--out-trajectory", default=None, help="Output trajectory PNG.")
    parser.add_argument("--out-profile", default=None, help="Output defect profile PNG.")
    parser.add_argument("--trajectory-index", type=int, default=0, help="Which saved trajectory to plot.")
    parser.add_argument("--gallery-indices", default=None, help="Optional comma-separated trajectory indices for a gallery plot.")
    parser.add_argument("--out-gallery", default=None, help="Optional gallery PNG path.")
    args = parser.parse_args()

    trajectory_path = resolve_path(args.trajectory, root=EDM_ROOT)
    defect_csv = resolve_path(args.defect_csv, root=EDM_ROOT)
    defect_matrix = resolve_path(args.defect_matrix, root=EDM_ROOT) if args.defect_matrix else None
    metadata = load_trajectory(trajectory_path).get("metadata", {})
    time_param = metadata.get("time_param", "unknown")
    out_trajectory = (
        resolve_path(args.out_trajectory, root=EDM_ROOT)
        if args.out_trajectory
        else EXP_ROOT / "figures" / f"trajectory_{time_param}.png"
    )
    out_profile = (
        resolve_path(args.out_profile, root=EDM_ROOT)
        if args.out_profile
        else EXP_ROOT / "figures" / f"defect_profile_{time_param}.png"
    )
    out_gallery = resolve_path(args.out_gallery, root=EDM_ROOT) if args.out_gallery else None

    plot_trajectory_2d(
        trajectory_path=trajectory_path,
        defect_csv=defect_csv,
        defect_matrix_path=defect_matrix,
        out_png=out_trajectory,
        out_pdf=out_trajectory.with_suffix(".pdf"),
        trajectory_index=int(args.trajectory_index),
        title=f"{time_param} trajectory with defect heat",
    )
    plot_defect_profile(
        defect_csv=defect_csv,
        defect_matrix_path=defect_matrix,
        out_png=out_profile,
        out_pdf=out_profile.with_suffix(".pdf"),
        title=f"{time_param} interval defect",
    )
    if args.gallery_indices:
        if out_gallery is None:
            out_gallery = EXP_ROOT / "figures" / f"trajectory_{time_param}_gallery.png"
        gallery_indices = [int(part) for part in args.gallery_indices.split(",") if part.strip()]
        plot_trajectory_gallery(
            trajectory_path=trajectory_path,
            defect_csv=defect_csv,
            defect_matrix_path=defect_matrix,
            out_png=out_gallery,
            out_pdf=out_gallery.with_suffix(".pdf"),
            trajectory_indices=gallery_indices,
            title=f"{time_param} multi-trajectory defect heatmaps",
        )
        print(f"Wrote trajectory gallery: {out_gallery}")
    print(f"Wrote trajectory figure: {out_trajectory}")
    print(f"Wrote defect profile figure: {out_profile}")


if __name__ == "__main__":
    main()
