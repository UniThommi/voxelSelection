#!/usr/bin/env python3
"""
Greedy Coverage Curve Comparison
=================================

Loads efficiency curves from multiple greedy NPZ result files and
plots them on a single figure. Optionally computes and displays the
theoretical upper bound ε_max (fraction of NCs coverable by any
combination of all valid voxels).

Usage:
    # Basic comparison
    python compareGreedyCurves.py \
        results/ncM1m1.npz \
        results/ncM1m1Weight.npz \
        results/muonM1m1W1.npz \
        -o comparison.png

    # With upper bound from HDF5
    python compareGreedyCurves.py \
        results/ncM1m1.npz \
        results/ncM6m1.npz \
        --hdf5 data.hdf5 --area-ratio \
        -o comparison.png

    # Custom labels
    python compareGreedyCurves.py \
        results/ncM1m1.npz "NC M=1" \
        results/ncM6m1.npz "NC M=6" \
        -o comparison.png

Author: Thomas Buerger (University of Tübingen)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_efficiency_curve(npz_path: str) -> tuple[np.ndarray, dict]:
    """
    Load efficiency curve and metadata from a greedy result NPZ file.

    Parameters
    ----------
    npz_path : str
        Path to the NPZ file.

    Returns
    -------
    efficiencies : np.ndarray
        Efficiency at each greedy step (length N).
    metadata : dict
        Run parameters extracted from the NPZ.
    """
    data = np.load(npz_path, allow_pickle=True)
    efficiencies = data["efficiencies"]

    metadata = {}
    for key in ["N", "M", "m", "W", "optimize", "min_spacing",
                "num_ncs", "num_primaries", "num_ge77_muons",
                "per_area"]:
        if key in data:
            val = data[key]
            # np.load wraps scalars in 0-d arrays
            metadata[key] = val.item() if val.ndim == 0 else val
    metadata["npz_path"] = npz_path

    return efficiencies, metadata


def compute_epsilon_max(
    hdf5_path: str,
    M: int = 1,
    m: int = 1,
    apply_area_ratio: bool = False,
) -> float:
    """
    Compute the theoretical upper bound: fraction of NCs that are
    seen by >= M valid voxels in total (i.e., the coverage if all
    voxels were selected).

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 data file.
    M : int
        Multiplicity threshold.
    m : int
        Hit threshold per voxel.
    apply_area_ratio : bool
        Apply area-dependent scaling before binarization.

    Returns
    -------
    float
        ε_max: fraction of NCs coverable.
    """
    # Import from greedy script
    try:
        from greedyVoxelSelection import load_and_binarize
    except ImportError:
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../core")
        from greedyVoxelSelection import load_and_binarize

    B, _, _, _, _ = load_and_binarize(
        hdf5_path, m=m, apply_area_ratio=apply_area_ratio, verbose=False,
    )

    # Number of voxels seeing each NC
    nnz_per_row = np.diff(B.tocsr().indptr)
    num_coverable = int(np.sum(nnz_per_row >= M))
    epsilon_max = num_coverable / B.shape[0]

    print(f"ε_max (M={M}, m={m}, area_ratio={apply_area_ratio}): "
          f"{epsilon_max:.4%} ({num_coverable:,} / {B.shape[0]:,})")

    return epsilon_max


def auto_label(metadata: dict, npz_path: str) -> str:
    """
    Generate a descriptive label from NPZ metadata.

    Parameters
    ----------
    metadata : dict
        Run parameters.
    npz_path : str
        Filename for fallback.

    Returns
    -------
    str
        Human-readable label.
    """
    opt = metadata.get("optimize", "?")
    parts = [opt]

    M = metadata.get("M", 1)
    if M != 1:
        parts.append(f"M={M}")

    m = metadata.get("m", 1)
    if m != 1:
        parts.append(f"m={m}")

    W = metadata.get("W", None)
    if W is not None and opt == "muon-ge77":
        parts.append(f"W={int(W)}")

    if metadata.get("per_area", False):
        parts.append("per-area")

    # Detect muon-weight and area-ratio from filename
    stem = Path(npz_path).stem
    if "_mw" in stem:
        mw_part = stem.split("_mw")[1].split("_")[0].split(".npz")[0]
        parts.append(f"mw={mw_part}")
    if "arearatio" in stem:
        parts.append("ratio")
    if "nospacing" in stem:
        parts.append("no-sp")

    return ", ".join(parts)


def plot_comparison(
    curves: list[tuple[np.ndarray, str]],
    output_path: Path,
    epsilon_max: Optional[float] = None,
    title: str = "Greedy Coverage Curves",
) -> None:
    """
    Plot multiple efficiency curves on one figure.

    Parameters
    ----------
    curves : list of (efficiencies, label)
        Each entry is an efficiency array and its display label.
    output_path : Path
        Where to save the plot.
    epsilon_max : float or None
        If given, draw a horizontal reference line.
    title : str
        Plot title.
    """
    fig, (ax_main, ax_zoom) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [2, 1]},
    )

    # Color cycle
    colors = plt.cm.tab10.colors
    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

    for i, (eff, label) in enumerate(curves):
        k_vals = np.arange(1, len(eff) + 1)
        color = colors[i % len(colors)]
        ls = line_styles[i % len(line_styles)]

        ax_main.plot(k_vals, eff * 100, label=label,
                     color=color, linestyle=ls, linewidth=1.8)
        ax_zoom.plot(k_vals, eff * 100, label=label,
                     color=color, linestyle=ls, linewidth=1.8)

    # Upper bound
    if epsilon_max is not None:
        for ax in [ax_main, ax_zoom]:
            ax.axhline(y=epsilon_max * 100, color="black",
                       linestyle="--", linewidth=1.2, alpha=0.7,
                       label=f"ε_max = {epsilon_max:.2%}")

    # Main plot formatting
    ax_main.set_xlabel("Number of selected PMTs (k)", fontsize=12)
    ax_main.set_ylabel("Detection Efficiency (%)", fontsize=12)
    ax_main.set_title(title, fontsize=13)
    ax_main.legend(fontsize=9, loc="lower right")
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(1, max(len(eff) for eff, _ in curves))

    # Zoom: first 50 voxels
    zoom_limit = min(50, min(len(eff) for eff, _ in curves))
    ax_zoom.set_xlabel("Number of selected PMTs (k)", fontsize=12)
    ax_zoom.set_ylabel("Detection Efficiency (%)", fontsize=12)
    ax_zoom.set_title(f"Zoom: first {zoom_limit} PMTs", fontsize=13)
    ax_zoom.set_xlim(1, zoom_limit)
    ax_zoom.legend(fontsize=8, loc="lower right")
    ax_zoom.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {output_path}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare greedy coverage curves from multiple runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Positional arguments are NPZ files, optionally followed by\n"
            "a custom label in quotes:\n"
            "  compareGreedyCurves.py run1.npz \"Label 1\" run2.npz \"Label 2\"\n"
            "If no label is given, one is auto-generated from metadata."
        ),
    )
    parser.add_argument("inputs", nargs="+",
                        help="NPZ files, optionally each followed by a label string.")
    parser.add_argument("-o", "--output", type=str, default="greedy_comparison.png",
                        help="Output plot path.")
    parser.add_argument("--hdf5", type=str, default=None,
                        help="HDF5 file for computing ε_max upper bound.")
    parser.add_argument("-M", type=int, default=1,
                        help="M threshold for ε_max computation.")
    parser.add_argument("-m", type=int, default=1,
                        help="m threshold for ε_max computation.")
    parser.add_argument("--area-ratio", action="store_true",
                        help="Apply area-ratio scaling for ε_max computation.")
    parser.add_argument("--title", type=str,
                        default="Greedy Coverage Curves",
                        help="Plot title.")

    args = parser.parse_args(argv)

    # Parse inputs: alternating NPZ paths and optional labels
    npz_files: list[str] = []
    labels: list[Optional[str]] = []

    i = 0
    inputs = args.inputs
    while i < len(inputs):
        if inputs[i].endswith(".npz"):
            npz_files.append(inputs[i])
            # Check if next arg is a label (not an NPZ)
            if i + 1 < len(inputs) and not inputs[i + 1].endswith(".npz"):
                labels.append(inputs[i + 1])
                i += 2
            else:
                labels.append(None)
                i += 1
        else:
            parser.error(f"Expected .npz file, got: {inputs[i]}")

    if len(npz_files) == 0:
        parser.error("No NPZ files provided.")

    # Load curves
    curves: list[tuple[np.ndarray, str]] = []
    for npz_path, custom_label in zip(npz_files, labels):
        eff, meta = load_efficiency_curve(npz_path)
        if custom_label is not None:
            label = custom_label
        else:
            label = auto_label(meta, npz_path)
        curves.append((eff, label))
        print(f"Loaded: {npz_path} ({len(eff)} steps) → \"{label}\"")

    # Compute upper bound if requested
    epsilon_max = None
    if args.hdf5 is not None:
        epsilon_max = compute_epsilon_max(
            args.hdf5, M=args.M, m=args.m,
            apply_area_ratio=args.area_ratio,
        )

    # Print summary table
    print(f"\n{'Label':<35} {'N':>5} {'Final ε':>10} {'ε at k=50':>10}")
    print("-" * 65)
    for eff, label in curves:
        final = eff[-1] if len(eff) > 0 else 0.0
        at_50 = eff[49] if len(eff) >= 50 else eff[-1]
        print(f"{label:<35} {len(eff):>5} {final:>10.4%} {at_50:>10.4%}")
    if epsilon_max is not None:
        print(f"{'ε_max':<35} {'':>5} {epsilon_max:>10.4%}")

    # Plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(curves, output_path, epsilon_max=epsilon_max,
                    title=args.title)


if __name__ == "__main__":
    main()