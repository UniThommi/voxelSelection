#!/usr/bin/env python3
"""
Sample 12 NC configurations spanning [eff_min, eff_max] evenly.
============================================================

Generates N_CONFIGS PMT configurations (default: 12) with ncM1m1 settings
whose NC detection efficiencies are as evenly spaced as possible between
the worst-case (``--worst`` greedy) and best-case (greedy) outcomes.

Algorithm
---------
1. Greedy run  → eff_max  (setup_09)
2. Worst-case greedy run → eff_min  (setup_00)
3. 8 intermediate target efficiencies, linearly spaced in (eff_min, eff_max)
4. For each target: binary search on ``random_fraction`` f ∈ [0, 1] where
   the greedy step picks the best voxel with probability (1-f) and a random
   valid voxel with probability f.  Average over ``--avg-seeds`` runs per f
   for stability.

Usage
-----
    python -m pmtopt.sample_efficiency_range \\
        --hdf5 data.hdf5 \\
        -N 300 \\
        --output-dir setups/ \\
        --seed 42

Author: Thomas Buerger (University of Tübingen)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

import h5py
import numpy as np
from scipy import sparse

from pmtopt.data_loading import load_raw_sparse, binarize_from_raw
from pmtopt.geometry import DEFAULT_AREA_RATIOS, PMT_RADIUS
from pmtopt.greedy import _apply_spacing, greedy_select_nc
from pmtopt.sensitivity import run_sensitivity


# ===================================================================
# Stochastic greedy (M=1 only, for ncM1m1)
# ===================================================================

def stochastic_greedy_nc_m1(
    B: sparse.csc_matrix,
    N: int,
    f: float,
    rng: np.random.Generator,
    centers: np.ndarray,
    layers: np.ndarray,
    min_spacing: float,
    correction_strength: float = 2.0,
) -> tuple[list[int], np.ndarray, float]:
    """
    Run a stochastic greedy selection for M=1 NC coverage.

    At each step:
    - With adaptive probability p: pick the voxel with highest marginal gain
      (greedy choice).
    - With probability (1 - p): pick the voxel with lowest marginal gain
      (anti-greedy choice).

    The base probability is f, corrected exponentially by the running deficit
    between expected and actual greedy picks:
        deficit = f * step - greedy_count_so_far
        p = clamp(f * exp(correction_strength * deficit / N), 0, 1)
    A positive deficit (too few greedy picks so far) raises p; a negative
    deficit lowers it, keeping the total greedy pick count close to f * N.

    Parameters
    ----------
    B : sparse.csc_matrix
        Binary NC × voxel matrix.
    N : int
        Number of voxels to select.
    f : float
        Target probability of greedy (best) pick in [0, 1].
        f=1 → pure greedy, f=0 → pure anti-greedy.
    rng : np.random.Generator
        Random number generator.
    centers : np.ndarray, shape (num_voxels, 3)
        Voxel center coordinates for spacing constraint.
    layers : np.ndarray of str
        Layer label per voxel.
    min_spacing : float
        Minimum distance between voxels on the same layer (mm).
    correction_strength : float
        Exponent scale for the adaptive correction. Higher values make the
        probability react more aggressively to deficits. Default: 2.0.

    Returns
    -------
    selected : list[int]
        Column indices of selected voxels.
    coverage_counts : np.ndarray, dtype int16
        Per-NC coverage count.
    efficiency : float
        Detection efficiency = (NCs with coverage >= 1) / num_ncs.
    """
    num_ncs, num_voxels = B.shape
    coverage_counts = np.zeros(num_ncs, dtype=np.int16)
    available = np.ones(num_voxels, dtype=bool)
    selected: list[int] = []

    enforce_spacing = (min_spacing > 0 and centers is not None
                       and layers is not None)
    min_spacing_sq = min_spacing ** 2
    greedy_count = 0

    for step in range(N):
        at_m1 = (coverage_counts == 0)
        gains = B.T.dot(at_m1.astype(np.int32))

        # Adaptive probability: correct for running deficit vs target f * N
        deficit = f * step - greedy_count
        p = float(np.clip(f * np.exp(correction_strength * deficit / N), 0.0, 1.0))

        if rng.random() < p:
            # Greedy pick: voxel with most undetected NCs
            gains[~available] = -1
            best_voxel = int(np.argmax(gains))
            greedy_count += 1
        else:
            # Anti-greedy pick: voxel with fewest undetected NCs
            gains[~available] = gains.max() + 1
            best_voxel = int(np.argmin(gains))

        # Update coverage
        s, e = B.indptr[best_voxel], B.indptr[best_voxel + 1]
        coverage_counts[B.indices[s:e]] += 1
        available[best_voxel] = False
        selected.append(best_voxel)

        if enforce_spacing:
            _apply_spacing(centers, layers, available, best_voxel,
                           min_spacing_sq, verbose=False)

    efficiency = float(np.sum(coverage_counts >= 1)) / num_ncs
    return selected, coverage_counts, efficiency



# ===================================================================
# JSON output
# ===================================================================

def write_setup_json(
    hdf5_path: str,
    voxel_ids: np.ndarray,
    selected_cols: list[int],
    efficiency: float,
    num_ncs: int,
    num_primaries: int,
    N: int,
    M: int,
    m: int,
    area_ratios: dict,
    seed: int,
    min_spacing: float,
    output_path: Path,
) -> None:
    """Write a setup JSON in the standard selected_voxels format."""
    selected_voxels_json = []
    with h5py.File(hdf5_path, "r") as f:
        for col_idx in selected_cols:
            vid = voxel_ids[col_idx]
            center = f[f"voxels/{vid}/center"][:].tolist()
            corners_x = f[f"voxels/{vid}/corners/x"][:].tolist()
            corners_y = f[f"voxels/{vid}/corners/y"][:].tolist()
            corners_z = f[f"voxels/{vid}/corners/z"][:].tolist()
            corners = [[x, y, z]
                       for x, y, z in zip(corners_x, corners_y, corners_z)]
            layer_raw = f[f"voxels/{vid}/layer"][()]
            layer = (layer_raw.decode() if isinstance(layer_raw, bytes)
                     else str(layer_raw))
            selected_voxels_json.append({
                "index": vid,
                "center": center,
                "corners": corners,
                "layer": layer,
            })

    json_data = {
        "config": {
            "optimize": "nc",
            "N": N,
            "M": M,
            "m": m,
            "W": None,
            "area_ratios": area_ratios,
            "areas": None,
            "seed": seed,
            "min_spacing": min_spacing,
            "per_area": False,
            "muon_weight_k": None,
        },
        "efficiency": efficiency,
        "num_ncs": num_ncs,
        "num_primaries": num_primaries,
        "selected_voxels": selected_voxels_json,
    }
    with open(output_path, "w") as jf:
        json.dump(json_data, jf, indent=2)


# ===================================================================
# Plotting
# ===================================================================

def plot_coverage_vs_f(
    results: list[dict],
    f_values: np.ndarray,
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """
    Plot achieved NC detection efficiency vs greedy fraction f for all setups.

    setup_00 (best) is plotted at f=1, setup_01 (worst) at f=0,
    intermediate setups at their respective f values. Points are sorted by f.
    """
    # Assign f values: index 0 → best (f=1), index 1 → worst (f=0), rest → f_values
    f_per_result = [1.0, 0.0] + list(f_values)
    pairs = sorted(
        zip(f_per_result, [r["achieved"] for r in results]),
        key=lambda x: x[0],
    )
    fs, effs = zip(*pairs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fs, effs, "o-", color="steelblue", markersize=6, linewidth=1.5)
    ax.set_xlabel("f  (probability of greedy pick)")
    ax.set_ylabel("NC detection efficiency")
    ax.set_title("NC coverage vs greedy fraction f")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = output_dir / "coverage_vs_f.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"Coverage plot saved to {out_path}")


# ===================================================================
# Main sampling logic
# ===================================================================

def sample_efficiency_range(
    hdf5_path: str,
    N: int,
    M: int,
    m: int,
    n_configs: int,
    seed: int,
    area_ratios: dict,
    min_spacing: float,
    output_dir: Path,
    sensitivity: bool,
    deltas: list[float] | None,
    correction_strength: float,
    verbose: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 65)
        print("Loading simulation data")
        print("=" * 65)

    (raw_rows, raw_cols, raw_vals,
     voxel_ids, centers, layers,
     num_ncs, num_primaries) = load_raw_sparse(hdf5_path, verbose=verbose)

    num_voxels = len(voxel_ids)

    B = binarize_from_raw(
        raw_rows, raw_cols, raw_vals,
        num_ncs=num_ncs,
        num_voxels=num_voxels,
        layers=layers,
        area_ratios=area_ratios,
        m=m,
        seed=seed,
    )

    if verbose:
        print(f"\nB matrix: {num_ncs} x {num_voxels}, nnz={B.nnz:,}")

    # ------------------------------------------------------------------
    # Step 1: compute endpoints
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 65)
        print("Step 1: computing efficiency bounds")
        print("=" * 65)

    # Best case: greedy
    t0 = time.time()
    sel_best, effs_best, cov_best, _ = greedy_select_nc(
        B, N, M,
        centers=centers, layers=layers, min_spacing=min_spacing,
        verbose=verbose,
    )
    eff_max = effs_best[-1]
    if verbose:
        print(f"\n  Greedy (best):  eff_max = {eff_max:.4%}  ({time.time()-t0:.1f}s)")

    # Worst case
    t0 = time.time()
    sel_worst, effs_worst, cov_worst, _ = greedy_select_nc(
        B, N, M,
        centers=centers, layers=layers, min_spacing=min_spacing,
        worst=True, verbose=verbose,
    )
    eff_min = effs_worst[-1]
    if verbose:
        print(f"  Worst-case:     eff_min = {eff_min:.4%}  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # Step 2: f values for intermediate configs
    # ------------------------------------------------------------------
    # n_configs total: setup_00 (best), setup_01 (worst), then n-2 intermediate.
    # f[k] = probability of picking the best voxel at each greedy step.
    # f = linspace(0, 1, n_configs)[1:-1] → n-2 values strictly in (0, 1).
    f_values = np.linspace(0, 1, n_configs)[1:-1]

    if verbose:
        print(f"\nIntermediate f values ({len(f_values)} configs):")
        print(f"  {np.array2string(f_values, precision=4)}")

    # ------------------------------------------------------------------
    # Step 3: generate configurations
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 65)
        print("Step 3: generating configurations")
        print("=" * 65)

    results: list[dict] = []

    def _add_result(i, final_selected, achieved, info):
        out_path = output_dir / f"setup_{i:02d}.json"
        write_setup_json(
            hdf5_path=hdf5_path,
            voxel_ids=voxel_ids,
            selected_cols=final_selected,
            efficiency=achieved,
            num_ncs=num_ncs,
            num_primaries=num_primaries,
            N=N, M=M, m=m,
            area_ratios=area_ratios,
            seed=seed,
            min_spacing=min_spacing,
            output_path=out_path,
        )
        results.append({
            "name": f"setup_{i:02d}",
            "achieved": achieved,
            "info": info,
            "selected": final_selected,
        })
        if verbose:
            print(f"  setup_{i:02d}  achieved={achieved:.4%}  [{info}]"
                  f"  -> {out_path.name}")

    # setup_00: best (greedy, deterministic)
    t0 = time.time()
    _add_result(0, sel_best, eff_max, "greedy optimum (deterministic)")
    if verbose:
        print(f"    ({time.time()-t0:.1f}s)")

    # setup_01: worst (anti-greedy, deterministic)
    t0 = time.time()
    _add_result(1, sel_worst, eff_min, "worst-case greedy (deterministic)")
    if verbose:
        print(f"    ({time.time()-t0:.1f}s)")

    # setup_02 .. setup_{n-1}: intermediate, one run per f value
    for k, f in enumerate(f_values):
        i = k + 2
        t0 = time.time()
        rng = np.random.default_rng(seed + i)
        final_selected, _, achieved = stochastic_greedy_nc_m1(
            B, N, f, rng, centers, layers, min_spacing, correction_strength
        )
        _add_result(i, final_selected, achieved, f"stochastic f={f:.4f}")
        if verbose:
            print(f"    ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary_path = output_dir / "setup_summary.txt"
    with open(summary_path, "w") as sf:
        sf.write(f"# NC Efficiency Range Sampling\n")
        sf.write(f"# N={N}, M={M}, m={m}, seed={seed}\n")
        sf.write(f"# eff_min={eff_min:.6f}, eff_max={eff_max:.6f}\n")
        sf.write(f"# range={eff_max - eff_min:.6f}\n\n")
        sf.write(f"{'Setup':<12} {'Achieved':>12}  Notes\n")
        sf.write("-" * 60 + "\n")
        for r in results:
            sf.write(f"{r['name']:<12} {r['achieved']:>10.4%}  {r['info']}\n")

    if verbose:
        print(f"\nSummary written to {summary_path}")

    # ------------------------------------------------------------------
    # Coverage plot
    # ------------------------------------------------------------------
    plot_coverage_vs_f(results, f_values, output_dir, verbose=verbose)

    # ------------------------------------------------------------------
    # Optional: sensitivity analysis on the best-case (greedy) config
    # ------------------------------------------------------------------
    if sensitivity:
        if verbose:
            print("\n" + "=" * 65)
            print("Running sensitivity analysis for all setups")
            print("=" * 65)
        for r in results:
            if verbose:
                print(f"\n  -- {r['name']} (eff={r['achieved']:.4%}) --")
            run_sensitivity(
                filepath=hdf5_path,
                N=N,
                m=m,
                area_ratios=area_ratios,
                optimize="nc",
                M=M,
                min_spacing=min_spacing,
                seed=seed,
                deltas=deltas,
                output_dir=str(output_dir / "sensitivity" / r["name"]),
                baseline_selected=r["selected"],
                baseline_eff=r["achieved"],
                raw_rows=raw_rows,
                raw_cols=raw_cols,
                raw_vals=raw_vals,
                voxel_ids=voxel_ids,
                centers=centers,
                layers=layers,
                num_ncs=num_ncs,
                num_primaries=num_primaries,
                verbose=verbose,
            )

    if verbose:
        print("\nDone.")


# ===================================================================
# CLI
# ===================================================================

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate N_CONFIGS PMT selections evenly spanning the NC "
            "detection efficiency range for ncM1m1."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hdf5", type=str, required=True,
                        help="Path to raw SSD HDF5 data file.")
    parser.add_argument("-N", type=int, required=True,
                        help="Number of PMTs per configuration.")
    parser.add_argument("-M", type=int, default=1,
                        help="Multiplicity threshold.")
    parser.add_argument("-m", type=int, default=1,
                        help="Per-voxel hit threshold for binarization.")
    parser.add_argument("--n-configs", type=int, default=12,
                        help="Number of configurations to generate.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed.")
    parser.add_argument("--min-spacing", type=float, default=2 * PMT_RADIUS,
                        help="Minimum voxel spacing on the same layer (mm).")
    parser.add_argument("--correction-strength", type=float, default=2.0,
                        help="Exponent scale for the adaptive f-correction. "
                             "Higher values react more aggressively to "
                             "greedy-pick deficits. Default: 2.0.")
    parser.add_argument("--output-dir", type=str, default="setups",
                        help="Output directory for setup JSON files.")
    parser.add_argument("--pit", type=float, default=None)
    parser.add_argument("--bot", type=float, default=None)
    parser.add_argument("--top", type=float, default=None)
    parser.add_argument("--wall", type=float, default=None)
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run sensitivity analysis on the best-case "
                             "(greedy) config after generating setups.")
    parser.add_argument("--deltas", type=str, default=None,
                        help="Comma-separated area-ratio perturbation values "
                             "for sensitivity analysis (e.g. '-0.1,0.1'). "
                             "Default: -0.20,-0.10,-0.05,+0.05,+0.10,+0.20.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    verbose = not args.quiet

    area_ratios = dict(DEFAULT_AREA_RATIOS)
    if args.pit is not None:
        area_ratios["pit"] = args.pit
    if args.bot is not None:
        area_ratios["bot"] = args.bot
    if args.top is not None:
        area_ratios["top"] = args.top
    if args.wall is not None:
        area_ratios["wall"] = args.wall

    deltas = None
    if args.deltas is not None:
        deltas = [float(d) for d in args.deltas.split(",")]

    sample_efficiency_range(
        hdf5_path=args.hdf5,
        N=args.N,
        M=args.M,
        m=args.m,
        n_configs=args.n_configs,
        seed=args.seed,
        area_ratios=area_ratios,
        min_spacing=args.min_spacing,
        correction_strength=args.correction_strength,
        output_dir=Path(args.output_dir),
        sensitivity=args.sensitivity,
        deltas=deltas,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()