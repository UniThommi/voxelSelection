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
) -> tuple[list[int], np.ndarray, float]:
    """
    Run a stochastic greedy selection for M=1 NC coverage.

    At each step:
    - With probability (1 - f): pick the voxel with highest marginal gain
      (standard greedy choice).
    - With probability f: pick the voxel with lowest marginal gain
      (anti-greedy choice), interpolating toward worst-case behaviour.

    Parameters
    ----------
    B : sparse.csc_matrix
        Binary NC × voxel matrix.
    N : int
        Number of voxels to select.
    f : float
        Random fraction in [0, 1]. f=0 → pure greedy, f=1 → pure random.
    rng : np.random.Generator
        Random number generator.
    centers : np.ndarray, shape (num_voxels, 3)
        Voxel center coordinates for spacing constraint.
    layers : np.ndarray of str
        Layer label per voxel.
    min_spacing : float
        Minimum distance between voxels on the same layer (mm).

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

    for _ in range(N):
        at_m1 = (coverage_counts == 0)
        gains = B.T.dot(at_m1.astype(np.int32))
        if rng.random() < f:
            # Anti-greedy pick: voxel with fewest undetected NCs
            gains[~available] = gains.max() + 1
            best_voxel = int(np.argmin(gains))
        else:
            # Greedy pick: voxel with most undetected NCs
            gains[~available] = -1
            best_voxel = int(np.argmax(gains))

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


def eval_fraction(
    f: float,
    B: sparse.csc_matrix,
    N: int,
    centers: np.ndarray,
    layers: np.ndarray,
    min_spacing: float,
    seeds: list[int],
) -> float:
    """
    Average efficiency over multiple random seeds for a given fraction f.
    """
    effs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        _, _, eff = stochastic_greedy_nc_m1(
            B, N, f, rng, centers, layers, min_spacing
        )
        effs.append(eff)
    return float(np.mean(effs))


def find_fraction_for_target(
    target: float,
    B: sparse.csc_matrix,
    N: int,
    centers: np.ndarray,
    layers: np.ndarray,
    min_spacing: float,
    seeds: list[int],
    tol: float = 0.002,
    max_iter: int = 20,
) -> tuple[float, float]:
    """
    Binary search for the random fraction f that achieves ``target`` efficiency.

    Returns
    -------
    best_f : float
        The fraction closest to producing ``target`` efficiency.
    achieved_eff : float
        Average efficiency at ``best_f``.
    """
    lo, hi = 0.0, 1.0
    eff_lo = eval_fraction(lo, B, N, centers, layers, min_spacing, seeds)
    eff_hi = eval_fraction(hi, B, N, centers, layers, min_spacing, seeds)

    # f=0 → high eff, f=1 → lower eff; target should be in [eff_hi, eff_lo]
    if target >= eff_lo:
        return lo, eff_lo
    if target <= eff_hi:
        return hi, eff_hi

    best_f, best_diff = lo, abs(eff_lo - target)
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        eff_mid = eval_fraction(mid, B, N, centers, layers, min_spacing, seeds)
        diff = abs(eff_mid - target)
        if diff < best_diff:
            best_f, best_diff = mid, diff
        if best_diff <= tol:
            break
        if eff_mid > target:
            lo = mid
        else:
            hi = mid

    return best_f, eval_fraction(best_f, B, N, centers, layers, min_spacing, seeds)


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
# Main sampling logic
# ===================================================================

def sample_efficiency_range(
    hdf5_path: str,
    N: int,
    M: int,
    m: int,
    n_configs: int,
    seed: int,
    avg_seeds: int,
    area_ratios: dict,
    min_spacing: float,
    tol: float,
    output_dir: Path,
    sensitivity: bool,
    deltas: list[float] | None,
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
    # Step 2: define target efficiencies
    # ------------------------------------------------------------------
    targets = [
        eff_min + i * (eff_max - eff_min) / (n_configs - 1)
        for i in range(n_configs)
    ]

    if verbose:
        print(f"\nTarget efficiencies ({n_configs} configs, "
              f"step = {(eff_max - eff_min) / (n_configs - 1):.4%}):")
        for i, t in enumerate(targets):
            print(f"  setup_{i:02d}  target = {t:.4%}")

    # ------------------------------------------------------------------
    # Step 3: generate configurations
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 65)
        print("Step 3: generating intermediate configurations")
        print("=" * 65)

    eval_seeds = [seed + 1000 + k for k in range(avg_seeds)]

    results: list[dict] = []

    for i in range(n_configs):
        target = targets[i]
        t0 = time.time()

        if i == 0:
            # Worst-case config (deterministic)
            final_selected = sel_worst
            achieved = eff_min
            info = "worst-case greedy (deterministic)"
        elif i == n_configs - 1:
            # Best-case config (deterministic)
            final_selected = sel_best
            achieved = eff_max
            info = "greedy optimum (deterministic)"
        else:
            # Binary search on random_fraction
            best_f, achieved = find_fraction_for_target(
                target, B, N, centers, layers, min_spacing,
                seeds=eval_seeds, tol=tol,
            )
            # Generate one final config at best_f with the main seed
            rng = np.random.default_rng(seed + i)
            final_selected, _, achieved_single = stochastic_greedy_nc_m1(
                B, N, best_f, rng, centers, layers, min_spacing
            )
            achieved = achieved_single
            info = f"stochastic f={best_f:.4f}"

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
            "target": target,
            "achieved": achieved,
            "info": info,
            "selected": final_selected,
        })

        if verbose:
            print(f"  setup_{i:02d}  target={target:.4%}  "
                  f"achieved={achieved:.4%}  [{info}]  "
                  f"({time.time()-t0:.1f}s)  -> {out_path.name}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary_path = output_dir / "setup_summary.txt"
    with open(summary_path, "w") as sf:
        sf.write(f"# NC Efficiency Range Sampling\n")
        sf.write(f"# N={N}, M={M}, m={m}, seed={seed}\n")
        sf.write(f"# eff_min={eff_min:.6f}, eff_max={eff_max:.6f}\n")
        sf.write(f"# range={eff_max - eff_min:.6f}\n\n")
        sf.write(f"{'Setup':<12} {'Target':>12} {'Achieved':>12} "
                 f"{'Error':>8}  Notes\n")
        sf.write("-" * 75 + "\n")
        for r in results:
            err = r["achieved"] - r["target"]
            sf.write(f"{r['name']:<12} {r['target']:>10.4%} "
                     f"{r['achieved']:>10.4%} {err:>+8.4%}  "
                     f"{r['info']}\n")

    if verbose:
        print(f"\nSummary written to {summary_path}")

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
    parser.add_argument("--avg-seeds", type=int, default=3,
                        help="Number of seeds to average over per binary "
                             "search evaluation (higher = more stable).")
    parser.add_argument("--tol", type=float, default=0.002,
                        help="Efficiency tolerance for binary search.")
    parser.add_argument("--min-spacing", type=float, default=2 * PMT_RADIUS,
                        help="Minimum voxel spacing on the same layer (mm).")
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
        avg_seeds=args.avg_seeds,
        area_ratios=area_ratios,
        min_spacing=args.min_spacing,
        tol=args.tol,
        output_dir=Path(args.output_dir),
        sensitivity=args.sensitivity,
        deltas=deltas,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()