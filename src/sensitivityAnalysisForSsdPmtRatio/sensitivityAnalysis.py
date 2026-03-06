#!/usr/bin/env python3
"""
Sensitivity Analysis for Area-Ratio Scaling in Greedy Voxel Selection
=====================================================================

Perturbs the SSD/PMT area ratios by a global factor δ and re-runs the
greedy voxel selection for each perturbation.  Compares the resulting
voxel sets against the nominal (δ=0) baseline via:

  - Jaccard similarity J_k(δ) for k ∈ {25, 50, 75, ..., N}
  - Per-area Jaccard similarity
  - Relative coverage change ΔC(δ)

Outputs:
  - JSON with all metrics
  - Plot: J_k(δ) curves (one line per δ, x-axis = k)
  - Plot: ΔC(δ) bar chart

Usage:
    python sensitivity_analysis.py <hdf5_file> -N 300 \\
        --optimize nc -M 1 -m 1 --area-ratio \\
        --output-dir sensitivity_results

    python sensitivity_analysis.py <hdf5_file> -N 300 \\
        --optimize muon-ge77 -W 1 -m 1 --area-ratio \\
        --output-dir sensitivity_results

Author: Ferundo (Thesis project, University of Tübingen)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../core")
# ---------------------------------------------------------------------------
# Import shared constants and functions from greedy script.
# If not importable, they are redefined below.
# ---------------------------------------------------------------------------
try:
    from greedyVoxelSelection import (
        PMT_RADIUS,
        AREA_RATIOS,
        AREA_SURFACES,
        MUON_TIME_WINDOW_MIN_NS,
        MUON_TIME_WINDOW_MAX_NS,
        R_PIT, R_ZYL_BOT, R_ZYL_TOP, R_ZYLINDER,
        Z_ORIGIN, Z_OFFSET, H_ZYLINDER, Z_BASE_GLOBAL,
        Z_CUT_BOT, Z_CUT_TOP, WALL_HEIGHT,
        compute_per_area_N,
        is_valid_pmt_position,
        get_valid_voxel_mask,
        load_muon_data,
        build_muon_index,
        greedy_select_nc,
        greedy_select_muon,
        muon_weight_delta,
        plot_muon_nc_histogram,
    )
    _IMPORTED_GREEDY = True
except ImportError:
    _IMPORTED_GREEDY = False
    print("WARNING: Could not import greedyVoxelSelection. "
          "Ensure it is on PYTHONPATH or in the same directory.",
          file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Core: load_and_binarize with custom ratios
# ---------------------------------------------------------------------------

def load_and_binarize_custom_ratios(
    filepath: str,
    m: int,
    area_ratios: dict[str, float],
    apply_area_ratio: bool = True,
    verbose: bool = False,
) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load HDF5 data and construct sparse binary matrix B with custom
    area ratios.  Identical to load_and_binarize in the greedy script
    but accepts arbitrary ratio values.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    m : int
        Minimum hits per voxel for a NC to count as "seen".
    area_ratios : dict[str, float]
        Area-dependent scaling ratios (SSD / PMT).
    apply_area_ratio : bool
        If True, scale hits before binarization.
    verbose : bool
        Print progress.

    Returns
    -------
    B : sparse.csc_matrix
        Binary (NCs x valid_voxels) matrix.
    voxel_ids : np.ndarray
        Voxel ID strings.
    centers : np.ndarray, shape (num_valid_voxels, 3)
        Voxel center coordinates.
    layers : np.ndarray of str
        Layer label per valid voxel.
    num_primaries : int
        Total number of primary events.
    """
    with h5py.File(filepath, "r") as f:
        voxel_keys = sorted(
            c.decode() if isinstance(c, bytes) else str(c)
            for c in f["target_columns"][:]
        )

        valid_mask = get_valid_voxel_mask(f, voxel_keys, verbose=False)
        valid_keys = [k for k, v in zip(voxel_keys, valid_mask) if v]
        num_voxels = len(valid_keys)

        centers = np.empty((num_voxels, 3), dtype=np.float64)
        layers = np.empty(num_voxels, dtype=object)
        for i, vkey in enumerate(valid_keys):
            centers[i] = f[f"voxels/{vkey}/center"][:]
            layer_raw = f[f"voxels/{vkey}/layer"][()]
            layers[i] = (layer_raw.decode() if isinstance(layer_raw, bytes)
                         else str(layer_raw))

        target_columns = [
            c.decode() if isinstance(c, bytes) else str(c)
            for c in f["target_columns"][:]
        ]
        target_col_to_idx = {c: i for i, c in enumerate(target_columns)}
        valid_col_indices_arr = np.array(
            [target_col_to_idx[k] for k in valid_keys]
        )

        num_ncs = f["target_matrix"].shape[0]
        num_primaries = int(f["primaries"][()])

        if verbose:
            print(f"  Loading: {num_ncs} NCs, {num_voxels} valid voxels")

        # Precompute per-column ratio vector
        if apply_area_ratio:
            ratio_vec = np.array(
                [area_ratios.get(layers[c], 1.0) for c in range(num_voxels)],
                dtype=np.float32,
            )

        # Row-block reading
        BATCH_SIZE = 1000
        rows_list: list[np.ndarray] = []
        cols_list: list[np.ndarray] = []
        target_dset = f["target_matrix"]

        for row_start in range(0, num_ncs, BATCH_SIZE):
            row_end = min(row_start + BATCH_SIZE, num_ncs)
            block = target_dset[row_start:row_end, :]
            block_valid = block[:, valid_col_indices_arr]

            if apply_area_ratio:
                mask = (block_valid / ratio_vec) >= m
            else:
                mask = block_valid >= m

            nc_idx, col_idx = np.nonzero(mask)
            if len(nc_idx) > 0:
                rows_list.append(nc_idx.astype(np.int64) + row_start)
                cols_list.append(col_idx.astype(np.int32))

        if len(rows_list) > 0:
            all_rows = np.concatenate(rows_list)
            all_cols = np.concatenate(cols_list)
        else:
            all_rows = np.array([], dtype=np.int64)
            all_cols = np.array([], dtype=np.int32)
        all_data = np.ones(len(all_rows), dtype=np.int8)

        B = sparse.coo_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(num_ncs, num_voxels),
            dtype=np.int8,
        ).tocsc()

        voxel_ids = np.array(valid_keys)

    return B, voxel_ids, centers, layers, num_primaries


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    union = len(set_a | set_b)
    if union == 0:
        return 1.0
    return len(set_a & set_b) / union


def compute_jaccard_curve(
    baseline_order: list[int],
    perturbed_order: list[int],
    k_values: list[int],
) -> list[float]:
    """
    Compute Jaccard similarity for top-k voxels at each k.

    Parameters
    ----------
    baseline_order : list[int]
        Voxel column indices from nominal greedy (in selection order).
    perturbed_order : list[int]
        Voxel column indices from perturbed greedy (in selection order).
    k_values : list[int]
        Values of k at which to compute J_k.

    Returns
    -------
    jaccards : list[float]
        J_k for each k in k_values.
    """
    jaccards = []
    for k in k_values:
        set_base = set(baseline_order[:k])
        set_pert = set(perturbed_order[:k])
        jaccards.append(jaccard(set_base, set_pert))
    return jaccards


def compute_per_area_jaccard(
    baseline_order: list[int],
    perturbed_order: list[int],
    layers: np.ndarray,
    N: int,
) -> dict[str, float]:
    """
    Compute Jaccard similarity per detector area for the full
    selection of N voxels.

    Parameters
    ----------
    baseline_order : list[int]
        Baseline voxel indices.
    perturbed_order : list[int]
        Perturbed voxel indices.
    layers : np.ndarray
        Layer label per voxel (indexed by column index).
    N : int
        Number of selected voxels.

    Returns
    -------
    per_area : dict[str, float]
        Jaccard per area.
    """
    base_set = set(baseline_order[:N])
    pert_set = set(perturbed_order[:N])
    per_area = {}

    for area in ["pit", "bot", "top", "wall"]:
        base_area = {v for v in base_set if layers[v] == area}
        pert_area = {v for v in pert_set if layers[v] == area}
        per_area[area] = jaccard(base_area, pert_area)

    return per_area


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_jaccard_curves(
    results: dict,
    k_values: list[int],
    output_path: Path,
    optimize_mode: str,
) -> None:
    """
    Plot J_k(δ) curves: one line per δ, x-axis = k.

    Parameters
    ----------
    results : dict
        Keys are delta values (float), values contain "jaccard_curve".
    k_values : list[int]
        k values for x-axis.
    output_path : Path
        Where to save.
    optimize_mode : str
        "nc" or "muon-ge77" for title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.RdYlGn_r
    deltas_sorted = sorted(results.keys(), key=lambda d: abs(d))

    for i, delta in enumerate(deltas_sorted):
        color = cmap(abs(delta) / 0.25)
        label = f"δ = {delta:+.0%}"
        jvals = results[delta]["jaccard_curve"]
        ax.plot(k_values, jvals, marker="o", markersize=3,
                label=label, color=color, linewidth=1.5)

    ax.set_xlabel("k (top-k voxels)", fontsize=12)
    ax.set_ylabel("Jaccard Similarity $J_k(\\delta)$", fontsize=12)
    ax.set_title(f"Sensitivity Analysis: Jaccard vs. Top-k "
                 f"(mode: {optimize_mode})", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(k_values[0] - 5, k_values[-1] + 5)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5,
               label="J = 0.8 threshold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Jaccard curve plot saved to {output_path}")


def plot_coverage_change(
    results: dict,
    output_path: Path,
    optimize_mode: str,
) -> None:
    """
    Bar chart of relative coverage change ΔC(δ).

    Parameters
    ----------
    results : dict
        Keys are delta values (float), values contain "delta_coverage".
    output_path : Path
        Where to save.
    optimize_mode : str
        "nc" or "muon-ge77" for title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    deltas_sorted = sorted(results.keys())
    delta_c = [results[d]["delta_coverage"] for d in deltas_sorted]
    labels = [f"{d:+.0%}" for d in deltas_sorted]
    colors = ["#d32f2f" if dc < -0.01
              else "#388e3c" if dc > 0.01
              else "#757575" for dc in delta_c]

    bars = ax.bar(labels, [dc * 100 for dc in delta_c], color=colors,
                  edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Ratio perturbation δ", fontsize=12)
    ax.set_ylabel("ΔC(δ) [%]", fontsize=12)
    ax.set_title(f"Relative Coverage Change "
                 f"(mode: {optimize_mode})", fontsize=13)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars
    for bar, dc in zip(bars, delta_c):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{dc:+.3%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Coverage change plot saved to {output_path}")


def plot_per_area_jaccard(
    results: dict,
    output_path: Path,
    optimize_mode: str,
) -> None:
    """
    Grouped bar chart: per-area Jaccard for each δ.

    Parameters
    ----------
    results : dict
        Keys are delta values, values contain "per_area_jaccard".
    output_path : Path
        Where to save.
    optimize_mode : str
        "nc" or "muon-ge77" for title.
    """
    areas = ["pit", "bot", "top", "wall"]
    deltas_sorted = sorted(results.keys())
    n_deltas = len(deltas_sorted)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_deltas)
    width = 0.18

    area_colors = {"pit": "#1976d2", "bot": "#388e3c",
                   "top": "#f57c00", "wall": "#7b1fa2"}

    for i, area in enumerate(areas):
        vals = [results[d]["per_area_jaccard"][area] for d in deltas_sorted]
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width, label=area.upper(),
               color=area_colors[area], edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Ratio perturbation δ", fontsize=12)
    ax.set_ylabel("Jaccard Similarity", fontsize=12)
    ax.set_title(f"Per-Area Jaccard Similarity "
                 f"(mode: {optimize_mode})", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:+.0%}" for d in deltas_sorted])
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Per-area Jaccard plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_single_greedy(
    filepath: str,
    N: int,
    m: int,
    area_ratios: dict[str, float],
    optimize: str,
    M: int = 1,
    W: int = 1,
    min_spacing: float = 0.0,
    per_area: bool = False,
    muon_weight_k: float | None = None,
    verbose: bool = False,
) -> tuple[list[int], float]:
    """
    Run a single greedy optimization with given area ratios.

    Returns the selected voxel column indices (in order) and the
    final detection efficiency/coverage.

    Memory note: B is constructed and released within this function
    to avoid accumulating matrices across perturbation runs.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file.
    N : int
        Number of voxels to select.
    m : int
        Hit threshold per voxel.
    area_ratios : dict[str, float]
        Area-dependent scaling ratios.
    optimize : str
        "nc" or "muon-ge77".
    M : int
        NC multiplicity threshold (nc mode).
    W : int
        Muon coincidence threshold (muon-ge77 mode).
    min_spacing : float
        Minimum spacing between selected voxels (mm).
    per_area : bool
        If True, optimize each area independently.
    verbose : bool
        Print progress.

    Returns
    -------
    selected_cols : list[int]
        Selected voxel column indices in greedy order.
    final_efficiency : float
        Final detection efficiency.
    """
    B, voxel_ids, centers, layers, num_primaries = (
        load_and_binarize_custom_ratios(
            filepath, m=m, area_ratios=area_ratios,
            apply_area_ratio=True, verbose=verbose,
        )
    )

    # Load muon IDs for NC-mode muon-awareness (no time filter)
    nc_muon_weight_data = None
    if optimize == "nc" and muon_weight_k is not None:
        with h5py.File(filepath, "r") as f:
            phi_columns = [c.decode() if isinstance(c, bytes) else str(c)
                           for c in f["phi_columns"][:]]
            phi_col_idx = {name: i for i, name in enumerate(phi_columns)}
            phi_matrix = f["phi_matrix"]
            nc_global_muon_id = phi_matrix[:, phi_col_idx["global_muon_id"]].astype(np.int64)

        if len(nc_global_muon_id) != B.shape[0]:
            raise RuntimeError(
                f"Muon data length ({len(nc_global_muon_id)}) != "
                f"num_ncs ({B.shape[0]})"
            )
        unique_muon_ids = np.unique(nc_global_muon_id)
        num_muons_nc = len(unique_muon_ids)
        nc_to_muon_local_nc = np.searchsorted(
            unique_muon_ids, nc_global_muon_id
        ).astype(np.int32)

        nc_muon_weight_data = {
            "nc_to_muon_local": nc_to_muon_local_nc,
            "num_muons": num_muons_nc,
        }

    if optimize == "muon-ge77":
        global_muon_id, nc_time_ns, nc_flag_ge77 = load_muon_data(
            filepath, num_ncs=B.shape[0], verbose=False,
        )
        (nc_to_muon_local, muon_nc_counts, ge77_muon_global_ids,
         eligible_nc_mask, num_ge77_muons) = build_muon_index(
            global_muon_id, nc_time_ns, nc_flag_ge77, verbose=False,
        )

    if per_area:
        allocation = compute_per_area_N(N, verbose=False)
        all_selected: list[int] = []

        if optimize == "muon-ge77":
            shared_nc_detected = np.zeros(B.shape[0], dtype=bool)
            shared_muon_counts = np.zeros(num_ge77_muons, dtype=np.int32)

        for area_name in ["pit", "bot", "top", "wall"]:
            n_area = allocation[area_name]
            if n_area == 0:
                continue

            area_mask = (layers == area_name)
            area_indices = np.where(area_mask)[0]
            if len(area_indices) == 0:
                continue

            B_area = B[:, area_indices]
            centers_area = centers[area_indices]
            layers_area = layers[area_indices]

            if optimize == "nc":
                sel_local, _, _, _ = greedy_select_nc(
                    B_area, N=n_area, M=M,
                    centers=centers_area, layers=layers_area,
                    min_spacing=min_spacing, verbose=False,
                )
                sel_global = [int(area_indices[i]) for i in sel_local]
                all_selected.extend(sel_global)

            elif optimize == "muon-ge77":
                sel_local, _, nc_det, muon_det = greedy_select_muon(
                    B_area, N=n_area, W=W,
                    nc_to_muon_local=nc_to_muon_local,
                    eligible_nc_mask=eligible_nc_mask,
                    num_ge77_muons=num_ge77_muons,
                    centers=centers_area, layers=layers_area,
                    min_spacing=min_spacing,
                    muon_weight_k=muon_weight_k,
                    verbose=False,
                )
                sel_global = [int(area_indices[i]) for i in sel_local]
                all_selected.extend(sel_global)
                shared_nc_detected |= nc_det
                shared_muon_counts += muon_det

            # Free sub-matrix
            del B_area

        selected_cols = all_selected

        if optimize == "nc":
            coverage_counts = np.zeros(B.shape[0], dtype=np.int16)
            for col in selected_cols:
                s, e = B.indptr[col], B.indptr[col + 1]
                coverage_counts[B.indices[s:e]] += 1
            final_eff = int(np.sum(coverage_counts >= M)) / B.shape[0]
        else:
            n_det = int(np.sum(shared_muon_counts >= W))
            final_eff = n_det / num_ge77_muons if num_ge77_muons > 0 else 0.0

    else:
        # Global optimization
        if optimize == "nc":
            mw_kwargs = {}
            if nc_muon_weight_data is not None:
                mw_kwargs = {
                    "muon_weight_k": muon_weight_k,
                    "nc_to_muon_local": nc_muon_weight_data["nc_to_muon_local"],
                    "num_muons": nc_muon_weight_data["num_muons"],
                }
            selected_cols, effs, _, _ = greedy_select_nc(
                B, N=N, M=M,
                centers=centers, layers=layers,
                min_spacing=min_spacing, verbose=False,
                **mw_kwargs,
            )
            final_eff = effs[-1] if effs else 0.0

        elif optimize == "muon-ge77":
            selected_cols, effs, _, _ = greedy_select_muon(
                B, N=N, W=W,
                nc_to_muon_local=nc_to_muon_local,
                eligible_nc_mask=eligible_nc_mask,
                num_ge77_muons=num_ge77_muons,
                centers=centers, layers=layers,
                min_spacing=min_spacing,
                muon_weight_k=muon_weight_k,
                verbose=False,
            )
            final_eff = effs[-1] if effs else 0.0

    # Explicitly free large objects
    del B

    return selected_cols, final_eff


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for area-ratio scaling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("hdf5_file", type=str,
                        help="Path to the HDF5 data file.")
    parser.add_argument("-N", type=int, required=True,
                        help="Number of voxels to select.")
    parser.add_argument("--optimize", type=str, default="nc",
                        choices=["nc", "muon-ge77"],
                        help="Optimization target.")
    parser.add_argument("-M", type=int, default=1,
                        help="NC multiplicity threshold (nc mode).")
    parser.add_argument("-m", type=int, default=1,
                        help="Hit threshold per voxel.")
    parser.add_argument("-W", type=int, default=1,
                        help="Muon coincidence threshold (muon-ge77 mode).")
    parser.add_argument("--no-spacing", action="store_true",
                        help="Disable minimum spacing constraint.")
    parser.add_argument("--per-area", action="store_true",
                        help="Optimize each area independently.")
    parser.add_argument("--output-dir", type=str, default="sensitivity_results",
                        help="Directory for output files.")
    parser.add_argument("--muon-weight", type=float, default=None, metavar="K",
                        help="Enable muon-level diminishing-returns weighting "
                             "with saturation constant k. Uses f(d) = 1 - exp(-d/k). "
                             "For 90%% saturation at 10 detected NCs, use k ≈ 4.34. "
                             "Not compatible with --per-area.")
    parser.add_argument("--deltas", type=str, default=None,
                        help="Comma-separated delta values (e.g. '-0.20,-0.10,...'). "
                             "Default: -0.20,-0.10,-0.05,+0.05,+0.10,+0.20")

    args = parser.parse_args(argv)

    if args.optimize == "muon-ge77" and args.M != 1:
        args.M = 1

    min_spacing = 0.0 if args.no_spacing else 2 * PMT_RADIUS

    if args.muon_weight is not None and args.per_area:
        parser.error("--muon-weight and --per-area cannot be combined. "
                     "This combination is not implemented.")

    # Parse deltas
    if args.deltas is not None:
        deltas = [float(d.strip()) for d in args.deltas.split(",")]
    else:
        deltas = [-0.20, -0.10, -0.05, +0.05, +0.10, +0.20]

    # k values for Jaccard curve (steps of 25 up to N)
    k_values = list(range(25, args.N + 1, 25))
    if k_values[-1] != args.N:
        k_values.append(args.N)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Sensitivity Analysis — Area Ratio Perturbation")
    print("=" * 65)
    print(f"  Mode:        {args.optimize}")
    print(f"  N:           {args.N}")
    print(f"  M:           {args.M}")
    print(f"  m:           {args.m}")
    if args.optimize == "muon-ge77":
        print(f"  W:           {args.W}")
    print(f"  Per-area:    {args.per_area}")
    print(f"  Spacing:     {min_spacing:.0f} mm")
    print(f"  Deltas:      {deltas}")
    print(f"  k-values:    {k_values}")
    print(f"  Nominal ratios: {AREA_RATIOS}")
    if args.muon_weight is not None:
        print(f"  Muon weight: k = {args.muon_weight:.4f}")
    print(f"  Output dir:  {output_dir}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Nominal (baseline) run with δ = 0
    # ------------------------------------------------------------------
    print(f"[Baseline] Running nominal greedy (δ = 0)...")
    t0 = time.time()

    baseline_selected, baseline_eff = run_single_greedy(
        filepath=args.hdf5_file,
        N=args.N, m=args.m,
        area_ratios=AREA_RATIOS,
        optimize=args.optimize,
        M=args.M, W=args.W,
        min_spacing=min_spacing,
        per_area=args.per_area,
        muon_weight_k=args.muon_weight,
        verbose=True,
    )

    t_baseline = time.time() - t0
    print(f"  Baseline efficiency: {baseline_eff:.4%}")
    print(f"  Baseline time: {t_baseline:.1f}s")

    # We need the layers array for per-area Jaccard.
    # Re-load once (lightweight, only voxel metadata).
    with h5py.File(args.hdf5_file, "r") as f:
        voxel_keys = sorted(
            c.decode() if isinstance(c, bytes) else str(c)
            for c in f["target_columns"][:]
        )
        valid_mask = get_valid_voxel_mask(f, voxel_keys, verbose=False)
        valid_keys = [k for k, v in zip(voxel_keys, valid_mask) if v]
        num_voxels = len(valid_keys)
        layers_all = np.empty(num_voxels, dtype=object)
        for i, vkey in enumerate(valid_keys):
            layer_raw = f[f"voxels/{vkey}/layer"][()]
            layers_all[i] = (layer_raw.decode() if isinstance(layer_raw, bytes)
                             else str(layer_raw))

    # ------------------------------------------------------------------
    # Step 2: Perturbed runs
    # ------------------------------------------------------------------
    results: dict[float, dict] = {}

    for idx, delta in enumerate(deltas):
        perturbed_ratios = {
            area: ratio * (1.0 + delta)
            for area, ratio in AREA_RATIOS.items()
        }

        print(f"\n[{idx+1}/{len(deltas)}] δ = {delta:+.0%} | "
              f"Ratios: { {a: f'{r:.4f}' for a, r in perturbed_ratios.items()} }")
        t0 = time.time()

        pert_selected, pert_eff = run_single_greedy(
            filepath=args.hdf5_file,
            N=args.N, m=args.m,
            area_ratios=perturbed_ratios,
            optimize=args.optimize,
            M=args.M, W=args.W,
            min_spacing=min_spacing,
            per_area=args.per_area,
            muon_weight_k=args.muon_weight,
            verbose=False,
        )

        dt = time.time() - t0
        print(f"  Efficiency: {pert_eff:.4%} "
              f"(ΔC = {(pert_eff - baseline_eff) / baseline_eff:+.4%})")
        print(f"  Time: {dt:.1f}s")

        # Compute metrics
        j_curve = compute_jaccard_curve(
            baseline_selected, pert_selected, k_values,
        )
        per_area_j = compute_per_area_jaccard(
            baseline_selected, pert_selected, layers_all, args.N,
        )
        delta_cov = ((pert_eff - baseline_eff) / baseline_eff
                     if baseline_eff > 0 else 0.0)

        results[delta] = {
            "selected_cols": pert_selected,
            "efficiency": pert_eff,
            "delta_coverage": delta_cov,
            "jaccard_curve": j_curve,
            "per_area_jaccard": per_area_j,
            "perturbed_ratios": perturbed_ratios,
            "runtime_s": dt,
        }

        # Print Jaccard summary
        j_50 = j_curve[k_values.index(50)] if 50 in k_values else None
        j_N = j_curve[-1]
        print(f"  J_50 = {j_50:.3f}" if j_50 is not None else "")
        print(f"  J_{args.N} = {j_N:.3f}")
        print(f"  Per-area J: {per_area_j}")

    # ------------------------------------------------------------------
    # Step 3: Save results
    # ------------------------------------------------------------------
    # JSON summary (no large arrays, just metrics)
    json_output = output_dir / f"sensitivity_{args.optimize}_N{args.N}.json"
    json_data = {
        "config": {
            "optimize": args.optimize,
            "N": args.N,
            "M": args.M,
            "m": args.m,
            "W": args.W,
            "per_area": args.per_area,
            "min_spacing": min_spacing,
            "nominal_ratios": AREA_RATIOS,
            "muon_weight_k": args.muon_weight,
            "deltas": deltas,
            "k_values": k_values,
        },
        "baseline": {
            "efficiency": baseline_eff,
            "selected_cols": baseline_selected,
            "runtime_s": t_baseline,
        },
        "perturbations": {
            f"{d:+.2f}": {
                "efficiency": r["efficiency"],
                "delta_coverage": r["delta_coverage"],
                "jaccard_curve": r["jaccard_curve"],
                "per_area_jaccard": r["per_area_jaccard"],
                "perturbed_ratios": r["perturbed_ratios"],
                "runtime_s": r["runtime_s"],
                "selected_cols": r["selected_cols"],
            }
            for d, r in results.items()
        },
    }

    with open(json_output, "w") as jf:
        json.dump(json_data, jf, indent=2)
    print(f"\nResults saved to {json_output}")

    # ------------------------------------------------------------------
    # Step 4: Plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")

    plot_jaccard_curves(
        results, k_values,
        output_dir / f"sensitivity_jaccard_{args.optimize}_N{args.N}.png",
        args.optimize,
    )

    plot_coverage_change(
        results,
        output_dir / f"sensitivity_coverage_{args.optimize}_N{args.N}.png",
        args.optimize,
    )

    plot_per_area_jaccard(
        results,
        output_dir / f"sensitivity_perarea_{args.optimize}_N{args.N}.png",
        args.optimize,
    )

    # ------------------------------------------------------------------
    # Step 5: Summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print(f"{'=' * 65}")
    print(f"{'δ':>8} | {'Efficiency':>11} | {'ΔC':>8} | "
          f"{'J_50':>6} | {'J_100':>6} | {'J_200':>6} | "
          f"{'J_' + str(args.N):>6}")
    print("-" * 65)

    # Baseline row
    print(f"{'0':>8} | {baseline_eff:>11.4%} | {'—':>8} | "
          f"{'—':>6} | {'—':>6} | {'—':>6} | {'—':>6}")

    for d in sorted(results.keys()):
        r = results[d]
        jc = r["jaccard_curve"]

        def _get_jk(k: int) -> str:
            if k in k_values:
                return f"{jc[k_values.index(k)]:.3f}"
            return "—"

        print(f"{d:>+8.0%} | {r['efficiency']:>11.4%} | "
              f"{r['delta_coverage']:>+8.3%} | "
              f"{_get_jk(50):>6} | {_get_jk(100):>6} | "
              f"{_get_jk(200):>6} | {_get_jk(args.N):>6}")

    total_time = t_baseline + sum(r["runtime_s"] for r in results.values())
    print(f"\nTotal runtime: {total_time:.0f}s ({total_time/3600:.1f}h)")


if __name__ == "__main__":
    main()