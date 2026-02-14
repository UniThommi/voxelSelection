#!/usr/bin/env python3
"""
Greedy Voxel Selection for Neutron Capture Detection
=====================================================

Solves the Maximum Coverage / Partial Set Multi-Cover Problem:
Given a set of neutron captures (NCs) and optical detector voxels,
select N voxels to maximize the number of NCs detected, where detection
requires at least M voxels exceeding a hit threshold m.

Theoretical Background
----------------------
For M=1 this is the classical Maximum k-Coverage Problem (NP-hard).
The greedy algorithm provides a (1 - 1/e) ≈ 0.632 approximation guarantee
[Nemhauser, Wolsey & Fisher, "An Analysis of Approximations for Maximizing
Submodular Set Functions", Math. Programming 14(1), 1978, Theorem 4.3].

For M>1 this is the Partial Set Multi-Cover Problem. Submodularity of the
objective breaks (marginal gains can increase), so the (1-1/e) guarantee
does not hold directly. However, the greedy remains the best polynomial-time
approach, as shown in the framework of concave coverage problems
[Barman, Fawzi & Fermé, "Tight Approximation Guarantees for Concave
Coverage Problems", Math. Programming 201, 2023, Theorem 1.1].

Algorithm
---------
Greedy iteration: In each of N rounds, select the voxel whose addition
causes the largest increase in the number of detected NCs. A NC contributes
to the marginal gain of voxel v if and only if:
  (a) its current coverage count c[i] == M - 1  (one short of threshold)
  (b) voxel v sees this NC (B[i,v] == 1)
This is the exact marginal gain of the objective function f(S).

Complexity: O(N * num_voxels * avg_nnz_per_col) per full evaluation.
With sparse CSC format, each iteration is O(num_voxels * avg_nnz_per_col).

Usage
-----
    python greedy_voxel_selection.py <hdf5_file> --N 50 --M 2 --m 1

Author: Ferundo (Thesis project, University of Tübingen)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from scipy import sparse

# ---------------------------------------------------------------------------
# Detector geometry constants (mm)
# ---------------------------------------------------------------------------
PMT_RADIUS = 131          # PMT physical radius
R_PIT = 3800              # Pit radius
R_ZYL_BOT = 3950          # Inner radius of bottom ring
R_ZYL_TOP = 1200          # Inner radius of top ring
R_ZYLINDER = 4300         # Apothem (outer radius of rings / wall)
Z_ORIGIN = 20
Z_OFFSET = -5000
H_ZYLINDER = 8900 - 1    # h - 1
Z_BASE_GLOBAL = Z_ORIGIN + Z_OFFSET


def is_valid_pmt_position(
    center: np.ndarray,
    layer: str,
    pmt_r: float = PMT_RADIUS,
) -> bool:
    """
    Check whether a PMT of given radius can be physically placed at
    the voxel center without protruding beyond the detector boundaries.

    Parameters
    ----------
    center : np.ndarray, shape (3,)
        Voxel center coordinates (x, y, z) in mm.
    layer : str
        Detector layer: "pit", "bot", "top", or "wall".
    pmt_r : float
        PMT radius in mm.

    Returns
    -------
    bool
        True if the PMT fits within the layer boundaries.
    """
    x, y, z = center
    r_center = np.sqrt(x**2 + y**2)

    if layer == "pit":
        return r_center + pmt_r <= R_PIT
    elif layer == "bot":
        return (r_center - pmt_r >= R_ZYL_BOT) and (r_center + pmt_r <= R_ZYLINDER)
    elif layer == "top":
        return (r_center - pmt_r >= R_ZYL_TOP) and (r_center + pmt_r <= R_ZYLINDER)
    elif layer == "wall":
        z_min_allowed = Z_BASE_GLOBAL + pmt_r
        z_max_allowed = Z_BASE_GLOBAL + H_ZYLINDER - pmt_r
        return z_min_allowed <= z <= z_max_allowed

    return False


def get_valid_voxel_mask(
    f: h5py.File,
    voxel_keys: list[str],
    verbose: bool = True,
) -> np.ndarray:
    """
    Read voxel center and layer from HDF5 and determine which voxels
    can physically host a PMT.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle.
    voxel_keys : list[str]
        Sorted list of voxel ID strings.
    verbose : bool
        Print filtering statistics.

    Returns
    -------
    valid_mask : np.ndarray of bool, shape (num_voxels,)
        True for voxels where a PMT can be placed.
    """
    num_voxels = len(voxel_keys)
    valid_mask = np.zeros(num_voxels, dtype=bool)
    layer_counts = {"pit": [0, 0], "bot": [0, 0], "top": [0, 0], "wall": [0, 0]}

    for col_idx, vkey in enumerate(voxel_keys):
        center = f[f"voxels/{vkey}/center"][:]
        layer_raw = f[f"voxels/{vkey}/layer"][()]
        # Handle bytes vs str (h5py can return either)
        layer = layer_raw.decode() if isinstance(layer_raw, bytes) else str(layer_raw)

        if layer in layer_counts:
            layer_counts[layer][0] += 1

        if is_valid_pmt_position(center, layer):
            valid_mask[col_idx] = True
            if layer in layer_counts:
                layer_counts[layer][1] += 1

    if verbose:
        total = num_voxels
        valid = int(valid_mask.sum())
        print(f"PMT placement filter (r_pmt = {PMT_RADIUS} mm):")
        for ly in ["pit", "bot", "top", "wall"]:
            tot, val = layer_counts[ly]
            print(f"  {ly:>4}: {val}/{tot} valid")
        print(f"  Total: {valid}/{total} valid "
              f"({total - valid} filtered out)")

    return valid_mask


def load_and_binarize(
    filepath: str,
    m: int = 1,
    verbose: bool = True,
) -> tuple[sparse.csc_matrix, np.ndarray, int]:
    """
    Load HDF5 data and construct sparse binary matrix B.
    
    B[i, j] = 1 iff voxel j registered >= m hits for NC i.
    Only voxels where a PMT can physically be placed are included.
    
    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    m : int
        Minimum number of hits per voxel for a NC to count as "seen".
    verbose : bool
        Print progress information.
    
    Returns
    -------
    B : sparse.csc_matrix
        Binary (NCs x valid_voxels) matrix in CSC format.
        CSC is chosen because the greedy algorithm accesses columns
        (= individual voxels) in each iteration.
    voxel_ids : np.ndarray
        Array of voxel ID strings, mapping column index -> voxel name.
    num_primaries : int
        Total number of primary events (for efficiency calculation).
    """
    with h5py.File(filepath, "r") as f:
        # Get voxel IDs from the target group
        voxel_keys = sorted(f["target"].keys())

        # --- PMT placement filter ---
        valid_mask = get_valid_voxel_mask(f, voxel_keys, verbose=verbose)
        valid_keys = [k for k, v in zip(voxel_keys, valid_mask) if v]
        num_voxels = len(valid_keys)
        
        # Determine number of NCs from first dataset
        num_ncs = f["target"][valid_keys[0]].shape[0]
        num_primaries = int(f["primaries"][()])
        
        if verbose:
            print(f"Loading data: {num_ncs} NCs, {num_voxels} valid voxels, "
                  f"{num_primaries} primaries")
            print(f"Binarization threshold m = {m}")
        
        # Build sparse matrix column by column.
        # We collect COO-format data (row_indices, col_indices, values)
        # then convert to CSC at the end. This is the most memory-efficient
        # way to construct a sparse matrix from columnar data.
        rows_list: list[np.ndarray] = []
        cols_list: list[np.ndarray] = []
        
        for col_idx, voxel_key in enumerate(valid_keys):
            if verbose and col_idx % 1000 == 0:
                print(f"  Loading voxel {col_idx}/{num_voxels}...", end="\r")
            
            hits = f["target"][voxel_key][:]
            
            # Binarize: which NCs have >= m hits in this voxel?
            # This is where the threshold m enters the pipeline.
            mask = hits >= m
            nc_indices = np.where(mask)[0]
            
            if len(nc_indices) > 0:
                rows_list.append(nc_indices)
                cols_list.append(np.full(len(nc_indices), col_idx, dtype=np.int32))
        
        if verbose:
            print()  # newline after progress
        
        # Concatenate and build sparse matrix
        all_rows = np.concatenate(rows_list)
        all_cols = np.concatenate(cols_list)
        all_data = np.ones(len(all_rows), dtype=np.int8)
        
        B = sparse.coo_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(num_ncs, num_voxels),
            dtype=np.int8,
        ).tocsc()
        
        nnz = B.nnz
        density = nnz / (num_ncs * num_voxels) * 100
        mem_mb = (B.data.nbytes + B.indices.nbytes + B.indptr.nbytes) / 1e6
        
        if verbose:
            print(f"Sparse matrix: {num_ncs} x {num_voxels}, "
                  f"nnz = {nnz:,} ({density:.3f}%), "
                  f"memory = {mem_mb:.1f} MB")
        
        voxel_ids = np.array(valid_keys)
    
    return B, voxel_ids, num_primaries


def greedy_select(
    B: sparse.csc_matrix,
    N: int,
    M: int,
    verbose: bool = True,
) -> tuple[list[int], list[float], np.ndarray]:
    """
    Greedy voxel selection maximizing threshold coverage.
    
    In each of N iterations, we select the voxel v* that maximizes
    the marginal gain:
    
        gain(v) = |{i : c[i] == M-1  AND  B[i,v] == 1}|
    
    where c[i] is the current coverage count of NC i (number of already-
    selected voxels that see NC i).
    
    Why only c[i] == M-1?
    ---------------------
    The objective f(S) = |{i : c[i] >= M}| changes by exactly the number
    of NCs that cross the threshold M when v is added. Since B is binary,
    adding v increments c[i] by at most 1. So only NCs at c[i] == M-1
    can cross the threshold. NCs with c[i] < M-1 need more voxels,
    and NCs with c[i] >= M are already detected.
    
    Submodularity note (M=1):
    -------------------------
    For M=1, gain(v) = |{i : c[i]==0 AND B[i,v]==1}|, i.e. the number
    of NEW NCs covered. This is exactly the standard greedy for Maximum
    k-Coverage with (1-1/e) guarantee [Nemhauser et al., 1978].
    
    Non-submodularity note (M>1):
    -----------------------------
    For M>1, the marginal gain of a voxel can INCREASE as more voxels
    are selected (NCs move from c < M-1 to c == M-1, becoming "available"
    for detection). This violates diminishing returns. The greedy is still
    the best polynomial approach [Barman et al., Math. Programming 201, 2023].
    
    Parameters
    ----------
    B : sparse.csc_matrix
        Binary matrix (NCs x voxels) in CSC format.
    N : int
        Number of voxels to select.
    M : int
        Multiplicity threshold for detection.
    verbose : bool
        Print per-iteration progress.
    
    Returns
    -------
    selected : list[int]
        Column indices of selected voxels (in selection order).
    efficiencies : list[float]
        Cumulative detection efficiency after each selection step.
    coverage_counts : np.ndarray
        Final coverage count per NC.
    """
    num_ncs, num_voxels = B.shape
    
    if N > num_voxels:
        raise ValueError(f"N={N} exceeds number of voxels ({num_voxels})")
    
    # c[i] = number of selected voxels that see NC i.
    # Using int16 is safe: max possible value is N, which is << 32767.
    coverage_counts = np.zeros(num_ncs, dtype=np.int16)
    
    # Track which voxels are still candidates
    available = np.ones(num_voxels, dtype=bool)
    
    selected: list[int] = []
    efficiencies: list[float] = []
    
    if verbose:
        print(f"\nGreedy selection: N={N}, M={M}")
        print(f"{'Step':>4} | {'Voxel':>8} | {'Gain':>8} | "
              f"{'Detected':>10} | {'Efficiency':>10}")
        print("-" * 60)
    
    for step in range(N):
        t0 = time.time()
        
        best_gain = -1
        best_voxel = -1
        
        # ---------------------------------------------------------------
        # Core of the greedy: evaluate marginal gain for all candidates.
        #
        # For each candidate voxel v (column in B), we need:
        #   gain(v) = number of NCs where c[i] == M-1 AND B[i,v] == 1
        #
        # Implementation via sparse CSC:
        #   B.indices[B.indptr[v]:B.indptr[v+1]] gives the row indices
        #   (NC indices) where B[:,v] is nonzero. We count how many of
        #   these have coverage_counts == M-1.
        # ---------------------------------------------------------------
        
        # Precompute the boolean mask of NCs at threshold boundary.
        # This avoids re-checking coverage_counts[row] == M-1 for every
        # voxel — instead we do a single vectorized comparison.
        at_threshold = (coverage_counts == (M - 1))
        
        candidate_indices = np.where(available)[0]
        
        # Vectorized gain computation for all candidates at once.
        # For each candidate column, sum the at_threshold values over
        # its nonzero rows. This is equivalent to: B.T @ at_threshold,
        # but we only compute it for available columns.
        #
        # Using sparse matrix-vector product: gains = B[:, candidates].T @ at_threshold
        # This leverages optimized BLAS routines inside scipy.sparse.
        
        if len(candidate_indices) == num_voxels - step:
            # All remaining columns — use full matrix product
            # B.T @ at_threshold gives gain for ALL voxels at once.
            # This is a single sparse matrix-vector multiply: O(nnz).
            all_gains = B.T.dot(at_threshold.astype(np.int32))
            all_gains[~available] = -1  # mask out already selected
            best_voxel = int(np.argmax(all_gains))
            best_gain = int(all_gains[best_voxel])
        else:
            # Subset of columns — slice and multiply
            B_sub = B[:, candidate_indices]
            sub_gains = B_sub.T.dot(at_threshold.astype(np.int32))
            best_sub_idx = int(np.argmax(sub_gains))
            best_gain = int(sub_gains[best_sub_idx])
            best_voxel = int(candidate_indices[best_sub_idx])
        
        # ---------------------------------------------------------------
        # Update coverage counts.
        # For all NCs seen by the selected voxel, increment c[i].
        # CSC format gives us direct access to the nonzero rows of
        # column best_voxel via B.indices[B.indptr[v]:B.indptr[v+1]].
        # ---------------------------------------------------------------
        col_start = B.indptr[best_voxel]
        col_end = B.indptr[best_voxel + 1]
        affected_ncs = B.indices[col_start:col_end]
        coverage_counts[affected_ncs] += 1
        
        # Mark voxel as selected
        available[best_voxel] = False
        selected.append(best_voxel)
        
        # Compute cumulative detection efficiency
        num_detected = int(np.sum(coverage_counts >= M))
        efficiency = num_detected / num_ncs
        efficiencies.append(efficiency)
        
        dt = time.time() - t0
        
        if verbose:
            print(f"{step+1:>4} | {best_voxel:>8} | {best_gain:>8} | "
                  f"{num_detected:>10} | {efficiency:>10.4%}  ({dt:.2f}s)")
    
    return selected, efficiencies, coverage_counts


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Greedy voxel selection for NC detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("hdf5_file", type=str, help="Path to the HDF5 data file.")
    parser.add_argument("-N", type=int, required=True,
                        help="Number of voxels to select.")
    parser.add_argument("-M", type=int, required=True,
                        help="Multiplicity threshold for detection "
                             "(number of voxels that must fire).")
    parser.add_argument("-m", type=int, default=1,
                        help="Hit threshold per voxel "
                             "(minimum hits for a voxel to count as firing).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (npz format). "
                             "If not specified, prints to stdout only.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output.")
    
    args = parser.parse_args(argv)
    verbose = not args.quiet
    
    # --- Step 1: Load and binarize ---
    if verbose:
        print("=" * 60)
        print("Greedy Voxel Selection for NC Detection")
        print("=" * 60)
    
    B, voxel_ids, num_primaries = load_and_binarize(
        args.hdf5_file, m=args.m, verbose=verbose,
    )
    
    # --- Step 2: Run greedy selection ---
    selected_cols, efficiencies, coverage_counts = greedy_select(
        B, N=args.N, M=args.M, verbose=verbose,
    )
    
    # --- Step 3: Report results ---
    selected_voxel_ids = voxel_ids[selected_cols]
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Final detection efficiency: {efficiencies[-1]:.4%}")
        print(f"Detected NCs: {int(np.sum(coverage_counts >= args.M)):,} "
              f"/ {B.shape[0]:,}")
        print(f"Total primaries: {num_primaries:,}")
        print(f"\nSelected voxels (in selection order):")
        for rank, (col, vid) in enumerate(zip(selected_cols, selected_voxel_ids)):
            eff_delta = efficiencies[rank] - (efficiencies[rank-1] if rank > 0 else 0.0)
            print(f"  {rank+1:>3}. Voxel {vid} (col {col}), "
                  f"cumulative eff = {efficiencies[rank]:.4%}, "
                  f"Δeff = {eff_delta:.4%}")
    
    # --- Step 4: Save results ---
    if args.output:
        np.savez(
            args.output,
            selected_columns=np.array(selected_cols),
            selected_voxel_ids=selected_voxel_ids,
            efficiencies=np.array(efficiencies),
            coverage_counts=coverage_counts,
            N=args.N,
            M=args.M,
            m=args.m,
            num_ncs=B.shape[0],
            num_primaries=num_primaries,
        )
        if verbose:
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()