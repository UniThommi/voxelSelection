#!/usr/bin/env python3
"""
Evaluate Detection Efficiency for a Fixed Voxel Set
=====================================================

Given a JSON file with selected voxels (same format as the project-wide
voxel JSON) and an HDF5 data file, compute the NC detection efficiency
for multiplicity threshold M and hit threshold m.

NCs that produce zero hits across ALL voxels in the HDF5 file are
classified as "dark" (never detectable) and excluded from the efficiency
calculation and coverage distribution.

Usage
-----
    python evaluate_efficiency.py <hdf5_file> <voxel_json> -M 2 -m 1

Author: Ferundo (Thesis project, University of Tübingen)
"""

import argparse
import json
import sys
from typing import Optional

import h5py
import numpy as np


def evaluate_efficiency(
    hdf5_path: str,
    json_path: str,
    M: int,
    m: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Compute detection efficiency for a fixed set of voxels.

    For each NC, count how many of the selected voxels register >= m hits.
    A NC is detected if that count >= M.

    NCs with zero total hits across ALL voxels in the HDF5 are excluded
    as "dark" (never detectable regardless of PMT placement).

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 data file.
    json_path : str
        Path to JSON file with selected voxels (list of objects with "index" key).
    M : int
        Multiplicity threshold (number of voxels that must fire).
    m : int
        Hit threshold per voxel.
    verbose : bool
        Print progress and results.

    Returns
    -------
    results : dict with keys:
        - num_detected: int
        - num_ncs_total: int (including dark NCs)
        - num_ncs_dark: int (NCs with zero hits on any voxel)
        - num_ncs: int (detectable NCs, excluding dark)
        - num_primaries: int
        - efficiency: float (relative to detectable NCs)
        - coverage_counts: np.ndarray (per-NC coverage count, all NCs)
        - dark_mask: np.ndarray (bool, True for dark NCs)
        - voxel_indices: list[str] (voxel index strings from JSON)
    """
    # Load voxel indices from JSON
    with open(json_path, "r") as f:
        voxel_data = json.load(f)

    voxel_indices = [v["index"] for v in voxel_data]
    num_voxels = len(voxel_indices)

    if verbose:
        print(f"Loaded {num_voxels} voxels from {json_path}")

    with h5py.File(hdf5_path, "r") as f:
        # Verify all voxel indices exist in the HDF5 target group
        available_keys = set(f["target"].keys())
        missing = [vi for vi in voxel_indices if vi not in available_keys]
        if missing:
            print(f"ERROR: {len(missing)} voxel indices not found in HDF5: "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
                  file=sys.stderr)
            sys.exit(1)

        all_voxel_keys = list(available_keys)
        num_ncs_total = f["target"][voxel_indices[0]].shape[0]
        num_primaries = int(f["primaries"][()])

        if verbose:
            print(f"HDF5: {num_ncs_total} NCs, {num_primaries} primaries")
            print(f"Parameters: M={M}, m={m}")

        # --- Identify dark NCs (zero hits across ALL voxels) ---
        if verbose:
            print(f"\nIdentifying dark NCs (scanning all {len(all_voxel_keys)} voxels)...")

        has_any_hit = np.zeros(num_ncs_total, dtype=bool)
        for i, key in enumerate(all_voxel_keys):
            if verbose and i % 500 == 0:
                print(f"  Scanning voxel {i}/{len(all_voxel_keys)}...", end="\r")
            hits = f["target"][key][:]
            has_any_hit |= (hits > 0)
        if verbose:
            print()

        dark_mask = ~has_any_hit
        num_ncs_dark = int(dark_mask.sum())
        num_ncs = num_ncs_total - num_ncs_dark

        if verbose:
            print(f"  Total NCs:      {num_ncs_total:,}")
            print(f"  Dark NCs:       {num_ncs_dark:,} ({num_ncs_dark / num_ncs_total:.2%})")
            print(f"  Detectable NCs: {num_ncs:,} ({num_ncs / num_ncs_total:.2%})")

        # --- Count coverage for selected voxels ---
        if verbose:
            print(f"\nComputing coverage for {num_voxels} selected voxels...")

        coverage_counts = np.zeros(num_ncs_total, dtype=np.int16)

        for i, vi in enumerate(voxel_indices):
            if verbose and i % 50 == 0:
                print(f"  Processing voxel {i}/{num_voxels}...", end="\r")
            hits = f["target"][vi][:]
            coverage_counts += (hits >= m).astype(np.int16)

        if verbose:
            print()

    # --- Results (excluding dark NCs) ---
    coverage_detectable = coverage_counts[~dark_mask]
    num_detected = int(np.sum(coverage_detectable >= M))
    efficiency = num_detected / num_ncs if num_ncs > 0 else 0.0

    if verbose:
        print(f"\nResults (dark NCs excluded):")
        print(f"  Detected NCs:  {num_detected:,} / {num_ncs:,}")
        print(f"  Efficiency:    {efficiency:.4%}")
        print(f"  Primaries:     {num_primaries:,}")

        # Coverage distribution (excluding dark NCs)
        print(f"\n  Coverage distribution (detectable NCs only):")
        max_cov = int(coverage_detectable.max()) if num_ncs > 0 else 0
        for c in range(min(max_cov + 1, M + 5)):
            count = int(np.sum(coverage_detectable == c))
            pct = count / num_ncs * 100 if num_ncs > 0 else 0.0
            marker = " <-- threshold M" if c == M else ""
            print(f"    c={c:>3}: {count:>8} ({pct:>6.2f}%){marker}")
        if max_cov >= M + 5:
            above = int(np.sum(coverage_detectable >= M + 5))
            print(f"    c>={M+5:>2}: {above:>8} ({above/num_ncs*100:>6.2f}%)")

    return {
        "num_detected": num_detected,
        "num_ncs_total": num_ncs_total,
        "num_ncs_dark": num_ncs_dark,
        "num_ncs": num_ncs,
        "num_primaries": num_primaries,
        "efficiency": efficiency,
        "coverage_counts": coverage_counts,
        "dark_mask": dark_mask,
        "voxel_indices": voxel_indices,
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate detection efficiency for a fixed voxel set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("hdf5_file", type=str,
                        help="Path to the HDF5 data file.")
    parser.add_argument("voxel_json", type=str,
                        help="Path to JSON file with selected voxels.")
    parser.add_argument("-M", type=int, required=True,
                        help="Multiplicity threshold for detection.")
    parser.add_argument("-m", type=int, default=1,
                        help="Hit threshold per voxel.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output.")

    args = parser.parse_args(argv)

    evaluate_efficiency(
        args.hdf5_file,
        args.voxel_json,
        M=args.M,
        m=args.m,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()