#!/usr/bin/env python3
"""
Greedy Voxel Selection for Neutron Capture Detection
=====================================================

Solves the Maximum Coverage / Partial Set Multi-Cover Problem:
Given a set of neutron captures (NCs) and optical detector voxels,
select N voxels to maximize the number of NCs detected, where detection
requires at least M voxels exceeding a hit threshold m.

Two optimization modes:
  - "nc":         Maximize number of detected NCs (original mode).
  - "muon-ge77":  Maximize number of Ge77 muons identified via
                  coincidence detection of W neutron captures within
                  a time window [1 µs, 200 µs] relative to the muon.
                  Forces M=1 internally.

Theoretical Background
----------------------
For M=1 this is the classical Maximum k-Coverage Problem (NP-hard).
The greedy algorithm provides a (1 - 1/e) ≈ 0.632 approximation guarantee
[Nemhauser, Wolsey & Fisher, "An Analysis of Approximations for Maximizing
Submodular Set Functions", Math. Programming 14(1), 1978, Theorem 4.3].

For M>1 (nc mode) or W>1 (muon-ge77 mode), this is the Partial Set
Multi-Cover Problem. Submodularity of the objective breaks, but the greedy
remains the best polynomial-time approach
[Barman, Fawzi & Fermé, "Tight Approximation Guarantees for Concave
Coverage Problems", Math. Programming 201, 2023, Theorem 1.1].

Algorithm
---------
NC mode:
    Greedy iteration: select the voxel whose addition causes the largest
    increase in the number of detected NCs. A NC contributes to the
    marginal gain of voxel v iff c[i] == M-1 and B[i,v] == 1.

Muon-Ge77 mode:
    Greedy iteration: select the voxel whose addition causes the largest
    increase in the number of detected Ge77 muons. A muon is detected
    when >= W of its NCs (within [1µs, 200µs]) are individually detected
    (i.e. seen by >= 1 selected voxel, since M=1).
    Marginal gain uses SpMV (B^T @ w) as upper bound, followed by exact
    deduplication over muon IDs for top candidates.

Usage
-----
    # NC mode (original)
    python greedyVoxelSelection.py <hdf5_file> -N 50 -M 2 -m 1

    # Muon-Ge77 mode
    python greedy_voxel_selection.py <hdf5_file> -N 50 -W 4 --optimize muon-ge77

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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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

# Muon-Ge77 time window (ns)
MUON_TIME_WINDOW_MIN_NS = 1_000.0    # 1 µs
MUON_TIME_WINDOW_MAX_NS = 200_000.0  # 200 µs

# Area-dependent hit scaling ratios.
# When enabled, raw hits are divided by the layer ratio before
# binarization: hits / ratio >= m. This accounts for differing
# optical photon collection efficiencies across detector regions.
AREA_RATIOS: dict[str, float] = {
    "pit":  2.0731,
    "bot":  2.3843,
    "top":  2.2004,
    "wall": 1.8776,
}

# ---------------------------------------------------------------------------
# Detector surface areas (mm²), derived from geometry constants.
# Matches GeometryConfig in zoneRatioAnalysis.py.
# ---------------------------------------------------------------------------
Z_CUT_BOT = Z_BASE_GLOBAL       # = Z_ORIGIN + Z_OFFSET = -4980
Z_CUT_TOP = Z_CUT_BOT + H_ZYLINDER - 2  # 3917
WALL_HEIGHT = Z_CUT_TOP - Z_CUT_BOT      # 8897

AREA_SURFACES: dict[str, float] = {
    "pit":  np.pi * R_PIT**2,
    "bot":  np.pi * (R_ZYLINDER**2 - R_ZYL_BOT**2),
    "top":  np.pi * (R_ZYLINDER**2 - R_ZYL_TOP**2),
    "wall": 2 * np.pi * R_ZYLINDER * WALL_HEIGHT,
}


def compute_per_area_N(
    N: int,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Distribute N PMTs across areas proportional to surface area.

    Uses the largest remainder method to ensure sum(N_area) == N exactly.

    Parameters
    ----------
    N : int
        Total number of PMTs to distribute.
    verbose : bool
        Print allocation table.

    Returns
    -------
    allocation : dict[str, int]
        Number of PMTs per area.
    """
    total_area = sum(AREA_SURFACES.values())
    # Exact fractional allocation
    fractions = {area: N * a / total_area for area, a in AREA_SURFACES.items()}
    # Floor allocation
    floors = {area: int(np.floor(f)) for area, f in fractions.items()}
    # Remainders
    remainders = {area: fractions[area] - floors[area] for area in fractions}
    # Distribute leftover to areas with largest remainders
    leftover = N - sum(floors.values())
    sorted_areas = sorted(remainders, key=lambda a: remainders[a], reverse=True)
    allocation = dict(floors)
    for i in range(leftover):
        allocation[sorted_areas[i]] += 1

    if verbose:
        print(f"\nPer-area PMT allocation (N={N}):")
        print(f"  {'Area':<6} {'N_PMTs':>7} {'Fläche (M mm²)':>16} "
              f"{'Dichte':>14} {'Abw. von Ziel':>14}")
        print(f"  {'-' * 58}")
        target_density = N / total_area
        for area in ["pit", "bot", "top", "wall"]:
            n_a = allocation[area]
            a_mm2 = AREA_SURFACES[area]
            density = n_a / a_mm2 if a_mm2 > 0 else 0.0
            dev = (density - target_density) / target_density * 100
            print(f"  {area:<6} {n_a:>7} {a_mm2/1e6:>16.2f} "
                  f"{density:>14.6e} {dev:>+13.1f}%")

    return allocation


def muon_weight_delta(d: np.ndarray, k: float) -> np.ndarray:
    """
    Marginal gain of detecting the (d+1)-th NC of a muon under
    exponential saturation weighting.

    f(d) = 1 - exp(-d / k)
    Δf(d) = f(d+1) - f(d) = exp(-d/k) * (1 - exp(-1/k))

    Parameters
    ----------
    d : np.ndarray
        Current number of detected NCs per muon (integer array).
    k : float
        Saturation constant. k = d* / ln(10) gives f(d*) = 0.9.

    Returns
    -------
    np.ndarray, dtype float32
        Marginal weight Δf(d) for each element.
    """
    c = 1.0 - np.exp(-1.0 / k)  # constant factor
    return (np.exp(-d.astype(np.float32) / k) * c).astype(np.float32)


def muon_weight_k_for_90pct(d_star: int) -> float:
    """
    Compute k such that f(d_star) = 0.9.

    Parameters
    ----------
    d_star : int
        Number of detected NCs at which 90% saturation is reached.

    Returns
    -------
    float
        Saturation constant k.
    """
    return d_star / np.log(10)


def plot_muon_nc_histogram(
    muon_det_counts: np.ndarray,
    output_path: Path,
    title_extra: str = "",
) -> None:
    """
    Histogram of detected NCs per muon.

    Parameters
    ----------
    muon_det_counts : np.ndarray, shape (num_muons,)
        Number of detected NCs per muon.
    output_path : Path
        Where to save.
    title_extra : str
        Additional info for title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    max_count = int(muon_det_counts.max()) if len(muon_det_counts) > 0 else 0
    bins = np.arange(0, max_count + 2) - 0.5

    ax.hist(muon_det_counts, bins=bins, edgecolor="black", linewidth=0.5,
            color="#1976d2", alpha=0.85)
    ax.set_xlabel("Detected NCs per muon", fontsize=12)
    ax.set_ylabel("Number of muons", fontsize=12)
    ax.set_title(f"Muon NC Detection Distribution"
                 + (f"\n{title_extra}" if title_extra else ""),
                 fontsize=13)
    ax.set_xlim(-0.5, min(max_count + 1.5, 50.5))
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate summary stats
    n_total = len(muon_det_counts)
    n_zero = int(np.sum(muon_det_counts == 0))
    mean_d = float(np.mean(muon_det_counts)) if n_total > 0 else 0.0
    median_d = float(np.median(muon_det_counts)) if n_total > 0 else 0.0
    textstr = (f"Total muons: {n_total:,}\n"
               f"Undetected (d=0): {n_zero:,}\n"
               f"Mean d: {mean_d:.1f}\n"
               f"Median d: {median_d:.0f}")
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Muon NC histogram saved to {output_path}")


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
    apply_area_ratio: bool = False,
    verbose: bool = True,
) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load HDF5 data and construct sparse binary matrix B.

    B[i, j] = 1 iff voxel j registered >= m hits for NC i.
    Only voxels where a PMT can physically be placed are included.

    If apply_area_ratio is True, raw hits are divided by the layer-
    dependent ratio before comparison: hits / ratio >= m.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    m : int
        Minimum number of hits per voxel for a NC to count as "seen".
    apply_area_ratio : bool
        If True, scale hits by area-dependent ratios before binarization.
    verbose : bool
        Print progress information.

    Returns
    -------
    B : sparse.csc_matrix
        Binary (NCs x valid_voxels) matrix in CSC format.
    voxel_ids : np.ndarray
        Array of voxel ID strings, mapping column index -> voxel name.
    centers : np.ndarray, shape (num_valid_voxels, 3)
        Voxel center coordinates (x, y, z) in mm.
    layers : np.ndarray of str, shape (num_valid_voxels,)
        Layer label per valid voxel.
    num_primaries : int
        Total number of primary events (for efficiency calculation).
    """
    with h5py.File(filepath, "r") as f:
        voxel_keys = sorted(
            c.decode() if isinstance(c, bytes) else str(c)
            for c in f["target_columns"][:]
        )

        # --- PMT placement filter ---
        valid_mask = get_valid_voxel_mask(f, voxel_keys, verbose=verbose)
        valid_keys = [k for k, v in zip(voxel_keys, valid_mask) if v]
        num_voxels = len(valid_keys)

        # Read centers and layers for valid voxels
        centers = np.empty((num_voxels, 3), dtype=np.float64)
        layers = np.empty(num_voxels, dtype=object)
        for i, vkey in enumerate(valid_keys):
            centers[i] = f[f"voxels/{vkey}/center"][:]
            layer_raw = f[f"voxels/{vkey}/layer"][()]
            layers[i] = layer_raw.decode() if isinstance(layer_raw, bytes) else str(layer_raw)

        # NEU:
        # Read target column index and map valid voxel keys to matrix columns
        target_columns = [c.decode() if isinstance(c, bytes) else str(c)
                          for c in f["target_columns"][:]]
        target_col_to_idx = {c: i for i, c in enumerate(target_columns)}
        valid_col_indices_arr = np.array([target_col_to_idx[k] for k in valid_keys])

        num_ncs = f["target_matrix"].shape[0]
        num_primaries = int(f["primaries"][()])

        if verbose:
            print(f"Loading data: {num_ncs} NCs, {num_voxels} valid voxels, "
                  f"{num_primaries} primaries")
            print(f"Binarization threshold m = {m}")
            if apply_area_ratio:
                print(f"Area ratio scaling enabled: {AREA_RATIOS}")

        # Precompute per-column area ratios for vectorized binarization
        if apply_area_ratio:
            ratio_vec = np.array(
                [AREA_RATIOS.get(layers[c], 1.0) for c in range(num_voxels)],
                dtype=np.float32,
            )

        # Row-block reading — aligned to chunk layout (1000, 9583)
        BATCH_SIZE = 1000  # matches HDF5 chunk row dimension
        rows_list: list[np.ndarray] = []
        cols_list: list[np.ndarray] = []

        target_dset = f["target_matrix"]
        total_batches = (num_ncs - 1) // BATCH_SIZE + 1
        t_load_start = time.time()
        t_last_report = t_load_start

        for batch_idx, row_start in enumerate(range(0, num_ncs, BATCH_SIZE)):
            row_end = min(row_start + BATCH_SIZE, num_ncs)

            # Read full chunk row slice, then extract valid columns in RAM
            block = target_dset[row_start:row_end, :]     # (batch, 9583)
            block_valid = block[:, valid_col_indices_arr]  # (batch, num_valid)

            # Vectorized binarization over entire block
            if apply_area_ratio:
                mask = (block_valid / ratio_vec) >= m
            else:
                mask = block_valid >= m

            nc_idx, col_idx = np.nonzero(mask)
            if len(nc_idx) > 0:
                rows_list.append(nc_idx.astype(np.int64) + row_start)
                cols_list.append(col_idx.astype(np.int32))

            # Progress report every ~5 seconds
            t_now = time.time()
            if verbose and (t_now - t_last_report >= 5.0 or batch_idx == total_batches - 1):
                elapsed = t_now - t_load_start
                frac = (batch_idx + 1) / total_batches
                eta = (elapsed / frac - elapsed) if frac > 0.01 else 0.0
                rows_per_sec = row_end / elapsed if elapsed > 0 else 0.0
                print(f"  Batch {batch_idx+1}/{total_batches} "
                      f"({frac:.1%}) | "
                      f"{rows_per_sec:,.0f} rows/s | "
                      f"elapsed {elapsed:.1f}s | "
                      f"ETA {eta:.1f}s", end="\r")
                t_last_report = t_now

        t_load_elapsed = time.time() - t_load_start
        if verbose:
            print(f"\n  Target matrix loaded and binarized in {t_load_elapsed:.1f}s "
                  f"({num_ncs / t_load_elapsed:,.0f} rows/s)")

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

    return B, voxel_ids, centers, layers, num_primaries


def load_muon_data(
    filepath: str,
    num_ncs: int,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load muon-related fields from the phi group in the HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    num_ncs : int
        Expected number of NCs (for consistency check).
    verbose : bool
        Print progress information.

    Returns
    -------
    global_muon_id : np.ndarray, shape (num_ncs,), dtype int64
        Global unique muon ID per NC.
    nc_time_ns : np.ndarray, shape (num_ncs,), dtype float64
        NC time relative to muon entry [ns].
    nc_flag_ge77 : np.ndarray, shape (num_ncs,), dtype bool
        True if this NC is a Ge77 capture.
    """
    with h5py.File(filepath, "r") as f:
        phi_columns = [c.decode() if isinstance(c, bytes) else str(c)
                       for c in f["phi_columns"][:]]
        phi_col_idx = {name: i for i, name in enumerate(phi_columns)}

        phi_matrix = f["phi_matrix"]
        global_muon_id = phi_matrix[:, phi_col_idx["global_muon_id"]].astype(np.int64)
        nc_time_ns = phi_matrix[:, phi_col_idx["nC_time_in_ns"]].astype(np.float64)
        nc_flag_ge77 = phi_matrix[:, phi_col_idx["nC_flag_Ge77"]].astype(bool)

    if len(global_muon_id) != num_ncs:
        raise ValueError(
            f"Muon data length ({len(global_muon_id)}) != num_ncs ({num_ncs})"
        )

    if verbose:
        n_ge77_ncs = int(nc_flag_ge77.sum())
        n_unique_muons = len(np.unique(global_muon_id))
        ge77_muon_ids = np.unique(global_muon_id[nc_flag_ge77])
        print(f"\nMuon data loaded:")
        print(f"  Total NCs: {num_ncs:,}")
        print(f"  Unique muons: {n_unique_muons:,}")
        print(f"  Ge77 NCs: {n_ge77_ncs:,}")
        print(f"  Ge77 muons: {len(ge77_muon_ids):,}")

    return global_muon_id, nc_time_ns, nc_flag_ge77


def build_muon_index(
    global_muon_id: np.ndarray,
    nc_time_ns: np.ndarray,
    nc_flag_ge77: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build data structures for muon-level optimization.

    Identifies Ge77 muons and their time-filtered NCs. Only NCs within
    the coincidence window [1 µs, 200 µs] are considered for detection.
    Ge77 NCs themselves are invisible (0 hits everywhere) and excluded
    from the detection counting — they only serve to flag the muon.

    Parameters
    ----------
    global_muon_id : np.ndarray, shape (num_ncs,)
        Global muon ID per NC.
    nc_time_ns : np.ndarray, shape (num_ncs,)
        NC time relative to muon [ns].
    nc_flag_ge77 : np.ndarray, shape (num_ncs,)
        Ge77 flag per NC.
    verbose : bool
        Print statistics.

    Returns
    -------
    nc_to_muon_local : np.ndarray, shape (num_ncs,), dtype int32
        Maps each NC index to a local Ge77-muon index (0..num_ge77_muons-1).
        Set to -1 for NCs not belonging to a Ge77 muon or outside the
        time window or that are Ge77 NCs themselves (invisible).
    muon_nc_counts : np.ndarray, shape (num_ge77_muons,), dtype int32
        Number of eligible (non-Ge77, time-filtered) NCs per Ge77 muon.
    ge77_muon_global_ids : np.ndarray, shape (num_ge77_muons,), dtype int64
        Global muon IDs of Ge77 muons (for output mapping).
    eligible_nc_mask : np.ndarray, shape (num_ncs,), dtype bool
        True for NCs that participate in muon-level optimization
        (non-Ge77, within time window, belonging to a Ge77 muon).
    num_ge77_muons : int
        Number of Ge77 muons.
    """
    num_ncs = len(global_muon_id)

    # Step 1: Identify Ge77 muons (muons with at least one Ge77 NC)
    ge77_muon_global_ids = np.unique(global_muon_id[nc_flag_ge77])
    num_ge77_muons = len(ge77_muon_global_ids)

    # Step 2: Build global -> local muon ID mapping
    global_to_local = {int(gid): lid for lid, gid in enumerate(ge77_muon_global_ids)}

    # Step 3: Time window filter
    in_time_window = (
        (nc_time_ns >= MUON_TIME_WINDOW_MIN_NS)
        & (nc_time_ns <= MUON_TIME_WINDOW_MAX_NS)
    )

    # Step 4: Eligible NCs = non-Ge77, in time window, belonging to Ge77 muon
    belongs_to_ge77_muon = np.isin(global_muon_id, ge77_muon_global_ids)
    eligible_nc_mask = belongs_to_ge77_muon & in_time_window & (~nc_flag_ge77)

    # Step 5: Map eligible NCs to local muon index (vectorized)
    nc_to_muon_local = np.full(num_ncs, -1, dtype=np.int32)
    eligible_indices = np.where(eligible_nc_mask)[0]

    # Vectorized mapping via searchsorted (ge77_muon_global_ids is sorted
    # from np.unique)
    eligible_global_ids = global_muon_id[eligible_indices]
    local_ids = np.searchsorted(ge77_muon_global_ids, eligible_global_ids)
    nc_to_muon_local[eligible_indices] = local_ids.astype(np.int32)

    # Step 6: Count eligible NCs per muon
    muon_nc_counts = np.bincount(local_ids, minlength=num_ge77_muons).astype(np.int32)

    if verbose:
        n_eligible = int(eligible_nc_mask.sum())
        n_muons_with_any = int((muon_nc_counts >= 1).sum())
        print(f"\nMuon index built:")
        print(f"  Ge77 muons: {num_ge77_muons:,}")
        print(f"  Eligible NCs (non-Ge77, in time window): {n_eligible:,}")
        print(f"  Ge77 muons with ≥1 eligible NC: {n_muons_with_any:,}")
        print(f"  Time window: [{MUON_TIME_WINDOW_MIN_NS/1e3:.0f} µs, "
              f"{MUON_TIME_WINDOW_MAX_NS/1e3:.0f} µs]")

    return (nc_to_muon_local, muon_nc_counts, ge77_muon_global_ids,
            eligible_nc_mask, num_ge77_muons)


def greedy_select_nc(
    B: sparse.csc_matrix,
    N: int,
    M: int,
    centers: np.ndarray | None = None,
    layers: np.ndarray | None = None,
    min_spacing: float = 0.0,
    muon_weight_k: float | None = None,
    nc_to_muon_local: np.ndarray | None = None,
    num_muons: int = 0,
    verbose: bool = True,
) -> tuple[list[int], list[float], np.ndarray, np.ndarray | None]:
    """
    Greedy voxel selection maximizing NC-level threshold coverage.

    In each of N iterations, select the voxel v* that maximizes:
        gain(v) = |{i : c[i] == M-1  AND  B[i,v] == 1}|

    Parameters
    ----------
    B : sparse.csc_matrix
        Binary matrix (NCs x voxels) in CSC format.
    N : int
        Number of voxels to select.
    M : int
        Multiplicity threshold for detection.
    centers : np.ndarray or None
        Voxel centers for spacing constraint.
    layers : np.ndarray or None
        Layer labels for spacing constraint.
    min_spacing : float
        Minimum distance between selected voxels on same layer (mm).
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
    muon_det_counts : np.ndarray or None
        If muon_weight_k is set, number of detected NCs per muon.
        None otherwise.
    """
    num_ncs, num_voxels = B.shape

    if N > num_voxels:
        raise ValueError(f"N={N} exceeds number of voxels ({num_voxels})")

    enforce_spacing = (min_spacing > 0) and (centers is not None) and (layers is not None)
    min_spacing_sq = min_spacing ** 2

    coverage_counts = np.zeros(num_ncs, dtype=np.int16)
    available = np.ones(num_voxels, dtype=bool)
    selected: list[int] = []
    efficiencies: list[float] = []

    # Muon-level tracking for diminishing-returns weighting
    use_muon_weight = (muon_weight_k is not None
                       and nc_to_muon_local is not None
                       and num_muons > 0)
    if use_muon_weight:
        # nc_detected_flag[i] = True once NC i is seen by any selected voxel
        nc_detected_flag = np.zeros(num_ncs, dtype=bool)
        # Number of detected NCs per muon
        muon_det_counts = np.zeros(num_muons, dtype=np.int32)
    else:
        nc_detected_flag = None
        muon_det_counts = None

    # Dynamic M ramp: when M>1, each step checks whether any NC has
    # reached c_i == M-1. If yes, optimize for M (promote M-1 -> M).
    # If no, fall back to M=1 (maximize first-time coverage).
    # This switches dynamically per step — no fixed phase boundary.
    use_dynamic_M = (M > 1)

    if verbose:
        spacing_str = f", min_spacing={min_spacing:.0f}mm" if enforce_spacing else ""
        weight_str = f", muon_weight_k={muon_weight_k:.2f}" if use_muon_weight else ""
        dynamic_str = f" (dynamic M: fallback to M=1 when no NCs at M-1)" if use_dynamic_M else ""
        print(f"\nGreedy selection (NC mode): N={N}, M={M}{spacing_str}{weight_str}{dynamic_str}")
        print(f"{'Step':>4} | {'Voxel':>8} | {'Gain':>8} | "
              f"{'Detected':>10} | {'Efficiency':>10}")
        print("-" * 60)

    for step in range(N):
        t0 = time.time()

        # Dynamic M: prefer promoting M-1 -> M when possible,
        # otherwise fall back to building coverage with M=1.
        if use_dynamic_M:
            # Priority: promote M-1 -> M if possible.
            # Otherwise, find the highest occupied level below M
            # and promote that. This avoids zero-gain steps.
            if np.any(coverage_counts == (M - 1)):
                effective_M = M
            else:
                # Find highest c where any NC sits, up to M-2
                max_c = 0
                for c_level in range(M - 2, -1, -1):
                    if np.any(coverage_counts == c_level):
                        max_c = c_level + 1
                        break
                effective_M = max(max_c, 1)
        else:
            effective_M = M
        at_threshold = (coverage_counts == (effective_M - 1))

        if use_muon_weight:
            # Weighted SpMV: w[i] = Δf(d_μ(i)) for NCs at threshold
            muon_ids_per_nc = nc_to_muon_local  # shape (num_ncs,)
            d_per_nc = muon_det_counts[muon_ids_per_nc]  # current d_μ for each NC
            weights = muon_weight_delta(d_per_nc, muon_weight_k)
            weights[~at_threshold] = 0.0
            all_gains = B.T.dot(weights)
        else:
            # SpMV: g[v] = number of NCs at threshold that voxel v sees
            all_gains = B.T.dot(at_threshold.astype(np.int32))
        all_gains[~available] = -1
        best_voxel = int(np.argmax(all_gains))
        best_gain = int(all_gains[best_voxel])

        # Update coverage counts
        col_start = B.indptr[best_voxel]
        col_end = B.indptr[best_voxel + 1]
        affected_ncs = B.indices[col_start:col_end]
        coverage_counts[affected_ncs] += 1

        # Update muon detection counts (track which NCs are newly seen)
        if use_muon_weight:
            newly_seen = affected_ncs[~nc_detected_flag[affected_ncs]]
            nc_detected_flag[affected_ncs] = True
            if len(newly_seen) > 0:
                muon_lids = nc_to_muon_local[newly_seen]
                increments = np.bincount(muon_lids, minlength=num_muons)
                muon_det_counts += increments.astype(np.int32)

        available[best_voxel] = False
        selected.append(best_voxel)

        # Spacing constraint
        if enforce_spacing:
            selected_center = centers[best_voxel]
            selected_layer = layers[best_voxel]
            same_layer_mask = (layers == selected_layer) & available
            same_layer_indices = np.where(same_layer_mask)[0]
            if len(same_layer_indices) > 0:
                diff = centers[same_layer_indices] - selected_center
                dist_sq = np.sum(diff ** 2, axis=1)
                too_close = same_layer_indices[dist_sq < min_spacing_sq]
                available[too_close] = False
                if verbose and len(too_close) > 0:
                    print(f"       └─ spacing: excluded {len(too_close)} "
                          f"voxels on layer '{selected_layer}'")

        num_detected = int(np.sum(coverage_counts >= M))
        efficiency = num_detected / num_ncs
        efficiencies.append(efficiency)

        dt = time.time() - t0

        if verbose:
            phase_tag = f" [M=1]" if (use_dynamic_M and effective_M == 1) else ""
            gain_str = f"{best_gain:>8.3f}" if use_muon_weight else f"{best_gain:>8}"
            print(f"{step+1:>4} | {best_voxel:>8} | {gain_str} | "
                  f"{num_detected:>10} | {efficiency:>10.4%}  ({dt:.2f}s){phase_tag}")

    return selected, efficiencies, coverage_counts, muon_det_counts


def greedy_select_muon(
    B: sparse.csc_matrix,
    N: int,
    W: int,
    nc_to_muon_local: np.ndarray,
    eligible_nc_mask: np.ndarray,
    num_ge77_muons: int,
    centers: np.ndarray | None = None,
    layers: np.ndarray | None = None,
    min_spacing: float = 0.0,
    muon_weight_k: float | None = None,
    verbose: bool = True,
) -> tuple[list[int], list[float], np.ndarray, np.ndarray]:
    """
    Greedy voxel selection maximizing Ge77 muon detection.

    M=1 is enforced: a NC is detected iff any selected voxel sees it.
    A Ge77 muon is detected iff >= W of its eligible NCs are detected.

    Marginal gain computation:
        1. SpMV upper bound: w[i] = 1 if NC i is undetected, belongs
           to a Ge77 muon at d_mu == W-1, and is eligible. Then
           g[v] = B^T @ w gives an upper bound (may count same muon
           multiple times if v covers multiple of its NCs).
        2. Exact deduplication: for the top-K candidates (by SpMV score),
           compute exact gain by counting unique muons that cross W.

    Parameters
    ----------
    B : sparse.csc_matrix
        Binary matrix (NCs x voxels) in CSC format.
    N : int
        Number of voxels to select.
    W : int
        Minimum number of detected NCs per muon for muon detection.
    nc_to_muon_local : np.ndarray, shape (num_ncs,)
        Maps NC index -> local Ge77-muon index (-1 if not eligible).
    eligible_nc_mask : np.ndarray, shape (num_ncs,), dtype bool
        True for NCs participating in muon optimization.
    num_ge77_muons : int
        Number of Ge77 muons.
    centers : np.ndarray or None
        Voxel centers for spacing constraint.
    layers : np.ndarray or None
        Layer labels for spacing constraint.
    min_spacing : float
        Minimum distance between selected voxels on same layer (mm).
    verbose : bool
        Print per-iteration progress.

    Returns
    -------
    selected : list[int]
        Column indices of selected voxels (in selection order).
    efficiencies : list[float]
        Cumulative muon detection efficiency after each step.
    nc_detected : np.ndarray, dtype bool
        Final detection status per NC (True if seen by any selected voxel).
    muon_detected_counts : np.ndarray, shape (num_ge77_muons,)
        Final number of detected eligible NCs per Ge77 muon.
    """
    num_ncs, num_voxels = B.shape

    if N > num_voxels:
        raise ValueError(f"N={N} exceeds number of voxels ({num_voxels})")

    enforce_spacing = (min_spacing > 0) and (centers is not None) and (layers is not None)
    min_spacing_sq = min_spacing ** 2

    # Number of top SpMV candidates to evaluate exactly.
    TOP_K = min(50, num_voxels)

    # NC detection status (M=1: detected iff seen by any selected voxel)
    nc_detected = np.zeros(num_ncs, dtype=bool)

    # Per-muon count of detected eligible NCs
    muon_detected_counts = np.zeros(num_ge77_muons, dtype=np.int32)

    available = np.ones(num_voxels, dtype=bool)
    selected: list[int] = []
    efficiencies: list[float] = []

    use_muon_weight = (muon_weight_k is not None)

    # Dynamic W ramp: when W>1, each step checks whether any muon has
    # reached d_μ == W-1. If yes, optimize for W (promote W-1 -> W).
    # If no, fall back to W=1 (maximize first-time NC detections).
    use_dynamic_W = (W > 1) and not use_muon_weight

    if verbose:
        spacing_str = f", min_spacing={min_spacing:.0f}mm" if enforce_spacing else ""
        weight_str = f", muon_weight_k={muon_weight_k:.2f}" if use_muon_weight else ""
        dynamic_str = f" (dynamic W: fallback to W=1 when no muons at W-1)" if use_dynamic_W else ""
        print(f"\nGreedy selection (muon-ge77 mode): N={N}, W={W}{spacing_str}{weight_str}{dynamic_str}")
        print(f"{'Step':>4} | {'Voxel':>8} | {'Gain':>8} | "
              f"{'Muons det.':>10} | {'Efficiency':>10}")
        print("-" * 60)

    for step in range(N):
        t0 = time.time()

        # -----------------------------------------------------------------
        # Step A: SpMV upper bound for marginal gain.
        #
        # Weight vector w[i] = 1 iff:
        #   (a) NC i is not yet detected (nc_detected[i] == False)
        #   (b) NC i is eligible (belongs to Ge77 muon, in time window)
        #   (c) NC i's muon is at d_mu == W-1 (one short of threshold)
        #
        # Then g = B^T @ w gives per-voxel upper bound on muon gain.
        # It overcounts when a voxel covers multiple NCs of the same
        # muon at W-1, but this is corrected in Step B.
        # -----------------------------------------------------------------
        # Dynamic W: prefer promoting W-1 -> W when possible.
        # Otherwise, find the highest occupied level below W
        # and promote that. This avoids zero-gain steps when all
        # muons have d_μ >= 1 but none has reached W-1.
        if use_dynamic_W:
            if np.any(muon_detected_counts == (W - 1)):
                effective_W = W
            else:
                # Find highest d where any muon sits, up to W-2
                max_d = 0
                for d_level in range(W - 2, -1, -1):
                    if np.any(muon_detected_counts == d_level):
                        max_d = d_level + 1
                        break
                effective_W = max(max_d, 1)
        else:
            effective_W = W
        muon_at_threshold = (muon_detected_counts == (effective_W - 1))

        # Build weight vector (vectorized)
        eligible_undetected = eligible_nc_mask & (~nc_detected)
        eligible_idx = np.where(eligible_undetected)[0]

        if use_muon_weight:
            # Weighted: w[i] = Δf(d_μ(i)) for eligible undetected NCs
            w = np.zeros(num_ncs, dtype=np.float32)
            if len(eligible_idx) > 0:
                muon_lids = nc_to_muon_local[eligible_idx]
                d_vals = muon_detected_counts[muon_lids]
                w[eligible_idx] = muon_weight_delta(d_vals, muon_weight_k)
        else:
            w = np.zeros(num_ncs, dtype=np.int32)
            if len(eligible_idx) > 0:
                muon_lids = nc_to_muon_local[eligible_idx]
                at_thresh = muon_at_threshold[muon_lids]
                w[eligible_idx[at_thresh]] = 1

        all_gains_upper = B.T.dot(w)
        all_gains_upper[~available] = -1

        # -----------------------------------------------------------------
        # Step B: Exact deduplication for top-K candidates.
        #
        # For each top-K candidate voxel v, find the NCs it would newly
        # detect, map them to muon IDs, and count unique muons that
        # cross the W threshold.
        # -----------------------------------------------------------------
        top_k_indices = np.argsort(all_gains_upper)[-TOP_K:][::-1]
        # Filter to those with positive upper bound
        top_k_indices = top_k_indices[all_gains_upper[top_k_indices] > 0]

        best_gain = 0.0 if use_muon_weight else 0
        best_voxel = -1

        if len(top_k_indices) == 0:
            # No voxel can improve the objective — pick arbitrary available
            avail_idx = np.where(available)[0]
            if len(avail_idx) > 0:
                best_voxel = int(avail_idx[0])
            else:
                raise RuntimeError(
                    f"No available voxels left at step {step+1}/{N}. "
                    f"Spacing constraint may be too tight."
                )
            best_gain = 0
        else:
            for v in top_k_indices:
                # NCs that voxel v sees (from CSC structure)
                col_start = B.indptr[v]
                col_end = B.indptr[v + 1]
                v_ncs = B.indices[col_start:col_end]

                # Filter to eligible, undetected NCs
                mask = eligible_nc_mask[v_ncs] & (~nc_detected[v_ncs])
                newly_detected_ncs = v_ncs[mask]

                if len(newly_detected_ncs) == 0:
                    continue

                muon_lids_v = nc_to_muon_local[newly_detected_ncs]

                if use_muon_weight:
                    # Weighted gain: sum Δf per unique muon
                    # For each muon, Δf depends on current d_μ.
                    # Multiple NCs of same muon: each increments d sequentially.
                    # Approximate: use Δf(d_μ) for each unique muon (ignoring
                    # intra-step multi-NC increments — conservative estimate).
                    unique_muons, counts = np.unique(muon_lids_v, return_counts=True)
                    d_vals = muon_detected_counts[unique_muons]
                    # Sum marginal gains: Σ_j Δf(d + j) for j=0..count-1
                    gain = 0.0
                    for um_idx in range(len(unique_muons)):
                        d_cur = d_vals[um_idx]
                        n_new = counts[um_idx]
                        for j in range(n_new):
                            gain += float(
                                muon_weight_delta(
                                    np.array([d_cur + j]), muon_weight_k
                                )[0]
                            )
                else:
                    # Unweighted: count unique muons at threshold
                    # (uses effective_W via muon_at_threshold computed above)
                    at_thresh_mask = muon_at_threshold[muon_lids_v]
                    muon_lids_at_thresh = muon_lids_v[at_thresh_mask]
                    gain = len(np.unique(muon_lids_at_thresh))

                if gain > best_gain:
                    best_gain = gain
                    best_voxel = int(v)

            # Fallback if no candidate has positive exact gain
            if best_voxel == -1:
                best_voxel = int(top_k_indices[0])
                best_gain = 0

        # -----------------------------------------------------------------
        # Step C: Update state after selecting best_voxel.
        # -----------------------------------------------------------------
        col_start = B.indptr[best_voxel]
        col_end = B.indptr[best_voxel + 1]
        affected_ncs = B.indices[col_start:col_end]

        # Find NCs newly detected by this voxel
        newly_detected_mask = ~nc_detected[affected_ncs]
        newly_detected_ncs = affected_ncs[newly_detected_mask]
        nc_detected[affected_ncs] = True

        # Update muon detected counts for eligible newly-detected NCs
        # (vectorized: filter to eligible, then bincount)
        eligible_new = newly_detected_ncs[eligible_nc_mask[newly_detected_ncs]]
        if len(eligible_new) > 0:
            muon_lids_new = nc_to_muon_local[eligible_new]
            increments = np.bincount(muon_lids_new, minlength=num_ge77_muons)
            muon_detected_counts += increments.astype(np.int32)

        available[best_voxel] = False
        selected.append(best_voxel)

        # Spacing constraint
        if enforce_spacing:
            selected_center = centers[best_voxel]
            selected_layer = layers[best_voxel]
            same_layer_mask = (layers == selected_layer) & available
            same_layer_indices = np.where(same_layer_mask)[0]
            if len(same_layer_indices) > 0:
                diff = centers[same_layer_indices] - selected_center
                dist_sq = np.sum(diff ** 2, axis=1)
                too_close = same_layer_indices[dist_sq < min_spacing_sq]
                available[too_close] = False
                if verbose and len(too_close) > 0:
                    print(f"       └─ spacing: excluded {len(too_close)} "
                          f"voxels on layer '{selected_layer}'")

        num_muons_detected = int(np.sum(muon_detected_counts >= W))
        efficiency = num_muons_detected / num_ge77_muons if num_ge77_muons > 0 else 0.0
        efficiencies.append(efficiency)

        dt = time.time() - t0

        if verbose:
            phase_tag = f" [W=1]" if (use_dynamic_W and effective_W == 1) else ""
            gain_str = f"{best_gain:>8.3f}" if use_muon_weight else f"{best_gain:>8}"
            print(f"{step+1:>4} | {best_voxel:>8} | {gain_str} | "
                  f"{num_muons_detected:>10} | {efficiency:>10.4%}  ({dt:.2f}s){phase_tag}")

    return selected, efficiencies, nc_detected, muon_detected_counts

def plot_selected_voxels(
    selected_centers: np.ndarray,
    selected_layers: np.ndarray,
    selected_ids: list[str],
    output_path: Path,
    title_extra: str = "",
) -> None:
    """
    Generate 3D scatter plot of selected voxel positions overlaid
    on a wireframe of the detector geometry.

    Parameters
    ----------
    selected_centers : np.ndarray, shape (N, 3)
        Voxel center coordinates (x, y, z) in mm.
    selected_layers : np.ndarray of str, shape (N,)
        Layer label per voxel.
    selected_ids : list[str]
        Voxel ID strings.
    output_path : Path
        Where to save the plot.
    title_extra : str
        Additional info for the plot title.
    """
    Z_BASE = Z_BASE_GLOBAL  # from module constants
    Z_TOP = Z_BASE + H_ZYLINDER

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # --- Detector wireframe ---
    theta = np.linspace(0, 2 * np.pi, 200)
    n_vert = 24
    theta_lines = np.linspace(0, 2 * np.pi, n_vert, endpoint=False)

    # Cylinder
    for z in [Z_BASE, Z_TOP]:
        ax.plot(R_ZYLINDER * np.cos(theta), R_ZYLINDER * np.sin(theta), z,
                color="gray", alpha=0.3, linewidth=0.5)
    for t in theta_lines:
        ax.plot([R_ZYLINDER * np.cos(t)] * 2, [R_ZYLINDER * np.sin(t)] * 2,
                [Z_BASE, Z_TOP], color="gray", alpha=0.3, linewidth=0.5)

    # Pit and bot ring boundaries
    ax.plot(R_PIT * np.cos(theta), R_PIT * np.sin(theta), Z_BASE,
            color="blue", alpha=0.7, linewidth=1.2, label=f"Pit (r={R_PIT})")
    ax.plot(R_ZYL_BOT * np.cos(theta), R_ZYL_BOT * np.sin(theta), Z_BASE,
            color="green", alpha=0.7, linewidth=1.2,
            label=f"Bot ring inner (r={R_ZYL_BOT})")

    # --- Selected voxels ---
    layer_markers = {"pit": "o", "bot": "s", "top": "^", "wall": "D"}

    for layer in ["pit", "bot", "top", "wall"]:
        mask = selected_layers == layer
        if not np.any(mask):
            continue
        pts = selected_centers[mask]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c="red", marker=layer_markers.get(layer, "o"),
            s=30, alpha=0.8, edgecolors="darkred", linewidths=0.5,
            label=f"Selected ({layer}: {mask.sum()})",
        )

    # --- Check constraints and mark violations ---
    boundary_failures = []
    for i, (c, lay) in enumerate(zip(selected_centers, selected_layers)):
        if not is_valid_pmt_position(c, lay):
            boundary_failures.append(i)

    distance_violations = []
    min_dist = 2 * PMT_RADIUS
    n = len(selected_centers)
    for i in range(n):
        diffs = selected_centers[i + 1:] - selected_centers[i]
        dists = np.linalg.norm(diffs, axis=1)
        too_close = np.where(dists < min_dist)[0]
        for idx in too_close:
            j = i + 1 + idx
            distance_violations.append((i, j, dists[idx]))

    if boundary_failures:
        fail_pts = selected_centers[boundary_failures]
        ax.scatter(
            fail_pts[:, 0], fail_pts[:, 1], fail_pts[:, 2],
            c="yellow", marker="x", s=100, linewidths=2,
            label=f"Boundary violations ({len(boundary_failures)})",
        )

    if distance_violations:
        for i, j, d in distance_violations:
            p1, p2 = selected_centers[i], selected_centers[j]
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color="yellow", linewidth=2, alpha=0.8,
            )

    # --- Formatting ---
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(
        f"Selected Voxels (N={len(selected_centers)}, "
        f"boundary fails={len(boundary_failures)}, "
        f"dist violations={len(distance_violations)})"
        + (f"\n{title_extra}" if title_extra else "")
    )
    ax.legend(loc="upper left", fontsize=8)

    # Per-area count annotation
    area_counts = {area: int(np.sum(selected_layers == area))
                   for area in ["pit", "bot", "top", "wall"]}
    count_str = "\n".join(f"{a.upper():>4}: {cnt:>3} PMTs"
                          for a, cnt in area_counts.items())
    count_str += f"\n{'Total':>4}: {len(selected_centers):>3} PMTs"
    ax.text2D(0.98, 0.50, count_str, transform=ax.transAxes, fontsize=9,
              verticalalignment="center", horizontalalignment="right",
              fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                        edgecolor="gray", alpha=0.85))

    max_range = max(R_ZYLINDER, (Z_TOP - Z_BASE) / 2)
    mid_z = (Z_BASE + Z_TOP) / 2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Print summary
    print(f"\nPlot saved to {output_path}")
    print(f"  Boundary check: {'PASS' if not boundary_failures else 'FAIL'}")
    print(f"  Distance check: {'PASS' if not distance_violations else 'FAIL'}")
    for layer in ["pit", "bot", "top", "wall"]:
        count = np.sum(selected_layers == layer)
        if count > 0:
            pts = selected_centers[selected_layers == layer]
            r_vals = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            print(f"  {layer:>4}: {count:>4} voxels "
                f"(r: {r_vals.min():.0f}-{r_vals.max():.0f} mm, "
                f"z: {pts[:, 2].min():.0f}-{pts[:, 2].max():.0f} mm)")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Greedy voxel selection for NC / Ge77-muon detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("hdf5_file", type=str, help="Path to the HDF5 data file.")
    parser.add_argument("-N", type=int, required=True,
                        help="Number of voxels to select.")
    parser.add_argument("--optimize", type=str, default="nc",
                        choices=["nc", "muon-ge77"],
                        help="Optimization target.")
    parser.add_argument("-M", type=int, default=1,
                        help="Multiplicity threshold for NC detection "
                             "(number of voxels that must fire). "
                             "Ignored in muon-ge77 mode (forced to 1).")
    parser.add_argument("-m", type=int, default=1,
                        help="Hit threshold per voxel "
                             "(minimum hits for a voxel to count as firing).")
    parser.add_argument("-W", type=int, default=None,
                        help="Muon coincidence threshold: minimum number of "
                             "detected NCs per muon. Required for muon-ge77 mode.")
    parser.add_argument("--no-spacing", action="store_true",
                        help="Disable minimum spacing constraint between "
                             "selected voxels (default: enforce 2*PMT_RADIUS).")
    parser.add_argument("--area-ratio", action="store_true",
                        help="Apply area-dependent hit scaling before "
                             "binarization (hits / ratio >= m).")
    parser.add_argument("--muon-weight", type=float, default=None, metavar="K",
                        help="Enable muon-level diminishing-returns weighting "
                             "with saturation constant k. Uses f(d) = 1 - exp(-d/k). "
                             "For 90%% saturation at 10 detected NCs, use k ≈ 4.34. "
                             "Not compatible with --per-area.")
    parser.add_argument("--per-area", action="store_true",
                        help="Optimize each detector area (pit, bot, top, wall) "
                             "independently with N proportional to surface area.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (npz format). "
                             "If not specified, auto-generates from parameters.")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for all output files (npz, json, png).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output.")

    args = parser.parse_args(argv)
    verbose = not args.quiet
    min_spacing = 0.0 if args.no_spacing else 2 * PMT_RADIUS

    # --- Validate arguments ---
    if args.optimize == "muon-ge77":
        if args.W is None:
            parser.error("--optimize muon-ge77 requires -W argument.")
        if args.M != 1:
            if verbose:
                print(f"Warning: muon-ge77 mode forces M=1 (was M={args.M}).")
            args.M = 1

    # Auto-generate output filename
    # Validate muon-weight + per-area incompatibility
    if args.muon_weight is not None and args.per_area:
        parser.error("--muon-weight and --per-area cannot be combined. "
                     "This combination is not implemented.")

    if args.muon_weight is not None and args.muon_weight <= 0:
        parser.error("--muon-weight must be a positive float.")

    # Auto-generate output filename
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        spacing_tag = "nospacing" if args.no_spacing else f"spacing{int(min_spacing)}mm"
        ratio_tag = "_arearatio" if args.area_ratio else ""
        perarea_tag = "_perarea" if args.per_area else ""
        mw_tag = f"_mw{args.muon_weight:.2f}" if args.muon_weight is not None else ""
        if args.optimize == "nc":
            args.output = f"greedy_N{args.N}_M{args.M}_m{args.m}_{spacing_tag}{ratio_tag}{perarea_tag}{mw_tag}.npz"
        else:
            args.output = (f"greedy_muon_N{args.N}_W{args.W}_m{args.m}_"
                           f"{spacing_tag}{ratio_tag}{perarea_tag}{mw_tag}.npz")

    args.output = str(output_dir / Path(args.output).name)

    # --- Step 1: Load and binarize ---
    if verbose:
        print("=" * 60)
        print(f"Greedy Voxel Selection — mode: {args.optimize}")
        print("=" * 60)

    B, voxel_ids, centers, layers, num_primaries = load_and_binarize(
        args.hdf5_file, m=args.m, apply_area_ratio=args.area_ratio,
        verbose=verbose,
    )

    # --- Step 2: Run greedy ---
    # Optionally load muon data (needed for muon-ge77 mode)
    muon_data = None
    # Load muon data for muon-ge77 mode OR for NC mode with muon weighting
    nc_muon_weight_data = None
    if args.optimize == "nc" and args.muon_weight is not None:
        # Load muon IDs for NC-mode muon-awareness (no time filter)
        with h5py.File(args.hdf5_file, "r") as f:
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

        # Validate: every NC must belong to a muon
        unique_muon_ids = np.unique(nc_global_muon_id)
        num_muons_nc = len(unique_muon_ids)
        # Build local muon index (no time filter, all NCs)
        global_to_local_nc = {int(gid): lid for lid, gid in enumerate(unique_muon_ids)}
        nc_to_muon_local_nc = np.array(
            [global_to_local_nc[int(gid)] for gid in nc_global_muon_id],
            dtype=np.int32,
        )

        if verbose:
            print(f"\nMuon weighting (NC mode):")
            print(f"  Unique muons: {num_muons_nc:,}")
            print(f"  k = {args.muon_weight:.4f}")
            print(f"  f(10) = {1 - np.exp(-10/args.muon_weight):.4f}")

        nc_muon_weight_data = {
            "nc_to_muon_local": nc_to_muon_local_nc,
            "num_muons": num_muons_nc,
            "unique_muon_ids": unique_muon_ids,
        }

    if args.optimize == "muon-ge77":
        global_muon_id, nc_time_ns, nc_flag_ge77 = load_muon_data(
            args.hdf5_file, num_ncs=B.shape[0], verbose=verbose,
        )
        (nc_to_muon_local, muon_nc_counts, ge77_muon_global_ids,
         eligible_nc_mask, num_ge77_muons) = build_muon_index(
            global_muon_id, nc_time_ns, nc_flag_ge77, verbose=verbose,
        )
        if verbose:
            max_ncs_per_muon = int(muon_nc_counts.max()) if num_ge77_muons > 0 else 0
            print(f"  Max eligible NCs per Ge77 muon: {max_ncs_per_muon}")
            n_feasible = int((muon_nc_counts >= args.W).sum())
            print(f"  Ge77 muons with ≥ W={args.W} eligible NCs: "
                  f"{n_feasible:,} / {num_ge77_muons:,}")
        muon_data = {
            "nc_to_muon_local": nc_to_muon_local,
            "eligible_nc_mask": eligible_nc_mask,
            "num_ge77_muons": num_ge77_muons,
            "ge77_muon_global_ids": ge77_muon_global_ids,
            "muon_detected_counts": None,
            "nc_detected": None,
        }

    if args.per_area:
        # --- Per-area optimization ---
        allocation = compute_per_area_N(args.N, verbose=verbose)
        all_selected_cols: list[int] = []

        # For muon mode: shared state across areas
        if args.optimize == "muon-ge77":
            shared_nc_detected = np.zeros(B.shape[0], dtype=bool)
            shared_muon_detected_counts = np.zeros(num_ge77_muons, dtype=np.int32)

        for area_name in ["pit", "bot", "top", "wall"]:
            n_area = allocation[area_name]
            if n_area == 0:
                if verbose:
                    print(f"\nSkipping {area_name}: 0 PMTs allocated.")
                continue

            # Build mask for voxels in this area
            area_mask = (layers == area_name)
            area_indices = np.where(area_mask)[0]
            n_voxels_area = len(area_indices)

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Per-area optimization: {area_name} "
                      f"(N={n_area}, {n_voxels_area} voxels)")
                print(f"{'=' * 60}")

            if n_voxels_area == 0:
                if verbose:
                    print(f"  No valid voxels in {area_name}, skipping.")
                continue

            # Extract sub-matrix for this area
            B_area = B[:, area_indices]
            centers_area = centers[area_indices]
            layers_area = layers[area_indices]

            if args.optimize == "nc":
                sel_local, eff_area, cov_area, _ = greedy_select_nc(
                    B_area, N=n_area, M=args.M,
                    centers=centers_area, layers=layers_area,
                    min_spacing=min_spacing, verbose=verbose,
                )
                # Map local column indices back to global
                sel_global = [int(area_indices[i]) for i in sel_local]
                all_selected_cols.extend(sel_global)

            elif args.optimize == "muon-ge77":
                # Run greedy on sub-matrix but with shared muon state.
                # We pass the full nc_to_muon_local etc. — the sub-matrix
                # B_area still has all NC rows, just fewer voxel columns.
                sel_local, eff_area, nc_det_area, muon_det_area = (
                    greedy_select_muon(
                        B_area, N=n_area, W=args.W,
                        nc_to_muon_local=nc_to_muon_local,
                        eligible_nc_mask=eligible_nc_mask,
                        num_ge77_muons=num_ge77_muons,
                        centers=centers_area, layers=layers_area,
                        min_spacing=min_spacing,
                        muon_weight_k=args.muon_weight,
                        verbose=verbose,
                    )
                )
                sel_global = [int(area_indices[i]) for i in sel_local]
                all_selected_cols.extend(sel_global)
                # Merge detection state: OR for nc_detected, MAX for muon counts
                shared_nc_detected |= nc_det_area
                # Muon counts: accumulate (each area contributes independently)
                shared_muon_detected_counts += muon_det_area

        selected_cols = all_selected_cols
        selected_voxel_ids = voxel_ids[selected_cols]

        # Final combined reporting
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Per-area optimization complete (N={args.N})")
            print(f"{'=' * 60}")

        if args.optimize == "nc":
            # Recompute coverage with all selected voxels
            coverage_counts = np.zeros(B.shape[0], dtype=np.int16)
            for col in selected_cols:
                col_start = B.indptr[col]
                col_end = B.indptr[col + 1]
                coverage_counts[B.indices[col_start:col_end]] += 1
            num_detected = int(np.sum(coverage_counts >= args.M))
            final_eff = num_detected / B.shape[0]
            efficiencies = [final_eff]  # only final combined efficiency

            if verbose:
                print(f"Final NC detection efficiency: {final_eff:.4%}")
                print(f"Detected NCs: {num_detected:,} / {B.shape[0]:,}")

            np.savez(
                args.output,
                selected_columns=np.array(selected_cols),
                selected_voxel_ids=selected_voxel_ids,
                efficiencies=np.array(efficiencies),
                coverage_counts=coverage_counts,
                N=args.N, M=args.M, m=args.m,
                optimize="nc", per_area=True,
                allocation=json.dumps(allocation),
                min_spacing=min_spacing,
                num_ncs=B.shape[0],
                num_primaries=num_primaries,
            )

        elif args.optimize == "muon-ge77":
            num_muons_detected = int(np.sum(shared_muon_detected_counts >= args.W))
            final_eff = num_muons_detected / num_ge77_muons if num_ge77_muons > 0 else 0.0
            efficiencies = [final_eff]

            if verbose:
                print(f"Final Ge77 muon detection efficiency: {final_eff:.4%}")
                print(f"Detected Ge77 muons: {num_muons_detected:,} "
                      f"/ {num_ge77_muons:,}")
                num_ncs_detected = int(shared_nc_detected.sum())
                print(f"NCs detected (any): {num_ncs_detected:,} / {B.shape[0]:,} "
                      f"({num_ncs_detected/B.shape[0]:.4%})")

            np.savez(
                args.output,
                selected_columns=np.array(selected_cols),
                selected_voxel_ids=selected_voxel_ids,
                efficiencies=np.array(efficiencies),
                muon_detected_counts=shared_muon_detected_counts,
                nc_detected=shared_nc_detected,
                ge77_muon_global_ids=ge77_muon_global_ids,
                N=args.N, W=args.W, m=args.m, M=1,
                optimize="muon-ge77", per_area=True,
                allocation=json.dumps(allocation),
                min_spacing=min_spacing,
                num_ncs=B.shape[0],
                num_ge77_muons=num_ge77_muons,
                num_primaries=num_primaries,
            )

    else:
        # --- Global optimization (original behavior) ---
        if args.optimize == "nc":
            mw_kwargs = {}
            if nc_muon_weight_data is not None:
                mw_kwargs = {
                    "muon_weight_k": args.muon_weight,
                    "nc_to_muon_local": nc_muon_weight_data["nc_to_muon_local"],
                    "num_muons": nc_muon_weight_data["num_muons"],
                }
            selected_cols, efficiencies, coverage_counts, muon_det_counts_nc = greedy_select_nc(
                B, N=args.N, M=args.M,
                centers=centers, layers=layers, min_spacing=min_spacing,
                verbose=verbose, **mw_kwargs,
            )
            selected_voxel_ids = voxel_ids[selected_cols]
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Final NC detection efficiency: {efficiencies[-1]:.4%}")
                print(f"Detected NCs: {int(np.sum(coverage_counts >= args.M)):,} "
                      f"/ {B.shape[0]:,}")
                print(f"Total primaries: {num_primaries:,}")

            np.savez(
                args.output,
                selected_columns=np.array(selected_cols),
                selected_voxel_ids=selected_voxel_ids,
                efficiencies=np.array(efficiencies),
                coverage_counts=coverage_counts,
                N=args.N, M=args.M, m=args.m,
                optimize="nc",
                min_spacing=min_spacing,
                num_ncs=B.shape[0],
                num_primaries=num_primaries,
            )

        elif args.optimize == "muon-ge77":
            selected_cols, efficiencies, nc_detected, muon_detected_counts = (
                greedy_select_muon(
                    B, N=args.N, W=args.W,
                    nc_to_muon_local=nc_to_muon_local,
                    eligible_nc_mask=eligible_nc_mask,
                    num_ge77_muons=num_ge77_muons,
                    centers=centers, layers=layers, min_spacing=min_spacing,
                    muon_weight_k=args.muon_weight,
                    verbose=verbose,
                )
            )

            selected_voxel_ids = voxel_ids[selected_cols]
            num_muons_detected = int(np.sum(muon_detected_counts >= args.W))
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Final Ge77 muon detection efficiency: {efficiencies[-1]:.4%}")
                print(f"Detected Ge77 muons: {num_muons_detected:,} "
                      f"/ {num_ge77_muons:,}")
                print(f"Total primaries: {num_primaries:,}")
                num_ncs_detected = int(nc_detected.sum())
                print(f"NCs detected (any): {num_ncs_detected:,} / {B.shape[0]:,} "
                      f"({num_ncs_detected/B.shape[0]:.4%})")

            np.savez(
                args.output,
                selected_columns=np.array(selected_cols),
                selected_voxel_ids=selected_voxel_ids,
                efficiencies=np.array(efficiencies),
                muon_detected_counts=muon_detected_counts,
                nc_detected=nc_detected,
                ge77_muon_global_ids=ge77_muon_global_ids,
                N=args.N, W=args.W, m=args.m, M=1,
                optimize="muon-ge77",
                min_spacing=min_spacing,
                num_ncs=B.shape[0],
                num_ge77_muons=num_ge77_muons,
                num_primaries=num_primaries,
            )

    if verbose:
        print(f"\nResults saved to {args.output}")
        print(f"\nSelected voxels (in selection order):")
        for rank, (col, vid) in enumerate(zip(selected_cols, selected_voxel_ids)):
            if len(efficiencies) > rank:
                eff_val = efficiencies[rank]
                eff_delta = eff_val - (efficiencies[rank-1] if rank > 0 else 0.0)
                print(f"  {rank+1:>3}. Voxel {vid} (col {col}), "
                      f"cumulative eff = {eff_val:.4%}, "
                      f"Δeff = {eff_delta:.4%}")
            else:
                print(f"  {rank+1:>3}. Voxel {vid} (col {col})")

    # --- Muon NC detection histogram ---
    if args.muon_weight is not None:
        hist_path = Path(args.output).parent / (
            Path(args.output).stem + "_muon_hist.png"
        )
        title_info = f"mode={args.optimize}, k={args.muon_weight:.2f}"
        if args.optimize == "nc" and nc_muon_weight_data is not None:
            plot_muon_nc_histogram(
                muon_det_counts_nc,
                hist_path,
                title_extra=title_info,
            )
        elif args.optimize == "muon-ge77":
            plot_muon_nc_histogram(
                muon_detected_counts,
                hist_path,
                title_extra=title_info + f", W={args.W}",
            )

    # --- Export selected voxels as JSON ---
    json_output = Path(args.output).with_suffix(".json")
    selected_voxels_json = []

    with h5py.File(args.hdf5_file, "r") as f:
        for col_idx in selected_cols:
            vid = voxel_ids[col_idx]
            center = f[f"voxels/{vid}/center"][:].tolist()
            corners_x = f[f"voxels/{vid}/corners/x"][:].tolist()
            corners_y = f[f"voxels/{vid}/corners/y"][:].tolist()
            corners_z = f[f"voxels/{vid}/corners/z"][:].tolist()
            corners = [[x, y, z] for x, y, z in zip(corners_x, corners_y, corners_z)]
            layer_raw = f[f"voxels/{vid}/layer"][()]
            layer = layer_raw.decode() if isinstance(layer_raw, bytes) else str(layer_raw)

            selected_voxels_json.append({
                "index": vid,
                "center": center,
                "corners": corners,
                "layer": layer,
            })

    with open(json_output, "w") as jf:
        json.dump(selected_voxels_json, jf, indent=2)

    if verbose:
        print(f"Selected voxels JSON saved to {json_output}")

    # --- Plot selected voxels ---
    plot_centers = centers[selected_cols]
    plot_layers = layers[selected_cols]
    plot_ids = [str(voxel_ids[c]) for c in selected_cols]
    plot_path = Path(args.output).with_suffix(".png")

    title_extra = f"mode={args.optimize}"
    if args.optimize == "nc":
        title_extra += f", M={args.M}, m={args.m}"
    else:
        title_extra += f", W={args.W}, m={args.m}"
    if args.per_area:
        title_extra += ", per-area"
    if args.area_ratio:
        title_extra += ", area-ratio"

    plot_selected_voxels(
        plot_centers, plot_layers, plot_ids,
        output_path=plot_path,
        title_extra=title_extra,
    )

if __name__ == "__main__":
    main()