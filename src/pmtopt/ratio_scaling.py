"""
SSD-to-PMT hit scaling with stochastic rounding.

Provides in-memory scaling (for greedy pipeline) and HDF5 file writing
(for creating ratio-adjusted data files).
"""

import shutil
import time
from pathlib import Path

import h5py
import numpy as np


def build_layer_map(
    f: h5py.File,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Read layer labels for all voxels and build column-index masks
    per area.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle.

    Returns
    -------
    layer_labels : np.ndarray of str, shape (num_voxels,)
        Layer label per voxel column.
    area_col_masks : dict[str, np.ndarray of bool]
        Boolean mask per area over voxel columns.
    """
    target_columns = [
        c.decode() if isinstance(c, bytes) else str(c)
        for c in f["target_columns"][:]
    ]
    num_voxels = len(target_columns)
    layer_labels = np.empty(num_voxels, dtype=object)

    for col_idx, vkey in enumerate(target_columns):
        layer_raw = f[f"voxels/{vkey}/layer"][()]
        layer_labels[col_idx] = (
            layer_raw.decode() if isinstance(layer_raw, bytes)
            else str(layer_raw)
        )

    area_col_masks = {}
    for area in ["pit", "bot", "top", "wall"]:
        area_col_masks[area] = (layer_labels == area)

    return layer_labels, area_col_masks


def build_ratio_vec(
    layer_labels: np.ndarray,
    ratios: dict[str, float],
) -> np.ndarray:
    """
    Build per-column ratio vector from layer labels and area ratios.

    Parameters
    ----------
    layer_labels : np.ndarray of str
        Layer label per voxel column.
    ratios : dict[str, float]
        SSD/PMT ratio per area.

    Returns
    -------
    ratio_vec : np.ndarray, dtype float64, shape (num_voxels,)
    """
    ratio_vec = np.ones(len(layer_labels), dtype=np.float64)
    for area, ratio in ratios.items():
        mask = (layer_labels == area)
        ratio_vec[mask] = ratio
    return ratio_vec


def stochastic_round_block(
    block: np.ndarray,
    ratio_vec: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Scale a block of integer hits by area ratios and apply stochastic
    rounding.

    For each element: floor(hits/ratio) + Bernoulli(fractional part).

    Parameters
    ----------
    block : np.ndarray, shape (batch_len, num_voxels), dtype int32
        Raw hit counts from SSD simulation.
    ratio_vec : np.ndarray, shape (num_voxels,), dtype float64
        Per-column ratio (SSD/PMT).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    rounded : np.ndarray, shape (batch_len, num_voxels), dtype int32
        Stochastically rounded hit counts.
    """
    scaled = block.astype(np.float64) / ratio_vec
    floor_vals = np.floor(scaled).astype(np.int32)
    fractional = scaled - floor_vals
    coin = rng.random(fractional.shape) < fractional
    return floor_vals + coin.astype(np.int32)


def write_ratio_hdf5(
    input_path: str,
    output_path: str,
    ratios: dict[str, float],
    seed: int = 42,
    batch_size: int = 1000,
    verbose: bool = True,
) -> None:
    """
    Copy HDF5 file, scale target_matrix hits by area ratios with
    stochastic rounding, and recompute region_matrix.

    Parameters
    ----------
    input_path : str
        Path to original HDF5 file.
    output_path : str
        Path for the output HDF5 file.
    ratios : dict[str, float]
        SSD/PMT ratios per area.
    seed : int
        Random seed for stochastic rounding.
    batch_size : int
        Number of NC rows per processing batch.
    verbose : bool
        Print progress.
    """
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"Copying {input_path} -> {output_path} ...")
    t0 = time.time()
    shutil.copy2(input_path, output_path)
    if verbose:
        print(f"  Copy done in {time.time() - t0:.1f}s")

    with h5py.File(output_path, "r+") as f:
        layer_labels, area_col_masks = build_layer_map(f)
        ratio_vec = build_ratio_vec(layer_labels, ratios)

        # Build region column map
        region_columns = [
            c.decode() if isinstance(c, bytes) else str(c)
            for c in f["region_columns"][:]
        ]
        region_col_map = {name: i for i, name in enumerate(region_columns)}

        num_voxels = len(layer_labels)
        target_dset = f["target_matrix"]
        region_dset = f["region_matrix"]
        num_ncs = target_dset.shape[0]
        num_regions = region_dset.shape[1]

        if verbose:
            for area, ratio in ratios.items():
                n_cols = int(area_col_masks[area].sum())
                print(f"  {area}: {n_cols} voxels, ratio = {ratio:.4f}")
            print(f"  target_matrix: {num_ncs:,} x {num_voxels}")
            print(f"  Seed: {seed}")

        # Read original region sums for cross-check
        original_region_sums = np.zeros(num_regions, dtype=np.float64)
        for row_start in range(0, num_ncs, batch_size):
            row_end = min(row_start + batch_size, num_ncs)
            block = region_dset[row_start:row_end, :]
            original_region_sums += block.astype(np.float64).sum(axis=0)

        if verbose:
            print("\nProcessing target_matrix...")

        total_batches = (num_ncs - 1) // batch_size + 1
        t_proc_start = time.time()
        t_last_report = t_proc_start
        new_region_sums = np.zeros(num_regions, dtype=np.int64)

        for batch_idx, row_start in enumerate(range(0, num_ncs, batch_size)):
            row_end = min(row_start + batch_size, num_ncs)
            batch_len = row_end - row_start

            block = target_dset[row_start:row_end, :]
            rounded = stochastic_round_block(block, ratio_vec, rng)

            target_dset[row_start:row_end, :] = rounded

            # Recompute region_matrix
            region_block = np.zeros((batch_len, num_regions), dtype=np.int32)
            for area in ratios:
                if area not in region_col_map:
                    continue
                reg_col = region_col_map[area]
                col_mask = area_col_masks[area]
                region_block[:, reg_col] = rounded[:, col_mask].sum(axis=1)

            region_dset[row_start:row_end, :] = region_block
            new_region_sums += region_block.astype(np.int64).sum(axis=0)

            t_now = time.time()
            if verbose and (t_now - t_last_report >= 5.0
                            or batch_idx == total_batches - 1):
                elapsed = t_now - t_proc_start
                frac = (batch_idx + 1) / total_batches
                eta = (elapsed / frac - elapsed) if frac > 0.01 else 0.0
                rows_per_sec = row_end / elapsed if elapsed > 0 else 0.0
                print(f"  Batch {batch_idx+1}/{total_batches} "
                      f"({frac:.1%}) | "
                      f"{rows_per_sec:,.0f} rows/s | "
                      f"elapsed {elapsed:.1f}s | "
                      f"ETA {eta:.1f}s", end="\r")
                t_last_report = t_now

        t_proc_elapsed = time.time() - t_proc_start
        if verbose:
            print(f"\n  Processing done in {t_proc_elapsed:.1f}s "
                  f"({num_ncs / t_proc_elapsed:,.0f} rows/s)")

        # Cross-check
        if verbose:
            print("\nCross-check: region_matrix sums")
            print(f"  {'Area':<6} {'Original':>14} {'Expected':>14} "
                  f"{'Actual':>14} {'Rel. Diff':>10}")
            print(f"  {'-' * 60}")

            for area in ["pit", "bot", "top", "wall"]:
                if area not in region_col_map:
                    continue
                reg_col = region_col_map[area]
                orig = original_region_sums[reg_col]
                expected = orig / ratios[area]
                actual = new_region_sums[reg_col]
                rel_diff = (actual - expected) / expected if expected > 0 else 0.0
                print(f"  {area:<6} {orig:>14,.0f} {expected:>14,.0f} "
                      f"{actual:>14,} {rel_diff:>+10.4%}")

    if verbose:
        print(f"\nOutput written to {output_path}")


def fmt_ratio_filename(ratios: dict[str, float]) -> str:
    """
    Format ratios for use in output filenames.

    Parameters
    ----------
    ratios : dict[str, float]
        Area ratios.

    Returns
    -------
    str
        Formatted string, e.g. "pit2p0731_bot2p3843_top2p2004_wall1p8776".
    """
    def _fmt(r: float) -> str:
        return f"{r:.4f}".replace(".", "p")

    return "_".join(f"{a}{_fmt(ratios[a])}" for a in ["pit", "bot", "top", "wall"])