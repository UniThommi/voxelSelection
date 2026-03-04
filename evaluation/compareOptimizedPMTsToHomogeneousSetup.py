#!/usr/bin/env python3
"""
Compare neutron capture detection between two PMT geometry simulations.

Sim 1: Homogeneous PMT placement (reference)
Sim 2: Optimized PMT placement

Detection criterion: An NC is "detected" if it has hits on >= m distinct
det_uids within 200 ns of the NC time.

Usage:
    python compare_nc_detection.py [--m_threshold 6]
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─── Configuration ───────────────────────────────────────────────────────────

SIM1_DIR = (
    "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/rawOpticalHomogeneousPMTsFromMusunNCs/run_20260216_195734"
)
SIM2_DIR = (
    "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/rawOpticalOptimizedPMTsFromMusunNCs/run_20260217_023810"
)
NC_CSV = (
    "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/gammaGunFormatOpticalSSD/merged_ncs.csv"
)
TIME_CUT_NS = 200.0


def load_optical_data(sim_dir: str) -> pd.DataFrame:
    """
    Load and concatenate optical photon data from all output_t*.hdf5 files.

    Returns:
        DataFrame with columns: det_uid, evtid, time_in_ns, muon_track_id, nC_track_id
    """
    hdf5_files = sorted(glob.glob(os.path.join(sim_dir, "output_t*.hdf5")))
    if not hdf5_files:
        raise FileNotFoundError(f"No output_t*.hdf5 files found in {sim_dir}")

    print(f"  Found {len(hdf5_files)} HDF5 files in {Path(sim_dir).name}")

    chunks: list[dict[str, np.ndarray]] = []
    fields = ["det_uid", "evtid", "time_in_ns", "muon_track_id", "nC_track_id"]

    for fpath in hdf5_files:
        with h5py.File(fpath, "r") as f:
            grp = f["hit/optical"]
            n_entries = int(grp["det_uid"]["entries"][()])
            if n_entries == 0:
                continue
            chunk = {field: grp[field]["pages"][:n_entries] for field in fields}
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No optical photon entries found in {sim_dir}")

    combined = {field: np.concatenate([c[field] for c in chunks]) for field in fields}
    df = pd.DataFrame(combined)
    print(f"  Loaded {len(df):,} optical photon hits")
    return df


def load_nc_data(csv_path: str) -> pd.DataFrame:
    """Load neutron capture metadata from CSV."""
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} neutron captures from CSV")
    return df


def apply_time_cut_and_count(
    optical_df: pd.DataFrame,
    nc_df: pd.DataFrame,
    m_threshold: int,
) -> set[tuple[int, int]]:
    """
    For each NC, filter optical photons arriving within TIME_CUT_NS of the
    NC time, then check if >= m_threshold distinct det_uids are hit.

    Returns:
        Set of (muon_id, nc_id) tuples that pass the detection criterion.
    """
    # Build NC time lookup: (muon_id, nc_id) -> nc_time
    nc_time_map = dict(
        zip(
            zip(nc_df["muon_id"], nc_df["nc_id"]),
            nc_df["nc_time"],
        )
    )

    # Map NC times onto optical DataFrame via vectorized lookup
    nc_keys = list(zip(optical_df["muon_track_id"], optical_df["nC_track_id"]))
    nc_times = np.array([nc_time_map.get(k, np.nan) for k in nc_keys])

    # Time since NC
    dt = optical_df["time_in_ns"].values - nc_times

    # Filter: valid match and within time window
    valid_mask = (~np.isnan(nc_times)) & (dt >= 0) & (dt <= TIME_CUT_NS)

    n_unmatched = np.isnan(nc_times).sum()
    if n_unmatched > 0:
        print(f"  Warning: {n_unmatched:,} photons could not be matched to an NC")

    filtered = optical_df.loc[valid_mask].copy()
    print(f"  {len(filtered):,} photons pass time cut (≤{TIME_CUT_NS} ns)")

    # Group by NC and count distinct det_uids
    grouped = (
        filtered.groupby(["muon_track_id", "nC_track_id"])["det_uid"]
        .nunique()
        .reset_index(name="n_det_uids")
    )

    max_det_uids = grouped["n_det_uids"].max()
    if max_det_uids > 300:
        raise ValueError(
            f"Sanity check failed: max distinct det_uids per NC = {max_det_uids} (>300). "
            f"Possible data mismatch or incorrect time cut."
        )

    detected = grouped.loc[grouped["n_det_uids"] >= m_threshold]
    detected_set = set(
        zip(detected["muon_track_id"], detected["nC_track_id"])
    )

    print(f"  {len(detected_set):,} NCs detected (≥{m_threshold} distinct det_uids)")
    return detected_set


def plot_nc_positions(
    nc_df: pd.DataFrame,
    detected: set[tuple[int, int]],
    all_nc_keys: set[tuple[int, int]],
    title: str,
    filename: str,
    output_dir: str,
) -> None:
    """
    Scatter plot of NC positions: green = detected, red = not detected.
    Only NCs present in all_nc_keys (i.e., the full NC set) are plotted.
    """
    nc_subset = nc_df[
        nc_df.apply(lambda r: (r["muon_id"], r["nc_id"]) in all_nc_keys, axis=1)
    ].copy()

    nc_subset["detected"] = nc_subset.apply(
        lambda r: (r["muon_id"], r["nc_id"]) in detected, axis=1
    )

    det = nc_subset[nc_subset["detected"]]
    undet = nc_subset[~nc_subset["detected"]]

    n_det = len(det)
    n_total = len(nc_subset)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{title}\nDetected: {n_det:,} / {n_total:,}", fontsize=13)

    projections = [
        ("nc_x", "nc_y", "X [m]", "Y [m]"),
        ("nc_x", "nc_z", "X [m]", "Z [m]"),
        ("nc_y", "nc_z", "Y [m]", "Z [m]"),
    ]

    for ax, (cx, cy, xlabel, ylabel) in zip(axes, projections):
        ax.scatter(
            undet[cx], undet[cy],
            c="red", s=4, alpha=0.3, label=f"Not detected ({n_total - n_det:,})",
        )
        ax.scatter(
            det[cx], det[cy],
            c="green", s=4, alpha=0.3, label=f"Detected ({n_det:,})",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, markerscale=3)
        ax.set_aspect("equal")

    plt.tight_layout()
    fpath = os.path.join(output_dir, filename)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


def plot_difference(
    nc_df: pd.DataFrame,
    only_set: set[tuple[int, int]],
    all_nc_keys: set[tuple[int, int]],
    title: str,
    filename: str,
    output_dir: str,
) -> None:
    """
    Plot NCs detected in one sim but not the other (green),
    with all other NCs in gray for context.
    """
    nc_subset = nc_df[
        nc_df.apply(lambda r: (r["muon_id"], r["nc_id"]) in all_nc_keys, axis=1)
    ].copy()

    nc_subset["highlighted"] = nc_subset.apply(
        lambda r: (r["muon_id"], r["nc_id"]) in only_set, axis=1
    )

    highlighted = nc_subset[nc_subset["highlighted"]]
    rest = nc_subset[~nc_subset["highlighted"]]

    n_only = len(highlighted)
    n_total = len(nc_subset)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{title}\n{n_only:,} / {n_total:,} NCs", fontsize=13)

    projections = [
        ("nc_x", "nc_y", "X [m]", "Y [m]"),
        ("nc_x", "nc_z", "X [m]", "Z [m]"),
        ("nc_y", "nc_z", "Y [m]", "Z [m]"),
    ]

    for ax, (cx, cy, xlabel, ylabel) in zip(axes, projections):
        ax.scatter(
            rest[cx], rest[cy],
            c="lightgray", s=3, alpha=0.2, label=f"Other ({n_total - n_only:,})",
        )
        ax.scatter(
            highlighted[cx], highlighted[cy],
            c="green", s=6, alpha=0.5, label=f"Exclusive ({n_only:,})",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, markerscale=3)
        ax.set_aspect("equal")

    plt.tight_layout()
    fpath = os.path.join(output_dir, filename)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


def main(m_threshold: int = 6) -> None:
    """Main analysis pipeline."""
    output_dir = os.path.join(os.getcwd(), "optimizedPMTsComparison")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # ── Load NC metadata ─────────────────────────────────────────────────
    print("Loading NC metadata...")
    nc_df = load_nc_data(NC_CSV)
    all_nc_keys = set(zip(nc_df["muon_id"], nc_df["nc_id"]))
    print(f"  Total unique NCs in CSV: {len(all_nc_keys):,}\n")

    # ── Load & process Sim 1 (homogeneous) ───────────────────────────────
    print("Loading Sim 1 (homogeneous PMTs)...")
    opt1 = load_optical_data(SIM1_DIR)
    print("Applying time cut & detection criterion...")
    detected_sim1 = apply_time_cut_and_count(opt1, nc_df, m_threshold)
    del opt1  # free memory
    print()

    # ── Load & process Sim 2 (optimized) ─────────────────────────────────
    print("Loading Sim 2 (optimized PMTs)...")
    opt2 = load_optical_data(SIM2_DIR)
    print("Applying time cut & detection criterion...")
    detected_sim2 = apply_time_cut_and_count(opt2, nc_df, m_threshold)
    del opt2  # free memory
    print()

    # ── Comparison ───────────────────────────────────────────────────────
    only_sim1 = detected_sim1 - detected_sim2
    only_sim2 = detected_sim2 - detected_sim1
    both = detected_sim1 & detected_sim2

    print("=" * 60)
    print(f"Detection threshold: ≥{m_threshold} distinct det_uids, ≤{TIME_CUT_NS} ns")
    print(f"Total NCs in dataset:          {len(all_nc_keys):>10,}")
    print(f"Detected by Sim 1 (homog.):    {len(detected_sim1):>10,}")
    print(f"Detected by Sim 2 (optimized): {len(detected_sim2):>10,}")
    print(f"Detected by both:              {len(both):>10,}")
    print(f"Only Sim 1 (not Sim 2):        {len(only_sim1):>10,}")
    print(f"Only Sim 2 (not Sim 1):        {len(only_sim2):>10,}")
    print(f"Improvement (Sim2 - Sim1):     {len(detected_sim2) - len(detected_sim1):>+10,}")
    print("=" * 60)
    print()

    # ── Plots ────────────────────────────────────────────────────────────
    print("Generating plots...")

    plot_nc_positions(
        nc_df, detected_sim1, all_nc_keys,
        title="Sim 1: Homogeneous PMTs",
        filename="sim1_homogeneous_nc_positions.png",
        output_dir=output_dir,
    )

    plot_nc_positions(
        nc_df, detected_sim2, all_nc_keys,
        title="Sim 2: Optimized PMTs",
        filename="sim2_optimized_nc_positions.png",
        output_dir=output_dir,
    )

    plot_difference(
        nc_df, only_sim1, all_nc_keys,
        title="Only detected by Sim 1 (homogeneous), NOT by Sim 2 (optimized)",
        filename="only_sim1_not_sim2.png",
        output_dir=output_dir,
    )

    plot_difference(
        nc_df, only_sim2, all_nc_keys,
        title="Only detected by Sim 2 (optimized), NOT by Sim 1 (homogeneous)",
        filename="only_sim2_not_sim1.png",
        output_dir=output_dir,
    )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare NC detection between two PMT geometry simulations."
    )
    parser.add_argument(
        "--m_threshold",
        type=int,
        default=6,
        help="Minimum number of distinct det_uids for detection (default: 6)",
    )
    args = parser.parse_args()
    main(m_threshold=args.m_threshold)