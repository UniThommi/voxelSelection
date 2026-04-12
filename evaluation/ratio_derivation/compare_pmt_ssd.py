#!/usr/bin/env python3
"""compare_pmt_ssd.py

Compare PMT simulation hits vs. SSD postprocessed voxel hits
for LEGEND neutron capture optical photon simulations.

Plots:
  1. Histogram of hit differences (Σhits_PMT - Σhits_Voxel) over all NCs, per pair
  2. Scatter: Σhits_PMT vs Σhits_Voxel (300 points)
  3. Spatial maps of relative deviation per detector region (bot/pit, top, wall)
  4. Boxplot of hit differences per pair (sorted by z-coordinate)
  5. Event-matched total hits correlation (PMT event vs SSD event)

Author: Thomas Buerger (University of Tübingen)
"""

import argparse
import glob
import os
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Default paths
# ─────────────────────────────────────────────────────────────
_DEFAULT_PMT_SIM_DIR = (
    "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/"
    "rawOpticalHomogeneousPMTsFromMusunNCs/run_20260216_195734"
)
_DEFAULT_SSD_POSTPROCESSED_FILE = (
    "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/"
    "MLFormatMusunNCsZylSSD300PMTs/ncscore_output_0.hdf5"
)
_DEFAULT_MERGED_NCS_CSV_DIR = (
    "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/"
    "gammaGunFormatOpticalSSD"
)
_DEFAULT_RAW_NC_SIM_DIR = (
    "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/"
    "rawMusunNCsSSD"
)

MAX_PMTS = 300

# Module-level path variables — set from CLI args in main()
PMT_SIM_DIR: str = _DEFAULT_PMT_SIM_DIR
SSD_POSTPROCESSED_FILE: str = _DEFAULT_SSD_POSTPROCESSED_FILE
MERGED_NCS_CSV_DIR: str = _DEFAULT_MERGED_NCS_CSV_DIR
RAW_NC_SIM_DIR: str = _DEFAULT_RAW_NC_SIM_DIR
OUTPUT_DIR: Path = Path.cwd() / "comparison_plots"


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare PMT simulation hits vs. SSD postprocessed voxel hits."
    )
    p.add_argument(
        '--pmt-sim-dir', type=str, default=_DEFAULT_PMT_SIM_DIR,
        help="Directory with raw optical PMT HDF5 files (output_t*.hdf5) "
             "(default: %(default)s)"
    )
    p.add_argument(
        '--ssd-file', type=str, default=_DEFAULT_SSD_POSTPROCESSED_FILE,
        help="SSD postprocessed HDF5 file (default: %(default)s)"
    )
    p.add_argument( # FIX not needed anymore
        '--merged-ncs-csv-dir', type=str, default=_DEFAULT_MERGED_NCS_CSV_DIR,
        help="Directory containing merged_ncs.csv for NC-to-run mapping "
             "(default: %(default)s)"
    )
    p.add_argument(
        '--raw-nc-sim-dir', type=str, default=_DEFAULT_RAW_NC_SIM_DIR,
        help="Directory containing raw NC simulation run subdirectories "
             "(default: %(default)s)"
    )
    p.add_argument(
        '--output-dir', type=Path, default=Path.cwd() / "comparison_plots",
        help="Output directory for plots (default: %(default)s)"
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────
def det_uid_to_voxel_id(det_uid: int) -> str:
    """Convert det_uid (int, always 8 digits) to voxel index string.

    - If second digit is '0': prefix is '10', voxel_id is 6 digits (e.g. 10000013 -> 000013)
    - If second digit is not '0': prefix is '1', voxel_id is 7 digits (e.g. 13001118 -> 3001118)
    """
    uid_str = str(det_uid)
    if uid_str[1] == "0":
        return uid_str[2:]  # Remove '10' prefix
    else:
        return uid_str[1:]  # Remove '1' prefix


def classify_region(voxel_id: str) -> str:
    """Classify voxel into detector region based on first two digits.

    00xxxx -> pit
    01xxxx -> bot
    99xxxx -> top
    else   -> wall
    """
    prefix = voxel_id[:2]
    if prefix == "00":
        return "pit"
    elif prefix == "01":
        return "bot"
    elif prefix == "99":
        return "top"
    else:
        return "wall"


# ─────────────────────────────────────────────────────────────
# Step 1: Load PMT simulation data
# ─────────────────────────────────────────────────────────────
def load_pmt_hits() -> dict:
    """Load PMT hits from all output files, return {voxel_id: total_hits}.

    Also validates that there are at most MAX_PMTS unique det_uids.
    """
    print("=" * 60)
    print("Loading PMT simulation data...")
    print(f"  Directory: {PMT_SIM_DIR}")

    pmt_files = sorted(glob.glob(os.path.join(PMT_SIM_DIR, "output_t*.hdf5")))
    if not pmt_files:
        raise FileNotFoundError(f"No output_t*.hdf5 files found in {PMT_SIM_DIR}")
    print(f"  Found {len(pmt_files)} files")

    all_det_uids = set()
    hits_per_voxel: dict = defaultdict(int)

    for fpath in pmt_files:
        with h5py.File(fpath, "r") as f:
            optical = f["hit"]["optical"]
            n_entries = int(optical["entries"][()])
            if n_entries == 0:
                continue

            det_uids = optical["det_uid"]["pages"][:]
            unique_uids = set(det_uids.tolist())
            all_det_uids.update(unique_uids)

            uid_values, counts = np.unique(det_uids, return_counts=True)
            for uid, count in zip(uid_values, counts):
                voxel_id = det_uid_to_voxel_id(int(uid))
                hits_per_voxel[voxel_id] += int(count)

    n_unique = len(all_det_uids)
    print(f"  Unique det_uids across all files: {n_unique}")

    if n_unique > MAX_PMTS:
        raise RuntimeError(
            f"ABORT: Found {n_unique} unique det_uids, expected at most {MAX_PMTS}!"
        )

    print(f"  Total voxel positions with hits: {len(hits_per_voxel)}")
    total_hits = sum(hits_per_voxel.values())
    print(f"  Total optical hits: {total_hits}")

    return dict(hits_per_voxel)


# ─────────────────────────────────────────────────────────────
# Step 2: Load SSD postprocessed voxel hits (only matching voxels)
# ─────────────────────────────────────────────────────────────
def load_ssd_voxel_hits(voxel_ids: list) -> dict:
    """Load SSD postprocessed hits for specified voxel IDs.

    Returns {voxel_id: total_hits_over_all_NCs}.
    """
    print("=" * 60)
    print("Loading SSD postprocessed voxel data...")
    print(f"  File: {SSD_POSTPROCESSED_FILE}")
    print(f"  Voxels to load: {len(voxel_ids)}")

    hits_per_voxel: dict = {}

    with h5py.File(SSD_POSTPROCESSED_FILE, "r") as f:
        all_col_ids = [c.decode() if isinstance(c, bytes) else str(c)
                       for c in f["target_columns"][:]]
        voxel_to_col = {v: i for i, v in enumerate(all_col_ids)}
        available_voxels = set(all_col_ids)
        mat = f["target_matrix"]

        missing = [v for v in voxel_ids if v not in available_voxels]
        if missing:
            print(f"  WARNING: {len(missing)} voxel IDs not found in SSD target:")
            for m in missing[:10]:
                print(f"    {m}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")

        for voxel_id in voxel_ids:
            if voxel_id in voxel_to_col:
                col = voxel_to_col[voxel_id]
                hits_per_voxel[voxel_id] = int(mat[:, col].sum())
            else:
                hits_per_voxel[voxel_id] = 0

    print(f"  Loaded {len(hits_per_voxel)} voxels")
    total_hits = sum(hits_per_voxel.values())
    print(f"  Total SSD voxel hits: {total_hits}")

    return hits_per_voxel


# ─────────────────────────────────────────────────────────────
# Step 3: Load voxel geometry for spatial plots
# ─────────────────────────────────────────────────────────────
def load_voxel_centers(voxel_ids: list) -> dict:
    """Load voxel center coordinates from SSD postprocessed file.

    Returns {voxel_id: np.array([x, y, z])}.
    """
    print("=" * 60)
    print("Loading voxel center coordinates...")

    centers: dict = {}

    with h5py.File(SSD_POSTPROCESSED_FILE, "r") as f:
        voxels_grp = f["voxels"]
        for voxel_id in voxel_ids:
            if voxel_id in voxels_grp:
                center = voxels_grp[voxel_id]["center"][:]
                centers[voxel_id] = center
            else:
                print(f"  WARNING: No center found for voxel {voxel_id}")

    print(f"  Loaded {len(centers)} voxel centers")
    return centers


# ─────────────────────────────────────────────────────────────
# Step 4: Load per-NC hits for boxplot (Plot 4)
# ─────────────────────────────────────────────────────────────
def load_pmt_hits_per_nc() -> dict:
    """Load PMT hits per NC event per voxel.

    Returns {voxel_id: {evtid: hit_count}}.
    """
    print("=" * 60)
    print("Loading PMT hits per NC event...")

    hits: dict = defaultdict(lambda: defaultdict(int))

    pmt_files = sorted(glob.glob(os.path.join(PMT_SIM_DIR, "output_t*.hdf5")))

    for fpath in pmt_files:
        with h5py.File(fpath, "r") as f:
            optical = f["hit"]["optical"]
            n_entries = int(optical["entries"][()])
            if n_entries == 0:
                continue

            det_uids = optical["det_uid"]["pages"][:]
            evtids = optical["evtid"]["pages"][:]

            for uid, evt in zip(det_uids, evtids):
                voxel_id = det_uid_to_voxel_id(int(uid))
                hits[voxel_id][int(evt)] += 1

    return dict(hits)


def load_ssd_hits_per_nc(voxel_ids: list) -> dict:
    """Load per-NC hit arrays for specified voxels from SSD postprocessed.

    Returns {voxel_id: np.array of hits per NC}.
    """
    print("Loading SSD per-NC hits for boxplot...")

    per_nc: dict = {}

    with h5py.File(SSD_POSTPROCESSED_FILE, "r") as f:
        all_col_ids = [c.decode() if isinstance(c, bytes) else str(c)
                       for c in f["target_columns"][:]]
        voxel_to_col = {v: i for i, v in enumerate(all_col_ids)}
        mat = f["target_matrix"]
        for voxel_id in voxel_ids:
            if voxel_id in voxel_to_col:
                col = voxel_to_col[voxel_id]
                per_nc[voxel_id] = mat[:, col][:]

    return per_nc


# ─────────────────────────────────────────────────────────────
# Plotting functions
# ─────────────────────────────────────────────────────────────
def plot1_hit_difference_histogram(
    pmt_hits: dict, ssd_hits: dict
) -> None:
    """Plot 1: Histogram of Δhits = Σhits_PMT - Σhits_Voxel per pair."""
    print("\n--- Plot 1: Hit Difference Histogram ---")

    voxel_ids = sorted(set(pmt_hits.keys()) & set(ssd_hits.keys()))
    deltas = np.array([pmt_hits[v] - ssd_hits[v] for v in voxel_ids])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute difference
    ax = axes[0]
    ax.hist(deltas, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Δ = 0")
    ax.axvline(np.mean(deltas), color="orange", linestyle="-", linewidth=1.5,
               label=f"Mean = {np.mean(deltas):.1f}")
    ax.set_xlabel("Δhits = Σhits_PMT − Σhits_Voxel")
    ax.set_ylabel("Number of PMT/Voxel pairs")
    ax.set_title("Absolute Hit Difference (summed over all NCs)")
    ax.legend()

    # Relative difference
    ax = axes[1]
    pmt_arr = np.array([pmt_hits[v] for v in voxel_ids], dtype=float)
    ssd_arr = np.array([ssd_hits[v] for v in voxel_ids], dtype=float)
    denom = 0.5 * (pmt_arr + ssd_arr)
    mask = denom > 0
    rel_deltas = np.full_like(deltas, dtype=float, fill_value=np.nan)
    rel_deltas[mask] = deltas[mask] / denom[mask]

    ax.hist(rel_deltas[mask], bins=50, edgecolor="black", alpha=0.7, color="coral")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(np.nanmean(rel_deltas), color="orange", linestyle="-", linewidth=1.5,
               label=f"Mean = {np.nanmean(rel_deltas):.4f}")
    ax.set_xlabel("Relative Δhits / mean(hits)")
    ax.set_ylabel("Number of PMT/Voxel pairs")
    ax.set_title("Relative Hit Difference (summed over all NCs)")
    ax.legend()

    plt.tight_layout()
    fpath = OUTPUT_DIR / "plot1_hit_difference_histogram.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


def plot2_scatter_pmt_vs_ssd(
    pmt_hits: dict, ssd_hits: dict
) -> None:
    """Plot 2: Scatter plot of total hits PMT vs Voxel."""
    print("\n--- Plot 2: Scatter PMT vs SSD ---")

    voxel_ids = sorted(set(pmt_hits.keys()) & set(ssd_hits.keys()))
    pmt_arr = np.array([pmt_hits[v] for v in voxel_ids])
    ssd_arr = np.array([ssd_hits[v] for v in voxel_ids])

    regions = [classify_region(v) for v in voxel_ids]
    region_colors = {"pit": "blue", "bot": "cyan", "top": "red", "wall": "green"}

    fig, ax = plt.subplots(figsize=(8, 8))

    for region, color in region_colors.items():
        mask = np.array([r == region for r in regions])
        if mask.any():
            ax.scatter(
                pmt_arr[mask], ssd_arr[mask],
                c=color, label=f"{region} ({mask.sum()})",
                alpha=0.7, edgecolors="black", linewidth=0.5, s=40
            )

    max_val = max(pmt_arr.max(), ssd_arr.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Σ hits PMT (over all NCs)")
    ax.set_ylabel("Σ hits SSD Voxel (over all NCs)")
    ax.set_title("PMT vs SSD Voxel Total Hits")
    ax.legend()
    ax.set_aspect("equal")

    plt.tight_layout()
    fpath = OUTPUT_DIR / "plot2_scatter_pmt_vs_ssd.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


def plot3_spatial_deviation(
    pmt_hits: dict,
    ssd_hits: dict,
    centers: dict,
) -> None:
    """Plot 3: Spatial maps of relative deviation per detector region."""
    print("\n--- Plot 3: Spatial Deviation Maps ---")

    voxel_ids = sorted(
        set(pmt_hits.keys()) & set(ssd_hits.keys()) & set(centers.keys())
    )

    data = []
    for v in voxel_ids:
        pmt_h = pmt_hits[v]
        ssd_h = ssd_hits[v]
        denom = 0.5 * (pmt_h + ssd_h)
        rel_dev = (pmt_h - ssd_h) / denom if denom > 0 else 0.0
        region = classify_region(v)
        cx, cy, cz = centers[v]
        r = np.sqrt(cx**2 + cy**2)
        phi = np.degrees(np.arctan2(cy, cx))
        data.append({
            "voxel_id": v, "region": region,
            "x": cx, "y": cy, "z": cz, "r": r, "phi": phi,
            "rel_dev": rel_dev,
        })

    df = pd.DataFrame(data)

    vmax = max(abs(df["rel_dev"].min()), abs(df["rel_dev"].max()))
    if vmax == 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Bot + Pit: x vs y
    ax = axes[0]
    subset = df[df["region"].isin(["bot", "pit"])]
    if not subset.empty:
        sc = ax.scatter(
            subset["x"], subset["y"], c=subset["rel_dev"],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            edgecolors="black", linewidth=0.5, s=50
        )
        plt.colorbar(sc, ax=ax, label="Relative deviation")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Bot + Pit (x-y projection)")
    ax.set_aspect("equal")

    # Top: x vs y
    ax = axes[1]
    subset = df[df["region"] == "top"]
    if not subset.empty:
        sc = ax.scatter(
            subset["x"], subset["y"], c=subset["rel_dev"],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            edgecolors="black", linewidth=0.5, s=50
        )
        plt.colorbar(sc, ax=ax, label="Relative deviation")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Top (x-y projection)")
    ax.set_aspect("equal")

    # Wall: z vs phi
    ax = axes[2]
    subset = df[df["region"] == "wall"]
    if not subset.empty:
        sc = ax.scatter(
            subset["phi"], subset["z"], c=subset["rel_dev"],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            edgecolors="black", linewidth=0.5, s=50
        )
        plt.colorbar(sc, ax=ax, label="Relative deviation")
    ax.set_xlabel("φ [deg]")
    ax.set_ylabel("z [mm]")
    ax.set_title("Wall (φ-z projection)")

    plt.tight_layout()
    fpath = OUTPUT_DIR / "plot3_spatial_deviation.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


def plot4_boxplot_per_pair(
    pmt_hits_per_nc: dict,
    ssd_hits_per_nc: dict,
    centers: dict,
) -> None:
    """Plot 4: Boxplot of hit distribution differences per pair, sorted by z."""
    print("\n--- Plot 4: Boxplot per PMT/Voxel pair ---")

    voxel_ids = sorted(
        set(pmt_hits_per_nc.keys()) & set(ssd_hits_per_nc.keys()) & set(centers.keys())
    )

    voxel_ids_sorted = sorted(voxel_ids, key=lambda v: centers[v][2])

    pmt_distributions = []
    ssd_distributions = []

    for v in voxel_ids_sorted:
        pmt_per_nc = (
            np.array(list(pmt_hits_per_nc[v].values()))
            if v in pmt_hits_per_nc else np.array([])
        )
        ssd_per_nc_arr = ssd_hits_per_nc.get(v, np.array([]))

        pmt_distributions.append(pmt_per_nc)
        ssd_distributions.append(ssd_per_nc_arr)

    n_pairs = len(voxel_ids_sorted)

    pmt_means = np.array([d.mean() if len(d) > 0 else 0 for d in pmt_distributions])
    pmt_stds = np.array([d.std() if len(d) > 0 else 0 for d in pmt_distributions])
    ssd_means = np.array([d.mean() if len(d) > 0 else 0 for d in ssd_distributions])
    ssd_stds = np.array([d.std() if len(d) > 0 else 0 for d in ssd_distributions])

    x = np.arange(n_pairs)

    fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    ax = axes[0]
    ax.errorbar(x, pmt_means, yerr=pmt_stds, fmt=".", color="steelblue",
                alpha=0.6, markersize=3, linewidth=0.5, label="PMT (mean ± std)")
    ax.errorbar(x, ssd_means, yerr=ssd_stds, fmt=".", color="coral",
                alpha=0.6, markersize=3, linewidth=0.5, label="SSD Voxel (mean ± std)")
    ax.set_ylabel("Hits per NC event")
    ax.set_title("Per-NC hit distribution per PMT/Voxel pair (sorted by z)")
    ax.legend()

    ax = axes[1]
    diff_means = pmt_means - ssd_means
    ax.bar(x, diff_means, color="steelblue", alpha=0.7, width=1.0)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("PMT/Voxel pair index (sorted by z-coordinate)")
    ax.set_ylabel("Δ mean hits per NC (PMT − SSD)")
    ax.set_title("Difference in mean hits per NC event")

    plt.tight_layout()
    fpath = OUTPUT_DIR / "plot4_boxplot_per_pair.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────
# Plot 5: Event-matched correlation (complex NC matching)
# ─────────────────────────────────────────────────────────────
def load_pmt_nc_identifiers() -> pd.DataFrame:
    """Load (muon_track_id, nC_track_id, evtid, total_hits) from PMT sim.

    Groups optical hits by evtid, extracts the muon_track_id and nC_track_id
    per event. Returns DataFrame with one row per NC event.
    """
    print("=" * 60)
    print("Loading PMT NC identifiers for event matching...")

    records = []
    pmt_files = sorted(glob.glob(os.path.join(PMT_SIM_DIR, "output_t*.hdf5")))

    for fpath in pmt_files:
        with h5py.File(fpath, "r") as f:
            optical = f["hit"]["optical"]
            n_entries = int(optical["entries"][()])
            if n_entries == 0:
                continue

            evtids = optical["evtid"]["pages"][:]
            muon_ids = optical["muon_track_id"]["pages"][:]
            nc_ids = optical["nC_track_id"]["pages"][:]

            unique_evts = np.unique(evtids)
            for evt in unique_evts:
                mask = evtids == evt
                total_hits = int(mask.sum())
                muon_id = int(muon_ids[mask][0])
                nc_id = int(nc_ids[mask][0])
                records.append({
                    "evtid": int(evt),
                    "muon_track_id": muon_id,
                    "nC_track_id": nc_id,
                    "total_hits_pmt": total_hits,
                })

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df)} unique NC events from PMT sim")
    return df


def load_merged_ncs_csv() -> pd.DataFrame:
    """Load merged_ncs.csv for NC-to-run mapping."""
    csv_files = glob.glob(os.path.join(MERGED_NCS_CSV_DIR, "merged_ncs.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"merged_ncs.csv not found in {MERGED_NCS_CSV_DIR}"
        )
    print(f"  Loading: {csv_files[0]}")
    df = pd.read_csv(csv_files[0])
    print(f"  Loaded {len(df)} NC entries from CSV")
    return df


def lookup_nc_properties(
    orig_muon_id: int, nc_id: int, run_id: int
) -> dict | None:
    """Look up NC properties from raw NC simulation files.

    Searches in rawMusunNCsSSD/run_{run_id:03d}/output_t*.hdf5
    for the event with evtid=orig_muon_id and nC_track_id=nc_id.
    Returns dict with NC properties or None.
    """
    run_dir = os.path.join(RAW_NC_SIM_DIR, f"run_{run_id:03d}")
    if not os.path.isdir(run_dir):
        return None

    raw_files = sorted(glob.glob(os.path.join(run_dir, "output_t*.hdf5")))

    for fpath in raw_files:
        with h5py.File(fpath, "r") as f:
            if "MyNeutronCaptureOutput" not in f["hit"]:
                continue

            nc_out = f["hit"]["MyNeutronCaptureOutput"]
            n_entries = int(nc_out["entries"][()])
            if n_entries == 0:
                continue

            evtids = nc_out["evtid"]["pages"][:]
            nc_ids_arr = nc_out["nC_track_id"]["pages"][:]

            mask = (evtids == orig_muon_id) & (nc_ids_arr == nc_id)
            if not mask.any():
                continue

            idx = np.where(mask)[0][0]

            props = {
                "nC_x_position_in_m": float(nc_out["nC_x_position_in_m"]["pages"][idx]),
                "nC_y_position_in_m": float(nc_out["nC_y_position_in_m"]["pages"][idx]),
                "nC_z_position_in_m": float(nc_out["nC_z_position_in_m"]["pages"][idx]),
                "nC_gamma_amount": int(nc_out["nC_gamma_amount"]["pages"][idx]),
                "nC_gamma_total_energy_in_keV": float(
                    nc_out["nC_gamma_total_energy_in_keV"]["pages"][idx]
                ),
                "gamma1_E_in_keV": float(nc_out["gamma1_E_in_keV"]["pages"][idx]),
                "gamma2_E_in_keV": float(nc_out["gamma2_E_in_keV"]["pages"][idx]),
                "gamma3_E_in_keV": float(nc_out["gamma3_E_in_keV"]["pages"][idx]),
                "gamma4_E_in_keV": float(nc_out["gamma4_E_in_keV"]["pages"][idx]),
            }
            return props

    return None


def plot5_event_matched_correlation(
    pmt_nc_df: pd.DataFrame,
    voxel_ids: list,
) -> None:
    """Plot 5: Matched NC events — total hits PMT vs total hits SSD."""
    print("\n--- Plot 5: Event-Matched Correlation ---")
    print("  This may take a while due to NC matching...")

    merged_csv = load_merged_ncs_csv()

    print("  Preloading SSD target arrays...")
    ssd_target_arrays: dict = {}
    with h5py.File(SSD_POSTPROCESSED_FILE, "r") as f:
        all_col_ids = [c.decode() if isinstance(c, bytes) else str(c)
                       for c in f["target_columns"][:]]
        voxel_to_col = {v: i for i, v in enumerate(all_col_ids)}
        mat = f["target_matrix"]
        for v in voxel_ids:
            if v in voxel_to_col:
                col = voxel_to_col[v]
                ssd_target_arrays[v] = mat[:, col][:]

    print("  Preloading SSD phi arrays for NC matching...")
    with h5py.File(SSD_POSTPROCESSED_FILE, "r") as f:
        phi_cols_list = [c.decode() if isinstance(c, bytes) else str(c)
                         for c in f["phi_columns"][:]]
        phi_col = {n: i for i, n in enumerate(phi_cols_list)}
        phi_mat = f["phi_matrix"][:]
        ssd_x      = phi_mat[:, phi_col["xNC_mm"]]
        ssd_y      = phi_mat[:, phi_col["yNC_mm"]]
        ssd_z      = phi_mat[:, phi_col["zNC_mm"]]
        ssd_ngamma = phi_mat[:, phi_col["#gamma"]]
        ssd_etot   = phi_mat[:, phi_col["E_gamma_tot_keV"]]
        ssd_e1     = phi_mat[:, phi_col["gammaE1_keV"]]
        ssd_e2     = phi_mat[:, phi_col["gammaE2_keV"]]
        ssd_e3     = phi_mat[:, phi_col["gammaE3_keV"]]
        ssd_e4     = phi_mat[:, phi_col["gammaE4_keV"]]

    matched_pmt_totals = []
    matched_ssd_totals = []
    n_matched = 0
    n_failed = 0
    n_total = len(pmt_nc_df)

    for i, row in pmt_nc_df.iterrows():
        if (i + 1) % 500 == 0:
            print(f"  Processing event {i + 1}/{n_total} "
                  f"(matched: {n_matched}, failed: {n_failed})")

        muon_id = row["muon_track_id"]
        nc_id = row["nC_track_id"]

        csv_match = merged_csv[
            (merged_csv["muon_id"] == muon_id) & (merged_csv["nc_id"] == nc_id)
        ]
        if csv_match.empty:
            n_failed += 1
            continue

        csv_row = csv_match.iloc[0]
        run_id = int(csv_row["run_id"])
        orig_muon_id = int(csv_row["orig_muon_id"])

        nc_props = lookup_nc_properties(orig_muon_id, nc_id, run_id)
        if nc_props is None:
            print(f"  WARNING: NC properties not found for orig_muon_id={orig_muon_id}, "
                  f"nc_id={nc_id}, run_id={run_id}")
            n_failed += 1
            continue

        target_x = np.float32(nc_props["nC_x_position_in_m"] * 1000.0)
        target_y = np.float32(nc_props["nC_y_position_in_m"] * 1000.0)
        target_z = np.float32(nc_props["nC_z_position_in_m"] * 1000.0)
        target_ngamma = np.float32(nc_props["nC_gamma_amount"])
        target_etot = np.float32(nc_props["nC_gamma_total_energy_in_keV"])
        target_e1 = np.float32(nc_props["gamma1_E_in_keV"])
        target_e2 = np.float32(nc_props["gamma2_E_in_keV"])
        target_e3 = np.float32(nc_props["gamma3_E_in_keV"])
        target_e4 = np.float32(nc_props["gamma4_E_in_keV"])

        mask = (
            (ssd_x == target_x)
            & (ssd_y == target_y)
            & (ssd_z == target_z)
            & (ssd_ngamma == target_ngamma)
            & (ssd_etot == target_etot)
            & (ssd_e1 == target_e1)
            & (ssd_e2 == target_e2)
            & (ssd_e3 == target_e3)
            & (ssd_e4 == target_e4)
        )

        indices = np.where(mask)[0]
        if len(indices) == 0:
            print(f"  WARNING: No match in SSD postprocessed for orig_muon_id={orig_muon_id}, "
                  f"nc_id={nc_id}")
            n_failed += 1
            continue

        ssd_event_idx = int(indices[0])

        ssd_total = sum(
            int(arr[ssd_event_idx]) for arr in ssd_target_arrays.values()
        )

        matched_pmt_totals.append(row["total_hits_pmt"])
        matched_ssd_totals.append(ssd_total)
        n_matched += 1

    print(f"  Matching complete: {n_matched} matched, {n_failed} failed")

    if n_matched == 0:
        print("  No events matched — skipping Plot 5")
        return

    pmt_arr = np.array(matched_pmt_totals)
    ssd_arr = np.array(matched_ssd_totals)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(pmt_arr, ssd_arr, alpha=0.3, s=10, color="steelblue")

    max_val = max(pmt_arr.max(), ssd_arr.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Total hits PMT (per NC event)")
    ax.set_ylabel("Total hits SSD 300 Voxels (per NC event)")
    ax.set_title(f"Event-Matched Hit Correlation ({n_matched} events)")
    ax.legend()

    plt.tight_layout()
    fpath = OUTPUT_DIR / "plot5_event_matched_correlation.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    global PMT_SIM_DIR, SSD_POSTPROCESSED_FILE, MERGED_NCS_CSV_DIR, RAW_NC_SIM_DIR
    global OUTPUT_DIR

    args = parse_args()

    PMT_SIM_DIR = args.pmt_sim_dir
    SSD_POSTPROCESSED_FILE = args.ssd_file
    MERGED_NCS_CSV_DIR = args.merged_ncs_csv_dir
    RAW_NC_SIM_DIR = args.raw_nc_sim_dir
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("PMT vs SSD Voxel Hit Comparison")
    print("=" * 60)

    # --- Cross-check: NC count consistency ---
    print("=" * 60)
    print("Cross-checking NC event counts...")
    pmt_files = sorted(glob.glob(os.path.join(PMT_SIM_DIR, "output_t*.hdf5")))
    all_evtids = set()
    for fpath in pmt_files:
        with h5py.File(fpath, "r") as f:
            vertices = f["hit"]["vertices"]
            n_entries = int(vertices["entries"][()])
            if n_entries == 0:
                continue
            evtids = vertices["evtid"]["pages"][:]
            all_evtids.update(evtids.tolist())
    n_pmt_events = len(all_evtids)

    with h5py.File(SSD_POSTPROCESSED_FILE, "r") as f:
        n_ssd_events = f["target_matrix"].shape[0]

    print(f"  Unique evtids in PMT vertices: {n_pmt_events}")
    print(f"  NC entries in SSD postprocessed: {n_ssd_events}")

    if n_pmt_events != n_ssd_events:
        raise RuntimeError(
            f"ABORT: NC count mismatch! PMT vertices: {n_pmt_events}, "
            f"SSD postprocessed: {n_ssd_events}"
        )
    print("  NC counts match")

    # --- Cross-check: zero-hit NC events ---
    print("\nComparing zero-hit NC events...")
    with h5py.File(SSD_POSTPROCESSED_FILE, "r") as f:
        ssd_voxel_keys = [c.decode() if isinstance(c, bytes) else str(c)
                          for c in f["target_columns"][:]]
        mat = f["target_matrix"]
        n_ncs = mat.shape[0]
        # Sum hits per NC across all voxels in batches to avoid huge allocation
        total_per_nc = np.zeros(n_ncs, dtype=np.int64)
        _BATCH = 1000
        for _rs in range(0, n_ncs, _BATCH):
            _re = min(_rs + _BATCH, n_ncs)
            total_per_nc[_rs:_re] = mat[_rs:_re, :].sum(axis=1)
        ssd_zero_ncs = int(np.sum(total_per_nc == 0))

    optical_evtids = set()
    for fpath in pmt_files:
        with h5py.File(fpath, "r") as f:
            n_opt = int(f["hit"]["optical"]["entries"][()])
            if n_opt > 0:
                optical_evtids.update(f["hit"]["optical"]["evtid"]["pages"][:].tolist())
    pmt_zero_ncs = n_pmt_events - len(optical_evtids)

    print(f"  Zero-hit NCs (SSD, all voxels = 0): {ssd_zero_ncs}")
    print(f"  Zero-hit NCs (PMT, no optical entry): {pmt_zero_ncs}")
    print(f"  Difference: {abs(ssd_zero_ncs - pmt_zero_ncs)}")

    # --- Load data for Plots 1-3 ---
    pmt_hits = load_pmt_hits()
    voxel_ids = sorted(pmt_hits.keys())

    with h5py.File(SSD_POSTPROCESSED_FILE, "r") as f:
        ssd_target_keys = set(
            c.decode() if isinstance(c, bytes) else str(c)
            for c in f["target_columns"][:]
        )
    matched_ids = [v for v in voxel_ids if v in ssd_target_keys]
    unmatched_ids = [v for v in voxel_ids if v not in ssd_target_keys]
    print(f"\n  PMT voxel IDs matched in SSD target: {len(matched_ids)}/{len(voxel_ids)}")
    if unmatched_ids:
        print(f"  Unmatched PMT voxel IDs (first 10): {unmatched_ids[:10]}")
        for uid in unmatched_ids[:5]:
            close = [k for k in ssd_target_keys if uid in k or k in uid]
            if close:
                print(f"    {uid} -> possible SSD matches: {close[:3]}")

    ssd_hits = load_ssd_voxel_hits(voxel_ids)
    centers = load_voxel_centers(voxel_ids)

    # --- Plots 1-3 ---
    plot1_hit_difference_histogram(pmt_hits, ssd_hits)
    plot2_scatter_pmt_vs_ssd(pmt_hits, ssd_hits)
    plot3_spatial_deviation(pmt_hits, ssd_hits, centers)

    # --- Plot 4: Per-NC distributions ---
    pmt_hits_per_nc = load_pmt_hits_per_nc()
    ssd_hits_per_nc = load_ssd_hits_per_nc(voxel_ids)
    plot4_boxplot_per_pair(pmt_hits_per_nc, ssd_hits_per_nc, centers)

    print("\n" + "=" * 60)
    print("Plots 1-4 saved successfully.")
    print("=" * 60)

    # --- Plot 5: Event matching (slow) ---
    try:
        pmt_nc_df = load_pmt_nc_identifiers()
        plot5_event_matched_correlation(pmt_nc_df, voxel_ids)
    except Exception as e:
        print(f"\n  Plot 5 failed: {e}")
        print("  Plots 1-4 are still saved.")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
