from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

NC_GROUP = "/hit/MyNeutronCaptureOutput"


@dataclass
class RunStats:
    """Aggregated statistics for a single simulation run."""
    run_name: str = ""
    n_files: int = 0
    nc_total: int = 0
    nc_ge77: int = 0
    muons_total: int = 0
    muons_with_ge77: int = 0


def read_pages(group: h5py.Group, field_name: str) -> np.ndarray:
    """Read a 'pages'-style field from the NC output group."""
    return group[field_name]["pages"][:]


def make_log_bins(vmin: int, vmax: int) -> np.ndarray:
    """
    Create bin edges that are linear within each decade:
    1,2,3,...,9, 10,20,30,...,90, 100,200,...,900, 1000,2000,...
    """
    edges: list[float] = []
    decade = 10 ** int(np.floor(np.log10(max(vmin, 1))))
    val = decade
    while val <= vmax:
        edges.append(val)
        val += decade
        if val >= decade * 10:
            decade *= 10
    edges.append(max(val, vmax + decade))  # upper edge of last bin
    return np.array(edges, dtype=float)


def analyze_file(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract evtid, nC_track_id, nC_flag_Ge77, and nC_time_in_ns from a single HDF5 file.

    Returns:
        Tuple of (evtid, nc_track_id, ge77_flag, nc_time), all 1D np.ndarray.
        Returns empty arrays if the group has no entries.
    """
    empty_i = np.array([], dtype=np.int32)
    empty_f = np.array([], dtype=np.float64)
    with h5py.File(filepath, "r") as f:
        if NC_GROUP not in f:
            return empty_i, empty_i, empty_i, empty_f

        grp = f[NC_GROUP]
        n_entries = int(grp["entries"][()])
        if n_entries == 0:
            return empty_i, empty_i, empty_i, empty_f

        evtid = read_pages(grp, "evtid")
        nc_track_id = read_pages(grp, "nC_track_id")
        ge77_flag = read_pages(grp, "nC_flag_Ge77")
        nc_time = read_pages(grp, "nC_time_in_ns")

    return evtid, nc_track_id, ge77_flag, nc_time


def analyze_run(run_dir: Path) -> RunStats:
    """
    Analyze all HDF5 files in a single run directory.

    Deduplicates NCs on unique (evtid, nc_track_id) pairs within a run,
    then counts unique muons (evtid) that have at least one Ge77-flagged NC.
    """
    stats = RunStats(run_name=run_dir.name)

    hdf5_files = sorted(run_dir.glob("output_t*.hdf5"))
    stats.n_files = len(hdf5_files)

    if stats.n_files == 0:
        print(f"  WARNING: No output_t*.hdf5 files found in {run_dir}")
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        return stats, empty_i, empty_i, empty_f, empty_f

    all_evtid: list[np.ndarray] = []
    all_nc_id: list[np.ndarray] = []
    all_ge77: list[np.ndarray] = []
    all_time: list[np.ndarray] = []

    for fp in hdf5_files:
        try:
            evtid, nc_id, ge77, nc_time = analyze_file(fp)
        except Exception as e:
            print(f"  ERROR reading {fp.name}: {e}")
            continue

        if evtid.size > 0:
            all_evtid.append(evtid)
            all_nc_id.append(nc_id)
            all_ge77.append(ge77)
            all_time.append(nc_time)

    if not all_evtid:
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        return stats, empty_i, empty_i, empty_f, empty_f

    evtid_arr = np.concatenate(all_evtid)
    nc_id_arr = np.concatenate(all_nc_id)
    ge77_arr = np.concatenate(all_ge77)
    time_arr = np.concatenate(all_time)

    # Deduplicate on unique (evtid, nc_track_id) pairs within this run.
    # For duplicates, keep the row with the max ge77 flag (1 wins over 0).
    pair_keys = np.stack([evtid_arr, nc_id_arr], axis=1)
    _, unique_idx, inverse = np.unique(
        pair_keys, axis=0, return_index=True, return_inverse=True
    )

    # For each unique pair, take max ge77 flag across duplicates
    unique_ge77 = np.zeros(len(unique_idx), dtype=np.int32)
    np.maximum.at(unique_ge77, inverse, ge77_arr)

    # For time, take the value at unique_idx (arbitrary pick among duplicates)
    unique_time = time_arr[unique_idx]

    unique_evtid = evtid_arr[unique_idx]
    ge77_mask = unique_ge77 == 1

    stats.nc_total = len(unique_idx)
    stats.nc_ge77 = int(ge77_mask.sum())
    stats.muons_total = len(np.unique(unique_evtid))
    stats.muons_with_ge77 = len(np.unique(unique_evtid[ge77_mask]))

    # Per-muon NC counts for histogram
    muon_ids, nc_counts = np.unique(unique_evtid, return_counts=True)

    # Per-muon NC counts only for muons with at least one Ge77 NC
    ge77_muon_ids = np.unique(unique_evtid[ge77_mask])
    ge77_muon_mask = np.isin(muon_ids, ge77_muon_ids)
    nc_counts_ge77 = nc_counts[ge77_muon_mask]

    # Time arrays: all NCs and Ge77-only NCs
    # For the "Ge77 muons only" time histogram: all NCs belonging to Ge77-producing muons
    is_ge77_muon = np.isin(unique_evtid, ge77_muon_ids)
    time_all = unique_time
    time_ge77_flag = unique_time[ge77_mask]
    time_ge77_muons_all = unique_time[is_ge77_muon]
    time_ge77_muons_ge77flag = unique_time[is_ge77_muon & ge77_mask]

    return stats, nc_counts, nc_counts_ge77, time_all, time_ge77_flag, time_ge77_muons_all, time_ge77_muons_ge77flag


base = Path("/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/rawMusunNCsSSD")

if not base.exists():
    print(f"ERROR: Base directory not found: {base}")
    sys.exit(1)

run_dirs = sorted(base.glob("run_0*"))
if not run_dirs:
    print(f"ERROR: No run_* directories found in {base}")
    sys.exit(1)

print(f"Base directory: {base}")
print(f"Found {len(run_dirs)} run directories\n")
print(f"{'Run':<12} {'Files':>6} {'NC total':>10} {'NC Ge77':>10} "
        f"{'Muons tot':>10} {'Muons Ge77':>11}")
print("-" * 65)

all_stats: list[RunStats] = []
all_nc_counts: list[np.ndarray] = []
all_nc_counts_ge77: list[np.ndarray] = []
all_time: list[np.ndarray] = []
all_time_ge77flag: list[np.ndarray] = []
all_time_ge77muons: list[np.ndarray] = []
all_time_ge77muons_ge77flag: list[np.ndarray] = []
for rd in run_dirs:
    print(f"Processing {rd.name}...", end=" ", flush=True)
    s, nc_counts, nc_counts_ge77, t_all, t_ge77f, t_ge77m, t_ge77mf = analyze_run(rd)
    all_stats.append(s)
    if nc_counts.size > 0:
        all_nc_counts.append(nc_counts)
    if nc_counts_ge77.size > 0:
        all_nc_counts_ge77.append(nc_counts_ge77)
    if t_all.size > 0:
        all_time.append(t_all)
        all_time_ge77flag.append(t_ge77f)
    if t_ge77m.size > 0:
        all_time_ge77muons.append(t_ge77m)
        all_time_ge77muons_ge77flag.append(t_ge77mf)
    print(f"\r{s.run_name:<12} {s.n_files:>6} {s.nc_total:>10} {s.nc_ge77:>10} "
            f"{s.muons_total:>10} {s.muons_with_ge77:>11}")

# --- Totals ---
print("-" * 65)
total_files = sum(s.n_files for s in all_stats)
total_nc = sum(s.nc_total for s in all_stats)
total_ge77 = sum(s.nc_ge77 for s in all_stats)
total_muons = sum(s.muons_total for s in all_stats)
total_muons_ge77 = sum(s.muons_with_ge77 for s in all_stats)

print(f"{'TOTAL':<12} {total_files:>6} {total_nc:>10} {total_ge77:>10} "
        f"{total_muons:>10} {total_muons_ge77:>11}")

if total_nc > 0:
    print(f"\nGe77 fraction: {total_ge77 / total_nc:.4f} "
            f"({total_ge77}/{total_nc})")
if total_muons > 0:
    print(f"Muons producing Ge77 capture: {total_muons_ge77 / total_muons:.4f} "
            f"({total_muons_ge77}/{total_muons})")

# --- Histogram: NCs per muon ---
if all_nc_counts:
    counts = np.concatenate(all_nc_counts)
    max_nc = int(counts.max())
    median_nc = np.median(counts)
    mean_nc = counts.mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = make_log_bins(int(counts.min()), int(counts.max()))
    ax.hist(counts, bins=bins, edgecolor="black", linewidth=0.5, color="#4C72B0")

    ax.set_xlabel("Number of neutron captures per muon", fontsize=13)
    ax.set_ylabel("Number of muons", fontsize=13)
    ax.set_title(
        f"NC multiplicity per muon (all runs, N = {len(counts)} muons)",
        fontsize=14,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(mean_nc, color="red", linestyle="--", linewidth=1.2,
                label=f"Mean = {mean_nc:.2f}")
    ax.axvline(median_nc, color="orange", linestyle="--", linewidth=1.2,
                label=f"Median = {median_nc:.0f}")
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)

    out_path = base / "nc_per_muon_histogram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHistogram saved to: {out_path}")

# --- Histogram: NCs per muon (Ge77-producing muons only) ---
if all_nc_counts_ge77:
    counts_ge77 = np.concatenate(all_nc_counts_ge77)
    max_nc = int(counts_ge77.max())
    median_nc = np.median(counts_ge77)
    mean_nc = counts_ge77.mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = make_log_bins(int(counts_ge77.min()), int(counts_ge77.max()))
    ax.hist(counts_ge77, bins=bins, edgecolor="black", linewidth=0.5,
            color="#C44E52")

    ax.set_xlabel("Number of neutron captures per muon", fontsize=13)
    ax.set_ylabel("Number of muons", fontsize=13)
    ax.set_title(
        f"NC multiplicity per Ge77-producing muon "
        f"(all runs, N = {len(counts_ge77)} muons)",
        fontsize=14,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(mean_nc, color="red", linestyle="--", linewidth=1.2,
                label=f"Mean = {mean_nc:.2f}")
    ax.axvline(median_nc, color="orange", linestyle="--", linewidth=1.2,
                label=f"Median = {median_nc:.0f}")
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)

    out_path = base / "nc_per_muon_ge77_histogram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogram (Ge77) saved to: {out_path}")

# --- Histogram: NC capture time (all muons), Ge77 NCs stacked in red ---
if all_time:
    t_all = np.concatenate(all_time)
    t_ge77f = np.concatenate(all_time_ge77flag) if all_time_ge77flag else np.array([])
    t_non_ge77 = t_all[~np.isin(np.arange(len(t_all)), [])]  # placeholder

    # Filter to positive times for log scale
    mask_pos = t_all > 0
    t_all_pos = t_all[mask_pos]

    if t_all_pos.size > 0:
        bins = make_log_bins(max(1, int(t_all_pos.min())), int(t_all_pos.max()))

        # Split into non-Ge77 and Ge77 for stacking
        # We need to re-collect with a flag; easier: histogram both and overlay
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(t_all_pos, bins=bins, edgecolor="black", linewidth=0.3,
                color="#4C72B0", label=f"All NCs (N={len(t_all_pos)})")

        if t_ge77f.size > 0:
            t_ge77f_pos = t_ge77f[t_ge77f > 0]
            if t_ge77f_pos.size > 0:
                ax.hist(t_ge77f_pos, bins=bins, edgecolor="black", linewidth=0.3,
                        color="#C44E52", label=f"Ge77 NCs (N={len(t_ge77f_pos)})")

        ax.set_xlabel("Neutron capture time [ns]", fontsize=13)
        ax.set_ylabel("Number of neutron captures", fontsize=13)
        ax.set_title("NC capture time — all muons", fontsize=14)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=11)
        ax.tick_params(labelsize=11)

        out_path = base / "nc_time_all_muons_histogram.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Time histogram (all muons) saved to: {out_path}")

# --- Histogram: NC capture time (Ge77-producing muons only), Ge77 NCs stacked in red ---
if all_time_ge77muons:
    t_gm = np.concatenate(all_time_ge77muons)
    t_gmf = np.concatenate(all_time_ge77muons_ge77flag) if all_time_ge77muons_ge77flag else np.array([])

    mask_pos = t_gm > 0
    t_gm_pos = t_gm[mask_pos]

    if t_gm_pos.size > 0:
        bins = make_log_bins(max(1, int(t_gm_pos.min())), int(t_gm_pos.max()))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(t_gm_pos, bins=bins, edgecolor="black", linewidth=0.3,
                color="#4C72B0", label=f"All NCs of Ge77 muons (N={len(t_gm_pos)})")

        if t_gmf.size > 0:
            t_gmf_pos = t_gmf[t_gmf > 0]
            if t_gmf_pos.size > 0:
                ax.hist(t_gmf_pos, bins=bins, edgecolor="black", linewidth=0.3,
                        color="#C44E52", label=f"Ge77 NCs (N={len(t_gmf_pos)})")

        ax.set_xlabel("Neutron capture time [ns]", fontsize=13)
        ax.set_ylabel("Number of neutron captures", fontsize=13)
        ax.set_title("NC capture time — Ge77-producing muons only", fontsize=14)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=11)
        ax.tick_params(labelsize=11)

        out_path = base / "nc_time_ge77_muons_histogram.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Time histogram (Ge77 muons) saved to: {out_path}")

# =============================================================================
# Muon property analysis: Ge77-producing vs non-Ge77 muons
#
# Plots:
# 1. Kinetic energy histogram (Ge77 vs non-Ge77, normalized)
# 2. Zenith angle histogram (Ge77 vs non-Ge77, normalized)
# 3. Azimuth angle histogram (Ge77 vs non-Ge77, normalized)
# 4. 3D scatter of Ge77-muon positions with momentum arrows + cylinder
#
# Append after imports of main analysis script.
# =============================================================================

VERTICES_GROUP = "/hit/vertices"
PARTICLES_GROUP = "/hit/particles"


def read_muon_data_file(
    filepath: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read vertices and particles data from a single HDF5 file.
    1:1 index mapping between vertices and particles.

    Returns:
        (evtid, x, y, z, ekin, px, py, pz) — all 1D arrays, same length.
    """
    empty_i = np.array([], dtype=np.int32)
    empty_f = np.array([], dtype=np.float64)
    empty = (empty_i, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f)
    with h5py.File(filepath, "r") as f:
        if VERTICES_GROUP not in f or PARTICLES_GROUP not in f:
            return empty

        vgrp = f[VERTICES_GROUP]
        pgrp = f[PARTICLES_GROUP]

        n_vert = int(vgrp["entries"][()])
        n_part = int(pgrp["entries"][()])
        if n_vert == 0 or n_part == 0:
            return empty

        assert n_vert == n_part, (
            f"vertices ({n_vert}) and particles ({n_part}) entry count mismatch in {filepath}"
        )

        evtid = vgrp["evtid"]["pages"][:]
        x = vgrp["xloc_in_m"]["pages"][:]
        y = vgrp["yloc_in_m"]["pages"][:]
        z = vgrp["zloc_in_m"]["pages"][:]
        ekin = pgrp["ekin_in_MeV"]["pages"][:]
        px = pgrp["px_in_MeV"]["pages"][:]
        py = pgrp["py_in_MeV"]["pages"][:]
        pz = pgrp["pz_in_MeV"]["pages"][:]

    return evtid, x, y, z, ekin, px, py, pz


def collect_muon_data(run_dirs: list[Path]) -> dict:
    """
    Collect all muon vertex/particle data across runs,
    and determine which evtids are Ge77-producing per run.

    Returns dict with arrays: x, y, z, ekin, px, py, pz, is_ge77
    """
    all_x, all_y, all_z = [], [], []
    all_ekin, all_px, all_py, all_pz = [], [], [], []
    all_is_ge77 = []

    for run_dir in run_dirs:
        hdf5_files = sorted(run_dir.glob("output_t*.hdf5"))
        if not hdf5_files:
            continue

        # --- Collect NC data for this run to find Ge77 muon evtids ---
        nc_evtids_list = []
        nc_ge77_list = []
        nc_trackid_list = []
        for fp in hdf5_files:
            try:
                with h5py.File(fp, "r") as f:
                    if NC_GROUP not in f:
                        continue
                    grp = f[NC_GROUP]
                    if int(grp["entries"][()]) == 0:
                        continue
                    nc_evtids_list.append(grp["evtid"]["pages"][:])
                    nc_trackid_list.append(grp["nC_track_id"]["pages"][:])
                    nc_ge77_list.append(grp["nC_flag_Ge77"]["pages"][:])
            except Exception:
                continue

        ge77_muon_evtids = set()
        if nc_evtids_list:
            nc_evtid = np.concatenate(nc_evtids_list)
            nc_tid = np.concatenate(nc_trackid_list)
            nc_ge77 = np.concatenate(nc_ge77_list)

            # Deduplicate NCs on (evtid, nc_track_id)
            pair_keys = np.stack([nc_evtid, nc_tid], axis=1)
            _, unique_idx, inverse = np.unique(
                pair_keys, axis=0, return_index=True, return_inverse=True
            )
            unique_ge77 = np.zeros(len(unique_idx), dtype=np.int32)
            np.maximum.at(unique_ge77, inverse, nc_ge77)
            unique_nc_evtid = nc_evtid[unique_idx]

            ge77_mask = unique_ge77 == 1
            ge77_muon_evtids = set(unique_nc_evtid[ge77_mask].tolist())

        # --- Collect vertex/particle data for this run ---
        run_evtid_set = set()
        for fp in hdf5_files:
            try:
                evtid, x, y, z, ekin, px, py, pz = read_muon_data_file(fp)
            except Exception as e:
                print(f"  ERROR reading muon data from {fp.name}: {e}")
                continue

            if evtid.size == 0:
                continue

            # Deduplicate: keep only first occurrence per evtid within run
            new_mask = np.array([eid not in run_evtid_set for eid in evtid])
            run_evtid_set.update(evtid[new_mask].tolist())

            evtid = evtid[new_mask]
            x, y, z = x[new_mask], y[new_mask], z[new_mask]
            ekin = ekin[new_mask]
            px, py, pz = px[new_mask], py[new_mask], pz[new_mask]

            is_ge77 = np.array([eid in ge77_muon_evtids for eid in evtid], dtype=bool)

            all_x.append(x)
            all_y.append(y)
            all_z.append(z)
            all_ekin.append(ekin)
            all_px.append(px)
            all_py.append(py)
            all_pz.append(pz)
            all_is_ge77.append(is_ge77)

    return {
        "x": np.concatenate(all_x),
        "y": np.concatenate(all_y),
        "z": np.concatenate(all_z),
        "ekin": np.concatenate(all_ekin),
        "px": np.concatenate(all_px),
        "py": np.concatenate(all_py),
        "pz": np.concatenate(all_pz),
        "is_ge77": np.concatenate(all_is_ge77),
    }


def plot_muon_properties(data: dict, base: Path) -> None:
    """Create all muon property comparison plots."""
    is_ge77 = data["is_ge77"]
    not_ge77 = ~is_ge77

    ekin = data["ekin"]
    px, py, pz = data["px"], data["py"], data["pz"]
    p_mag = np.sqrt(px**2 + py**2 + pz**2)

    # Zenith angle: theta = arccos(pz / |p|)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = np.where(p_mag > 0, pz / p_mag, 0.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))

    # Azimuth angle: phi = atan2(py, px)
    phi_deg = np.degrees(np.arctan2(py, px))

    n_ge77 = int(is_ge77.sum())
    n_other = int(not_ge77.sum())
    print(f"\nMuon properties: {n_ge77} Ge77-producing, {n_other} non-Ge77")

    # --- 1. Kinetic energy histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ekin_ge77 = ekin[is_ge77 & (ekin > 0)]
    ekin_other = ekin[not_ge77 & (ekin > 0)]

    if ekin_ge77.size > 0 and ekin_other.size > 0:
        vmin = max(1, int(min(ekin_ge77.min(), ekin_other.min())))
        vmax = int(max(ekin_ge77.max(), ekin_other.max()))
        bins = make_log_bins(vmin, vmax)

        w_other = np.ones(len(ekin_other)) / len(ekin_other)
        w_ge77 = np.ones(len(ekin_ge77)) / len(ekin_ge77)
        ax.hist(ekin_other, bins=bins, weights=w_other, edgecolor="black",
                linewidth=0.3, color="#4C72B0", alpha=0.7,
                label=f"Non-Ge77 (N={len(ekin_other)})")
        ax.hist(ekin_ge77, bins=bins, weights=w_ge77, edgecolor="black",
                linewidth=0.3, color="#C44E52", alpha=0.7,
                label=f"Ge77 (N={len(ekin_ge77)})")

        ax.set_xlabel("Muon kinetic energy [MeV]", fontsize=13)
        ax.set_ylabel("Fraction of muons", fontsize=13)
        ax.set_title("Muon kinetic energy: Ge77 vs non-Ge77", fontsize=14)
        ax.set_xscale("log")
        ax.legend(fontsize=11)
        ax.tick_params(labelsize=11)

        out_path = base / "muon_energy_ge77_comparison.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Energy histogram saved to: {out_path}")
    plt.close(fig)

    # --- 2. Zenith angle histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    theta_ge77 = theta_deg[is_ge77]
    theta_other = theta_deg[not_ge77]

    bins_angle = np.linspace(0, 180, 91)
    w_other = np.ones(len(theta_other)) / len(theta_other)
    w_ge77 = np.ones(len(theta_ge77)) / len(theta_ge77)
    ax.hist(theta_other, bins=bins_angle, weights=w_other, edgecolor="black",
            linewidth=0.3, color="#4C72B0", alpha=0.7,
            label=f"Non-Ge77 (N={len(theta_other)})")
    ax.hist(theta_ge77, bins=bins_angle, weights=w_ge77, edgecolor="black",
            linewidth=0.3, color="#C44E52", alpha=0.7,
            label=f"Ge77 (N={len(theta_ge77)})")

    ax.set_xlabel("Zenith angle θ [deg]", fontsize=13)
    ax.set_ylabel("Fraction of muons", fontsize=13)
    ax.set_title("Muon zenith angle: Ge77 vs non-Ge77", fontsize=14)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)

    out_path = base / "muon_zenith_ge77_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Zenith histogram saved to: {out_path}")

    # --- 3. Azimuth angle histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    phi_ge77 = phi_deg[is_ge77]
    phi_other = phi_deg[not_ge77]

    bins_phi = np.linspace(-180, 180, 91)
    w_other = np.ones(len(phi_other)) / len(phi_other)
    w_ge77 = np.ones(len(phi_ge77)) / len(phi_ge77)
    ax.hist(phi_other, bins=bins_phi, weights=w_other, edgecolor="black",
            linewidth=0.3, color="#4C72B0", alpha=0.7,
            label=f"Non-Ge77 (N={len(phi_other)})")
    ax.hist(phi_ge77, bins=bins_phi, weights=w_ge77, edgecolor="black",
            linewidth=0.3, color="#C44E52", alpha=0.7,
            label=f"Ge77 (N={len(phi_ge77)})")

    ax.set_xlabel("Azimuth angle φ [deg]", fontsize=13)
    ax.set_ylabel("Fraction of muons", fontsize=13)
    ax.set_title("Muon azimuth angle: Ge77 vs non-Ge77", fontsize=14)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)

    out_path = base / "muon_azimuth_ge77_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Azimuth histogram saved to: {out_path}")

    # --- 4. 3D scatter of Ge77 muon positions with momentum arrows ---
    fig = plt.figure(figsize=(12, 10))
    ax3d = fig.add_subplot(111, projection="3d")

    # Ge77 muon positions in mm
    x_ge77 = data["x"][is_ge77] * 1000
    y_ge77 = data["y"][is_ge77] * 1000
    z_ge77 = data["z"][is_ge77] * 1000
    px_ge77 = px[is_ge77]
    py_ge77 = py[is_ge77]
    pz_ge77 = pz[is_ge77]

    # Normalize momentum for arrow direction
    p_mag_ge77 = np.sqrt(px_ge77**2 + py_ge77**2 + pz_ge77**2)
    p_mag_ge77 = np.where(p_mag_ge77 > 0, p_mag_ge77, 1.0)
    arrow_scale = 500.0  # mm
    dx = px_ge77 / p_mag_ge77 * arrow_scale
    dy = py_ge77 / p_mag_ge77 * arrow_scale
    dz = pz_ge77 / p_mag_ge77 * arrow_scale

    ax3d.scatter(x_ge77, y_ge77, z_ge77, c="red", s=10, alpha=0.6,
                 label=f"Ge77 muon vertex (N={n_ge77})")

    for i in range(len(x_ge77)):
        ax3d.plot(
            [x_ge77[i], x_ge77[i] + dx[i]],
            [y_ge77[i], y_ge77[i] + dy[i]],
            [z_ge77[i], z_ge77[i] + dz[i]],
            color="red", alpha=0.3, linewidth=0.5,
        )

    # Draw cylinder: radius=4300mm, z from -5000 to 3900 mm
    cyl_r = 4300.0
    cyl_z_min, cyl_z_max = -5000.0, 3900.0
    theta_cyl = np.linspace(0, 2 * np.pi, 80)
    z_cyl_arr = np.array([cyl_z_min, cyl_z_max])
    theta_grid, z_grid = np.meshgrid(theta_cyl, z_cyl_arr)
    x_cyl = cyl_r * np.cos(theta_grid)
    y_cyl = cyl_r * np.sin(theta_grid)

    ax3d.plot_surface(x_cyl, y_cyl, z_grid, alpha=0.08, color="gray")

    # Top and bottom circles
    circle_theta = np.linspace(0, 2 * np.pi, 100)
    cx = cyl_r * np.cos(circle_theta)
    cy = cyl_r * np.sin(circle_theta)
    ax3d.plot(cx, cy, cyl_z_min, color="gray", linewidth=0.8, alpha=0.5)
    ax3d.plot(cx, cy, cyl_z_max, color="gray", linewidth=0.8, alpha=0.5)

    ax3d.set_xlabel("X [mm]", fontsize=11)
    ax3d.set_ylabel("Y [mm]", fontsize=11)
    ax3d.set_zlabel("Z [mm]", fontsize=11)
    ax3d.set_title(f"Ge77-producing muon vertices (N={n_ge77})", fontsize=14)
    ax3d.legend(fontsize=10)

    out_path = base / "muon_ge77_3d_positions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"3D plot saved to: {out_path}")


# --- Run ---
base = Path("/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/rawMusunNCsSSD")
run_dirs = sorted(base.glob("run_0*"))

print(f"Collecting muon vertex/particle data from {len(run_dirs)} runs...")
muon_data = collect_muon_data(run_dirs)
plot_muon_properties(muon_data, base)

# =============================================================================
# Muon property analysis: NC-producing vs non-NC-producing muons
#
# Plots:
# 1. Kinetic energy histogram (NC vs no-NC, normalized)
# 2. Zenith angle histogram (NC vs no-NC, normalized)
# 3. Azimuth angle histogram (NC vs no-NC, normalized)
# 4. 3D scatter of NC-producing muon positions with momentum arrows + cylinder
#
# Append after imports of main analysis script.
# Uses: NC_GROUP, make_log_bins, h5py, np, plt, Path
# =============================================================================

VERTICES_GROUP = "/hit/vertices"
PARTICLES_GROUP = "/hit/particles"


def read_muon_data_file_nc(
    filepath: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read vertices and particles data from a single HDF5 file.
    1:1 index mapping between vertices and particles.
    """
    empty_i = np.array([], dtype=np.int32)
    empty_f = np.array([], dtype=np.float64)
    empty = (empty_i, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f)
    with h5py.File(filepath, "r") as f:
        if VERTICES_GROUP not in f or PARTICLES_GROUP not in f:
            return empty

        vgrp = f[VERTICES_GROUP]
        pgrp = f[PARTICLES_GROUP]

        n_vert = int(vgrp["entries"][()])
        n_part = int(pgrp["entries"][()])
        if n_vert == 0 or n_part == 0:
            return empty

        assert n_vert == n_part, (
            f"vertices ({n_vert}) and particles ({n_part}) mismatch in {filepath}"
        )

        evtid = vgrp["evtid"]["pages"][:]
        x = vgrp["xloc_in_m"]["pages"][:]
        y = vgrp["yloc_in_m"]["pages"][:]
        z = vgrp["zloc_in_m"]["pages"][:]
        ekin = pgrp["ekin_in_MeV"]["pages"][:]
        px = pgrp["px_in_MeV"]["pages"][:]
        py = pgrp["py_in_MeV"]["pages"][:]
        pz = pgrp["pz_in_MeV"]["pages"][:]

    return evtid, x, y, z, ekin, px, py, pz


def collect_muon_data_nc(run_dirs: list[Path]) -> dict:
    """
    Collect all muon vertex/particle data across runs,
    and determine which evtids produce any NC (regardless of Ge77).
    """
    all_x, all_y, all_z = [], [], []
    all_ekin, all_px, all_py, all_pz = [], [], [], []
    all_has_nc = []

    for run_dir in run_dirs:
        hdf5_files = sorted(run_dir.glob("output_t*.hdf5"))
        if not hdf5_files:
            continue

        # --- Collect NC evtids for this run (any NC, deduplicated) ---
        nc_evtids_list = []
        nc_trackid_list = []
        for fp in hdf5_files:
            try:
                with h5py.File(fp, "r") as f:
                    if NC_GROUP not in f:
                        continue
                    grp = f[NC_GROUP]
                    if int(grp["entries"][()]) == 0:
                        continue
                    nc_evtids_list.append(grp["evtid"]["pages"][:])
                    nc_trackid_list.append(grp["nC_track_id"]["pages"][:])
            except Exception:
                continue

        nc_muon_evtids = set()
        if nc_evtids_list:
            nc_evtid = np.concatenate(nc_evtids_list)
            nc_tid = np.concatenate(nc_trackid_list)

            # Deduplicate on (evtid, nc_track_id)
            pair_keys = np.stack([nc_evtid, nc_tid], axis=1)
            unique_pairs = np.unique(pair_keys, axis=0)
            nc_muon_evtids = set(unique_pairs[:, 0].tolist())

        # --- Collect vertex/particle data for this run ---
        run_evtid_set = set()
        for fp in hdf5_files:
            try:
                evtid, x, y, z, ekin, px, py, pz = read_muon_data_file_nc(fp)
            except Exception as e:
                print(f"  ERROR reading muon data from {fp.name}: {e}")
                continue

            if evtid.size == 0:
                continue

            # Deduplicate: first occurrence per evtid within run
            new_mask = np.array([eid not in run_evtid_set for eid in evtid])
            run_evtid_set.update(evtid[new_mask].tolist())

            evtid = evtid[new_mask]
            x, y, z = x[new_mask], y[new_mask], z[new_mask]
            ekin = ekin[new_mask]
            px, py, pz = px[new_mask], py[new_mask], pz[new_mask]

            has_nc = np.array([eid in nc_muon_evtids for eid in evtid], dtype=bool)

            all_x.append(x)
            all_y.append(y)
            all_z.append(z)
            all_ekin.append(ekin)
            all_px.append(px)
            all_py.append(py)
            all_pz.append(pz)
            all_has_nc.append(has_nc)

    return {
        "x": np.concatenate(all_x),
        "y": np.concatenate(all_y),
        "z": np.concatenate(all_z),
        "ekin": np.concatenate(all_ekin),
        "px": np.concatenate(all_px),
        "py": np.concatenate(all_py),
        "pz": np.concatenate(all_pz),
        "has_nc": np.concatenate(all_has_nc),
    }


def plot_muon_properties_nc(data: dict, base: Path) -> None:
    """Create muon property comparison plots: NC-producing vs no-NC."""
    has_nc = data["has_nc"]
    no_nc = ~has_nc

    ekin = data["ekin"]
    px, py, pz = data["px"], data["py"], data["pz"]
    p_mag = np.sqrt(px**2 + py**2 + pz**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = np.where(p_mag > 0, pz / p_mag, 0.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))

    phi_deg = np.degrees(np.arctan2(py, px))

    n_nc = int(has_nc.sum())
    n_no_nc = int(no_nc.sum())
    print(f"\nMuon properties: {n_nc} NC-producing, {n_no_nc} non-NC")

    # --- 1. Kinetic energy histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ekin_nc = ekin[has_nc & (ekin > 0)]
    ekin_no = ekin[no_nc & (ekin > 0)]

    if ekin_nc.size > 0 and ekin_no.size > 0:
        vmin = max(1, int(min(ekin_nc.min(), ekin_no.min())))
        vmax = int(max(ekin_nc.max(), ekin_no.max()))
        bins = make_log_bins(vmin, vmax)

        w_no = np.ones(len(ekin_no)) / len(ekin_no)
        w_nc = np.ones(len(ekin_nc)) / len(ekin_nc)
        ax.hist(ekin_no, bins=bins, weights=w_no, edgecolor="black",
                linewidth=0.3, color="#4C72B0", alpha=0.7,
                label=f"No NC (N={len(ekin_no)})")
        ax.hist(ekin_nc, bins=bins, weights=w_nc, edgecolor="black",
                linewidth=0.3, color="#C44E52", alpha=0.7,
                label=f"NC-producing (N={len(ekin_nc)})")

        ax.set_xlabel("Muon kinetic energy [MeV]", fontsize=13)
        ax.set_ylabel("Fraction of muons", fontsize=13)
        ax.set_title("Muon kinetic energy: NC-producing vs no-NC", fontsize=14)
        ax.set_xscale("log")
        ax.legend(fontsize=11)
        ax.tick_params(labelsize=11)

        out_path = base / "muon_energy_nc_comparison.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Energy histogram saved to: {out_path}")
    plt.close(fig)

    # --- 2. Zenith angle histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    theta_nc = theta_deg[has_nc]
    theta_no = theta_deg[no_nc]

    bins_angle = np.linspace(0, 180, 91)
    w_no = np.ones(len(theta_no)) / len(theta_no)
    w_nc = np.ones(len(theta_nc)) / len(theta_nc)
    ax.hist(theta_no, bins=bins_angle, weights=w_no, edgecolor="black",
            linewidth=0.3, color="#4C72B0", alpha=0.7,
            label=f"No NC (N={len(theta_no)})")
    ax.hist(theta_nc, bins=bins_angle, weights=w_nc, edgecolor="black",
            linewidth=0.3, color="#C44E52", alpha=0.7,
            label=f"NC-producing (N={len(theta_nc)})")

    ax.set_xlabel("Zenith angle θ [deg]", fontsize=13)
    ax.set_ylabel("Fraction of muons", fontsize=13)
    ax.set_title("Muon zenith angle: NC-producing vs no-NC", fontsize=14)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)

    out_path = base / "muon_zenith_nc_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Zenith histogram saved to: {out_path}")

    # --- 3. Azimuth angle histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    phi_nc = phi_deg[has_nc]
    phi_no = phi_deg[no_nc]

    bins_phi = np.linspace(-180, 180, 91)
    w_no = np.ones(len(phi_no)) / len(phi_no)
    w_nc = np.ones(len(phi_nc)) / len(phi_nc)
    ax.hist(phi_no, bins=bins_phi, weights=w_no, edgecolor="black",
            linewidth=0.3, color="#4C72B0", alpha=0.7,
            label=f"No NC (N={len(phi_no)})")
    ax.hist(phi_nc, bins=bins_phi, weights=w_nc, edgecolor="black",
            linewidth=0.3, color="#C44E52", alpha=0.7,
            label=f"NC-producing (N={len(phi_nc)})")

    ax.set_xlabel("Azimuth angle φ [deg]", fontsize=13)
    ax.set_ylabel("Fraction of muons", fontsize=13)
    ax.set_title("Muon azimuth angle: NC-producing vs no-NC", fontsize=14)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)

    out_path = base / "muon_azimuth_nc_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Azimuth histogram saved to: {out_path}")

    # --- 4. 3D scatter of NC-producing muon positions with momentum arrows ---
    fig = plt.figure(figsize=(12, 10))
    ax3d = fig.add_subplot(111, projection="3d")

    x_nc = data["x"][has_nc] * 1000
    y_nc = data["y"][has_nc] * 1000
    z_nc = data["z"][has_nc] * 1000
    px_nc = px[has_nc]
    py_nc = py[has_nc]
    pz_nc = pz[has_nc]

    p_mag_nc = np.sqrt(px_nc**2 + py_nc**2 + pz_nc**2)
    p_mag_nc = np.where(p_mag_nc > 0, p_mag_nc, 1.0)
    arrow_scale = 500.0
    dx = px_nc / p_mag_nc * arrow_scale
    dy = py_nc / p_mag_nc * arrow_scale
    dz = pz_nc / p_mag_nc * arrow_scale

    ax3d.scatter(x_nc, y_nc, z_nc, c="red", s=10, alpha=0.6,
                 label=f"NC-producing muon vertex (N={n_nc})")

    for i in range(len(x_nc)):
        ax3d.plot(
            [x_nc[i], x_nc[i] + dx[i]],
            [y_nc[i], y_nc[i] + dy[i]],
            [z_nc[i], z_nc[i] + dz[i]],
            color="red", alpha=0.3, linewidth=0.5,
        )

    # Cylinder: radius=4300mm, z from -5000 to 3900 mm
    cyl_r = 4300.0
    cyl_z_min, cyl_z_max = -5000.0, 3900.0
    theta_cyl = np.linspace(0, 2 * np.pi, 80)
    z_cyl_arr = np.array([cyl_z_min, cyl_z_max])
    theta_grid, z_grid = np.meshgrid(theta_cyl, z_cyl_arr)
    x_cyl = cyl_r * np.cos(theta_grid)
    y_cyl = cyl_r * np.sin(theta_grid)

    ax3d.plot_surface(x_cyl, y_cyl, z_grid, alpha=0.08, color="gray")

    circle_theta = np.linspace(0, 2 * np.pi, 100)
    cx = cyl_r * np.cos(circle_theta)
    cy = cyl_r * np.sin(circle_theta)
    ax3d.plot(cx, cy, cyl_z_min, color="gray", linewidth=0.8, alpha=0.5)
    ax3d.plot(cx, cy, cyl_z_max, color="gray", linewidth=0.8, alpha=0.5)

    ax3d.set_xlabel("X [mm]", fontsize=11)
    ax3d.set_ylabel("Y [mm]", fontsize=11)
    ax3d.set_zlabel("Z [mm]", fontsize=11)
    ax3d.set_title(f"NC-producing muon vertices (N={n_nc})", fontsize=14)
    ax3d.legend(fontsize=10)

    out_path = base / "muon_nc_3d_positions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"3D plot saved to: {out_path}")


# --- Run ---
base = Path("/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/rawMusunNCsSSD")
run_dirs = sorted(base.glob("run_0*"))

print(f"Collecting muon vertex/particle data from {len(run_dirs)} runs...")
muon_data_nc = collect_muon_data_nc(run_dirs)
plot_muon_properties_nc(muon_data_nc, base)

