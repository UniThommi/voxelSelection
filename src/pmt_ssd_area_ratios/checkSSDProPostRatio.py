#!/usr/bin/env python3
"""
SSD Raw vs. SSD Postprocessed Efficiency Comparison
====================================================
Verifies that the postprocessing pipeline does not lose optical photons.

Comparisons:
  A) Global: Total hits (raw vs postprocessed), overall
  C) Per-NC Event Match: 5 matched events, total hit comparison

Note: Per-voxel comparison is not possible because raw SSD has only
4 detector UIDs (1965-1968), while postprocessed distributes hits
across ~9500 voxels. The mapping is one-to-many.

Author: Thomas Buerger (University of Tübingen)
"""

import gc
import glob
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import psutil


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
SSD_RAW_DIR = (
    "/pscratch/sd/t/tbuerger/data/proofeDetectionEfficiencies/"
    "experimentEfficiencyRatio/PMTs_div_SSD_1/"
    "ratio_0.588_SSD/run_20260209_211629"
)
SSD_POSTPROCESSED_DIR = (
    "/pscratch/sd/t/tbuerger/data/proofeDetectionEfficiencies/"
    "experimentEfficiencyRatio/PMTs_div_SSD_1/"
    "ratio_0.588_SSD_postprocessed"
)

N_EVENT_MATCHES = 5

OUTPUT_DIR = Path.cwd() / "ssd_raw_vs_postprocessed_plots"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Geometry (same as detectionRatio.py)
# ─────────────────────────────────────────────────────────────
@dataclass
class GeometryConfig:
    """Detector geometry parameters."""
    h: float = 8900
    r_zylinder: float = 4300
    z_origin: float = 20

    @property
    def h_zylinder(self) -> float:
        return self.h + 20 - self.z_origin

    @property
    def z_cut_bot(self) -> float:
        return -4979

    @property
    def z_cut_top(self) -> float:
        return self.z_cut_bot + self.h_zylinder - 2


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def get_chunk_size() -> int:
    """Determine optimal chunk size based on available memory."""
    available_gb = psutil.virtual_memory().available / (1024**3)
    if available_gb > 400:
        return 50_000
    elif available_gb > 200:
        return 30_000
    elif available_gb > 50:
        return 20_000
    elif available_gb > 30:
        return 15_000
    elif available_gb > 20:
        return 10_000
    return 5_000


def check_radial_momentum(
    x: np.ndarray, y: np.ndarray,
    px: np.ndarray, py: np.ndarray,
) -> np.ndarray:
    """Check if momentum points radially outward (dot product <= 0)."""
    return (x * px + y * py) >= 0


def load_nc_data_dict(f: h5py.File) -> Dict[Tuple[int, int], float]:
    """Load NC event data: (evtid, nC_track_id) -> nC_time."""
    nc_out = f["hit"]["MyNeutronCaptureOutput"]
    evtids = nc_out["evtid"]["pages"][:]
    nc_ids = nc_out["nC_track_id"]["pages"][:]
    nc_times = nc_out["nC_time_in_ns"]["pages"][:]

    return {
        (int(evtids[i]), int(nc_ids[i])): float(nc_times[i])
        for i in range(len(evtids))
    }


# ─────────────────────────────────────────────────────────────
# A) SSD Raw: Total filtered hits
# ─────────────────────────────────────────────────────────────
def process_ssd_raw(
    raw_dir: str, geometry: GeometryConfig, chunk_size: int
) -> Tuple[int, int, Dict[int, int]]:
    """Process SSD raw data with time + momentum filters.

    Returns:
        (total_hits, total_nc_events, hits_per_det_uid)
    """
    print("=" * 60)
    print("Processing SSD raw data...")
    print(f"  Directory: {raw_dir}")

    hdf5_files = sorted(glob.glob(os.path.join(raw_dir, "output_t*.hdf5")))
    if not hdf5_files:
        raise FileNotFoundError(f"No output_t*.hdf5 in {raw_dir}")
    print(f"  Found {len(hdf5_files)} files")

    total_hits = 0
    total_nc_events = 0
    hits_per_uid: Dict[int, int] = defaultdict(int)
    z_cut_bot = geometry.z_cut_bot
    z_cut_top = geometry.z_cut_top

    for file_idx, fpath in enumerate(hdf5_files):
        if (file_idx + 1) % 50 == 0:
            print(f"  File {file_idx + 1}/{len(hdf5_files)}")

        with h5py.File(fpath, "r") as f:
            nc_data_dict = load_nc_data_dict(f)
            total_nc_events += len(nc_data_dict)

            optical = f["hit"]["optical"]
            total_photons = len(optical["x_position_in_m"]["pages"])
            if total_photons == 0:
                continue

            num_chunks = (total_photons - 1) // chunk_size + 1

            for ci in range(num_chunks):
                s = ci * chunk_size
                e = min(s + chunk_size, total_photons)

                x = np.array(optical["x_position_in_m"]["pages"][s:e], dtype=np.float32) * 1000
                y = np.array(optical["y_position_in_m"]["pages"][s:e], dtype=np.float32) * 1000
                z = np.array(optical["z_position_in_m"]["pages"][s:e], dtype=np.float32) * 1000
                px = np.array(optical["x_momentum_direction"]["pages"][s:e], dtype=np.float32)
                py = np.array(optical["y_momentum_direction"]["pages"][s:e], dtype=np.float32)
                pz = np.array(optical["z_momentum_direction"]["pages"][s:e], dtype=np.float32)

                evtids = optical["evtid"]["pages"][s:e]
                nc_ids = optical["nC_track_id"]["pages"][s:e]
                times = optical["time_in_ns"]["pages"][s:e]
                det_uids = optical["det_uid"]["pages"][s:e]

                # Filter 1: Time window (0-200 ns after NC)
                nc_times_arr = np.full(len(times), np.inf, dtype=np.float32)
                for idx in range(len(evtids)):
                    key = (int(evtids[idx]), int(nc_ids[idx]))
                    if key in nc_data_dict:
                        nc_times_arr[idx] = nc_data_dict[key]

                valid_nc = nc_times_arr != np.inf
                time_mask = (
                    valid_nc
                    & (times >= nc_times_arr)
                    & (times <= nc_times_arr + 200.0)
                )

                x = x[time_mask]
                y = y[time_mask]
                z = z[time_mask]
                px = px[time_mask]
                py = py[time_mask]
                pz = pz[time_mask]
                det_uids_filtered = det_uids[time_mask]

                if len(x) == 0:
                    continue

                # Filter 2: Momentum direction
                mask_bot = z <= z_cut_bot
                mask_top = z >= z_cut_top
                mask_barrel = ~mask_bot & ~mask_top

                final_mask = np.zeros(len(z), dtype=bool)
                final_mask[mask_bot] = pz[mask_bot] <= 0
                final_mask[mask_top] = pz[mask_top] >= 0
                if np.any(mask_barrel):
                    final_mask[mask_barrel] = check_radial_momentum(
                        x[mask_barrel], y[mask_barrel],
                        px[mask_barrel], py[mask_barrel],
                    )

                det_uids_final = det_uids_filtered[final_mask]
                total_hits += len(det_uids_final)

                uid_vals, counts = np.unique(det_uids_final, return_counts=True)
                for uid, count in zip(uid_vals, counts):
                    hits_per_uid[int(uid)] += int(count)

                del x, y, z, px, py, pz, evtids, nc_ids, times, det_uids
                del nc_times_arr, valid_nc, time_mask, det_uids_filtered
                del mask_bot, mask_top, mask_barrel, final_mask, det_uids_final
                gc.collect()

    print(f"  Total NC events: {total_nc_events:,}")
    print(f"  Unique det_uids: {sorted(hits_per_uid.keys())}")
    print(f"  Total filtered hits: {total_hits:,}")
    for uid, count in sorted(hits_per_uid.items()):
        print(f"    det_uid {uid}: {count:,} hits")

    return total_hits, total_nc_events, dict(hits_per_uid)


# ─────────────────────────────────────────────────────────────
# A) SSD Postprocessed: Total hits
# ─────────────────────────────────────────────────────────────
def process_ssd_postprocessed(pp_dir: str) -> Tuple[int, int]:
    """Load total hits from all postprocessed HDF5 files.

    Returns:
        (total_hits, total_nc_events)
    """
    print("=" * 60)
    print("Loading SSD postprocessed data...")
    print(f"  Directory: {pp_dir}")

    hdf5_files = sorted(glob.glob(os.path.join(pp_dir, "*.hdf5")))
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files in {pp_dir}")
    print(f"  Found {len(hdf5_files)} files: {[os.path.basename(f) for f in hdf5_files]}")

    total_hits = 0
    total_nc_events = 0

    for fpath in hdf5_files:
        fname = os.path.basename(fpath)
        with h5py.File(fpath, "r") as f:
            target = f["target"]
            voxel_keys = list(target.keys())
            n_ncs = target[voxel_keys[0]].shape[0]
            total_nc_events += n_ncs

            file_hits = 0
            for voxel_id in voxel_keys:
                file_hits += int(np.sum(target[voxel_id][:]))
            total_hits += file_hits

            print(f"  {fname}: {n_ncs:,} NCs, {len(voxel_keys)} voxels, {file_hits:,} hits")

    print(f"  Total NC events: {total_nc_events:,}")
    print(f"  Total postprocessed hits: {total_hits:,}")

    return total_hits, total_nc_events


# ─────────────────────────────────────────────────────────────
# C) Per-NC event matching: sample from postprocessed, find in raw
# ─────────────────────────────────────────────────────────────
def sample_nc_from_postprocessed(
    pp_dir: str, n_samples: int = 5
) -> List[Dict]:
    """Sample N random NC events from postprocessed files.

    For each sampled NC, extracts properties from phi/ and total hits
    summed over all voxels. Memory-efficient: loads one NC at a time.
    """
    print("=" * 60)
    print(f"Sampling {n_samples} NC events from postprocessed data...")

    hdf5_files = sorted(glob.glob(os.path.join(pp_dir, "*.hdf5")))

    # Count total NCs per file
    file_nc_counts = []
    for fpath in hdf5_files:
        with h5py.File(fpath, "r") as f:
            first_key = next(iter(f["target"].keys()))
            n = f["target"][first_key].shape[0]
            file_nc_counts.append((fpath, n))

    total_ncs = sum(n for _, n in file_nc_counts)
    print(f"  Total NCs across files: {total_ncs:,}")

    # Sample global indices
    rng = np.random.default_rng(seed=42)
    global_indices = sorted(
        rng.choice(total_ncs, size=min(n_samples, total_ncs), replace=False)
    )
    print(f"  Sampled global indices: {global_indices}")

    # Map global index -> (file, local_index)
    samples = []
    offset = 0
    sample_ptr = 0

    for fpath, n_ncs in file_nc_counts:
        if sample_ptr >= len(global_indices):
            break

        while sample_ptr < len(global_indices) and global_indices[sample_ptr] < offset + n_ncs:
            local_idx = global_indices[sample_ptr] - offset

            with h5py.File(fpath, "r") as f:
                phi = f["phi"]
                props = {
                    "pp_file": fpath,
                    "pp_index": int(local_idx),
                    "x_mm": float(phi["xNC_mm"][local_idx]),
                    "y_mm": float(phi["yNC_mm"][local_idx]),
                    "z_mm": float(phi["zNC_mm"][local_idx]),
                    "ngamma": int(phi["#gamma"][local_idx]),
                    "etot_keV": float(phi["E_gamma_tot_keV"][local_idx]),
                    "e1_keV": float(phi["gammaE1_keV"][local_idx]),
                    "e2_keV": float(phi["gammaE2_keV"][local_idx]),
                    "e3_keV": float(phi["gammaE3_keV"][local_idx]),
                    "e4_keV": float(phi["gammaE4_keV"][local_idx]),
                }

                # Sum hits across all voxels for this single NC index
                pp_total = 0
                for voxel_id in f["target"].keys():
                    pp_total += int(f["target"][voxel_id][local_idx])
                props["pp_total_hits"] = pp_total

            samples.append(props)
            sample_ptr += 1

        offset += n_ncs

    for i, s in enumerate(samples):
        print(f"  NC {i+1}: pos=({s['x_mm']:.1f}, {s['y_mm']:.1f}, {s['z_mm']:.1f}) mm, "
              f"ngamma={s['ngamma']}, pp_hits={s['pp_total_hits']}")

    return samples


def find_nc_in_raw_and_count_hits(
    raw_dir: str,
    nc_sample: Dict,
    geometry: GeometryConfig,
) -> Optional[int]:
    """Find a specific NC event in raw data and return total filtered hits.

    Matches on NC position (mm->m) and gamma properties.
    Applies full filter pipeline (time + momentum).

    Returns total filtered hits, or None if NC not found.
    """
    target_x_m = nc_sample["x_mm"] #/ 1000.0
    target_y_m = nc_sample["y_mm"] #/ 1000.0
    target_z_m = nc_sample["z_mm"] #/ 1000.0
    target_ngamma = nc_sample["ngamma"]
    target_etot = nc_sample["etot_keV"]
    target_e1 = nc_sample["e1_keV"]
    target_e2 = nc_sample["e2_keV"]
    target_e3 = nc_sample["e3_keV"]
    target_e4 = nc_sample["e4_keV"]

    z_cut_bot = geometry.z_cut_bot
    z_cut_top = geometry.z_cut_top

    hdf5_files = sorted(glob.glob(os.path.join(raw_dir, "output_t*.hdf5")))

    for fpath in hdf5_files:
        with h5py.File(fpath, "r") as f:
            nc_out = f["hit"]["MyNeutronCaptureOutput"]
            n_nc = len(nc_out["evtid"]["pages"])
            if n_nc == 0:
                continue

            # Load NC metadata (small arrays, no OOM risk)
            nc_evtids = nc_out["evtid"]["pages"][:]
            nc_ids = nc_out["nC_track_id"]["pages"][:]
            nc_x = nc_out["nC_x_position_in_m"]["pages"][:]
            nc_y = nc_out["nC_y_position_in_m"]["pages"][:]
            nc_z = nc_out["nC_z_position_in_m"]["pages"][:]
            nc_ngamma = nc_out["nC_gamma_amount"]["pages"][:]
            nc_etot_arr = nc_out["nC_gamma_total_energy_in_keV"]["pages"][:]
            nc_e1_arr = nc_out["gamma1_E_in_keV"]["pages"][:]
            nc_e2_arr = nc_out["gamma2_E_in_keV"]["pages"][:]
            nc_e3_arr = nc_out["gamma3_E_in_keV"]["pages"][:]
            nc_e4_arr = nc_out["gamma4_E_in_keV"]["pages"][:]

            # Match with tolerance for float roundtrip
            tol = 1e-3
            match_mask = (
                (np.abs(nc_x - target_x_m) < tol)
                & (np.abs(nc_y - target_y_m) < tol)
                & (np.abs(nc_z - target_z_m) < tol)
                & (nc_ngamma == target_ngamma)
                & (np.abs(nc_etot_arr - target_etot) < 0.1)
                & (np.abs(nc_e1_arr - target_e1) < 0.1)
                & (np.abs(nc_e2_arr - target_e2) < 0.1)
                & (np.abs(nc_e3_arr - target_e3) < 0.1)
                & (np.abs(nc_e4_arr - target_e4) < 0.1)
            )

            match_indices = np.where(match_mask)[0]
            if len(match_indices) == 0:
                # Free memory before continuing
                del nc_evtids, nc_ids, nc_x, nc_y, nc_z
                del nc_ngamma, nc_etot_arr, nc_e1_arr, nc_e2_arr, nc_e3_arr, nc_e4_arr
                continue

            midx = match_indices[0]
            evtid = int(nc_evtids[midx])
            nc_track_id = int(nc_ids[midx])
            nc_time = float(nc_out["nC_time_in_ns"]["pages"][midx])

            # Free NC metadata
            del nc_evtids, nc_ids, nc_x, nc_y, nc_z
            del nc_ngamma, nc_etot_arr, nc_e1_arr, nc_e2_arr, nc_e3_arr, nc_e4_arr

            print(f"    Found in {os.path.basename(fpath)}: "
                  f"evtid={evtid}, nC_track_id={nc_track_id}")

            # Extract optical photons for this NC event only
            optical = f["hit"]["optical"]
            total_photons = len(optical["evtid"]["pages"])
            if total_photons == 0:
                return 0

            # Stream through optical data in chunks to avoid OOM
            filtered_count = 0
            chunk_size = get_chunk_size()
            num_chunks = (total_photons - 1) // chunk_size + 1

            for ci in range(num_chunks):
                cs = ci * chunk_size
                ce = min(cs + chunk_size, total_photons)

                opt_evtids = optical["evtid"]["pages"][cs:ce]
                opt_nc_ids = optical["nC_track_id"]["pages"][cs:ce]

                evt_mask = (opt_evtids == evtid) & (opt_nc_ids == nc_track_id)
                n_match = int(evt_mask.sum())
                if n_match == 0:
                    del opt_evtids, opt_nc_ids
                    continue

                # Load position/momentum only for matched photons
                x = np.array(optical["x_position_in_m"]["pages"][cs:ce], dtype=np.float32)[evt_mask] * 1000
                y = np.array(optical["y_position_in_m"]["pages"][cs:ce], dtype=np.float32)[evt_mask] * 1000
                z = np.array(optical["z_position_in_m"]["pages"][cs:ce], dtype=np.float32)[evt_mask] * 1000
                px = np.array(optical["x_momentum_direction"]["pages"][cs:ce], dtype=np.float32)[evt_mask]
                py = np.array(optical["y_momentum_direction"]["pages"][cs:ce], dtype=np.float32)[evt_mask]
                pz = np.array(optical["z_momentum_direction"]["pages"][cs:ce], dtype=np.float32)[evt_mask]
                times = optical["time_in_ns"]["pages"][cs:ce][evt_mask]

                # Time filter
                time_ok = (times >= nc_time) & (times <= nc_time + 200.0)
                x, y, z = x[time_ok], y[time_ok], z[time_ok]
                px, py, pz = px[time_ok], py[time_ok], pz[time_ok]

                if len(x) == 0:
                    del opt_evtids, opt_nc_ids
                    continue

                # Momentum filter
                mask_bot = z <= z_cut_bot
                mask_top = z >= z_cut_top
                mask_barrel = ~mask_bot & ~mask_top

                final_mask = np.zeros(len(z), dtype=bool)
                final_mask[mask_bot] = pz[mask_bot] <= 0
                final_mask[mask_top] = pz[mask_top] >= 0
                if np.any(mask_barrel):
                    final_mask[mask_barrel] = check_radial_momentum(
                        x[mask_barrel], y[mask_barrel],
                        px[mask_barrel], py[mask_barrel],
                    )

                filtered_count += int(final_mask.sum())

                del opt_evtids, opt_nc_ids, x, y, z, px, py, pz, times
                del mask_bot, mask_top, mask_barrel, final_mask
                gc.collect()

            return filtered_count

    return None


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────
def plot_a_global_summary(
    raw_total: int, pp_total: int,
    raw_nc: int, pp_nc: int,
    hits_per_uid: Dict[int, int],
) -> None:
    """Plot A: Global summary with bar chart."""
    print("\n--- Comparison A: Global Summary ---")

    ratio = raw_total / pp_total if pp_total > 0 else float("nan")

    print(f"\n  {'':20s} {'Raw':>14s} {'Postprocessed':>14s}")
    print("  " + "-" * 50)
    print(f"  {'NC events':20s} {raw_nc:>14,} {pp_nc:>14,}")
    print(f"  {'Total filtered hits':20s} {raw_total:>14,} {pp_total:>14,}")
    print(f"  {'Ratio (Raw/PP)':20s} {ratio:>14.6f}")
    print(f"  {'Hits per NC (Raw)':20s} {raw_total / raw_nc:>14.4f}")
    print(f"  {'Hits per NC (PP)':20s} {pp_total / pp_nc:>14.4f}")
    print()
    print(f"  Raw det_uid breakdown:")
    for uid, count in sorted(hits_per_uid.items()):
        print(f"    UID {uid}: {count:>12,} ({100 * count / raw_total:.1f}%)")

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total hits
    ax = axes[0]
    bars = ax.bar(
        ["SSD Raw", "SSD Postprocessed"],
        [raw_total, pp_total],
        color=["steelblue", "coral"],
        edgecolor="black",
    )
    ax.set_ylabel("Total photon hits")
    ax.set_title(f"Global Hit Comparison (Ratio = {ratio:.4f})")
    for bar, val in zip(bars, [raw_total, pp_total]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,}", ha="center", va="bottom", fontsize=10)

    # Hits per NC
    ax = axes[1]
    raw_per_nc = raw_total / raw_nc if raw_nc > 0 else 0
    pp_per_nc = pp_total / pp_nc if pp_nc > 0 else 0
    ratio_per_nc = raw_per_nc / pp_per_nc if pp_per_nc > 0 else float("nan")
    bars = ax.bar(
        ["SSD Raw", "SSD Postprocessed"],
        [raw_per_nc, pp_per_nc],
        color=["steelblue", "coral"],
        edgecolor="black",
    )
    ax.set_ylabel("Hits per NC event")
    ax.set_title(f"Normalized per NC (Ratio = {ratio_per_nc:.4f})")
    for bar, val in zip(bars, [raw_per_nc, pp_per_nc]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fpath = OUTPUT_DIR / "A_global_summary.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


def plot_c_event_match(matched_events: List[Dict]) -> None:
    """Plot C: Per-NC event comparison bar chart."""
    print(f"\n--- Comparison C: {len(matched_events)} Matched Events ---")

    if not matched_events:
        print("  No events matched - skipping Plot C")
        return

    n = len(matched_events)

    # Print table
    print(f"\n  {'NC#':>4s} {'PP hits':>10s} {'Raw hits':>10s} {'Ratio':>10s} "
          f"{'Position (mm)':>30s}")
    print("  " + "-" * 70)
    for i, evt in enumerate(matched_events):
        ratio = evt["raw_hits"] / evt["pp_hits"] if evt["pp_hits"] > 0 else float("nan")
        pos = f"({evt['x_mm']:.0f}, {evt['y_mm']:.0f}, {evt['z_mm']:.0f})"
        print(f"  {i+1:>4d} {evt['pp_hits']:>10d} {evt['raw_hits']:>10d} "
              f"{ratio:>10.4f} {pos:>30s}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(max(8, 2 * n), 5))

    x = np.arange(n)
    width = 0.35

    pp_vals = [e["pp_hits"] for e in matched_events]
    raw_vals = [e["raw_hits"] for e in matched_events]

    ax.bar(x - width / 2, raw_vals, width, label="SSD Raw", color="steelblue", edgecolor="black")
    ax.bar(x + width / 2, pp_vals, width, label="SSD Postprocessed", color="coral", edgecolor="black")

    for i, (rv, pv) in enumerate(zip(raw_vals, pp_vals)):
        ratio = rv / pv if pv > 0 else float("nan")
        ax.text(x[i], max(rv, pv) * 1.02, f"R={ratio:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("NC event")
    ax.set_ylabel("Total photon hits")
    ax.set_title(f"Event-Matched Hit Comparison ({n} NCs)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"NC {i+1}" for i in range(n)])
    ax.legend()

    plt.tight_layout()
    fpath = OUTPUT_DIR / "C_event_matched_comparison.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    geometry = GeometryConfig()
    chunk_size = get_chunk_size()

    print("=" * 70)
    print("SSD Raw vs. Postprocessed Efficiency Comparison")
    print("=" * 70)
    print(f"  Raw dir:  {SSD_RAW_DIR}")
    print(f"  PP dir:   {SSD_POSTPROCESSED_DIR}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print("=" * 70)

    # --- A: Global comparison ---
    raw_total, raw_nc, hits_per_uid = process_ssd_raw(SSD_RAW_DIR, geometry, chunk_size)
    pp_total, pp_nc = process_ssd_postprocessed(SSD_POSTPROCESSED_DIR)

    if raw_nc != pp_nc:
        print(f"\n  ⚠ NC count mismatch: Raw={raw_nc:,}, PP={pp_nc:,}")
    else:
        print(f"\n  ✓ NC counts match: {raw_nc:,}")

    plot_a_global_summary(raw_total, pp_total, raw_nc, pp_nc, hits_per_uid)

    # --- C: Event-level matching ---
    print("\n" + "=" * 60)
    print(f"Event-level matching ({N_EVENT_MATCHES} events)...")

    nc_samples = sample_nc_from_postprocessed(
        SSD_POSTPROCESSED_DIR, n_samples=N_EVENT_MATCHES
    )

    matched_events = []
    for j, nc_sample in enumerate(nc_samples):
        print(f"\n  Event {j+1}/{len(nc_samples)}: "
              f"pos=({nc_sample['x_mm']:.1f}, {nc_sample['y_mm']:.1f}, {nc_sample['z_mm']:.1f}) mm, "
              f"PP hits={nc_sample['pp_total_hits']}")

        raw_hits = find_nc_in_raw_and_count_hits(SSD_RAW_DIR, nc_sample, geometry)

        if raw_hits is None:
            print(f"    ✗ Not found in raw data")
            continue

        ratio = raw_hits / nc_sample["pp_total_hits"] if nc_sample["pp_total_hits"] > 0 else float("nan")
        print(f"    ✓ Raw={raw_hits}, PP={nc_sample['pp_total_hits']}, Ratio={ratio:.4f}")

        matched_events.append({
            "x_mm": nc_sample["x_mm"],
            "y_mm": nc_sample["y_mm"],
            "z_mm": nc_sample["z_mm"],
            "pp_hits": nc_sample["pp_total_hits"],
            "raw_hits": raw_hits,
        })

    plot_c_event_match(matched_events)

    # --- Final Summary ---
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    ratio = raw_total / pp_total if pp_total > 0 else float("nan")
    print(f"  Global ratio (Raw / Postprocessed): {ratio:.6f}")

    if abs(ratio - 1.0) < 0.005:
        print("  ✓ No significant photon loss detected (<0.5%)")
    elif ratio < 1.0:
        loss_pct = (1.0 - ratio) * 100
        print(f"  ⚠ Postprocessed has MORE hits than raw by {loss_pct:.2f}%")
        print(f"    -> Possible: postprocessing adds photons or uses different filters")
    else:
        loss_pct = (ratio - 1.0) * 100
        print(f"  ⚠ Postprocessed LOSES {loss_pct:.2f}% of photons vs raw")

    if matched_events:
        event_ratios = [
            e["raw_hits"] / e["pp_hits"]
            for e in matched_events if e["pp_hits"] > 0
        ]
        if event_ratios:
            print(f"\n  Event-level ratios: "
                  f"mean={np.mean(event_ratios):.4f}, "
                  f"std={np.std(event_ratios):.4f}, "
                  f"range=[{min(event_ratios):.4f}, {max(event_ratios):.4f}]")

    print(f"\n  Plots saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()