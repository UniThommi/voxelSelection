#!/usr/bin/env python3
# findWallGradientReason.py
"""
Diagnostic: Compare z-distribution of NC sources vs detected photon z-positions
for SSD and PMT on the wall, to understand the wall ratio gradient.

Produces:
1. NC source z-distribution (where do neutron captures happen?)
2. Detected photon z-distribution on wall (SSD vs PMT)
3. 2D histogram: NC source z vs detected photon z (SSD and PMT separately)
4. Incidence angle distribution on wall per z-region

Author: Thomas Buerger (University of Tübingen)
"""

import h5py
import numpy as np
import gc
import psutil
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/pscratch/sd/t/tbuerger/data/proofeDetectionEfficiencies/"
                "experimentEfficiencyRatio/PMTs_div_SSD_1")
SSD_DIR = BASE_DIR / "ratio_1_SSD"
PMT_DIR = BASE_DIR / "ratio_1_PMTs"
PMT_JSON_PATH = Path("/global/cfs/projectdirs/legend/users/tbuerger/sim/data/"
                     "optPhotonSensitiveSurface/homogeneous300PMTpositions_currentDist.json")
OUTPUT_DIR = BASE_DIR / "zone_analysis" / "diagnostics"

# Geometry
R_ZYLINDER = 4300.0
Z_CUT_BOT = -4979.0
Z_CUT_TOP = 3918.0

# SSD UIDs
SSD_UID_WALL = 1965

NC_TIME_WINDOW = 200.0  # ns

# Areas to analyze
DIAGNOSTIC_AREAS = ['pit', 'top', 'wall', 'bot']


# =============================================================================
# HELPERS (from zoneRatioAnalysis.py)
# =============================================================================

def get_chunk_size() -> int:
    available_mem_gb = psutil.virtual_memory().available / (1024**3)
    if available_mem_gb > 400: return 50000
    elif available_mem_gb > 200: return 30000
    elif available_mem_gb > 50: return 20000
    elif available_mem_gb > 30: return 15000
    elif available_mem_gb > 20: return 10000
    else: return 5000


def load_nc_data_dict(f: h5py.File) -> Dict:
    nc_evtid = f['hit']['MyNeutronCaptureOutput']['evtid']['pages'][:]
    nc_nC_id = f['hit']['MyNeutronCaptureOutput']['nC_track_id']['pages'][:]
    nc_time = f['hit']['MyNeutronCaptureOutput']['nC_time_in_ns']['pages'][:]
    nc_x = f['hit']['MyNeutronCaptureOutput']['nC_x_position_in_m']['pages'][:] * 1000
    nc_y = f['hit']['MyNeutronCaptureOutput']['nC_y_position_in_m']['pages'][:] * 1000
    nc_z = f['hit']['MyNeutronCaptureOutput']['nC_z_position_in_m']['pages'][:] * 1000
    nc_data = {}
    for idx in range(len(nc_evtid)):
        key = (nc_evtid[idx], nc_nC_id[idx])
        nc_data[key] = {
            'nC_time': nc_time[idx],
            'nC_x': nc_x[idx],
            'nC_y': nc_y[idx],
            'nC_z': nc_z[idx],
            'nC_r': np.sqrt(nc_x[idx]**2 + nc_y[idx]**2),
        }
    return nc_data


def load_pmt_uids(json_path: Path) -> set:
    """Load set of valid PMT UIDs."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    uids = set()
    for entry in data:
        index = entry['index']
        uid_normal = '10' + index
        if len(uid_normal) == 8:
            uids.add(int(uid_normal))
        else:
            uid_overflow = '1' + index
            if len(uid_overflow) == 8:
                uids.add(int(uid_overflow))
    return uids


def checkRadialMomentumVectorized(x, y, z, px, py, pz):
    return (x * px + y * py) >= 0


# =============================================================================
# DATA COLLECTION
# =============================================================================

def process_files_diagnostic(
    base_path: Path,
    chunk_size: int,
    is_pmt: bool,
    pmt_uids: set = None,
    areas: List[str] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Process SSD or PMT files and collect per area:
    - NC source z-positions
    - Detected photon z-positions
    - Incidence angles on surface
    - Time since NC (t_photon - t_NC)
    - Momentum components (pr, pz) at detection

    Returns dict: {area_name: {field: np.ndarray}}
    """
    if areas is None:
        areas = ['wall']

    SSD_UIDS = {'pit': 1966, 'bot': 1967, 'top': 1968, 'wall': 1965}

    # Collectors per area
    collectors = {}
    for area in areas:
        collectors[area] = {
            'nc_source_z': [],
            'nc_source_r': [],
            'detection_z': [],
            'detection_r': [],
            'incidence_angles': [],
            'time_since_nc': [],
            'pr': [],
            'pz_momentum': [],
        }

    run_dirs = sorted(base_path.glob("run_*"))
    setup_name = "PMT" if is_pmt else "SSD"
    print(f"\nProcessing {setup_name}: {len(run_dirs)} runs, areas={areas}")

    # For PMT: build wall/pit/top/bot UID sets
    pmt_uids_by_layer = {'pit': set(), 'bot': set(), 'top': set(), 'wall': set()}
    if is_pmt:
        with open(PMT_JSON_PATH, 'r') as f:
            data = json.load(f)
        for entry in data:
            layer = entry['layer'].lower()
            if layer not in pmt_uids_by_layer:
                continue
            index = entry['index']
            uid_normal = '10' + index
            if len(uid_normal) == 8:
                pmt_uids_by_layer[layer].add(int(uid_normal))
            else:
                uid_overflow = '1' + index
                if len(uid_overflow) == 8:
                    pmt_uids_by_layer[layer].add(int(uid_overflow))
        for layer in areas:
            if layer in pmt_uids_by_layer:
                print(f"  {layer} PMT UIDs: {len(pmt_uids_by_layer[layer])}")

    total_files = 0
    for run_idx, run_dir in enumerate(run_dirs, 1):
        hdf5_files = sorted(run_dir.glob("output_t*.hdf5"))
        print(f"  Run {run_idx}/{len(run_dirs)}: {run_dir.name} ({len(hdf5_files)} files)")

        for file_idx, hdf5_file in enumerate(hdf5_files):
            if (file_idx + 1) % 50 == 0:
                print(f"    File {file_idx + 1}/{len(hdf5_files)}")
            total_files += 1

            try:
                with h5py.File(hdf5_file, 'r') as f:
                    nc_data = load_nc_data_dict(f)
                    total = len(f['hit']['optical']['x_position_in_m']['pages'])
                    num_chunks = (total - 1) // chunk_size + 1

                    for chunk_idx in range(num_chunks):
                        cs = chunk_idx * chunk_size
                        ce = min(cs + chunk_size, total)

                        x = np.array(f['hit']['optical']['x_position_in_m']['pages'][cs:ce],
                                     dtype=np.float32) * 1000
                        y = np.array(f['hit']['optical']['y_position_in_m']['pages'][cs:ce],
                                     dtype=np.float32) * 1000
                        z = np.array(f['hit']['optical']['z_position_in_m']['pages'][cs:ce],
                                     dtype=np.float32) * 1000
                        px = np.array(f['hit']['optical']['x_momentum_direction']['pages'][cs:ce],
                                      dtype=np.float32)
                        py = np.array(f['hit']['optical']['y_momentum_direction']['pages'][cs:ce],
                                      dtype=np.float32)
                        pz_arr = np.array(f['hit']['optical']['z_momentum_direction']['pages'][cs:ce],
                                          dtype=np.float32)
                        evtid = f['hit']['optical']['evtid']['pages'][cs:ce]
                        nc_id = f['hit']['optical']['nC_track_id']['pages'][cs:ce]
                        time = f['hit']['optical']['time_in_ns']['pages'][cs:ce]
                        det_uid = f['hit']['optical']['det_uid']['pages'][cs:ce]

                        # NC time filter + source positions
                        nc_times = np.full(len(time), np.inf, dtype=np.float32)
                        nc_source_z_arr = np.full(len(time), np.nan, dtype=np.float32)
                        nc_source_r_arr = np.full(len(time), np.nan, dtype=np.float32)
                        for idx in range(len(evtid)):
                            key = (evtid[idx], nc_id[idx])
                            if key in nc_data:
                                nc_times[idx] = nc_data[key]['nC_time']
                                nc_source_z_arr[idx] = nc_data[key]['nC_z']
                                nc_source_r_arr[idx] = nc_data[key]['nC_r']

                        time_mask = ((nc_times != np.inf) &
                                     (time >= nc_times) &
                                     (time <= nc_times + NC_TIME_WINDOW))

                        x = x[time_mask]; y = y[time_mask]; z = z[time_mask]
                        px = px[time_mask]; py = py[time_mask]; pz_arr = pz_arr[time_mask]
                        det_uid = det_uid[time_mask]
                        nc_source_z_arr = nc_source_z_arr[time_mask]
                        nc_source_r_arr = nc_source_r_arr[time_mask]
                        t_since_nc = (time[time_mask] - nc_times[time_mask]).astype(np.float32)

                        # Momentum filter
                        mask_bot = z <= Z_CUT_BOT
                        mask_top = z >= Z_CUT_TOP
                        mask_barrel = ~mask_bot & ~mask_top

                        final_mask = np.zeros_like(z, dtype=bool)
                        final_mask[mask_bot] = pz_arr[mask_bot] <= 0
                        final_mask[mask_top] = pz_arr[mask_top] >= 0
                        if np.any(mask_barrel):
                            final_mask[mask_barrel] = checkRadialMomentumVectorized(
                                x[mask_barrel], y[mask_barrel], z[mask_barrel],
                                px[mask_barrel], py[mask_barrel], pz_arr[mask_barrel])

                        x = x[final_mask]; y = y[final_mask]; z = z[final_mask]
                        px = px[final_mask]; py = py[final_mask]; pz_arr = pz_arr[final_mask]
                        det_uid = det_uid[final_mask]
                        nc_source_z_arr = nc_source_z_arr[final_mask]
                        nc_source_r_arr = nc_source_r_arr[final_mask]
                        t_since_nc = t_since_nc[final_mask]

                        pr = np.sqrt(px**2 + py**2)

                        # Per-area collection
                        for area in areas:
                            if is_pmt:
                                area_mask = np.isin(det_uid, list(pmt_uids_by_layer.get(area, set())))
                            else:
                                area_mask = det_uid == SSD_UIDS.get(area, -1)

                            if not np.any(area_mask):
                                continue

                            det_z = z[area_mask]
                            src_z = nc_source_z_arr[area_mask]
                            t_nc = t_since_nc[area_mask]
                            pr_a = pr[area_mask]
                            pz_a = pz_arr[area_mask]

                            det_r = np.sqrt(x[area_mask]**2 + y[area_mask]**2)
                            src_r = nc_source_r_arr[area_mask]

                            collectors[area]['detection_z'].append(det_z)
                            collectors[area]['detection_r'].append(det_r)
                            collectors[area]['nc_source_z'].append(src_z)
                            collectors[area]['nc_source_r'].append(src_r)
                            collectors[area]['time_since_nc'].append(t_nc)
                            collectors[area]['pr'].append(pr_a)
                            collectors[area]['pz_momentum'].append(pz_a)

                            # Incidence angle
                            if area == 'wall':
                                x_a = x[area_mask]; y_a = y[area_mask]
                                px_a = px[area_mask]; py_a = py[area_mask]
                                r_a = np.sqrt(x_a**2 + y_a**2)
                                r_a = np.maximum(r_a, 1e-6)
                                cos_theta = np.abs(px_a * (x_a/r_a) + py_a * (y_a/r_a))
                                cos_theta = np.clip(cos_theta, 0.0, 1.0)
                                angles = np.degrees(np.arccos(cos_theta))
                            elif area in ('pit', 'bot'):
                                # Normal is -z (downward)
                                cos_theta = np.abs(pz_a)
                                cos_theta = np.clip(cos_theta, 0.0, 1.0)
                                angles = np.degrees(np.arccos(cos_theta))
                            elif area == 'top':
                                # Normal is +z (upward)
                                cos_theta = np.abs(pz_a)
                                cos_theta = np.clip(cos_theta, 0.0, 1.0)
                                angles = np.degrees(np.arccos(cos_theta))
                            else:
                                angles = np.zeros(np.sum(area_mask))

                            collectors[area]['incidence_angles'].append(
                                np.column_stack([det_z, angles]))

                        del x, y, z, px, py, pz_arr, evtid, nc_id, time, det_uid
                        del nc_times, nc_source_z_arr, nc_source_r_arr, t_since_nc
                        gc.collect()

            except Exception as e:
                print(f"    Error: {e}")

    print(f"  Total files: {total_files}")

    # Concatenate per area
    result = {}
    for area in areas:
        c = collectors[area]
        result[area] = {
            'detection_z': np.concatenate(c['detection_z']) if c['detection_z'] else np.array([]),
            'detection_r': np.concatenate(c['detection_r']) if c['detection_r'] else np.array([]),
            'nc_source_z': np.concatenate(c['nc_source_z']) if c['nc_source_z'] else np.array([]),
            'nc_source_r': np.concatenate(c['nc_source_r']) if c['nc_source_r'] else np.array([]),
            'incidence_angles': np.concatenate(c['incidence_angles']) if c['incidence_angles'] else np.array([]).reshape(0, 2),
            'time_since_nc': np.concatenate(c['time_since_nc']) if c['time_since_nc'] else np.array([]),
            'pr': np.concatenate(c['pr']) if c['pr'] else np.array([]),
            'pz_momentum': np.concatenate(c['pz_momentum']) if c['pz_momentum'] else np.array([]),
        }
        print(f"  {area}: {len(result[area]['detection_z']):,} photon hits")

    return result


# =============================================================================
# PLOTTING
# =============================================================================

def make_diagnostic_plots(ssd_data, pmt_data, output_dir, areas,
                          ssd_all_ncs=None, pmt_all_ncs=None):
    """Generate all diagnostic plots per area."""

    z_bins = np.linspace(Z_CUT_BOT, Z_CUT_TOP, 50)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    r_bins = np.linspace(0, R_ZYLINDER + 200, 50)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2

    # Geometry markers
    MARKERS = {
        'pit': [('Skirt r=3800', 3800, 'r')],
        'bot': [('SSD r=4300', 4300, 'r'), ('Foot r=3950', 3950, 'g')],
        'top': [('SSD r=4300', 4300, 'r'), ('Cryo neck r=1200', 1200, 'g')],
        'wall': [('SSD r=4300', 4300, 'r')],
    }
    Z_MARKERS = {
        'wall': [],  # no fixed z-markers needed
    }

    def add_r_markers(ax, area, vertical=True):
        for label, val, col in MARKERS.get(area, []):
            if vertical:
                ax.axvline(val, color=col, linestyle=':', linewidth=1.5,
                           alpha=0.7, label=label)
            else:
                ax.axhline(val, color=col, linestyle=':', linewidth=1.5,
                           alpha=0.7, label=label)

    def add_photon_counts(ax, ssd, pmt):
        n_ssd = len(ssd['detection_z'])
        n_pmt = len(pmt['detection_z'])
        ax.text(0.98, 0.98,
                f'SSD: {n_ssd:,} photons\nPMT: {n_pmt:,} photons',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # =========================================================================
    # General NC plots: ALL NCs from MyNeutronCaptureOutput
    # =========================================================================
    print("\n  --- General NC distribution plots (all NCs, not detection-filtered) ---")

    if ssd_all_ncs is not None and pmt_all_ncs is not None:
        for field, label, bins, xlabel, fname in [
            ('nc_z', 'NC source z', z_bins, 'NC source z [mm]', 'all_nc_z'),
            ('nc_r', 'NC source r', r_bins, 'NC source r [mm]', 'all_nc_r'),
        ]:
            fig, ax = plt.subplots(figsize=(12, 6))

            n_ssd = len(ssd_all_ncs[field])
            n_pmt = len(pmt_all_ncs[field])

            if n_ssd > 0:
                ax.hist(ssd_all_ncs[field], bins=bins, alpha=0.6, density=True,
                        label=f'SSD sim (N={n_ssd:,})', color='blue')
            if n_pmt > 0:
                ax.hist(pmt_all_ncs[field], bins=bins, alpha=0.6, density=True,
                        label=f'PMT sim (N={n_pmt:,})', color='red')

            if field == 'nc_r':
                for a in areas:
                    for mk_label, val, col in MARKERS.get(a, []):
                        ax.axvline(val, color=col, linestyle=':', linewidth=1.5,
                                   alpha=0.7, label=mk_label)
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())
            else:
                ax.legend()

            ax.set_xlabel(xlabel)
            ax.set_ylabel('Probability density')
            ax.set_title(f'All NCs (unfiltered): {label} distribution')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f"{fname}.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {fname}.png")

    # =========================================================================
    # Per-area plots
    # =========================================================================
    for area in areas:
        ssd = ssd_data.get(area)
        pmt = pmt_data.get(area)
        if ssd is None or pmt is None:
            continue
        if len(ssd['detection_z']) == 0 and len(pmt['detection_z']) == 0:
            continue

        print(f"\n  --- Plots for {area.upper()} ---")

        n_ssd_total = len(ssd['time_since_nc'])
        n_pmt_total = len(pmt['time_since_nc'])

        # =====================================================================
        # Plot 1: NC source z-distribution
        # =====================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        if len(ssd['nc_source_z']) > 0:
            ax.hist(ssd['nc_source_z'], bins=z_bins, alpha=0.6,
                    label=f'SSD sim (N={len(ssd["nc_source_z"]):,})',
                    density=True, color='blue')
        if len(pmt['nc_source_z']) > 0:
            ax.hist(pmt['nc_source_z'], bins=z_bins, alpha=0.6,
                    label=f'PMT sim (N={len(pmt["nc_source_z"]):,})',
                    density=True, color='red')
        ax.set_xlabel('NC source z [mm]')
        ax.set_ylabel('Probability density')
        ax.set_title(f'{area.upper()}: Z-distribution of NC sources (detected photons only)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{area}_nc_source_z.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {area}_nc_source_z.png")

        # =====================================================================
        # Plot 1b: NC source r-distribution
        # =====================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        if len(ssd['nc_source_r']) > 0:
            ax.hist(ssd['nc_source_r'], bins=r_bins, alpha=0.6,
                    label=f'SSD sim (N={len(ssd["nc_source_r"]):,})',
                    density=True, color='blue')
        if len(pmt['nc_source_r']) > 0:
            ax.hist(pmt['nc_source_r'], bins=r_bins, alpha=0.6,
                    label=f'PMT sim (N={len(pmt["nc_source_r"]):,})',
                    density=True, color='red')
        add_r_markers(ax, area)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.set_xlabel('NC source r [mm]')
        ax.set_ylabel('Probability density')
        ax.set_title(f'{area.upper()}: R-distribution of NC sources (detected photons only)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{area}_nc_source_r.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {area}_nc_source_r.png")

        # =====================================================================
        # Plot 2: Detection position (z for wall, r for others)
        # =====================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        if area == 'wall':
            if len(ssd['detection_z']) > 0:
                ax.hist(ssd['detection_z'], bins=z_bins, alpha=0.6,
                        label=f'SSD (N={len(ssd["detection_z"]):,})',
                        density=True, color='blue')
            if len(pmt['detection_z']) > 0:
                ax.hist(pmt['detection_z'], bins=z_bins, alpha=0.6,
                        label=f'PMT (N={len(pmt["detection_z"]):,})',
                        density=True, color='red')
            ax.set_xlabel('Detection z [mm]')
            ax.set_title(f'{area.upper()}: Z-distribution of detected photons')
        else:
            if len(ssd['detection_r']) > 0:
                ax.hist(ssd['detection_r'], bins=r_bins, alpha=0.6,
                        label=f'SSD (N={len(ssd["detection_r"]):,})',
                        density=True, color='blue')
            if len(pmt['detection_r']) > 0:
                ax.hist(pmt['detection_r'], bins=r_bins, alpha=0.6,
                        label=f'PMT (N={len(pmt["detection_r"]):,})',
                        density=True, color='red')
            add_r_markers(ax, area)
            ax.set_xlabel('Detection r [mm]')
            ax.set_title(f'{area.upper()}: R-distribution of detected photons')
        ax.set_ylabel('Probability density')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{area}_detection_pos.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {area}_detection_pos.png")

        # =====================================================================
        # Plot 3a: 2D histogram NC source z vs detection z
        # =====================================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        # Find global vmax for shared scale
        counts_all = []
        for data in [ssd, pmt]:
            src_z = data['nc_source_z']
            det_z = data['detection_z']
            valid = ~np.isnan(src_z) & ~np.isnan(det_z)
            if np.sum(valid) > 0:
                h_tmp, _, _ = np.histogram2d(src_z[valid], det_z[valid],
                                             bins=[z_bins, z_bins])
                counts_all.append(h_tmp.max())
        global_vmax = max(counts_all) if counts_all else 1

        for ax, data, title in [(ax1, ssd, 'SSD'), (ax2, pmt, 'PMT')]:
            src_z = data['nc_source_z']
            det_z = data['detection_z']
            valid = ~np.isnan(src_z) & ~np.isnan(det_z)
            if np.sum(valid) > 0:
                h = ax.hist2d(src_z[valid], det_z[valid], bins=[z_bins, z_bins],
                              cmap='inferno',
                              norm=matplotlib.colors.LogNorm(vmin=1, vmax=global_vmax))
                fig.colorbar(h[3], ax=ax, label='Counts')
                ax.plot([Z_CUT_BOT, Z_CUT_TOP], [Z_CUT_BOT, Z_CUT_TOP],
                        'w--', linewidth=1, alpha=0.5)
            n_valid = np.sum(valid)
            ax.set_xlabel('NC source z [mm]')
            ax.set_ylabel('Detection z [mm]')
            ax.set_title(f'{title} (N={n_valid:,})')
        fig.suptitle(f'{area.upper()}: NC source z vs detection z', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{area}_nc_vs_det_z_2d.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {area}_nc_vs_det_z_2d.png")

        # =====================================================================
        # Plot 3b: 2D histogram NC source r vs detection r
        # =====================================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        counts_all = []
        for data in [ssd, pmt]:
            src_r = data['nc_source_r']
            det_r = data['detection_r']
            valid = ~np.isnan(src_r) & ~np.isnan(det_r)
            if np.sum(valid) > 0:
                h_tmp, _, _ = np.histogram2d(src_r[valid], det_r[valid],
                                             bins=[r_bins, r_bins])
                counts_all.append(h_tmp.max())
        global_vmax_r = max(counts_all) if counts_all else 1

        for ax, data, title in [(ax1, ssd, 'SSD'), (ax2, pmt, 'PMT')]:
            src_r = data['nc_source_r']
            det_r = data['detection_r']
            valid = ~np.isnan(src_r) & ~np.isnan(det_r)
            if np.sum(valid) > 0:
                h = ax.hist2d(src_r[valid], det_r[valid], bins=[r_bins, r_bins],
                              cmap='inferno',
                              norm=matplotlib.colors.LogNorm(vmin=1, vmax=global_vmax_r))
                fig.colorbar(h[3], ax=ax, label='Counts')
                add_r_markers(ax, area, vertical=True)
                add_r_markers(ax, area, vertical=False)
            n_valid = np.sum(valid)
            ax.set_xlabel('NC source r [mm]')
            ax.set_ylabel('Detection r [mm]')
            ax.set_title(f'{title} (N={n_valid:,})')
        fig.suptitle(f'{area.upper()}: NC source r vs detection r', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{area}_nc_vs_det_r_2d.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {area}_nc_vs_det_r_2d.png")

        # =====================================================================
        # Plot 4: Time since NC
        # =====================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        t_bins = np.linspace(0, NC_TIME_WINDOW, 100)

        if n_ssd_total > 0:
            mean_t_ssd = np.mean(ssd['time_since_nc'])
            weights_ssd = np.ones(n_ssd_total) / n_ssd_total
            ax.hist(ssd['time_since_nc'], bins=t_bins, alpha=0.6,
                    weights=weights_ssd,
                    label=f'SSD (N={n_ssd_total:,}, ⟨t⟩={mean_t_ssd:.1f} ns)',
                    color='blue')
            ax.axvline(mean_t_ssd, color='blue', linestyle='--', linewidth=1.5)
        if n_pmt_total > 0:
            mean_t_pmt = np.mean(pmt['time_since_nc'])
            weights_pmt = np.ones(n_pmt_total) / n_pmt_total
            ax.hist(pmt['time_since_nc'], bins=t_bins, alpha=0.6,
                    weights=weights_pmt,
                    label=f'PMT (N={n_pmt_total:,}, ⟨t⟩={mean_t_pmt:.1f} ns)',
                    color='red')
            ax.axvline(mean_t_pmt, color='red', linestyle='--', linewidth=1.5)

        ax.set_xlabel('Time since NC [ns]')
        ax.set_ylabel('Fraction of photons')
        ax.set_title(f'{area.upper()}: Photon arrival time relative to NC')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{area}_time_since_nc.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {area}_time_since_nc.png")

        # =====================================================================
        # Plot 5: Radial momentum pr
        # =====================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        pr_bins = np.linspace(0, 1, 80)
        if n_ssd_total > 0:
            weights_ssd = np.ones(n_ssd_total) / n_ssd_total
            ax.hist(ssd['pr'], bins=pr_bins, alpha=0.6,
                    weights=weights_ssd, label=f'SSD (N={n_ssd_total:,})', color='blue')
        if n_pmt_total > 0:
            weights_pmt = np.ones(n_pmt_total) / n_pmt_total
            ax.hist(pmt['pr'], bins=pr_bins, alpha=0.6,
                    weights=weights_pmt, label=f'PMT (N={n_pmt_total:,})', color='red')
        ax.set_xlabel('p_r = √(p_x² + p_y²)')
        ax.set_ylabel('Fraction of photons')
        ax.set_title(f'{area.upper()}: Radial momentum component at detection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{area}_momentum_pr.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {area}_momentum_pr.png")

        # =====================================================================
        # Plot 6: Vertical momentum pz
        # =====================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        pz_bins = np.linspace(-1, 1, 80)
        if n_ssd_total > 0:
            weights_ssd = np.ones(n_ssd_total) / n_ssd_total
            ax.hist(ssd['pz_momentum'], bins=pz_bins, alpha=0.6,
                    weights=weights_ssd, label=f'SSD (N={n_ssd_total:,})', color='blue')
        if n_pmt_total > 0:
            weights_pmt = np.ones(n_pmt_total) / n_pmt_total
            ax.hist(pmt['pz_momentum'], bins=pz_bins, alpha=0.6,
                    weights=weights_pmt, label=f'PMT (N={n_pmt_total:,})', color='red')
        ax.set_xlabel('p_z')
        ax.set_ylabel('Fraction of photons')
        ax.set_title(f'{area.upper()}: Vertical momentum component at detection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{area}_momentum_pz.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {area}_momentum_pz.png")

        # =====================================================================
        # Plot 7: Incidence angle distribution
        # =====================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        angle_bins = np.linspace(0, 90, 60)
        if len(ssd['incidence_angles']) > 0:
            ssd_angles = ssd['incidence_angles'][:, 1]
            n_ssd_a = len(ssd_angles)
            weights_ssd = np.ones(n_ssd_a) / n_ssd_a
            ax.hist(ssd_angles, bins=angle_bins, alpha=0.6,
                    weights=weights_ssd, label=f'SSD (N={n_ssd_a:,})', color='blue')
        if len(pmt['incidence_angles']) > 0:
            pmt_angles = pmt['incidence_angles'][:, 1]
            n_pmt_a = len(pmt_angles)
            weights_pmt = np.ones(n_pmt_a) / n_pmt_a
            ax.hist(pmt_angles, bins=angle_bins, alpha=0.6,
                    weights=weights_pmt, label=f'PMT (N={n_pmt_a:,})', color='red')
        ax.set_xlabel('Incidence angle [deg] (0° = normal)')
        ax.set_ylabel('Fraction of photons')
        ax.set_title(f'{area.upper()}: Incidence angle distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{area}_incidence_angle.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {area}_incidence_angle.png")

        # =====================================================================
        # Plot 8: Mean incidence angle vs coordinate
        # =====================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        if area == 'wall':
            coord_bins = z_bins
            coord_centers = z_centers
            xlabel = 'Detection z [mm]'
        else:
            coord_bins = r_bins
            coord_centers = r_centers
            xlabel = 'Detection r [mm]'
            add_r_markers(ax, area)

        for data, label, color in [(ssd, 'SSD', 'blue'), (pmt, 'PMT', 'red')]:
            if len(data['incidence_angles']) == 0:
                continue
            if area == 'wall':
                det_coord = data['incidence_angles'][:, 0]
            else:
                det_coord = data['detection_r'][:len(data['incidence_angles'])]
            angles = data['incidence_angles'][:, 1]
            mean_angles = []
            for i in range(len(coord_bins) - 1):
                mask = (det_coord >= coord_bins[i]) & (det_coord < coord_bins[i+1])
                if np.sum(mask) > 10:
                    mean_angles.append(np.mean(angles[mask]))
                else:
                    mean_angles.append(np.nan)
            ax.plot(coord_centers, mean_angles, 'o-', color=color,
                    linewidth=2, markersize=4, label=label)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Mean incidence angle [deg]')
        ax.set_title(f'{area.upper()}: Mean incidence angle vs position')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{area}_mean_angle_vs_pos.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {area}_mean_angle_vs_pos.png")

        # =====================================================================
        # Plot 9 (wall only): Incidence angle vs z heatmap (SSD + PMT side by side)
        # =====================================================================
        if area == 'wall':
            angle_bins_hm = np.linspace(0, 90, 60)
            z_bins_hm = np.linspace(Z_CUT_BOT, Z_CUT_TOP, 50)

            # Find global vmax
            counts_hm = []
            for data in [ssd, pmt]:
                if len(data['incidence_angles']) > 0:
                    h_tmp, _, _ = np.histogram2d(
                        data['incidence_angles'][:, 1],
                        data['incidence_angles'][:, 0],
                        bins=[angle_bins_hm, z_bins_hm])
                    counts_hm.append(h_tmp.max())
            global_vmax_hm = max(counts_hm) if counts_hm else 1

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

            for ax, data, title in [(ax1, ssd, 'SSD'), (ax2, pmt, 'PMT')]:
                if len(data['incidence_angles']) > 0:
                    angles = data['incidence_angles'][:, 1]
                    det_z = data['incidence_angles'][:, 0]
                    h = ax.hist2d(angles, det_z,
                                  bins=[angle_bins_hm, z_bins_hm],
                                  cmap='inferno',
                                  norm=matplotlib.colors.LogNorm(
                                      vmin=1, vmax=global_vmax_hm))
                    fig.colorbar(h[3], ax=ax, label='Counts')
                ax.set_xlabel('Incidence angle [deg] (0° = normal)')
                ax.set_ylabel('Detection z [mm]')
                n_a = len(data['incidence_angles'])
                ax.set_title(f'{title} (N={n_a:,})')

            fig.suptitle('WALL: Incidence angle vs z-position', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / "wall_angle_vs_z_heatmap.png",
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: wall_angle_vs_z_heatmap.png")

def collect_all_nc_positions(base_path: Path) -> Dict[str, np.ndarray]:
    """
    Read ALL NC positions from MyNeutronCaptureOutput (not filtered by detection).
    """
    nc_z_list = []
    nc_r_list = []

    run_dirs = sorted(base_path.glob("run_*"))
    print(f"\n  Collecting all NC positions from {len(run_dirs)} runs...")

    total_ncs = 0
    for run_dir in run_dirs:
        hdf5_files = sorted(run_dir.glob("output_t*.hdf5"))
        for hdf5_file in hdf5_files:
            try:
                with h5py.File(hdf5_file, 'r') as f:
                    nc_x = f['hit']['MyNeutronCaptureOutput']['nC_x_position_in_m']['pages'][:] * 1000
                    nc_y = f['hit']['MyNeutronCaptureOutput']['nC_y_position_in_m']['pages'][:] * 1000
                    nc_z = f['hit']['MyNeutronCaptureOutput']['nC_z_position_in_m']['pages'][:] * 1000
                    nc_z_list.append(nc_z)
                    nc_r_list.append(np.sqrt(nc_x**2 + nc_y**2))
                    total_ncs += len(nc_z)
            except Exception as e:
                print(f"    Error: {e}")

    result = {
        'nc_z': np.concatenate(nc_z_list) if nc_z_list else np.array([]),
        'nc_r': np.concatenate(nc_r_list) if nc_r_list else np.array([]),
    }
    print(f"  Total NCs: {total_ncs:,}")
    return result

# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunk_size = get_chunk_size()

    print("=" * 70)
    print("NC Source vs Detection Diagnostic Analysis")
    print("=" * 70)
    print(f"Chunk size: {chunk_size}")
    print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"Areas: {DIAGNOSTIC_AREAS}")

    pmt_uids = load_pmt_uids(PMT_JSON_PATH)
    print(f"Total PMT UIDs: {len(pmt_uids)}")

    # Collect ALL NC positions (not filtered by detection)
    print("\n--- Collecting all NC positions ---")
    ssd_all_ncs = collect_all_nc_positions(SSD_DIR)
    pmt_all_ncs = collect_all_nc_positions(PMT_DIR)

    # Process SSD
    ssd_data = process_files_diagnostic(SSD_DIR, chunk_size, is_pmt=False,
                                         areas=DIAGNOSTIC_AREAS)

    # Process PMT
    pmt_data = process_files_diagnostic(PMT_DIR, chunk_size, is_pmt=True,
                                         pmt_uids=pmt_uids, areas=DIAGNOSTIC_AREAS)

    # Generate plots
    print("\nGenerating diagnostic plots...")
    make_diagnostic_plots(ssd_data, pmt_data, OUTPUT_DIR, DIAGNOSTIC_AREAS,
                          ssd_all_ncs=ssd_all_ncs, pmt_all_ncs=pmt_all_ncs)

    print("\nDone.")


if __name__ == "__main__":
    main()