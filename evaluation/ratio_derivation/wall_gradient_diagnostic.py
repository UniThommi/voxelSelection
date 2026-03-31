#!/usr/bin/env python3
"""wall_gradient_diagnostic.py

Diagnostic: Compare z-distribution of NC sources vs detected photon positions
for SSD and PMT across all detector areas, to understand the wall ratio gradient.

Produces (per area):
  - NC source z/r distribution (where do neutron captures happen?)
  - Detected photon z/r distribution (SSD vs PMT)
  - 2D histograms: NC source z vs detected photon z/r
  - Time since NC distribution
  - Radial/vertical momentum distributions
  - Incidence angle distribution and mean angle vs position
  - (Wall only) Incidence angle vs z heatmap
"""

import argparse
import gc
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import psutil
import h5py
from pathlib import Path
from typing import Dict, List, Tuple

from pmtopt.geometry import R_ZYLINDER, Z_CUT_BOT, Z_CUT_TOP
from ratio_analysis.photon_filters import (
    get_chunk_size, checkRadialMomentumVectorized,
    SSD_UID_PIT, SSD_UID_BOT, SSD_UID_TOP, SSD_UID_WALL,
)


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_BASE = Path("/pscratch/sd/t/tbuerger/data/proofeDetectionEfficiencies/"
             "experimentEfficiencyRatio/PMTs_div_SSD_1")
_DEFAULT_SSD_DIR    = _BASE / "ratio_1_SSD"
_DEFAULT_PMT_DIR    = _BASE / "ratio_1_PMTs"
_DEFAULT_PMT_JSON   = Path(
    "/global/cfs/projectdirs/legend/users/tbuerger/sim/data/"
    "optPhotonSensitiveSurface/homogeneous300PMTpositions_currentDist.json"
)
_DEFAULT_OUTPUT_DIR = _BASE / "zone_analysis" / "diagnostics"

NC_TIME_WINDOW = 200.0  # ns


SSD_UIDS = {'pit': SSD_UID_PIT, 'bot': SSD_UID_BOT, 'top': SSD_UID_TOP, 'wall': SSD_UID_WALL}


# ---------------------------------------------------------------------------
# Extended NC data loader — also stores NC source positions (nC_x/y/z/r)
# ---------------------------------------------------------------------------
def load_nc_data_dict(f: h5py.File) -> Dict:
    """Load NC data: (evtid, nC_track_id) → {nC_time, nC_x, nC_y, nC_z, nC_r}."""
    nc   = f['hit']['MyNeutronCaptureOutput']
    evtid  = nc['evtid']['pages'][:]
    nC_id  = nc['nC_track_id']['pages'][:]
    t_nc   = nc['nC_time_in_ns']['pages'][:]
    x_nc   = nc['nC_x_position_in_m']['pages'][:] * 1000
    y_nc   = nc['nC_y_position_in_m']['pages'][:] * 1000
    z_nc   = nc['nC_z_position_in_m']['pages'][:] * 1000

    nc_data = {}
    for i in range(len(evtid)):
        key = (evtid[i], nC_id[i])
        nc_data[key] = {
            'nC_time': t_nc[i],
            'nC_x':    x_nc[i],
            'nC_y':    y_nc[i],
            'nC_z':    z_nc[i],
            'nC_r':    np.sqrt(x_nc[i]**2 + y_nc[i]**2),
        }
    return nc_data


def load_pmt_uids_by_layer(json_path: Path) -> Dict[str, set]:
    """Load PMT UIDs grouped by layer."""
    with open(json_path) as f:
        data = json.load(f)
    by_layer: Dict[str, set] = {'pit': set(), 'bot': set(), 'top': set(), 'wall': set()}
    for entry in data:
        layer = entry['layer'].lower()
        if layer not in by_layer:
            continue
        index = entry['index']
        uid_n = '10' + index
        if len(uid_n) == 8:
            by_layer[layer].add(int(uid_n))
        else:
            uid_o = '1' + index
            if len(uid_o) == 8:
                by_layer[layer].add(int(uid_o))
    return by_layer


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def process_files_diagnostic(
    base_path: Path,
    chunk_size: int,
    is_pmt: bool,
    pmt_json_path: Path,
    areas: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Process SSD or PMT files; collect per-area diagnostic arrays.

    Returns dict: {area: {field: np.ndarray}} with fields:
      nc_source_z, nc_source_r, detection_z, detection_r,
      incidence_angles, time_since_nc, pr, pz_momentum
    """
    pmt_uids_by_layer: Dict[str, set] = {}
    if is_pmt:
        pmt_uids_by_layer = load_pmt_uids_by_layer(pmt_json_path)
        for layer in areas:
            print(f"  {layer} PMT UIDs: {len(pmt_uids_by_layer.get(layer, set()))}")

    collectors = {
        area: {
            'nc_source_z': [], 'nc_source_r': [],
            'detection_z': [], 'detection_r': [],
            'incidence_angles': [],
            'time_since_nc': [], 'pr': [], 'pz_momentum': [],
        }
        for area in areas
    }

    run_dirs = sorted(base_path.glob("run_*"))
    setup_name = "PMT" if is_pmt else "SSD"
    print(f"\nProcessing {setup_name}: {len(run_dirs)} runs, areas={areas}")

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

                    for ci in range(num_chunks):
                        cs = ci * chunk_size
                        ce = min(cs + chunk_size, total)

                        x   = np.array(f['hit']['optical']['x_position_in_m']['pages'][cs:ce],
                                       dtype=np.float32) * 1000
                        y   = np.array(f['hit']['optical']['y_position_in_m']['pages'][cs:ce],
                                       dtype=np.float32) * 1000
                        z   = np.array(f['hit']['optical']['z_position_in_m']['pages'][cs:ce],
                                       dtype=np.float32) * 1000
                        px  = np.array(f['hit']['optical']['x_momentum_direction']['pages'][cs:ce],
                                       dtype=np.float32)
                        py  = np.array(f['hit']['optical']['y_momentum_direction']['pages'][cs:ce],
                                       dtype=np.float32)
                        pz_arr = np.array(f['hit']['optical']['z_momentum_direction']['pages'][cs:ce],
                                          dtype=np.float32)
                        evtid  = f['hit']['optical']['evtid']['pages'][cs:ce]
                        nc_id  = f['hit']['optical']['nC_track_id']['pages'][cs:ce]
                        time   = f['hit']['optical']['time_in_ns']['pages'][cs:ce]
                        det_uid = f['hit']['optical']['det_uid']['pages'][cs:ce]

                        # NC time filter + capture source positions
                        nc_times       = np.full(len(time), np.inf, dtype=np.float32)
                        nc_source_z_a  = np.full(len(time), np.nan, dtype=np.float32)
                        nc_source_r_a  = np.full(len(time), np.nan, dtype=np.float32)
                        for i in range(len(evtid)):
                            key = (evtid[i], nc_id[i])
                            if key in nc_data:
                                nc_times[i]      = nc_data[key]['nC_time']
                                nc_source_z_a[i] = nc_data[key]['nC_z']
                                nc_source_r_a[i] = nc_data[key]['nC_r']

                        tmask = ((nc_times != np.inf) & (time >= nc_times)
                                 & (time <= nc_times + NC_TIME_WINDOW))
                        x  = x[tmask];  y  = y[tmask];  z  = z[tmask]
                        px = px[tmask]; py = py[tmask]; pz_arr = pz_arr[tmask]
                        det_uid        = det_uid[tmask]
                        nc_source_z_a  = nc_source_z_a[tmask]
                        nc_source_r_a  = nc_source_r_a[tmask]
                        t_since_nc = (time[tmask] - nc_times[tmask]).astype(np.float32)

                        # Momentum filter
                        mb   = z <= Z_CUT_BOT
                        mt   = z >= Z_CUT_TOP
                        mbar = ~mb & ~mt
                        fmask = np.zeros(len(z), dtype=bool)
                        fmask[mb] = pz_arr[mb] <= 0
                        fmask[mt] = pz_arr[mt] >= 0
                        if np.any(mbar):
                            fmask[mbar] = checkRadialMomentumVectorized(
                                x[mbar], y[mbar], z[mbar],
                                px[mbar], py[mbar], pz_arr[mbar])

                        x  = x[fmask];  y  = y[fmask];  z  = z[fmask]
                        px = px[fmask]; py = py[fmask]; pz_arr = pz_arr[fmask]
                        det_uid       = det_uid[fmask]
                        nc_source_z_a = nc_source_z_a[fmask]
                        nc_source_r_a = nc_source_r_a[fmask]
                        t_since_nc    = t_since_nc[fmask]
                        pr = np.sqrt(px**2 + py**2)

                        # Per-area collection
                        for area in areas:
                            if is_pmt:
                                amask = np.isin(det_uid, list(pmt_uids_by_layer.get(area, set())))
                            else:
                                amask = det_uid == SSD_UIDS.get(area, -1)
                            if not np.any(amask):
                                continue

                            det_z = z[amask]
                            det_r = np.sqrt(x[amask]**2 + y[amask]**2)
                            src_z = nc_source_z_a[amask]
                            src_r = nc_source_r_a[amask]
                            t_nc  = t_since_nc[amask]
                            pr_a  = pr[amask]
                            pz_a  = pz_arr[amask]

                            collectors[area]['detection_z'].append(det_z)
                            collectors[area]['detection_r'].append(det_r)
                            collectors[area]['nc_source_z'].append(src_z)
                            collectors[area]['nc_source_r'].append(src_r)
                            collectors[area]['time_since_nc'].append(t_nc)
                            collectors[area]['pr'].append(pr_a)
                            collectors[area]['pz_momentum'].append(pz_a)

                            # Incidence angle
                            if area == 'wall':
                                x_a = x[amask]; y_a = y[amask]
                                px_a = px[amask]; py_a = py[amask]
                                r_a = np.maximum(np.sqrt(x_a**2 + y_a**2), 1e-6)
                                cos_theta = np.clip(
                                    np.abs(px_a * (x_a / r_a) + py_a * (y_a / r_a)), 0.0, 1.0)
                            else:
                                cos_theta = np.clip(np.abs(pz_a), 0.0, 1.0)
                            angles = np.degrees(np.arccos(cos_theta))
                            collectors[area]['incidence_angles'].append(
                                np.column_stack([det_z, angles]))

                        del x, y, z, px, py, pz_arr, evtid, nc_id, time, det_uid
                        del nc_times, nc_source_z_a, nc_source_r_a, t_since_nc
                        gc.collect()

            except Exception as e:
                print(f"    Error: {e}")

    print(f"  Total files: {total_files}")

    result = {}
    for area in areas:
        c = collectors[area]
        result[area] = {
            'detection_z':    np.concatenate(c['detection_z'])    if c['detection_z']    else np.array([]),
            'detection_r':    np.concatenate(c['detection_r'])    if c['detection_r']    else np.array([]),
            'nc_source_z':    np.concatenate(c['nc_source_z'])    if c['nc_source_z']    else np.array([]),
            'nc_source_r':    np.concatenate(c['nc_source_r'])    if c['nc_source_r']    else np.array([]),
            'time_since_nc':  np.concatenate(c['time_since_nc'])  if c['time_since_nc']  else np.array([]),
            'pr':             np.concatenate(c['pr'])             if c['pr']             else np.array([]),
            'pz_momentum':    np.concatenate(c['pz_momentum'])    if c['pz_momentum']    else np.array([]),
            'incidence_angles': (np.concatenate(c['incidence_angles'])
                                 if c['incidence_angles']
                                 else np.array([]).reshape(0, 2)),
        }
        print(f"  {area}: {len(result[area]['detection_z']):,} photon hits")

    return result


def collect_all_nc_positions(base_path: Path) -> Dict[str, np.ndarray]:
    """Read ALL NC positions from MyNeutronCaptureOutput (not filtered by detection)."""
    nc_z_list, nc_r_list = [], []
    run_dirs = sorted(base_path.glob("run_*"))
    print(f"\n  Collecting all NC positions from {len(run_dirs)} runs...")
    total_ncs = 0
    for run_dir in run_dirs:
        for hdf5_file in sorted(run_dir.glob("output_t*.hdf5")):
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
    print(f"  Total NCs: {total_ncs:,}")
    return {
        'nc_z': np.concatenate(nc_z_list) if nc_z_list else np.array([]),
        'nc_r': np.concatenate(nc_r_list) if nc_r_list else np.array([]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_diagnostic_plots(ssd_data, pmt_data, output_dir: Path, areas: List[str],
                          ssd_all_ncs=None, pmt_all_ncs=None) -> None:
    z_bins     = np.linspace(Z_CUT_BOT, Z_CUT_TOP, 50)
    z_centers  = (z_bins[:-1] + z_bins[1:]) / 2
    r_bins     = np.linspace(0, R_ZYLINDER + 200, 50)
    r_centers  = (r_bins[:-1] + r_bins[1:]) / 2

    MARKERS = {
        'pit':  [('Skirt r=3800', 3800, 'r')],
        'bot':  [('SSD r=4300', 4300, 'r'), ('Foot r=3950', 3950, 'g')],
        'top':  [('SSD r=4300', 4300, 'r'), ('Cryo neck r=1200', 1200, 'g')],
        'wall': [('SSD r=4300', 4300, 'r')],
    }

    def add_r_markers(ax, area, vertical=True):
        for label, val, col in MARKERS.get(area, []):
            if vertical:
                ax.axvline(val, color=col, linestyle=':', linewidth=1.5, alpha=0.7, label=label)
            else:
                ax.axhline(val, color=col, linestyle=':', linewidth=1.5, alpha=0.7, label=label)

    # ------------------------------------------------------------------
    # General NC plots (unfiltered NC source distributions)
    # ------------------------------------------------------------------
    print("\n  --- General NC distribution plots ---")
    if ssd_all_ncs is not None and pmt_all_ncs is not None:
        for field, bins, xlabel, fname in [
            ('nc_z', z_bins, 'NC source z [mm]', 'all_nc_z'),
            ('nc_r', r_bins, 'NC source r [mm]', 'all_nc_r'),
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
                ax.legend(dict(zip(labels, handles)).values(),
                          dict(zip(labels, handles)).keys())
            else:
                ax.legend()
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Probability density')
            ax.set_title(f'All NCs (unfiltered): {field} distribution')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f"{fname}.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {fname}.png")

    # ------------------------------------------------------------------
    # Per-area plots
    # ------------------------------------------------------------------
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

        # Plot 1: NC source z
        fig, ax = plt.subplots(figsize=(12, 6))
        if len(ssd['nc_source_z']) > 0:
            ax.hist(ssd['nc_source_z'], bins=z_bins, alpha=0.6, density=True,
                    label=f'SSD (N={len(ssd["nc_source_z"]):,})', color='blue')
        if len(pmt['nc_source_z']) > 0:
            ax.hist(pmt['nc_source_z'], bins=z_bins, alpha=0.6, density=True,
                    label=f'PMT (N={len(pmt["nc_source_z"]):,})', color='red')
        ax.set_xlabel('NC source z [mm]'); ax.set_ylabel('Probability density')
        ax.set_title(f'{area.upper()}: Z-distribution of NC sources (detected photons only)')
        ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(output_dir / f"{area}_nc_source_z.png", dpi=150, bbox_inches='tight')
        plt.close(); print(f"    Saved: {area}_nc_source_z.png")

        # Plot 1b: NC source r
        fig, ax = plt.subplots(figsize=(12, 6))
        if len(ssd['nc_source_r']) > 0:
            ax.hist(ssd['nc_source_r'], bins=r_bins, alpha=0.6, density=True,
                    label=f'SSD (N={len(ssd["nc_source_r"]):,})', color='blue')
        if len(pmt['nc_source_r']) > 0:
            ax.hist(pmt['nc_source_r'], bins=r_bins, alpha=0.6, density=True,
                    label=f'PMT (N={len(pmt["nc_source_r"]):,})', color='red')
        add_r_markers(ax, area)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
        ax.set_xlabel('NC source r [mm]'); ax.set_ylabel('Probability density')
        ax.set_title(f'{area.upper()}: R-distribution of NC sources (detected photons only)')
        ax.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(output_dir / f"{area}_nc_source_r.png", dpi=150, bbox_inches='tight')
        plt.close(); print(f"    Saved: {area}_nc_source_r.png")

        # Plot 2: Detection position
        fig, ax = plt.subplots(figsize=(12, 6))
        if area == 'wall':
            if len(ssd['detection_z']) > 0:
                ax.hist(ssd['detection_z'], bins=z_bins, alpha=0.6, density=True,
                        label=f'SSD (N={len(ssd["detection_z"]):,})', color='blue')
            if len(pmt['detection_z']) > 0:
                ax.hist(pmt['detection_z'], bins=z_bins, alpha=0.6, density=True,
                        label=f'PMT (N={len(pmt["detection_z"]):,})', color='red')
            ax.set_xlabel('Detection z [mm]')
            ax.set_title(f'{area.upper()}: Z-distribution of detected photons')
        else:
            if len(ssd['detection_r']) > 0:
                ax.hist(ssd['detection_r'], bins=r_bins, alpha=0.6, density=True,
                        label=f'SSD (N={len(ssd["detection_r"]):,})', color='blue')
            if len(pmt['detection_r']) > 0:
                ax.hist(pmt['detection_r'], bins=r_bins, alpha=0.6, density=True,
                        label=f'PMT (N={len(pmt["detection_r"]):,})', color='red')
            add_r_markers(ax, area)
            ax.set_xlabel('Detection r [mm]')
            ax.set_title(f'{area.upper()}: R-distribution of detected photons')
        ax.set_ylabel('Probability density')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
        ax.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(output_dir / f"{area}_detection_pos.png", dpi=150, bbox_inches='tight')
        plt.close(); print(f"    Saved: {area}_detection_pos.png")

        # Plot 3a: 2D NC source z vs detection z
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        counts_all = []
        for data in [ssd, pmt]:
            src_z, det_z = data['nc_source_z'], data['detection_z']
            valid = ~np.isnan(src_z) & ~np.isnan(det_z)
            if valid.sum() > 0:
                h_tmp, _, _ = np.histogram2d(src_z[valid], det_z[valid], bins=[z_bins, z_bins])
                counts_all.append(h_tmp.max())
        global_vmax = max(counts_all) if counts_all else 1
        for ax, data, title in [(ax1, ssd, 'SSD'), (ax2, pmt, 'PMT')]:
            src_z, det_z = data['nc_source_z'], data['detection_z']
            valid = ~np.isnan(src_z) & ~np.isnan(det_z)
            if valid.sum() > 0:
                h = ax.hist2d(src_z[valid], det_z[valid], bins=[z_bins, z_bins],
                              cmap='inferno', norm=matplotlib.colors.LogNorm(vmin=1, vmax=global_vmax))
                fig.colorbar(h[3], ax=ax, label='Counts')
                ax.plot([Z_CUT_BOT, Z_CUT_TOP], [Z_CUT_BOT, Z_CUT_TOP], 'w--', linewidth=1, alpha=0.5)
            ax.set_xlabel('NC source z [mm]'); ax.set_ylabel('Detection z [mm]')
            ax.set_title(f'{title} (N={valid.sum():,})')
        fig.suptitle(f'{area.upper()}: NC source z vs detection z', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{area}_nc_vs_det_z_2d.png", dpi=150, bbox_inches='tight')
        plt.close(); print(f"    Saved: {area}_nc_vs_det_z_2d.png")

        # Plot 3b: 2D NC source r vs detection r
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        counts_all = []
        for data in [ssd, pmt]:
            src_r, det_r = data['nc_source_r'], data['detection_r']
            valid = ~np.isnan(src_r) & ~np.isnan(det_r)
            if valid.sum() > 0:
                h_tmp, _, _ = np.histogram2d(src_r[valid], det_r[valid], bins=[r_bins, r_bins])
                counts_all.append(h_tmp.max())
        global_vmax_r = max(counts_all) if counts_all else 1
        for ax, data, title in [(ax1, ssd, 'SSD'), (ax2, pmt, 'PMT')]:
            src_r, det_r = data['nc_source_r'], data['detection_r']
            valid = ~np.isnan(src_r) & ~np.isnan(det_r)
            if valid.sum() > 0:
                h = ax.hist2d(src_r[valid], det_r[valid], bins=[r_bins, r_bins],
                              cmap='inferno', norm=matplotlib.colors.LogNorm(vmin=1, vmax=global_vmax_r))
                fig.colorbar(h[3], ax=ax, label='Counts')
                add_r_markers(ax, area, vertical=True)
                add_r_markers(ax, area, vertical=False)
            ax.set_xlabel('NC source r [mm]'); ax.set_ylabel('Detection r [mm]')
            ax.set_title(f'{title} (N={valid.sum():,})')
        fig.suptitle(f'{area.upper()}: NC source r vs detection r', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{area}_nc_vs_det_r_2d.png", dpi=150, bbox_inches='tight')
        plt.close(); print(f"    Saved: {area}_nc_vs_det_r_2d.png")

        # Plot 4: Time since NC
        fig, ax = plt.subplots(figsize=(12, 6))
        t_bins = np.linspace(0, NC_TIME_WINDOW, 100)
        for data, label, color in [(ssd, 'SSD', 'blue'), (pmt, 'PMT', 'red')]:
            n = len(data['time_since_nc'])
            if n > 0:
                mean_t = np.mean(data['time_since_nc'])
                ax.hist(data['time_since_nc'], bins=t_bins, alpha=0.6,
                        weights=np.ones(n) / n,
                        label=f'{label} (N={n:,}, ⟨t⟩={mean_t:.1f} ns)', color=color)
                ax.axvline(mean_t, color=color, linestyle='--', linewidth=1.5)
        ax.set_xlabel('Time since NC [ns]'); ax.set_ylabel('Fraction of photons')
        ax.set_title(f'{area.upper()}: Photon arrival time relative to NC')
        ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(output_dir / f"{area}_time_since_nc.png", dpi=150, bbox_inches='tight')
        plt.close(); print(f"    Saved: {area}_time_since_nc.png")

        # Plot 5: Radial momentum pr
        fig, ax = plt.subplots(figsize=(12, 6))
        pr_bins = np.linspace(0, 1, 80)
        for data, label, color in [(ssd, 'SSD', 'blue'), (pmt, 'PMT', 'red')]:
            n = len(data['pr'])
            if n > 0:
                ax.hist(data['pr'], bins=pr_bins, alpha=0.6,
                        weights=np.ones(n) / n, label=f'{label} (N={n:,})', color=color)
        ax.set_xlabel('p_r = √(p_x² + p_y²)'); ax.set_ylabel('Fraction of photons')
        ax.set_title(f'{area.upper()}: Radial momentum component at detection')
        ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(output_dir / f"{area}_momentum_pr.png", dpi=150, bbox_inches='tight')
        plt.close(); print(f"    Saved: {area}_momentum_pr.png")

        # Plot 6: Vertical momentum pz
        fig, ax = plt.subplots(figsize=(12, 6))
        pz_bins = np.linspace(-1, 1, 80)
        for data, label, color in [(ssd, 'SSD', 'blue'), (pmt, 'PMT', 'red')]:
            n = len(data['pz_momentum'])
            if n > 0:
                ax.hist(data['pz_momentum'], bins=pz_bins, alpha=0.6,
                        weights=np.ones(n) / n, label=f'{label} (N={n:,})', color=color)
        ax.set_xlabel('p_z'); ax.set_ylabel('Fraction of photons')
        ax.set_title(f'{area.upper()}: Vertical momentum component at detection')
        ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(output_dir / f"{area}_momentum_pz.png", dpi=150, bbox_inches='tight')
        plt.close(); print(f"    Saved: {area}_momentum_pz.png")

        # Plot 7: Incidence angle distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        angle_bins = np.linspace(0, 90, 60)
        for data, label, color in [(ssd, 'SSD', 'blue'), (pmt, 'PMT', 'red')]:
            if len(data['incidence_angles']) > 0:
                angles = data['incidence_angles'][:, 1]
                n = len(angles)
                ax.hist(angles, bins=angle_bins, alpha=0.6,
                        weights=np.ones(n) / n, label=f'{label} (N={n:,})', color=color)
        ax.set_xlabel('Incidence angle [deg] (0° = normal)'); ax.set_ylabel('Fraction of photons')
        ax.set_title(f'{area.upper()}: Incidence angle distribution')
        ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(output_dir / f"{area}_incidence_angle.png", dpi=150, bbox_inches='tight')
        plt.close(); print(f"    Saved: {area}_incidence_angle.png")

        # Plot 8: Mean incidence angle vs coordinate
        fig, ax = plt.subplots(figsize=(12, 6))
        if area == 'wall':
            coord_bins, coord_centers, xlabel = z_bins, z_centers, 'Detection z [mm]'
        else:
            coord_bins, coord_centers, xlabel = r_bins, r_centers, 'Detection r [mm]'
            add_r_markers(ax, area)

        for data, label, color in [(ssd, 'SSD', 'blue'), (pmt, 'PMT', 'red')]:
            if len(data['incidence_angles']) == 0:
                continue
            det_coord = (data['incidence_angles'][:, 0] if area == 'wall'
                         else data['detection_r'][:len(data['incidence_angles'])])
            angles = data['incidence_angles'][:, 1]
            mean_angles = [
                np.mean(angles[(det_coord >= coord_bins[i]) & (det_coord < coord_bins[i+1])])
                if np.sum((det_coord >= coord_bins[i]) & (det_coord < coord_bins[i+1])) > 10
                else np.nan
                for i in range(len(coord_bins) - 1)
            ]
            ax.plot(coord_centers, mean_angles, 'o-', color=color,
                    linewidth=2, markersize=4, label=label)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
        ax.set_xlabel(xlabel); ax.set_ylabel('Mean incidence angle [deg]')
        ax.set_title(f'{area.upper()}: Mean incidence angle vs position')
        ax.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(output_dir / f"{area}_mean_angle_vs_pos.png", dpi=150, bbox_inches='tight')
        plt.close(); print(f"    Saved: {area}_mean_angle_vs_pos.png")

        # Plot 9 (wall only): Incidence angle vs z heatmap
        if area == 'wall':
            angle_bins_hm = np.linspace(0, 90, 60)
            z_bins_hm     = np.linspace(Z_CUT_BOT, Z_CUT_TOP, 50)
            counts_hm = []
            for data in [ssd, pmt]:
                if len(data['incidence_angles']) > 0:
                    h_tmp, _, _ = np.histogram2d(
                        data['incidence_angles'][:, 1], data['incidence_angles'][:, 0],
                        bins=[angle_bins_hm, z_bins_hm])
                    counts_hm.append(h_tmp.max())
            global_vmax_hm = max(counts_hm) if counts_hm else 1

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            for ax, data, title in [(ax1, ssd, 'SSD'), (ax2, pmt, 'PMT')]:
                if len(data['incidence_angles']) > 0:
                    h = ax.hist2d(data['incidence_angles'][:, 1], data['incidence_angles'][:, 0],
                                  bins=[angle_bins_hm, z_bins_hm], cmap='inferno',
                                  norm=matplotlib.colors.LogNorm(vmin=1, vmax=global_vmax_hm))
                    fig.colorbar(h[3], ax=ax, label='Counts')
                ax.set_xlabel('Incidence angle [deg] (0° = normal)')
                ax.set_ylabel('Detection z [mm]')
                n_a = len(data['incidence_angles'])
                ax.set_title(f'{title} (N={n_a:,})')
            fig.suptitle('WALL: Incidence angle vs z-position', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / "wall_angle_vs_z_heatmap.png", dpi=150, bbox_inches='tight')
            plt.close(); print(f"    Saved: wall_angle_vs_z_heatmap.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NC source vs. photon detection diagnostic analysis.")
    p.add_argument('--ssd-dir',    type=Path, default=_DEFAULT_SSD_DIR,
                   help="SSD run_* base directory (default: %(default)s)")
    p.add_argument('--pmt-dir',    type=Path, default=_DEFAULT_PMT_DIR,
                   help="PMT run_* base directory (default: %(default)s)")
    p.add_argument('--pmt-json',   type=Path, default=_DEFAULT_PMT_JSON,
                   help="PMT positions JSON file (default: %(default)s)")
    p.add_argument('--output-dir', type=Path, default=_DEFAULT_OUTPUT_DIR,
                   help="Output directory for plots (default: %(default)s)")
    p.add_argument('--areas', nargs='+', default=['pit', 'top', 'wall', 'bot'],
                   metavar='AREA',
                   help="Detector areas to analyze (default: pit top wall bot)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = get_chunk_size()

    print("=" * 70)
    print("NC Source vs Detection Diagnostic Analysis")
    print("=" * 70)
    print(f"SSD dir:    {args.ssd_dir}")
    print(f"PMT dir:    {args.pmt_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Chunk size: {chunk_size}")
    print(f"Memory:     {psutil.virtual_memory().available / (1024**3):.1f} GB available")
    print(f"Areas:      {args.areas}")

    # Collect all NC positions (unfiltered)
    print("\n--- Collecting all NC positions ---")
    ssd_all_ncs = collect_all_nc_positions(args.ssd_dir)
    pmt_all_ncs = collect_all_nc_positions(args.pmt_dir)

    # Process SSD
    ssd_data = process_files_diagnostic(
        args.ssd_dir, chunk_size, is_pmt=False,
        pmt_json_path=args.pmt_json, areas=args.areas)

    # Process PMT
    pmt_data = process_files_diagnostic(
        args.pmt_dir, chunk_size, is_pmt=True,
        pmt_json_path=args.pmt_json, areas=args.areas)

    # Generate plots
    print("\nGenerating diagnostic plots...")
    make_diagnostic_plots(ssd_data, pmt_data, args.output_dir, args.areas,
                          ssd_all_ncs=ssd_all_ncs, pmt_all_ncs=pmt_all_ncs)

    print("\nDone.")


if __name__ == "__main__":
    main()
