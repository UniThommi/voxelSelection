#!/usr/bin/env python3
# optimize_detection_time.py
"""
Optimize photon detection time by comparing different pit/wall distance configurations.
Reads PMT simulation data from multiple geometry scans and compares timing distributions.

Author: Diagnostic analysis
Date: 2026-02-27
"""

import h5py
import numpy as np
import gc
import psutil
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

SCAN_DIR = Path("/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/"
                "optimalDistance/homogeneousNCs")
BASELINE_DIR = Path("/pscratch/sd/t/tbuerger/data/proofeDetectionEfficiencies/"
                    "experimentEfficiencyRatio/PMTs_div_SSD_1/ratio_1_PMTs")
PMT_JSON_PATH = Path("/global/cfs/projectdirs/legend/users/tbuerger/sim/data/"
                     "optPhotonSensitiveSurface/homogeneous300PMTpositions_currentDist.json")
OUTPUT_DIR = Path("/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/"
                  "optimalDistance/homogeneousNCs/timing_analysis")


# Geometry (baseline)
R_ZYLINDER = 4300.0
Z_CUT_BOT = -4979.0
Z_CUT_TOP = 3918.0

# SSD UIDs
SSD_UIDS = {'pit': 1966, 'bot': 1967, 'top': 1968, 'wall': 1965}

# NC time window
NC_TIME_WINDOW = 200.0  # ns

# Expected PMT count
EXPECTED_PMTS = 300

# Timing metric: 'mean', 'median', 'p90', 'p95', 'p99'
TIMING_METRIC = 'mean'

# Areas to analyze
ANALYSIS_AREAS = ['pit', 'wall']

# NC detection efficiency analysis
NC_EFFICIENCY_PLOT = True
GE77_MUON_ONLY = False  # If True, only count NCs from muons that produced a Ge-77 NC
MULTIPLICITY_M = 6      # Minimum number of PMTs that must fire
MULTIPLICITY_m = 1      # Minimum photons per PMT to count as "fired"


# =============================================================================
# HELPERS
# =============================================================================

def get_chunk_size() -> int:
    available_mem_gb = psutil.virtual_memory().available / (1024**3)
    if available_mem_gb > 400: return 50000
    elif available_mem_gb > 200: return 30000
    elif available_mem_gb > 50: return 20000
    elif available_mem_gb > 30: return 15000
    elif available_mem_gb > 20: return 10000
    else: return 5000


def compute_timing_metric(times: np.ndarray, metric: str) -> float:
    """Compute timing metric from array of detection times."""
    if len(times) == 0:
        return np.nan
    if metric == 'mean':
        return np.mean(times)
    elif metric == 'median':
        return np.median(times)
    elif metric == 'p90':
        return np.percentile(times, 90)
    elif metric == 'p95':
        return np.percentile(times, 95)
    elif metric == 'p99':
        return np.percentile(times, 99)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use: mean, median, p90, p95, p99")


def metric_label(metric: str) -> str:
    """Human-readable label for timing metric."""
    labels = {
        'mean': 'Mean',
        'median': 'Median',
        'p90': '90th percentile',
        'p95': '95th percentile',
        'p99': '99th percentile',
    }
    return labels.get(metric, metric)


def load_nc_data_dict(f: h5py.File) -> Dict:
    nc_evtid = f['hit']['MyNeutronCaptureOutput']['evtid']['pages'][:]
    nc_nC_id = f['hit']['MyNeutronCaptureOutput']['nC_track_id']['pages'][:]
    nc_time = f['hit']['MyNeutronCaptureOutput']['nC_time_in_ns']['pages'][:]
    nc_flag_ge77 = f['hit']['MyNeutronCaptureOutput']['nC_flag_Ge77']['pages'][:]
    nc_data = {}
    for idx in range(len(nc_evtid)):
        key = (nc_evtid[idx], nc_nC_id[idx])
        nc_data[key] = {
            'nC_time': nc_time[idx],
            'evtid': nc_evtid[idx],
            'flag_ge77': nc_flag_ge77[idx],
        }
    return nc_data


def filter_ge77_muon_ncs(nc_data: Dict) -> Dict:
    """
    Filter NC data to only include NCs from muons (events) that produced
    at least one Ge-77 NC. Returns filtered nc_data dict.
    """
    # Find evtids with at least one Ge-77 NC
    ge77_evtids = set()
    for key, val in nc_data.items():
        if val['flag_ge77'] == 1:
            ge77_evtids.add(val['evtid'])

    # Keep all NCs from those events
    filtered = {k: v for k, v in nc_data.items() if v['evtid'] in ge77_evtids}
    return filtered


def load_pmt_uids(json_path: Path) -> Tuple[set, Dict[str, set]]:
    """Load PMT UIDs and group by layer."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_uids = set()
    uids_by_layer = {'pit': set(), 'bot': set(), 'top': set(), 'wall': set()}
    for entry in data:
        index = entry['index']
        layer = entry['layer'].lower()
        uid_normal = '10' + index
        if len(uid_normal) == 8:
            uid = int(uid_normal)
        else:
            uid_overflow = '1' + index
            if len(uid_overflow) == 8:
                uid = int(uid_overflow)
            else:
                continue
        all_uids.add(uid)
        if layer in uids_by_layer:
            uids_by_layer[layer].add(uid)
    return all_uids, uids_by_layer


def checkRadialMomentumVectorized(x, y, z, px, py, pz):
    return (x * px + y * py) >= 0


# =============================================================================
# SCAN CONFIGURATION PARSER
# =============================================================================

@dataclass
class ScanConfig:
    name: str
    pit_offset: int  # mm shift in z
    wall_offset: int  # mm shift in r
    data_dir: Path
    run_dir: Path


def parse_scan_configs() -> List[ScanConfig]:
    """Parse all scan directories and build config list."""
    configs = []

    run_dirs = sorted(BASELINE_DIR.glob("run_*"))
    hdf5_direct = sorted(BASELINE_DIR.glob("output_t*.hdf5"))
    if run_dirs:
        if len(run_dirs) > 1:
            print(f"  ⚠️  WARNING: {len(run_dirs)} run dirs in baseline, using latest: {run_dirs[-1].name}")
        baseline_run_dir = run_dirs[-1]
    elif hdf5_direct:
        print(f"  Baseline: HDF5 files found directly in {BASELINE_DIR.name}")
        baseline_run_dir = BASELINE_DIR
    else:
        raise FileNotFoundError(f"No run dirs or HDF5 files in baseline: {BASELINE_DIR}")
    configs.append(ScanConfig(
        name="pit+0_wall+0",
        pit_offset=0,
        wall_offset=0,
        data_dir=BASELINE_DIR,
        run_dir=baseline_run_dir,
    ))

    # Scan directories
    if SCAN_DIR.exists():
        for scan_subdir in sorted(SCAN_DIR.iterdir()):
            if not scan_subdir.is_dir():
                continue
            # Parse name: pit+100_wall+500
            match = re.match(r'pit([+-]\d+)_wall([+-]\d+)', scan_subdir.name)
            if not match:
                continue
            pit_off = int(match.group(1))
            wall_off = int(match.group(2))

            run_dirs = sorted(scan_subdir.glob("run_*"))
            hdf5_direct = sorted(scan_subdir.glob("output_t*.hdf5"))
            if run_dirs:
                if len(run_dirs) > 1:
                    print(f"  ⚠️  WARNING: {len(run_dirs)} run dirs in {scan_subdir.name}, "
                          f"using latest: {run_dirs[-1].name}")
                scan_run_dir = run_dirs[-1]
            elif hdf5_direct:
                print(f"  {scan_subdir.name}: HDF5 files found directly")
                scan_run_dir = scan_subdir
            else:
                print(f"  ⚠️  WARNING: No run dirs or HDF5 files in {scan_subdir.name}, skipping")
                continue

            configs.append(ScanConfig(
                name=scan_subdir.name,
                pit_offset=pit_off,
                wall_offset=wall_off,
                data_dir=scan_subdir,
                run_dir=scan_run_dir,
            ))

    return configs


# =============================================================================
# DATA PROCESSING
# =============================================================================

def process_scan(
    config: ScanConfig,
    chunk_size: int,
    pmt_uids_by_layer: Dict[str, set],
    all_pmt_uids: set,
    areas: List[str],
    collect_nc_efficiency: bool = False,
    ge77_muon_only: bool = False,
) -> Dict[str, Dict]:
    """
    Process one scan configuration and collect timing data per area,
    plus optionally NC detection efficiency data.

    Returns: {
        area: {'time_since_nc': np.ndarray, 'n_photons': int},
        '_nc_efficiency': {  # only if collect_nc_efficiency=True
            'total_ncs': int,
            'detected_ncs': int,
            'nc_details': Dict[tuple, Dict]  # per-NC photon counts per PMT
        }
    }
    """
    print(f"\n  Processing: {config.name} ({config.run_dir.name})")

    z_cut_bot = Z_CUT_BOT - config.pit_offset
    z_cut_top = Z_CUT_TOP

    collectors = {area: [] for area in areas}

    # NC efficiency tracking: {(evtid, nc_id): {pmt_uid: photon_count}}
    nc_pmt_hits = {} if collect_nc_efficiency else None
    total_ncs = 0
    ge77_muon_ncs = 0

    hdf5_files = sorted(config.run_dir.glob("output_t*.hdf5"))
    print(f"    Files: {len(hdf5_files)}")

    # PMT sanity check
    if hdf5_files:
        with h5py.File(hdf5_files[0], 'r') as f:
            det_uids_sample = f['hit']['optical']['det_uid']['pages'][:]
            unique_uids = set(det_uids_sample.tolist())
            matched = unique_uids & all_pmt_uids
            print(f"    PMT UID check: {len(matched)} of {len(all_pmt_uids)} expected UIDs "
                  f"found in first file")
            if len(matched) < EXPECTED_PMTS * 0.9:
                print(f"    ⚠️  WARNING: Only {len(matched)}/{EXPECTED_PMTS} PMTs detected!")

    for file_idx, hdf5_file in enumerate(hdf5_files):
        if (file_idx + 1) % 50 == 0:
            print(f"    File {file_idx + 1}/{len(hdf5_files)}")

        try:
            with h5py.File(hdf5_file, 'r') as f:
                nc_data = load_nc_data_dict(f)

                # Count NCs
                if collect_nc_efficiency:
                    if ge77_muon_only:
                        nc_data_filtered = filter_ge77_muon_ncs(nc_data)
                        ge77_muon_ncs += len(nc_data_filtered)
                        total_ncs += len(nc_data)
                        # Use filtered for efficiency, but original for timing
                        nc_data_eff = nc_data_filtered
                    else:
                        nc_data_eff = nc_data
                        total_ncs += len(nc_data)

                    # Initialize hit counters for NCs in this file
                    for key in nc_data_eff:
                        if key not in nc_pmt_hits:
                            nc_pmt_hits[key] = {}

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
                    time = np.array(f['hit']['optical']['time_in_ns']['pages'][cs:ce],
                                    dtype=np.float32)
                    det_uid = f['hit']['optical']['det_uid']['pages'][cs:ce]

                    # NC time filter
                    nc_times = np.full(len(time), np.inf, dtype=np.float32)
                    for idx in range(len(evtid)):
                        key = (evtid[idx], nc_id[idx])
                        if key in nc_data:
                            nc_times[idx] = nc_data[key]['nC_time']

                    time_mask = ((nc_times != np.inf) &
                                 (time >= nc_times) &
                                 (time <= nc_times + NC_TIME_WINDOW))

                    x = x[time_mask]; y = y[time_mask]; z = z[time_mask]
                    px = px[time_mask]; py = py[time_mask]; pz_arr = pz_arr[time_mask]
                    det_uid_f = det_uid[time_mask]
                    evtid_f = evtid[time_mask]
                    nc_id_f = nc_id[time_mask]
                    t_since_nc = (time[time_mask] - nc_times[time_mask])

                    # Momentum filter
                    mask_bot = z <= z_cut_bot
                    mask_top = z >= z_cut_top
                    mask_barrel = ~mask_bot & ~mask_top

                    final_mask = np.zeros_like(z, dtype=bool)
                    final_mask[mask_bot] = pz_arr[mask_bot] <= 0
                    final_mask[mask_top] = pz_arr[mask_top] >= 0
                    if np.any(mask_barrel):
                        final_mask[mask_barrel] = checkRadialMomentumVectorized(
                            x[mask_barrel], y[mask_barrel], z[mask_barrel],
                            px[mask_barrel], py[mask_barrel], pz_arr[mask_barrel])

                    det_uid_f = det_uid_f[final_mask]
                    evtid_f = evtid_f[final_mask]
                    nc_id_f = nc_id_f[final_mask]
                    t_since_nc = t_since_nc[final_mask]

                    # Per-area timing collection
                    for area in areas:
                        area_mask = np.isin(det_uid_f, list(pmt_uids_by_layer.get(area, set())))
                        if np.any(area_mask):
                            collectors[area].append(t_since_nc[area_mask])

                    # NC efficiency: count photons per PMT per NC (all areas combined)
                    if collect_nc_efficiency:
                        # Only count hits on any PMT
                        pmt_mask = np.isin(det_uid_f, list(all_pmt_uids))
                        if np.any(pmt_mask):
                            evt_pmt = evtid_f[pmt_mask]
                            nc_pmt = nc_id_f[pmt_mask]
                            uid_pmt = det_uid_f[pmt_mask]

                            for i in range(len(evt_pmt)):
                                key = (int(evt_pmt[i]), int(nc_pmt[i]))
                                if key in nc_pmt_hits:
                                    uid = int(uid_pmt[i])
                                    nc_pmt_hits[key][uid] = nc_pmt_hits[key].get(uid, 0) + 1

                    del x, y, z, px, py, pz_arr, evtid, nc_id, time, det_uid
                    del det_uid_f, evtid_f, nc_id_f, nc_times, t_since_nc
                    gc.collect()

        except Exception as e:
            print(f"    Error: {e}")

    # Build result
    result = {}
    for area in areas:
        times = np.concatenate(collectors[area]) if collectors[area] else np.array([])
        result[area] = {
            'time_since_nc': times,
            'n_photons': len(times),
        }
        print(f"    {area}: {len(times):,} photons")

    # NC efficiency
    if collect_nc_efficiency:
        n_total = ge77_muon_ncs if ge77_muon_only else total_ncs
        n_detected = 0
        for key, pmt_counts in nc_pmt_hits.items():
            n_fired_pmts = sum(1 for count in pmt_counts.values() if count >= MULTIPLICITY_m)
            if n_fired_pmts >= MULTIPLICITY_M:
                n_detected += 1

        eff = n_detected / n_total * 100 if n_total > 0 else 0.0
        result['_nc_efficiency'] = {
            'total_ncs': n_total,
            'detected_ncs': n_detected,
            'efficiency': eff,
        }
        filter_str = "Ge-77 muon NCs" if ge77_muon_only else "all NCs"
        print(f"    NC efficiency ({filter_str}): {n_detected:,} / {n_total:,} = {eff:.2f}% "
              f"(M≥{MULTIPLICITY_M}, m≥{MULTIPLICITY_m})")

    return result


# =============================================================================
# PLOTTING
# =============================================================================

def plot_timing_optimization(
    all_results: Dict[str, Dict[str, Dict]],
    configs: List[ScanConfig],
    metric: str,
    output_dir: Path,
):
    """
    Plot timing metric vs configuration for each area.

    all_results: {config_name: {area: {time_since_nc, n_photons}}}
    """
    for area in ANALYSIS_AREAS:
        fig, ax = plt.subplots(figsize=(14, 7))

        # Collect data points
        labels = []
        metric_values = []
        n_photons_list = []
        pit_offsets = []
        wall_offsets = []

        for config in configs:
            data = all_results.get(config.name, {}).get(area)
            if data is None or data['n_photons'] == 0:
                continue

            val = compute_timing_metric(data['time_since_nc'], metric)
            labels.append(config.name)
            metric_values.append(val)
            n_photons_list.append(data['n_photons'])
            pit_offsets.append(config.pit_offset)
            wall_offsets.append(config.wall_offset)

        if not metric_values:
            print(f"  No data for {area}, skipping plot")
            continue

        metric_values = np.array(metric_values)
        n_photons_list = np.array(n_photons_list)

        # Find baseline and optimum
        baseline_idx = None
        for i, config in enumerate(configs):
            if config.name in labels:
                idx = labels.index(config.name)
                if config.pit_offset == 0 and config.wall_offset == 0:
                    baseline_idx = idx

        optimum_idx = np.argmin(metric_values)

        # Bar colors
        colors = []
        for i in range(len(labels)):
            if i == baseline_idx:
                colors.append('orange')
            elif i == optimum_idx:
                colors.append('green')
            else:
                colors.append('steelblue')

        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, metric_values, color=colors, alpha=0.8, edgecolor='black')

        # Value labels on bars
        for i, (bar, val) in enumerate(zip(bars, metric_values)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f} ns', ha='center', va='bottom', fontsize=9,
                    fontweight='bold' if i == optimum_idx else 'normal')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(f'{metric_label(metric)} detection time [ns]')
        ax.set_title(f'{area.upper()}: {metric_label(metric)} photon detection time vs geometry')
        ax.grid(True, alpha=0.3, axis='y')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='orange', edgecolor='black', label='Baseline (original)'),
            Patch(facecolor='green', edgecolor='black', label='Optimum'),
            Patch(facecolor='steelblue', edgecolor='black', label='Scan point'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # Photon counts below plot
        photon_text = "  |  ".join([f"{l}: {n:,}" for l, n in zip(labels, n_photons_list)])
        fig.text(0.5, -0.02, f"Detected photons:  {photon_text}",
                 ha='center', fontsize=8, style='italic')

        plt.tight_layout()
        plt.savefig(output_dir / f"{area}_timing_optimization_{metric}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {area}_timing_optimization_{metric}.png")

        # Also plot timing distributions for comparison
        fig, ax = plt.subplots(figsize=(14, 7))
        t_bins = np.linspace(0, NC_TIME_WINDOW, 100)

        for i, config in enumerate(configs):
            data = all_results.get(config.name, {}).get(area)
            if data is None or data['n_photons'] == 0:
                continue

            times = data['time_since_nc']
            weights = np.ones(len(times)) / len(times)
            val = compute_timing_metric(times, metric)

            if config.pit_offset == 0 and config.wall_offset == 0:
                color = 'orange'
                lw = 2.5
            elif i == optimum_idx:
                color = 'green'
                lw = 2.5
            else:
                color = None
                lw = 1.5

            ax.hist(times, bins=t_bins, alpha=0.4, weights=weights,
                    label=f'{config.name} ({metric_label(metric)}={val:.1f} ns)',
                    color=color, linewidth=lw, histtype='step')

        ax.set_xlabel('Time since NC [ns]')
        ax.set_ylabel('Fraction of photons')
        ax.set_title(f'{area.upper()}: Detection time distributions')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{area}_timing_distributions_{metric}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {area}_timing_distributions_{metric}.png")

def plot_nc_efficiency(
    all_results: Dict[str, Dict],
    configs: List[ScanConfig],
    output_dir: Path,
):
    """Plot NC detection efficiency vs configuration."""
    fig, ax = plt.subplots(figsize=(14, 7))

    labels = []
    efficiencies = []
    detected_list = []
    total_list = []

    for config in configs:
        data = all_results.get(config.name, {}).get('_nc_efficiency')
        if data is None:
            continue
        labels.append(config.name)
        efficiencies.append(data['efficiency'])
        detected_list.append(data['detected_ncs'])
        total_list.append(data['total_ncs'])

    if not efficiencies:
        print("  No NC efficiency data, skipping plot")
        return

    efficiencies = np.array(efficiencies)

    # Find baseline and optimum
    baseline_idx = None
    for i, label in enumerate(labels):
        if label == "pit+0_wall+0":
            baseline_idx = i
    optimum_idx = np.argmax(efficiencies)

    colors = []
    for i in range(len(labels)):
        if i == baseline_idx:
            colors.append('orange')
        elif i == optimum_idx:
            colors.append('green')
        else:
            colors.append('steelblue')

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, efficiencies, color=colors, alpha=0.8, edgecolor='black')

    for i, (bar, val) in enumerate(zip(bars, efficiencies)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9,
                fontweight='bold' if i == optimum_idx else 'normal')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('NC detection efficiency [%]')

    filter_str = "Ge-77 muon NCs" if GE77_MUON_ONLY else "all NCs"
    if MULTIPLICITY_m != 1:
        print(f"  ⚠️  WARNING: Non-default photon threshold m={MULTIPLICITY_m}")
    ax.set_title(f'NC Detection Efficiency ({filter_str}, M≥{MULTIPLICITY_M}, m≥{MULTIPLICITY_m})')
    ax.grid(True, alpha=0.3, axis='y')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='orange', edgecolor='black', label='Baseline (original)'),
        Patch(facecolor='green', edgecolor='black', label='Optimum'),
        Patch(facecolor='steelblue', edgecolor='black', label='Scan point'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # NC counts below plot
    nc_text = "  |  ".join([f"{l}: {d:,}/{t:,}" for l, d, t
                            in zip(labels, detected_list, total_list)])
    fig.text(0.5, -0.02, f"Detected/Total NCs:  {nc_text}",
             ha='center', fontsize=8, style='italic')

    plt.tight_layout()
    fname = f"nc_efficiency_M{MULTIPLICITY_M}_m{MULTIPLICITY_m}"
    if GE77_MUON_ONLY:
        fname += "_ge77"
    plt.savefig(output_dir / f"{fname}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunk_size = get_chunk_size()

    print("=" * 70)
    print("Detection Time Optimization Analysis")
    print("=" * 70)
    print(f"Timing metric: {TIMING_METRIC} ({metric_label(TIMING_METRIC)})")
    print(f"Areas: {ANALYSIS_AREAS}")
    print(f"Chunk size: {chunk_size}")
    print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")

    # Load PMT UIDs
    all_uids, pmt_uids_by_layer = load_pmt_uids(PMT_JSON_PATH)
    print(f"Total PMT UIDs: {len(all_uids)}")
    for layer, uids in pmt_uids_by_layer.items():
        if layer in ANALYSIS_AREAS:
            print(f"  {layer}: {len(uids)} PMTs")

    # Parse scan configurations
    print("\nParsing scan configurations...")
    configs = parse_scan_configs()
    print(f"Found {len(configs)} configurations:")
    for c in configs:
        print(f"  {c.name}: pit={c.pit_offset:+d}mm, wall={c.wall_offset:+d}mm "
              f"→ {c.run_dir}")

    # Process each configuration
    print("\n" + "=" * 70)
    print("Processing scan data...")
    print("=" * 70)

    all_results = {}
    for config in configs:
        result = process_scan(config, chunk_size, pmt_uids_by_layer, all_uids,
                              ANALYSIS_AREAS,
                              collect_nc_efficiency=NC_EFFICIENCY_PLOT,
                              ge77_muon_only=GE77_MUON_ONLY)
        all_results[config.name] = result

    # Summary table
    print("\n" + "=" * 70)
    print(f"TIMING SUMMARY ({metric_label(TIMING_METRIC)})")
    print("=" * 70)
    print(f"\n  {'Config':<25} ", end="")
    for area in ANALYSIS_AREAS:
        print(f"{'t_' + area + ' [ns]':>15} {'N_' + area:>12} ", end="")
    if NC_EFFICIENCY_PLOT:
        print(f"{'NC eff [%]':>12} {'det/total':>15}", end="")
    print()
    print("  " + "-" * (25 + len(ANALYSIS_AREAS) * 29 + (29 if NC_EFFICIENCY_PLOT else 0)))

    for config in configs:
        data = all_results.get(config.name, {})
        print(f"  {config.name:<25} ", end="")
        for area in ANALYSIS_AREAS:
            area_data = data.get(area, {'time_since_nc': np.array([]), 'n_photons': 0})
            val = compute_timing_metric(area_data['time_since_nc'], TIMING_METRIC)
            n = area_data['n_photons']
            print(f"{val:>15.2f} {n:>12,} ", end="")
        if NC_EFFICIENCY_PLOT:
            nc_eff = data.get('_nc_efficiency', {})
            eff = nc_eff.get('efficiency', 0.0)
            det = nc_eff.get('detected_ncs', 0)
            tot = nc_eff.get('total_ncs', 0)
            print(f"{eff:>12.2f} {det:>7,}/{tot:,}", end="")
        print()

    # Plots
    print("\nGenerating plots...")
    plot_timing_optimization(all_results, configs, TIMING_METRIC, OUTPUT_DIR)

    if NC_EFFICIENCY_PLOT:
        plot_nc_efficiency(all_results, configs, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()