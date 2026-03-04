#!/usr/bin/env python3
# optimize_detection_time.py
"""
Optimize photon detection time by comparing different pit/wall distance configurations.

Two-Simulation Workflow:
  Sim1 (NC_DATA_DIR): Muon → Neutron Capture data (MyNeutronCaptureOutput), NO optical photons
  Sim2 (SCAN_DIR / BASELINE_DIR): Gamma vertices → Optical photons on detectors

Matching:
  Sim2 optical/muon_track_id  ==  Sim1 MyNeutronCaptureOutput/evtid
  Sim2 optical/nC_track_id    ==  Sim1 MyNeutronCaptureOutput/nC_track_id

Author: Diagnostic analysis
Date: 2026-02-28
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
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Sim1: NC data (MyNeutronCaptureOutput) ---
NC_DATA_DIR = Path("/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/"
                   "rawMusunNCs/run_001")

# --- Sim2: Optical photon data ---
SCAN_DIR = Path("/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/"
                "optimalDistance/musunNCs")
BASELINE_DIR = Path("/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/rawOpticalHomogeneousPMTsFromMusunNCs/run_001")
PMT_JSON_PATH = Path("/global/cfs/projectdirs/legend/users/tbuerger/sim/data/"
                     "optPhotonSensitiveSurface/homogeneous300PMTpositions_currentDist.json")
OUTPUT_DIR = Path("/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/optimalDistance/musunNCs/timing_analysis")

# Geometry (baseline)
R_ZYLINDER = 4300.0
Z_CUT_BOT = -4979.0
Z_CUT_TOP = 3918.0

# SSD UIDs (not used for PMT analysis, kept for reference)
SSD_UIDS = {'pit': 1966, 'bot': 1967, 'top': 1968, 'wall': 1965}

# NC time window
NC_TIME_WINDOW = 200.0  # ns

# Expected PMT count
EXPECTED_PMTS = 300
EXPECTED_PMTS_WARNING = 292  # Known reduced config, warn but don't abort

# Timing metric: 'mean', 'median', 'p90', 'p95', 'p99'
TIMING_METRIC = 'mean'

# Areas to analyze
ANALYSIS_AREAS = ['pit', 'wall']

# NC detection efficiency analysis
NC_EFFICIENCY_PLOT = True
GE77_MUON_ONLY = True  # If True, only count NCs from muons that produced a Ge-77 NC
MULTIPLICITY_M = 6      # Minimum number of PMTs that must fire
MULTIPLICITY_m = 1      # Minimum photons per PMT to count as "fired"

# Area-restricted multiplicity: if True, only PMTs on the analysis area count
# for the multiplicity condition. Evaluated per ANALYSIS_AREAS entry separately.
AREA_RESTRICTED_MULTIPLICITY = True


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


def checkRadialMomentumVectorized(x, y, z, px, py, pz):
    return (x * px + y * py) >= 0


# =============================================================================
# PMT UID LOADING
# =============================================================================

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


def validate_pmt_count(
    hdf5_file: Path,
    all_pmt_uids: set,
    config_name: str,
) -> int:
    """
    Validate how many expected PMTs are actually hit in a sample file.
    Returns number of matched PMTs.

    Raises RuntimeError if count is neither EXPECTED_PMTS nor EXPECTED_PMTS_WARNING.
    Prints warning if count == EXPECTED_PMTS_WARNING.
    """
    with h5py.File(hdf5_file, 'r') as f:
        det_uids_sample = f['hit']['optical']['det_uid']['pages'][:]
    unique_uids = set(det_uids_sample.tolist())
    matched = len(unique_uids & all_pmt_uids)

    if matched == EXPECTED_PMTS:
        print(f"    PMT check [{config_name}]: {matched}/{EXPECTED_PMTS} PMTs ✓")
    elif matched == EXPECTED_PMTS_WARNING:
        print(f"    ⚠️  WARNING [{config_name}]: Only {matched}/{EXPECTED_PMTS} PMTs detected! "
              f"(known reduced configuration)")
    else:
        raise RuntimeError(
            f"PMT count mismatch in {config_name}: found {matched} PMTs in data, "
            f"expected {EXPECTED_PMTS} (or {EXPECTED_PMTS_WARNING} for reduced config). "
            f"File: {hdf5_file}"
        )
    return matched


# =============================================================================
# NC DATA LOADING FROM SIM1
# =============================================================================

def load_nc_data_from_sim1(nc_data_dir: Path) -> Dict[Tuple[int, int], dict]:
    """
    Load all NC data from Sim1 files (MyNeutronCaptureOutput).

    All Sim2 configurations share the same NC run, so this is loaded once.

    Returns:
        {(evtid, nC_track_id): {'nC_time': float, 'flag_ge77': int, 'evtid': int}}
    """
    print(f"\n  Loading NC data from Sim1: {nc_data_dir}")

    hdf5_files = sorted(nc_data_dir.glob("output_t*.hdf5"))
    if not hdf5_files:
        hdf5_files = sorted(nc_data_dir.glob("output_*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in NC_DATA_DIR: {nc_data_dir}")

    print(f"    Found {len(hdf5_files)} Sim1 files")

    nc_data = {}
    duplicate_count = 0

    for file_idx, hdf5_file in enumerate(hdf5_files):
        if (file_idx + 1) % 50 == 0:
            print(f"    Loading Sim1 file {file_idx + 1}/{len(hdf5_files)}")

        with h5py.File(hdf5_file, 'r') as f:
            nc_out = f['hit']['MyNeutronCaptureOutput']
            nc_evtid = nc_out['evtid']['pages'][:]
            nc_nC_id = nc_out['nC_track_id']['pages'][:]
            nc_time = nc_out['nC_time_in_ns']['pages'][:]
            nc_flag_ge77 = nc_out['nC_flag_Ge77']['pages'][:]

            for idx in range(len(nc_evtid)):
                key = (int(nc_evtid[idx]), int(nc_nC_id[idx]))
                if key in nc_data:
                    duplicate_count += 1
                    continue
                nc_data[key] = {
                    'nC_time': float(nc_time[idx]),
                    'evtid': int(nc_evtid[idx]),
                    'flag_ge77': int(nc_flag_ge77[idx]),
                }

    if duplicate_count > 0:
        print(f"    ⚠️  WARNING: {duplicate_count} duplicate (evtid, nC_id) entries skipped")

    print(f"    ✓ Loaded {len(nc_data)} unique NC events")
    return nc_data


def filter_ge77_muon_ncs(nc_data: Dict[Tuple[int, int], dict]) -> Dict[Tuple[int, int], dict]:
    """
    Filter NC data to only include NCs from muons (events) that produced
    at least one Ge-77 NC.
    """
    ge77_evtids: Set[int] = set()
    for key, val in nc_data.items():
        if val['flag_ge77'] == 1:
            ge77_evtids.add(val['evtid'])

    filtered = {k: v for k, v in nc_data.items() if v['evtid'] in ge77_evtids}
    print(f"    Ge-77 filter: {len(ge77_evtids)} muons with Ge-77, "
          f"{len(filtered)}/{len(nc_data)} NCs retained")
    return filtered


# =============================================================================
# SCAN CONFIGURATION PARSER
# =============================================================================

@dataclass
class ScanConfig:
    name: str
    pit_offset: int  # mm shift in z
    wall_offset: int  # mm shift in r
    data_dir: Path


def parse_scan_configs() -> List[ScanConfig]:
    """Parse all scan directories and build config list."""
    configs = []

    # Baseline: detect whether HDF5 files are directly in dir or in run_* subdirs
    hdf5_direct = sorted(BASELINE_DIR.glob("output_t*.hdf5"))
    run_dirs = sorted(BASELINE_DIR.glob("run_*"))

    if run_dirs:
        if len(run_dirs) > 1:
            print(f"  ⚠️  WARNING: {len(run_dirs)} run dirs in baseline, "
                  f"using latest: {run_dirs[-1].name}")
        baseline_data_dir = run_dirs[-1]
    elif hdf5_direct:
        print(f"  Baseline: HDF5 files found directly in {BASELINE_DIR.name}")
        baseline_data_dir = BASELINE_DIR
    else:
        raise FileNotFoundError(
            f"No run dirs or HDF5 files in baseline: {BASELINE_DIR}")

    configs.append(ScanConfig(
        name="pit+0_wall+0",
        pit_offset=0,
        wall_offset=0,
        data_dir=baseline_data_dir,
    ))

    # Scan directories
    if SCAN_DIR.exists():
        for scan_subdir in sorted(SCAN_DIR.iterdir()):
            if not scan_subdir.is_dir():
                continue
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
                scan_data_dir = run_dirs[-1]
            elif hdf5_direct:
                print(f"  {scan_subdir.name}: HDF5 files found directly")
                scan_data_dir = scan_subdir
            else:
                print(f"  ⚠️  WARNING: No run dirs or HDF5 files in "
                      f"{scan_subdir.name}, skipping")
                continue

            configs.append(ScanConfig(
                name=scan_subdir.name,
                pit_offset=pit_off,
                wall_offset=wall_off,
                data_dir=scan_data_dir,
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
    nc_data: Dict[Tuple[int, int], dict],
    collect_nc_efficiency: bool = False,
    ge77_muon_only: bool = False,
    area_restricted_multiplicity: bool = False,
) -> Dict[str, Dict]:
    """
    Process one scan configuration using two-sim workflow.

    NC data is pre-loaded from Sim1 and matched against Sim2 optical photons
    via (muon_track_id == evtid, nC_track_id == nC_track_id).

    Returns: {
        area: {'time_since_nc': np.ndarray, 'n_photons': int},
        '_nc_efficiency': {  # only if collect_nc_efficiency=True
            'total_ncs': int,
            'detected_ncs': int,
            'efficiency': float,
            # if area_restricted_multiplicity:
            'per_area': {area: {'detected': int, 'efficiency': float}}
        }
    }
    """
    print(f"\n  Processing: {config.name} ({config.data_dir})")

    z_cut_bot = Z_CUT_BOT - config.pit_offset
    z_cut_top = Z_CUT_TOP

    # Prepare NC data (optionally filter for Ge-77 muons)
    if ge77_muon_only:
        nc_data_eff = filter_ge77_muon_ncs(nc_data)
    else:
        nc_data_eff = nc_data

    total_ncs = len(nc_data_eff)

    collectors = {area: [] for area in areas}

    # NC efficiency tracking:
    # {(evtid, nc_id): {pmt_uid: photon_count}}  (all PMTs)
    nc_pmt_hits: Optional[Dict[Tuple[int, int], Dict[int, int]]] = None
    if collect_nc_efficiency:
        nc_pmt_hits = {key: {} for key in nc_data_eff}

    hdf5_files = sorted(config.data_dir.glob("output_t*.hdf5"))
    if not hdf5_files:
        hdf5_files = sorted(config.data_dir.glob("output_*.hdf5"))
    print(f"    Sim2 files: {len(hdf5_files)}")

    if not hdf5_files:
        print(f"    ⚠️  WARNING: No files found, skipping")
        result = {area: {'time_since_nc': np.array([]), 'n_photons': 0}
                  for area in areas}
        if collect_nc_efficiency:
            result['_nc_efficiency'] = _build_nc_efficiency_result(
                nc_pmt_hits, total_ncs, all_pmt_uids, pmt_uids_by_layer,
                areas, area_restricted_multiplicity)
        return result

    # PMT validation on first file
    validate_pmt_count(hdf5_files[0], all_pmt_uids, config.name)

    for file_idx, hdf5_file in enumerate(hdf5_files):
        if (file_idx + 1) % 50 == 0:
            print(f"    File {file_idx + 1}/{len(hdf5_files)}")

        try:
            with h5py.File(hdf5_file, 'r') as f:
                total = len(f['hit']['optical']['x_position_in_m']['pages'])
                num_chunks = (total - 1) // chunk_size + 1

                for chunk_idx in range(num_chunks):
                    cs = chunk_idx * chunk_size
                    ce = min(cs + chunk_size, total)

                    x = np.array(f['hit']['optical']['x_position_in_m']['pages'][cs:ce],
                                 dtype=np.float64) * 1000
                    y = np.array(f['hit']['optical']['y_position_in_m']['pages'][cs:ce],
                                 dtype=np.float64) * 1000
                    z = np.array(f['hit']['optical']['z_position_in_m']['pages'][cs:ce],
                                 dtype=np.float64) * 1000
                    px = np.array(f['hit']['optical']['x_momentum_direction']['pages'][cs:ce],
                                  dtype=np.float64)
                    py = np.array(f['hit']['optical']['y_momentum_direction']['pages'][cs:ce],
                                  dtype=np.float64)
                    pz_arr = np.array(f['hit']['optical']['z_momentum_direction']['pages'][cs:ce],
                                      dtype=np.float64)

                    # Two-sim matching fields
                    muon_ids = np.array(f['hit']['optical']['muon_track_id']['pages'][cs:ce],
                                        dtype=np.int32)
                    nc_ids = np.array(f['hit']['optical']['nC_track_id']['pages'][cs:ce],
                                      dtype=np.int32)
                    time = np.array(f['hit']['optical']['time_in_ns']['pages'][cs:ce],
                                    dtype=np.float64)
                    det_uid = np.array(f['hit']['optical']['det_uid']['pages'][cs:ce],
                                       dtype=np.int32)

                    # NC time lookup: muon_track_id (Sim2) == evtid (Sim1)
                    nc_times = np.full(len(time), np.inf, dtype=np.float64)
                    for idx in range(len(muon_ids)):
                        key = (int(muon_ids[idx]), int(nc_ids[idx]))
                        if key in nc_data:
                            nc_times[idx] = nc_data[key]['nC_time']

                    has_nc = nc_times != np.inf
                    relative_times = np.where(has_nc, time.astype(np.float64) - nc_times, np.inf)

                    # RuntimeError on negative relative times (allow small numerical noise)
                    negative_mask = has_nc & (relative_times < -1.0)
                    if np.any(negative_mask):
                        neg_indices = np.where(negative_mask)[0]
                        first_bad = neg_indices[0]
                        raise RuntimeError(
                            f"FATAL: Negative relative photon time detected!\n"
                            f"  Config: {config.name}, File: {hdf5_file.name}\n"
                            f"  muon_track_id={int(muon_ids[first_bad])}, "
                            f"nC_track_id={int(nc_ids[first_bad])}\n"
                            f"  photon_time={float(time[first_bad]):.4f} ns, "
                            f"nC_time={float(nc_times[first_bad]):.4f} ns\n"
                            f"  relative_time={float(relative_times[first_bad]):.4f} ns\n"
                            f"  {int(np.sum(negative_mask))} photons with negative relative time."
                        )

                    # Time window filter
                    time_mask = has_nc & (relative_times >= -1.0) & (relative_times <= NC_TIME_WINDOW)

                    x = x[time_mask]; y = y[time_mask]; z = z[time_mask]
                    px = px[time_mask]; py = py[time_mask]; pz_arr = pz_arr[time_mask]
                    det_uid_f = det_uid[time_mask]
                    muon_ids_f = muon_ids[time_mask]
                    nc_ids_f = nc_ids[time_mask]
                    t_since_nc = relative_times[time_mask].astype(np.float32)

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
                    muon_ids_f = muon_ids_f[final_mask]
                    nc_ids_f = nc_ids_f[final_mask]
                    t_since_nc = t_since_nc[final_mask]

                    # Per-area timing collection
                    for area in areas:
                        area_uids = pmt_uids_by_layer.get(area, set())
                        area_mask = np.isin(det_uid_f, list(area_uids))
                        if np.any(area_mask):
                            collectors[area].append(t_since_nc[area_mask])

                    # NC efficiency: count photons per PMT per NC (all PMTs)
                    if nc_pmt_hits is not None:
                        pmt_mask = np.isin(det_uid_f, list(all_pmt_uids))
                        if np.any(pmt_mask):
                            evt_pmt = muon_ids_f[pmt_mask]
                            nc_pmt = nc_ids_f[pmt_mask]
                            uid_pmt = det_uid_f[pmt_mask]

                            for i in range(len(evt_pmt)):
                                key = (int(evt_pmt[i]), int(nc_pmt[i]))
                                if key in nc_pmt_hits:
                                    uid = int(uid_pmt[i])
                                    nc_pmt_hits[key][uid] = nc_pmt_hits[key].get(uid, 0) + 1

                    del x, y, z, px, py, pz_arr, muon_ids, nc_ids, time, det_uid
                    del det_uid_f, muon_ids_f, nc_ids_f, nc_times, t_since_nc
                    gc.collect()

        except RuntimeError:
            raise
        except Exception as e:
            print(f"    Error processing {hdf5_file.name}: {e}")
            raise

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
        result['_nc_efficiency'] = _build_nc_efficiency_result(
            nc_pmt_hits, total_ncs, all_pmt_uids, pmt_uids_by_layer,
            areas, area_restricted_multiplicity)

    return result


def _build_nc_efficiency_result(
    nc_pmt_hits: Optional[Dict[Tuple[int, int], Dict[int, int]]],
    total_ncs: int,
    all_pmt_uids: set,
    pmt_uids_by_layer: Dict[str, set],
    areas: List[str],
    area_restricted: bool,
) -> Dict:
    """
    Build NC efficiency result dict.

    If area_restricted=False:
        Count NC as detected if >= MULTIPLICITY_M PMTs (any layer) fired with >= MULTIPLICITY_m photons.
    If area_restricted=True:
        Per area: count NC as detected if >= MULTIPLICITY_M PMTs **on that area** fired.
    """
    if nc_pmt_hits is None:
        return {'total_ncs': 0, 'detected_ncs': 0, 'efficiency': 0.0}

    per_area = {}
    filter_str = "Ge-77 muon NCs" if GE77_MUON_ONLY else "all NCs"

    for area in areas:
        if area == 'pit':
            # Pit is always evaluated with pit-PMTs only
            eligible_uids = pmt_uids_by_layer.get('pit', set())
            restriction_str = "pit PMTs only"
        elif area == 'wall':
            if area_restricted:
                # Wall only
                eligible_uids = pmt_uids_by_layer.get('wall', set())
                restriction_str = "wall PMTs only"
            else:
                # Wall + bot + top
                eligible_uids = (pmt_uids_by_layer.get('wall', set()) |
                                 pmt_uids_by_layer.get('bot', set()) |
                                 pmt_uids_by_layer.get('top', set()))
                restriction_str = "wall+bot+top PMTs"
        else:
            # Fallback for any other area
            eligible_uids = pmt_uids_by_layer.get(area, set())
            restriction_str = f"{area} PMTs only"

        n_detected = 0
        for key, pmt_counts in nc_pmt_hits.items():
            n_fired = sum(
                1 for uid, count in pmt_counts.items()
                if uid in eligible_uids and count >= MULTIPLICITY_m
            )
            if n_fired >= MULTIPLICITY_M:
                n_detected += 1

        eff = n_detected / total_ncs * 100 if total_ncs > 0 else 0.0
        per_area[area] = {'detected': n_detected, 'efficiency': eff}

        print(f"    NC efficiency {area} ({filter_str}): "
              f"{n_detected:,} / {total_ncs:,} = {eff:.2f}% "
              f"(M≥{MULTIPLICITY_M}, m≥{MULTIPLICITY_m}, {restriction_str})")

    return {
        'total_ncs': total_ncs,
        'per_area': per_area,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_timing_optimization(
    all_results: Dict[str, Dict[str, Dict]],
    configs: List[ScanConfig],
    metric: str,
    output_dir: Path,
):
    """Plot timing metric vs configuration for each area."""
    for area in ANALYSIS_AREAS:
        fig, ax = plt.subplots(figsize=(14, 7))

        labels = []
        metric_values = []
        n_photons_list = []

        for config in configs:
            data = all_results.get(config.name, {}).get(area)
            if data is None or data['n_photons'] == 0:
                continue

            val = compute_timing_metric(data['time_since_nc'], metric)
            labels.append(config.name)
            metric_values.append(val)
            n_photons_list.append(data['n_photons'])

        if not metric_values:
            print(f"  No data for {area}, skipping plot")
            continue

        metric_values = np.array(metric_values)
        n_photons_list = np.array(n_photons_list)

        # Find baseline and optimum
        baseline_idx = None
        for i, label in enumerate(labels):
            if label == "pit+0_wall+0":
                baseline_idx = i
        optimum_idx = np.argmin(metric_values)

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

        for i, (bar, val) in enumerate(zip(bars, metric_values)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f} ns', ha='center', va='bottom', fontsize=9,
                    fontweight='bold' if i == optimum_idx else 'normal')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(f'{metric_label(metric)} detection time [ns]')
        ax.set_title(f'{area.upper()}: {metric_label(metric)} photon detection time vs geometry')
        ax.grid(True, alpha=0.3, axis='y')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='orange', edgecolor='black', label='Baseline (original)'),
            Patch(facecolor='green', edgecolor='black', label='Optimum'),
            Patch(facecolor='steelblue', edgecolor='black', label='Scan point'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        photon_text = "  |  ".join([f"{l}: {n:,}" for l, n in zip(labels, n_photons_list)])
        fig.text(0.5, -0.02, f"Detected photons:  {photon_text}",
                 ha='center', fontsize=8, style='italic')

        plt.tight_layout()
        plt.savefig(output_dir / f"{area}_timing_optimization_{metric}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {area}_timing_optimization_{metric}.png")

        # Timing distributions
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
    area_restricted: bool = False,
):
    """Plot NC detection efficiency vs configuration, always per area."""
    for area in ANALYSIS_AREAS:
        _plot_nc_efficiency_single(
            all_results, configs, output_dir, area=area,
            area_restricted=area_restricted)


def _plot_nc_efficiency_single(
    all_results: Dict[str, Dict],
    configs: List[ScanConfig],
    output_dir: Path,
    area: str,
    area_restricted: bool = False,
):
    """Plot NC efficiency for a single area."""
    fig, ax = plt.subplots(figsize=(14, 7))

    labels = []
    efficiencies = []
    detected_list = []
    total_list = []

    for config in configs:
        nc_eff = all_results.get(config.name, {}).get('_nc_efficiency')
        if nc_eff is None:
            continue

        per_area = nc_eff.get('per_area', {})
        area_data = per_area.get(area)
        if area_data is None:
            continue
        labels.append(config.name)
        efficiencies.append(area_data['efficiency'])
        detected_list.append(area_data['detected'])
        total_list.append(nc_eff['total_ncs'])

    if not efficiencies:
        print(f"  No NC efficiency data{f' for {area}' if area else ''}, skipping plot")
        return

    efficiencies = np.array(efficiencies)

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
    mult_str = f"{area.upper()} PMTs only" if area_restricted else "all PMTs for multiplicity"
    if MULTIPLICITY_m != 1:
        print(f"  ⚠️  WARNING: Non-default photon threshold m={MULTIPLICITY_m}")
    ax.set_title(f'{area.upper()}: NC Detection Efficiency ({filter_str}, M≥{MULTIPLICITY_M}, '
                 f'm≥{MULTIPLICITY_m}, {mult_str})')
    ax.grid(True, alpha=0.3, axis='y')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='orange', edgecolor='black', label='Baseline (original)'),
        Patch(facecolor='green', edgecolor='black', label='Optimum'),
        Patch(facecolor='steelblue', edgecolor='black', label='Scan point'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    nc_text = "  |  ".join([f"{l}: {d:,}/{t:,}" for l, d, t
                            in zip(labels, detected_list, total_list)])
    fig.text(0.5, -0.02, f"Detected/Total NCs:  {nc_text}",
             ha='center', fontsize=8, style='italic')

    plt.tight_layout()
    fname = f"nc_efficiency_M{MULTIPLICITY_M}_m{MULTIPLICITY_m}_{area}"
    if GE77_MUON_ONLY:
        fname += "_ge77"
    if area_restricted:
        fname += "_restricted"
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
    print("Detection Time Optimization Analysis (Two-Sim Workflow)")
    print("=" * 70)
    print(f"NC data (Sim1):  {NC_DATA_DIR}")
    print(f"Scan dir (Sim2): {SCAN_DIR}")
    print(f"Baseline (Sim2): {BASELINE_DIR}")
    print(f"Timing metric:   {TIMING_METRIC} ({metric_label(TIMING_METRIC)})")
    print(f"Areas:           {ANALYSIS_AREAS}")
    print(f"Chunk size:      {chunk_size}")
    print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"NC efficiency:   {NC_EFFICIENCY_PLOT}")
    print(f"Ge-77 only:      {GE77_MUON_ONLY}")
    print(f"Multiplicity:    M≥{MULTIPLICITY_M}, m≥{MULTIPLICITY_m}")
    print(f"Area-restricted: {AREA_RESTRICTED_MULTIPLICITY}")

    # Load PMT UIDs
    all_uids, pmt_uids_by_layer = load_pmt_uids(PMT_JSON_PATH)
    print(f"\nTotal PMT UIDs: {len(all_uids)}")
    for layer, uids in pmt_uids_by_layer.items():
        if layer in ANALYSIS_AREAS:
            print(f"  {layer}: {len(uids)} PMTs")

    # Load NC data from Sim1 (shared across all scan configs)
    nc_data = load_nc_data_from_sim1(NC_DATA_DIR)

    # Parse scan configurations
    print("\nParsing scan configurations...")
    configs = parse_scan_configs()
    print(f"Found {len(configs)} configurations:")
    for c in configs:
        print(f"  {c.name}: pit={c.pit_offset:+d}mm, wall={c.wall_offset:+d}mm "
              f"→ {c.data_dir}")

    # Process each configuration
    print("\n" + "=" * 70)
    print("Processing scan data...")
    print("=" * 70)

    all_results = {}
    for config in configs:
        result = process_scan(
            config, chunk_size, pmt_uids_by_layer, all_uids,
            ANALYSIS_AREAS, nc_data,
            collect_nc_efficiency=NC_EFFICIENCY_PLOT,
            ge77_muon_only=GE77_MUON_ONLY,
            area_restricted_multiplicity=AREA_RESTRICTED_MULTIPLICITY,
        )
        all_results[config.name] = result

    # Summary table
    print("\n" + "=" * 70)
    print(f"TIMING SUMMARY ({metric_label(TIMING_METRIC)})")
    print("=" * 70)

    # Header
    header = f"  {'Config':<25} "
    for area in ANALYSIS_AREAS:
        header += f"{'t_' + area + ' [ns]':>15} {'N_' + area:>12} "
    if NC_EFFICIENCY_PLOT:
        for area in ANALYSIS_AREAS:
            header += f"{'eff_' + area + ' [%]':>14} "
    print(header)
    sep_len = 25 + len(ANALYSIS_AREAS) * 29
    if NC_EFFICIENCY_PLOT:
        sep_len += len(ANALYSIS_AREAS) * 16
    print("  " + "-" * sep_len)

    for config in configs:
        data = all_results.get(config.name, {})
        line = f"  {config.name:<25} "
        for area in ANALYSIS_AREAS:
            area_data = data.get(area, {'time_since_nc': np.array([]), 'n_photons': 0})
            val = compute_timing_metric(area_data['time_since_nc'], TIMING_METRIC)
            n = area_data['n_photons']
            line += f"{val:>15.2f} {n:>12,} "
        if NC_EFFICIENCY_PLOT:
            nc_eff = data.get('_nc_efficiency', {})
            per_area = nc_eff.get('per_area', {})
            for area in ANALYSIS_AREAS:
                area_eff = per_area.get(area, {}).get('efficiency', 0.0)
                line += f"{area_eff:>14.2f} "
        print(line)

    # Plots
    print("\nGenerating plots...")
    plot_timing_optimization(all_results, configs, TIMING_METRIC, OUTPUT_DIR)

    if NC_EFFICIENCY_PLOT:
        plot_nc_efficiency(all_results, configs, OUTPUT_DIR,
                           area_restricted=AREA_RESTRICTED_MULTIPLICITY)

    print("\nDone.")


if __name__ == "__main__":
    main()