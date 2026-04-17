#!/usr/bin/env python3
"""zone_ratio_analysis.py

Unified CLI for zone-based SSD/PMT photon detection efficiency ratio analysis.

Modes:
  homogeneous  NC from uniform neutron source simulation (evtid-based NC keys)
  musun        NC from muon-seeded simulation (muon_track_id-based NC keys from CSV)

Usage:
  python zone_ratio_analysis.py --mode homogeneous --ssd-dir /path/ssd --pmt-dir /path/pmt
  python zone_ratio_analysis.py --mode musun --compare ref_zones.json --output-dir ./out
"""

import argparse
import gc
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from ratio_analysis.geometry import GeometryConfig
from ratio_analysis.pmt_data import (
    load_pmt_data, get_pmts_by_layer, build_uid_to_pmt_map, crosscheck_uids,
)
from ratio_analysis.photon_filters import (
    get_chunk_size, checkRadialMomentumVectorized,
    SSD_UID_PIT, SSD_UID_BOT, SSD_UID_TOP, SSD_UID_WALL,
    PMT_CATHODE_RADIUS, MC_SAMPLES,
)
from ratio_analysis.nc_data import (
    load_nc_data_dict_homogeneous, load_nc_data_dict_musun,
    count_nc_from_hdf5, count_nc_from_csv,
)
from ratio_analysis.zones import (
    Zone,
    build_radial_zones, build_z_zones,
    assign_radial_zone, assign_z_zone,
    load_reference_json, build_zones_from_reference,
)
from ratio_analysis.process_ssd import process_all_files_ssd
from ratio_analysis.process_pmt import process_all_files_pmt
from ratio_analysis.plotting import (
    plot_radial_zones, plot_wall_zones, plot_snr_scan,
    plot_comparison, plot_area_flux,
)
from ratio_analysis.io import scan_optimal_zones, save_results_txt, save_results_json


# ---------------------------------------------------------------------------
# Default paths (mode-specific; overridden by CLI args)
# ---------------------------------------------------------------------------
_MUSUN_BASE = Path("/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface")
_MUSUN_DEFAULTS = {
    'ssd_dir':    _MUSUN_BASE / "rawOpticalSSDFromMusunNCS",
    'pmt_dir':    _MUSUN_BASE / "rawOpticalHomogeneousPMTsFromMusunNCs",
    'output_dir': _MUSUN_BASE / "zone_analysis_results",
    'compare':    Path("/pscratch/sd/t/tbuerger/data/proofeDetectionEfficiencies/"
                       "experimentEfficiencyRatio/PMTs_div_SSD_1/zone_analysis/"
                       "zone_ratio_results.json"),
}

_HOMO_BASE = Path("/pscratch/sd/t/tbuerger/data/proofeDetectionEfficiencies/"
                  "experimentEfficiencyRatio/PMTs_div_SSD_1")
_HOMO_DEFAULTS = {
    'ssd_dir':    _HOMO_BASE / "ratio_1_SSD",
    'pmt_dir':    _HOMO_BASE / "ratio_1_PMTs",
    'output_dir': _HOMO_BASE / "zone_analysis",
}

_PMT_JSON_DEFAULT = Path(
    "/global/cfs/projectdirs/legend/users/tbuerger/sim/data/"
    "optPhotonSensitiveSurface/homogeneous300PMTpositions_currentDist.json"
)

FRACTION_TOLERANCE = 0.01
ZONE_SCAN_RANGE    = range(2, 30)


# ---------------------------------------------------------------------------
# Zone-build helpers
# ---------------------------------------------------------------------------

def _build_bot_zone(geometry: GeometryConfig, pmt_by_layer: Dict) -> Tuple:
    bot_bounds = [geometry.r_zyl_bot, geometry.r_zylinder]
    bot_zones = [Zone(
        zone_id=0, area_name="bot",
        boundary_low=geometry.r_zyl_bot, boundary_high=geometry.r_zylinder,
        area_mm2=geometry.area_bot,
        pmt_fractions={pmt.index: 1.0 for pmt in pmt_by_layer['bot']},
        effective_n_pmts=float(len(pmt_by_layer['bot'])),
    )]
    print(f"  bot: 1 zone, {len(pmt_by_layer['bot'])} PMTs, "
          f"area={geometry.area_bot:.0f}mm²")
    return bot_zones, bot_bounds


def _validate_fractions(all_zones: List[Zone], pmts: list) -> None:
    pmt_frac_sums: Dict[str, float] = {}
    for z in all_zones:
        for pmt_idx, frac in (z.pmt_fractions or {}).items():
            pmt_frac_sums[pmt_idx] = pmt_frac_sums.get(pmt_idx, 0.0) + frac

    unassigned = {p.index for p in pmts} - set(pmt_frac_sums.keys())
    if unassigned:
        print(f"  ⚠️  {len(unassigned)} PMTs not assigned to any zone: "
              f"{sorted(unassigned)[:5]}...")

    bad = {i: s for i, s in pmt_frac_sums.items() if abs(s - 1.0) > FRACTION_TOLERANCE}
    if bad:
        print(f"  ⚠️  {len(bad)} PMTs with fraction sum != 1.0 "
              f"(>{FRACTION_TOLERANCE*100:.0f}% off):")
        for idx, s in sorted(bad.items(), key=lambda x: abs(x[1] - 1.0), reverse=True)[:5]:
            print(f"    PMT {idx}: sum = {s:.4f}")
    else:
        print(f"  ✅ All {len(pmt_frac_sums)} assigned PMTs have fraction sums within "
              f"{FRACTION_TOLERANCE*100:.0f}% of 1.0")


# ---------------------------------------------------------------------------
# Chunk-level filter helpers (zone scan inner loop)
# ---------------------------------------------------------------------------

def _filter_ssd_chunk(
    f, cs: int, ce: int,
    primary_field: str, nc_data_dict: Dict,
    geometry: GeometryConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read one SSD HDF5 chunk; apply NC time + momentum filters.

    Returns (x_f, y_f, z_f, uid_f) in mm, after both filters.
    """
    x   = np.array(f['hit']['optical']['x_position_in_m']['pages'][cs:ce],
                   dtype=np.float64) * 1000
    y   = np.array(f['hit']['optical']['y_position_in_m']['pages'][cs:ce],
                   dtype=np.float64) * 1000
    z   = np.array(f['hit']['optical']['z_position_in_m']['pages'][cs:ce],
                   dtype=np.float64) * 1000
    px  = np.array(f['hit']['optical']['x_momentum_direction']['pages'][cs:ce],
                   dtype=np.float64)
    py  = np.array(f['hit']['optical']['y_momentum_direction']['pages'][cs:ce],
                   dtype=np.float64)
    pz_arr = np.array(f['hit']['optical']['z_momentum_direction']['pages'][cs:ce],
                      dtype=np.float64)
    pids = f['hit']['optical'][primary_field]['pages'][cs:ce]
    nids = f['hit']['optical']['nC_track_id']['pages'][cs:ce]
    time = np.array(f['hit']['optical']['time_in_ns']['pages'][cs:ce], dtype=np.float64)
    uid  = f['hit']['optical']['det_uid']['pages'][cs:ce]

    # NC time filter
    nc_times = np.full(len(time), np.inf, dtype=np.float64)
    for i in range(len(pids)):
        key = (int(pids[i]), int(nids[i]))
        if key in nc_data_dict:
            nc_times[i] = nc_data_dict[key]['nC_time']
    tmask = (nc_times != np.inf) & (time >= nc_times) & (time <= nc_times + 200.0)
    x  = x[tmask];  y  = y[tmask];  z  = z[tmask]
    px = px[tmask]; py = py[tmask]; pz_arr = pz_arr[tmask]
    uid = uid[tmask]

    # Momentum (inward-facing) filter
    mb = z <= geometry.z_cut_bot
    mt = z >= geometry.z_cut_top
    mbar = ~mb & ~mt
    fmask = np.zeros(len(z), dtype=bool)
    fmask[mb] = pz_arr[mb] <= 0
    fmask[mt] = pz_arr[mt] >= 0
    if np.any(mbar):
        fmask[mbar] = checkRadialMomentumVectorized(
            x[mbar], y[mbar], z[mbar], px[mbar], py[mbar], pz_arr[mbar])

    return x[fmask], y[fmask], z[fmask], uid[fmask]


def _filter_pmt_chunk(
    f, cs: int, ce: int,
    primary_field: str, nc_data_dict: Dict,
) -> Tuple[Dict[int, int], int]:
    """Read one PMT HDF5 chunk; apply NC time filter.

    Returns ({uid: photon_count}, n_unmatched) for PMT UIDs only.
    n_unmatched counts photons whose (primary_id, nc_id) key was absent from
    nc_data_dict (they carry no NC time and are silently dropped).
    """
    pids = f['hit']['optical'][primary_field]['pages'][cs:ce]
    nids = f['hit']['optical']['nC_track_id']['pages'][cs:ce]
    time = np.array(f['hit']['optical']['time_in_ns']['pages'][cs:ce], dtype=np.float64)
    uid  = f['hit']['optical']['det_uid']['pages'][cs:ce]

    nc_times = np.full(len(time), np.inf, dtype=np.float64)
    n_unmatched = 0
    for i in range(len(pids)):
        key = (int(pids[i]), int(nids[i]))
        if key in nc_data_dict:
            nc_times[i] = nc_data_dict[key]['nC_time']
        else:
            n_unmatched += 1
    tmask = (nc_times != np.inf) & (time >= nc_times) & (time <= nc_times + 200.0)
    uid_f = uid[tmask]
    pmt_mask = (uid_f >= 10_000_000) & (uid_f < 1_000_000_000)
    uid_f = uid_f[pmt_mask]

    counts: Dict[int, int] = {}
    uids, cnts = np.unique(uid_f, return_counts=True)
    for u, c in zip(uids, cnts):
        counts[int(u)] = counts.get(int(u), 0) + int(c)
    return counts, n_unmatched


# ---------------------------------------------------------------------------
# Per-area zone-scan processors
# ---------------------------------------------------------------------------

def _scan_ssd_for_area(
    mode: str,
    ssd_dir: Path,
    area_name: str,
    cfg: Dict,
    zone_configs: Dict[int, List[Zone]],
    bounds_configs: Dict[int, List[float]],
    bot_bounds: List[float],
    geometry: GeometryConfig,
    chunk_size: int,
    bot_counted: bool,
    bot_ssd_counts_scan: Dict[str, int],
) -> Tuple[Dict[int, Dict[str, int]], int]:
    """Single-pass SSD scan for one area across all zone configs."""
    ssd_counts_by_n = {n: {f"{area_name}_{i}": 0 for i in range(n)} for n in zone_configs}
    ssd_scan_nc = 0

    for run_dir in sorted(ssd_dir.glob("run_*")):
        if mode == 'musun':
            nc_csv = run_dir / "merged_ncs.csv"
            if not nc_csv.exists():
                print(f"    ⚠️ No merged_ncs.csv in {run_dir.name}, skipping")
                continue
            ssd_scan_nc += count_nc_from_csv(nc_csv)
            nc_data_dict_run = load_nc_data_dict_musun(nc_csv)
            primary_field = 'muon_track_id'

        for hdf5_file in sorted(run_dir.glob("output_t*.hdf5")):
            if mode == 'homogeneous':
                ssd_scan_nc += count_nc_from_hdf5(hdf5_file)

            try:
                with h5py.File(hdf5_file, 'r') as f:
                    if mode == 'homogeneous':
                        nc_data_dict = load_nc_data_dict_homogeneous(f)
                        primary_field = 'evtid'
                    else:
                        nc_data_dict = nc_data_dict_run

                    total = len(f['hit']['optical']['x_position_in_m']['pages'])
                    num_chunks = (total - 1) // chunk_size + 1

                    for ci in range(num_chunks):
                        cs = ci * chunk_size
                        ce = min(cs + chunk_size, total)
                        x_f, y_f, z_f, uid_f = _filter_ssd_chunk(
                            f, cs, ce, primary_field, nc_data_dict, geometry)

                        if area_name == 'pit':
                            amask = uid_f == SSD_UID_PIT
                            coord = np.sqrt(x_f[amask]**2 + y_f[amask]**2)
                        elif area_name == 'top':
                            amask = uid_f == SSD_UID_TOP
                            coord = np.sqrt(x_f[amask]**2 + y_f[amask]**2)
                        elif area_name == 'wall':
                            amask = uid_f == SSD_UID_WALL
                            coord = z_f[amask]
                        else:
                            continue

                        if not np.any(amask):
                            continue

                        for n in zone_configs:
                            if cfg['type'] == 'radial':
                                zids = assign_radial_zone(coord, bounds_configs[n])
                            else:
                                zids = assign_z_zone(coord, bounds_configs[n])
                            for zi in range(n):
                                ssd_counts_by_n[n][f"{area_name}_{zi}"] += int(
                                    np.sum(zids == zi))

                        if not bot_counted:
                            bmask = uid_f == SSD_UID_BOT
                            if np.any(bmask):
                                r_bot = np.sqrt(x_f[bmask]**2 + y_f[bmask]**2)
                                zids_bot = assign_radial_zone(r_bot, bot_bounds)
                                bot_ssd_counts_scan["bot_0"] += int(
                                    np.sum(zids_bot == 0))

                        del x_f, y_f, z_f, uid_f
                        gc.collect()
            except Exception as e:
                print(f"    Error processing {hdf5_file.name}: {e}")

    return ssd_counts_by_n, ssd_scan_nc


def _scan_pmt_for_area(
    mode: str,
    pmt_dir: Path,
    area_name: str,
    zone_configs: Dict[int, List[Zone]],
    frac_configs: Dict[int, Dict],
    uid_to_pmt: Dict,
    bot_zone_frac: Dict[str, float],
    chunk_size: int,
    bot_counted: bool,
    bot_pmt_counts_scan: Dict[str, float],
) -> Tuple[Dict[int, Dict[str, float]], int]:
    """Single-pass PMT scan for one area across all zone configs."""
    pmt_counts_by_n = {n: {f"{area_name}_{i}": 0.0 for i in range(n)} for n in zone_configs}
    pmt_scan_nc = 0

    for run_dir in sorted(pmt_dir.glob("run_*")):
        if mode == 'musun':
            nc_csv = run_dir / "merged_ncs.csv"
            if not nc_csv.exists():
                print(f"    ⚠️ No merged_ncs.csv in {run_dir.name}, skipping")
                continue
            pmt_scan_nc += count_nc_from_csv(nc_csv)
            nc_data_dict_run = load_nc_data_dict_musun(nc_csv)
            primary_field = 'muon_track_id'

        for hdf5_file in sorted(run_dir.glob("output_t*.hdf5")):
            if mode == 'homogeneous':
                pmt_scan_nc += count_nc_from_hdf5(hdf5_file)

            try:
                pmt_photon_counts: Dict[int, int] = {}
                with h5py.File(hdf5_file, 'r') as f:
                    if mode == 'homogeneous':
                        nc_data_dict = load_nc_data_dict_homogeneous(f)
                        primary_field = 'evtid'
                    else:
                        nc_data_dict = nc_data_dict_run

                    total = len(f['hit']['optical']['x_position_in_m']['pages'])
                    num_chunks = (total - 1) // chunk_size + 1
                    for ci in range(num_chunks):
                        cs = ci * chunk_size
                        ce = min(cs + chunk_size, total)
                        chunk_cnts, _ = _filter_pmt_chunk(
                            f, cs, ce, primary_field, nc_data_dict)
                        for u, c in chunk_cnts.items():
                            pmt_photon_counts[u] = pmt_photon_counts.get(u, 0) + c

                # Distribute fractionally for each N
                for n in zone_configs:
                    for uid_int, n_ph in pmt_photon_counts.items():
                        if uid_int not in uid_to_pmt:
                            continue
                        pmt = uid_to_pmt[uid_int]
                        if pmt.layer != area_name:
                            continue
                        for zone_key, pmt_fracs in frac_configs[n].items():
                            if pmt.index in pmt_fracs:
                                pmt_counts_by_n[n][zone_key] += n_ph * pmt_fracs[pmt.index]

                # Count bot PMT photons once (during first area scan)
                if not bot_counted:
                    for uid_int, n_ph in pmt_photon_counts.items():
                        if uid_int not in uid_to_pmt:
                            continue
                        pmt = uid_to_pmt[uid_int]
                        if pmt.layer != 'bot':
                            continue
                        if pmt.index in bot_zone_frac:
                            bot_pmt_counts_scan["bot_0"] += n_ph * bot_zone_frac[pmt.index]

            except Exception as e:
                print(f"    Error processing {hdf5_file.name}: {e}")

    return pmt_counts_by_n, pmt_scan_nc


# ---------------------------------------------------------------------------
# Zone scan orchestrator
# ---------------------------------------------------------------------------

def _run_zone_scan(
    mode: str,
    ssd_dir: Path,
    pmt_dir: Path,
    geometry: GeometryConfig,
    chunk_size: int,
    pmt_by_layer: Dict,
    uid_to_pmt: Dict,
    zone_scan_areas: List[str],
    n_pit: int,
    n_top: int,
    n_wall: int,
    bot_bounds: List[float],
    bot_zone_frac: Dict[str, float],
    snr_threshold: float,
    min_pmts_per_zone: int,
) -> Tuple[Dict, Dict, Dict[str, int], Dict[str, float]]:
    """Run zone scan for each requested area.

    Returns:
        (optimal_zones, scan_results, bot_ssd_counts_scan, bot_pmt_counts_scan)
        where optimal_zones contains:
          - optimal_zones['pit' | 'top' | 'wall'] = optimal N
          - optimal_zones['{area}_ssd_counts'] = ssd_counts dict for optimal N
          - optimal_zones['{area}_pmt_counts'] = pmt_counts dict for optimal N
          - optimal_zones['{area}_ssd_nc'] / '_pmt_nc'
    """
    all_area_cfgs = {
        'pit':  {'r_min': 0.0,               'r_max': geometry.r_pit,      'type': 'radial',
                 'pmts': pmt_by_layer['pit']},
        'top':  {'r_min': geometry.r_zyl_top, 'r_max': geometry.r_zylinder, 'type': 'radial',
                 'pmts': pmt_by_layer['top']},
        'wall': {'z_min': geometry.z_cut_bot, 'z_max': geometry.z_cut_top,  'type': 'z',
                 'pmts': pmt_by_layer['wall']},
    }
    scan_areas = {k: v for k, v in all_area_cfgs.items() if k in zone_scan_areas}

    optimal_zones: Dict = {}
    scan_results:  Dict[str, Dict] = {}
    bot_ssd_counts_scan: Dict[str, int]   = {"bot_0": 0}
    bot_pmt_counts_scan: Dict[str, float] = {"bot_0": 0.0}
    bot_counted = False

    for area_name, cfg in scan_areas.items():
        # Build zone configs for each N in scan range
        zone_configs:  Dict[int, List[Zone]] = {}
        bounds_configs: Dict[int, List[float]] = {}
        frac_configs:   Dict[int, Dict]        = {}

        for n in ZONE_SCAN_RANGE:
            n_pmts = len(cfg['pmts'])
            if n_pmts / n < min_pmts_per_zone:
                print(f"    Skipping N={n}+ for {area_name}: "
                      f"{n_pmts} PMTs / {n} zones < {min_pmts_per_zone}")
                break
            if cfg['type'] == 'radial':
                zones_n, bounds_n = build_radial_zones(
                    cfg['pmts'], n, cfg['r_min'], cfg['r_max'], area_name)
            else:
                zones_n, bounds_n = build_z_zones(
                    cfg['pmts'], n, cfg['z_min'], cfg['z_max'],
                    area_name, geometry.r_zylinder)
            zone_configs[n]  = zones_n
            bounds_configs[n] = bounds_n
            frac_configs[n]  = {f"{area_name}_{z.zone_id}": z.pmt_fractions
                                for z in zones_n}

        print(f"\n  Scanning {area_name}: processing SSD files...")
        ssd_counts_by_n, ssd_scan_nc = _scan_ssd_for_area(
            mode, ssd_dir, area_name, cfg, zone_configs, bounds_configs,
            bot_bounds, geometry, chunk_size, bot_counted, bot_ssd_counts_scan)

        print(f"  Scanning {area_name}: processing PMT files...")
        pmt_counts_by_n, pmt_scan_nc = _scan_pmt_for_area(
            mode, pmt_dir, area_name, zone_configs, frac_configs,
            uid_to_pmt, bot_zone_frac, chunk_size, bot_counted, bot_pmt_counts_scan)

        optimal_n, scan_data = scan_optimal_zones(
            ssd_counts_by_n, pmt_counts_by_n,
            ssd_scan_nc, pmt_scan_nc,
            area_name, zone_configs, snr_threshold, min_pmts_per_zone,
        )
        optimal_zones[area_name] = optimal_n
        scan_results[area_name]  = scan_data
        optimal_zones[f"{area_name}_ssd_counts"] = ssd_counts_by_n[optimal_n]
        optimal_zones[f"{area_name}_pmt_counts"] = pmt_counts_by_n[optimal_n]
        optimal_zones[f"{area_name}_ssd_nc"]     = ssd_scan_nc
        optimal_zones[f"{area_name}_pmt_nc"]     = pmt_scan_nc

        if not bot_counted:
            bot_counted = True

    return optimal_zones, scan_results, bot_ssd_counts_scan, bot_pmt_counts_scan


# ---------------------------------------------------------------------------
# Per-PMT-position ratio analysis
# ---------------------------------------------------------------------------

def _accumulate_pmt_hits_per_uid(
    mode: str,
    pmt_dir: Path,
    chunk_size: int,
) -> Tuple[Dict[int, int], int, int]:
    """Scan all PMT run directories and accumulate hit counts per detector UID.

    Applies the same NC time-window filter (hit time in [NC_time, NC_time+200 ns])
    as the main zone-scan pipeline by reusing _filter_pmt_chunk.

    Opens each HDF5 file only once: NC data dict and photon hits are both read
    within the same ``with h5py.File(...)`` block.

    Args:
        mode:       'homogeneous' or 'musun'
        pmt_dir:    base directory that contains run_* subdirectories
        chunk_size: number of rows per HDF5 read chunk

    Returns:
        (uid_counts, total_nc, n_unmatched)
        uid_counts:  {uid: total_photon_count} (only UIDs with >= 1 time-filtered hit)
        total_nc:    number of NC events across all processed files
        n_unmatched: photons whose (primary_id, nc_id) key was absent from nc_data_dict
    """
    uid_counts: Dict[int, int] = {}
    total_nc: int = 0
    n_unmatched: int = 0

    for run_dir in sorted(pmt_dir.glob("run_*")):
        if mode == 'musun':
            nc_csv = run_dir / "merged_ncs.csv"
            if not nc_csv.exists():
                print(f"    [per-PMT] No merged_ncs.csv in {run_dir.name}, skipping")
                continue
            nc_data_dict_run = load_nc_data_dict_musun(nc_csv)
            total_nc += len(nc_data_dict_run)
            primary_field = 'muon_track_id'

        for hdf5_file in sorted(run_dir.glob("output_t*.hdf5")):
            try:
                with h5py.File(hdf5_file, 'r') as f:
                    if mode == 'homogeneous':
                        # Load NC dict and photon hits in a single open
                        nc_data_dict = load_nc_data_dict_homogeneous(f)
                        primary_field = 'evtid'
                        total_nc += len(
                            f['hit']['MyNeutronCaptureOutput']['evtid']['pages'])
                    else:
                        nc_data_dict = nc_data_dict_run

                    total = len(f['hit']['optical']['x_position_in_m']['pages'])
                    num_chunks = (total - 1) // chunk_size + 1
                    for ci in range(num_chunks):
                        cs = ci * chunk_size
                        ce = min(cs + chunk_size, total)
                        chunk_counts, chunk_unmatched = _filter_pmt_chunk(
                            f, cs, ce, primary_field, nc_data_dict)
                        for uid, cnt in chunk_counts.items():
                            uid_counts[uid] = uid_counts.get(uid, 0) + cnt
                        n_unmatched += chunk_unmatched
            except Exception as e:
                print(f"    [per-PMT] Error in {hdf5_file.name}: {e}")

    return uid_counts, total_nc, n_unmatched


def _load_ssd_voxel_hits_from_postprocessed(
    ssd_postprocessed_file: Path,
) -> Tuple[Dict[str, int], int]:
    """Sum target_matrix over all NC rows to get total photon hits per voxel.

    The postprocessed file is assumed to already carry the NC time-window
    filter applied during simPostProcessing -- no additional filtering is done.
    Reads in batches of 1 000 NC rows to limit peak memory.
    Accumulates in float64 (from int32 on disk) to avoid overflow at large row counts.

    Args:
        ssd_postprocessed_file: path to ncscore_output_*.hdf5

    Returns:
        ({voxel_id: total_hits_over_all_NCs}, n_ncs)
    """
    with h5py.File(ssd_postprocessed_file, 'r') as f:
        voxel_ids: List[str] = [
            c.decode() if isinstance(c, bytes) else str(c)
            for c in f['target_columns'][:]
        ]
        mat = f['target_matrix']
        n_ncs, n_vox = mat.shape
        col_sums = np.zeros(n_vox, dtype=np.float64)
        _BATCH = 1000
        for rs in range(0, n_ncs, _BATCH):
            re = min(rs + _BATCH, n_ncs)
            col_sums += np.array(mat[rs:re, :], dtype=np.float64).sum(axis=0)

    return {vid: int(round(s)) for vid, s in zip(voxel_ids, col_sums)}, n_ncs


def _plot_per_pmt_delta_histogram(
    all_deltas: List[float],
    area_deltas: Dict[str, List[float]],
    area_ratios: Dict[str, float],
    out_path: Path,
) -> None:
    """Two-panel figure: histogram of delta_hits coloured by area (left) and
    per-area mean +/- std bar chart (right).

    delta_hits = (SSD hits / area_ratio) - PMT hits.
    A value of 0 means perfect calibration; the representative voxel
    (chosen as the median-SSD-hit pair) has delta = 0 by construction.
    """
    area_colors = {
        'pit':  'royalblue',
        'bot':  'darkcyan',
        'top':  'firebrick',
        'wall': 'forestgreen',
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: stacked histogram coloured by area
    ax = axes[0]
    bins = np.histogram_bin_edges(all_deltas, bins=30)
    for area in ['pit', 'bot', 'top', 'wall']:
        if area_deltas.get(area):
            ax.hist(
                area_deltas[area], bins=bins,
                color=area_colors[area], alpha=0.6,
                label=f"{area}  (N={len(area_deltas[area])})",
                edgecolor='black', linewidth=0.4,
            )
    ax.axvline(0,
               color='black', linestyle='--', linewidth=1.5,
               label='delta = 0  (perfect match)')
    ax.axvline(np.mean(all_deltas),
               color='darkorange', linestyle='-', linewidth=1.5,
               label=f'Global mean = {np.mean(all_deltas):.1f}')
    ax.set_xlabel("delta_hits = (SSD hits / area ratio) - PMT hits  [photons]", fontsize=11)
    ax.set_ylabel("Number of PMT positions", fontsize=11)
    ax.set_title(
        "Per-PMT hit deviation after per-area calibration\n"
        "(representative voxel = closest to median SSD hits; delta = 0 by construction)",
        fontsize=10,
    )
    ax.legend(fontsize=8)

    # Right: per-area mean +/- std bar chart
    ax = axes[1]
    valid_areas = [a for a in ['pit', 'bot', 'top', 'wall'] if area_deltas.get(a)]
    means  = [np.mean(area_deltas[a]) for a in valid_areas]
    stds   = [np.std(area_deltas[a])  for a in valid_areas]
    colors = [area_colors[a]          for a in valid_areas]
    x = np.arange(len(valid_areas))
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.75,
           edgecolor='black', capsize=6, error_kw={'linewidth': 1.5})
    ax.axhline(0, color='black', linestyle='--', linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{a}\n(ratio = {area_ratios.get(a, float('nan')):.3f})" for a in valid_areas],
        fontsize=9,
    )
    ax.set_ylabel("Mean delta_hits  [photons]  (mean +/- std)", fontsize=11)
    ax.set_title("Per-area mean hit deviation after calibration", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def analyze_per_pmt_ratio(
    mode: str,
    pmt_dir: Path,
    ssd_postprocessed_file: Path,
    pmt_json: Path,
    geometry: GeometryConfig,
    chunk_size: int,
    output_dir: Path,
) -> None:
    """Per-PMT-position calibration ratio analysis (independent of zone scan).

    Derives one efficiency ratio per detector area from the median-hit
    representative voxel/PMT pair, scales all SSD voxel hits in the area by
    that ratio, then measures the residual hit deviation at every PMT position.

    Algorithm
    ---------
    1. Load 300 PMT positions (JSON) -> UID -> voxel_id mapping via pmt.index.
    2. Scan raw PMT sim files with the NC time-window filter (+/-200 ns around
       each NC event, same filter as the main pipeline) -> per-UID photon count.
    3. Load SSD postprocessed target_matrix (already time-filtered during
       simPostProcessing) -> sum over NC rows -> per-voxel photon count.
    4. For each area (pit / bot / top / wall):
         a. Select the voxel/PMT pair whose SSD hit count is closest to the
            area median -> the "representative" pair.
         b. ratio_area = ssd_hits[repr] / pmt_hits[repr]
            Note: ratio_area < 1 if the SSD surface collected fewer photons
            than the PMT at the same position.
    5. For every PMT position in the area:
            scaled_hits = ssd_hits / ratio_area
            delta_hits  = scaled_hits - pmt_hits
       By construction, delta = 0 for the representative voxel; all other
       values show the residual non-uniformity within the area.
    6. Plot a histogram of delta_hits over all ~300 PMT-matched voxels.

    Assumptions / uncertainties
    ---------------------------
    * pmt.index (from the JSON) == voxel_id in SSD target_columns.
      Both derive from the same det_uid by stripping the '10' / '1' prefix
      (consistent between pmt_data.py and compare_pmt_ssd.py).
    * The SSD postprocessed target_matrix carries the same 200 ns time window
      as the PMT filtering applied here (confirmed by user).
    * PMT positions with pmt_hits == 0 in the representative voxel produce an
      undefined ratio and the area is excluded with a warning.

    Args:
        mode:                   'homogeneous' or 'musun'
        pmt_dir:                raw PMT simulation base directory (run_* subdirs)
        ssd_postprocessed_file: ncscore_output_*.hdf5 with target_matrix
        pmt_json:               JSON file listing the 300 PMT positions
        geometry:               GeometryConfig (kept for API symmetry; not used here)
        chunk_size:             HDF5 read chunk size (rows per chunk)
        output_dir:             directory where the output plot is saved
    """
    print("\n" + "=" * 80)
    print("Per-PMT-position ratio analysis")
    print("=" * 80)
    print(f"  Mode:               {mode}")
    print(f"  PMT dir:            {pmt_dir}")
    print(f"  SSD postprocessed:  {ssd_postprocessed_file}")
    print(f"  Output dir:         {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1 -- PMT positions
    print("\n[1/4] Loading PMT positions...")
    pmts       = load_pmt_data(pmt_json)
    uid_to_pmt = build_uid_to_pmt_map(pmts)
    print(f"  {len(pmts)} PMTs loaded, {len(uid_to_pmt)} UID mappings")

    # 2 -- PMT hit counts (NC time filter)
    print("\n[2/4] Accumulating PMT hits per position (NC time filter)...")
    pmt_uid_hits, pmt_nc, n_unmatched = _accumulate_pmt_hits_per_uid(
        mode, pmt_dir, chunk_size)
    n_with_hits = sum(1 for v in pmt_uid_hits.values() if v > 0)
    print(f"  {len(pmt_uid_hits)} UIDs observed; {n_with_hits} with hits > 0")
    print(f"  PMT simulation: {pmt_nc:,} NC events")
    if n_unmatched:
        print(f"  WARNING: {n_unmatched:,} photons had no NC match in nc_data_dict "
              f"(dropped by time filter)")

    # 3 -- SSD postprocessed voxel hits
    print("\n[3/4] Loading SSD postprocessed voxel hits...")
    ssd_voxel_hits, ssd_nc = _load_ssd_voxel_hits_from_postprocessed(
        ssd_postprocessed_file)
    print(f"  {len(ssd_voxel_hits)} voxels; "
          f"total hits = {sum(ssd_voxel_hits.values()):,}")
    print(f"  SSD postprocessed: {ssd_nc:,} NC events")

    if pmt_nc != ssd_nc:
        raise ValueError(
            f"NC count mismatch: PMT simulation has {pmt_nc:,} NCs but "
            f"SSD postprocessed file has {ssd_nc:,} NCs. "
            "Ensure both datasets were produced from the same simulation run."
        )

    # 4 -- Build per-position records and per-area ratios
    print("\n[4/4] Computing per-area calibration ratios and delta hits...")

    records: List[Dict] = []
    n_no_pmt = 0
    n_no_ssd = 0
    for uid, pmt in uid_to_pmt.items():
        p_hits = pmt_uid_hits.get(uid, 0)
        s_hits = ssd_voxel_hits.get(pmt.index, 0)
        if uid not in pmt_uid_hits:
            n_no_pmt += 1
        if pmt.index not in ssd_voxel_hits:
            n_no_ssd += 1
        records.append({
            'uid':      uid,
            'voxel_id': pmt.index,
            'layer':    pmt.layer,
            'pmt_hits': p_hits,
            'ssd_hits': s_hits,
        })
    if n_no_pmt:
        print(f"  INFO: {n_no_pmt} PMT positions with no time-filtered photons in sim "
              f"(normal for low-rate NCs)")
    if n_no_ssd:
        print(f"  WARNING: {n_no_ssd} PMT positions have no matching voxel in the "
              f"SSD postprocessed file (voxel_id not in target_columns)")

    # Per-area calibration: median representative voxel
    areas = ['pit', 'bot', 'top', 'wall']
    area_ratios: Dict[str, float] = {}

    print(f"\n  {'Area':<6} {'Repr voxel':<14} {'SSD hits':>10} "
          f"{'PMT hits':>10} {'Ratio':>8}  (ratio < 1 means SSD < PMT)")
    print("  " + "-" * 62)

    for area in areas:
        area_recs = [r for r in records if r['layer'] == area]
        if not area_recs:
            print(f"  {area:<6}  -- no matched voxels in this area")
            area_ratios[area] = float('nan')
            continue

        ssd_vals   = np.array([r['ssd_hits'] for r in area_recs], dtype=float)
        median_ssd = np.median(ssd_vals)
        # Voxel closest to median SSD hit count -> representative
        idx_repr   = int(np.argmin(np.abs(ssd_vals - median_ssd)))
        repr_rec   = area_recs[idx_repr]

        if repr_rec['pmt_hits'] == 0:
            print(f"  {area:<6}  WARNING: representative PMT has 0 photon hits -- "
                  f"ratio undefined, area skipped")
            area_ratios[area] = float('nan')
            continue

        ratio = repr_rec['ssd_hits'] / repr_rec['pmt_hits']
        area_ratios[area] = ratio
        print(f"  {area:<6} {repr_rec['voxel_id']:<14} "
              f"{repr_rec['ssd_hits']:>10d} {repr_rec['pmt_hits']:>10d} {ratio:>8.4f}")

    # Compute delta_hits for every PMT position with a valid area ratio
    all_deltas:  List[float]            = []
    area_deltas: Dict[str, List[float]] = {a: [] for a in areas}

    for r in records:
        ratio = area_ratios.get(r['layer'], float('nan'))
        if np.isnan(ratio) or ratio == 0.0:
            continue
        delta = r['ssd_hits'] / ratio - r['pmt_hits']
        all_deltas.append(delta)
        area_deltas[r['layer']].append(delta)

    if not all_deltas:
        print("\n  WARNING: No valid delta values -- check input paths and data.")
        return

    print(f"\n  Delta-hits summary over {len(all_deltas)} PMT positions:")
    print(f"    Global mean   = {np.mean(all_deltas):+.2f}")
    print(f"    Global std    = {np.std(all_deltas):.2f}")
    print(f"    Global median = {np.median(all_deltas):+.2f}")
    print(f"    Min / Max     = {np.min(all_deltas):+.1f} / {np.max(all_deltas):+.1f}")
    for area in areas:
        if area_deltas[area]:
            print(f"    {area:<5}  mean={np.mean(area_deltas[area]):+.2f}  "
                  f"std={np.std(area_deltas[area]):.2f}  N={len(area_deltas[area])}")

    out_path = output_dir / "per_pmt_delta_hits.png"
    _plot_per_pmt_delta_histogram(all_deltas, area_deltas, area_ratios, out_path)
    print(f"\n  Saved: {out_path}")
    print("Per-PMT ratio analysis done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zone-based SSD/PMT photon detection efficiency ratio analysis."
    )
    p.add_argument('--mode', required=True, choices=['homogeneous', 'musun'],
                   help="Simulation mode")
    p.add_argument('--ssd-dir', type=Path, default=None,
                   help="SSD run_* base directory (default: mode-specific)")
    p.add_argument('--pmt-dir', type=Path, default=None,
                   help="PMT run_* base directory (default: mode-specific)")
    p.add_argument('--pmt-json', type=Path, default=_PMT_JSON_DEFAULT,
                   help="PMT positions JSON file")
    p.add_argument('--output-dir', type=Path, default=None,
                   help="Output directory for results/plots (default: mode-specific)")
    p.add_argument('--n-pit',  type=int, default=5, help="Initial pit zones (default: 5)")
    p.add_argument('--n-top',  type=int, default=4, help="Initial top zones (default: 4)")
    p.add_argument('--n-wall', type=int, default=8, help="Initial wall zones (default: 8)")
    p.add_argument('--zone-scan', nargs='*', default=['pit', 'top', 'wall'],
                   metavar='AREA',
                   help="Areas to zone-scan for optimal N "
                        "(default: pit top wall; pass empty list to disable)")
    p.add_argument('--snr-threshold', type=float, default=3.0,
                   help="Minimum SNR for zone scan (default: 3.0)")
    p.add_argument('--min-pmts', type=int, default=8,
                   help="Minimum effective PMTs per zone in scan (default: 8)")
    p.add_argument('--compare', type=Path, default=None, metavar='REF_JSON',
                   help="Load reference zone boundaries from JSON; skip optimization. "
                        "NOTE: the original musun_pmt_ssd_zone_ratio_analysis.py always "
                        "ran in compare mode (COMPARE_MODE=True hardcoded). To replicate "
                        "that default behaviour pass the reference JSON explicitly, e.g.: "
                        "--compare <path>/zone_ratio_results.json  "
                        "(the musun default path is stored in _MUSUN_DEFAULTS['compare'])")
    p.add_argument('--geometry', type=str, default='currentDist',
                   help="Geometry name tag stored in output JSON metadata")
    # per-PMT-position ratio analysis (independent mode)
    p.add_argument('--per-pmt-ratio', action='store_true',
                   help="Run per-PMT-position ratio analysis instead of the zone scan. "
                        "Requires --ssd-postprocessed-file.")
    p.add_argument('--ssd-postprocessed-file', type=Path, default=None,
                   metavar='HDF5',
                   help="SSD postprocessed ncscore_output_*.hdf5 (target_matrix). "
                        "Required when --per-pmt-ratio is set.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Fill mode-specific path defaults
    if args.ssd_dir is None:
        args.ssd_dir = (_MUSUN_DEFAULTS['ssd_dir'] if args.mode == 'musun'
                        else _HOMO_DEFAULTS['ssd_dir'])
    if args.pmt_dir is None:
        args.pmt_dir = (_MUSUN_DEFAULTS['pmt_dir'] if args.mode == 'musun'
                        else _HOMO_DEFAULTS['pmt_dir'])
    if args.output_dir is None:
        args.output_dir = (_MUSUN_DEFAULTS['output_dir'] if args.mode == 'musun'
                           else _HOMO_DEFAULTS['output_dir'])

    compare_mode   = args.compare is not None
    zone_scan_areas = args.zone_scan if args.zone_scan else []

    geometry   = GeometryConfig(geometry_name=args.geometry)
    chunk_size = get_chunk_size()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Independent per-PMT-position analysis (short-circuits zone scan)
    if args.per_pmt_ratio:
        if args.ssd_postprocessed_file is None:
            raise ValueError(
                "--ssd-postprocessed-file is required when --per-pmt-ratio is set"
            )
        analyze_per_pmt_ratio(
            mode=args.mode,
            pmt_dir=args.pmt_dir,
            ssd_postprocessed_file=args.ssd_postprocessed_file,
            pmt_json=args.pmt_json,
            geometry=geometry,
            chunk_size=chunk_size,
            output_dir=args.output_dir,
        )
        return

    output_file = args.output_dir / "zone_ratio_results.txt"

    print("=" * 80)
    print("Zone-based SSD vs. PMT Photon Detection Efficiency Analysis")
    print("=" * 80)
    print(f"Mode:        {args.mode}")
    print(f"Geometry:    {geometry.geometry_name}")
    print(f"SSD dir:     {args.ssd_dir}")
    print(f"PMT dir:     {args.pmt_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Chunk size:  {chunk_size}")
    print(f"Memory:      {psutil.virtual_memory().available / (1024**3):.1f} GB available")
    print(f"Zone config: Pit={args.n_pit}, Wall={args.n_wall}, Top={args.n_top}, Bot=1")
    if compare_mode:
        print(f"Compare mode: {args.compare}")
    else:
        print(f"Zone scan areas: "
              f"{zone_scan_areas if zone_scan_areas else 'none (manual zones)'}")
        print(f"Min PMTs per zone: {args.min_pmts}")
    print(f"PMT cathode radius: {PMT_CATHODE_RADIUS} mm")
    print(f"MC samples for overlap: {MC_SAMPLES}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load PMT positions
    # -------------------------------------------------------------------------
    print("\nLoading PMT positions...")
    pmts          = load_pmt_data(args.pmt_json)
    pmt_by_layer  = get_pmts_by_layer(pmts)
    uid_to_pmt    = build_uid_to_pmt_map(pmts)
    print(f"  Total PMTs: {len(pmts)}, UID mappings: {len(uid_to_pmt)}")
    for layer, lp in pmt_by_layer.items():
        print(f"  {layer}: {len(lp)} PMTs")

    # -------------------------------------------------------------------------
    # Build zone boundaries
    # -------------------------------------------------------------------------
    if compare_mode:
        print(f"\n📊 COMPARE MODE: Loading reference from {args.compare}")
        ref_data = load_reference_json(args.compare)
        (pit_zones, top_zones, wall_zones, bot_zones,
         pit_bounds, top_bounds, wall_bounds, bot_bounds
         ) = build_zones_from_reference(ref_data, pmts, pmt_by_layer, geometry)
    else:
        print("\nComputing zone boundaries with fractional PMT assignment...")
        pit_zones,  pit_bounds  = build_radial_zones(
            pmt_by_layer['pit'],  args.n_pit,
            0.0, geometry.r_pit, "pit")
        top_zones,  top_bounds  = build_radial_zones(
            pmt_by_layer['top'],  args.n_top,
            geometry.r_zyl_top, geometry.r_zylinder, "top")
        wall_zones, wall_bounds = build_z_zones(
            pmt_by_layer['wall'], args.n_wall,
            geometry.z_cut_bot, geometry.z_cut_top, "wall", geometry.r_zylinder)

    bot_zones, bot_bounds = _build_bot_zone(geometry, pmt_by_layer)

    # Build zone_fractions lookup used by full-processing path and bot counting
    all_zones = pit_zones + top_zones + wall_zones + bot_zones
    zone_fractions: Dict[str, Dict[str, float]] = {
        f"{z.area_name}_{z.zone_id}": (z.pmt_fractions or {})
        for z in all_zones
    }
    all_zone_keys = [f"{z.area_name}_{z.zone_id}" for z in all_zones]

    print("\nValidating fractional PMT assignment...")
    _validate_fractions(all_zones, pmts)

    # -------------------------------------------------------------------------
    # Zone scan (find optimal N per area)
    # -------------------------------------------------------------------------
    optimal_zones_dict: Dict = {}
    scan_results:         Dict = {}
    bot_ssd_counts_scan:  Dict[str, int]   = {"bot_0": 0}
    bot_pmt_counts_scan:  Dict[str, float] = {"bot_0": 0.0}

    if zone_scan_areas and not compare_mode:
        print("\n" + "=" * 80)
        print(f"ZONE SCAN: Finding optimal number of zones for {zone_scan_areas}")
        print("=" * 80)

        (optimal_zones_dict, scan_results,
         bot_ssd_counts_scan, bot_pmt_counts_scan) = _run_zone_scan(
            args.mode, args.ssd_dir, args.pmt_dir, geometry, chunk_size,
            pmt_by_layer, uid_to_pmt, zone_scan_areas,
            args.n_pit, args.n_top, args.n_wall,
            bot_bounds, zone_fractions.get("bot_0", {}),
            args.snr_threshold, args.min_pmts,
        )

        n_pit_opt  = optimal_zones_dict.get('pit',  args.n_pit)
        n_top_opt  = optimal_zones_dict.get('top',  args.n_top)
        n_wall_opt = optimal_zones_dict.get('wall', args.n_wall)

        print("\n" + "=" * 80)
        print(f"OPTIMAL ZONE COUNTS: pit={n_pit_opt}, "
              f"top={n_top_opt}, wall={n_wall_opt}, bot=1")
        print("=" * 80)

        if scan_results:
            plot_snr_scan(scan_results, args.snr_threshold, optimal_zones_dict,
                          args.output_dir / "zone_snr_scan.png")

        # Rebuild only scanned areas with optimal N
        print("\nRebuilding zones with optimal counts...")
        if 'pit' in zone_scan_areas:
            pit_zones, pit_bounds = build_radial_zones(
                pmt_by_layer['pit'], n_pit_opt, 0.0, geometry.r_pit, "pit")
        if 'top' in zone_scan_areas:
            top_zones, top_bounds = build_radial_zones(
                pmt_by_layer['top'], n_top_opt,
                geometry.r_zyl_top, geometry.r_zylinder, "top")
        if 'wall' in zone_scan_areas:
            wall_zones, wall_bounds = build_z_zones(
                pmt_by_layer['wall'], n_wall_opt,
                geometry.z_cut_bot, geometry.z_cut_top, "wall", geometry.r_zylinder)

        all_zones = pit_zones + top_zones + wall_zones + bot_zones
        zone_fractions = {
            f"{z.area_name}_{z.zone_id}": (z.pmt_fractions or {})
            for z in all_zones
        }
        all_zone_keys = [f"{z.area_name}_{z.zone_id}" for z in all_zones]

    # -------------------------------------------------------------------------
    # Process files (or reuse scan counts)
    # -------------------------------------------------------------------------
    all_areas_scanned = (
        not compare_mode
        and bool(zone_scan_areas)
        and set(zone_scan_areas) >= {'pit', 'top', 'wall'}
    )

    if all_areas_scanned:
        print("\nReusing photon counts from zone scan...")
        ssd_counts: Dict[str, int] = {}
        pmt_counts: Dict[str, float] = {}
        for area in ['pit', 'top', 'wall']:
            ssd_counts.update(optimal_zones_dict[f"{area}_ssd_counts"])
            pmt_counts.update(optimal_zones_dict[f"{area}_pmt_counts"])
        ssd_counts["bot_0"] = bot_ssd_counts_scan["bot_0"]
        pmt_counts["bot_0"] = bot_pmt_counts_scan["bot_0"]
        ssd_nc    = optimal_zones_dict["pit_ssd_nc"]
        pmt_nc    = optimal_zones_dict["pit_pmt_nc"]
        ssd_files = len(list(args.ssd_dir.glob("run_*/output_t*.hdf5")))
        pmt_files = len(list(args.pmt_dir.glob("run_*/output_t*.hdf5")))
    else:
        print("\n[1/2] Processing SSD simulation data...")
        ssd_counts, ssd_files, ssd_nc = process_all_files_ssd(
            args.ssd_dir, geometry, chunk_size,
            pit_bounds, top_bounds, wall_bounds, bot_bounds,
            len(pit_zones), len(top_zones), len(wall_zones), 1,
            args.mode,
        )
        print("\n[2/2] Processing PMT simulation data...")
        pmt_counts, pmt_files, pmt_nc, observed_uids = process_all_files_pmt(
            args.pmt_dir, geometry, chunk_size,
            uid_to_pmt, zone_fractions, all_zone_keys,
            args.mode,
        )
        if not compare_mode:
            assert ssd_files == pmt_files, \
                f"File count mismatch: SSD={ssd_files}, PMT={pmt_files}"
        crosscheck_uids(uid_to_pmt, observed_uids, "PMT")

    if ssd_nc != pmt_nc:
        nc_diff     = abs(ssd_nc - pmt_nc)
        nc_diff_pct = 100.0 * nc_diff / max(ssd_nc, pmt_nc)
        print(f"\n⚠️  WARNING: NC mismatch: SSD={ssd_nc:,}, PMT={pmt_nc:,} "
              f"(diff={nc_diff:,}, {nc_diff_pct:.2f}%)")

    # -------------------------------------------------------------------------
    # Compute and print results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 130)
    print("RESULTS - ZONE RATIOS")
    print("=" * 130)

    area_mean_density: Dict[str, float] = {}
    for area in ['pit', 'bot', 'top', 'wall']:
        area_zones = [z for z in all_zones if z.area_name == area]
        total_eff  = sum(z.effective_n_pmts for z in area_zones)
        total_area = sum(z.area_mm2 for z in area_zones)
        area_mean_density[area] = total_eff / total_area if total_area > 0 else 0.0

    header_ext = (f"{'AREA':<8} {'ZONE':<6} {'BOUNDARY':<30} {'SSD_PH':>12} "
                  f"{'PMT_PH':>12} {'PH/NC_SSD':>12} {'PH/NC_PMT':>12} "
                  f"{'RAW_RATIO':>10} {'CORR_RATIO':>10} {'EFF_PMTs':>10} "
                  f"{'AREA_mm2':>12} {'DENSITY':>14} {'DENS_DEV':>9}")
    sep = "-" * len(header_ext)
    print(header_ext)
    print(sep)

    results = []
    for zone in all_zones:
        key    = f"{zone.area_name}_{zone.zone_id}"
        ssd_ph = ssd_counts.get(key, 0)
        pmt_ph = pmt_counts.get(key, 0.0)

        ssd_per_nc = ssd_ph / ssd_nc if ssd_nc > 0 else 0.0
        pmt_per_nc = pmt_ph / pmt_nc if pmt_nc > 0 else 0.0
        raw_ratio  = ssd_per_nc / pmt_per_nc if pmt_per_nc > 0 else float('nan')

        mean_d = area_mean_density[zone.area_name]
        if mean_d > 0 and zone.pmt_density > 0 and not np.isnan(raw_ratio):
            corr_ratio = raw_ratio * (zone.pmt_density / mean_d)
        else:
            corr_ratio = float('nan')

        dens_dev = (zone.pmt_density - mean_d) / mean_d * 100 if mean_d > 0 else 0.0

        if zone.area_name in ('pit', 'bot', 'top'):
            bstr = f"r=[{zone.boundary_low:.0f}, {zone.boundary_high:.0f}]mm"
        else:
            bstr = f"z=[{zone.boundary_low:.0f}, {zone.boundary_high:.0f}]mm"

        print(f"{zone.area_name:<8} {zone.zone_id:<6} {bstr:<30} "
              f"{ssd_ph:>12d} {pmt_ph:>12.1f} {ssd_per_nc:>12.6f} {pmt_per_nc:>12.6f} "
              f"{raw_ratio:>10.4f} {corr_ratio:>10.4f} {zone.effective_n_pmts:>10.2f} "
              f"{zone.area_mm2:>12.0f} {zone.pmt_density:>14.6e} {dens_dev:>+8.1f}%")

        results.append({
            'zone': zone, 'key': key,
            'ssd_photons': ssd_ph, 'pmt_photons': pmt_ph,
            'ssd_per_nc': ssd_per_nc, 'pmt_per_nc': pmt_per_nc,
            'ratio': raw_ratio, 'corr_ratio': corr_ratio, 'dens_dev': dens_dev,
        })

    print(sep)

    # Per-area aggregates
    print(f"\n{'AREA':<10} {'SSD_TOTAL':>12} {'PMT_TOTAL':>12} {'PH/NC_SSD':>12} "
          f"{'PH/NC_PMT':>12} {'RATIO':>10}")
    print("-" * 70)
    for area in ['pit', 'bot', 'top', 'wall']:
        ar     = [r for r in results if r['zone'].area_name == area]
        ssd_t  = sum(r['ssd_photons'] for r in ar)
        pmt_t  = sum(r['pmt_photons'] for r in ar)
        ssd_pnc = ssd_t / ssd_nc if ssd_nc > 0 else 0.0
        pmt_pnc = pmt_t / pmt_nc if pmt_nc > 0 else 0.0
        ratio   = ssd_pnc / pmt_pnc if pmt_pnc > 0 else float('nan')
        print(f"{area:<10} {ssd_t:>12d} {pmt_t:>12.1f} "
              f"{ssd_pnc:>12.6f} {pmt_pnc:>12.6f} {ratio:>10.4f}")
    g_ssd    = sum(r['ssd_photons'] for r in results)
    g_pmt    = sum(r['pmt_photons'] for r in results)
    g_ssd_pnc = g_ssd / ssd_nc if ssd_nc > 0 else 0.0
    g_pmt_pnc = g_pmt / pmt_nc if pmt_nc > 0 else 0.0
    g_ratio   = g_ssd_pnc / g_pmt_pnc if g_pmt_pnc > 0 else float('nan')
    print("-" * 70)
    print(f"{'TOTAL':<10} {g_ssd:>12d} {g_pmt:>12.1f} "
          f"{g_ssd_pnc:>12.6f} {g_pmt_pnc:>12.6f} {g_ratio:>10.4f}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    save_results_txt(
        output_file, results, pit_zones, top_zones, wall_zones,
        ssd_nc, pmt_nc, ssd_files, pmt_files,
        zone_scan_areas, args.snr_threshold, args.min_pmts,
    )
    save_results_json(
        args.output_dir / "zone_ratio_results.json",
        results, pit_zones, top_zones, wall_zones,
        pit_bounds, top_bounds, wall_bounds, bot_bounds,
        ssd_nc, pmt_nc, ssd_files, pmt_files, geometry.geometry_name,
    )

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    print("\nGenerating plots...")
    all_ratios = [r['corr_ratio'] for r in results if not np.isnan(r['corr_ratio'])]
    if all_ratios:
        global_norm = plt.Normalize(vmin=min(all_ratios) * 0.9, vmax=max(all_ratios) * 1.1)
    else:
        global_norm = plt.Normalize(0, 1)

    for area, zones_list, layer_pmts, r_min, r_max in [
        ("pit",  pit_zones,  pmt_by_layer['pit'],  0.0,               geometry.r_pit),
        ("top",  top_zones,  pmt_by_layer['top'],  geometry.r_zyl_top, geometry.r_zylinder),
        ("bot",  bot_zones,  pmt_by_layer['bot'],  geometry.r_zyl_bot, geometry.r_zylinder),
    ]:
        ratios = [
            next((r['corr_ratio'] for r in results
                  if r['zone'].area_name == area and r['zone'].zone_id == z.zone_id),
                 float('nan'))
            for z in zones_list
        ]
        plot_radial_zones(zones_list, layer_pmts, ratios, area, r_min, r_max,
                          args.output_dir / f"zone_ratio_{area}.png", global_norm)

    wall_ratios = [
        next((r['corr_ratio'] for r in results
              if r['zone'].area_name == 'wall' and r['zone'].zone_id == z.zone_id),
             float('nan'))
        for z in wall_zones
    ]
    plot_wall_zones(wall_zones, pmt_by_layer['wall'], wall_ratios,
                    geometry.r_zylinder, args.output_dir / "zone_ratio_wall.png", global_norm)

    if compare_mode:
        print("\nGenerating comparison plots...")
        plot_comparison(results, ref_data, args.output_dir)

    for area in ['pit', 'top', 'wall']:
        area_results = [r for r in results if r['zone'].area_name == area]
        plot_area_flux(area_results, area,
                       args.output_dir / f"{area}_ssd_vs_pmt_flux.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
