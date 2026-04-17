"""Process PMT raw HDF5 files, counting photons per zone."""

import gc
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from .geometry import GeometryConfig
from .pmt_data import PMTInfo
from .nc_data import (
    load_nc_data_dict_homogeneous,
    load_nc_data_dict_musun,
    count_nc_from_csv,
)


def process_pmt_file(
    file_path: Path,
    geometry: GeometryConfig,
    chunk_size: int,
    nc_data_dict: Dict,
    primary_id_field: str,
    uid_to_pmt: Dict[int, PMTInfo],
    zone_fractions: Dict[str, Dict[str, float]],
    all_zone_keys: List[str],
) -> Tuple[Dict[str, float], set]:
    """Process one PMT raw HDF5 file.

    Counts photons per PMT UID (filtered by NC time window), then distributes
    fractionally across zones using precomputed overlap fractions.

    Args:
        primary_id_field: 'evtid' or 'muon_track_id'

    Returns:
        (zone_counts, observed_uids)
    """
    pmt_photon_counts: Dict[int, int] = {}
    observed_uids: set = set()

    try:
        with h5py.File(file_path, 'r') as f:
            total = len(f['hit']['optical']['x_position_in_m']['pages'])
            num_chunks = (total - 1) // chunk_size + 1

            for chunk_idx in range(num_chunks):
                cs = chunk_idx * chunk_size
                ce = min(cs + chunk_size, total)

                primary_ids  = f['hit']['optical'][primary_id_field]['pages'][cs:ce]
                nc_track_ids = f['hit']['optical']['nC_track_id']['pages'][cs:ce]
                time         = np.array(
                    f['hit']['optical']['time_in_ns']['pages'][cs:ce], dtype=np.float64)
                det_uid      = f['hit']['optical']['det_uid']['pages'][cs:ce]

                # NC time filter
                nc_times = np.full(len(time), np.inf, dtype=np.float64)
                for idx in range(len(primary_ids)):
                    key = (int(primary_ids[idx]), int(nc_track_ids[idx]))
                    if key in nc_data_dict:
                        nc_times[idx] = nc_data_dict[key]['nC_time']
                time_mask = (nc_times != np.inf) & (time >= nc_times) & (time <= nc_times + 200.0)

                det_uid_filtered = det_uid[time_mask]

                # Valid PMT UIDs (8 digits starting with 1)
                pmt_mask = ((det_uid_filtered >= 10_000_000) &
                            (det_uid_filtered < 1_000_000_000))
                det_uid_filtered = det_uid_filtered[pmt_mask]

                unique_uids, uid_counts = np.unique(det_uid_filtered, return_counts=True)
                for uid_val, cnt in zip(unique_uids, uid_counts):
                    uid_int = int(uid_val)
                    pmt_photon_counts[uid_int] = pmt_photon_counts.get(uid_int, 0) + int(cnt)
                    observed_uids.add(uid_int)

                del primary_ids, nc_track_ids, time, det_uid, nc_times, time_mask
                del det_uid_filtered
                gc.collect()

    except Exception as e:
        print(f"Error processing PMT {file_path}: {e}")
        return {k: 0.0 for k in all_zone_keys}, set()

    # Distribute photons fractionally across zones
    zone_counts: Dict[str, float] = {k: 0.0 for k in all_zone_keys}
    for uid_int, n_photons in pmt_photon_counts.items():
        if uid_int not in uid_to_pmt:
            continue
        pmt = uid_to_pmt[uid_int]
        for zone_key, pmt_fracs in zone_fractions.items():
            if pmt.index in pmt_fracs:
                zone_counts[zone_key] += n_photons * pmt_fracs[pmt.index]

    return zone_counts, observed_uids


def process_all_files_pmt(
    base_path: Path,
    geometry: GeometryConfig,
    chunk_size: int,
    uid_to_pmt: Dict[int, PMTInfo],
    zone_fractions: Dict[str, Dict[str, float]],
    all_zone_keys: List[str],
    mode: str,
) -> Tuple[Dict[str, float], int, int, set]:
    """Process all PMT run directories.

    Args:
        mode: 'homogeneous' or 'musun'

    Returns:
        (total_counts, total_files, total_nc, all_observed_uids)
    """
    total_counts: Dict[str, float] = {k: 0.0 for k in all_zone_keys}
    total_files = 0
    total_nc = 0
    all_observed_uids: set = set()

    run_dirs = sorted(base_path.glob("run_*"))
    print(f"Processing PMT setup [{mode}]: found {len(run_dirs)} runs")

    for run_idx, run_dir in enumerate(run_dirs, 1):
        print(f"  Run {run_idx}/{len(run_dirs)}: {run_dir.name}")
        hdf5_files = sorted(run_dir.glob("output_t*.hdf5"))

        if mode == 'musun':
            nc_csv = run_dir / "merged_ncs.csv"
            if not nc_csv.exists():
                print(f"    ⚠️ No merged_ncs.csv in {run_dir.name}, skipping")
                continue
            total_nc += count_nc_from_csv(nc_csv)
            nc_data_dict_run = load_nc_data_dict_musun(nc_csv)
            primary_id_field = 'muon_track_id'

        for file_idx, hdf5_file in enumerate(hdf5_files):
            if (file_idx + 1) % 50 == 0:
                print(f"    Processing file {file_idx + 1}/{len(hdf5_files)}")
            total_files += 1

            if mode == 'homogeneous':
                with h5py.File(hdf5_file, 'r') as f:
                    total_nc += len(
                        f['hit']['MyNeutronCaptureOutput']['evtid']['pages']
                    )
                    nc_data_dict = load_nc_data_dict_homogeneous(f)
                primary_id_field = 'evtid'
            else:
                nc_data_dict = nc_data_dict_run

            file_counts, file_uids = process_pmt_file(
                hdf5_file, geometry, chunk_size, nc_data_dict, primary_id_field,
                uid_to_pmt, zone_fractions, all_zone_keys,
            )
            for k, v in file_counts.items():
                total_counts[k] += v
            all_observed_uids.update(file_uids)

        area_sums = {}
        for k, v in total_counts.items():
            area = k.split('_')[0]
            area_sums[area] = area_sums.get(area, 0) + v
        print(f"    Run {run_idx} complete: {area_sums}")

    return total_counts, total_files, total_nc, all_observed_uids
