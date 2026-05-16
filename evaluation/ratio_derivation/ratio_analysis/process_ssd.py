"""Process SSD raw HDF5 files, counting photons per zone."""

import gc
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .geometry import GeometryConfig
from .photon_filters import (
    checkRadialMomentumVectorized,
    SSD_UID_PIT, SSD_UID_BOT, SSD_UID_TOP, SSD_UID_WALL,
)
from .nc_data import (
    load_nc_data_dict_homogeneous,
    load_nc_data_dict_musun,
    count_nc_from_hdf5,
    count_nc_from_csv,
)
from .zones import assign_radial_zone, assign_z_zone


def process_ssd_file(
    file_path: Path,
    geometry: GeometryConfig,
    chunk_size: int,
    nc_data_dict: Dict,
    primary_id_field: str,
    pit_boundaries: List[float],
    top_boundaries: List[float],
    wall_boundaries: List[float],
    bot_boundaries: List[float],
    n_pit: int, n_top: int, n_wall: int, n_bot: int,
) -> Dict[str, int]:
    """Process one SSD raw HDF5 file; count photons per zone.

    Args:
        nc_data_dict: pre-loaded NC event lookup: (primary_id, nC_id) → {'nC_time': float}
        primary_id_field: HDF5 field name for the primary event ID:
            'evtid' for homogeneous, 'muon_track_id' for musun.
    """
    counts: Dict[str, int] = {}
    for area, n in [('pit', n_pit), ('top', n_top), ('wall', n_wall), ('bot', n_bot)]:
        for i in range(n):
            counts[f"{area}_{i}"] = 0

    try:
        with h5py.File(file_path, 'r') as f:
            total = len(f['hit']['optical']['x_position_in_m']['pages'])
            num_chunks = (total - 1) // chunk_size + 1
            z_cut_bot = geometry.z_cut_bot
            z_cut_top = geometry.z_cut_top

            for chunk_idx in range(num_chunks):
                cs = chunk_idx * chunk_size
                ce = min(cs + chunk_size, total)

                x  = np.array(f['hit']['optical']['x_position_in_m']['pages'][cs:ce],
                               dtype=np.float64) * 1000
                y  = np.array(f['hit']['optical']['y_position_in_m']['pages'][cs:ce],
                               dtype=np.float64) * 1000
                z  = np.array(f['hit']['optical']['z_position_in_m']['pages'][cs:ce],
                               dtype=np.float64) * 1000
                px = np.array(f['hit']['optical']['x_momentum_direction']['pages'][cs:ce],
                               dtype=np.float64)
                py = np.array(f['hit']['optical']['y_momentum_direction']['pages'][cs:ce],
                               dtype=np.float64)
                pz = np.array(f['hit']['optical']['z_momentum_direction']['pages'][cs:ce],
                               dtype=np.float64)
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

                x  = x[time_mask];  y  = y[time_mask];  z  = z[time_mask]
                px = px[time_mask]; py = py[time_mask]; pz = pz[time_mask]
                det_uid = det_uid[time_mask]

                # Momentum filter
                mask_bot    = z <= z_cut_bot
                mask_top    = z >= z_cut_top
                mask_barrel = ~mask_bot & ~mask_top

                final_mask = np.zeros(len(z), dtype=bool)
                final_mask[mask_bot] = pz[mask_bot] <= 0
                final_mask[mask_top] = pz[mask_top] >= 0
                if np.any(mask_barrel):
                    final_mask[mask_barrel] = checkRadialMomentumVectorized(
                        x[mask_barrel], y[mask_barrel], z[mask_barrel],
                        px[mask_barrel], py[mask_barrel], pz[mask_barrel],
                    )

                x_f   = x[final_mask];   y_f = y[final_mask];  z_f = z[final_mask]
                uid_f = det_uid[final_mask]
                r_f   = np.sqrt(x_f**2 + y_f**2)

                # Pit
                pit_mask = uid_f == SSD_UID_PIT
                if np.any(pit_mask):
                    zids = assign_radial_zone(r_f[pit_mask], pit_boundaries)
                    for zi in range(n_pit):
                        counts[f"pit_{zi}"] += int(np.sum(zids == zi))

                # Bot
                bot_mask = uid_f == SSD_UID_BOT
                if np.any(bot_mask):
                    zids = assign_radial_zone(r_f[bot_mask], bot_boundaries)
                    for zi in range(n_bot):
                        counts[f"bot_{zi}"] += int(np.sum(zids == zi))

                # Top
                top_mask = uid_f == SSD_UID_TOP
                if np.any(top_mask):
                    zids = assign_radial_zone(r_f[top_mask], top_boundaries)
                    for zi in range(n_top):
                        counts[f"top_{zi}"] += int(np.sum(zids == zi))

                # Wall
                wall_mask = uid_f == SSD_UID_WALL
                if np.any(wall_mask):
                    zids = assign_z_zone(z_f[wall_mask], wall_boundaries)
                    for zi in range(n_wall):
                        counts[f"wall_{zi}"] += int(np.sum(zids == zi))

                del x, y, z, px, py, pz, primary_ids, nc_track_ids, time, det_uid
                del nc_times, time_mask, mask_bot, mask_top, mask_barrel, final_mask
                del x_f, y_f, z_f, uid_f, r_f
                gc.collect()

    except Exception as e:
        print(f"Error processing SSD {file_path}: {e}")

    return counts


def process_all_files_ssd(
    base_path: Path,
    geometry: GeometryConfig,
    chunk_size: int,
    pit_boundaries: List[float],
    top_boundaries: List[float],
    wall_boundaries: List[float],
    bot_boundaries: List[float],
    n_pit: int, n_top: int, n_wall: int, n_bot: int,
    mode: str,
    multi_ssd_dir: Optional[Path] = None,
    runs: Optional[List[int]] = None,
) -> Tuple[Dict[str, int], int, int]:
    """Process all SSD run directories, optionally summing multiple independent
    optical simulations of the same neutron captures.

    Args:
        base_path:      SSD base directory containing run_* subdirectories.
        multi_ssd_dir:  If provided, each run_* must also contain sim_*
                        subdirectories (run_*/sim_*/output_t*.hdf5).  Hits from
                        all sim_* directories are summed with those from
                        base_path.  NC events are counted only from base_path
                        (all sims share the same physical NCs).
        mode:           'homogeneous' or 'musun'
        runs:           if not None, restrict to these run IDs (e.g. [1] for run_001)

    Returns:
        (total_counts, total_files, total_nc)
    """
    all_keys = ([f"pit_{i}"  for i in range(n_pit)]  +
                [f"top_{i}"  for i in range(n_top)]  +
                [f"wall_{i}" for i in range(n_wall)] +
                [f"bot_{i}"  for i in range(n_bot)])
    total_counts: Dict[str, int] = {k: 0 for k in all_keys}
    total_files = 0
    total_nc = 0

    run_dirs = sorted(base_path.glob("run_*"))
    if runs is not None:
        runs_set = set(runs)
        run_dirs = [r for r in run_dirs
                    if r.name[4:].isdigit() and int(r.name[4:]) in runs_set]
    multi_label = f" + multi_ssd ({multi_ssd_dir.name})" if multi_ssd_dir else ""
    print(f"Processing SSD setup [{mode}]{multi_label}: found {len(run_dirs)} runs")

    for run_idx, run_dir in enumerate(run_dirs, 1):
        print(f"  Run {run_idx}/{len(run_dirs)}: {run_dir.name}")

        if mode == 'musun':
            nc_csv = run_dir / "merged_ncs.csv"
            if not nc_csv.exists():
                print(f"    ⚠️ No merged_ncs.csv in {run_dir.name}, skipping")
                continue
            total_nc += count_nc_from_csv(nc_csv)
            nc_data_dict_run = load_nc_data_dict_musun(nc_csv)
            primary_id_field = 'muon_track_id'

        # Collect all (file, count_nc_flag) pairs for this run.
        # count_nc_flag=True only for base_path files; all sims share the same NCs.
        source_files: List[Tuple[Path, bool]] = [
            (f, True) for f in sorted(run_dir.glob("output_t*.hdf5"))
        ]
        if multi_ssd_dir is not None:
            multi_run_dir = multi_ssd_dir / run_dir.name
            if multi_run_dir.exists():
                for sim_dir in sorted(multi_run_dir.glob("sim_*")):
                    for f in sorted(sim_dir.glob("output_t*.hdf5")):
                        source_files.append((f, False))
            else:
                print(f"    ⚠️ {run_dir.name} not found in multi_ssd_dir, "
                      f"skipping multi sims for this run")

        for file_idx, (hdf5_file, count_nc_flag) in enumerate(source_files):
            if (file_idx + 1) % 50 == 0:
                print(f"    Processing file {file_idx + 1}/{len(source_files)}")
            total_files += 1

            if mode == 'homogeneous':
                with h5py.File(hdf5_file, 'r') as f:
                    if count_nc_flag:
                        total_nc += len(
                            f['hit']['MyNeutronCaptureOutput']['evtid']['pages']
                        )
                    nc_data_dict = load_nc_data_dict_homogeneous(f)
                primary_id_field = 'evtid'
            else:
                nc_data_dict = nc_data_dict_run

            file_counts = process_ssd_file(
                hdf5_file, geometry, chunk_size, nc_data_dict, primary_id_field,
                pit_boundaries, top_boundaries, wall_boundaries, bot_boundaries,
                n_pit, n_top, n_wall, n_bot,
            )
            for k, v in file_counts.items():
                total_counts[k] += v

        area_sums = {}
        for k, v in total_counts.items():
            area = k.split('_')[0]
            area_sums[area] = area_sums.get(area, 0) + v
        print(f"    Run {run_idx} complete: {area_sums}")

    return total_counts, total_files, total_nc
