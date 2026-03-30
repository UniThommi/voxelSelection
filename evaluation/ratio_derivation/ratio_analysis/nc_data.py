"""Neutron capture event data loading — homogeneous and musun variants."""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def load_nc_data_dict_homogeneous(f: h5py.File) -> Dict[Tuple, Dict]:
    """Load NC event data from HDF5 (homogeneous NC simulations).

    Key: (evtid, nC_track_id) → {'nC_time': float}
    Matches against hit/optical/evtid and hit/optical/nC_track_id.
    """
    nc_out = f['hit']['MyNeutronCaptureOutput']
    nc_evtid = nc_out['evtid']['pages'][:]
    nc_nC_id = nc_out['nC_track_id']['pages'][:]
    nc_time  = nc_out['nC_time_in_ns']['pages'][:]

    nc_data_dict = {}
    for idx in range(len(nc_evtid)):
        key = (nc_evtid[idx], nc_nC_id[idx])
        nc_data_dict[key] = {'nC_time': nc_time[idx]}
    return nc_data_dict


def load_nc_data_dict_musun(csv_path: Path) -> Dict[Tuple, Dict]:
    """Load NC event data from merged_ncs.csv (musun simulations).

    Key: (muon_track_id, nC_track_id) → {'nC_time': float}
    Matches against hit/optical/muon_track_id and hit/optical/nC_track_id.
    """
    nc_data_dict = {}
    with open(csv_path, 'r') as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            muon_id = int(parts[0])
            nc_id   = int(parts[1])
            nc_time = float(parts[5])
            nc_data_dict[(muon_id, nc_id)] = {'nC_time': nc_time}
    return nc_data_dict


def count_nc_from_hdf5(file_path: Path) -> int:
    """Return number of NC events in a raw HDF5 file (homogeneous)."""
    with h5py.File(file_path, 'r') as f:
        return len(f['hit']['MyNeutronCaptureOutput']['evtid']['pages'])


def count_nc_from_csv(csv_path: Path) -> int:
    """Return number of NC events in merged_ncs.csv (musun), excluding header."""
    with open(csv_path, 'r') as f:
        return sum(1 for _ in f) - 1
