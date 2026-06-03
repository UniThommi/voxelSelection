"""
Level-1 MUSUN analysis: MUSUN muon simulation → Neutron Capture (NC) extraction.

10 simulation runs with 1×10^7 muons each.

Outputs (all to <output_path>/musun_nc_analysis/):
  - Standard distributions: all muons + Ge77 only + Ge77 vs non-Ge77 + NC vs non-NC
  - NC spatial / time distributions
  - Outlier analysis (muons above 99th-percentile NC count on log scale)
  - Convergence analysis: W1 distance vs number of runs (k) and vs sample size (N)
  - statistics.txt
"""
from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from itertools import combinations as _combinations
from scipy.stats import wasserstein_distance

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NC_GROUP = "/hit/MyNeutronCaptureOutput"
GAMMA_GROUP = "/hit/CaptureGammas"
VERTICES_GROUP = "/hit/vertices"
PARTICLES_GROUP = "/hit/particles"

NC_CUT_100 = 100

# NC-level fields used for the 3-level statistical convergence test
NC_FIELDS_FOR_CONV: dict[str, str] = {
    "nc_time_ns":  "NC time [ns]",
    "nc_x_m":      "NC x position [m]",
    "nc_y_m":      "NC y position [m]",
    "nc_z_m":      "NC z position [m]",
    "nc_r_m":      "NC r position [m]",
    "nc_phi_rad":  "NC φ [rad]",
    "nc_ge77":     "Ge77 flag",
    "nc_n_gammas": "N capture gammas",
    "nc_counts":   "NC count per muon",
}
# nc_phi_rad uses circular W1 (see _w1_circular_phi); all others use linear W1.
_CIRCULAR_FIELDS: frozenset[str] = frozenset({"nc_phi_rad"})

DEFAULT_DATA_PATH = (
    "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/rawMusunNCs"
)
NUM_RUNS_DEFAULT = 10
N_PERMUTATIONS = 100  # random subsets per k in convergence analysis
RANDOM_SEED = 42
W1_THRESHOLD_FRAC = 0.05   # convergence threshold = 5% of W1 at k=1
MAX_SCATTER_PTS = 5_000    # max points for 3-D scatter / arrow plots

COLORS = {
    "blue": "#4C72B0",
    "red": "#C44E52",
    "green": "#55A868",
    "orange": "#DD8452",
    "purple": "#8172B2",
}

_T0_GLOBAL = time.perf_counter()


def _log_resources(label: str, t_ref: float | None = None) -> float:
    """Print wall-clock time and RSS; return current time."""
    now = time.perf_counter()
    elapsed = now - (t_ref if t_ref is not None else _T0_GLOBAL)
    if _HAS_PSUTIL:
        import os
        rss_gb = _psutil.Process(os.getpid()).memory_info().rss / 1e9
        print(f"  [RESOURCE] {label:50s}  "
              f"elapsed={elapsed:7.1f}s  RSS={rss_gb:.2f} GB", flush=True)
    else:
        print(f"  [RESOURCE] {label:50s}  "
              f"elapsed={elapsed:7.1f}s  (no psutil)", flush=True)
    return now


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class RunData:
    """Extracted and derived data for one simulation run."""

    run_name: str = ""
    n_files: int = 0
    n_muons_total: int = 0  # unique evtids found in vertices/particles groups

    # NC-level (one row per unique (evtid, nC_track_id) after deduplication)
    nc_evtid: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    nc_ge77: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    nc_time_ns: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    nc_phi_rad: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    nc_x_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    nc_y_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    nc_z_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    nc_r_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    # Per NC-producing muon (sorted by evtid)
    muon_nc_evtids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    nc_counts: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    # Ge77-producing muon evtids
    ge77_evtids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    # Number of Ge77-flagged NCs per Ge77-producing muon (length == ge77_evtids.size)
    ge77_nc_per_ge77muon: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    # All muons: one row per unique evtid (first occurrence)
    muon_evtid: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    muon_ekin_mev: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    muon_zenith_deg: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    muon_azimuth_deg: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    muon_x_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    muon_y_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    muon_z_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    muon_px_mev: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    muon_py_mev: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    muon_pz_mev: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    # True if the NC occurred inside the water volume
    nc_is_water: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    # Material name per NC (resolved per-file from //hit/materials/, dtype=object)
    nc_material_name: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    # Number of capture gammas per NC
    nc_n_gammas: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))


# ---------------------------------------------------------------------------
# HDF5 reading helpers
# ---------------------------------------------------------------------------
def _pages(grp: h5py.Group, name: str) -> np.ndarray:
    """Read grp[name]['pages'] dataset."""
    return grp[name]["pages"][:]


def read_nc_data_file(fp: Path) -> dict[str, np.ndarray]:
    """Return NC fields from one HDF5 file; empty arrays if group absent/empty."""
    empty: dict[str, np.ndarray] = {
        "evtid":       np.array([], dtype=np.int64),
        "track_id":    np.array([], dtype=np.int64),
        "ge77":        np.array([], dtype=np.int32),
        "time_ns":     np.array([], dtype=np.float64),
        "x_m":         np.array([], dtype=np.float64),
        "y_m":         np.array([], dtype=np.float64),
        "z_m":         np.array([], dtype=np.float64),
        "material_id": np.array([], dtype=np.int32),
    }
    try:
        with h5py.File(fp, "r") as f:
            if NC_GROUP not in f:
                return empty
            grp = f[NC_GROUP]
            if int(grp["entries"][()]) == 0:
                return empty
            return {
                "evtid":       _pages(grp, "evtid").astype(np.int64),
                "track_id":    _pages(grp, "nC_track_id").astype(np.int64),
                "ge77":        _pages(grp, "nC_flag_Ge77").astype(np.int32),
                "time_ns":     _pages(grp, "nC_time_in_ns"),
                "x_m":         _pages(grp, "nC_x_position_in_m"),
                "y_m":         _pages(grp, "nC_y_position_in_m"),
                "z_m":         _pages(grp, "nC_z_position_in_m"),
                "material_id": _pages(grp, "nC_material_id").astype(np.int32),
            }
    except Exception as exc:
        print(f"  ERROR reading NC data from {fp.name}: {exc}")
        return empty


def read_material_map(fp: Path) -> dict[int, str]:
    """Read material-ID → name mapping from //hit/materials/ in an HDF5 file."""
    try:
        with h5py.File(fp, "r") as f:
            grp = f["hit/materials"]
            raw_names = grp["materialNames"]["pages"][:]
            raw_ids   = grp["materialsID"]["pages"][:]
        return {
            int(mid): (name.decode() if isinstance(name, bytes) else str(name))
            for mid, name in zip(raw_ids, raw_names)
        }
    except Exception as exc:
        print(f"  WARNING: could not read material map from {fp.name}: {exc}")
        return {}


def read_gamma_count_file(fp: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (evtid, nc_id) arrays from /hit/CaptureGammas (one row per gamma)."""
    empty = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    try:
        with h5py.File(fp, "r") as f:
            if GAMMA_GROUP not in f:
                return empty
            grp = f[GAMMA_GROUP]
            if int(grp["entries"][()]) == 0:
                return empty
            evtid = _pages(grp, "evtid").astype(np.int64)
            nc_id = _pages(grp, "nc_id").astype(np.int64)
        return evtid, nc_id
    except Exception as exc:
        print(f"  ERROR reading gamma data from {fp.name}: {exc}")
        return empty


def read_muon_data_file(fp: Path) -> dict[str, np.ndarray]:
    """Return muon kinematics + positions from one HDF5 file.

    Joins /hit/vertices (positions) with /hit/particles (kinematics) on
    evtid.  Zenith and azimuth are computed from the momentum vector.
    """
    empty: dict[str, np.ndarray] = {
        k: np.array([], dtype=dt)
        for k, dt in [
            ("evtid", np.int64), ("ekin_mev", np.float64),
            ("zenith_deg", np.float64), ("azimuth_deg", np.float64),
            ("x_m", np.float64), ("y_m", np.float64), ("z_m", np.float64),
            ("px_mev", np.float64), ("py_mev", np.float64), ("pz_mev", np.float64),
        ]
    }
    try:
        with h5py.File(fp, "r") as f:
            if VERTICES_GROUP not in f or PARTICLES_GROUP not in f:
                return empty
            vgrp, pgrp = f[VERTICES_GROUP], f[PARTICLES_GROUP]
            if int(vgrp["entries"][()]) == 0 or int(pgrp["entries"][()]) == 0:
                return empty

            evtid_v = _pages(vgrp, "evtid").astype(np.int64)
            x_m = _pages(vgrp, "xloc_in_m")
            y_m = _pages(vgrp, "yloc_in_m")
            z_m = _pages(vgrp, "zloc_in_m")

            evtid_p = _pages(pgrp, "evtid").astype(np.int64)
            ekin = _pages(pgrp, "ekin_in_MeV")
            px = _pages(pgrp, "px_in_MeV")
            py = _pages(pgrp, "py_in_MeV")
            pz = _pages(pgrp, "pz_in_MeV")

        # Align by evtid (both are sorted per MUSUN event order, but join safely)
        sv = np.argsort(evtid_v, kind="stable")
        sp = np.argsort(evtid_p, kind="stable")
        common, iv, ip = np.intersect1d(
            evtid_v[sv], evtid_p[sp], return_indices=True
        )
        if common.size == 0:
            return empty

        evtid = common
        x_m = x_m[sv][iv]
        y_m = y_m[sv][iv]
        z_m = z_m[sv][iv]
        ekin = ekin[sp][ip]
        px = px[sp][ip]
        py = py[sp][ip]
        pz = pz[sp][ip]

        p_mag = np.sqrt(px**2 + py**2 + pz**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_theta = np.where(p_mag > 0, pz / p_mag, 0.0)
        zenith_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        azimuth_deg = np.degrees(np.arctan2(py, px))

        return {
            "evtid": evtid,
            "ekin_mev": ekin,
            "zenith_deg": zenith_deg,
            "azimuth_deg": azimuth_deg,
            "x_m": x_m,
            "y_m": y_m,
            "z_m": z_m,
            "px_mev": px,
            "py_mev": py,
            "pz_mev": pz,
        }
    except Exception as exc:
        print(f"  ERROR reading muon data from {fp.name}: {exc}")
        return empty


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------
def load_run(run_dir: Path) -> RunData:
    """Load, merge and deduplicate all HDF5 files in one run directory."""
    rd = RunData(run_name=run_dir.name)
    hdf5_files = sorted(run_dir.glob("output_t*.hdf5"))
    rd.n_files = len(hdf5_files)

    if rd.n_files == 0:
        print(f"  WARNING: no output_t*.hdf5 found in {run_dir}", flush=True)
        return rd

    nc_parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("evtid", "track_id", "ge77", "time_ns", "x_m", "y_m", "z_m")
    }
    nc_mat_name_parts: list[np.ndarray] = []
    mu_parts: dict[str, list[np.ndarray]] = {
        k: [] for k in (
            "evtid", "ekin_mev", "zenith_deg", "azimuth_deg",
            "x_m", "y_m", "z_m", "px_mev", "py_mev", "pz_mev",
        )
    }
    gamma_evtid_l: list[np.ndarray] = []
    gamma_nc_id_l: list[np.ndarray] = []

    for fp in hdf5_files:
        nc = read_nc_data_file(fp)
        if nc["evtid"].size > 0:
            for k in nc_parts:
                nc_parts[k].append(nc[k])
            # Read per-file material map and immediately resolve IDs → names
            fmat_map = read_material_map(fp)
            mat_names = np.array(
                [fmat_map.get(int(mid), f"ID:{mid}") for mid in nc["material_id"]],
                dtype=object,
            )
            nc_mat_name_parts.append(mat_names)

        mu = read_muon_data_file(fp)
        if mu["evtid"].size > 0:
            for k in mu_parts:
                mu_parts[k].append(mu[k])

        ge, gn = read_gamma_count_file(fp)
        if ge.size > 0:
            gamma_evtid_l.append(ge)
            gamma_nc_id_l.append(gn)

    # ---- NC deduplication on (evtid, nC_track_id) ----
    if nc_parts["evtid"]:
        evtid_arr    = np.concatenate(nc_parts["evtid"])
        track_arr    = np.concatenate(nc_parts["track_id"])
        ge77_arr     = np.concatenate(nc_parts["ge77"])
        time_arr     = np.concatenate(nc_parts["time_ns"])
        x_arr        = np.concatenate(nc_parts["x_m"])
        y_arr        = np.concatenate(nc_parts["y_m"])
        z_arr        = np.concatenate(nc_parts["z_m"])
        mat_name_arr = np.concatenate(nc_mat_name_parts)

        pair_keys = np.stack([evtid_arr, track_arr], axis=1)
        _, unique_idx, inverse = np.unique(
            pair_keys, axis=0, return_index=True, return_inverse=True
        )
        # For duplicate NC records keep max Ge77 flag (1 beats 0)
        unique_ge77 = np.zeros(len(unique_idx), dtype=np.int32)
        np.maximum.at(unique_ge77, inverse, ge77_arr)

        unique_track_ids = track_arr[unique_idx]  # kept for gamma-count matching

        rd.nc_evtid   = evtid_arr[unique_idx]
        rd.nc_ge77    = unique_ge77
        rd.nc_time_ns = time_arr[unique_idx]
        rd.nc_phi_rad = np.arctan2(y_arr[unique_idx], x_arr[unique_idx])
        rd.nc_x_m     = x_arr[unique_idx].astype(np.float64)
        rd.nc_y_m     = y_arr[unique_idx].astype(np.float64)
        rd.nc_z_m     = z_arr[unique_idx]
        rd.nc_r_m     = np.sqrt(x_arr[unique_idx]**2 + y_arr[unique_idx]**2)

        # Material names were already resolved per-file; apply dedup index
        rd.nc_material_name = mat_name_arr[unique_idx]
        rd.nc_is_water      = rd.nc_material_name == "Water"

        # Cross-check: every NC with ge77_flag=1 must be in EnrichedGermanium0.913.
        # The reverse is not required — captures on Ge-70/72/73/74 are in germanium
        # but do not produce Ge-77.
        ge77_mask = rd.nc_ge77 == 1
        ge_mask   = rd.nc_material_name == "EnrichedGermanium0.913"
        n_flagged_not_ge = int((ge77_mask & ~ge_mask).sum())
        if n_flagged_not_ge > 0:
            raise RuntimeError(
                f"Ge77 flag / EnrichedGermanium0.913 mismatch in {run_dir}:\n"
                f"  ge77_flag=1 but NOT in EnrichedGermanium0.913: {n_flagged_not_ge} NCs"
            )

        # Gamma counts per NC
        if gamma_evtid_l:
            all_ge = np.concatenate(gamma_evtid_l)
            all_gn = np.concatenate(gamma_nc_id_l)
            gamma_pairs = np.stack([all_ge, all_gn], axis=1)
            upairs, gcounts = np.unique(gamma_pairs, axis=0, return_counts=True)
            gamma_lookup = {
                (int(p[0]), int(p[1])): int(c)
                for p, c in zip(upairs, gcounts)
            }
            del all_ge, all_gn, gamma_pairs, upairs, gcounts
        else:
            gamma_lookup = {}
        rd.nc_n_gammas = np.array(
            [gamma_lookup.get((int(e), int(t)), 0)
             for e, t in zip(rd.nc_evtid, unique_track_ids)],
            dtype=np.int32,
        )

        # Per-muon NC counts
        mu_nc_ids, counts = np.unique(rd.nc_evtid, return_counts=True)
        rd.muon_nc_evtids = mu_nc_ids
        rd.nc_counts = counts

        # Ge77 info
        ge77_nc_mask = rd.nc_ge77 == 1
        rd.ge77_evtids = np.unique(rd.nc_evtid[ge77_nc_mask])
        ge77_nc_evtids_only = rd.nc_evtid[ge77_nc_mask]
        _, ge77_nc_per = np.unique(ge77_nc_evtids_only, return_counts=True)
        rd.ge77_nc_per_ge77muon = ge77_nc_per

    # ---- Muon deduplication: keep first occurrence per evtid ----
    if mu_parts["evtid"]:
        evtid_all = np.concatenate(mu_parts["evtid"])
        _, first_idx = np.unique(evtid_all, return_index=True)
        rd.muon_evtid = evtid_all[first_idx]
        for k in ("ekin_mev", "zenith_deg", "azimuth_deg",
                  "x_m", "y_m", "z_m", "px_mev", "py_mev", "pz_mev"):
            arr = np.concatenate(mu_parts[k])
            setattr(rd, f"muon_{k}", arr[first_idx])
        rd.n_muons_total = rd.muon_evtid.size

    return rd


# ---------------------------------------------------------------------------
# Aggregation across all runs
# ---------------------------------------------------------------------------
def aggregate_runs(run_list: list[RunData]) -> dict:
    """Combine all runs into flat arrays; add boolean muon flags."""
    nc_ge77_l, nc_time_l, nc_phi_l, nc_z_l, nc_r_m_l, nc_is_ge77mu_l, nc_is_water_l, nc_n_gammas_l = [], [], [], [], [], [], [], []
    nc_counts_all_l, nc_counts_ge77mu_l, nc_counts_noge77mu_l = [], [], []
    ge77_nc_per_mu_l = []
    mu_ekin_l, mu_zen_l, mu_az_l = [], [], []
    mu_x_l, mu_y_l, mu_z_l = [], [], []
    mu_px_l, mu_py_l, mu_pz_l = [], [], []
    mu_is_ge77_l, mu_has_nc_l = [], []

    stats = dict(n_muons_total=0, n_nc_total=0, n_nc_ge77=0,
                 n_muons_ge77=0, n_muons_with_nc=0)

    for rd in run_list:
        stats["n_muons_total"] += rd.n_muons_total
        stats["n_nc_total"] += rd.nc_evtid.size
        stats["n_nc_ge77"] += int((rd.nc_ge77 == 1).sum())
        stats["n_muons_ge77"] += rd.ge77_evtids.size
        stats["n_muons_with_nc"] += rd.muon_nc_evtids.size

        if rd.nc_evtid.size > 0:
            nc_ge77_l.append(rd.nc_ge77)
            nc_time_l.append(rd.nc_time_ns)
            nc_phi_l.append(rd.nc_phi_rad)
            nc_z_l.append(rd.nc_z_m)
            nc_r_m_l.append(rd.nc_r_m)
            nc_is_ge77mu_l.append(np.isin(rd.nc_evtid, rd.ge77_evtids))
            nc_is_water_l.append(rd.nc_is_water)
            nc_n_gammas_l.append(rd.nc_n_gammas)

        if rd.nc_counts.size > 0:
            ge77_mu_mask = np.isin(rd.muon_nc_evtids, rd.ge77_evtids)
            nc_counts_all_l.append(rd.nc_counts)
            nc_counts_ge77mu_l.append(rd.nc_counts[ge77_mu_mask])
            nc_counts_noge77mu_l.append(rd.nc_counts[~ge77_mu_mask])

        if rd.ge77_nc_per_ge77muon.size > 0:
            ge77_nc_per_mu_l.append(rd.ge77_nc_per_ge77muon)

        if rd.muon_evtid.size > 0:
            is_ge77 = np.isin(rd.muon_evtid, rd.ge77_evtids)
            has_nc = np.isin(rd.muon_evtid, rd.muon_nc_evtids)
            mu_ekin_l.append(rd.muon_ekin_mev)
            mu_zen_l.append(rd.muon_zenith_deg)
            mu_az_l.append(rd.muon_azimuth_deg)
            mu_x_l.append(rd.muon_x_m)
            mu_y_l.append(rd.muon_y_m)
            mu_z_l.append(rd.muon_z_m)
            mu_px_l.append(rd.muon_px_mev)
            mu_py_l.append(rd.muon_py_mev)
            mu_pz_l.append(rd.muon_pz_mev)
            mu_is_ge77_l.append(is_ge77)
            mu_has_nc_l.append(has_nc)

    def cat(lst: list, dtype=None) -> np.ndarray:
        if not lst:
            return np.array([], dtype=dtype)
        return np.concatenate(lst)

    agg: dict = {
        # NC-level
        "nc_ge77": cat(nc_ge77_l, np.int32),
        "nc_time_ns": cat(nc_time_l, np.float64),
        "nc_phi_rad": cat(nc_phi_l, np.float64),
        "nc_z_m": cat(nc_z_l, np.float64),
        "nc_r_m": cat(nc_r_m_l, np.float64),
        "nc_is_ge77mu": cat(nc_is_ge77mu_l, bool),
        "nc_is_water":  cat(nc_is_water_l,  bool),
        "nc_n_gammas":  cat(nc_n_gammas_l,  np.int32),
        # Per-muon NC counts
        "nc_counts": cat(nc_counts_all_l, np.int64),
        "nc_counts_ge77mu": cat(nc_counts_ge77mu_l, np.int64),
        "nc_counts_noge77mu": cat(nc_counts_noge77mu_l, np.int64),
        "ge77_nc_per_mu": cat(ge77_nc_per_mu_l, np.int64),
        # Muon-level
        "mu_ekin_mev": cat(mu_ekin_l, np.float64),
        "mu_zenith_deg": cat(mu_zen_l, np.float64),
        "mu_azimuth_deg": cat(mu_az_l, np.float64),
        "mu_x_m": cat(mu_x_l, np.float64),
        "mu_y_m": cat(mu_y_l, np.float64),
        "mu_z_m": cat(mu_z_l, np.float64),
        "mu_px_mev": cat(mu_px_l, np.float64),
        "mu_py_mev": cat(mu_py_l, np.float64),
        "mu_pz_mev": cat(mu_pz_l, np.float64),
        "mu_is_ge77": cat(mu_is_ge77_l, bool),
        "mu_has_nc": cat(mu_has_nc_l, bool),
    }
    agg.update(stats)
    return agg


# ---------------------------------------------------------------------------
# Plot utilities
# ---------------------------------------------------------------------------
def make_log_bins(vmin: float, vmax: float) -> np.ndarray:
    """Linear bins within each decade: 1,2,...,9,10,20,...,90,100,..."""
    if vmax <= vmin or vmin <= 0:
        return np.array([max(vmin, 1e-10), max(vmax, 2e-10)])
    edges: list[float] = []
    decade = 10.0 ** int(np.floor(np.log10(max(vmin, 1e-10))))
    val = decade
    while val <= vmax * 1.01:
        edges.append(val)
        val += decade
        if val >= decade * 10:
            decade *= 10
    if not edges or edges[-1] < vmax:
        edges.append(val)
    return np.array(edges, dtype=float)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.name}")


def _draw_cylinder(ax: "plt.Axes", r: float = 4300.0,
                   z_min: float = -5000.0, z_max: float = 3900.0) -> None:
    """Draw a wireframe cylinder on a 3-D axes (dimensions in mm)."""
    theta = np.linspace(0, 2 * np.pi, 80)
    z_grid, th_grid = np.meshgrid([z_min, z_max], theta)
    ax.plot_surface(r * np.cos(th_grid), r * np.sin(th_grid), z_grid,
                    alpha=0.06, color="gray")
    for z in (z_min, z_max):
        ax.plot(r * np.cos(theta), r * np.sin(theta), z,
                color="gray", linewidth=0.8, alpha=0.5)


# ---------------------------------------------------------------------------
# Muon property plots  (4 variants × 3 observables = 12 PNGs)
# ---------------------------------------------------------------------------
def _single_hist(
    data: np.ndarray, color: str, title: str, xlabel: str,
    out_path: Path, bins: np.ndarray,
    log_x: bool = False, log_y: bool = False, normalized: bool = True,
) -> None:
    if data.size == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    w = np.ones(len(data)) / len(data) if normalized else None
    ax.hist(data, bins=bins, weights=w, color=color,
            edgecolor="black", linewidth=0.3, alpha=0.85)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Fraction" if normalized else "Count", fontsize=13)
    ax.set_title(f"{title}  (N = {len(data):,})", fontsize=14)
    ax.tick_params(labelsize=11)
    _save(fig, out_path)


def _comparison_hist(
    d1: np.ndarray, d2: np.ndarray,
    l1: str, l2: str, c1: str, c2: str,
    title: str, xlabel: str, out_path: Path,
    bins: np.ndarray,
    log_x: bool = False, log_y: bool = False, normalized: bool = True,
) -> None:
    if d1.size == 0 and d2.size == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for data, label, color in [(d1, l1, c1), (d2, l2, c2)]:
        if data.size == 0:
            continue
        w = np.ones(len(data)) / len(data) if normalized else None
        ax.hist(data, bins=bins, weights=w, color=color, alpha=0.7,
                edgecolor="black", linewidth=0.3, label=f"{label} (N={len(data):,})")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Fraction" if normalized else "Count", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    _save(fig, out_path)


def _ratio_plot(
    d1: np.ndarray, d2: np.ndarray,
    l1: str, l2: str,
    bins: np.ndarray,
    xlabel: str, out_path: Path,
    log_x: bool = False,
) -> None:
    """Two-panel figure: normalised-fraction ratio and raw-count ratio (d2 / d1).
    Bins where the d1 count is zero are skipped (no marker).
    """
    if d1.size == 0 or d2.size == 0:
        return
    h1_raw, _ = np.histogram(d1, bins=bins)
    h2_raw, _ = np.histogram(d2, bins=bins)
    h1_norm = h1_raw / d1.size
    h2_norm = h2_raw / d2.size
    bc = 0.5 * (bins[:-1] + bins[1:])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Ratio  {l2} / {l1}", fontsize=13)
    for ax, num, den, ylabel, title, zoom_y in [
        (axes[0], h2_norm, h1_norm,
         f"({l2} fraction) / ({l1} fraction)", "Normalised-fraction ratio", False),
        (axes[1], h2_raw.astype(float), h1_raw.astype(float),
         f"({l2} count) / ({l1} count)", "Raw-count ratio", True),
    ]:
        valid = den > 0
        ratio_vals = num[valid] / den[valid]
        ax.plot(bc[valid], ratio_vals,
                "o-", color=COLORS["red"], markersize=4, linewidth=1.2)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="ratio = 1")
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=10)
        if zoom_y and ratio_vals.size > 0:
            y_lo = float(ratio_vals.min())
            y_hi = float(ratio_vals.max())
            spread = y_hi - y_lo
            margin = max(0.15 * spread, 0.05)
            ax.set_ylim(y_lo - margin, y_hi + margin)
    plt.tight_layout()
    _save(fig, out_path)


def plot_muon_distributions(agg: dict, out_dir: Path) -> None:
    """Azimuth, zenith, energy — 4 variants each."""
    is_ge77 = agg["mu_is_ge77"]
    has_nc = agg["mu_has_nc"]
    ekin = agg["mu_ekin_mev"]
    zenith = agg["mu_zenith_deg"]
    azimuth = agg["mu_azimuth_deg"]

    observables = [
        {
            "data": ekin,
            "mask_ge77": is_ge77 & (ekin > 0),
            "mask_noge77": ~is_ge77 & (ekin > 0),
            "mask_nc": has_nc & (ekin > 0),
            "mask_nonc": ~has_nc & (ekin > 0),
            "all_mask": ekin > 0,
            "xlabel": "Muon kinetic energy [MeV]",
            "base": "muon_energy",
            "title": "Muon kinetic energy",
            "log_x": True,
            "bins_fn": lambda d: make_log_bins(max(1.0, float(d.min())), float(d.max())),
        },
        {
            "data": zenith,
            "mask_ge77": is_ge77,
            "mask_noge77": ~is_ge77,
            "mask_nc": has_nc,
            "mask_nonc": ~has_nc,
            "all_mask": np.ones(len(zenith), dtype=bool),
            "xlabel": "Zenith angle θ [°]",
            "base": "muon_zenith",
            "title": "Muon zenith angle",
            "log_x": False,
            "bins_fn": lambda _: np.linspace(0, 180, 91),
        },
        {
            "data": azimuth,
            "mask_ge77": is_ge77,
            "mask_noge77": ~is_ge77,
            "mask_nc": has_nc,
            "mask_nonc": ~has_nc,
            "all_mask": np.ones(len(azimuth), dtype=bool),
            "xlabel": "Azimuth angle φ [°]",
            "base": "muon_azimuth",
            "title": "Muon azimuth angle",
            "log_x": False,
            "bins_fn": lambda _: np.linspace(-180, 180, 91),
        },
    ]

    for obs in observables:
        d = obs["data"]
        m_all = obs["all_mask"]
        d_all = d[m_all]
        if d_all.size == 0:
            continue
        bins = obs["bins_fn"](d_all)
        lx = obs["log_x"]
        base = obs["base"]
        xl = obs["xlabel"]
        title = obs["title"]

        _comparison_hist(
            d[obs["mask_noge77"]], d[obs["mask_ge77"]],
            "Non-Ge77", "Ge77", COLORS["blue"], COLORS["red"],
            f"{title}: Ge77 vs non-Ge77", xl,
            out_dir / f"{base}_ge77_vs_noge77.png", bins, log_x=lx,
        )
        _ratio_plot(
            d[obs["mask_noge77"]], d[obs["mask_ge77"]],
            "Non-Ge77", "Ge77", bins, xl,
            out_dir / f"{base}_ge77_vs_noge77_ratio.png", log_x=lx,
        )
        _comparison_hist(
            d[obs["mask_nonc"]], d[obs["mask_nc"]],
            "No NC", "NC-producing", COLORS["blue"], COLORS["red"],
            f"{title}: NC-producing vs non-NC", xl,
            out_dir / f"{base}_nc_vs_nonc.png", bins, log_x=lx,
        )


def plot_ge77_fraction_bar(agg: dict, out_dir: Path) -> None:
    """Bar chart showing Ge77-producing muon fraction with count annotations."""
    n_total = agg["n_muons_total"]
    n_ge77 = agg["n_muons_ge77"]
    if n_total == 0:
        return
    n_other = n_total - n_ge77
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Ge77-producing", "Non-Ge77"],
        [n_ge77 / n_total, n_other / n_total],
        color=[COLORS["red"], COLORS["blue"]],
        edgecolor="black", linewidth=0.8,
    )
    for bar, count in zip(bars, [n_ge77, n_other]):
        ax.annotate(
            f"{count:,}\n({count / n_total:.3%})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", fontsize=11,
        )
    ax.set_ylabel("Fraction of muons", fontsize=13)
    ax.set_title(f"Ge77-producing muon fraction  (total: {n_total:,})", fontsize=14)
    ax.set_ylim(0, max(n_ge77, n_other) / n_total * 1.3)
    ax.tick_params(labelsize=11)
    _save(fig, out_dir / "ge77_fraction_bar.png")


# ---------------------------------------------------------------------------
# NC count per muon
# ---------------------------------------------------------------------------
def plot_nc_count_per_muon(agg: dict, out_dir: Path) -> None:
    """NC count per muon: comparison overlay (Ge77 vs non-Ge77) with log-spaced bins."""
    counts_all = agg["nc_counts"]
    counts_ge77 = agg["nc_counts_ge77mu"]
    counts_noge77 = agg["nc_counts_noge77mu"]

    if counts_all.size == 0:
        return

    max_count = int(counts_all.max())
    bins = np.logspace(0, np.log10(max(max_count, 10)), 50)

    fig, ax = plt.subplots(figsize=(10, 6))
    for data, label, color in [
        (counts_noge77, "Non-Ge77 NC-producing", COLORS["blue"]),
        (counts_ge77, "Ge77-producing", COLORS["red"]),
    ]:
        if data.size == 0:
            continue
        ax.hist(data, bins=bins, color=color, alpha=0.7,
                edgecolor="black", linewidth=0.3, label=f"{label} (N={len(data):,})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1.0]))
    ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))
    ax.set_xlabel("NC count per muon", fontsize=13)
    ax.set_ylabel("Number of muons", fontsize=13)
    ax.set_title("NC count per muon: Ge77 vs non-Ge77 NC-producing", fontsize=14)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    _save(fig, out_dir / "nc_count_per_muon_comparison.png")
    _ratio_plot(
        counts_noge77, counts_ge77,
        "Non-Ge77 NC-producing", "Ge77-producing", bins,
        "NC count per muon",
        out_dir / "nc_count_per_muon_comparison_ratio.png",
        log_x=True,
    )


# ---------------------------------------------------------------------------
# Ge77 NCs per Ge77 muon
# ---------------------------------------------------------------------------
def plot_ge77_per_ge77muon(agg: dict, out_dir: Path) -> None:
    """Bar chart: how many Ge77-flagged NCs does each Ge77 muon produce?"""
    data = agg["ge77_nc_per_mu"]
    if data.size == 0:
        return

    x_max = min(int(data.max()), 7)
    x_vals = np.arange(1, x_max + 1)
    n_total = len(data)
    bar_counts = np.array([(data == x).sum() for x in x_vals], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x_vals, bar_counts, color=COLORS["purple"],
                  edgecolor="black", linewidth=0.8)

    for bar, cnt in zip(bars, bar_counts):
        pct = cnt / n_total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.015,
            f"{cnt:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, rotation=90,
        )

    ax.set_xlabel("Number of Ge77-flagged NCs per Ge77-producing muon", fontsize=13)
    ax.set_ylabel("Number of Ge77-producing muons", fontsize=13)
    ax.set_title(
        f"Ge77 captures per Ge77-producing muon  (N = {n_total:,})", fontsize=14
    )
    ax.set_xticks(x_vals)
    ax.set_xlim(0.4, x_max + 0.6)
    ax.set_ylim(0, max(bar_counts) * 1.55)
    ax.tick_params(labelsize=11)
    _save(fig, out_dir / "ge77_per_ge77muon.png")


# ---------------------------------------------------------------------------
# NC time distributions
# ---------------------------------------------------------------------------
def plot_nc_times(agg: dict, out_dir: Path) -> None:
    """NC capture time: all muons + Ge77-muons (all NCs + Ge77-flagged)."""
    nc_time = agg["nc_time_ns"]
    nc_ge77_flag = agg["nc_ge77"]
    nc_is_ge77mu = agg["nc_is_ge77mu"]

    if nc_time.size == 0:
        return

    pos = nc_time > 0
    t_pos = nc_time[pos]
    if t_pos.size == 0:
        return
    bins = make_log_bins(max(1.0, float(t_pos.min())), float(t_pos.max()))

    # ---- All muons: all NCs + Ge77-flagged NCs highlighted ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(t_pos, bins=bins, color=COLORS["blue"], edgecolor="black",
            linewidth=0.3, label=f"All NCs (N={len(t_pos):,})")
    ge77_t = nc_time[pos & (nc_ge77_flag == 1)]
    if ge77_t.size > 0:
        ax.hist(ge77_t, bins=bins, color=COLORS["red"], edgecolor="black",
                linewidth=0.3, label=f"Ge77-flagged NCs (N={len(ge77_t):,})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("NC capture time [ns]", fontsize=13)
    ax.set_ylabel("Number of NCs", fontsize=13)
    ax.set_title("NC capture time — all muons", fontsize=14)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    _save(fig, out_dir / "nc_time_all.png")

    # ---- Ge77-producing muons only ----
    ge77mu_mask = pos & nc_is_ge77mu
    t_ge77mu = nc_time[ge77mu_mask]
    if t_ge77mu.size == 0:
        return
    bins2 = make_log_bins(max(1.0, float(t_ge77mu.min())), float(t_ge77mu.max()))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(t_ge77mu, bins=bins2, color=COLORS["blue"], edgecolor="black",
            linewidth=0.3,
            label=f"All NCs of Ge77 muons (N={len(t_ge77mu):,})")
    ge77_only = nc_time[ge77mu_mask & (nc_ge77_flag == 1)]
    if ge77_only.size > 0:
        ax.hist(ge77_only, bins=bins2, color=COLORS["red"], edgecolor="black",
                linewidth=0.3, label=f"Ge77-flagged NCs (N={len(ge77_only):,})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("NC capture time [ns]", fontsize=13)
    ax.set_ylabel("Number of NCs", fontsize=13)
    ax.set_title("NC capture time — Ge77-producing muons only", fontsize=14)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    _save(fig, out_dir / "nc_time_ge77muons.png")


# ---------------------------------------------------------------------------
# NC spatial distributions
# ---------------------------------------------------------------------------
def plot_nc_positions(agg: dict, out_dir: Path) -> None:
    """
    1-D histograms of φ, z, and r (all NCs vs Ge77-muon NCs) with ratio plots.
    2-D heatmaps: φ vs z and r vs z per subset (shared colorbar scale within subset).
    """
    phi = np.degrees(agg["nc_phi_rad"])   # convert to degrees for readability
    z_mm = agg["nc_z_m"] * 1e3           # metres → mm
    r_mm = agg["nc_r_m"] * 1e3           # metres → mm
    is_ge77mu = agg["nc_is_ge77mu"]

    if phi.size == 0:
        return

    # ---- 1-D φ: comparison overlay (all NCs vs Ge77-muon NCs) ----
    bins_phi = np.linspace(-180, 180, 73)
    phi_ge77mu = phi[is_ge77mu]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(phi, bins=bins_phi, color=COLORS["blue"], alpha=0.7,
            edgecolor="black", linewidth=0.3, label=f"All NCs (N={phi.size:,})")
    if phi_ge77mu.size > 0:
        ax.hist(phi_ge77mu, bins=bins_phi, color=COLORS["red"], alpha=0.7,
                edgecolor="black", linewidth=0.3,
                label=f"NCs of Ge77-producing muons (N={phi_ge77mu.size:,})")
    ax.set_xlabel("NC azimuth φ [°]", fontsize=13)
    ax.set_ylabel("Number of NCs", fontsize=13)
    ax.set_title("NC azimuthal position: all NCs vs Ge77-muon NCs", fontsize=14)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    _save(fig, out_dir / "nc_phi_1d_comparison.png")
    _ratio_plot(
        phi, phi_ge77mu,
        "All NCs", "NCs of Ge77-producing muons", bins_phi,
        "NC azimuth φ [°]",
        out_dir / "nc_phi_1d_comparison_ratio.png",
    )

    # ---- 1-D z: comparison overlay (all NCs vs Ge77-muon NCs) ----
    bins_z = np.linspace(float(z_mm.min()), float(z_mm.max()), 100)
    z_ge77mu = z_mm[is_ge77mu]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(z_mm, bins=bins_z, color=COLORS["blue"], alpha=0.7,
            edgecolor="black", linewidth=0.3, label=f"All NCs (N={z_mm.size:,})")
    if z_ge77mu.size > 0:
        ax.hist(z_ge77mu, bins=bins_z, color=COLORS["red"], alpha=0.7,
                edgecolor="black", linewidth=0.3,
                label=f"NCs of Ge77-producing muons (N={z_ge77mu.size:,})")
    ax.set_xlabel("NC z position [mm]", fontsize=13)
    ax.set_ylabel("Number of NCs", fontsize=13)
    ax.set_title("NC z position: all NCs vs Ge77-muon NCs", fontsize=14)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    _save(fig, out_dir / "nc_z_1d_comparison.png")
    _ratio_plot(
        z_mm, z_ge77mu,
        "All NCs", "NCs of Ge77-producing muons", bins_z,
        "NC z position [mm]",
        out_dir / "nc_z_1d_comparison_ratio.png",
    )

    # ---- 1-D r: comparison overlay (all NCs vs Ge77-muon NCs) ----
    bins_r = np.linspace(0.0, float(r_mm.max()), 73)
    r_ge77mu = r_mm[is_ge77mu]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(r_mm, bins=bins_r, color=COLORS["blue"], alpha=0.7,
            edgecolor="black", linewidth=0.3, label=f"All NCs (N={r_mm.size:,})")
    if r_ge77mu.size > 0:
        ax.hist(r_ge77mu, bins=bins_r, color=COLORS["red"], alpha=0.7,
                edgecolor="black", linewidth=0.3,
                label=f"NCs of Ge77-producing muons (N={r_ge77mu.size:,})")
    ax.set_xlabel("NC radial position r [mm]", fontsize=13)
    ax.set_ylabel("Number of NCs", fontsize=13)
    ax.set_title("NC radial position: all NCs vs Ge77-muon NCs", fontsize=14)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    _save(fig, out_dir / "nc_r_1d_comparison.png")
    _ratio_plot(
        r_mm, r_ge77mu,
        "All NCs", "NCs of Ge77-producing muons", bins_r,
        "NC radial position r [mm]",
        out_dir / "nc_r_1d_comparison_ratio.png",
    )

    # ---- 2-D φ vs z AND r vs z (shared colorbar scale per subset) ----
    for mask, tag, title_base in [
        (np.ones(phi.size, dtype=bool), "all", "NC positions — all NCs"),
        (is_ge77mu, "ge77muons", "NC positions — NCs of Ge77-producing muons"),
    ]:
        ph = phi[mask]
        zm = z_mm[mask]
        rm = r_mm[mask]
        if ph.size == 0:
            continue

        H_phi, _, _ = np.histogram2d(ph, zm, bins=[72, 100])
        H_r,   _, _ = np.histogram2d(rm, zm, bins=[72, 100])
        vmax = max(float(H_phi.max()), float(H_r.max()))

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f"{title_base}  (N = {ph.size:,})", fontsize=14)

        hh1 = axes[0].hist2d(ph, zm, bins=[72, 100], cmap="viridis", vmin=0, vmax=vmax)
        axes[0].set_xlabel("NC azimuth φ [°]", fontsize=13)
        axes[0].set_ylabel("NC z position [mm]", fontsize=13)
        axes[0].set_title("φ vs z", fontsize=12)
        axes[0].tick_params(labelsize=11)

        axes[1].hist2d(rm, zm, bins=[72, 100], cmap="viridis", vmin=0, vmax=vmax)
        axes[1].set_xlabel("NC radial position r [mm]", fontsize=13)
        axes[1].set_ylabel("NC z position [mm]", fontsize=13)
        axes[1].set_title("r vs z", fontsize=12)
        axes[1].tick_params(labelsize=11)

        fig.colorbar(hh1[3], ax=axes, label="Number of NCs")
        plt.tight_layout()
        _save(fig, out_dir / f"nc_phi_r_z_2d_{tag}.png")


# ---------------------------------------------------------------------------
# 3-D muon position scatter plots
# ---------------------------------------------------------------------------
def _plot_3d_scatter(
    x_mm: np.ndarray, y_mm: np.ndarray, z_mm: np.ndarray,
    px: np.ndarray, py: np.ndarray, pz: np.ndarray,
    title: str, out_path: Path,
    max_pts: int = MAX_SCATTER_PTS,
) -> None:
    """3-D scatter of muon entry points with down-sampled momentum arrows."""
    n = len(x_mm)
    if n == 0:
        return
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(n, min(n, max_pts), replace=False)
    xs, ys, zs = x_mm[idx], y_mm[idx], z_mm[idx]
    pxs, pys, pzs = px[idx], py[idx], pz[idx]

    p_mag = np.sqrt(pxs**2 + pys**2 + pzs**2)
    p_mag = np.where(p_mag > 0, p_mag, 1.0)
    scale = 500.0   # arrow length in mm
    dx, dy, dz = pxs / p_mag * scale, pys / p_mag * scale, pzs / p_mag * scale

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c="red", s=8, alpha=0.5, label=f"Muon vertex (N={n:,})")
    for i in range(len(xs)):
        ax.plot([xs[i], xs[i] + dx[i]], [ys[i], ys[i] + dy[i]],
                [zs[i], zs[i] + dz[i]], color="red", alpha=0.2, linewidth=0.4)
    _draw_cylinder(ax)
    ax.set_xlabel("X [mm]", fontsize=11)
    ax.set_ylabel("Y [mm]", fontsize=11)
    ax.set_zlabel("Z [mm]", fontsize=11)
    ax.set_title(
        f"{title}  (N = {n:,}; showing {min(n, max_pts):,})", fontsize=13
    )
    ax.legend(fontsize=10)
    _save(fig, out_path)


def plot_muon_3d_scatters(agg: dict, out_dir: Path) -> None:
    """3-D position scatter for Ge77 muons and NC-producing muons."""
    x_mm = agg["mu_x_m"] * 1e3
    y_mm = agg["mu_y_m"] * 1e3
    z_mm = agg["mu_z_m"] * 1e3
    px = agg["mu_px_mev"]
    py = agg["mu_py_mev"]
    pz = agg["mu_pz_mev"]
    is_ge77 = agg["mu_is_ge77"]
    has_nc = agg["mu_has_nc"]

    for mask, fname, title in [
        (is_ge77, "muon_ge77_3d.png", "Ge77-producing muon entry vertices"),
        (has_nc, "muon_nc_3d.png", "NC-producing muon entry vertices"),
    ]:
        if mask.sum() == 0:
            continue
        _plot_3d_scatter(
            x_mm[mask], y_mm[mask], z_mm[mask],
            px[mask], py[mask], pz[mask],
            title, out_dir / fname,
        )


# ---------------------------------------------------------------------------
# NC material distribution
# ---------------------------------------------------------------------------
def plot_nc_material_distribution(run_list: list[RunData], out_dir: Path) -> None:
    """Grouped bar chart of NC material fractions: Ge77-muon NCs vs non-Ge77-muon NCs,
    plus a ratio panel. Materials are sorted by total NC count (descending)."""
    counts_ge77:   dict[str, int] = {}
    counts_noge77: dict[str, int] = {}

    for rd in run_list:
        if rd.nc_evtid.size == 0 or rd.nc_material_name.size == 0:
            continue
        is_ge77mu = np.isin(rd.nc_evtid, rd.ge77_evtids)
        for name in np.unique(rd.nc_material_name):
            mat_mask = rd.nc_material_name == name
            counts_ge77[name]   = counts_ge77.get(name, 0)   + int((mat_mask & is_ge77mu).sum())
            counts_noge77[name] = counts_noge77.get(name, 0) + int((mat_mask & ~is_ge77mu).sum())

    all_names = sorted(
        set(counts_ge77) | set(counts_noge77),
        key=lambda n: counts_ge77.get(n, 0) + counts_noge77.get(n, 0),
        reverse=True,
    )
    if not all_names:
        return

    n_ge77   = sum(counts_ge77.values())
    n_noge77 = sum(counts_noge77.values())
    ge77_f   = np.array([counts_ge77.get(n, 0)   / max(n_ge77,   1) for n in all_names])
    noge77_f = np.array([counts_noge77.get(n, 0) / max(n_noge77, 1) for n in all_names])

    x = np.arange(len(all_names))
    w = 0.35
    fig_w = max(10, len(all_names) * 1.6 + 4)

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 6))
    fig.suptitle("NC material distribution: Ge77-muon NCs vs non-Ge77-muon NCs",
                 fontsize=13)

    # ---- Left: overlaid grouped bars (normalised fractions) ----
    ax = axes[0]
    ax.bar(x - w / 2, noge77_f, w,
           label=f"Non-Ge77 muon NCs (N={n_noge77:,})",
           color=COLORS["blue"], edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.bar(x + w / 2, ge77_f, w,
           label=f"Ge77-muon NCs (N={n_ge77:,})",
           color=COLORS["red"],  edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of NCs", fontsize=12)
    ax.set_title("Normalised fraction per material", fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=9)

    # ---- Right: ratio (Ge77 fraction / non-Ge77 fraction per material) ----
    ax2 = axes[1]
    valid = noge77_f > 0
    ratios = np.where(valid, ge77_f / np.where(valid, noge77_f, 1.0), np.nan)
    bars = ax2.bar(x[valid], ratios[valid],
                   color=COLORS["red"], edgecolor="black", linewidth=0.5, alpha=0.85)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="ratio = 1")
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_names, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("(Ge77 fraction) / (non-Ge77 fraction)", fontsize=11)
    ax2.set_title("Ratio per material", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.tick_params(labelsize=9)

    plt.tight_layout()
    _save(fig, out_dir / "nc_material_ge77_vs_noge77.png")


# ---------------------------------------------------------------------------
# NC gamma count distribution
# ---------------------------------------------------------------------------
def plot_nc_gamma_count(agg: dict, out_dir: Path) -> None:
    """Normalised histogram of capture-gamma multiplicity: Ge77-muon NCs vs non-Ge77."""
    n_gammas  = agg["nc_n_gammas"]
    is_ge77mu = agg["nc_is_ge77mu"]

    if n_gammas.size == 0:
        return

    ge77_g   = n_gammas[is_ge77mu]
    noge77_g = n_gammas[~is_ge77mu]

    max_g = int(n_gammas.max())
    bins  = np.arange(0, max_g + 2) - 0.5   # integer-centred half-open bins

    _comparison_hist(
        noge77_g, ge77_g,
        "Non-Ge77 muon NCs", "Ge77-muon NCs",
        COLORS["blue"], COLORS["red"],
        "Capture-gamma multiplicity: Ge77 vs non-Ge77 muons",
        "Number of capture gammas per NC",
        out_dir / "nc_gamma_count_ge77_vs_noge77.png",
        bins, log_x=False, log_y=True,
    )
    _ratio_plot(
        noge77_g, ge77_g,
        "Non-Ge77 muon NCs", "Ge77-muon NCs", bins,
        "Number of capture gammas per NC",
        out_dir / "nc_gamma_count_ge77_vs_noge77_ratio.png",
    )


# ---------------------------------------------------------------------------
# Outlier analysis
# ---------------------------------------------------------------------------
def analyze_outliers(agg: dict, out_dir: Path) -> dict:
    """
    Outlier muons = NC count above 99th percentile on log scale.
    threshold = exp(np.percentile(np.log(nc_counts), 99))
    """
    counts = agg["nc_counts"]
    if counts.size == 0:
        return {}

    threshold = float(np.exp(np.percentile(np.log(counts), 99)))
    outlier_mask = counts > threshold
    n_outliers = int(outlier_mask.sum())
    print(f"\n  Outlier threshold (99th pct log scale): {threshold:.1f} NCs/muon")
    print(f"  Outlier muons: {n_outliers:,}  "
          f"({n_outliers / len(counts) * 100:.2f}% of NC-producing muons)")

    # Fraction of total NCs from outlier muons
    outlier_nc_total = int(counts[outlier_mask].sum())
    total_nc = int(agg["n_nc_total"])
    frac_nc = outlier_nc_total / total_nc if total_nc > 0 else 0.0
    print(f"  NCs from outlier muons: {outlier_nc_total:,} ({frac_nc * 100:.2f}% of all NCs)")

    # ---- Plot 1: NC count distribution with outlier threshold ----
    bins = make_log_bins(1, int(counts.max()))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(counts, bins=bins, color=COLORS["blue"], edgecolor="black", linewidth=0.4,
            label="All NC-producing muons")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"99th pct threshold = {threshold:.0f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("NC count per muon", fontsize=13)
    ax.set_ylabel("Number of muons", fontsize=13)
    ax.set_title(
        f"NC count per muon with outlier threshold  "
        f"({n_outliers:,} outliers = {frac_nc * 100:.1f}% of NCs)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    _save(fig, out_dir / "outlier_nc_count.png")

    max_single_count = int(counts[outlier_mask].max()) if n_outliers > 0 else 0

    if total_nc > 0 and max_single_count / total_nc > 0.05:
        print(
            f"\n  *** WARNING: single outlier muon has {max_single_count:,} NCs "
            f"= {max_single_count / total_nc * 100:.1f}% of all NCs. "
            f"NC spatial distribution may be dominated by this muon. ***"
        )

    return {
        "n_outliers": n_outliers,
        "threshold": threshold,
        "frac_nc": frac_nc,
        "max_single_count": max_single_count,
    }


# ---------------------------------------------------------------------------
# Outlier muon fingerprint analysis
# ---------------------------------------------------------------------------
def _muon_fingerprint(rd: RunData, evtid: int) -> tuple[float, ...] | None:
    """Return (ekin, px, py, pz, x, y, z) for evtid, or None if not found.

    muon_evtid is sorted (np.unique in load_run), so searchsorted is O(log n).
    """
    idx = int(np.searchsorted(rd.muon_evtid, evtid))
    if idx >= len(rd.muon_evtid) or rd.muon_evtid[idx] != evtid:
        return None
    return (
        float(rd.muon_ekin_mev[idx]),
        float(rd.muon_px_mev[idx]),
        float(rd.muon_py_mev[idx]),
        float(rd.muon_pz_mev[idx]),
        float(rd.muon_x_m[idx]),
        float(rd.muon_y_m[idx]),
        float(rd.muon_z_m[idx]),
    )


def _fp_rounded(fp: tuple[float, ...], sigfigs: int = 5) -> tuple[float, ...]:
    """Round each component of a fingerprint to sigfigs significant figures."""
    out: list[float] = []
    for v in fp:
        if v == 0.0:
            out.append(0.0)
        else:
            mag = 10 ** (sigfigs - 1 - int(np.floor(np.log10(abs(v)))))
            out.append(round(v * mag) / mag)
    return tuple(out)


def analyze_repeated_outlier_muons(
    run_list: list[RunData],
    out_dir: Path,
    top_n: int = 10,
) -> None:
    """Check whether the top-N highest-NC muons repeat within or across runs.

    Within each run: scans the top_n*10 outlier muons for pairs of different
    evtids sharing the same kinematic fingerprint — a MUSUN trajectory replay.

    Across runs: compares the top-N fingerprints from every run and counts
    in how many runs each fingerprint appears, using both exact float equality
    and a 5-significant-figure tolerance.

    Produces one PNG heatmap: rows = unique fingerprints (sorted by max NC
    count), columns = runs, colour = NC count (log scale), grey = absent.
    """
    n_runs = len(run_list)
    if n_runs == 0:
        return

    # ------------------------------------------------------------------ #
    # 1. Collect top-N fingerprints per run                               #
    # ------------------------------------------------------------------ #
    # run_top[i] = list of (nc_count, fingerprint) sorted desc by nc_count
    run_top: list[list[tuple[int, tuple[float, ...]]]] = []

    for rd in run_list:
        if rd.nc_counts.size == 0:
            run_top.append([])
            continue
        n = min(top_n, len(rd.nc_counts))
        top_idx = np.argsort(rd.nc_counts)[-n:][::-1]
        entries: list[tuple[int, tuple[float, ...]]] = []
        for i in top_idx:
            evtid = int(rd.muon_nc_evtids[i])
            nc_count = int(rd.nc_counts[i])
            fp = _muon_fingerprint(rd, evtid)
            if fp is not None:
                entries.append((nc_count, fp))
        run_top.append(entries)

    # ------------------------------------------------------------------ #
    # 2. Within-run duplicate check                                       #
    # ------------------------------------------------------------------ #
    # Among the top_n*10 NC-producing muons in each run, look for two
    # different evtids that share the same rounded fingerprint.
    print(f"\n  Within-run fingerprint check (top-{top_n * 10} NC muons per run):")
    any_within = False
    for rd in run_list:
        if rd.nc_counts.size == 0:
            continue
        n_check = min(top_n * 10, len(rd.nc_counts))
        check_idx = np.argsort(rd.nc_counts)[-n_check:]

        fp_seen: dict[tuple, list[int]] = {}
        for i in check_idx:
            evtid = int(rd.muon_nc_evtids[i])
            fp = _muon_fingerprint(rd, evtid)
            if fp is None:
                continue
            fp_r = _fp_rounded(fp)
            fp_seen.setdefault(fp_r, []).append(evtid)

        dups = {fp: evtids for fp, evtids in fp_seen.items() if len(evtids) > 1}
        if dups:
            any_within = True
            print(f"    {rd.run_name}: {len(dups)} repeated trajectory(ies) "
                  f"among top-{n_check}")
            for fp_r, evtids in list(dups.items())[:3]:
                print(f"      evtids {evtids[:6]}  ekin={fp_r[0]:.5g} MeV  "
                      f"x={fp_r[4]:.4g} m")
        else:
            print(f"    {rd.run_name}: no duplicates among top-{n_check}")

    if not any_within:
        print("    → No within-run trajectory replays detected.")

    # ------------------------------------------------------------------ #
    # 3. Across-run fingerprint matching                                  #
    # ------------------------------------------------------------------ #
    # fp_exact[fp]   = {run_idx: nc_count}
    # fp_tol[fp_r]   = {run_idx: nc_count}  (5 sig-fig rounded keys)
    fp_exact: dict[tuple, dict[int, int]] = {}
    fp_tol:   dict[tuple, dict[int, int]] = {}

    for run_idx, entries in enumerate(run_top):
        for nc_count, fp in entries:
            d = fp_exact.setdefault(fp, {})
            d[run_idx] = max(d.get(run_idx, 0), nc_count)

            fp_r = _fp_rounded(fp)
            d2 = fp_tol.setdefault(fp_r, {})
            d2[run_idx] = max(d2.get(run_idx, 0), nc_count)

    multi_exact = sum(1 for d in fp_exact.values() if len(d) > 1)
    multi_tol   = sum(1 for d in fp_tol.values()   if len(d) > 1)
    n_exact = len(fp_exact)
    n_tol   = len(fp_tol)

    print(f"\n  Across-run match (exact):          "
          f"{multi_exact}/{n_exact} fingerprints in >1 run")
    print(f"  Across-run match (5 sig-fig tol.): "
          f"{multi_tol}/{n_tol} fingerprints in >1 run")

    for fp, run_nc in sorted(fp_exact.items(),
                             key=lambda kv: max(kv[1].values()), reverse=True):
        if len(run_nc) > 1:
            runs_str = ", ".join(
                f"{run_list[ri].run_name}(NC={nc:,})"
                for ri, nc in sorted(run_nc.items())
            )
            print(f"    ekin={fp[0]:.5g} MeV  x={fp[4]:.4g} m  → {runs_str}")

    # ------------------------------------------------------------------ #
    # 4. Heatmap                                                          #
    # ------------------------------------------------------------------ #
    sorted_fps = sorted(
        fp_exact.keys(),
        key=lambda fp: max(fp_exact[fp].values()),
        reverse=True,
    )
    n_fps = len(sorted_fps)
    if n_fps == 0:
        return

    nc_matrix = np.full((n_fps, n_runs), np.nan)
    for row, fp in enumerate(sorted_fps):
        for col, nc_count in fp_exact[fp].items():
            nc_matrix[row, col] = float(nc_count)

    run_names = [rd.run_name for rd in run_list]
    row_labels = []
    for fp in sorted_fps:
        max_nc = max(fp_exact[fp].values())
        n_present = len(fp_exact[fp])
        row_labels.append(
            f"E={fp[0] / 1000:.2f} TeV  "
            f"p=({fp[1]:.0f},{fp[2]:.0f},{fp[3]:.0f}) MeV  "
            f"[{n_present}/{n_runs}]  max NC={max_nc:,}"
        )

    valid = nc_matrix[~np.isnan(nc_matrix)]
    vmin = float(valid.min()) if valid.size > 0 else 1.0
    vmax = float(valid.max()) if valid.size > 0 else 2.0
    norm = mcolors.LogNorm(vmin=max(vmin, 1.0), vmax=max(vmax, 2.0))
    text_thresh = np.exp(0.5 * (np.log(max(vmin, 1.0)) + np.log(max(vmax, 1.0))))

    fig_h = max(5.0, n_fps * 0.75 + 3.0)
    fig_w = max(9.0, n_runs * 1.5 + 5.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.plasma.copy()
    cmap.set_bad("#d0d0d0")
    im = ax.imshow(nc_matrix, aspect="auto", cmap=cmap, norm=norm)
    plt.colorbar(im, ax=ax, label="NC count (log scale)", pad=0.01)

    ax.set_xticks(range(n_runs))
    ax.set_xticklabels(run_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_fps))
    ax.set_yticklabels(row_labels, fontsize=7)

    for row in range(n_fps):
        for col in range(n_runs):
            v = nc_matrix[row, col]
            if not np.isnan(v):
                txt_color = "white" if v > text_thresh else "black"
                ax.text(col, row, f"{int(v):,}",
                        ha="center", va="center", fontsize=7, color=txt_color)
            else:
                ax.text(col, row, "—",
                        ha="center", va="center", fontsize=8, color="#aaaaaa")

    ax.set_title(
        f"Top-{top_n} NC muons per run — kinematic fingerprint matching\n"
        f"Exact: {multi_exact}/{n_exact} fingerprints in >1 run  |  "
        f"Tolerance (5 s.f.): {multi_tol}/{n_tol} in >1 run",
        fontsize=12,
    )
    ax.set_xlabel("Simulation run", fontsize=11)
    ax.set_ylabel(
        f"Muon fingerprint — top-{top_n} per run, sorted by max NC count",
        fontsize=10,
    )
    plt.tight_layout()
    _save(fig, out_dir / "outlier_muon_fingerprint_heatmap.png")


# ---------------------------------------------------------------------------
# Material breakdown of cut NCs (muons with >NC_MUON_CUT NCs)
# ---------------------------------------------------------------------------
NC_MUON_CUT = 1_000


def plot_cut_nc_material(run_list: list[RunData], out_dir: Path) -> None:
    """Bar chart: water vs other material for NCs from muons with >NC_MUON_CUT NCs."""
    n_water = 0
    n_other = 0
    for rd in run_list:
        if rd.nc_counts.size == 0:
            continue
        cut_mask = rd.nc_counts > NC_MUON_CUT
        if not cut_mask.any():
            continue
        cut_evtids = rd.muon_nc_evtids[cut_mask]
        nc_sel = np.isin(rd.nc_evtid, cut_evtids)
        n_water += int(rd.nc_is_water[nc_sel].sum())
        n_other += int((~rd.nc_is_water[nc_sel]).sum())

    total = n_water + n_other
    if total == 0:
        print(f"  No NCs from muons with >{NC_MUON_CUT:,} NCs; skipping material plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Water", "Other material"],
        [n_water / total, n_other / total],
        color=[COLORS["blue"], COLORS["orange"]],
        edgecolor="black", linewidth=0.8,
    )
    for bar, count in zip(bars, [n_water, n_other]):
        ax.annotate(
            f"{count:,}\n({count / total:.3%})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", fontsize=11,
        )
    ax.set_ylabel("Fraction of NCs", fontsize=13)
    ax.set_title(
        f"Material of NCs from muons with >{NC_MUON_CUT:,} NCs  (total: {total:,})",
        fontsize=13,
    )
    ax.set_ylim(0, max(n_water, n_other) / total * 1.3)
    ax.tick_params(labelsize=11)
    _save(fig, out_dir / "cut_nc_material.png")


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------
def _transform(data: np.ndarray, obs: str) -> np.ndarray:
    """Apply observable-specific transform before computing distances."""
    if obs in ("nc_count", "energy"):
        return np.log(data + 1.0)
    return data   # zenith and azimuth: raw degrees


def _w1_sorted(sorted_p: np.ndarray, sorted_q: np.ndarray) -> float:
    """Wasserstein-1 distance between two *pre-sorted* empirical distributions.

    Memory layout (peak = 3 × (n+m) × 8 bytes):
      1. all_vals  — merged sorted union, freed after np.diff
      2. deltas    — interval widths (nm-1,)
      3. cdf_diff  — |F_p - F_q| at each breakpoint (nm-1,)

    Returns np.nan if either array is empty.
    """
    n, m = len(sorted_p), len(sorted_q)
    if n == 0 or m == 0:
        return float("nan")

    nm = n + m
    all_vals = np.empty(nm, dtype=np.float64)
    all_vals[:n] = sorted_p
    all_vals[n:] = sorted_q
    all_vals.sort(kind="mergesort")   # O(nm): fast for two sorted halves

    deltas = np.diff(all_vals)        # (nm-1,) — new allocation
    bpts = all_vals[:-1]              # view, no copy

    cdf_diff = (np.searchsorted(sorted_p, bpts, side="right") / n
                - np.searchsorted(sorted_q, bpts, side="right") / m)
    del all_vals
    np.abs(cdf_diff, out=cdf_diff)

    w1 = float(np.dot(cdf_diff, deltas))
    del cdf_diff, deltas
    return w1


def _merge_sorted_runs(sorted_runs: list[np.ndarray],
                       indices: list[int]) -> np.ndarray:
    """Concatenate pre-sorted per-run arrays and mergesort the result.

    Using numpy's 'mergesort' on k pre-sorted blocks of total size N is
    O(N log k), much faster than O(N log N) for unsorted data.
    """
    total = sum(len(sorted_runs[i]) for i in indices)
    buf = np.empty(total, dtype=np.float64)
    off = 0
    for i in indices:
        n_i = len(sorted_runs[i])
        buf[off:off + n_i] = sorted_runs[i]
        off += n_i
    buf.sort(kind="mergesort")
    return buf


def convergence_analysis(
    run_list: list[RunData], out_dir: Path, full_convergence: bool = False
) -> dict[str, int]:
    """
    Compute W1 convergence metrics vs number of runs k.

    For NC count per muon: produces W1 figures for the full distribution
    and cut distributions (cuts at NC_CUT_100 and NC_MUON_CUT).

    If full_convergence=True, also produces W1 figures for zenith,
    azimuth, and energy.

    Returns recommended minimum k per observable (threshold = 5% of W1(k=1)).
    """
    n_runs = len(run_list)
    if n_runs < 2:
        print("  Convergence analysis requires >= 2 runs; skipping.")
        return {}

    rng = random.Random(RANDOM_SEED)
    run_indices = list(range(n_runs))
    k_vals = list(range(1, n_runs + 1))

    def _compute_metrics(
        sorted_runs_local: list[np.ndarray],
    ) -> tuple[list[float], np.ndarray]:
        """W1 vs k for given pre-sorted per-run arrays."""
        sorted_ref = _merge_sorted_runs(sorted_runs_local, run_indices)
        if sorted_ref.size == 0:
            del sorted_ref
            return [], np.zeros((n_runs, N_PERMUTATIONS))
        det_w1: list[float] = []
        for k in k_vals:
            sk = _merge_sorted_runs(sorted_runs_local, list(range(k)))
            w1 = _w1_sorted(sk, sorted_ref)
            del sk
            det_w1.append(w1 if not np.isnan(w1) else 0.0)
        rand_w1_m = np.zeros((n_runs, N_PERMUTATIONS))
        for perm_i in range(N_PERMUTATIONS):
            shuffled = run_indices.copy()
            rng.shuffle(shuffled)
            for k in k_vals:
                sk = _merge_sorted_runs(sorted_runs_local, shuffled[:k])
                w1 = _w1_sorted(sk, sorted_ref)
                del sk
                rand_w1_m[k - 1, perm_i] = w1 if not np.isnan(w1) else 0.0
        del sorted_ref
        return det_w1, rand_w1_m

    def _rec_k_from_w1(det_w1: list[float]) -> tuple[int, float]:
        if not det_w1 or det_w1[0] <= 0:
            return n_runs, 0.0
        thr = W1_THRESHOLD_FRAC * det_w1[0]
        rec = next((k for k, w in enumerate(det_w1, 1) if w <= thr), n_runs)
        return rec, thr

    def _draw_w1_panel(
        ax: "plt.Axes",
        title: str,
        det_vals: list[float],
        r_mean: np.ndarray,
        r_std: np.ndarray,
        ylabel: str,
        threshold: float | None,
        rec_k: int | None,
        w1_at_k1: float | None = None,
        show_deterministic: bool = True,
    ) -> None:
        if show_deterministic:
            ax.plot(k_vals, det_vals, "o-", color=COLORS["blue"],
                    linewidth=1.8, markersize=5, label="Deterministic (cumulative)")
        ax.plot(k_vals, r_mean, "s--", color=COLORS["orange"],
                linewidth=1.5, markersize=5, label="Random subsets (mean)")
        ax.fill_between(k_vals, r_mean - r_std, r_mean + r_std,
                        color=COLORS["orange"], alpha=0.25,
                        label=f"±1σ  ({N_PERMUTATIONS} permutations)")
        ax.axhline(0, color="gray", linestyle=":", linewidth=1.0,
                   label="Reference")
        if threshold and threshold > 0 and rec_k is not None:
            ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
                       label=f"5% threshold = {threshold:.4f}")
            ax.axvline(rec_k, color="red", linestyle=":", linewidth=1.2,
                       label=f"Rec. k = {rec_k}")
        ax.set_xlabel("Number of runs k", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(k_vals)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=9)
        if w1_at_k1 is not None and w1_at_k1 > 0:
            ax2 = ax.twinx()
            y_lo, y_hi = ax.get_ylim()
            ax2.set_ylim(y_lo / w1_at_k1 * 100.0, y_hi / w1_at_k1 * 100.0)
            ax2.set_ylabel("% of W₁(k=1)", fontsize=10)
            ax2.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
            ax2.tick_params(labelsize=9)

    recommendations: dict[str, int] = {}
    t_conv = _log_resources("convergence start")

    print("\n  Convergence: NC count per muon ...", flush=True)

    nc_full_raw:   list[np.ndarray] = []
    nc_cut100_raw: list[np.ndarray] = []
    nc_cut_raw:    list[np.ndarray] = []
    for rd in run_list:
        full   = rd.nc_counts.astype(np.float64)
        cut100 = full[full <= NC_CUT_100]
        cut    = full[full <= NC_MUON_CUT]
        nc_full_raw.append(np.sort(full))
        nc_cut100_raw.append(np.sort(cut100) if cut100.size > 0
                             else np.array([], dtype=np.float64))
        nc_cut_raw.append(np.sort(cut) if cut.size > 0
                          else np.array([], dtype=np.float64))

    dw1_fr,    rw1_fr    = _compute_metrics(nc_full_raw)
    dw1_c100r, rw1_c100r = _compute_metrics(nc_cut100_raw)
    dw1_cr,    rw1_cr    = _compute_metrics(nc_cut_raw)
    del nc_full_raw, nc_cut100_raw, nc_cut_raw
    gc.collect()
    _log_resources("nc_count: all metrics computed", t_conv)

    rec_full,   thr_fr    = _rec_k_from_w1(dw1_fr)
    rec_cut100, thr_c100r = _rec_k_from_w1(dw1_c100r)
    rec_cut,    thr_cr    = _rec_k_from_w1(dw1_cr)
    recommendations["nc_count"]         = rec_full
    recommendations["nc_count_cut_100"] = rec_cut100
    recommendations["nc_count_cut"]     = rec_cut
    print(f"    Full dist       — W1 thr = {thr_fr:.4f}    →  rec k = {rec_full}")
    print(f"    Cut ≤{NC_CUT_100} dist  — W1 thr = {thr_c100r:.4f}  →  rec k = {rec_cut100}")
    print(f"    Cut ≤{NC_MUON_CUT:,} dist — W1 thr = {thr_cr:.4f}    →  rec k = {rec_cut}")

    # Full distribution: 1×1 W1 figure
    if dw1_fr:
        rw1_mean = rw1_fr.mean(axis=1)
        rw1_std  = rw1_fr.std(axis=1)
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.suptitle("Convergence: NC count per muon — full distribution", fontsize=14)
        _draw_w1_panel(ax, "W₁  —  Full (all muons)",
                       dw1_fr, rw1_mean, rw1_std,
                       "W₁ (NC count)", None, None,
                       w1_at_k1=dw1_fr[0] if dw1_fr[0] > 0 else None,
                       show_deterministic=False)
        plt.tight_layout()
        _save(fig, out_dir / "convergence_nc_count_full.png")

    # Cut distributions: 2×1 W1 figure
    rows_data = [
        (f"Cut ≤ {NC_CUT_100}",     dw1_c100r, rw1_c100r, "NC count", None, None),
        (f"Cut ≤ {NC_MUON_CUT:,}", dw1_cr,    rw1_cr,    "NC count", None, None),
    ]
    if any(d for d, *_ in rows_data):
        fig, axes = plt.subplots(2, 1, figsize=(9, 10))
        fig.suptitle("Convergence: NC count per muon — cut distributions", fontsize=14)
        for row, (row_title, d_w1, r_w1, xlabel, thr, rec) in enumerate(rows_data):
            if not d_w1:
                continue
            rw1_mean = r_w1.mean(axis=1)
            rw1_std  = r_w1.std(axis=1)
            _draw_w1_panel(axes[row], f"W₁  —  {row_title}",
                           d_w1, rw1_mean, rw1_std,
                           f"W₁ ({xlabel})", thr, rec,
                           w1_at_k1=d_w1[0] if d_w1[0] > 0 else None,
                           show_deterministic=False)
        plt.tight_layout()
        _save(fig, out_dir / "convergence_nc_count_cuts.png")

    if not full_convergence:
        return recommendations

    def get_raw_obs(rd: RunData, obs: str) -> np.ndarray:
        if obs == "zenith":
            return rd.muon_zenith_deg
        if obs == "azimuth":
            return rd.muon_azimuth_deg
        if obs == "energy":
            return rd.muon_ekin_mev[rd.muon_ekin_mev > 0]
        raise ValueError(obs)

    other_observables = [
        ("zenith",  "Zenith [°]",          "Muon zenith"),
        ("azimuth", "Azimuth [°]",         "Muon azimuth"),
        ("energy",  "log(E_kin [MeV]+1)",  "Muon energy"),
    ]

    for obs_key, obs_xlabel, obs_title in other_observables:
        print(f"\n  Convergence: {obs_title} ...", flush=True)
        t_obs = _log_resources(f"{obs_key}: begin")

        sorted_runs: list[np.ndarray] = []
        for rd in run_list:
            raw = _transform(get_raw_obs(rd, obs_key), obs_key)
            sorted_runs.append(np.sort(raw))
        t_obs = _log_resources(f"{obs_key}: sorted {n_runs} runs", t_obs)

        det_w1, rand_w1 = _compute_metrics(sorted_runs)
        del sorted_runs
        gc.collect()
        _log_resources(f"{obs_key}: metrics done", t_obs)

        if not det_w1:
            continue

        rec_k, threshold = _rec_k_from_w1(det_w1)
        recommendations[obs_key] = rec_k
        print(f"    W1 threshold = {threshold:.4f}  →  recommended k = {rec_k}")

        rand_w1_mean = rand_w1.mean(axis=1)
        rand_w1_std  = rand_w1.std(axis=1)

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.suptitle(f"Convergence analysis: {obs_title}", fontsize=14)
        _draw_w1_panel(ax, "W₁ distance", det_w1, rand_w1_mean, rand_w1_std,
                       f"W₁ ({obs_xlabel})", threshold, rec_k,
                       w1_at_k1=det_w1[0] if det_w1[0] > 0 else None)
        _save(fig, out_dir / f"convergence_{obs_key}.png")
        _log_resources(f"{obs_key}: saved — total conv elapsed", t_conv)

    return recommendations


# ---------------------------------------------------------------------------
# Pairwise Wasserstein distance analysis
# ---------------------------------------------------------------------------
def pairwise_w1_analysis(run_list: list[RunData], out_dir: Path) -> None:
    """Empirical null distribution of W1 distances from all C(n,2) run pairs.

    For each observable (zenith, azimuth, energy, NC count per muon) computes
    all unique pairwise W1 distances, derives summary statistics, z-scores,
    and range-normalised distances, and saves a report + two plots per
    observable (histogram/boxplot and heatmap).
    """
    n_runs = len(run_list)
    if n_runs < 2:
        print("  Pairwise W1: need ≥ 2 runs; skipping.")
        return

    run_labels = [rd.run_name for rd in run_list]
    pair_indices = list(_combinations(range(n_runs), 2))
    n_pairs = len(pair_indices)

    def _get_zenith(rd: RunData) -> np.ndarray:
        return rd.muon_zenith_deg

    def _get_azimuth(rd: RunData) -> np.ndarray:
        return rd.muon_azimuth_deg

    def _get_energy(rd: RunData) -> np.ndarray:
        e = rd.muon_ekin_mev
        return np.log(e[e > 0] + 1.0)

    def _get_nc_count(rd: RunData) -> np.ndarray:
        return rd.nc_counts.astype(np.float64)

    observables = [
        ("zenith",   "Muon zenith",       "Zenith [°]",           _get_zenith),
        ("azimuth",  "Muon azimuth",      "Azimuth [°]",          _get_azimuth),
        ("energy",   "Muon energy",       "log(E_kin [MeV] + 1)", _get_energy),
        ("nc_count", "NC count per muon", "NC count (raw)",       _get_nc_count),
    ]

    report_lines: list[str] = [
        "=== Pairwise Wasserstein Distance Analysis ===",
        f"Runs: {n_runs}  |  Pairs: C({n_runs},2) = {n_pairs}",
        "",
        "Purpose: estimate the null distribution of W1 distances expected from",
        "finite-sample fluctuations when all runs share the same underlying",
        "physical process.  The 95th and 99th percentiles define empirical",
        "acceptance thresholds for future simulation runs.",
        "",
    ]

    for obs_key, obs_title, obs_xlabel, getter in observables:
        print(f"\n  Pairwise W1: {obs_title} ...", flush=True)

        arrays = [getter(rd) for rd in run_list]
        if any(a.size == 0 for a in arrays):
            print(f"    WARNING: empty array for at least one run; skipping {obs_title}.")
            continue
        sorted_arrays = [np.sort(a) for a in arrays]

        # Compute all C(n_runs, 2) pairwise W1 distances
        pw_w1 = np.zeros((n_runs, n_runs))
        pair_vals: list[float] = []
        for i, j in pair_indices:
            w1 = _w1_sorted(sorted_arrays[i], sorted_arrays[j])
            pw_w1[i, j] = w1
            pw_w1[j, i] = w1
            pair_vals.append(w1)
        pv = np.array(pair_vals, dtype=np.float64)

        # Summary statistics
        mean_w = float(pv.mean())
        med_w  = float(np.median(pv))
        std_w  = float(pv.std(ddof=0))
        min_w  = float(pv.min())
        max_w  = float(pv.max())
        p95_w  = float(np.percentile(pv, 95))
        p99_w  = float(np.percentile(pv, 99))

        # Normalizations
        all_data   = np.concatenate(arrays)
        data_range = float(all_data.max() - all_data.min())
        del all_data

        z_scores   = (pv - mean_w) / std_w  if std_w > 0.0  else np.zeros_like(pv)
        norm_range = pv / data_range         if data_range > 0.0 else np.zeros_like(pv)

        # Build symmetric matrices
        z_mat   = np.zeros((n_runs, n_runs))
        rng_mat = np.zeros((n_runs, n_runs))
        for idx, (i, j) in enumerate(pair_indices):
            z_mat[i, j]   = z_mat[j, i]   = z_scores[idx]
            rng_mat[i, j] = rng_mat[j, i] = norm_range[idx]

        n_exceed_p99 = int(np.sum(pv > p99_w))
        consistent   = n_exceed_p99 == 0

        # ---- Text report ----
        sep   = "=" * 62
        hdr   = "           " + "".join(f"  {lb:>8s}" for lb in run_labels)

        def _mat_block(mat: np.ndarray, fmt: str) -> list[str]:
            rows = [hdr]
            for i in range(n_runs):
                row = f"  {run_labels[i]:>8s}" + "".join(
                    "     --  " if i == j else f"  {mat[i, j]:{fmt}}"
                    for j in range(n_runs)
                )
                rows.append(row)
            return rows

        report_lines += [
            sep,
            f"Observable: {obs_title}",
            sep,
            "",
            "  Summary statistics (absolute W1):",
            f"    Mean:                  {mean_w:>14.6f}",
            f"    Median:                {med_w:>14.6f}",
            f"    Std (ddof=0):          {std_w:>14.6f}",
            f"    Min:                   {min_w:>14.6f}",
            f"    Max:                   {max_w:>14.6f}",
            f"    95th percentile:       {p95_w:>14.6f}  [empirical threshold]",
            f"    99th percentile:       {p99_w:>14.6f}  [strict threshold]",
            f"    Data range:            {data_range:>14.6f}",
            (f"    W1_mean / data range:  {mean_w/data_range:>14.6f}"
             if data_range > 0 else "    W1_mean / data range:       (range = 0)"),
            "",
            "  Pairwise W1 matrix (absolute):",
        ] + _mat_block(pw_w1, "8.4f") + [
            "",
            "  Pairwise W1 matrix (z-score = (W − mean) / std):",
        ] + _mat_block(z_mat, "8.3f") + [
            "",
            "  Pairwise W1 matrix (W / data range):",
        ] + _mat_block(rng_mat, "8.6f") + [
            "",
            "  Interpretation:",
            f"    Intrinsic fluctuation (1σ):   {std_w:.6f}"
            + (f"  ({std_w / data_range * 100:.4f}% of data range)"
               if data_range > 0 else ""),
            f"    'Normal' W1 range:            [{mean_w - std_w:.4f}, {mean_w + std_w:.4f}]  (mean ± 1σ)",
            f"    Empirical threshold (95th):   {p95_w:.6f}",
            f"    Strict threshold (99th):      {p99_w:.6f}",
            f"    Pairs exceeding 99th pct:     {n_exceed_p99}/{n_pairs}",
            f"    Runs mutually consistent:     {'YES' if consistent else 'NO — inspect heatmap'}",
            "",
            "  Acceptance criterion for future runs:",
            f"    W1(new run, combined reference) ≤ {p95_w:.6f}  (95th pct, recommended)",
            f"    W1(new run, combined reference) ≤ {p99_w:.6f}  (99th pct, strict)",
            "",
        ]

        # ---- Plot 1: Histogram + Boxplot ----
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Pairwise W₁ — {obs_title}  ({n_pairs} pairs, {n_runs} runs)",
            fontsize=13,
        )

        ax = axes[0]
        n_bins = max(7, min(20, n_pairs // 3))
        ax.hist(pv, bins=n_bins, color=COLORS["blue"], edgecolor="black",
                linewidth=0.6, alpha=0.8)
        ax.axvline(mean_w, color=COLORS["orange"], linewidth=1.8,
                   linestyle="--", label=f"Mean = {mean_w:.4f}")
        ax.axvline(mean_w - std_w, color=COLORS["orange"], linewidth=1.0,
                   linestyle=":", alpha=0.7, label=f"±1σ = {std_w:.4f}")
        ax.axvline(mean_w + std_w, color=COLORS["orange"], linewidth=1.0,
                   linestyle=":", alpha=0.7)
        ax.axvline(p95_w, color=COLORS["red"], linewidth=1.5,
                   linestyle="--", label=f"95th pct = {p95_w:.4f}")
        ax.axvline(p99_w, color=COLORS["purple"], linewidth=1.5,
                   linestyle=":", label=f"99th pct = {p99_w:.4f}")
        ax.set_xlabel(f"W₁ ({obs_xlabel})", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Distribution of pairwise W₁", fontsize=11)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=9)

        ax = axes[1]
        ax.boxplot(
            pv, vert=True, patch_artist=True,
            boxprops=dict(facecolor=COLORS["blue"], alpha=0.5),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", markersize=5, alpha=0.6,
                            markerfacecolor=COLORS["blue"]),
        )
        ax.axhline(p95_w, color=COLORS["red"], linewidth=1.5,
                   linestyle="--", label=f"95th pct = {p95_w:.4f}")
        ax.axhline(p99_w, color=COLORS["purple"], linewidth=1.5,
                   linestyle=":", label=f"99th pct = {p99_w:.4f}")
        ax.set_ylabel(f"W₁ ({obs_xlabel})", fontsize=11)
        ax.set_title("Boxplot of pairwise W₁", fontsize=11)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=9)
        ax.set_xticks([])

        plt.tight_layout()
        _save(fig, out_dir / f"pairwise_w1_{obs_key}_dist.png")

        # ---- Plot 2: Heatmap ----
        off_diag = pw_w1[~np.eye(n_runs, dtype=bool)]
        vmax = float(off_diag.max()) if off_diag.size > 0 else 1.0

        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(pw_w1, cmap="viridis", aspect="equal",
                       vmin=0.0, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"W₁ ({obs_xlabel})", fontsize=11)
        ax.set_xticks(range(n_runs))
        ax.set_yticks(range(n_runs))
        ax.set_xticklabels(run_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(run_labels, fontsize=9)
        for i in range(n_runs):
            for j in range(n_runs):
                val = pw_w1[i, j]
                txt = "—" if i == j else f"{val:.3f}"
                brightness = val / vmax if (vmax > 0 and i != j) else 0.0
                txt_color  = "white" if brightness < 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=txt_color)
        ax.set_title(f"Pairwise W₁ matrix — {obs_title}", fontsize=12)
        plt.tight_layout()
        _save(fig, out_dir / f"pairwise_w1_{obs_key}_heatmap.png")

    # Write report
    out_path = out_dir / "pairwise_w1_statistics.txt"
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\n  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Statistical convergence test (3 levels)
# ---------------------------------------------------------------------------

def _extract_nc_fields_run(rd: RunData) -> dict[str, np.ndarray]:
    """Extract NC-level fields from a RunData for convergence analysis."""
    return {
        "nc_time_ns":  rd.nc_time_ns.astype(np.float64),
        "nc_x_m":      rd.nc_x_m.astype(np.float64),
        "nc_y_m":      rd.nc_y_m.astype(np.float64),
        "nc_z_m":      rd.nc_z_m.astype(np.float64),
        "nc_r_m":      rd.nc_r_m.astype(np.float64),
        "nc_phi_rad":  rd.nc_phi_rad.astype(np.float64),
        "nc_ge77":     rd.nc_ge77.astype(np.float64),
        "nc_n_gammas": rd.nc_n_gammas.astype(np.float64),
        "nc_counts":   rd.nc_counts.astype(np.float64),  # per-muon count
    }


def _extract_nc_fields_run_ge77(rd: RunData) -> dict[str, np.ndarray]:
    """Like _extract_nc_fields_run but restricted to Ge77-producing muons."""
    nc_ge77_mu_mask   = np.isin(rd.nc_evtid, rd.ge77_evtids)
    counts_ge77_mask  = np.isin(rd.muon_nc_evtids, rd.ge77_evtids)
    return {
        "nc_time_ns":  rd.nc_time_ns[nc_ge77_mu_mask].astype(np.float64),
        "nc_x_m":      rd.nc_x_m[nc_ge77_mu_mask].astype(np.float64),
        "nc_y_m":      rd.nc_y_m[nc_ge77_mu_mask].astype(np.float64),
        "nc_z_m":      rd.nc_z_m[nc_ge77_mu_mask].astype(np.float64),
        "nc_r_m":      rd.nc_r_m[nc_ge77_mu_mask].astype(np.float64),
        "nc_phi_rad":  rd.nc_phi_rad[nc_ge77_mu_mask].astype(np.float64),
        "nc_ge77":     rd.nc_ge77[nc_ge77_mu_mask].astype(np.float64),
        "nc_n_gammas": rd.nc_n_gammas[nc_ge77_mu_mask].astype(np.float64),
        "nc_counts":   rd.nc_counts[counts_ge77_mask].astype(np.float64),
    }


def load_nc_fields_flat(
    hdf5_files: list[Path],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load and deduplicate NC fields from a flat list of HDF5 files.

    Deduplication is on (evtid, nC_track_id), same logic as load_run.
    Returns (all_fields, ge77_fields) where ge77_fields is restricted to
    NCs belonging to Ge77-producing muons.
    """
    parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("evtid", "track_id", "ge77", "time_ns", "x_m", "y_m", "z_m")
    }
    gamma_evtid_l: list[np.ndarray] = []
    gamma_nc_id_l: list[np.ndarray] = []

    for fp in hdf5_files:
        nc = read_nc_data_file(fp)
        if nc["evtid"].size > 0:
            for k in parts:
                parts[k].append(nc[k])
        ge, gn = read_gamma_count_file(fp)
        if ge.size > 0:
            gamma_evtid_l.append(ge)
            gamma_nc_id_l.append(gn)

    empty = {k: np.array([], dtype=np.float64) for k in NC_FIELDS_FOR_CONV}
    if not parts["evtid"]:
        return empty, empty

    evtid_arr = np.concatenate(parts["evtid"])
    track_arr = np.concatenate(parts["track_id"])
    ge77_arr  = np.concatenate(parts["ge77"])
    time_arr  = np.concatenate(parts["time_ns"])
    x_arr     = np.concatenate(parts["x_m"])
    y_arr     = np.concatenate(parts["y_m"])
    z_arr     = np.concatenate(parts["z_m"])

    pair_keys = np.stack([evtid_arr, track_arr], axis=1)
    _, unique_idx, inverse = np.unique(
        pair_keys, axis=0, return_index=True, return_inverse=True
    )
    unique_ge77 = np.zeros(len(unique_idx), dtype=np.int32)
    np.maximum.at(unique_ge77, inverse, ge77_arr)
    unique_evtids = evtid_arr[unique_idx]
    unique_tracks = track_arr[unique_idx]

    if gamma_evtid_l:
        all_ge = np.concatenate(gamma_evtid_l)
        all_gn = np.concatenate(gamma_nc_id_l)
        gamma_pairs = np.stack([all_ge, all_gn], axis=1)
        upairs, gcounts = np.unique(gamma_pairs, axis=0, return_counts=True)
        gamma_lookup = {
            (int(p[0]), int(p[1])): int(c) for p, c in zip(upairs, gcounts)
        }
        del all_ge, all_gn, gamma_pairs, upairs, gcounts
    else:
        gamma_lookup = {}

    nc_n_gammas = np.array(
        [gamma_lookup.get((int(e), int(t)), 0)
         for e, t in zip(unique_evtids, unique_tracks)],
        dtype=np.float64,
    )

    sorted_mu_evtids, nc_counts_per_mu = np.unique(unique_evtids, return_counts=True)

    nc_time   = time_arr[unique_idx].astype(np.float64)
    nc_z      = z_arr[unique_idx].astype(np.float64)
    nc_r      = np.sqrt(x_arr[unique_idx] ** 2 + y_arr[unique_idx] ** 2)
    nc_phi    = np.arctan2(y_arr[unique_idx], x_arr[unique_idx])
    nc_ge77f  = unique_ge77.astype(np.float64)
    nc_counts = nc_counts_per_mu.astype(np.float64)

    nc_x = x_arr[unique_idx].astype(np.float64)
    nc_y = y_arr[unique_idx].astype(np.float64)

    all_fields: dict[str, np.ndarray] = {
        "nc_time_ns":  nc_time,
        "nc_x_m":      nc_x,
        "nc_y_m":      nc_y,
        "nc_z_m":      nc_z,
        "nc_r_m":      nc_r,
        "nc_phi_rad":  nc_phi,
        "nc_ge77":     nc_ge77f,
        "nc_n_gammas": nc_n_gammas,
        "nc_counts":   nc_counts,
    }

    # Ge77-only: restrict to NCs from muons that produced ≥1 Ge77 NC
    ge77_muon_evtids = np.unique(unique_evtids[unique_ge77 == 1])
    nc_ge77_mu_mask  = np.isin(unique_evtids, ge77_muon_evtids)
    mu_ge77_mask     = np.isin(sorted_mu_evtids, ge77_muon_evtids)

    ge77_fields: dict[str, np.ndarray] = {
        "nc_time_ns":  nc_time[nc_ge77_mu_mask],
        "nc_x_m":      nc_x[nc_ge77_mu_mask],
        "nc_y_m":      nc_y[nc_ge77_mu_mask],
        "nc_z_m":      nc_z[nc_ge77_mu_mask],
        "nc_r_m":      nc_r[nc_ge77_mu_mask],
        "nc_phi_rad":  nc_phi[nc_ge77_mu_mask],
        "nc_ge77":     nc_ge77f[nc_ge77_mu_mask],
        "nc_n_gammas": nc_n_gammas[nc_ge77_mu_mask],
        "nc_counts":   nc_counts[mu_ge77_mask],
    }

    return all_fields, ge77_fields


def level2_pairwise_fluctuations(
    run_list: list[RunData],
    extractor=None,
    label: str = "all muons",
) -> dict[str, tuple[float, float]]:
    """Compute all C(n,2) pairwise W1 distances between runs per NC field.

    extractor: callable(RunData) -> dict[str, np.ndarray]; defaults to
               _extract_nc_fields_run (all muons).
    Returns {field_key: (mean_w1, std_w1)}.
    """
    if extractor is None:
        extractor = _extract_nc_fields_run
    n_runs = len(run_list)
    pair_indices = list(_combinations(range(n_runs), 2))
    n_pairs = len(pair_indices)

    print(f"\n{'='*60}")
    print(f"Level 2 — Leave-One-Out Pairwise W1 Fluctuations ({label})")
    print(f"  Runs: {n_runs}  |  Pairs: C({n_runs},2) = {n_pairs}")

    fluctuations: dict[str, tuple[float, float]] = {}

    for field_key, field_label in NC_FIELDS_FOR_CONV.items():
        arrays = [extractor(rd)[field_key] for rd in run_list]
        if any(a.size == 0 for a in arrays):
            print(f"  SKIP {field_key}: empty array in ≥1 run")
            continue
        sorted_arrs = [np.sort(a) for a in arrays]

        w1_vals: list[float] = []
        for i, j in pair_indices:
            if field_key in _CIRCULAR_FIELDS:
                w1 = _w1_circular_phi(sorted_arrs[i], sorted_arrs[j])
            else:
                w1 = _w1_sorted(sorted_arrs[i], sorted_arrs[j])
            if not np.isnan(w1):
                w1_vals.append(w1)

        if not w1_vals:
            continue
        arr = np.array(w1_vals)
        mean_w1 = float(arr.mean())
        std_w1  = float(arr.std())
        fluctuations[field_key] = (mean_w1, std_w1)
        print(f"  {field_key:20s}: mean W1 = {mean_w1:.6g}  std = {std_w1:.6g}")

    return fluctuations


# Sub-directory names and their expected muon counts for Level-3 learning curve.
# Each tuple is (directory_name, expected_total_muons).  The vertex cross-check
# verifies the actual count via //hit/vertices/evtid/pages.
_SUBSAMPLE_DIRS: list[tuple[str, int]] = [
    ("1e4", 10_000),
    ("1e5", 100_000),
    ("1e6", 1_000_000),
    ("1e7", 10_000_000),
]


def _w1_circular_phi(a: np.ndarray, b: np.ndarray, K: int = 200) -> float:
    """Wasserstein-1 for angles in [-π, π] under the circular arc metric
    d(θ₁, θ₂) = min(|θ₁−θ₂|, 2π−|θ₁−θ₂|).

    Algorithm
    ---------
    The standard linear W1 on [-π, π] treats the wrap-around at ±π as a
    large cost, which is incorrect for angular data.  The true circular W1
    is the minimum linear W1 achievable by cyclically shifting one
    distribution along the circle.

    1. Map both arrays to [0, 2π) via ``% (2π)``.
    2. Try K uniformly-spaced cyclic shifts c_k = k·(2π/K) for k=0,...,K-1.
    3. For each shift c, the sorted shifted array (b + c) % 2π is obtained
       in O(1) by splitting at the wrap-around index via np.searchsorted
       and concatenating the two halves (no full re-sort required).
    4. Evaluate the linear W1 against a via _w1_sorted  [O(n+m) per
       call since both inputs are sorted].
    5. Return the minimum W1 over all K candidates.

    Approximation
    -------------
    The approximation error in the optimal shift is ≤ 2π/K radians.
    For a near-uniform distribution (azimuthal symmetry in the water tank)
    the resulting W1 error is bounded by (2π/K) × max_density ≈ 2π/K ÷ 2π
    = 1/K.  With K=200 this gives < 0.5 % relative error in W1, well below
    the statistical noise at any subsample size used here.

    Complexity: O(K · (n + m)).  With K=200 and n=m=1e7, roughly 4×10⁹
    elementary operations (~2–4 s on a modern node).

    References
    ----------
    Rabin J. & Peyré G. (2011), "Wasserstein regularization of imaging
    problem", IEEE ICIP — circular optimal transport on the 1-torus.
    Delon J. et al. (2010), "Fast Earth Mover's Distance", SIAM IS — exact
    O(n log n) algorithm for equal-size empirical measures on the circle.
    """
    TWO_PI = 2.0 * np.pi
    a_s = np.sort(a % TWO_PI)
    b_s = np.sort(b % TWO_PI)
    best = np.inf
    for k in range(K):
        c = k * TWO_PI / K
        # Find index where b_s[idx:] + c wraps past 2π
        idx = int(np.searchsorted(b_s, TWO_PI - c, side="left"))
        if idx == len(b_s):
            # All values shift without wrapping
            b_c = b_s + c
        elif idx == 0:
            # All values wrap: subtract 2π after adding c
            b_c = b_s + c - TWO_PI
        else:
            b_c = np.concatenate([b_s[idx:] + c - TWO_PI, b_s[:idx] + c])
        w1 = _w1_sorted(a_s, b_c)
        if w1 < best:
            best = w1
    return best


def _count_muons_in_dir(hdf5_files: list[Path]) -> int:
    """Count unique muon evtids across all HDF5 files via //hit/vertices/evtid/pages.

    Each muon appears exactly once in the vertices group, so the number of
    unique evtids equals the number of simulated muons.  This is the ground-
    truth count used to validate that a sub-directory contains the expected N.
    """
    all_evtids: list[np.ndarray] = []
    for fp in hdf5_files:
        try:
            with h5py.File(fp, "r") as f:
                if VERTICES_GROUP not in f:
                    continue
                vgrp = f[VERTICES_GROUP]
                if int(vgrp["entries"][()]) == 0:
                    continue
                all_evtids.append(_pages(vgrp, "evtid").astype(np.int64))
        except Exception as exc:
            print(f"  WARNING: could not read vertices from {fp.name}: {exc}")
    if not all_evtids:
        return 0
    return int(np.unique(np.concatenate(all_evtids)).size)


def _compute_swd_all_params(
    sub_fields: dict[str, np.ndarray],
    ref_fields: dict[str, np.ndarray],
    n_projections: int = 500,
    seed: int = 42,
) -> float:
    """Sliced Wasserstein Distance (SWD) across all NC parameters at once.

    Feature matrix (D=10 columns per NC event)
    -------------------------------------------
    nc_time_ns, nc_x_m, nc_y_m, nc_z_m, nc_r_m, nc_ge77, nc_n_gammas,
    nc_counts   →  8 columns used directly.
    nc_phi_rad  →  (cos φ, sin φ)  — 2 columns preserving circular topology
                   so that angles near ±π are treated as neighbours.

    Normalization
    -------------
    Each column is z-scored using the reference mean and std to put all
    parameters on a comparable scale.  Columns with zero std are dropped.

    SWD computation
    ---------------
    The SWD is the average W1 along K=n_projections random unit directions
    in R^D, computed via ot.sliced_wasserstein_distance (POT library).
    SWD → W₁ as K → ∞  (Rabin et al. 2012, "Wasserstein Barycenter and
    its Application to Texture Mixing").  At N=1e8 (sub == ref) SWD = 0.

    Returns np.nan if any required field is empty.
    """
    import ot  # POT — already in project dependencies

    SCALAR_KEYS = [
        "nc_time_ns", "nc_x_m", "nc_y_m", "nc_z_m", "nc_r_m",
        "nc_ge77", "nc_n_gammas", "nc_counts",
    ]

    def _build_matrix(fields: dict[str, np.ndarray]) -> np.ndarray | None:
        cols: list[np.ndarray] = []
        for k in SCALAR_KEYS:
            arr = fields.get(k, np.array([]))
            if arr.size == 0:
                return None
            cols.append(arr.astype(np.float32))
        phi = fields.get("nc_phi_rad", np.array([]))
        if phi.size == 0:
            return None
        cols.append(np.cos(phi).astype(np.float32))
        cols.append(np.sin(phi).astype(np.float32))
        return np.stack(cols, axis=1)  # shape (N, D)

    X_ref = _build_matrix(ref_fields)
    X_sub = _build_matrix(sub_fields)
    if X_ref is None or X_sub is None:
        return float("nan")

    # Z-score normalise using reference statistics
    ref_mean = X_ref.mean(axis=0)
    ref_std  = X_ref.std(axis=0)
    keep = ref_std > 0
    if not keep.any():
        return float("nan")
    X_ref = ((X_ref - ref_mean) / np.where(keep, ref_std, 1.0))[:, keep]
    X_sub = ((X_sub - ref_mean) / np.where(keep, ref_std, 1.0))[:, keep]

    rng = np.random.default_rng(seed)
    return float(ot.sliced_wasserstein_distance(
        X_sub, X_ref, n_projections=n_projections, seed=int(rng.integers(2**31))
    ))


def _plot_relative_improvement(
    n_arr: np.ndarray,
    w1_data: dict[str, list[float]],
    out_dir: Path,
    title_suffix: str = "All muons",
    fname_suffix: str = "",
) -> None:
    """Plot B: relative W1 improvement per decade.

    ΔW1_rel(N) = [W1(N/10) − W1(N)] / W1(N/10) × 100 %

    Shows how much each additional decade of statistics reduces the W1
    distance to the 1e8 reference.  The N=1e7 point is annotated
    explicitly.  N=1e8 (W1=0) is excluded — division by W1(N/10)=W1(1e7)
    would give 100 % and the point is trivial.
    """
    n_fields = len(NC_FIELDS_FOR_CONV)
    ncols = 3
    nrows = (n_fields + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes_flat = np.array(axes).flatten()

    # Exclude the N=1e8 anchor (W1=0) from relative improvement
    mask_excl = n_arr < 1e8 - 1
    n_plot = n_arr[mask_excl]

    for idx, (field_key, field_label) in enumerate(NC_FIELDS_FOR_CONV.items()):
        ax = axes_flat[idx]
        w1_arr = np.array(w1_data[field_key])[mask_excl]

        # ΔW1_rel(N) = [W1(N/10) - W1(N)] / W1(N/10) * 100
        # Computed at each consecutive pair; result assigned to the larger N.
        delta_vals: list[float] = []
        delta_n:    list[float] = []
        for k in range(1, len(n_plot)):
            w_prev = w1_arr[k - 1]
            w_curr = w1_arr[k]
            if w_prev > 0 and not np.isnan(w_prev) and not np.isnan(w_curr):
                delta_vals.append((w_prev - w_curr) / w_prev * 100.0)
                delta_n.append(n_plot[k])

        if not delta_n:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(field_label, fontsize=12)
            continue

        ax.bar(range(len(delta_n)), delta_vals, color=COLORS["blue"], alpha=0.8)
        ax.set_xticks(range(len(delta_n)))
        ax.set_xticklabels([f"1e{int(np.log10(n))}" for n in delta_n], fontsize=9)

        # Mark N=1e7 bar
        for ki, n_val in enumerate(delta_n):
            if abs(n_val - 1e7) < 1:
                ax.bar(ki, delta_vals[ki], color=COLORS["red"], alpha=0.9,
                       label=f"N=1e7: {delta_vals[ki]:.1f}%")
                break

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("N (larger end of decade)", fontsize=10)
        ax.set_ylabel("ΔW1_rel [%]", fontsize=10)
        ax.set_title(field_label, fontsize=12)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=9)

    for extra in axes_flat[n_fields:]:
        extra.set_visible(False)

    fig.suptitle(
        f"Relative W1 Improvement per Decade — {title_suffix}\n"
        r"$\Delta W1_\mathrm{rel}(N) = [W1(N/10)-W1(N)]\,/\,W1(N/10)\times100\%$",
        fontsize=13,
    )
    plt.tight_layout()
    fname = f"convergence_relative_improvement{fname_suffix}.png"
    _save(fig, out_dir / fname)


def _convergence_summary(
    n_arr: np.ndarray,
    w1_all: dict[str, list[float]],
    w1_ge77: dict[str, list[float]],
    out_dir: Path,
) -> None:
    """Print and save convergence criteria table for both populations.

    Criterion A — absolute threshold:
        W1(N) < f × W1(N=1e4)   for f ∈ {0.01, 0.05, 0.10}

    Criterion B — relative improvement threshold:
        ΔW1_rel(N→10N) < ε_rel  for ε_rel ∈ {1%, 2%, 5%}

    Outputs:
    - Console: one line per parameter × criterion, ✓/✗ at N=1e7.
    - LaTeX: booktabs-style table saved to out_dir/convergence_summary_table.tex.
    """
    F_VALS   = [0.01, 0.05, 0.10]
    EPS_VALS = [1.0,  2.0,  5.0]

    def _min_n_criterion_a(w1_vals: list[float], f: float) -> float | None:
        """Return smallest N where W1(N) < f * W1(1e4), or None."""
        arr = np.array(w1_vals)
        # First value corresponds to 1e4 (first subsample dir)
        valid = ~np.isnan(arr) & (arr > 0)
        if not valid.any():
            return None
        w1_ref = arr[valid][0]
        threshold = f * w1_ref
        for n_val, w in zip(n_arr, arr):
            if n_val >= 1e8 - 1:
                break  # skip the 0-anchor
            if not np.isnan(w) and w < threshold:
                return float(n_val)
        return None

    def _min_n_criterion_b(w1_vals: list[float], eps: float) -> float | None:
        """Return smallest N (larger end) where ΔW1_rel < eps [%], or None."""
        arr = np.array(w1_vals)
        mask_excl = n_arr < 1e8 - 1
        arr_ex = arr[mask_excl]
        n_ex   = n_arr[mask_excl]
        for k in range(1, len(n_ex)):
            w_prev, w_curr = arr_ex[k - 1], arr_ex[k]
            if w_prev > 0 and not np.isnan(w_prev) and not np.isnan(w_curr):
                rel = (w_prev - w_curr) / w_prev * 100.0
                if rel < eps:
                    return float(n_ex[k])
        return None

    def _passes_at_1e7(w1_vals: list[float], threshold_n: float | None) -> bool:
        if threshold_n is None:
            return False
        return threshold_n <= 1e7 + 1

    header = (
        f"\n{'='*80}\n"
        "Convergence Summary\n"
        f"{'='*80}"
    )
    print(header)

    tex_rows: list[str] = []
    tex_rows.append(r"\begin{table}[htbp]")
    tex_rows.append(r"\centering")
    tex_rows.append(r"\caption{Convergence criteria for NC parameters (all muons / Ge77 muons).}")
    tex_rows.append(r"\begin{tabular}{llccc}")
    tex_rows.append(r"\toprule")
    tex_rows.append(
        r"Parameter & Criterion & Threshold & Min.\ sufficient $N$ (all) "
        r"& $N=10^7$ sufficient? \\"
    )
    tex_rows.append(r"\midrule")

    for pop_label, w1_data in [("all muons", w1_all), ("Ge77 muons", w1_ge77)]:
        print(f"\n  Population: {pop_label}")
        print(f"  {'Parameter':<22} {'Crit':<4} {'Thr':<6} {'Min N':>10}  {'✓/✗ at 1e7'}")
        print(f"  {'-'*60}")

        for field_key, field_label in NC_FIELDS_FOR_CONV.items():
            w1_vals = w1_data[field_key]

            for f in F_VALS:
                mn = _min_n_criterion_a(w1_vals, f)
                ok = _passes_at_1e7(w1_vals, mn)
                mn_str = f"{mn:.0e}" if mn is not None else "never"
                sym = "✓" if ok else "✗"
                tex_sym = r"\checkmark" if ok else r"$\times$"
                print(f"  {field_label:<22} A    {f:<6.2f} {mn_str:>10}  {sym}")
                tex_rows.append(
                    f"{field_label} & A & $f={f:.2f}$ & {mn_str} & {tex_sym} \\\\"
                )

            for eps in EPS_VALS:
                mn = _min_n_criterion_b(w1_vals, eps)
                ok = _passes_at_1e7(w1_vals, mn)
                mn_str = f"{mn:.0e}" if mn is not None else "never"
                sym = "✓" if ok else "✗"
                tex_sym = r"\checkmark" if ok else r"$\times$"
                print(f"  {field_label:<22} B    {eps:<6.1f}% {mn_str:>10}  {sym}")
                tex_rows.append(
                    f"{field_label} & B & $\\varepsilon={eps:.0f}\\%$ & {mn_str} & {tex_sym} \\\\"
                )

        tex_rows.append(r"\midrule")

    tex_rows.append(r"\bottomrule")
    tex_rows.append(r"\end{tabular}")
    tex_rows.append(r"\end{table}")

    # ---------------------------------------------------------------------- #
    # Overall verdict: is N=1e7 sufficient across ALL parameters × populations?
    # ---------------------------------------------------------------------- #
    verdict_results: dict[tuple[str, str], tuple[bool, int, int]] = {}

    for f in F_VALS:
        key = ("A", f"{f:.2f}")
        passes: list[bool] = []
        for w1_data in [w1_all, w1_ge77]:
            for field_key in NC_FIELDS_FOR_CONV:
                mn = _min_n_criterion_a(w1_data[field_key], f)
                passes.append(_passes_at_1e7(w1_data[field_key], mn))
        verdict_results[key] = (all(passes), sum(passes), len(passes))

    for eps in EPS_VALS:
        key = ("B", f"{eps:.0f}%")
        passes = []
        for w1_data in [w1_all, w1_ge77]:
            for field_key in NC_FIELDS_FOR_CONV:
                mn = _min_n_criterion_b(w1_data[field_key], eps)
                passes.append(_passes_at_1e7(w1_data[field_key], mn))
        verdict_results[key] = (all(passes), sum(passes), len(passes))

    print(f"\n{'='*80}")
    print("OVERALL VERDICT: Is N = 1×10⁷ sufficient?")
    print(f"{'='*80}")
    print(f"  (checking {len(NC_FIELDS_FOR_CONV)} parameters × 2 populations = "
          f"{len(NC_FIELDS_FOR_CONV) * 2} parameter×population combinations)")
    print()
    for (crit, thr), (all_pass, n_pass, n_total) in verdict_results.items():
        sym = "✓  SUFFICIENT" if all_pass else "✗  NEED 1e8 "
        print(f"  Criterion {crit}  threshold={thr:<5s}  "
              f"{n_pass:2d}/{n_total} combinations pass  →  {sym}")

    any_all_pass = any(v[0] for v in verdict_results.values())
    print()
    if any_all_pass:
        # Find the strictest criterion that still passes
        passing_keys = [(c, t) for (c, t), (ok, _, _) in verdict_results.items() if ok]
        print("  CONCLUSION: N = 1×10⁷ is SUFFICIENT under at least one tested criterion.")
        print(f"  Passing criterion(s): {', '.join(f'Crit {c} thr={t}' for c, t in passing_keys)}")
    else:
        print("  CONCLUSION: N = 1×10⁷ is INSUFFICIENT — N = 1×10⁸ is REQUIRED")
        print("              under all tested criteria for at least one parameter.")
    print(f"{'='*80}\n")

    # Add verdict to LaTeX output
    tex_rows.append("")
    tex_rows.append(r"\begin{table}[htbp]")
    tex_rows.append(r"\centering")
    tex_rows.append(r"\caption{Overall verdict: is $N=10^7$ sufficient?}")
    tex_rows.append(r"\begin{tabular}{llcc}")
    tex_rows.append(r"\toprule")
    tex_rows.append(r"Criterion & Threshold & Combinations passing & Verdict \\")
    tex_rows.append(r"\midrule")
    for (crit, thr), (all_pass, n_pass, n_total) in verdict_results.items():
        verdict_str = r"\checkmark\ Sufficient" if all_pass else r"$\times$\ Need $10^8$"
        tex_rows.append(
            f"{crit} & {thr} & ${n_pass}/{n_total}$ & {verdict_str} \\\\"
        )
    tex_rows.append(r"\midrule")
    overall_str = (
        r"\textbf{Sufficient}" if any_all_pass
        else r"\textbf{Need $10^8$}"
    )
    tex_rows.append(rf"\multicolumn{{4}}{{l}}{{Overall: {overall_str}}} \\")
    tex_rows.append(r"\bottomrule")
    tex_rows.append(r"\end{tabular}")
    tex_rows.append(r"\end{table}")

    tex_path = out_dir / "convergence_summary_table.tex"
    tex_path.write_text("\n".join(tex_rows) + "\n", encoding="utf-8")
    print(f"\n  Saved LaTeX table: {tex_path.name}")


def _fit_power_law(
    n_arr: np.ndarray, w1_arr: np.ndarray
) -> tuple[float, float, float] | None:
    """Fit W1(N) = a · N^{-α} via log-log OLS on valid (N, W1) pairs.

    Returns (a, alpha, alpha_se) or None if fewer than 3 valid points.
    alpha_se is the OLS standard error of the exponent from the polyfit
    covariance matrix.  The theoretical i.i.d. value is alpha = 0.5
    (Fournier & Guillin 2015, Ann. Probab. 43(6)).
    """
    valid = ~np.isnan(w1_arr) & (w1_arr > 0) & (n_arr > 0)
    if valid.sum() < 3:
        return None
    coeffs, cov = np.polyfit(
        np.log10(n_arr[valid]), np.log10(w1_arr[valid]), 1, cov=True
    )
    alpha_se = float(np.sqrt(cov[0, 0]))
    return float(10 ** coeffs[1]), float(-coeffs[0]), alpha_se


def _plot_learning_curve(
    n_arr: np.ndarray,
    w1_data: dict[str, list[float]],
    fluctuations: dict[str, tuple[float, float]],
    out_dir: Path,
    title_suffix: str = "All muons",
    fname_suffix: str = "",
) -> None:
    n_fields = len(NC_FIELDS_FOR_CONV)
    ncols = 3
    nrows = (n_fields + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes_flat = np.array(axes).flatten()

    for idx, (field_key, field_label) in enumerate(NC_FIELDS_FOR_CONV.items()):
        ax = axes_flat[idx]
        w1_arr = np.array(w1_data[field_key])
        # valid: subsample points with positive W1 (excludes the N=1e8 zero anchor)
        valid = ~np.isnan(w1_arr) & (w1_arr > 0)

        ax.scatter(n_arr[valid], w1_arr[valid],
                   color=COLORS["blue"], zorder=5, s=60, label="W1 to 1e8 ref")

        anchor_mask = (n_arr >= 1e8 - 1) & (w1_arr == 0.0)
        if anchor_mask.any():
            ax.axvline(1e8, color="gray", linestyle=":", linewidth=0.8,
                       label="N=1e8 (W1=0, ref=self)")

        fit = _fit_power_law(n_arr, w1_arr)
        if fit is not None:
            a_f, b_f, b_se = fit
            n_min = float(n_arr[valid].min()) if valid.any() else 1e4
            n_plot = np.logspace(np.log10(n_min), 8.5, 300)
            ax.plot(n_plot, a_f * n_plot ** (-b_f), "--",
                    color=COLORS["red"], linewidth=1.5,
                    label=f"α={b_f:.3f}±{b_se:.3f} (fit, F&G 2015)")
            w1_1e8 = a_f * 1e8 ** (-b_f)
            ax.scatter([1e8], [w1_1e8], marker="*", s=120,
                       color=COLORS["red"], zorder=6,
                       label=f"D(1e8)≈{w1_1e8:.4g}")
            # Theoretical α=0.5 reference anchored at first valid data point
            n0 = float(n_arr[valid][0])
            w0 = float(w1_arr[valid][0])
            ax.plot(n_plot, w0 * (n_plot / n0) ** (-0.5), ":",
                    color="gray", linewidth=1.2, alpha=0.7,
                    label="α=0.5 (i.i.d. CLT theory)")

        if field_key in fluctuations:
            m_fl, s_fl = fluctuations[field_key]
            ax.axhline(m_fl, color=COLORS["green"], linestyle="--",
                       linewidth=1.0, label=f"L2 mean={m_fl:.4g}")
            lo = max(m_fl - s_fl, 1e-30)
            ax.axhspan(lo, m_fl + s_fl, alpha=0.15, color=COLORS["green"])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N muon samples", fontsize=11)
        ax.set_ylabel("W1 distance", fontsize=11)
        ax.set_title(field_label, fontsize=12)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=9)

    for extra in axes_flat[n_fields:]:
        extra.set_visible(False)

    fig.suptitle(f"Convergence Learning Curve: W1 vs Sample Size — {title_suffix}",
                 fontsize=14)
    plt.tight_layout()
    fname = f"convergence_learning_curve{fname_suffix}.png"
    _save(fig, out_dir / fname)


def _plot_learning_curve_normalized(
    n_arr: np.ndarray,
    w1_data: dict[str, list[float]],
    fluctuations: dict[str, tuple[float, float]],
    out_dir: Path,
    title_suffix: str = "All muons",
    fname_suffix: str = "",
) -> None:
    n_fields = len(NC_FIELDS_FOR_CONV)
    ncols = 3
    nrows = (n_fields + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes_flat = np.array(axes).flatten()

    for idx, (field_key, field_label) in enumerate(NC_FIELDS_FOR_CONV.items()):
        ax = axes_flat[idx]

        if field_key not in fluctuations or fluctuations[field_key][0] == 0:
            ax.text(0.5, 0.5, "No L2 baseline", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(field_label, fontsize=12)
            continue

        m_fl, s_fl = fluctuations[field_key]
        w1_arr  = np.array(w1_data[field_key])
        w1_norm = w1_arr / m_fl
        valid = ~np.isnan(w1_norm) & (w1_norm > 0)

        ax.scatter(n_arr[valid], w1_norm[valid],
                   color=COLORS["blue"], zorder=5, s=60, label="W1/mean(L2)")

        fit = _fit_power_law(n_arr, w1_norm)
        if fit is not None:
            a_f, b_f, _ = fit
            n_min = float(n_arr[valid].min()) if valid.any() else 1e4
            n_plot = np.logspace(np.log10(n_min), 8.5, 300)
            ax.plot(n_plot, a_f * n_plot ** (-b_f), "--",
                    color=COLORS["red"], linewidth=1.5,
                    label=f"{a_f:.3g}·n^(−{b_f:.3f})")
            val_1e8 = a_f * 1e8 ** (-b_f)
            ax.scatter([1e8], [val_1e8], marker="*", s=120,
                       color=COLORS["red"], zorder=6,
                       label=f"at 1e8: {val_1e8:.4g}")
            ax.axvline(1e8, color="gray", linestyle=":", linewidth=0.8)

        ax.axhline(1.0, color=COLORS["green"], linestyle="--",
                   linewidth=1.0, label="= L2 mean fluctuation")
        rel_std = s_fl / m_fl if m_fl > 0 else 0.0
        ax.axhspan(max(1.0 - rel_std, 1e-30), 1.0 + rel_std,
                   alpha=0.15, color=COLORS["green"])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N muon samples", fontsize=11)
        ax.set_ylabel("W1 / mean(W1_L2)", fontsize=11)
        ax.set_title(field_label, fontsize=12)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=9)

    for extra in axes_flat[n_fields:]:
        extra.set_visible(False)

    fig.suptitle(
        f"Normalized Convergence: W1 / L2 Mean Fluctuation — {title_suffix}",
        fontsize=14,
    )
    plt.tight_layout()
    fname = f"convergence_learning_curve_normalized{fname_suffix}.png"
    _save(fig, out_dir / fname)


def _plot_learning_curve_comparison(
    n_arr: np.ndarray,
    w1_all: dict[str, list[float]],
    w1_ge77: dict[str, list[float]],
    out_dir: Path,
) -> None:
    """One subplot per field: both all-muon and Ge77-only W1 curves on the same axes."""
    n_fields = len(NC_FIELDS_FOR_CONV)
    ncols = 3
    nrows = (n_fields + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes_flat = np.array(axes).flatten()

    for idx, (field_key, field_label) in enumerate(NC_FIELDS_FOR_CONV.items()):
        ax = axes_flat[idx]
        w1_a = np.array(w1_all[field_key])
        w1_g = np.array(w1_ge77[field_key])

        for w1_arr, color, marker, subset_label in [
            (w1_a, COLORS["blue"],   "o", "All muons"),
            (w1_g, COLORS["orange"], "s", "Ge77 muons"),
        ]:
            valid = ~np.isnan(w1_arr) & (w1_arr > 0)
            if not valid.any():
                continue
            ax.scatter(n_arr[valid], w1_arr[valid],
                       color=color, marker=marker, zorder=5, s=60,
                       label=subset_label)
            fit = _fit_power_law(n_arr, w1_arr)
            if fit is not None:
                a_f, b_f, _ = fit
                n_min = float(n_arr[valid].min())
                n_plot = np.logspace(np.log10(n_min), 8.5, 300)
                ax.plot(n_plot, a_f * n_plot ** (-b_f), "--",
                        color=color, linewidth=1.5, alpha=0.8,
                        label=f"n^(−{b_f:.2f})")

        ax.axvline(1e8, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N muon samples", fontsize=11)
        ax.set_ylabel("W1 distance", fontsize=11)
        ax.set_title(field_label, fontsize=12)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=9)

    for extra in axes_flat[n_fields:]:
        extra.set_visible(False)

    fig.suptitle("W1 Convergence: All muons vs Ge77 muons", fontsize=14)
    plt.tight_layout()
    _save(fig, out_dir / "convergence_learning_curve_comparison.png")


def _plot_swd_learning_curve(
    n_arr: np.ndarray,
    swd_all: list[float],
    swd_ge77: list[float],
    out_dir: Path,
) -> None:
    """Single plot: Sliced Wasserstein Distance (all params combined) vs N."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for swd_vals, color, marker, label in [
        (swd_all,  COLORS["blue"],   "o", "All muons"),
        (swd_ge77, COLORS["orange"], "s", "Ge77 muons"),
    ]:
        arr = np.array(swd_vals)
        valid = ~np.isnan(arr) & (arr > 0)
        if valid.any():
            ax.scatter(n_arr[valid], arr[valid],
                       color=color, marker=marker, s=60, zorder=5, label=label)
            fit = _fit_power_law(n_arr, arr)
            if fit is not None:
                a_f, b_f, _ = fit
                n_min = float(n_arr[valid].min())
                n_plot = np.logspace(np.log10(n_min), 8.5, 300)
                ax.plot(n_plot, a_f * n_plot ** (-b_f), "--",
                        color=color, linewidth=1.5, alpha=0.8,
                        label=f"{label}: n^(−{b_f:.2f})")
        # Plot the N=1e8 zero anchor explicitly on the x-axis
        ax.scatter([1e8], [1e-10], marker="*", s=150, color=color,
                   zorder=6, alpha=0.6)

    ax.axvline(1e8, color="gray", linestyle=":", linewidth=0.8,
               label="N=1e8 ref (SWD=0)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N muon samples", fontsize=12)
    ax.set_ylabel("Sliced Wasserstein Distance", fontsize=12)
    ax.set_title("SWD (all NC parameters combined) vs Sample Size", fontsize=13)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    _save(fig, out_dir / "convergence_swd_combined.png")


def level3_learning_curve(
    data_path: Path,
    run_list: list[RunData],
    fluctuations: dict[str, tuple[float, float]],
    fluctuations_ge77: dict[str, tuple[float, float]],
    out_dir: Path,
) -> None:
    """Load sub-directory data; compute W1 vs 1e8 reference; fit power law; save plots.

    For each entry in _SUBSAMPLE_DIRS the directory data_path/<dir>/ is loaded
    in full (no subsampling).  Before loading, _count_muons_in_dir verifies that
    the number of unique muon evtids in //hit/vertices/evtid/pages exactly matches
    the expected count from the directory name.  A mismatch raises RuntimeError.

    Circular W1 is used for nc_phi_rad (see _w1_circular_phi).
    An anchor at N=1e8 with W1=0 is appended; the power-law fit excludes it via
    the (w1 > 0) mask.
    A Sliced Wasserstein Distance (SWD) across all parameters is computed as a
    combined multivariate convergence metric.
    """
    print(f"\n{'='*60}")
    print("Level 3 — Learning Curve & Power Law Extrapolation")

    # Build reference arrays from all 10 runs (= full 1e8 dataset).
    ref_all:  dict[str, np.ndarray] = {}
    ref_ge77: dict[str, np.ndarray] = {}
    for field_key in NC_FIELDS_FOR_CONV:
        all_parts  = [_extract_nc_fields_run(rd)[field_key]      for rd in run_list]
        ge77_parts = [_extract_nc_fields_run_ge77(rd)[field_key] for rd in run_list]
        ref_all[field_key]  = np.sort(np.concatenate([a for a in all_parts  if a.size > 0]))
        ref_ge77[field_key] = np.sort(np.concatenate([a for a in ge77_parts if a.size > 0]))

    # Raw (unsorted) reference dicts for SWD feature-matrix construction
    ref_all_raw:  dict[str, np.ndarray] = {
        k: np.concatenate([_extract_nc_fields_run(rd)[k] for rd in run_list])
        for k in NC_FIELDS_FOR_CONV
    }
    ref_ge77_raw: dict[str, np.ndarray] = {
        k: np.concatenate([_extract_nc_fields_run_ge77(rd)[k] for rd in run_list])
        for k in NC_FIELDS_FOR_CONV
    }

    total_nc      = sum(rd.nc_evtid.size for rd in run_list)
    total_ge77_nc = sum(int((rd.nc_ge77 == 1).sum()) for rd in run_list)
    print(f"  Reference (all):  {total_nc:,} NCs | "
          f"Ge77-muon subset: {total_ge77_nc:,} Ge77-flagged NCs")

    w1_all:  dict[str, list[float]] = {k: [] for k in NC_FIELDS_FOR_CONV}
    w1_ge77: dict[str, list[float]] = {k: [] for k in NC_FIELDS_FOR_CONV}
    swd_all:  list[float] = []
    swd_ge77: list[float] = []
    n_vals: list[float] = []

    for size_dir, n_expected in _SUBSAMPLE_DIRS:
        subdir = data_path / size_dir
        if not subdir.exists():
            raise RuntimeError(
                f"Level 3: required sub-directory not found: {subdir}\n"
                f"Expected sub-directories under {data_path}: "
                f"{[d for d, _ in _SUBSAMPLE_DIRS]}"
            )
        hdf5_files = sorted(subdir.glob("output_t*.hdf5"))
        if not hdf5_files:
            hdf5_files = sorted(subdir.glob("*.hdf5"))
        if not hdf5_files:
            raise RuntimeError(
                f"Level 3: no HDF5 files found in required sub-directory: {subdir}"
            )

        # --- Vertex cross-check: actual muon count must match directory name ---
        print(f"  Verifying {size_dir}/  ({len(hdf5_files)} files) ...", flush=True)
        n_actual = _count_muons_in_dir(hdf5_files)
        if n_actual != n_expected:
            raise RuntimeError(
                f"Level 3 vertex cross-check FAILED for {subdir}:\n"
                f"  Expected {n_expected:,} unique muon evtids (from directory name '{size_dir}'),\n"
                f"  but //hit/vertices/evtid/pages contains {n_actual:,} unique evtids.\n"
                f"  Ensure each sub-directory is a simulation of exactly the stated "
                f"number of muons."
            )
        print(f"    OK: {n_actual:,} unique muon evtids == expected {n_expected:,}.")

        print(f"  Loading {size_dir}/  (N={n_expected:.0e}) ...", flush=True)
        sub_all_fields, sub_ge77_fields = load_nc_fields_flat(hdf5_files)
        n_vals.append(float(n_expected))

        for field_key in NC_FIELDS_FOR_CONV:
            for sub_fields, w1_dict, ref_sorted in [
                (sub_all_fields,  w1_all,  ref_all),
                (sub_ge77_fields, w1_ge77, ref_ge77),
            ]:
                sub_arr = sub_fields[field_key]
                ref_s   = ref_sorted[field_key]
                if sub_arr.size == 0 or ref_s.size == 0:
                    w1_dict[field_key].append(float("nan"))
                    continue
                if field_key in _CIRCULAR_FIELDS:
                    w1 = _w1_circular_phi(sub_arr, ref_s)
                else:
                    sub_s = np.sort(sub_arr)
                    w1 = _w1_sorted(sub_s, ref_s)
                w1_dict[field_key].append(float(w1))
            print(f"    {field_key:20s}: W1(all)={w1_all[field_key][-1]:.6g}"
                  f"  W1(ge77)={w1_ge77[field_key][-1]:.6g}")

        # Sliced Wasserstein Distance over all parameters combined
        swd_a = _compute_swd_all_params(sub_all_fields,  ref_all_raw)
        swd_g = _compute_swd_all_params(sub_ge77_fields, ref_ge77_raw)
        swd_all.append(swd_a)
        swd_ge77.append(swd_g)
        print(f"    {'SWD (all params)':20s}: SWD(all)={swd_a:.6g}  SWD(ge77)={swd_g:.6g}")

    if len(n_vals) == 0:
        raise RuntimeError(
            "Level 3: none of the expected sub-directories were found or passed "
            f"the vertex cross-check under {data_path}.\n"
            f"Expected: {[d for d, _ in _SUBSAMPLE_DIRS]}"
        )
    if len(n_vals) < 2:
        print("  Only 1 sample size available; power-law fit requires ≥ 2. "
              "Add more sub-directories.")
        return

    # Append N=1e8 anchor: comparing the full reference to itself gives W1=0.
    # The power-law fit excludes this point via the (w1 > 0) mask.
    n_vals.append(1e8)
    for fk in NC_FIELDS_FOR_CONV:
        w1_all[fk].append(0.0)
        w1_ge77[fk].append(0.0)
    swd_all.append(0.0)
    swd_ge77.append(0.0)
    print("  Added N=1e8 anchor (W1=0, SWD=0) — reference vs itself.")

    n_arr = np.array(n_vals, dtype=np.float64)

    _plot_learning_curve(n_arr, w1_all,  fluctuations,      out_dir,
                         title_suffix="All muons")
    _plot_learning_curve(n_arr, w1_ge77, fluctuations_ge77, out_dir,
                         title_suffix="Ge77 muons", fname_suffix="_ge77")
    _plot_learning_curve_normalized(n_arr, w1_all,  fluctuations,      out_dir,
                                    title_suffix="All muons")
    _plot_learning_curve_normalized(n_arr, w1_ge77, fluctuations_ge77, out_dir,
                                    title_suffix="Ge77 muons", fname_suffix="_ge77")
    _plot_learning_curve_comparison(n_arr, w1_all, w1_ge77, out_dir)

    _plot_relative_improvement(n_arr, w1_all,  out_dir, title_suffix="All muons")
    _plot_relative_improvement(n_arr, w1_ge77, out_dir,
                               title_suffix="Ge77 muons", fname_suffix="_ge77")

    # SWD combined plot (one subplot, both populations)
    _plot_swd_learning_curve(n_arr, swd_all, swd_ge77, out_dir)

    _convergence_summary(n_arr, w1_all, w1_ge77, out_dir)


def run_statistical_convergence_test(
    data_path: Path,
    run_list: list[RunData],
    out_dir: Path,
) -> None:
    """Orchestrate the W1-based statistical convergence test.

    Level 2: Pairwise W1 fluctuation scale from all C(R,2) run pairs,
             for all muons and Ge77 muons separately.
    Level 3: W1 learning curve for subsamples N ∈ {1e4,1e5,1e6,1e7} vs the
             full 1e8 reference distribution, with power-law fit W1(N) = a·N^{-α}.
             Raises RuntimeError if no sub-sampled data directories exist.

    The scientific basis for the convergence limits (global absolute threshold
    and relative improvement threshold) is printed and saved to
    convergence_theory.txt.
    """
    n_muons_per_run = int(np.mean([rd.n_muons_total for rd in run_list]))
    total_muons     = sum(rd.n_muons_total for rd in run_list)

    print(f"\n{'='*60}")
    print("W1 Statistical Convergence Test")
    print(f"  {len(run_list)} runs  |  ~{n_muons_per_run:.2e} muons/run  "
          f"|  total ~{total_muons:.2e} muons")
    print(f"{'='*60}")

    fluctuations = level2_pairwise_fluctuations(
        run_list, extractor=_extract_nc_fields_run, label="all muons"
    )
    fluctuations_ge77 = level2_pairwise_fluctuations(
        run_list, extractor=_extract_nc_fields_run_ge77, label="Ge77 muons"
    )

    level3_learning_curve(data_path, run_list, fluctuations, fluctuations_ge77, out_dir)

    print(f"\n{'='*60}")
    print("Convergence test complete.")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Statistics output
# ---------------------------------------------------------------------------
def write_statistics(
    agg: dict,
    outlier_info: dict,
    recommendations: dict[str, int],
    run_list: list[RunData],
    out_dir: Path,
) -> None:
    """Write a statistics.txt summary file."""
    lines = [
        "=== MUSUN NC Analysis — Statistics ===",
        "",
        f"Runs loaded:               {len(run_list)}",
        f"Total muons:               {agg['n_muons_total']:,}",
        f"Muons producing NCs:       {agg['n_muons_with_nc']:,}"
        f"  ({agg['n_muons_with_nc'] / max(agg['n_muons_total'], 1) * 100:.3f}%)",
        f"Ge77-producing muons:      {agg['n_muons_ge77']:,}"
        f"  ({agg['n_muons_ge77'] / max(agg['n_muons_total'], 1) * 100:.3f}%)",
        f"Total NCs:                 {agg['n_nc_total']:,}",
        f"Ge77-flagged NCs:          {agg['n_nc_ge77']:,}"
        f"  ({agg['n_nc_ge77'] / max(agg['n_nc_total'], 1) * 100:.3f}%)",
        "",
        "--- Outlier analysis ---",
    ]
    if outlier_info:
        lines += [
            f"Threshold (99th pct log):  {outlier_info.get('threshold', 'n/a'):.1f} NCs/muon",
            f"Outlier muons:             {outlier_info.get('n_outliers', 'n/a'):,}",
            f"NC fraction from outliers: {outlier_info.get('frac_nc', 0) * 100:.2f}%",
        ]
        max_single = outlier_info.get("max_single_count", 0)
        if max_single > 0:
            lines.append(
                f"Max single-muon NC count:  {max_single:,}"
                f"  ({max_single / max(agg['n_nc_total'], 1) * 100:.2f}% of all NCs)"
            )
    else:
        lines.append("  (no outlier data)")

    lines += ["", "--- Convergence: recommended minimum number of runs ---"]
    obs_labels = {
        "nc_count":         "NC count per muon (full)",
        "nc_count_cut_100": f"NC count per muon (cut ≤{NC_CUT_100})",
        "nc_count_cut":     f"NC count per muon (cut ≤{NC_MUON_CUT:,})",
        "zenith":           "Muon zenith",
        "azimuth":          "Muon azimuth",
        "energy":           "Muon energy",
    }
    for key, label in obs_labels.items():
        k = recommendations.get(key, "n/a")
        lines.append(f"  {label:<25} k = {k}")

    lines += ["", "--- Per-run summary ---",
              f"{'Run':<12} {'Files':>6} {'NCs':>10} {'Ge77 NCs':>10} "
              f"{'NC muons':>10} {'Ge77 muons':>11}"]
    lines.append("-" * 65)
    for rd in run_list:
        lines.append(
            f"{rd.run_name:<12} {rd.n_files:>6} {rd.nc_evtid.size:>10} "
            f"{int((rd.nc_ge77 == 1).sum()):>10} "
            f"{rd.muon_nc_evtids.size:>10} {rd.ge77_evtids.size:>11}"
        )

    txt = "\n".join(lines) + "\n"
    out_path = out_dir / "statistics.txt"
    out_path.write_text(txt)
    print(f"  saved: {out_path.name}")
    print(txt)


# ---------------------------------------------------------------------------
# Unified W1 convergence entry point
# ---------------------------------------------------------------------------
def run_all_w1_convergence_analysis(
    data_path: Path,
    run_list: list[RunData],
    out_dir: Path,
) -> dict[str, int]:
    """Run all W1-based convergence analyses in one call.

    Order:
      1. W1 vs cumulative runs k  (convergence_analysis, full mode)
      2. Pairwise W1 for muon parameters  (pairwise_w1_analysis)
      3. Pairwise W1 fluctuation scale for NC fields  (level2_pairwise_fluctuations)
      4. Learning curve W1(N) vs 1e8 reference for NC fields  (level3_learning_curve)
         — requires sub-directories {data_path}/1e4/, 1e5/, 1e6/, 1e7/ to exist;
           raises RuntimeError if any are missing.

    Returns the k-recommendations dict from convergence_analysis.
    """
    recommendations = convergence_analysis(run_list, out_dir, full_convergence=True)
    pairwise_w1_analysis(run_list, out_dir)
    run_statistical_convergence_test(data_path, run_list, out_dir)
    return recommendations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Level-1 MUSUN analysis: muon simulation → NC extraction."
    )
    p.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Root directory containing run_001/ … run_010/ sub-directories.",
    )
    p.add_argument(
        "--output-path",
        default=DEFAULT_DATA_PATH,
        help="Base output directory.  A 'musun_nc_analysis/' sub-directory is "
             "created inside it.  Defaults to --data-path.",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=NUM_RUNS_DEFAULT,
        help=f"Number of runs to load (default: {NUM_RUNS_DEFAULT}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_base = Path(args.output_path) if args.output_path else data_path
    out_dir = output_base / "musun_nc_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"ERROR: data path not found: {data_path}")
        sys.exit(1)

    run_dirs = sorted(data_path.glob("run_*"))[: args.runs]
    if not run_dirs:
        print(f"ERROR: no run_* directories in {data_path}")
        sys.exit(1)
    print(f"Data path  : {data_path}")
    print(f"Output dir : {out_dir}")
    print(f"Runs found : {len(run_dirs)}\n")

    t_main = _log_resources("main start")

    # ---- Load ----
    run_list: list[RunData] = []
    for rd_path in run_dirs:
        print(f"Loading {rd_path.name} ...", flush=True)
        t_run = time.perf_counter()
        rd = load_run(rd_path)
        run_list.append(rd)
        elapsed = time.perf_counter() - t_run
        print(
            f"  {rd.n_files} files | "
            f"{rd.nc_evtid.size:,} NCs | "
            f"{rd.muon_nc_evtids.size:,} NC muons | "
            f"{rd.ge77_evtids.size:,} Ge77 muons | "
            f"{rd.n_muons_total:,} total muons | "
            f"{elapsed:.1f}s"
        )
    _log_resources("all runs loaded", t_main)

    # ---- Aggregate ----
    print("\nAggregating ...", flush=True)
    agg = aggregate_runs(run_list)
    _log_resources("aggregation done", t_main)
    print(
        f"Total: {agg['n_muons_total']:,} muons | "
        f"{agg['n_nc_total']:,} NCs | "
        f"{agg['n_muons_ge77']:,} Ge77 muons"
    )

    # ---- Plots ----
    print("\n--- Standard muon distributions ---")
    plot_muon_distributions(agg, out_dir)
    plot_ge77_fraction_bar(agg, out_dir)
    _log_resources("muon distribution plots done", t_main)

    print("\n--- NC distributions ---")
    plot_nc_count_per_muon(agg, out_dir)
    plot_ge77_per_ge77muon(agg, out_dir)
    plot_nc_times(agg, out_dir)
    plot_nc_positions(agg, out_dir)
    plot_nc_material_distribution(run_list, out_dir)
    plot_nc_gamma_count(agg, out_dir)
    _log_resources("NC distribution plots done", t_main)

    print("\n--- Outlier analysis ---")
    outlier_info = analyze_outliers(agg, out_dir)
    _log_resources("outlier analysis done", t_main)

    print("\n--- Outlier muon fingerprint analysis ---")
    analyze_repeated_outlier_muons(run_list, out_dir)
    _log_resources("fingerprint analysis done", t_main)

    print("\n--- Cut NC material ---")
    plot_cut_nc_material(run_list, out_dir)
    _log_resources("cut NC material done", t_main)

    print("\n--- W1 Convergence Analysis ---")
    recommendations = run_all_w1_convergence_analysis(data_path, run_list, out_dir)
    _log_resources("W1 convergence analysis done", t_main)

    print("\n--- Statistics ---")
    write_statistics(agg, outlier_info, recommendations, run_list, out_dir)
    _log_resources("total runtime", t_main)

    print("\nDone.")


if __name__ == "__main__":
    main()
