"""
Level-1 MUSUN analysis: MUSUN muon simulation → Neutron Capture (NC) extraction.

10 simulation runs with 1×10^7 muons each.

Outputs (all to <output_path>/musun_nc_analysis/):
  - Standard distributions: all muons + Ge77 only + Ge77 vs non-Ge77 + NC vs non-NC
  - NC spatial / time distributions
  - Outlier analysis (muons above 99th-percentile NC count on log scale)
  - Monte Carlo uncertainty analysis: standard error ΔE_N = S/√N vs sample size N
  - statistics.txt
"""
from __future__ import annotations

import argparse
import gc
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

DEFAULT_DATA_PATH = (
    "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/rawMusunNCs"
)
NUM_RUNS_DEFAULT = 10
RANDOM_SEED = 42
MAX_SCATTER_PTS = 5_000    # max points for 3-D scatter / arrow plots
MC_N_DRAWS = 20            # random subsamples drawn per N grid point
MC_N_SIZES = 20            # number of log-spaced N grid points

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
    ax.hist(data, bins=bins, density=normalized, color=color,
            edgecolor="black", linewidth=0.3, alpha=0.85)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Probability density" if normalized else "Count", fontsize=13)
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
        ax.hist(data, bins=bins, density=normalized, color=color, alpha=0.7,
                edgecolor="black", linewidth=0.3, label=f"{label} (N={len(data):,})")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Probability density" if normalized else "Count", fontsize=13)
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


# ---------------------------------------------------------------------------
# Publication-quality comparison utilities  (axis-level building blocks)
# ---------------------------------------------------------------------------

def plot_normalized_histogram(
    ax: "plt.Axes",
    d1: np.ndarray, d2: np.ndarray,
    l1: str, l2: str, c1: str, c2: str,
    bins: np.ndarray,
    log_x: bool = False, log_y: bool = False,
) -> None:
    """Overlay density-normalised histograms for two classes on *ax*.

    Uses density=True: bin value = count / (N * Δx).  For variable-width bins
    (e.g. log-spaced energy) this is the only correct normalisation — equal-
    fraction weighting (1/N) would make wide bins appear falsely large because
    the comparison is not per-unit-x.  Identical bin edges are applied to both
    distributions so every bin represents the same x-interval.
    """
    for data, label, color in [(d1, l1, c1), (d2, l2, c2)]:
        if data.size == 0:
            continue
        ax.hist(data, bins=bins, density=True, color=color, alpha=0.7,
                edgecolor="black", linewidth=0.3,
                label=f"{label}  (N={data.size:,})")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel("Probability density", fontsize=11)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)


def plot_cdf(
    ax: "plt.Axes",
    d1: np.ndarray, d2: np.ndarray,
    l1: str, l2: str, c1: str, c2: str,
    bins: np.ndarray,
    log_x: bool = False,
) -> None:
    """Plot empirical CDFs for two classes using shared bin edges on *ax*.

    CDF[i] = fraction of events with value ≤ bins[i+1] (right bin edge).
    Using the same bin edges as the density histogram ensures that CDF
    breakpoints correspond exactly to histogram bins.  The vertical separation
    between the two CDFs at any x indicates cumulative discrimination power:
    larger gaps → the two distributions have diverged more by that point.
    """
    for data, label, color in [(d1, l1, c1), (d2, l2, c2)]:
        if data.size == 0:
            continue
        h, _ = np.histogram(data, bins=bins)
        cdf = np.cumsum(h).astype(float) / data.size
        x_step = np.concatenate([[bins[0]], bins[1:]])
        y_step = np.concatenate([[0.0], cdf])
        ax.step(x_step, y_step, where="post", color=color, linewidth=1.5,
                label=f"{label}  (N={data.size:,})")
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Cumulative probability", fontsize=11)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)


def plot_ratio(
    ax: "plt.Axes",
    d_num: np.ndarray, d_den: np.ndarray,
    l_num: str, l_den: str,
    bins: np.ndarray,
    log_x: bool = False,
) -> None:
    """Plot density ratio R(x) = p_num(x) / p_den(x) with Poisson error bars.

    Density in bin i:  p_i = k_i / (N * Δx_i),  k_i = raw bin count.
    Poisson uncertainty on the numerator counts: σ_k = √k_i.
    Propagated to the ratio:  σ_R = R / √k_i  (numerator errors only, as the
    denominator population is large enough to treat as a reference).
    Bins where the denominator density is zero are skipped.  R > 1 means the
    numerator class is enriched; R < 1 means it is depleted.
    """
    if d_num.size == 0 or d_den.size == 0:
        return
    bin_widths = np.diff(bins)
    h_num, _ = np.histogram(d_num, bins=bins)
    h_den, _ = np.histogram(d_den, bins=bins)
    p_num = h_num / (d_num.size * bin_widths)
    p_den = h_den / (d_den.size * bin_widths)
    bc = 0.5 * (bins[:-1] + bins[1:])

    valid = p_den > 0
    ratio = np.where(valid, p_num / np.where(valid, p_den, 1.0), np.nan)
    sigma = np.where(
        valid & (h_num > 0), ratio / np.sqrt(h_num.astype(float)), np.nan
    )

    mask = valid & np.isfinite(ratio)
    ax.errorbar(
        bc[mask], ratio[mask], yerr=sigma[mask],
        fmt="o-", color=COLORS["red"], markersize=4, linewidth=1.2,
        elinewidth=0.8, capsize=3, label=f"{l_num} / {l_den}",
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="R = 1")
    if log_x:
        ax.set_xscale("log")
    ax.set_ylabel(f"Density ratio  {l_num} / {l_den}", fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)


def _analysis_panel(
    d_num: np.ndarray, d_den: np.ndarray,
    l_num: str, l_den: str,
    c_num: str, c_den: str,
    bins: np.ndarray,
    xlabel: str, title: str, out_path: Path,
    log_x: bool = False, log_y: bool = False,
) -> None:
    """Three-panel comparison figure: density histogram | CDF | ratio with errors.

    Combines plot_normalized_histogram, plot_cdf, and plot_ratio into a single
    publication-quality figure.  The numerator class (d_num, typically Ge77)
    carries Poisson error bars in the ratio panel; the denominator (d_den) is
    treated as the reference distribution.  Reading the three panels together:
      left   — shape differences at each x value (probability density)
      centre  — cumulative discrimination power (CDF)
      right   — pointwise enrichment factor with statistical significance (ratio)
    """
    if d_num.size == 0 and d_den.size == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=13)

    plot_normalized_histogram(axes[0], d_num, d_den, l_num, l_den,
                               c_num, c_den, bins, log_x, log_y)
    axes[0].set_xlabel(xlabel, fontsize=11)
    axes[0].set_title("Probability density", fontsize=12)

    plot_cdf(axes[1], d_num, d_den, l_num, l_den, c_num, c_den, bins, log_x)
    axes[1].set_xlabel(xlabel, fontsize=11)
    axes[1].set_title("Cumulative distribution function", fontsize=12)

    plot_ratio(axes[2], d_num, d_den, l_num, l_den, bins, log_x)
    axes[2].set_xlabel(xlabel, fontsize=11)
    axes[2].set_title(f"Density ratio: {l_num} / {l_den}", fontsize=12)

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

        d_ge77   = d[obs["mask_ge77"]]
        d_noge77 = d[obs["mask_noge77"]]
        _comparison_hist(
            d_noge77, d_ge77,
            "Non-Ge77", "Ge77", COLORS["blue"], COLORS["red"],
            f"{title}: Ge77 vs non-Ge77", xl,
            out_dir / f"{base}_ge77_vs_noge77.png", bins, log_x=lx,
        )
        _analysis_panel(
            d_ge77, d_noge77,
            "Ge77", "Non-Ge77", COLORS["red"], COLORS["blue"],
            bins, xl, f"{title}: Ge77 vs non-Ge77 muons",
            out_dir / f"{base}_ge77_vs_noge77_panel.png", log_x=lx,
        )
        d_nc   = d[obs["mask_nc"]]
        d_nonc = d[obs["mask_nonc"]]
        _comparison_hist(
            d_nonc, d_nc,
            "No NC", "NC-producing", COLORS["blue"], COLORS["red"],
            f"{title}: NC-producing vs non-NC", xl,
            out_dir / f"{base}_nc_vs_nonc.png", bins, log_x=lx,
        )
        _analysis_panel(
            d_nc, d_nonc,
            "NC-producing", "No NC", COLORS["red"], COLORS["blue"],
            bins, xl, f"{title}: NC-producing vs non-NC muons",
            out_dir / f"{base}_nc_vs_nonc_panel.png", log_x=lx,
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
    _analysis_panel(
        counts_ge77, counts_noge77,
        "Ge77-producing", "Non-Ge77 NC-producing",
        COLORS["red"], COLORS["blue"],
        bins, "NC count per muon",
        "NC count per muon: Ge77-producing vs non-Ge77 NC-producing muons",
        out_dir / "nc_count_per_muon_comparison_panel.png",
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
    """NC capture time: Ge77-muon NCs vs non-Ge77-muon NCs (muon-level split)."""
    nc_time      = agg["nc_time_ns"]
    nc_is_ge77mu = agg["nc_is_ge77mu"]

    if nc_time.size == 0:
        return

    pos = nc_time > 0
    t_ge77mu   = nc_time[pos & nc_is_ge77mu]
    t_noge77mu = nc_time[pos & ~nc_is_ge77mu]

    t_all_pos = nc_time[pos]
    if t_all_pos.size == 0:
        return
    bins = make_log_bins(max(1.0, float(t_all_pos.min())), float(t_all_pos.max()))

    _comparison_hist(
        t_noge77mu, t_ge77mu,
        "Non-Ge77-muon NCs", "Ge77-muon NCs",
        COLORS["blue"], COLORS["red"],
        "NC capture time: Ge77-muon NCs vs non-Ge77-muon NCs",
        "NC capture time [ns]",
        out_dir / "nc_time_ge77_vs_noge77.png",
        bins, log_x=True, log_y=True,
    )
    _analysis_panel(
        t_ge77mu, t_noge77mu,
        "Ge77-muon NCs", "Non-Ge77-muon NCs",
        COLORS["red"], COLORS["blue"],
        bins, "NC capture time [ns]",
        "NC capture time: Ge77-muon NCs vs non-Ge77-muon NCs",
        out_dir / "nc_time_ge77_vs_noge77_panel.png",
        log_x=True,
    )


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

    # ---- 1-D φ: Ge77-muon NCs vs non-Ge77-muon NCs ----
    bins_phi     = np.linspace(-180, 180, 73)
    phi_ge77mu   = phi[is_ge77mu]
    phi_noge77mu = phi[~is_ge77mu]
    _comparison_hist(
        phi_noge77mu, phi_ge77mu,
        "Non-Ge77-muon NCs", "Ge77-muon NCs",
        COLORS["blue"], COLORS["red"],
        "NC azimuthal position: Ge77-muon NCs vs non-Ge77-muon NCs",
        "NC azimuth φ [°]",
        out_dir / "nc_phi_1d_comparison.png",
        bins_phi,
    )
    _analysis_panel(
        phi_ge77mu, phi_noge77mu,
        "Ge77-muon NCs", "Non-Ge77-muon NCs",
        COLORS["red"], COLORS["blue"],
        bins_phi, "NC azimuth φ [°]",
        "NC azimuthal position: Ge77-muon NCs vs non-Ge77-muon NCs",
        out_dir / "nc_phi_1d_comparison_panel.png",
    )

    # ---- 1-D z: Ge77-muon NCs vs non-Ge77-muon NCs ----
    bins_z     = np.linspace(float(z_mm.min()), float(z_mm.max()), 100)
    z_ge77mu   = z_mm[is_ge77mu]
    z_noge77mu = z_mm[~is_ge77mu]
    _comparison_hist(
        z_noge77mu, z_ge77mu,
        "Non-Ge77-muon NCs", "Ge77-muon NCs",
        COLORS["blue"], COLORS["red"],
        "NC z position: Ge77-muon NCs vs non-Ge77-muon NCs",
        "NC z position [mm]",
        out_dir / "nc_z_1d_comparison.png",
        bins_z,
    )
    _analysis_panel(
        z_ge77mu, z_noge77mu,
        "Ge77-muon NCs", "Non-Ge77-muon NCs",
        COLORS["red"], COLORS["blue"],
        bins_z, "NC z position [mm]",
        "NC z position: Ge77-muon NCs vs non-Ge77-muon NCs",
        out_dir / "nc_z_1d_comparison_panel.png",
    )

    # ---- 1-D r: Ge77-muon NCs vs non-Ge77-muon NCs ----
    bins_r     = np.linspace(0.0, float(r_mm.max()), 73)
    r_ge77mu   = r_mm[is_ge77mu]
    r_noge77mu = r_mm[~is_ge77mu]
    _comparison_hist(
        r_noge77mu, r_ge77mu,
        "Non-Ge77-muon NCs", "Ge77-muon NCs",
        COLORS["blue"], COLORS["red"],
        "NC radial position: Ge77-muon NCs vs non-Ge77-muon NCs",
        "NC radial position r [mm]",
        out_dir / "nc_r_1d_comparison.png",
        bins_r,
    )
    _analysis_panel(
        r_ge77mu, r_noge77mu,
        "Ge77-muon NCs", "Non-Ge77-muon NCs",
        COLORS["red"], COLORS["blue"],
        bins_r, "NC radial position r [mm]",
        "NC radial position: Ge77-muon NCs vs non-Ge77-muon NCs",
        out_dir / "nc_r_1d_comparison_panel.png",
    )

    # ---- 2-D φ vs z AND r vs z (separate colorbars, r-density correction) ----
    for mask, tag, title_base in [
        (np.ones(phi.size, dtype=bool), "all", "NC positions — all NCs"),
        (is_ge77mu, "ge77muons", "NC positions — NCs of Ge77-producing muons"),
    ]:
        ph = phi[mask]
        zm = z_mm[mask]
        rm = r_mm[mask]
        if ph.size == 0:
            continue

        # r vs z: divide by bin-centre r to correct for cylindrical area growth (dA ∝ r·dr·dz)
        r_bins_2d = np.linspace(0.0, float(rm.max()) * 1.02, 73)
        z_bins_2d = np.linspace(float(zm.min()), float(zm.max()), 101)
        H_r, r_edges, z_edges = np.histogram2d(rm, zm, bins=[r_bins_2d, z_bins_2d])
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        with np.errstate(invalid="ignore", divide="ignore"):
            H_r_density = H_r / r_centers[:, None]

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
        fig.suptitle(f"{title_base}  (N = {ph.size:,})", fontsize=14)

        # φ vs z: raw counts, colorbar placed directly right of this subplot
        hh1 = axes[0].hist2d(ph, zm, bins=[72, 100], cmap="viridis", vmin=0)
        axes[0].set_xlabel("NC azimuth φ [°]", fontsize=13)
        axes[0].set_ylabel("NC z position [mm]", fontsize=13)
        axes[0].set_title("φ vs z", fontsize=12)
        axes[0].tick_params(labelsize=11)
        fig.colorbar(hh1[3], ax=axes[0], label="Number of NCs")

        # r vs z: NCs / r to correct for cylindrical volume element
        pc = axes[1].pcolormesh(r_edges, z_edges, H_r_density.T, cmap="viridis", vmin=0)
        axes[1].set_xlabel("NC radial position r [mm]", fontsize=13)
        axes[1].set_ylabel("NC z position [mm]", fontsize=13)
        axes[1].set_title("r vs z  (NCs / r)", fontsize=12)
        axes[1].tick_params(labelsize=11)
        fig.colorbar(pc, ax=axes[1], label="NCs / r  [mm⁻¹]")

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
    _analysis_panel(
        ge77_g, noge77_g,
        "Ge77-muon NCs", "Non-Ge77-muon NCs",
        COLORS["red"], COLORS["blue"],
        bins, "Number of capture gammas per NC",
        "Capture-gamma multiplicity: Ge77-muon NCs vs non-Ge77-muon NCs",
        out_dir / "nc_gamma_count_ge77_vs_noge77_panel.png",
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
# Muon-level MC uncertainty analysis — helpers
# ---------------------------------------------------------------------------

def _read_nc_full(fp: Path) -> dict[str, np.ndarray]:
    """Read all NC fields needed for muon-level MC analysis from one HDF5 file.

    Reads gamma count and total gamma energy directly from NC_GROUP, avoiding
    the need to cross-join the CaptureGammas group.
    """
    empty: dict[str, np.ndarray] = {
        k: np.array([], dtype=dt)
        for k, dt in [
            ("evtid", np.int64), ("track_id", np.int64),
            ("ge77", np.int32), ("time_ns", np.float64),
            ("x_m", np.float64), ("y_m", np.float64), ("z_m", np.float64),
            ("n_gammas", np.int32), ("gamma_energy_kev", np.float64),
        ]
    }
    try:
        with h5py.File(fp, "r") as f:
            if NC_GROUP not in f:
                return empty
            grp = f[NC_GROUP]
            if int(grp["entries"][()]) == 0:
                return empty
            return {
                "evtid":            _pages(grp, "evtid").astype(np.int64),
                "track_id":         _pages(grp, "nC_track_id").astype(np.int64),
                "ge77":             _pages(grp, "nC_flag_Ge77").astype(np.int32),
                "time_ns":          _pages(grp, "nC_time_in_ns"),
                "x_m":              _pages(grp, "nC_x_position_in_m"),
                "y_m":              _pages(grp, "nC_y_position_in_m"),
                "z_m":              _pages(grp, "nC_z_position_in_m"),
                "n_gammas":         _pages(grp, "nC_gamma_amount").astype(np.int32),
                "gamma_energy_kev": _pages(grp, "nC_gamma_total_energy_in_keV"),
            }
    except Exception as exc:
        print(f"  ERROR reading NC data from {fp.name}: {exc}")
        return empty


def _load_single_run_for_mc(run_dir: Path) -> dict:
    """Load one run directory and return muon-level arrays.

    Reads all output_t*.hdf5 files, deduplicates NCs on (evtid, nC_track_id),
    deduplicates muons on evtid (first occurrence), then maps each NC to its
    muon to accumulate per-muon statistics.

    Returns dict keys:
        n_mu_total, mu_is_ge77, mu_nc_count,
        mu_nc_pos (N×3 mean position, metres),
        mu_nc_first_time (seconds of first NC per muon),
        mu_nc_mean_gammas, mu_nc_mean_energy_mev.
    All arrays have length n_mu_total; muons without NCs receive 0 / zero-vector.
    """
    _empty: dict = {
        "n_mu_total":            0,
        "mu_is_ge77":            np.array([], dtype=bool),
        "mu_nc_count":           np.array([], dtype=np.float64),
        "mu_nc_pos":             np.zeros((0, 3), dtype=np.float64),
        "mu_nc_first_time":      np.array([], dtype=np.float64),
        "mu_nc_mean_gammas":     np.array([], dtype=np.float64),
        "mu_nc_mean_energy_mev": np.array([], dtype=np.float64),
    }

    hdf5_files = sorted(run_dir.glob("output_t*.hdf5"))
    if not hdf5_files:
        print(f"  WARNING: no output_t*.hdf5 found in {run_dir}", flush=True)
        return _empty

    nc_ev_l: list[np.ndarray] = []
    nc_tr_l: list[np.ndarray] = []
    nc_ge77_l: list[np.ndarray] = []
    nc_t_l: list[np.ndarray] = []
    nc_x_l: list[np.ndarray] = []
    nc_y_l: list[np.ndarray] = []
    nc_z_l: list[np.ndarray] = []
    nc_ng_l: list[np.ndarray] = []
    nc_en_l: list[np.ndarray] = []
    mu_ev_l: list[np.ndarray] = []

    for fp in hdf5_files:
        nc = _read_nc_full(fp)
        if nc["evtid"].size > 0:
            nc_ev_l.append(nc["evtid"])
            nc_tr_l.append(nc["track_id"])
            nc_ge77_l.append(nc["ge77"])
            nc_t_l.append(nc["time_ns"])
            nc_x_l.append(nc["x_m"])
            nc_y_l.append(nc["y_m"])
            nc_z_l.append(nc["z_m"])
            nc_ng_l.append(nc["n_gammas"].astype(np.float64))
            nc_en_l.append(nc["gamma_energy_kev"])
        mu = read_muon_data_file(fp)
        if mu["evtid"].size > 0:
            mu_ev_l.append(mu["evtid"])

    # ---- Muon deduplication (first occurrence per evtid) ----
    mu_evtid: np.ndarray = (
        np.unique(np.concatenate(mu_ev_l)) if mu_ev_l
        else np.array([], dtype=np.int64)
    )
    n_mu = mu_evtid.size
    if n_mu == 0:
        return _empty

    # ---- NC deduplication on (evtid, nC_track_id) ----
    if nc_ev_l:
        ev_all  = np.concatenate(nc_ev_l)
        tr_all  = np.concatenate(nc_tr_l)
        pair_keys = np.stack([ev_all, tr_all], axis=1)
        _, uniq_idx, inverse = np.unique(
            pair_keys, axis=0, return_index=True, return_inverse=True
        )
        # Resolve duplicate NC records: keep max Ge77 flag
        ge77_flat  = np.concatenate(nc_ge77_l)
        uniq_ge77  = np.zeros(uniq_idx.size, dtype=np.int32)
        np.maximum.at(uniq_ge77, inverse, ge77_flat)

        nc_evtid = ev_all[uniq_idx]
        nc_ge77  = uniq_ge77
        nc_x     = np.concatenate(nc_x_l)[uniq_idx]
        nc_y     = np.concatenate(nc_y_l)[uniq_idx]
        nc_z     = np.concatenate(nc_z_l)[uniq_idx]
        nc_time  = np.concatenate(nc_t_l)[uniq_idx]
        nc_ngam  = np.concatenate(nc_ng_l)[uniq_idx]
        nc_energ = np.concatenate(nc_en_l)[uniq_idx]
        del ev_all, tr_all, pair_keys, ge77_flat, uniq_idx, inverse
    else:
        nc_evtid = np.array([], dtype=np.int64)
        nc_ge77  = np.array([], dtype=np.int32)
        nc_x = nc_y = nc_z = nc_time = nc_ngam = nc_energ = np.array([], dtype=np.float64)

    # ---- Ge77 muon set ----
    ge77_evtids = (
        np.unique(nc_evtid[nc_ge77 == 1]) if nc_evtid.size > 0
        else np.array([], dtype=np.int64)
    )
    mu_is_ge77 = np.isin(mu_evtid, ge77_evtids)

    # ---- Map NC evtids → muon array indices (mu_evtid is sorted by np.unique) ----
    mu_nc_count      = np.zeros(n_mu, dtype=np.float64)
    mu_nc_pos_x      = np.zeros(n_mu, dtype=np.float64)
    mu_nc_pos_y      = np.zeros(n_mu, dtype=np.float64)
    mu_nc_pos_z      = np.zeros(n_mu, dtype=np.float64)
    mu_nc_first_time = np.full(n_mu, np.inf, dtype=np.float64)
    mu_nc_ngam_sum   = np.zeros(n_mu, dtype=np.float64)
    mu_nc_energ_sum  = np.zeros(n_mu, dtype=np.float64)

    if nc_evtid.size > 0:
        nc_mu_idx = np.searchsorted(mu_evtid, nc_evtid)
        # Guard against NC evtids absent from the muon list
        valid = (
            (nc_mu_idx < n_mu)
            & (mu_evtid[np.clip(nc_mu_idx, 0, n_mu - 1)] == nc_evtid)
        )
        if not valid.all():
            nc_mu_idx = nc_mu_idx[valid]
            nc_x      = nc_x[valid];  nc_y = nc_y[valid];  nc_z = nc_z[valid]
            nc_time   = nc_time[valid]
            nc_ngam   = nc_ngam[valid]
            nc_energ  = nc_energ[valid]

        mu_nc_count   = np.bincount(nc_mu_idx, minlength=n_mu).astype(np.float64)
        mu_nc_pos_x   = np.bincount(nc_mu_idx, weights=nc_x,    minlength=n_mu)
        mu_nc_pos_y   = np.bincount(nc_mu_idx, weights=nc_y,    minlength=n_mu)
        mu_nc_pos_z   = np.bincount(nc_mu_idx, weights=nc_z,    minlength=n_mu)
        mu_nc_ngam_sum  = np.bincount(nc_mu_idx, weights=nc_ngam,  minlength=n_mu)
        mu_nc_energ_sum = np.bincount(nc_mu_idx, weights=nc_energ, minlength=n_mu)
        np.minimum.at(mu_nc_first_time, nc_mu_idx, nc_time)

    has_nc    = mu_nc_count > 0
    inv_count = np.where(has_nc, 1.0 / np.where(has_nc, mu_nc_count, 1.0), 0.0)

    mu_nc_pos = np.stack([
        mu_nc_pos_x * inv_count,
        mu_nc_pos_y * inv_count,
        mu_nc_pos_z * inv_count,
    ], axis=1)
    mu_nc_mean_gammas     = mu_nc_ngam_sum  * inv_count
    mu_nc_mean_energy_mev = mu_nc_energ_sum * inv_count / 1e3   # keV → MeV
    mu_nc_first_time      = np.where(has_nc, mu_nc_first_time, 0.0)

    return {
        "n_mu_total":            n_mu,
        "mu_is_ge77":            mu_is_ge77,
        "mu_nc_count":           mu_nc_count,
        "mu_nc_pos":             mu_nc_pos,
        "mu_nc_first_time":      mu_nc_first_time,
        "mu_nc_mean_gammas":     mu_nc_mean_gammas,
        "mu_nc_mean_energy_mev": mu_nc_mean_energy_mev,
    }


def _load_dataset_for_mc(dataset_dir: Path) -> dict:
    """Load a full dataset directory for muon-level MC analysis.

    Auto-detects layout:
      - Multi-run: run_*/ sub-directories each containing output_t*.hdf5 files.
      - Single-run: output_t*.hdf5 files directly in dataset_dir.

    Concatenates per-run muon-level arrays; evtid spaces are independent between
    runs so concatenation is correct without composite key bookkeeping.
    """
    run_dirs = sorted(dataset_dir.glob("run_*/"))
    if not run_dirs:
        run_dirs = [dataset_dir]

    parts: list[dict] = []
    for rd_path in run_dirs:
        print(f"    {rd_path.name} ...", end="  ", flush=True)
        p = _load_single_run_for_mc(rd_path)
        print(f"N_mu={p['n_mu_total']:,}", flush=True)
        parts.append(p)

    n_mu_total = sum(p["n_mu_total"] for p in parts)
    if n_mu_total == 0:
        return {
            "n_mu_total":            0,
            "mu_is_ge77":            np.array([], dtype=bool),
            "mu_nc_count":           np.array([], dtype=np.float64),
            "mu_nc_pos":             np.zeros((0, 3), dtype=np.float64),
            "mu_nc_first_time":      np.array([], dtype=np.float64),
            "mu_nc_mean_gammas":     np.array([], dtype=np.float64),
            "mu_nc_mean_energy_mev": np.array([], dtype=np.float64),
        }

    return {
        "n_mu_total": n_mu_total,
        "mu_is_ge77": np.concatenate([p["mu_is_ge77"] for p in parts]),
        "mu_nc_count": np.concatenate([p["mu_nc_count"] for p in parts]),
        "mu_nc_pos": np.concatenate([p["mu_nc_pos"] for p in parts], axis=0),
        "mu_nc_first_time": np.concatenate([p["mu_nc_first_time"] for p in parts]),
        "mu_nc_mean_gammas": np.concatenate([p["mu_nc_mean_gammas"] for p in parts]),
        "mu_nc_mean_energy_mev": np.concatenate(
            [p["mu_nc_mean_energy_mev"] for p in parts]
        ),
    }


def _mc_sem_fixed(
    data: np.ndarray,
    n_vals: np.ndarray,
    n_draws: int = MC_N_DRAWS,
    seed: int = RANDOM_SEED,
    vector: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """MC standard error on a caller-supplied N grid (fixed across datasets).

    data  : (N_total,) scalar or (N_total, 3) vector muon-level array.
    n_vals: 1-D int array of sample sizes; all values must be ≤ N_total.
    vector: if True, computes ‖ΔE⃗‖₂ = ‖(std_c/√N)‖ for each component c.

    Returns (sem_mean, sem_std) each of shape (len(n_vals),).
    """
    N_full = len(data)
    rng    = np.random.default_rng(seed)
    sem_means = np.zeros(len(n_vals), dtype=np.float64)
    sem_stds  = np.zeros(len(n_vals), dtype=np.float64)

    for i, n in enumerate(n_vals):
        vals: list[float] = []
        for _ in range(n_draws):
            idx = rng.choice(N_full, size=int(n), replace=True)
            if vector:
                sample = data[idx]                         # (n, 3)
                std_c  = np.std(sample, axis=0, ddof=1)   # (3,)
                vals.append(float(np.linalg.norm(std_c / np.sqrt(n))))
            else:
                s = float(np.std(data[idx], ddof=1))
                vals.append(s / np.sqrt(n))
        sem_means[i] = float(np.mean(vals))
        sem_stds[i]  = float(np.std(vals))

    return sem_means, sem_stds


# ---------------------------------------------------------------------------
# Muon-level MC uncertainty analysis — main function
# ---------------------------------------------------------------------------

def mc_uncertainty_analysis_muon_level(
    dataset_dirs: dict[str, Path],
    out_dir: Path,
) -> None:
    """MC standard error scaling with muon-level bootstrap across three datasets.

    For each dataset the full N_mu_total muon array is constructed: muons without
    NCs (or without Ge77) carry value 0 / zero-vector.  Bootstrap samples are
    drawn with replacement from this sparse array, so the variance correctly
    reflects the dilution by non-contributing muons.

    Observables (5 total, 3×2 subplot grid, last cell empty):
        nc_count        — NC count per muon              [scalar]
        nc_position     — mean NC position (x,y,z)       [vector → ‖ΔE⃗‖₂]
        nc_time         — first NC capture time           [scalar]
        nc_n_gammas     — mean gammas per NC per muon     [scalar]
        nc_gamma_energy — mean total gamma energy per NC  [scalar, MeV]

    Two populations per observable:
        all  — muons with ≥1 NC contribute their observable; others contribute 0
        Ge77 — Ge77-producing muons contribute their observable; others contribute 0

    N grid: log-spaced from 10³ to 10⁷ (fixed, MC_N_SIZES points, MC_N_DRAWS draws).

    A second y-axis (right, log scale) shows ρ_N = ΔE_N / |E_N| with a 5 %
    threshold line and N* crossing annotations.

    dataset_dirs keys must be "1e6", "1e7", "1e8".  The "1e8" path must contain
    run_001/…run_010/ sub-directories; "1e6" and "1e7" paths contain HDF5 files
    directly.

    Output: mc_uncertainty_scaling_muon_level.png
    """
    print("\n--- Monte Carlo uncertainty analysis (muon level) ---", flush=True)
    _t0 = time.perf_counter()

    N_FIXED   = 10**7
    N_MIN_FIX = 10**3
    RHO_THRESH = 0.05

    n_vals = np.unique(
        np.logspace(np.log10(N_MIN_FIX), np.log10(N_FIXED), MC_N_SIZES)
        .round().astype(int)
        .clip(2, N_FIXED)
    )  # int array, shape (≤MC_N_SIZES,)

    DS_COLORS: dict[str, str] = {
        "1e6": COLORS["blue"],
        "1e7": COLORS["green"],
        "1e8": COLORS["red"],
    }

    observables: list[dict] = [
        {
            "key":    "nc_count",
            "label":  "NC count per muon",
            "ylabel": r"$\Delta E_N = S/\sqrt{N}$  [NCs]",
            "vector": False,
        },
        {
            "key":    "nc_position",
            "label":  "NC position (x, y, z)",
            "ylabel": r"$\|\Delta\vec{E}_N\|_2$  [m]",
            "vector": True,
        },
        {
            "key":    "nc_time",
            "label":  "NC capture time",
            "ylabel": r"$\Delta E_N = S/\sqrt{N}$  [ns]",
            "vector": False,
        },
        {
            "key":    "nc_n_gammas",
            "label":  "Capture gammas per NC",
            "ylabel": r"$\Delta E_N = S/\sqrt{N}$  [gammas]",
            "vector": False,
        },
        {
            "key":    "nc_gamma_energy",
            "label":  "Total gamma energy per NC",
            "ylabel": r"$\Delta E_N = S/\sqrt{N}$  [MeV]",
            "vector": False,
        },
    ]

    # results[(ds_key, obs_key, pop)] → {sem_mean, sem_std, E_full, n_mu_total, k_ge77}
    results: dict[tuple, dict] = {}

    for i_ds, (ds_key, ds_dir) in enumerate(dataset_dirs.items()):
        print(f"\n  Dataset '{ds_key}'  ({ds_dir})", flush=True)
        t_ds = time.perf_counter()
        mu_data   = _load_dataset_for_mc(ds_dir)
        n_mu_total = mu_data["n_mu_total"]
        mu_is_ge77 = mu_data["mu_is_ge77"]
        k_ge77 = int(mu_is_ge77.sum())
        print(f"  N_mu={n_mu_total:,}  k_ge77={k_ge77:,}", flush=True)

        obs_arr: dict[str, np.ndarray] = {
            "nc_count":       mu_data["mu_nc_count"],
            "nc_position":    mu_data["mu_nc_pos"],
            "nc_time":        mu_data["mu_nc_first_time"],
            "nc_n_gammas":    mu_data["mu_nc_mean_gammas"],
            "nc_gamma_energy": mu_data["mu_nc_mean_energy_mev"],
        }

        for i_obs, obs in enumerate(observables):
            okey   = obs["key"]
            is_vec = obs["vector"]
            d_all  = obs_arr[okey]   # reference into mu_data; no copy

            # Ge77 array: identical to d_all for Ge77 muons, zero elsewhere
            if is_vec:
                d_ge77 = d_all.copy()
                d_ge77[~mu_is_ge77] = 0.0
            else:
                d_ge77 = np.where(mu_is_ge77, d_all, 0.0)

            for i_pop, (pop, data) in enumerate(
                [("all", d_all), ("ge77", d_ge77)]
            ):
                if pop == "ge77" and k_ge77 == 0:
                    continue

                if is_vec:
                    E_full = float(np.linalg.norm(np.mean(data, axis=0)))
                else:
                    E_full = float(np.mean(data))

                seed = RANDOM_SEED + i_obs * 12 + i_ds * 2 + i_pop
                sem_mean, sem_std = _mc_sem_fixed(
                    data, n_vals,
                    n_draws=MC_N_DRAWS, seed=seed, vector=is_vec,
                )
                results[(ds_key, okey, pop)] = {
                    "sem_mean":   sem_mean,
                    "sem_std":    sem_std,
                    "E_full":     E_full,
                    "n_mu_total": n_mu_total,
                    "k_ge77":     k_ge77,
                }

            del d_ge77   # free the per-observable copy; d_all is a reference

        del mu_data, obs_arr, mu_is_ge77
        gc.collect()
        _log_resources(f"  dataset '{ds_key}' done", t_ds)

    # ---- Plot ----------------------------------------------------------------
    n_obs  = len(observables)
    ncols  = 2
    nrows  = (n_obs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5.5 * nrows))
    axes_flat  = np.array(axes).flatten()
    n_arr_f    = n_vals.astype(np.float64)

    for idx, obs in enumerate(observables):
        ax   = axes_flat[idx]
        okey = obs["key"]

        ax2 = ax.twinx()
        ax2.set_yscale("log")
        ax2.set_ylabel(r"$\rho_N = \Delta E_N\,/\,|E_N|$",
                       fontsize=9, color="dimgray")
        ax2.tick_params(labelsize=8, colors="dimgray")
        ax2.axhline(RHO_THRESH, color="gray", linestyle=":",
                    linewidth=0.9, alpha=0.55)

        ref_n0: float | None   = None
        ref_sem0: float | None = None

        for ds_key in dataset_dirs:
            color = DS_COLORS.get(ds_key, "black")

            for pop, ls_left in [("all", "-"), ("ge77", "--")]:
                key = (ds_key, okey, pop)
                if key not in results:
                    continue
                r        = results[key]
                sem_mean = r["sem_mean"]
                sem_std  = r["sem_std"]
                E_full   = r["E_full"]
                k        = r["k_ge77"]
                n_mu_tot = r["n_mu_total"]

                if pop == "all":
                    label = f"{ds_key} — all  (N={n_mu_tot:,})"
                elif ds_key == "1e6":
                    label = f"{ds_key} — Ge77  (k≈{k}, high variance)"
                else:
                    label = f"{ds_key} — Ge77  (k≈{k})"

                # ΔE_N on left axis
                ax.plot(n_arr_f, sem_mean, ls_left, color=color,
                        linewidth=1.5, label=label)
                _rel = np.where(sem_mean > 0, sem_std / sem_mean, 0.0)
                ax.fill_between(
                    n_arr_f,
                    np.maximum(sem_mean * np.exp(-_rel), 1e-30),
                    sem_mean * np.exp(_rel),
                    color=color, alpha=0.12,
                )

                # Anchor 1/√N reference at first valid point of the 1e8 all curve
                if ds_key == "1e8" and pop == "all" and ref_n0 is None:
                    valid = np.isfinite(sem_mean) & (sem_mean > 0)
                    if valid.any():
                        i0      = int(np.argmax(valid))
                        ref_n0  = float(n_arr_f[i0])
                        ref_sem0 = float(sem_mean[i0])

                # ρ_N on right axis (same color, dotted)
                if abs(E_full) > 0:
                    rho    = sem_mean / abs(E_full)
                    finite = np.isfinite(rho) & (rho > 0)
                    if finite.any():
                        ax2.plot(n_arr_f[finite], rho[finite],
                                 ":", color=color, linewidth=0.9, alpha=0.65)

                        # Annotate N* (first N where ρ < 5 %)
                        below = np.where(finite & (rho < RHO_THRESH))[0]
                        if below.size > 0:
                            n_star = n_arr_f[below[0]]
                            ax2.annotate(
                                f"N*≈{n_star:.0e}",
                                xy=(n_star, RHO_THRESH),
                                xytext=(4, 4), textcoords="offset points",
                                fontsize=6, color=color, alpha=0.85,
                            )
                        else:
                            # ρ never crosses 5 % — annotate at rightmost point
                            ax2.text(
                                n_arr_f[-1], rho[finite][-1],
                                "ρ > 5%",
                                fontsize=6, color=color, alpha=0.70,
                                ha="right", va="bottom",
                            )

        # 1/√N reference on left axis
        if ref_n0 is not None and ref_sem0 is not None:
            n_ref = np.logspace(np.log10(ref_n0), np.log10(float(N_FIXED)), 100)
            ax.plot(n_ref, ref_sem0 * np.sqrt(ref_n0 / n_ref),
                    ":", color="gray", linewidth=1.2, alpha=0.7,
                    label=r"$1/\sqrt{N}$ ref.")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Sample size $N$", fontsize=11)
        ax.set_ylabel(obs["ylabel"], fontsize=10)
        ax.set_title(obs["label"], fontsize=12)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=9)

    for extra in axes_flat[n_obs:]:
        extra.set_visible(False)

    fig.suptitle(
        "Monte Carlo Uncertainty Scaling — Muon-Level Sampling\n"
        r"Left axis: $\Delta E_N = S/\sqrt{N}$  |  "
        r"Right axis (dotted): $\rho_N = \Delta E_N\,/\,|E_N|$, threshold 5\,\%"
        rf"   ({MC_N_DRAWS} draws per grid point)",
        fontsize=12,
    )
    plt.tight_layout()
    _save(fig, out_dir / "mc_uncertainty_scaling_muon_level.png")
    _log_resources("mc_uncertainty_analysis_muon_level done", _t0)


# ---------------------------------------------------------------------------
# Statistics output
# ---------------------------------------------------------------------------
def write_statistics(
    agg: dict,
    outlier_info: dict,
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
    p.add_argument(
        "--data-path-1e6",
        default=None,
        help="Directory containing the 1e6-muon dataset (single HDF5 file).  "
             "Defaults to <data-path>/1e6/.",
    )
    p.add_argument(
        "--data-path-1e7",
        default=None,
        help="Directory containing the 1e7-muon dataset (single HDF5 file).  "
             "Defaults to <data-path>/1e7/.",
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
    _log_resources("plot_muon_distributions", t_main)
    plot_ge77_fraction_bar(agg, out_dir)
    _log_resources("plot_ge77_fraction_bar", t_main)

    print("\n--- NC distributions ---")
    plot_nc_count_per_muon(agg, out_dir)
    _log_resources("plot_nc_count_per_muon", t_main)
    plot_ge77_per_ge77muon(agg, out_dir)
    _log_resources("plot_ge77_per_ge77muon", t_main)
    plot_nc_times(agg, out_dir)
    _log_resources("plot_nc_times", t_main)
    plot_nc_positions(agg, out_dir)
    _log_resources("plot_nc_positions", t_main)
    plot_nc_material_distribution(run_list, out_dir)
    _log_resources("plot_nc_material_distribution", t_main)
    plot_nc_gamma_count(agg, out_dir)
    _log_resources("plot_nc_gamma_count", t_main)

    print("\n--- Outlier analysis ---")
    outlier_info = analyze_outliers(agg, out_dir)
    _log_resources("analyze_outliers", t_main)

    print("\n--- Outlier muon fingerprint analysis ---")
    analyze_repeated_outlier_muons(run_list, out_dir)
    _log_resources("analyze_repeated_outlier_muons", t_main)

    print("\n--- Cut NC material ---")
    plot_cut_nc_material(run_list, out_dir)
    _log_resources("plot_cut_nc_material", t_main)

    print("\n--- Monte Carlo Uncertainty Analysis (muon level) ---")
    _path_1e6 = Path(args.data_path_1e6) if args.data_path_1e6 else data_path / "1e6"
    _path_1e7 = Path(args.data_path_1e7) if args.data_path_1e7 else data_path / "1e7"
    _ds_dirs: dict[str, Path] = {}
    for _ds_key, _ds_path in [("1e6", _path_1e6), ("1e7", _path_1e7), ("1e8", data_path)]:
        if _ds_path.exists():
            _ds_dirs[_ds_key] = _ds_path
        else:
            print(f"  Skipping dataset '{_ds_key}': path not found: {_ds_path}")
    if _ds_dirs:
        mc_uncertainty_analysis_muon_level(_ds_dirs, out_dir)
        _log_resources("mc_uncertainty_analysis_muon_level done", t_main)
    else:
        print("  No dataset paths found — skipping muon-level MC analysis.")

    # write_statistics only needs 5 scalar counts from agg; extract them before freeing.
    agg_stats = {k: agg[k] for k in ("n_muons_total", "n_nc_total", "n_nc_ge77",
                                      "n_muons_ge77", "n_muons_with_nc")}
    del agg
    gc.collect()
    _log_resources("freed agg", t_main)

    print("\n--- Statistics ---")
    write_statistics(agg_stats, outlier_info, run_list, out_dir)
    _log_resources("total runtime", t_main)

    print("\nDone.")


if __name__ == "__main__":
    main()
