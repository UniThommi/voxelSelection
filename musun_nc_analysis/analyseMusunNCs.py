"""
Level-1 MUSUN analysis: MUSUN muon simulation → Neutron Capture (NC) extraction.

10 simulation runs with 1×10^7 muons each.

Outputs (all to <output_path>/musun_nc_analysis/):
  - Standard distributions: all muons + Ge77 only + Ge77 vs non-Ge77 + NC vs non-NC
  - NC spatial / time distributions
  - Outlier analysis (muons above 99th-percentile NC count on log scale)
  - Convergence analysis: W1 + KS vs number of runs
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
from scipy.stats import ks_2samp, wasserstein_distance

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
N_PERMUTATIONS = 20   # random subsets per k in convergence analysis
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
        rd.nc_z_m     = z_arr[unique_idx]
        rd.nc_r_m     = np.sqrt(x_arr[unique_idx]**2 + y_arr[unique_idx]**2)

        # Material names were already resolved per-file; apply dedup index
        rd.nc_material_name = mat_name_arr[unique_idx]
        rd.nc_is_water      = rd.nc_material_name == "Water"

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


def _w1_ks_sorted(sorted_p: np.ndarray, sorted_q: np.ndarray) -> tuple[float, float]:
    """W1 and KS between two *pre-sorted* empirical distributions.

    Memory layout (peak = 3 × (n+m) × 8 bytes, vs scipy's ~4×):
      1. all_vals  — merged sorted union, freed after np.diff
      2. deltas    — interval widths (nm-1,)
      3. cdf_diff  — |F_p - F_q| at each breakpoint (nm-1,)
    """
    n, m = len(sorted_p), len(sorted_q)
    if n == 0 or m == 0:
        return float("nan"), float("nan")

    nm = n + m
    all_vals = np.empty(nm, dtype=np.float64)
    all_vals[:n] = sorted_p
    all_vals[n:] = sorted_q
    all_vals.sort(kind="mergesort")   # O(nm): fast for two sorted halves

    deltas = np.diff(all_vals)        # (nm-1,) — new allocation
    bpts = all_vals[:-1]              # view, no copy

    # |F_p(x) - F_q(x)| at each left-breakpoint
    cdf_diff = (np.searchsorted(sorted_p, bpts, side="right") / n
                - np.searchsorted(sorted_q, bpts, side="right") / m)
    del all_vals                      # free nm floats before taking abs + dot
    np.abs(cdf_diff, out=cdf_diff)

    w1 = float(np.dot(cdf_diff, deltas))
    ks = float(cdf_diff.max())
    del cdf_diff, deltas
    return w1, ks


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
    Compute W1 and KS convergence metrics vs number of runs k.

    For NC count per muon: produces two figures (log-transformed and raw),
    each with two rows (full distribution / cut at NC_MUON_CUT) × two
    columns (W1 / KS).

    If full_convergence=True, also produces single W1+KS figures for zenith,
    azimuth, and energy.

    Returns recommended minimum k per observable.
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
    ) -> tuple[list[float], list[float], np.ndarray, np.ndarray]:
        """W1 and KS vs k for given pre-sorted per-run arrays."""
        sorted_ref = _merge_sorted_runs(sorted_runs_local, run_indices)
        if sorted_ref.size == 0:
            del sorted_ref
            return [], [], np.zeros((n_runs, N_PERMUTATIONS)), np.zeros((n_runs, N_PERMUTATIONS))
        det_w1, det_ks = [], []
        for k in k_vals:
            sk = _merge_sorted_runs(sorted_runs_local, list(range(k)))
            w1, ks = _w1_ks_sorted(sk, sorted_ref)
            del sk
            det_w1.append(w1 if not np.isnan(w1) else 0.0)
            det_ks.append(ks if not np.isnan(ks) else 0.0)
        rand_w1_m = np.zeros((n_runs, N_PERMUTATIONS))
        rand_ks_m = np.zeros((n_runs, N_PERMUTATIONS))
        for perm_i in range(N_PERMUTATIONS):
            shuffled = run_indices.copy()
            rng.shuffle(shuffled)
            for k in k_vals:
                sk = _merge_sorted_runs(sorted_runs_local, shuffled[:k])
                w1, ks = _w1_ks_sorted(sk, sorted_ref)
                del sk
                rand_w1_m[k - 1, perm_i] = w1 if not np.isnan(w1) else 0.0
                rand_ks_m[k - 1, perm_i] = ks if not np.isnan(ks) else 0.0
        del sorted_ref
        return det_w1, det_ks, rand_w1_m, rand_ks_m

    def _rec_k_from_w1(det_w1: list[float]) -> tuple[int, float]:
        if not det_w1 or det_w1[0] <= 0:
            return n_runs, 0.0
        thr = W1_THRESHOLD_FRAC * det_w1[0]
        rec = next((k for k, w in enumerate(det_w1, 1) if w <= thr), n_runs)
        return rec, thr

    def _draw_panel(
        ax: "plt.Axes",
        title: str,
        det_vals: list[float],
        r_mean: np.ndarray,
        r_std: np.ndarray,
        ylabel: str,
        threshold: float | None,
        rec_k: int | None,
    ) -> None:
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

    recommendations: dict[str, int] = {}
    t_conv = _log_resources("convergence start")

    # ------------------------------------------------------------------ #
    # NC count per muon — 4 figures:                                      #
    #   full dist  : 1×2 (W1 | KS), log and raw variants                 #
    #   cuts ≤100 and ≤1000: 2×2 (rows=cut | cols=W1/KS), log and raw    #
    # ------------------------------------------------------------------ #
    print("\n  Convergence: NC count per muon ...", flush=True)

    nc_full_log:   list[np.ndarray] = []
    nc_cut100_log: list[np.ndarray] = []
    nc_cut_log:    list[np.ndarray] = []
    nc_full_raw:   list[np.ndarray] = []
    nc_cut100_raw: list[np.ndarray] = []
    nc_cut_raw:    list[np.ndarray] = []
    for rd in run_list:
        full   = rd.nc_counts.astype(np.float64)
        cut100 = full[full <= NC_CUT_100]
        cut    = full[full <= NC_MUON_CUT]
        nc_full_log.append(np.sort(np.log(full + 1.0)))
        nc_cut100_log.append(
            np.sort(np.log(cut100 + 1.0)) if cut100.size > 0
            else np.array([], dtype=np.float64)
        )
        nc_cut_log.append(
            np.sort(np.log(cut + 1.0)) if cut.size > 0
            else np.array([], dtype=np.float64)
        )
        nc_full_raw.append(np.sort(full))
        nc_cut100_raw.append(np.sort(cut100) if cut100.size > 0
                             else np.array([], dtype=np.float64))
        nc_cut_raw.append(np.sort(cut) if cut.size > 0
                          else np.array([], dtype=np.float64))

    dw1_fl,    dks_fl,    rw1_fl,    rks_fl    = _compute_metrics(nc_full_log)
    dw1_c100l, dks_c100l, rw1_c100l, rks_c100l = _compute_metrics(nc_cut100_log)
    dw1_cl,    dks_cl,    rw1_cl,    rks_cl    = _compute_metrics(nc_cut_log)
    dw1_fr,    dks_fr,    rw1_fr,    rks_fr    = _compute_metrics(nc_full_raw)
    dw1_c100r, dks_c100r, rw1_c100r, rks_c100r = _compute_metrics(nc_cut100_raw)
    dw1_cr,    dks_cr,    rw1_cr,    rks_cr    = _compute_metrics(nc_cut_raw)
    del nc_full_log, nc_cut100_log, nc_cut_log, nc_full_raw, nc_cut100_raw, nc_cut_raw
    gc.collect()
    _log_resources("nc_count: all metrics computed", t_conv)

    rec_full,   thr_fl    = _rec_k_from_w1(dw1_fl)
    rec_cut100, thr_c100l = _rec_k_from_w1(dw1_c100l)
    rec_cut,    thr_cl    = _rec_k_from_w1(dw1_cl)
    recommendations["nc_count"]         = rec_full
    recommendations["nc_count_cut_100"] = rec_cut100
    recommendations["nc_count_cut"]     = rec_cut
    print(f"    Full dist       — W1 thr = {thr_fl:.4f}    →  rec k = {rec_full}")
    print(f"    Cut ≤{NC_CUT_100} dist  — W1 thr = {thr_c100l:.4f}  →  rec k = {rec_cut100}")
    print(f"    Cut ≤{NC_MUON_CUT:,} dist — W1 thr = {thr_cl:.4f}    →  rec k = {rec_cut}")

    # -- Full distribution: 1×2 figure per transform variant --
    for fig_tag, fig_label, d_w1, d_ks, r_w1, r_ks, xlabel, thr, rec in [
        ("log", "log-transformed",
         dw1_fl, dks_fl, rw1_fl, rks_fl, "log(NC+1)", thr_fl, rec_full),
        ("raw", "raw (no transform)",
         dw1_fr, dks_fr, rw1_fr, rks_fr, "NC count", None, None),
    ]:
        if not d_w1:
            continue
        rw1_mean = r_w1.mean(axis=1)
        rw1_std  = r_w1.std(axis=1)
        rks_mean = r_ks.mean(axis=1)
        rks_std  = r_ks.std(axis=1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"Convergence: NC count per muon — full distribution  ({fig_label})",
            fontsize=14,
        )
        _draw_panel(axes[0], "W₁  —  Full (all muons)",
                    d_w1, rw1_mean, rw1_std,
                    f"W₁ ({xlabel})", thr, rec)
        _draw_panel(axes[1], "KS  —  Full (all muons)",
                    d_ks, rks_mean, rks_std,
                    "KS statistic D", None, None)
        plt.tight_layout()
        _save(fig, out_dir / f"convergence_nc_count_full_{fig_tag}.png")

    # -- Cut distributions: 2×2 figure per transform variant --
    for fig_tag, fig_label, rows_data in [
        ("log", "log-transformed", [
            (f"Cut ≤ {NC_CUT_100}",
             dw1_c100l, dks_c100l, rw1_c100l, rks_c100l, "log(NC+1)", thr_c100l, rec_cut100),
            (f"Cut ≤ {NC_MUON_CUT:,}",
             dw1_cl, dks_cl, rw1_cl, rks_cl, "log(NC+1)", thr_cl, rec_cut),
        ]),
        ("raw", "raw (no transform)", [
            (f"Cut ≤ {NC_CUT_100}",
             dw1_c100r, dks_c100r, rw1_c100r, rks_c100r, "NC count", None, None),
            (f"Cut ≤ {NC_MUON_CUT:,}",
             dw1_cr, dks_cr, rw1_cr, rks_cr, "NC count", None, None),
        ]),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Convergence: NC count per muon — cut distributions  ({fig_label})",
            fontsize=14,
        )
        for row, (row_title, d_w1, d_ks, r_w1, r_ks, xlabel, thr, rec) in enumerate(rows_data):
            if not d_w1:
                continue
            rw1_mean = r_w1.mean(axis=1)
            rw1_std  = r_w1.std(axis=1)
            rks_mean = r_ks.mean(axis=1)
            rks_std  = r_ks.std(axis=1)
            _draw_panel(axes[row, 0],
                        f"W₁  —  {row_title}",
                        d_w1, rw1_mean, rw1_std,
                        f"W₁ ({xlabel})", thr, rec)
            _draw_panel(axes[row, 1],
                        f"KS  —  {row_title}",
                        d_ks, rks_mean, rks_std,
                        "KS statistic D", None, None)
        plt.tight_layout()
        _save(fig, out_dir / f"convergence_nc_count_cuts_{fig_tag}.png")

    # ------------------------------------------------------------------ #
    # Other observables (zenith, azimuth, energy) — only if requested     #
    # ------------------------------------------------------------------ #
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

        det_w1, det_ks, rand_w1, rand_ks = _compute_metrics(sorted_runs)
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
        rand_ks_mean = rand_ks.mean(axis=1)
        rand_ks_std  = rand_ks.std(axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Convergence analysis: {obs_title}", fontsize=14)
        for ax, metric, det_vals, r_mean, r_std, ylabel in [
            (axes[0], "W₁ distance", det_w1, rand_w1_mean, rand_w1_std,
             f"W₁ ({obs_xlabel})"),
            (axes[1], "KS statistic", det_ks, rand_ks_mean, rand_ks_std,
             "KS statistic D"),
        ]:
            thr_v = threshold if metric == "W₁ distance" else None
            rec_v = rec_k    if metric == "W₁ distance" else None
            _draw_panel(ax, metric, det_vals, r_mean, r_std, ylabel, thr_v, rec_v)

        _save(fig, out_dir / f"convergence_{obs_key}.png")
        _log_resources(f"{obs_key}: saved — total conv elapsed", t_conv)

    return recommendations


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
        "--full-convergence",
        action="store_true",
        default=False,
        help="Also compute convergence for zenith, azimuth, and energy (slow).",
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

    print("\n--- Convergence analysis ---")
    recommendations = convergence_analysis(run_list, out_dir,
                                           full_convergence=args.full_convergence)
    _log_resources("convergence analysis done", t_main)

    print("\n--- Statistics ---")
    write_statistics(agg, outlier_info, recommendations, run_list, out_dir)
    _log_resources("total runtime", t_main)

    print("\nDone.")


if __name__ == "__main__":
    main()
