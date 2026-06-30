"""
Light analysis for homogeneous PMT simulation (sim 2).

Correlates optical photons from sim 2 (homogeneous PMT run) with NC events
from sim 1 (MUSUN NC run).  All filters follow the same logic as
zone_ratio_analysis.py.

Photon → NC mapping uses the triple
  (run_id, muon_id, nc_id)  [sim 2]  ==  (run_id, evtid, nc_id)  [sim 1]
where run_id is the shared run_* directory name, muon_id is sim 2's
optical/muon_track_id (== sim 1's evtid), and nc_id is nC_track_id in both.
Each run is loaded in isolation, so the (muon_id, nc_id) pair is unique per run.

Filter pipeline (applied in order):
  1. PMT UID mask:   det_uid in [10_000_000, 1_000_000_000)
  2. NC match:       (evtid, nC_track_id) present in sim 1 NC set (RuntimeError otherwise)
  3. Time filter:    photon_time_relative <= 200 ns
  4. NC time window: NC capture in [1 µs, 200 µs] after muon

Run script: python lightAnalysisHomogeneousPMTs.py
"""
from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from analyseMusunNCs import (
    read_nc_data_file, read_material_map,
    COLORS, RANDOM_SEED,
    _log_resources, _save,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPTICAL_GROUP = "/hit/optical"
DEFAULT_SIM2_PATH = (
    "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface"
    "/rawOpticalHomogeneousPMTsFromMusunNCs"
)
DEFAULT_SIM1_PATH = (
    "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/rawMusunNCs"
)
NUM_RUNS_DEFAULT = 10
PMT_UID_MIN = 10_000_000
PMT_UID_MAX = 1_000_000_000
EXPECTED_PMT_COUNT = 300
TIME_FILTER_NS = 200.0
# Shared bin edges for relative-time histograms (all photons, full range)
TIME_HIST_BINS  = np.linspace(-50.0, 2000.0, 261)  # 260 bins, 10 ns each
NC_TIME_LOW_NS  = 1_000.0    # 1 µs  — lower bound of NC detection window
NC_TIME_HIGH_NS = 200_000.0  # 200 µs — upper bound of NC detection window
WL_BINS = np.linspace(200.0, 800.0, 121)            # 5 nm bins, 200–800 nm


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class LightRunData:
    """Per-run optical photon data after all filters."""
    run_name: str = ""
    # One entry per NC from sim 1 (including 0-photon NCs)
    nc_photon_counts: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int64))
    nc_is_ge77: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=bool))
    nc_x_m: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64))
    nc_y_m: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64))
    nc_z_m: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64))
    # True if the NC occurred in the water volume (resolved per-file from sim 1)
    nc_is_water: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=bool))
    # Distinct PMTs (det_uid count) that fired per NC
    nc_pmt_counts: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int64))
    # NC capture time relative to muon [ns] (from sim 1)
    nc_time_ns: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64))
    # True if the NC occurred in EnrichedGermanium0.913 (resolved per-file from sim 1)
    nc_in_germanium: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=bool))
    # One entry per NC-producing muon from sim 1
    # muon_photon_counts: total photons summed over the muon's window NCs
    muon_photon_counts: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int64))
    # Distinct PMTs (det_uid) unioned over the muon's window NCs
    muon_pmt_counts: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int64))
    # True if ANY of the muon's NCs (any time) carries the Ge77 flag
    muon_is_ge77: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=bool))
    # As muon_photon_counts / muon_pmt_counts, but WITHOUT the 200 ns photon cut
    # (Filter 3 skipped; the [1 µs, 200 µs] NC window is still applied).
    muon_photon_counts_nocut: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int64))
    muon_pmt_counts_nocut: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int64))


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------
def _pages(grp: h5py.Group, name: str) -> np.ndarray:
    return grp[name]["pages"][:]


def _load_optical_file(fp: Path) -> dict | None:
    """Load optical photon fields from one sim 2 HDF5 file."""
    try:
        with h5py.File(fp, "r") as f:
            if OPTICAL_GROUP not in f:
                return None
            grp = f[OPTICAL_GROUP]
            evtid = _pages(grp, "muon_track_id").astype(np.int64)
            if evtid.size == 0:
                return None
            return {
                "evtid":         evtid,
                "nc_track_id":   _pages(grp, "nC_track_id").astype(np.int64),
                "det_uid":       _pages(grp, "det_uid").astype(np.int64),
                "time_ns":       _pages(grp, "time_in_ns").astype(np.float64),
                "wavelength_nm": _pages(grp, "wavelength_in_nm").astype(np.float64),
            }
    except Exception as exc:
        print(f"  ERROR reading {fp.name}: {exc}")
        return None


def _load_nc_keys(sim1_run_dir: Path) -> tuple[
    dict[tuple[int, int], int],                        # nc_ge77
    dict[tuple[int, int], float],                       # nc_time (ns, relative to muon)
    dict[tuple[int, int], tuple[float, float, float]],  # nc_pos
    dict[tuple[int, int], bool],                        # nc_is_water
    dict[tuple[int, int], bool],                        # nc_in_germanium
    dict[int, list[tuple[int, int]]],                   # muon_nc
]:
    """
    Read all unique NC (evtid, track_id) keys from a sim 1 run directory.

    The material map is read per HDF5 file (each file is an independent
    Geant4 process and may assign different IDs to the same materials).

    Returns:
      nc_ge77:        (evtid, tid) -> ge77 flag (0 or 1)
      nc_time:        (evtid, tid) -> NC capture time relative to muon [ns]
      nc_pos:         (evtid, tid) -> (x_m, y_m, z_m)
      nc_is_water:    (evtid, tid) -> True if NC occurred in water
      nc_in_germanium:(evtid, tid) -> True if NC occurred in EnrichedGermanium0.913
      muon_nc:        muon_evtid  -> list of NC keys belonging to that muon
    """
    nc_ge77:         dict[tuple[int, int], int] = {}
    nc_time:         dict[tuple[int, int], float] = {}
    nc_pos:          dict[tuple[int, int], tuple[float, float, float]] = {}
    nc_is_water:     dict[tuple[int, int], bool] = {}
    nc_in_germanium: dict[tuple[int, int], bool] = {}
    muon_nc: dict[int, list] = defaultdict(list)

    for fp in sorted(sim1_run_dir.glob("output_t*.hdf5")):
        data = read_nc_data_file(fp)
        if data["evtid"].size == 0:
            continue
        # Per-file material map: each Geant4 process may assign different IDs
        fmat_map = read_material_map(fp)
        water_ids = {k for k, v in fmat_map.items() if v == "Water"}
        ge_ids    = {k for k, v in fmat_map.items() if v == "EnrichedGermanium0.913"}
        for eid, tid, ge77, t_ns, x, y, z, mid in zip(
            data["evtid"].tolist(), data["track_id"].tolist(),
            data["ge77"].tolist(),
            data["time_ns"].tolist(),
            data["x_m"].tolist(), data["y_m"].tolist(), data["z_m"].tolist(),
            data["material_id"].tolist(),
        ):
            key = (eid, tid)
            if key not in nc_ge77:
                nc_ge77[key]         = ge77
                nc_time[key]         = float(t_ns)
                nc_pos[key]          = (x, y, z)
                nc_is_water[key]     = (int(mid) in water_ids)
                nc_in_germanium[key] = (int(mid) in ge_ids)
                muon_nc[eid].append(key)
            elif ge77 > nc_ge77[key]:
                nc_ge77[key] = ge77

    # Cross-check: every NC with ge77_flag=1 must be in EnrichedGermanium0.913.
    # The reverse is not required — captures on Ge-70/72/73/74 are in germanium
    # but do not produce Ge-77.
    ge77_keys      = {k for k, v in nc_ge77.items()         if v == 1}
    germanium_keys = {k for k, v in nc_in_germanium.items() if v}
    flagged_outside_ge = ge77_keys - germanium_keys
    if flagged_outside_ge:
        raise RuntimeError(
            f"Ge77 flag / EnrichedGermanium0.913 mismatch in {sim1_run_dir}:\n"
            f"  ge77_flag=1 but NOT in EnrichedGermanium0.913: {len(flagged_outside_ge)} NCs"
        )

    return nc_ge77, nc_time, nc_pos, nc_is_water, nc_in_germanium, dict(muon_nc)


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------
def load_light_run(
    sim2_run_dir: Path,
    sim1_run_dir: Path,
    global_pmt_uids: set,
    filter_counts: dict,
    all_times_hist: np.ndarray,
    ge77_times_hist: np.ndarray,
    all_wl_hist: np.ndarray,
) -> LightRunData:
    """
    Load one sim 2 run, apply all filters, return per-NC and per-muon counts.

    Histogram arrays are pre-allocated and accumulated in-place (no sampling):
      all_times_hist / ge77_times_hist — relative photon time, before 200 ns cut
      all_wl_hist                      — wavelength of every NC-matched photon
                                         (all NCs, before the time/window cuts)

    Filter pipeline:
      1. PMT UID mask (det_uid in [PMT_UID_MIN, PMT_UID_MAX))
      2. NC match: validate every photon has a known NC in sim 1 (RuntimeError otherwise)
      3. Time filter: keep photons with t <= 200 ns of NC (prompt signal window)
      4. NC time window: only count photons from NCs in [1 µs, 200 µs] after muon

    Mutates all passed-in accumulator arguments.
    """
    rd = LightRunData(run_name=sim2_run_dir.name)

    nc_ge77, nc_time, nc_pos, nc_is_water, nc_in_germanium, muon_nc = _load_nc_keys(sim1_run_dir)
    if not nc_ge77:
        print(f"  WARNING: no NC data in sim 1 run {sim1_run_dir.name}")
        return rd

    ge77_nc_set  = frozenset(k for k, g in nc_ge77.items() if g == 1)
    valid_nc_set = frozenset(nc_ge77.keys())
    window_nc_set = frozenset(
        k for k, t in nc_time.items()
        if NC_TIME_LOW_NS <= t <= NC_TIME_HIGH_NS
    )
    print(f"  Sim1 window NCs [{NC_TIME_LOW_NS:.0f}, {NC_TIME_HIGH_NS:.0f}] ns: "
          f"{len(window_nc_set):,} / {len(nc_ge77):,}")
    nc_counter: dict[tuple[int, int], int]  = {k: 0 for k in window_nc_set}
    nc_pmt_set:  dict[tuple[int, int], set] = {k: set() for k in window_nc_set}
    # No-200ns-cut twins: same window NCs, but every NC-matched photon counts
    # (Filter 3 skipped) so we can quantify the effect of the 200 ns cut.
    nc_counter_nocut: dict[tuple[int, int], int]  = {k: 0 for k in window_nc_set}
    nc_pmt_set_nocut: dict[tuple[int, int], set] = {k: set() for k in window_nc_set}

    for fp in sorted(sim2_run_dir.glob("output_t*.hdf5")):
        data = _load_optical_file(fp)
        if data is None:
            continue

        evtid   = data["evtid"]
        nc_tid  = data["nc_track_id"]
        det_uid = data["det_uid"]
        time_ns = data["time_ns"]
        wl_nm   = data["wavelength_nm"]

        filter_counts["n_raw"] += len(evtid)

        # ---- Filter 1: PMT UID mask ----------------------------------------
        pmt_mask = (det_uid >= PMT_UID_MIN) & (det_uid < PMT_UID_MAX)
        global_pmt_uids.update(int(u) for u in det_uid[pmt_mask])

        evtid_p   = evtid[pmt_mask]
        nc_tid_p  = nc_tid[pmt_mask]
        det_uid_p = det_uid[pmt_mask]
        time_ns_p = time_ns[pmt_mask]
        wl_nm_p   = wl_nm[pmt_mask]
        filter_counts["n_after_pmt"] += len(evtid_p)

        if evtid_p.size == 0:
            continue

        # ---- Filter 2: NC match (every photon must exist in sim 1) ----------
        match_mask = np.fromiter(
            ((int(e), int(t)) in valid_nc_set
             for e, t in zip(evtid_p.tolist(), nc_tid_p.tolist())),
            dtype=bool, count=len(evtid_p),
        )
        n_unmatched = int((~match_mask).sum())
        if n_unmatched > 0:
            raise RuntimeError(
                f"{n_unmatched} photon(s) in {fp} passed PMT mask "
                "but have no matching (evtid, nC_track_id) in sim 1. "
                "Sim 2 should only contain NC events that exist in sim 1."
            )
        evtid_m   = evtid_p[match_mask]
        nc_tid_m  = nc_tid_p[match_mask]
        det_uid_m = det_uid_p[match_mask]
        time_ns_m = time_ns_p[match_mask]
        wl_nm_m   = wl_nm_p[match_mask]
        filter_counts["n_after_nc_match"] += len(evtid_m)

        if evtid_m.size == 0:
            continue

        # Wavelength histogram: every NC-matched photon (all NCs, no time/window cut)
        if wl_nm_m.size > 0:
            h_wl, _ = np.histogram(wl_nm_m, bins=WL_BINS)
            all_wl_hist += h_wl

        # Compute photon time relative to NC by subtracting sim 1 NC capture time
        nc_time_arr = np.fromiter(
            (nc_time[(int(e), int(t))]
             for e, t in zip(evtid_m.tolist(), nc_tid_m.tolist())),
            dtype=np.float64, count=len(evtid_m),
        )
        time_ns_rel = time_ns_m - nc_time_arr

        # Sanity check: photons must not arrive before their NC
        bad = time_ns_rel < -1.0
        if bad.any():
            raise RuntimeError(
                f"{bad.sum()} photon(s) with NC-relative time < -1 ns in {fp} "
                f"(min = {time_ns_rel[bad].min():.3f} ns). "
                "Photon arrival time must not precede the NC capture."
            )

        # Accumulate NC-relative time histograms (before 200 ns cut)
        h, _ = np.histogram(time_ns_rel, bins=TIME_HIST_BINS)
        all_times_hist += h
        ge77_mask_m = np.fromiter(
            ((int(e), int(t)) in ge77_nc_set
             for e, t in zip(evtid_m.tolist(), nc_tid_m.tolist())),
            dtype=bool, count=len(evtid_m),
        )
        t_ge77_rel = time_ns_rel[ge77_mask_m]
        if t_ge77_rel.size > 0:
            h_ge77, _ = np.histogram(t_ge77_rel, bins=TIME_HIST_BINS)
            ge77_times_hist += h_ge77

        # ---- No-200ns-cut path: apply only the NC time window (skip Filter 3) ----
        window_mask_m = np.fromiter(
            ((int(e), int(t)) in window_nc_set
             for e, t in zip(evtid_m.tolist(), nc_tid_m.tolist())),
            dtype=bool, count=len(evtid_m),
        )
        evtid_nw   = evtid_m[window_mask_m]
        nc_tid_nw  = nc_tid_m[window_mask_m]
        det_uid_nw = det_uid_m[window_mask_m]
        if evtid_nw.size > 0:
            c_nc = Counter(zip(evtid_nw.tolist(), nc_tid_nw.tolist()))
            for key, cnt in c_nc.items():
                nc_counter_nocut[key] += cnt
            for e_v, t_v, d_v in zip(
                evtid_nw.tolist(), nc_tid_nw.tolist(), det_uid_nw.tolist()
            ):
                nc_pmt_set_nocut[(int(e_v), int(t_v))].add(int(d_v))

        # ---- Filter 3: keep photons within 200 ns of NC (prompt signal) ----
        time_mask = time_ns_rel <= TIME_FILTER_NS
        evtid_t   = evtid_m[time_mask]
        nc_tid_t  = nc_tid_m[time_mask]
        det_uid_t = det_uid_m[time_mask]
        filter_counts["n_after_time"] += len(evtid_t)

        if evtid_t.size == 0:
            continue

        # ---- Filter 4: NC must be within the [1 µs, 200 µs] muon window ----
        window_mask = np.fromiter(
            ((int(e), int(t)) in window_nc_set
             for e, t in zip(evtid_t.tolist(), nc_tid_t.tolist())),
            dtype=bool, count=len(evtid_t),
        )
        evtid_w   = evtid_t[window_mask]
        nc_tid_w  = nc_tid_t[window_mask]
        det_uid_w = det_uid_t[window_mask]
        filter_counts["n_after_window"] = (
            filter_counts.get("n_after_window", 0) + len(evtid_w)
        )

        if evtid_w.size == 0:
            continue

        # Count photons per NC (window-valid NCs only)
        c = Counter(zip(evtid_w.tolist(), nc_tid_w.tolist()))
        for key, cnt in c.items():
            nc_counter[key] += cnt

        # Distinct PMTs per NC
        for e_v, t_v, d_v in zip(
            evtid_w.tolist(), nc_tid_w.tolist(), det_uid_w.tolist()
        ):
            nc_pmt_set[(int(e_v), int(t_v))].add(int(d_v))

    # Build output arrays — one entry per window NC only
    nc_keys_list = sorted(window_nc_set)
    rd.nc_photon_counts = np.array(
        [nc_counter[k] for k in nc_keys_list], dtype=np.int64)
    rd.nc_pmt_counts = np.array(
        [len(nc_pmt_set[k]) for k in nc_keys_list], dtype=np.int64)
    rd.nc_time_ns = np.array(
        [nc_time[k] for k in nc_keys_list], dtype=np.float64)
    rd.nc_is_ge77 = np.array(
        [nc_ge77[k] == 1 for k in nc_keys_list], dtype=bool)
    rd.nc_x_m = np.array([nc_pos[k][0] for k in nc_keys_list], dtype=np.float64)
    rd.nc_y_m = np.array([nc_pos[k][1] for k in nc_keys_list], dtype=np.float64)
    rd.nc_z_m = np.array([nc_pos[k][2] for k in nc_keys_list], dtype=np.float64)
    rd.nc_is_water = np.array(
        [nc_is_water.get(k, False) for k in nc_keys_list], dtype=bool)
    rd.nc_in_germanium = np.array(
        [nc_in_germanium.get(k, False) for k in nc_keys_list], dtype=bool)

    # Per NC-producing muon: total photons, distinct PMTs, and Ge77-muon flag.
    # A muon is a Ge77 muon if ANY of its NCs (any time) carries the Ge77 flag
    # (matches analyseMusunNCs's ge77_evtids / nc_is_ge77mu definition).
    muon_photon_counts: list[int] = []
    muon_pmt_counts:    list[int] = []
    muon_is_ge77:       list[bool] = []
    muon_photon_counts_nocut: list[int] = []
    muon_pmt_counts_nocut:    list[int] = []
    for keys in muon_nc.values():
        muon_photon_counts.append(sum(nc_counter.get(k, 0) for k in keys))
        pmts: set = set()
        for k in keys:
            pmts |= nc_pmt_set.get(k, set())
        muon_pmt_counts.append(len(pmts))
        muon_is_ge77.append(any(nc_ge77.get(k, 0) == 1 for k in keys))
        # No-200ns-cut twins (same muon ordering)
        muon_photon_counts_nocut.append(
            sum(nc_counter_nocut.get(k, 0) for k in keys))
        pmts_nc: set = set()
        for k in keys:
            pmts_nc |= nc_pmt_set_nocut.get(k, set())
        muon_pmt_counts_nocut.append(len(pmts_nc))
    rd.muon_photon_counts = np.array(muon_photon_counts, dtype=np.int64)
    rd.muon_pmt_counts    = np.array(muon_pmt_counts,    dtype=np.int64)
    rd.muon_is_ge77       = np.array(muon_is_ge77,       dtype=bool)
    rd.muon_photon_counts_nocut = np.array(muon_photon_counts_nocut, dtype=np.int64)
    rd.muon_pmt_counts_nocut    = np.array(muon_pmt_counts_nocut,    dtype=np.int64)

    return rd


# ---------------------------------------------------------------------------
# Filter impact plots
# ---------------------------------------------------------------------------
def plot_filter_impact_pmt(filter_counts: dict, out_dir: Path) -> None:
    """Bar chart: photon counts before/after PMT UID mask."""
    n_raw = filter_counts["n_raw"]
    n_pmt = filter_counts["n_after_pmt"]
    n_non_pmt = n_raw - n_pmt

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["All photons", "PMT photons\n(kept)", "Non-PMT\n(removed)"]
    vals   = [n_raw, n_pmt, n_non_pmt]
    clrs   = [COLORS["blue"], COLORS["green"], COLORS["red"]]
    bars = ax.bar(labels, vals, color=clrs, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, vals):
        pct = val / n_raw * 100 if n_raw > 0 else 0.0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of photons", fontsize=13)
    ax.set_title("Filter impact: PMT UID mask", fontsize=14)
    ax.tick_params(labelsize=11)
    _save(fig, out_dir / "filter_impact_pmt.png")


def plot_filter_impact_time(
    all_times_hist: np.ndarray, filter_counts: dict, out_dir: Path
) -> None:
    """
    Relative-time histogram (all PMT photons, before 200 ns cut).

    all_times_hist contains counts for TIME_HIST_BINS.  Every photon
    that passed the PMT mask is included — no sampling.
    """
    if all_times_hist.sum() == 0:
        return

    n_pmt  = filter_counts["n_after_pmt"]
    n_time = filter_counts["n_after_time"]
    frac_removed = (n_pmt - n_time) / n_pmt * 100 if n_pmt > 0 else 0.0

    bin_centers = 0.5 * (TIME_HIST_BINS[:-1] + TIME_HIST_BINS[1:])
    cut_idx = int(np.searchsorted(TIME_HIST_BINS, TIME_FILTER_NS, side="left"))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(bin_centers, all_times_hist,
           width=TIME_HIST_BINS[1] - TIME_HIST_BINS[0],
           color=COLORS["blue"], alpha=0.8, linewidth=0,
           label="Kept (t ≤ 200 ns)")
    # Shade removed region (photons arriving > 200 ns after NC)
    ax.bar(bin_centers[cut_idx:], all_times_hist[cut_idx:],
           width=TIME_HIST_BINS[1] - TIME_HIST_BINS[0],
           color="red", alpha=0.5, linewidth=0,
           label=f"Removed (t > {TIME_FILTER_NS:.0f} ns)  ({frac_removed:.1f}% of PMT photons)")
    ax.axvline(TIME_FILTER_NS, color="red", linestyle="--", linewidth=2,
               label=f"{TIME_FILTER_NS:.0f} ns signal window boundary")
    ax.set_xlabel("Photon time relative to NC  [ns]  (t_photon − t_NC)", fontsize=13)
    ax.set_ylabel("Photon count  (all runs, all PMT photons)", fontsize=13)
    ax.set_title("Time filter: keep photons within 200 ns of NC", fontsize=14)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    _save(fig, out_dir / "filter_impact_time.png")


# ---------------------------------------------------------------------------
# Light-distribution histograms
# ---------------------------------------------------------------------------
def _photon_hist(
    counts: np.ndarray,
    color: str,
    xlabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Log-log histogram for a photon count array (includes 0s in title)."""
    if counts.size == 0:
        return
    counts_pos = counts[counts > 0]
    n_zero = int((counts == 0).sum())
    n_total = len(counts)

    fig, ax = plt.subplots(figsize=(10, 6))
    if counts_pos.size > 0:
        bins = np.logspace(0, np.log10(max(float(counts_pos.max()), 10)), 50)
        ax.hist(counts_pos, bins=bins, color=color, alpha=0.8,
                edgecolor="black", linewidth=0.3,
                label=f">0 photons  (N = {counts_pos.size:,})")
        ax.axvline(float(np.mean(counts_pos)), color="red", linestyle="--",
                   linewidth=1.5, label=f"Mean: {np.mean(counts_pos):.1f}")
        ax.axvline(float(np.median(counts_pos)), color="darkred",
                   linestyle=":", linewidth=1.5,
                   label=f"Median: {np.median(counts_pos):.0f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1.0]))
    ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(
        f"{title}\n"
        f"N_total={n_total:,}  |  zero-photon: {n_zero:,}"
        f" ({n_zero / n_total * 100:.1f}%)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    _save(fig, out_path)


def plot_light_per_nc(light_run_list: list[LightRunData], out_dir: Path) -> None:
    counts = np.concatenate(
        [rd.nc_photon_counts for rd in light_run_list
         if rd.nc_photon_counts.size > 0]
    )
    _photon_hist(counts, COLORS["blue"],
                 "Photon count per NC", "Detected photons per NC",
                 out_dir / "light_per_nc.png")


# ---------------------------------------------------------------------------
# Photon count per muon: Ge77 muons vs non-Ge77 muons
# ---------------------------------------------------------------------------
def plot_photon_count_comparison(
    light_run_list: list[LightRunData], out_dir: Path,
    *,
    attr: str = "muon_photon_counts",
    title: str = "Photon count per muon: Ge77 muons vs non-Ge77 muons",
    out_name: str = "photon_count_ge77_vs_noge77.png",
) -> None:
    """Photon count per muon: Ge77 muons vs non-Ge77 muons, density + ratio panel.

    A Ge77 muon is a muon with at least one Ge77-flagged NC; photon counts are
    summed over each muon's window NCs (per-muon aggregate).

    ``attr`` selects which per-muon array to use — ``muon_photon_counts`` (with
    the 200 ns photon cut) or ``muon_photon_counts_nocut`` (200 ns cut removed).
    """
    parts_ge77   = [getattr(rd, attr)[rd.muon_is_ge77]
                    for rd in light_run_list if getattr(rd, attr).size > 0]
    parts_noge77 = [getattr(rd, attr)[~rd.muon_is_ge77]
                    for rd in light_run_list if getattr(rd, attr).size > 0]
    if not parts_ge77 and not parts_noge77:
        return

    ge77_counts   = np.concatenate(parts_ge77)   if parts_ge77   else np.array([], dtype=np.int64)
    noge77_counts = np.concatenate(parts_noge77) if parts_noge77 else np.array([], dtype=np.int64)

    all_pos = np.concatenate(
        [ge77_counts[ge77_counts > 0], noge77_counts[noge77_counts > 0]]
    )
    if all_pos.size == 0:
        return
    bins = np.logspace(0, np.log10(max(float(all_pos.max()), 10)), 50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13)

    for data, label, color in [
        (noge77_counts, "Non-Ge77 muons", COLORS["blue"]),
        (ge77_counts,   "Ge77 muons",     COLORS["red"]),
    ]:
        d_pos  = data[data > 0]
        n_zero = int((data == 0).sum())
        if d_pos.size == 0:
            continue
        axes[0].hist(d_pos, bins=bins, density=True, color=color, alpha=0.70,
                     edgecolor="black", linewidth=0.3,
                     label=f"{label}  (N>0={d_pos.size:,}, zero={n_zero:,})")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Photon count per muon (>0)", fontsize=12)
    axes[0].set_ylabel("Probability density", fontsize=12)
    axes[0].set_title("Density comparison", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].tick_params(labelsize=10)

    bin_widths   = np.diff(bins)
    h_ge77,   _  = np.histogram(ge77_counts[ge77_counts > 0],     bins=bins)
    h_noge77, _  = np.histogram(noge77_counts[noge77_counts > 0], bins=bins)
    n_ge77_pos   = max(int((ge77_counts   > 0).sum()), 1)
    n_noge77_pos = max(int((noge77_counts > 0).sum()), 1)
    p_ge77   = h_ge77   / (n_ge77_pos   * bin_widths)
    p_noge77 = h_noge77 / (n_noge77_pos * bin_widths)
    bc    = 0.5 * (bins[:-1] + bins[1:])
    valid = p_noge77 > 0
    ratio = np.where(valid, p_ge77 / np.where(valid, p_noge77, 1.0), np.nan)
    sigma = np.where(valid & (h_ge77 > 0), ratio / np.sqrt(h_ge77.astype(float)), np.nan)
    mask  = valid & np.isfinite(ratio)
    if mask.any():
        axes[1].errorbar(bc[mask], ratio[mask], yerr=sigma[mask],
                         fmt="o-", color=COLORS["red"], markersize=4, linewidth=1.2,
                         elinewidth=0.8, capsize=3)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="R = 1")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Photon count per muon (>0)", fontsize=12)
    axes[1].set_ylabel("Density ratio  Ge77 / non-Ge77", fontsize=12)
    axes[1].set_title("Density ratio", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].tick_params(labelsize=10)

    plt.tight_layout()
    _save(fig, out_dir / out_name)


# ---------------------------------------------------------------------------
# PMT multiplicity per NC
# ---------------------------------------------------------------------------
def plot_pmt_multiplicity(
    light_run_list: list[LightRunData], out_dir: Path,
    *,
    attr: str = "muon_pmt_counts",
    title: str = "Distinct PMTs per muon: Ge77 muons vs non-Ge77 muons",
    out_name: str = "pmt_multiplicity_ge77_vs_noge77.png",
) -> None:
    """Distinct PMTs per muon (det_uid count unioned over the muon's window NCs):
    Ge77 muons vs non-Ge77 muons, density + ratio panel.

    ``attr`` selects ``muon_pmt_counts`` (with the 200 ns photon cut) or
    ``muon_pmt_counts_nocut`` (200 ns cut removed)."""
    parts_ge77   = [getattr(rd, attr)[rd.muon_is_ge77]
                    for rd in light_run_list if getattr(rd, attr).size > 0]
    parts_noge77 = [getattr(rd, attr)[~rd.muon_is_ge77]
                    for rd in light_run_list if getattr(rd, attr).size > 0]
    if not parts_ge77 and not parts_noge77:
        return

    ge77_pmt   = np.concatenate(parts_ge77)   if parts_ge77   else np.array([], dtype=np.int64)
    noge77_pmt = np.concatenate(parts_noge77) if parts_noge77 else np.array([], dtype=np.int64)
    all_pmt    = np.concatenate([ge77_pmt, noge77_pmt])
    if all_pmt.size == 0:
        return

    max_pmt = int(all_pmt.max())
    bins = np.arange(0, max_pmt + 2) - 0.5

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13)

    for data, label, color in [
        (noge77_pmt, "Non-Ge77 muons", COLORS["blue"]),
        (ge77_pmt,   "Ge77 muons",     COLORS["red"]),
    ]:
        if data.size == 0:
            continue
        axes[0].hist(data, bins=bins, density=True, color=color, alpha=0.70,
                     edgecolor="black", linewidth=0.3,
                     label=f"{label}  (N={data.size:,})")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Distinct PMTs per muon", fontsize=12)
    axes[0].set_ylabel("Probability density", fontsize=12)
    axes[0].set_title("Density comparison", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].tick_params(labelsize=10)

    bin_widths = np.diff(bins)
    h_ge77,   _ = np.histogram(ge77_pmt,   bins=bins)
    h_noge77, _ = np.histogram(noge77_pmt, bins=bins)
    n_ge77   = max(ge77_pmt.size,   1)
    n_noge77 = max(noge77_pmt.size, 1)
    p_ge77   = h_ge77   / (n_ge77   * bin_widths)
    p_noge77 = h_noge77 / (n_noge77 * bin_widths)
    bc    = 0.5 * (bins[:-1] + bins[1:])
    valid = p_noge77 > 0
    ratio = np.where(valid, p_ge77 / np.where(valid, p_noge77, 1.0), np.nan)
    sigma = np.where(valid & (h_ge77 > 0), ratio / np.sqrt(h_ge77.astype(float)), np.nan)
    mask  = valid & np.isfinite(ratio)
    if mask.any():
        axes[1].errorbar(bc[mask], ratio[mask], yerr=sigma[mask],
                         fmt="o-", color=COLORS["red"], markersize=4, linewidth=1.2,
                         elinewidth=0.8, capsize=3)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="R = 1")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Distinct PMTs per muon", fontsize=12)
    axes[1].set_ylabel("Density ratio  Ge77 / non-Ge77", fontsize=12)
    axes[1].set_title("Density ratio", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].tick_params(labelsize=10)

    plt.tight_layout()
    _save(fig, out_dir / out_name)


# ---------------------------------------------------------------------------
# All-muon cut vs no-cut comparison (photon yield / PMT multiplicity)
# ---------------------------------------------------------------------------
def plot_cut_comparison_all_muons(
    light_run_list: list[LightRunData], out_dir: Path,
    *,
    attr_cut: str,
    attr_nocut: str,
    xlabel: str,
    suptitle: str,
    out_name: str,
    bin_mode: str,  # "log" (photon yield) or "int" (PMT multiplicity)
    muon_mask_attr: str | None = None,
) -> None:
    """Two-panel comparison over NC-producing muons: with 200 ns cut (left)
    vs without it (right).  The right (no-cut) panel uses a logarithmic y-axis.

    Each panel is annotated with mean and median (vertical lines) and a title
    line giving N, the zero-detected fraction, std, and max of the >0 values.

    ``muon_mask_attr`` optionally names a per-run boolean attribute (e.g.
    ``muon_is_ge77``) used to restrict the muons; when None all muons are used.
    """
    def _concat(attr: str) -> np.ndarray:
        arrs = []
        for rd in light_run_list:
            vals = getattr(rd, attr)
            if vals.size == 0:
                continue
            if muon_mask_attr is not None:
                vals = vals[getattr(rd, muon_mask_attr)]
            if vals.size > 0:
                arrs.append(vals)
        return np.concatenate(arrs) if arrs else np.array([], dtype=np.int64)

    cut   = _concat(attr_cut)
    nocut = _concat(attr_nocut)
    if cut.size == 0 and nocut.size == 0:
        return

    # Shared bins so the two panels are directly comparable
    if bin_mode == "log":
        all_pos = np.concatenate([cut[cut > 0], nocut[nocut > 0]])
        if all_pos.size == 0:
            return
        bins = np.logspace(0, np.log10(max(float(all_pos.max()), 10)), 50)
    else:
        all_v = np.concatenate([cut, nocut])
        max_v = int(all_v.max()) if all_v.size > 0 else 1
        bins = np.arange(0, max_v + 2) - 0.5

    # Shared x and y axes so both panels use identical scales and limits
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    fig.suptitle(suptitle, fontsize=14)

    panels = [
        (axes[0], cut,   "With 200 ns cut", COLORS["blue"]),
        (axes[1], nocut, "No 200 ns cut",   COLORS["orange"]),
    ]
    for ax, data, label, color in panels:
        pos = data[data > 0]
        # Log-mode drops the zeros (no log-x bin for 0); int-mode keeps them.
        hist_data = pos if bin_mode == "log" else data
        if hist_data.size > 0:
            ax.hist(hist_data, bins=bins, color=color, alpha=0.80,
                    edgecolor="black", linewidth=0.3)
        if bin_mode == "log":
            ax.set_xscale("log")
        ax.set_yscale("log")

        if pos.size > 0:
            mean = float(pos.mean())
            med  = float(np.median(pos))
            ax.axvline(mean, color="red", linestyle="--", linewidth=1.5,
                       label=f"Mean: {mean:.1f}")
            ax.axvline(med, color="darkred", linestyle=":", linewidth=1.5,
                       label=f"Median: {med:.0f}")
            std = float(pos.std())
            mx  = int(pos.max())
        else:
            std, mx = 0.0, 0

        n      = data.size
        n_zero = int((data == 0).sum())
        frac   = n_zero / n * 100 if n > 0 else 0.0
        ax.set_title(
            f"{label}\n"
            f"N = {n:,}  |  zero: {n_zero:,} ({frac:.1f}%)  |  "
            f"std = {std:.1f}  |  max = {mx:,}",
            fontsize=11,
        )
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Number of muons", fontsize=12)
        if pos.size > 0:
            ax.legend(fontsize=10)
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    _save(fig, out_dir / out_name)


# ---------------------------------------------------------------------------
# Ge77 muons with zero detected signal
# ---------------------------------------------------------------------------
def plot_ge77_muon_zero_photons(
    light_run_list: list[LightRunData], out_dir: Path,
) -> None:
    """Bar chart: how many Ge77 muons have 0 detected photons / PMTs.

    A Ge77 muon is a muon with at least one Ge77-flagged NC.  Four metric
    variants are shown side by side, each split into "≥1 detected" (green) and
    "0 detected" (red):

      * Photons, 200 ns cut      — muon_photon_counts
      * Photons, no 200 ns cut   — muon_photon_counts_nocut
      * PMTs,    200 ns cut       — muon_pmt_counts
      * PMTs,    no 200 ns cut    — muon_pmt_counts_nocut

    All four share the same Ge77-muon denominator (counts summed/unioned over
    each muon's window NCs).
    """
    parts = [getattr(rd, "muon_photon_counts")[rd.muon_is_ge77]
             for rd in light_run_list if rd.muon_is_ge77.size > 0]
    if not parts or sum(p.size for p in parts) == 0:
        print("  No Ge77 muons found — skipping zero-photon bar chart.")
        return

    def _ge77(attr: str) -> np.ndarray:
        arrs = [getattr(rd, attr)[rd.muon_is_ge77]
                for rd in light_run_list if rd.muon_is_ge77.size > 0]
        return np.concatenate(arrs) if arrs else np.array([], dtype=np.int64)

    metrics = [
        ("Photons\n(200 ns cut)",  "muon_photon_counts"),
        ("Photons\n(no cut)",      "muon_photon_counts_nocut"),
        ("Distinct PMTs\n(200 ns cut)", "muon_pmt_counts"),
        ("Distinct PMTs\n(no cut)",     "muon_pmt_counts_nocut"),
    ]

    n_total = _ge77("muon_photon_counts").size
    labels, n_with_list, n_zero_list = [], [], []
    for label, attr in metrics:
        vals = _ge77(attr)
        n_zero = int((vals == 0).sum())
        labels.append(label)
        n_with_list.append(vals.size - n_zero)
        n_zero_list.append(n_zero)

    x = np.arange(len(metrics))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars_with = ax.bar(x - width / 2, n_with_list, width,
                       color=COLORS["green"], edgecolor="black", linewidth=0.8,
                       label="≥1 detected")
    bars_zero = ax.bar(x + width / 2, n_zero_list, width,
                       color=COLORS["red"], edgecolor="black", linewidth=0.8,
                       label="0 detected")

    for bars, vals in [(bars_with, n_with_list), (bars_zero, n_zero_list)]:
        for bar, val in zip(bars, vals):
            pct = val / n_total * 100 if n_total > 0 else 0.0
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:,}\n({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Number of Ge77 muons", fontsize=13)
    ax.set_title(
        f"Ge77 muons with zero detected signal  (N = {n_total:,} Ge77 muons)",
        fontsize=14,
    )
    ax.margins(y=0.15)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    _save(fig, out_dir / "ge77_muon_zero_photons.png")


# ---------------------------------------------------------------------------
# Wavelength of all NC photons
# ---------------------------------------------------------------------------
def plot_wavelength_all_ncs(
    all_wl_hist: np.ndarray,
    out_dir: Path,
) -> None:
    """Histogram of the wavelengths of every NC-matched photon (all NCs).

    Includes all photons that pass the PMT mask and have a matching NC in sim 1
    (before the 200 ns time cut and the muon time window).  A photon that passes
    the PMT mask without a matching NC already raises a RuntimeError during
    loading, so this histogram covers every NC photon.
    """
    if all_wl_hist.sum() == 0:
        return

    wl_centers = 0.5 * (WL_BINS[:-1] + WL_BINS[1:])
    bin_width  = WL_BINS[1] - WL_BINS[0]
    n_all      = int(all_wl_hist.sum())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(wl_centers, all_wl_hist, width=bin_width, color=COLORS["blue"],
           alpha=0.80, linewidth=0)
    ax.set_xlabel("Wavelength  [nm]", fontsize=13)
    ax.set_ylabel("Photon count", fontsize=13)
    ax.set_title(
        f"Wavelength of all NC photons  (NC-matched, N = {n_all:,})",
        fontsize=14,
    )
    ax.tick_params(labelsize=11)
    _save(fig, out_dir / "wavelength_all_ncs.png")


# ---------------------------------------------------------------------------
# Ge77 light plots
# ---------------------------------------------------------------------------
def plot_ge77_light(
    light_run_list: list[LightRunData],
    ge77_times_hist: np.ndarray,
    out_dir: Path,
) -> None:
    """Ge77-specific light analysis — 4 plots."""
    parts_counts = [rd.nc_photon_counts[rd.nc_is_ge77]
                    for rd in light_run_list
                    if rd.nc_photon_counts.size > 0 and rd.nc_is_ge77.any()]
    if not parts_counts:
        print("  No Ge77 NCs found — skipping Ge77 light plots.")
        return

    ge77_counts = np.concatenate(parts_counts)
    ge77_x = np.concatenate([rd.nc_x_m[rd.nc_is_ge77] for rd in light_run_list
                              if rd.nc_is_ge77.any()])
    ge77_y = np.concatenate([rd.nc_y_m[rd.nc_is_ge77] for rd in light_run_list
                              if rd.nc_is_ge77.any()])
    ge77_z = np.concatenate([rd.nc_z_m[rd.nc_is_ge77] for rd in light_run_list
                              if rd.nc_is_ge77.any()])

    n_ge77 = len(ge77_counts)
    n_with  = int((ge77_counts > 0).sum())
    n_zero  = n_ge77 - n_with

    # ---- 1: 3-D NC positions coloured by photon count ----------------------
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    clrs = ["gold" if c > 0 else "gray" for c in ge77_counts]
    ax.scatter(ge77_x * 1000, ge77_y * 1000, ge77_z * 1000,
               c=clrs, s=15, alpha=0.6, edgecolors="none")
    ax.set_xlabel("X [mm]", fontsize=11)
    ax.set_ylabel("Y [mm]", fontsize=11)
    ax.set_zlabel("Z [mm]", fontsize=11)
    ax.set_title(f"Ge77 NC positions  (N = {n_ge77:,})", fontsize=13)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gold",
               markersize=9, label=f"With photons ({n_with:,})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=9, label=f"No photons ({n_zero:,})"),
    ], fontsize=10, loc="upper right")
    _save(fig, out_dir / "ge77_nc_positions_3d.png")

    # ---- 2: Pie chart — with/without photons --------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    ax2.pie(
        [n_with, n_zero],
        labels=["With photons", "No photons"],
        autopct="%1.2f%%",
        colors=["gold", "lightgray"],
        explode=(0.05, 0),
        startangle=90,
        textprops={"fontsize": 12, "fontweight": "bold"},
        wedgeprops={"linewidth": 1.5, "edgecolor": "black"},
    )
    ax2.set_title(
        f"Ge77 NCs: photon detection  (N = {n_ge77:,})\n"
        "Note: 0 direct photons expected from Ge77 NC itself",
        fontsize=13,
    )
    _save(fig2, out_dir / "ge77_light_production.png")

    # ---- 3: Photon count histogram ------------------------------------------
    counts_pos = ge77_counts[ge77_counts > 0]
    if counts_pos.size > 0:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        bins = np.logspace(0, np.log10(max(float(counts_pos.max()), 10)), 40)
        ax3.hist(counts_pos, bins=bins, color="gold", alpha=0.85,
                 edgecolor="black", linewidth=1.0)
        ax3.axvline(float(np.mean(counts_pos)), color="red", linestyle="--",
                    linewidth=2, label=f"Mean: {np.mean(counts_pos):.1f}")
        ax3.axvline(float(np.median(counts_pos)), color="darkred",
                    linestyle=":", linewidth=2,
                    label=f"Median: {np.median(counts_pos):.0f}")
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.set_xlabel("Photon count per Ge77 NC", fontsize=13)
        ax3.set_ylabel("Number of Ge77 NCs", fontsize=13)
        ax3.set_title(
            f"Ge77 NC photon distribution  "
            f"(NCs with >0 photons, n = {counts_pos.size:,})",
            fontsize=13,
        )
        ax3.legend(fontsize=11)
        ax3.tick_params(labelsize=11)
        _save(fig3, out_dir / "ge77_photon_count_histogram.png")

    # ---- 4: Ge77 photon time distribution (before 200 ns cut) --------------
    if ge77_times_hist.sum() > 0:
        bin_centers = 0.5 * (TIME_HIST_BINS[:-1] + TIME_HIST_BINS[1:])
        bin_width   = TIME_HIST_BINS[1] - TIME_HIST_BINS[0]
        cut_idx      = int(np.searchsorted(TIME_HIST_BINS, TIME_FILTER_NS, side="left"))
        n_after_cut  = int(ge77_times_hist[cut_idx:].sum())
        n_total_hist = int(ge77_times_hist.sum())
        pct_removed  = n_after_cut / n_total_hist * 100 if n_total_hist > 0 else 0.0

        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.bar(bin_centers, ge77_times_hist, width=bin_width,
                color="gold", alpha=0.85, linewidth=0,
                label="Ge77 photons (all runs, all photons)")
        ax4.bar(bin_centers[cut_idx:], ge77_times_hist[cut_idx:],
                width=bin_width, color="red", alpha=0.5, linewidth=0,
                label=f"Removed (t > {TIME_FILTER_NS:.0f} ns)  ({pct_removed:.1f}%)")
        ax4.axvline(TIME_FILTER_NS, color="red", linestyle="--", linewidth=2)
        ax4.set_yscale("log")
        ax4.set_xlabel("Photon time relative to NC  [ns]", fontsize=13)
        ax4.set_ylabel("Photon count  (all runs, all photons)", fontsize=13)
        ax4.set_title(
            "Ge77 photon time distribution\n"
            "Note: 0 direct photons expected from Ge77 NC itself",
            fontsize=13,
        )
        ax4.legend(fontsize=11)
        ax4.tick_params(labelsize=11)
        _save(fig4, out_dir / "ge77_photon_time_distribution.png")

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def write_statistics(
    light_run_list: list[LightRunData],
    filter_counts: dict,
    global_pmt_uids: set,
    out_dir: Path,
) -> None:
    nc_all = np.concatenate(
        [rd.nc_photon_counts for rd in light_run_list
         if rd.nc_photon_counts.size > 0]
    ) if any(rd.nc_photon_counts.size > 0 for rd in light_run_list) else np.array([])
    mu_all = np.concatenate(
        [rd.muon_photon_counts for rd in light_run_list
         if rd.muon_photon_counts.size > 0]
    ) if any(rd.muon_photon_counts.size > 0 for rd in light_run_list) else np.array([])

    def _pct(num: int, den: int) -> str:
        return f"{num / den * 100:.2f}%" if den > 0 else "n/a"

    n_raw = filter_counts["n_raw"]
    lines = [
        "=== Light Analysis — Homogeneous PMT Simulation ===",
        "",
        f"Runs loaded:              {len(light_run_list)}",
        f"Unique PMT detector UIDs: {len(global_pmt_uids)}",
        "",
        "--- Filter pipeline ---",
        f"Total photons (raw):      {n_raw:,}",
        f"After PMT UID mask:       {filter_counts['n_after_pmt']:,}"
        f"  ({_pct(filter_counts['n_after_pmt'], n_raw)})",
        f"After NC match filter:    {filter_counts['n_after_nc_match']:,}"
        f"  ({_pct(filter_counts['n_after_nc_match'], n_raw)})",
        f"After <= 200 ns NC-rel.:  {filter_counts['n_after_time']:,}"
        f"  ({_pct(filter_counts['n_after_time'], n_raw)})",
        f"After NC time window:     {filter_counts.get('n_after_window', 0):,}"
        f"  ({_pct(filter_counts.get('n_after_window', 0), n_raw)})",
        "",
        "--- Per-NC photon statistics ---",
        f"Total NCs (incl. 0):      {nc_all.size:,}",
    ]
    if nc_all.size > 0:
        n_pos = int((nc_all > 0).sum())
        lines.append(
            f"NCs with >0 photons:      {n_pos:,}  ({_pct(n_pos, nc_all.size)})")
        pos = nc_all[nc_all > 0]
        if pos.size > 0:
            lines += [
                f"Mean photons (>0):        {pos.mean():.1f}",
                f"Median photons (>0):      {int(np.median(pos))}",
                f"Max photons:              {pos.max():,}",
            ]
    lines += [
        "",
        "--- Per-muon photon statistics ---",
        f"Total muons (incl. 0):    {mu_all.size:,}",
    ]
    if mu_all.size > 0:
        n_pos_mu = int((mu_all > 0).sum())
        lines.append(
            f"Muons with >0 photons:    {n_pos_mu:,}  ({_pct(n_pos_mu, mu_all.size)})")
        pos_mu = mu_all[mu_all > 0]
        if pos_mu.size > 0:
            lines += [
                f"Mean photons (>0):        {pos_mu.mean():.1f}",
                f"Median photons (>0):      {int(np.median(pos_mu))}",
                f"Max photons:              {pos_mu.max():,}",
            ]
    out_path = out_dir / "statistics.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optical photon light analysis for homogeneous PMT simulation."
    )
    parser.add_argument("--data-path", default=DEFAULT_SIM2_PATH,
                        help="Sim 2 root directory (default: %(default)s)")
    parser.add_argument("--sim1-path", default=DEFAULT_SIM1_PATH,
                        help="Sim 1 (MUSUN NC) root directory (default: %(default)s)")
    parser.add_argument("--output-path", default=DEFAULT_SIM2_PATH,
                        help="Base output directory.  A 'light_analysis/' sub-directory "
                             "is created inside it.  Defaults to --data-path.")
    parser.add_argument("--runs", type=int, default=NUM_RUNS_DEFAULT,
                        help="Max number of runs to process (default: %(default)s)")
    args = parser.parse_args()

    sim2_path   = Path(args.data_path)
    sim1_path   = Path(args.sim1_path)
    output_base = Path(args.output_path) if args.output_path else sim2_path
    out_dir     = output_base / "light_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sim 2 path : {sim2_path}")
    print(f"Sim 1 path : {sim1_path}")
    print(f"Output     : {out_dir}")

    t_main = _log_resources("startup")

    run_dirs_2 = sorted(sim2_path.glob("run_*"))[:args.runs]
    if not run_dirs_2:
        print(f"ERROR: no run_* directories in {sim2_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(run_dirs_2)} sim 2 run(s).", flush=True)

    # Global accumulators
    global_pmt_uids: set = set()
    filter_counts = {
        "n_raw": 0, "n_after_pmt": 0,
        "n_after_time": 0, "n_after_nc_match": 0,
        "n_after_window": 0,
    }
    n_bins = len(TIME_HIST_BINS) - 1
    all_times_hist  = np.zeros(n_bins, dtype=np.int64)
    ge77_times_hist = np.zeros(n_bins, dtype=np.int64)
    n_wl_bins = len(WL_BINS) - 1
    all_wl_hist  = np.zeros(n_wl_bins, dtype=np.int64)

    # Load all runs
    light_run_list: list[LightRunData] = []
    for i, run_dir_2 in enumerate(run_dirs_2, 1):
        run_name  = run_dir_2.name
        run_dir_1 = sim1_path / run_name
        if not run_dir_1.is_dir():
            print(f"  WARNING: sim 1 run {run_name} not found — skipping.")
            continue
        print(f"\n[{i}/{len(run_dirs_2)}] {run_name} ...", flush=True)
        lrd = load_light_run(
            run_dir_2, run_dir_1,
            global_pmt_uids, filter_counts,
            all_times_hist, ge77_times_hist,
            all_wl_hist,
        )
        light_run_list.append(lrd)
        _log_resources(f"run {run_name} done", t_main)

    if not light_run_list:
        print("ERROR: no runs loaded.", file=sys.stderr)
        sys.exit(1)

    # PMT count check
    n_pmt_uids = len(global_pmt_uids)
    if n_pmt_uids != EXPECTED_PMT_COUNT:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_PMT_COUNT} unique PMT detector UIDs, "
            f"found {n_pmt_uids}. Check detector geometry or UID range "
            f"[{PMT_UID_MIN}, {PMT_UID_MAX})."
        )
    print(f"\nPMT UID check: {n_pmt_uids} unique UIDs — OK")

    print("\n--- Filter impact plots ---")
    plot_filter_impact_pmt(filter_counts, out_dir)
    plot_filter_impact_time(all_times_hist, filter_counts, out_dir)
    _log_resources("filter plots done", t_main)

    print("\n--- Light histograms ---")
    plot_light_per_nc(light_run_list, out_dir)
    _log_resources("light histograms done", t_main)

    print("\n--- Ge77 light plots ---")
    plot_ge77_light(light_run_list, ge77_times_hist, out_dir)
    _log_resources("Ge77 plots done", t_main)

    print("\n--- Comparison plots ---")
    plot_photon_count_comparison(light_run_list, out_dir)
    plot_pmt_multiplicity(light_run_list, out_dir)
    # All-muon cut vs no-cut comparison (200 ns cut on vs off; window kept)
    plot_cut_comparison_all_muons(
        light_run_list, out_dir,
        attr_cut="muon_photon_counts",
        attr_nocut="muon_photon_counts_nocut",
        xlabel="Photon count per muon (>0)",
        suptitle="Photon yield per muon (all muons): 200 ns cut vs no cut",
        out_name="photon_yield_all_muons_cut_vs_nocut.png",
        bin_mode="log",
    )
    plot_cut_comparison_all_muons(
        light_run_list, out_dir,
        attr_cut="muon_pmt_counts",
        attr_nocut="muon_pmt_counts_nocut",
        xlabel="Distinct PMTs per muon",
        suptitle="PMT multiplicity per muon (all muons): 200 ns cut vs no cut",
        out_name="pmt_multiplicity_all_muons_cut_vs_nocut.png",
        bin_mode="int",
    )
    # Same comparison, restricted to Ge77 muons (muon with >=1 Ge77-flagged NC)
    plot_cut_comparison_all_muons(
        light_run_list, out_dir,
        attr_cut="muon_pmt_counts",
        attr_nocut="muon_pmt_counts_nocut",
        xlabel="Distinct PMTs per muon",
        suptitle="PMT multiplicity per muon (Ge77 muons): 200 ns cut vs no cut",
        out_name="pmt_multiplicity_ge77_muons_cut_vs_nocut.png",
        bin_mode="int",
        muon_mask_attr="muon_is_ge77",
    )
    plot_ge77_muon_zero_photons(light_run_list, out_dir)
    plot_wavelength_all_ncs(all_wl_hist, out_dir)
    _log_resources("comparison plots done", t_main)

    write_statistics(
        light_run_list, filter_counts, global_pmt_uids, out_dir)

    _log_resources("TOTAL", t_main)
    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
