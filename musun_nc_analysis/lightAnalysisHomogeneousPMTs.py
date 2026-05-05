"""
Light analysis for homogeneous PMT simulation (sim 2).

Correlates optical photons from sim 2 (homogeneous PMT run) with NC events
from sim 1 (MUSUN NC run).  All filters follow the same logic as
zone_ratio_analysis.py.

Filter pipeline (applied in order):
  1. PMT UID mask:   det_uid in [10_000_000, 1_000_000_000)
  2. Time filter:    photon_time_relative >= 200 ns
  3. NC match:       (evtid, nC_track_id) present in sim 1 NC set

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
    read_nc_data_file,
    _w1_ks_sorted, _merge_sorted_runs,
    COLORS, N_PERMUTATIONS, RANDOM_SEED, W1_THRESHOLD_FRAC,
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
TIME_HIST_BINS = np.linspace(-50.0, 2000.0, 261)  # 260 bins, 10 ns each


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
    # One entry per NC-producing muon from sim 1
    muon_photon_counts: np.ndarray = field(
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
            evtid = _pages(grp, "evtid").astype(np.int64)
            if evtid.size == 0:
                return None
            return {
                "evtid":       evtid,
                "nc_track_id": _pages(grp, "nC_track_id").astype(np.int64),
                "det_uid":     _pages(grp, "det_uid").astype(np.int64),
                "time_ns":     _pages(grp, "time_in_ns").astype(np.float64),
            }
    except Exception as exc:
        print(f"  ERROR reading {fp.name}: {exc}")
        return None


def _load_nc_keys(sim1_run_dir: Path) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], tuple[float, float, float]],
    dict[int, list[tuple[int, int]]],
]:
    """
    Read all unique NC (evtid, track_id) keys from a sim 1 run directory.

    Returns:
      nc_ge77:  (evtid, tid) -> ge77 flag (0 or 1)
      nc_pos:   (evtid, tid) -> (x_m, y_m, z_m)
      muon_nc:  muon_evtid  -> list of NC keys belonging to that muon
    """
    nc_ge77: dict[tuple[int, int], int] = {}
    nc_pos: dict[tuple[int, int], tuple[float, float, float]] = {}
    muon_nc: dict[int, list] = defaultdict(list)

    for fp in sorted(sim1_run_dir.glob("output_t*.hdf5")):
        data = read_nc_data_file(fp)
        if data["evtid"].size == 0:
            continue
        for eid, tid, ge77, x, y, z in zip(
            data["evtid"].tolist(), data["track_id"].tolist(),
            data["ge77"].tolist(),
            data["x_m"].tolist(), data["y_m"].tolist(), data["z_m"].tolist(),
        ):
            key = (eid, tid)
            if key not in nc_ge77:
                nc_ge77[key] = ge77
                nc_pos[key] = (x, y, z)
                muon_nc[eid].append(key)
            elif ge77 > nc_ge77[key]:
                nc_ge77[key] = ge77

    return nc_ge77, nc_pos, dict(muon_nc)


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
) -> LightRunData:
    """
    Load one sim 2 run, apply all filters, return per-NC and per-muon counts.

    all_times_hist / ge77_times_hist are pre-allocated bin-count arrays
    (shape = len(TIME_HIST_BINS)-1) that are accumulated in-place using
    np.histogram so that every photon contributes — no sampling.

    Mutates global_pmt_uids, filter_counts, all_times_hist, ge77_times_hist.
    """
    rd = LightRunData(run_name=sim2_run_dir.name)

    nc_ge77, nc_pos, muon_nc = _load_nc_keys(sim1_run_dir)
    if not nc_ge77:
        print(f"  WARNING: no NC data in sim 1 run {sim1_run_dir.name}")
        return rd

    ge77_nc_set = frozenset(k for k, g in nc_ge77.items() if g == 1)
    valid_nc_set = frozenset(nc_ge77.keys())
    # Photon counter initialised to 0 for every sim 1 NC
    nc_counter: dict[tuple[int, int], int] = {k: 0 for k in nc_ge77}

    for fp in sorted(sim2_run_dir.glob("output_t*.hdf5")):
        data = _load_optical_file(fp)
        if data is None:
            continue

        evtid    = data["evtid"]
        nc_tid   = data["nc_track_id"]
        det_uid  = data["det_uid"]
        time_ns  = data["time_ns"]

        # Cross-check: relative times must not be < -1 ns
        bad = time_ns < -1.0
        if bad.any():
            raise RuntimeError(
                f"{bad.sum()} photon(s) with relative time < -1 ns in {fp} "
                f"(min = {time_ns[bad].min():.3f} ns). "
                "Expected photon times to be relative to NC time."
            )

        filter_counts["n_raw"] += len(evtid)

        # ---- Filter 1: PMT UID mask ----------------------------------------
        pmt_mask = (det_uid >= PMT_UID_MIN) & (det_uid < PMT_UID_MAX)
        global_pmt_uids.update(int(u) for u in det_uid[pmt_mask])

        evtid_p  = evtid[pmt_mask]
        nc_tid_p = nc_tid[pmt_mask]
        time_ns_p = time_ns[pmt_mask]
        filter_counts["n_after_pmt"] += len(evtid_p)

        # Accumulate full relative-time histogram (all PMT photons, before 200 ns cut)
        if time_ns_p.size > 0:
            h, _ = np.histogram(time_ns_p, bins=TIME_HIST_BINS)
            all_times_hist += h

        # Accumulate Ge77 time histogram (PMT photons from Ge77 NCs, before 200 ns cut)
        if time_ns_p.size > 0:
            ge77_mask_p = np.fromiter(
                ((int(e), int(t)) in ge77_nc_set
                 for e, t in zip(evtid_p.tolist(), nc_tid_p.tolist())),
                dtype=bool, count=len(evtid_p),
            )
            t_ge77 = time_ns_p[ge77_mask_p]
            if t_ge77.size > 0:
                h_ge77, _ = np.histogram(t_ge77, bins=TIME_HIST_BINS)
                ge77_times_hist += h_ge77

        # ---- Filter 2: time >= 200 ns ---------------------------------------
        time_mask = time_ns_p >= TIME_FILTER_NS
        evtid_t  = evtid_p[time_mask]
        nc_tid_t = nc_tid_p[time_mask]
        filter_counts["n_after_time"] += len(evtid_t)

        if evtid_t.size == 0:
            continue

        # ---- Filter 3: NC match (every photon must exist in sim 1) ----------
        match_mask = np.fromiter(
            ((int(e), int(t)) in valid_nc_set
             for e, t in zip(evtid_t.tolist(), nc_tid_t.tolist())),
            dtype=bool, count=len(evtid_t),
        )
        n_unmatched = int((~match_mask).sum())
        if n_unmatched > 0:
            raise RuntimeError(
                f"{n_unmatched} photon(s) in {fp} passed PMT mask and 200 ns cut "
                "but have no matching (evtid, nC_track_id) in sim 1. "
                "Sim 2 should only contain NC events that exist in sim 1."
            )
        filter_counts["n_after_nc_match"] += int(match_mask.sum())

        # Count photons per NC using Counter (batch, no per-photon loop)
        c = Counter(zip(evtid_t.tolist(), nc_tid_t.tolist()))
        for key, cnt in c.items():
            nc_counter[key] += cnt

    # Build output arrays — one entry per sim 1 NC (preserves 0-photon NCs)
    nc_keys_list = list(nc_ge77.keys())
    rd.nc_photon_counts = np.array(
        [nc_counter[k] for k in nc_keys_list], dtype=np.int64)
    rd.nc_is_ge77 = np.array(
        [nc_ge77[k] == 1 for k in nc_keys_list], dtype=bool)
    rd.nc_x_m = np.array([nc_pos[k][0] for k in nc_keys_list], dtype=np.float64)
    rd.nc_y_m = np.array([nc_pos[k][1] for k in nc_keys_list], dtype=np.float64)
    rd.nc_z_m = np.array([nc_pos[k][2] for k in nc_keys_list], dtype=np.float64)

    muon_counts = [
        sum(nc_counter.get(k, 0) for k in keys)
        for keys in muon_nc.values()
    ]
    rd.muon_photon_counts = np.array(muon_counts, dtype=np.int64)

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
    cut_idx = int(np.searchsorted(TIME_HIST_BINS, TIME_FILTER_NS, side="left")) - 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(bin_centers, all_times_hist,
           width=TIME_HIST_BINS[1] - TIME_HIST_BINS[0],
           color=COLORS["blue"], alpha=0.8, linewidth=0)
    # Shade removed region
    ax.bar(bin_centers[:cut_idx], all_times_hist[:cut_idx],
           width=TIME_HIST_BINS[1] - TIME_HIST_BINS[0],
           color="red", alpha=0.5, linewidth=0,
           label=f"Removed by 200 ns cut ({frac_removed:.1f}% of PMT photons)")
    ax.axvline(TIME_FILTER_NS, color="red", linestyle="--", linewidth=2,
               label=f"{TIME_FILTER_NS:.0f} ns cut")
    ax.set_xlabel("Relative photon time [ns]  (t_photon − t_NC)", fontsize=13)
    ax.set_ylabel("Photon count  (all runs, all photons)", fontsize=13)
    ax.set_title("Filter impact: 200 ns time window", fontsize=14)
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


def plot_light_per_muon(light_run_list: list[LightRunData], out_dir: Path) -> None:
    counts = np.concatenate(
        [rd.muon_photon_counts for rd in light_run_list
         if rd.muon_photon_counts.size > 0]
    )
    _photon_hist(counts, COLORS["purple"],
                 "Photon count per muon", "Detected photons per muon",
                 out_dir / "light_per_muon.png")


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
        cut_idx     = int(np.searchsorted(TIME_HIST_BINS, TIME_FILTER_NS, side="left")) - 1
        n_before_cut = int(ge77_times_hist[:cut_idx].sum())
        n_total_hist = int(ge77_times_hist.sum())
        pct_before   = n_before_cut / n_total_hist * 100 if n_total_hist > 0 else 0.0

        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.bar(bin_centers, ge77_times_hist, width=bin_width,
                color="gold", alpha=0.85, linewidth=0,
                label="Ge77 photons (all runs, all photons)")
        ax4.bar(bin_centers[:cut_idx], ge77_times_hist[:cut_idx],
                width=bin_width, color="red", alpha=0.5, linewidth=0,
                label=f"Removed by 200 ns cut ({pct_before:.1f}%)")
        ax4.axvline(TIME_FILTER_NS, color="red", linestyle="--", linewidth=2)
        ax4.set_yscale("log")
        ax4.set_xlabel("Relative photon time [ns]", fontsize=13)
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
# Convergence analysis
# ---------------------------------------------------------------------------
def convergence_analysis_light(
    light_run_list: list[LightRunData],
    out_dir: Path,
) -> dict[str, int]:
    """W1 + KS convergence vs number of runs for light-per-NC and per-muon."""
    n_runs = len(light_run_list)
    if n_runs < 2:
        print("  Convergence requires >= 2 runs; skipping.")
        return {}

    rng = random.Random(RANDOM_SEED)
    run_indices = list(range(n_runs))

    observables = [
        ("light_per_nc",   "log(photons/NC + 1)",   "Light per NC"),
        ("light_per_muon", "log(photons/muon + 1)", "Light per muon"),
    ]

    def get_raw(lrd: LightRunData, obs: str) -> np.ndarray:
        return lrd.nc_photon_counts if obs == "light_per_nc" else lrd.muon_photon_counts

    recommendations: dict[str, int] = {}
    t_conv = _log_resources("light convergence start")

    for obs_key, obs_xlabel, obs_title in observables:
        print(f"\n  Convergence: {obs_title} ...", flush=True)
        t_obs = _log_resources(f"{obs_key}: begin")

        # Sort each run's transformed data once
        sorted_runs: list[np.ndarray] = []
        for lrd in light_run_list:
            raw = np.log(get_raw(lrd, obs_key).astype(np.float64) + 1.0)
            sorted_runs.append(np.sort(raw))
        t_obs = _log_resources(f"{obs_key}: sorted {n_runs} runs", t_obs)

        sorted_ref = _merge_sorted_runs(sorted_runs, run_indices)
        if sorted_ref.size == 0:
            del sorted_runs, sorted_ref
            continue
        t_obs = _log_resources(
            f"{obs_key}: reference  N={sorted_ref.size:,}", t_obs)

        k_vals = list(range(1, n_runs + 1))

        det_w1, det_ks = [], []
        for k in k_vals:
            sorted_k = _merge_sorted_runs(sorted_runs, list(range(k)))
            w1, ks = _w1_ks_sorted(sorted_k, sorted_ref)
            del sorted_k
            det_w1.append(w1)
            det_ks.append(ks)

        rand_w1 = np.zeros((n_runs, N_PERMUTATIONS))
        rand_ks = np.zeros((n_runs, N_PERMUTATIONS))
        for perm_i in range(N_PERMUTATIONS):
            shuffled = run_indices.copy()
            rng.shuffle(shuffled)
            for k in k_vals:
                sorted_k = _merge_sorted_runs(sorted_runs, shuffled[:k])
                w1, ks = _w1_ks_sorted(sorted_k, sorted_ref)
                del sorted_k
                rand_w1[k - 1, perm_i] = w1
                rand_ks[k - 1, perm_i] = ks
        t_obs = _log_resources(f"{obs_key}: random subsets done", t_obs)

        del sorted_runs, sorted_ref
        gc.collect()
        _log_resources(f"{obs_key}: freed — total conv elapsed", t_conv)

        rand_w1_mean = rand_w1.mean(axis=1)
        rand_w1_std  = rand_w1.std(axis=1)
        rand_ks_mean = rand_ks.mean(axis=1)
        rand_ks_std  = rand_ks.std(axis=1)

        w1_at_k1  = det_w1[0]
        threshold = W1_THRESHOLD_FRAC * w1_at_k1 if w1_at_k1 > 0 else 0.0
        rec_k = next(
            (k for k, w in enumerate(det_w1, start=1) if w <= threshold),
            n_runs,
        )
        recommendations[obs_key] = rec_k
        print(f"    W1 threshold = {threshold:.4f}  → recommended k = {rec_k}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Convergence: {obs_title}", fontsize=14)
        for ax, metric, det_vals, r_mean, r_std, ylabel in [
            (axes[0], "W₁ distance", det_w1, rand_w1_mean, rand_w1_std,
             f"W₁ ({obs_xlabel})"),
            (axes[1], "KS statistic", det_ks, rand_ks_mean, rand_ks_std,
             "KS statistic D"),
        ]:
            ax.plot(k_vals, det_vals, "o-", color=COLORS["blue"],
                    linewidth=1.8, markersize=5, label="Deterministic (cumulative)")
            ax.plot(k_vals, r_mean, "s--", color=COLORS["orange"],
                    linewidth=1.5, markersize=5, label="Random subsets (mean)")
            ax.fill_between(k_vals, r_mean - r_std, r_mean + r_std,
                            color=COLORS["orange"], alpha=0.25,
                            label=f"±1σ  ({N_PERMUTATIONS} permutations)")
            ax.axhline(0, color="gray", linestyle=":", linewidth=1.0,
                       label=f"Reference (k={n_runs} vs k={n_runs})")
            if metric == "W₁ distance" and threshold > 0:
                ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
                           label=f"5% threshold = {threshold:.4f}")
                ax.axvline(rec_k, color="red", linestyle=":", linewidth=1.2,
                           label=f"Rec. k = {rec_k}")
            ax.set_xlabel("Number of runs k", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(metric, fontsize=13)
            ax.set_xticks(k_vals)
            ax.legend(fontsize=9)
            ax.tick_params(labelsize=10)
        _save(fig, out_dir / f"convergence_{obs_key}.png")

    return recommendations


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def write_statistics(
    light_run_list: list[LightRunData],
    filter_counts: dict,
    global_pmt_uids: set,
    recommendations: dict[str, int],
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
        f"After >= 200 ns cut:      {filter_counts['n_after_time']:,}"
        f"  ({_pct(filter_counts['n_after_time'], n_raw)})",
        f"After NC match filter:    {filter_counts['n_after_nc_match']:,}"
        f"  ({_pct(filter_counts['n_after_nc_match'], n_raw)})",
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
    lines += [
        "",
        "--- Convergence: recommended minimum runs ---",
        f"Light per NC:             {recommendations.get('light_per_nc', 'n/a')}",
        f"Light per muon:           {recommendations.get('light_per_muon', 'n/a')}",
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
                        help="Output directory (default: <data-path>/light_analysis)")
    parser.add_argument("--runs", type=int, default=NUM_RUNS_DEFAULT,
                        help="Max number of runs to process (default: %(default)s)")
    args = parser.parse_args()

    sim2_path = Path(args.data_path)
    sim1_path = Path(args.sim1_path)
    out_dir = (Path(args.output_path) if args.output_path
               else sim2_path / "light_analysis")
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
    }
    n_bins = len(TIME_HIST_BINS) - 1
    all_times_hist  = np.zeros(n_bins, dtype=np.int64)
    ge77_times_hist = np.zeros(n_bins, dtype=np.int64)

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
    plot_light_per_muon(light_run_list, out_dir)
    _log_resources("light histograms done", t_main)

    print("\n--- Ge77 light plots ---")
    plot_ge77_light(light_run_list, ge77_times_hist, out_dir)
    _log_resources("Ge77 plots done", t_main)

    print("\n--- Convergence analysis ---")
    recommendations = convergence_analysis_light(light_run_list, out_dir)
    _log_resources("convergence done", t_main)

    write_statistics(
        light_run_list, filter_counts, global_pmt_uids, recommendations, out_dir)

    _log_resources("TOTAL", t_main)
    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
