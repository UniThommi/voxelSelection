#!/usr/bin/env python3
"""
PMT Coverage Comparison: Homogeneous vs. Optimized PMT Setup
=============================================================
Compares NC detection and Ge-77 muon classification between two PMT
configurations using Geant4/REMAGE simulation data.

Usage:
    python compare_pmt_coverage.py --optimized-path /path/to/optimized \
        [--muon-path ...] [--homogeneous-path ...] [--W 6] [--M 6] [--m 1]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
TIME_CUT_NC_NS = 200.0  # photon must arrive within 200 ns of NC
FLOAT_TOL_NS = -1.0  # tolerance for negative NC times (float rounding)
MUON_WINDOW_LO_NS = 1_000.0  # 1 µs
MUON_WINDOW_HI_NS = 200_000.0  # 200 µs
MAX_DET_UIDS = 300
NUM_RUNS = 10

DEFAULT_MUON_PATH = (
    "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/rawMusunNCs"
)
DEFAULT_HOMOGENEOUS_PATH = (
    "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/"
    "rawOpticalHomogeneousPMTsFromMusunNCs"
)


# ──────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────
@dataclass
class RunData:
    """Holds loaded data for a single run."""

    run_id: int
    nc_df: pd.DataFrame  # from Sim 1: MyNeutronCaptureOutput
    optical_hom: pd.DataFrame  # from Sim 2 homogeneous
    optical_opt: pd.DataFrame  # from Sim 2 optimized
    n_vertices_hom: int  # vertex count in Sim 2 homogeneous
    n_vertices_opt: int  # vertex count in Sim 2 optimized


@dataclass
class DetectionResult:
    """Aggregated detection results across all runs for one setup."""

    label: str
    # Per-NC info
    nc_total: int = 0
    nc_any_photon: int = 0  # ≥1 photon (no time cut)
    nc_photon_within_200ns: int = 0  # ≥1 photon within 200 ns
    nc_photon_only_outside_200ns: int = 0  # photons exist but all >200 ns
    nc_detected: int = 0  # ≥M PMTs with ≥m hits within 200 ns
    # Ge77 muon NC subset
    ge77_nc_total: int = 0
    ge77_nc_any_photon: int = 0
    ge77_nc_photon_within_200ns: int = 0
    ge77_nc_photon_only_outside_200ns: int = 0
    ge77_nc_detected: int = 0
    # Multiplicity histogram: PMT count per NC
    multiplicity_counts: list[int] = field(default_factory=list)
    # Muon classification
    n_muons_with_nc: int = 0
    n_ge77_muons: int = 0
    n_non_ge77_muons: int = 0
    tp: int = 0  # true positive: Ge77 muon classified as Ge77
    fp: int = 0  # false positive: non-Ge77 muon classified as Ge77
    tn: int = 0  # true negative: non-Ge77 muon not classified as Ge77
    fn: int = 0  # false negative: Ge77 muon not classified as Ge77
    # W histogram: detected NCs per muon in [1µs, 200µs]
    w_hist_ge77: list[int] = field(default_factory=list)
    w_hist_non_ge77: list[int] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# HDF5 I/O
# ──────────────────────────────────────────────────────────────────────
def _read_pages(group: h5py.Group, field_name: str) -> np.ndarray:
    """Read a field stored in LGDO column format (pages array)."""
    return group[field_name]["pages"][:]


def load_sim1_nc(run_dir: str) -> pd.DataFrame:
    """Load MyNeutronCaptureOutput from all output_t*.hdf5 in a run dir."""
    files = sorted(glob.glob(os.path.join(run_dir, "output_t*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No output_t*.hdf5 in {run_dir}")

    frames = []
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            grp = f["hit"]["MyNeutronCaptureOutput"]
            n_entries = int(grp["entries"][()])
            if n_entries == 0:
                continue
            df = pd.DataFrame(
                {
                    "muon_id": _read_pages(grp, "evtid"),
                    "nc_id": _read_pages(grp, "nC_track_id"),
                    "nc_time_ns": _read_pages(grp, "nC_time_in_ns"),
                    "flag_ge77": _read_pages(grp, "nC_flag_Ge77"),
                    "nc_x": _read_pages(grp, "nC_x_position_in_m"),
                    "nc_y": _read_pages(grp, "nC_y_position_in_m"),
                    "nc_z": _read_pages(grp, "nC_z_position_in_m"),
                }
            )
            frames.append(df)

    if not frames:
        raise ValueError(f"No NC entries found in {run_dir}")
    return pd.concat(frames, ignore_index=True)


def load_sim2_optical(run_dir: str) -> pd.DataFrame:
    """Load optical hit data from all output_t*.hdf5 in a run dir."""
    files = sorted(glob.glob(os.path.join(run_dir, "output_t*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No output_t*.hdf5 in {run_dir}")

    frames = []
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            grp = f["hit"]["optical"]
            n_entries = int(grp["entries"][()])
            if n_entries == 0:
                continue
            df = pd.DataFrame(
                {
                    "muon_track_id": _read_pages(grp, "muon_track_id"),
                    "nC_track_id": _read_pages(grp, "nC_track_id"),
                    "det_uid": _read_pages(grp, "det_uid"),
                    "time_in_ns": _read_pages(grp, "time_in_ns"),
                }
            )
            frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=["muon_track_id", "nC_track_id", "det_uid", "time_in_ns"]
        )
    return pd.concat(frames, ignore_index=True)


def count_sim2_vertices(run_dir: str) -> int:
    """Count total vertex entries across all output_t*.hdf5 in a run dir."""
    files = sorted(glob.glob(os.path.join(run_dir, "output_t*.hdf5")))
    total = 0
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            total += int(f["hit"]["vertices"]["entries"][()])
    return total


# ──────────────────────────────────────────────────────────────────────
# Data loading & validation
# ──────────────────────────────────────────────────────────────────────
def load_all_runs(
    muon_path: str, hom_path: str, opt_path: str
) -> list[RunData]:
    """Load and validate data for all available runs."""
    runs: list[RunData] = []
    skipped: list[int] = []

    for i in range(1, NUM_RUNS + 1):
        run_label = f"run_{i:03d}"
        muon_dir = os.path.join(muon_path, run_label)
        hom_dir = os.path.join(hom_path, run_label)
        opt_dir = os.path.join(opt_path, run_label)

        try:
            nc_df = load_sim1_nc(muon_dir)
            optical_hom = load_sim2_optical(hom_dir)
            optical_opt = load_sim2_optical(opt_dir)
            n_vert_hom = count_sim2_vertices(hom_dir)
            n_vert_opt = count_sim2_vertices(opt_dir)
        except Exception as e:
            print(f"[SKIP] {run_label}: {e}")
            skipped.append(i)
            continue

        runs.append(
            RunData(
                run_id=i,
                nc_df=nc_df,
                optical_hom=optical_hom,
                optical_opt=optical_opt,
                n_vertices_hom=n_vert_hom,
                n_vertices_opt=n_vert_opt,
            )
        )

    if skipped:
        print(f"\n[WARNING] Skipped runs: {skipped}")
    if not runs:
        raise RuntimeError("No valid runs loaded.")

    print(f"\nLoaded {len(runs)} runs successfully.")
    return runs


def validate_runs(runs: list[RunData]) -> None:
    """Run validation checks on loaded data."""
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # 1. Vertex count match per run
    for rd in runs:
        if rd.n_vertices_hom != rd.n_vertices_opt:
            raise RuntimeError(
                f"Run {rd.run_id:03d}: vertex count mismatch — "
                f"homogeneous={rd.n_vertices_hom}, "
                f"optimized={rd.n_vertices_opt}"
            )
    print("[PASS] Vertex counts match across setups for all runs.")

    # 2. NC time validation
    for rd in runs:
        min_time = rd.nc_df["nc_time_ns"].min()
        if min_time < FLOAT_TOL_NS:
            raise RuntimeError(
                f"Run {rd.run_id:03d}: NC time {min_time:.3f} ns < "
                f"{FLOAT_TOL_NS} ns tolerance."
            )
        if min_time < 0:
            print(
                f"[WARN] Run {rd.run_id:03d}: min NC time = "
                f"{min_time:.3f} ns (within tolerance)."
            )
    print("[PASS] No NC times below tolerance threshold.")

    # 3. Unique det_uid count per setup
    for label, accessor in [
        ("homogeneous", lambda rd: rd.optical_hom),
        ("optimized", lambda rd: rd.optical_opt),
    ]:
        all_uids = set()
        for rd in runs:
            df = accessor(rd)
            if len(df) > 0:
                all_uids.update(df["det_uid"].unique())
        n_uids = len(all_uids)
        if n_uids > MAX_DET_UIDS:
            raise RuntimeError(
                f"{label}: {n_uids} unique det_uids > {MAX_DET_UIDS}. "
                f"Possible data mismatch."
            )
        if n_uids < MAX_DET_UIDS:
            print(
                f"[WARN] {label}: only {n_uids} unique det_uids "
                f"(expected ~{MAX_DET_UIDS}). Some PMTs may have "
                f"detected nothing."
            )
        else:
            print(f"[PASS] {label}: {n_uids} unique det_uids.")

    # 4. All Sim 1 NCs should have corresponding events in Sim 2
    for rd in runs:
        sim1_keys = set(zip(rd.nc_df["muon_id"], rd.nc_df["nc_id"]))
        for label, opt_df in [
            ("homogeneous", rd.optical_hom),
            ("optimized", rd.optical_opt),
        ]:
            if len(opt_df) == 0:
                continue
            sim2_keys = set(
                zip(opt_df["muon_track_id"], opt_df["nC_track_id"])
            )
            unmatched = sim2_keys - sim1_keys
            if unmatched:
                print(
                    f"[WARN] Run {rd.run_id:03d} {label}: "
                    f"{len(unmatched)} optical (muon,nc) keys not in Sim 1."
                )

    # 5. No negative photon times
    for rd in runs:
        for label, opt_df in [
            ("homogeneous", rd.optical_hom),
            ("optimized", rd.optical_opt),
        ]:
            if len(opt_df) == 0:
                continue
            min_t = opt_df["time_in_ns"].min()
            if min_t < FLOAT_TOL_NS:
                raise RuntimeError(
                    f"Run {rd.run_id:03d} {label}: photon time "
                    f"{min_t:.3f} ns < {FLOAT_TOL_NS} ns."
                )

    print("[PASS] No invalid photon times.")

    # 6. Check for duplicate NCs in Sim 1
    for rd in runs:
        dupes = rd.nc_df.duplicated(subset=["muon_id", "nc_id"], keep=False)
        n_dupes = dupes.sum()
        if n_dupes > 0:
            print(
                f"[WARN] Run {rd.run_id:03d}: {n_dupes} duplicate "
                f"(muon_id, nc_id) rows in Sim 1 NC data."
            )

    print("\nValidation complete.\n")


# ──────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────
def detect_ncs(
    optical_df: pd.DataFrame,
    nc_df: pd.DataFrame,
    M_threshold: int,
    m_threshold: int,
) -> tuple[set[tuple[int, int]], pd.DataFrame]:
    """
    Determine which NCs are detected.

    Returns:
        - Set of (muon_id, nc_id) tuples that pass detection.
        - DataFrame with per-NC multiplicity (number of firing PMTs).
    """
    if len(optical_df) == 0:
        return set(), pd.DataFrame(
            columns=["muon_track_id", "nC_track_id", "n_firing_pmts"]
        )

    # Build NC time lookup
    nc_time_map = dict(
        zip(
            zip(nc_df["muon_id"], nc_df["nc_id"]),
            nc_df["nc_time_ns"],
        )
    )

    # Map NC times onto optical DataFrame
    nc_keys = list(
        zip(optical_df["muon_track_id"], optical_df["nC_track_id"])
    )
    nc_times = np.array([nc_time_map.get(k, np.nan) for k in nc_keys])

    # Time since NC
    dt = optical_df["time_in_ns"].values - nc_times

    # Filter: within 200 ns of NC (with float tolerance)
    valid_mask = (~np.isnan(nc_times)) & (dt >= FLOAT_TOL_NS) & (
        dt <= TIME_CUT_NC_NS
    )

    filtered = optical_df.loc[valid_mask].copy()

    # Count hits per (NC, det_uid)
    hits_per_pmt = (
        filtered.groupby(["muon_track_id", "nC_track_id", "det_uid"])
        .size()
        .reset_index(name="n_hits")
    )

    # Filter PMTs with >= m hits
    firing_pmts = hits_per_pmt.loc[hits_per_pmt["n_hits"] >= m_threshold]

    # Count firing PMTs per NC
    multiplicity = (
        firing_pmts.groupby(["muon_track_id", "nC_track_id"])["det_uid"]
        .nunique()
        .reset_index(name="n_firing_pmts")
    )

    # Detected NCs: multiplicity >= M
    detected = multiplicity.loc[multiplicity["n_firing_pmts"] >= M_threshold]
    detected_set = set(
        zip(detected["muon_track_id"], detected["nC_track_id"])
    )

    return detected_set, multiplicity


def compute_detectability(
    optical_df: pd.DataFrame,
    nc_df: pd.DataFrame,
    all_nc_keys: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """
    Compute detectability categories for NCs.

    Returns:
        - any_photon: NCs with ≥1 photon on any PMT (no time cut)
        - within_200ns: NCs with ≥1 photon within 200 ns
        - only_outside_200ns: NCs with photons but all >200 ns after NC
    """
    if len(optical_df) == 0:
        return set(), set(), set()

    nc_time_map = dict(
        zip(
            zip(nc_df["muon_id"], nc_df["nc_id"]),
            nc_df["nc_time_ns"],
        )
    )

    nc_keys = list(
        zip(optical_df["muon_track_id"], optical_df["nC_track_id"])
    )
    nc_times = np.array([nc_time_map.get(k, np.nan) for k in nc_keys])

    matched_mask = ~np.isnan(nc_times)

    # All NCs with ≥1 photon (no time cut)
    any_photon = set(
        zip(
            optical_df.loc[matched_mask, "muon_track_id"],
            optical_df.loc[matched_mask, "nC_track_id"],
        )
    )

    # NCs with ≥1 photon within 200 ns
    dt = optical_df["time_in_ns"].values - nc_times
    within_mask = matched_mask & (dt >= FLOAT_TOL_NS) & (
        dt <= TIME_CUT_NC_NS
    )
    within_200ns = set(
        zip(
            optical_df.loc[within_mask, "muon_track_id"],
            optical_df.loc[within_mask, "nC_track_id"],
        )
    )

    only_outside = any_photon - within_200ns

    return any_photon, within_200ns, only_outside


def analyze_setup(
    runs: list[RunData],
    setup: str,
    M_threshold: int,
    m_threshold: int,
    W_threshold: int,
) -> DetectionResult:
    """Run full analysis for one PMT setup."""
    label = "Homogeneous" if setup == "hom" else "Optimized"
    result = DetectionResult(label=label)

    all_multiplicities: list[int] = []
    all_w_ge77: list[int] = []
    all_w_non_ge77: list[int] = []

    for rd in runs:
        nc_df = rd.nc_df
        optical_df = rd.optical_hom if setup == "hom" else rd.optical_opt

        # All NC keys for this run
        all_nc_keys = set(zip(nc_df["muon_id"], nc_df["nc_id"]))
        result.nc_total += len(all_nc_keys)

        # Detectability
        any_photon, within_200ns, only_outside = compute_detectability(
            optical_df, nc_df, all_nc_keys
        )
        result.nc_any_photon += len(any_photon)
        result.nc_photon_within_200ns += len(within_200ns)
        result.nc_photon_only_outside_200ns += len(only_outside)

        # Detection
        detected_set, multiplicity_df = detect_ncs(
            optical_df, nc_df, M_threshold, m_threshold
        )
        result.nc_detected += len(detected_set)

        # Collect multiplicities (including 0 for undetected NCs)
        mult_map = dict(
            zip(
                zip(
                    multiplicity_df["muon_track_id"],
                    multiplicity_df["nC_track_id"],
                ),
                multiplicity_df["n_firing_pmts"],
            )
        )
        for key in all_nc_keys:
            all_multiplicities.append(mult_map.get(key, 0))

        # Ge77 muon identification (ground truth from Sim 1)
        ge77_muons = set(
            nc_df.loc[nc_df["flag_ge77"] == 1, "muon_id"].unique()
        )
        non_ge77_muons = (
            set(nc_df["muon_id"].unique()) - ge77_muons
        )
        all_muons = set(nc_df["muon_id"].unique())

        result.n_muons_with_nc += len(all_muons)
        result.n_ge77_muons += len(ge77_muons)
        result.n_non_ge77_muons += len(non_ge77_muons)

        # Ge77 NC subset
        ge77_nc_mask = nc_df["muon_id"].isin(ge77_muons)
        ge77_nc_keys = set(
            zip(
                nc_df.loc[ge77_nc_mask, "muon_id"],
                nc_df.loc[ge77_nc_mask, "nc_id"],
            )
        )
        result.ge77_nc_total += len(ge77_nc_keys)
        result.ge77_nc_any_photon += len(any_photon & ge77_nc_keys)
        result.ge77_nc_photon_within_200ns += len(
            within_200ns & ge77_nc_keys
        )
        result.ge77_nc_photon_only_outside_200ns += len(
            only_outside & ge77_nc_keys
        )
        result.ge77_nc_detected += len(detected_set & ge77_nc_keys)

        # Muon classification: count detected NCs per muon in [1µs, 200µs]
        # First, filter NCs to the muon time window
        nc_in_window = nc_df.loc[
            (nc_df["nc_time_ns"] >= MUON_WINDOW_LO_NS)
            & (nc_df["nc_time_ns"] <= MUON_WINDOW_HI_NS)
        ]
        nc_in_window_keys = set(
            zip(nc_in_window["muon_id"], nc_in_window["nc_id"])
        )

        # Detected NCs that are also in the time window
        detected_in_window = detected_set & nc_in_window_keys

        # Count per muon
        muon_detected_counts: dict[int, int] = {}
        for muon_id, _ in detected_in_window:
            muon_detected_counts[muon_id] = (
                muon_detected_counts.get(muon_id, 0) + 1
            )

        # Classify each muon
        for muon_id in all_muons:
            w_count = muon_detected_counts.get(muon_id, 0)
            is_ge77_truth = muon_id in ge77_muons
            is_classified_ge77 = w_count >= W_threshold

            if is_ge77_truth:
                all_w_ge77.append(w_count)
                if is_classified_ge77:
                    result.tp += 1
                else:
                    result.fn += 1
            else:
                all_w_non_ge77.append(w_count)
                if is_classified_ge77:
                    result.fp += 1
                else:
                    result.tn += 1

    result.multiplicity_counts = all_multiplicities
    result.w_hist_ge77 = all_w_ge77
    result.w_hist_non_ge77 = all_w_non_ge77

    return result


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────
COLOR_HOM = "#4C72B0"
COLOR_OPT = "#DD8452"
FIGSIZE = (14, 7)


def _annotate_bar(ax: plt.Axes, bar, value: int, total: Optional[int] = None) -> None:
    """Annotate a bar with count and optionally percentage."""
    text = f"{value:,}"
    if total and total > 0:
        text += f"\n({100 * value / total:.1f}%)"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        text,
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def plot_nc_detection_overview(
    hom: DetectionResult,
    opt: DetectionResult,
    output_dir: str,
    M: int,
    m: int,
) -> None:
    """Plot 1: NC Detection Overview — grouped bars."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    categories = [
        "Total NCs",
        "Detectable\n(≥1 photon)",
        "Detectable\n(within 200ns)",
        "Detectable\n(only >200ns)",
        f"Detected\n(M≥{M}, m≥{m})",
    ]
    hom_vals = [
        hom.nc_total,
        hom.nc_any_photon,
        hom.nc_photon_within_200ns,
        hom.nc_photon_only_outside_200ns,
        hom.nc_detected,
    ]
    opt_vals = [
        opt.nc_total,
        opt.nc_any_photon,
        opt.nc_photon_within_200ns,
        opt.nc_photon_only_outside_200ns,
        opt.nc_detected,
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars_h = ax.bar(x - width / 2, hom_vals, width, label=hom.label, color=COLOR_HOM)
    bars_o = ax.bar(x + width / 2, opt_vals, width, label=opt.label, color=COLOR_OPT)

    for bar, val in zip(bars_h, hom_vals):
        _annotate_bar(ax, bar, val, hom.nc_total)
    for bar, val in zip(bars_o, opt_vals):
        _annotate_bar(ax, bar, val, opt.nc_total)

    ax.set_ylabel("Number of NCs")
    ax.set_title(
        "NC Detection Overview\n"
        "(only muons with ≥1 NC; total NCs identical for both setups)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Add summary text
    delta_detected = opt.nc_detected - hom.nc_detected
    sign = "+" if delta_detected >= 0 else ""
    summary = (
        f"Δ detected NCs (opt−hom): {sign}{delta_detected:,}  "
        f"({sign}{100 * delta_detected / max(hom.nc_detected, 1):.1f}%)"
    )
    ax.text(
        0.5,
        -0.12,
        summary,
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        fontstyle="italic",
    )

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_nc_detection_overview.png"), dpi=150)
    plt.close(fig)
    print("  Saved 01_nc_detection_overview.png")


def plot_multiplicity_histogram(
    hom: DetectionResult,
    opt: DetectionResult,
    output_dir: str,
    M: int,
) -> None:
    """Plot 2: PMT multiplicity histogram per NC."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    max_mult = max(
        max(hom.multiplicity_counts) if hom.multiplicity_counts else 0,
        max(opt.multiplicity_counts) if opt.multiplicity_counts else 0,
    )
    bins = np.arange(0, max_mult + 2) - 0.5

    ax.hist(
        hom.multiplicity_counts,
        bins=bins,
        alpha=0.6,
        label=hom.label,
        color=COLOR_HOM,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.hist(
        opt.multiplicity_counts,
        bins=bins,
        alpha=0.6,
        label=opt.label,
        color=COLOR_OPT,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.axvline(M - 0.5, color="red", linestyle="--", linewidth=1.5, label=f"M threshold = {M}")
    ax.set_xlabel("PMT Multiplicity (# firing PMTs per NC)")
    ax.set_ylabel("Number of NCs")
    ax.set_title(
        "PMT Multiplicity Distribution per NC\n"
        "(# distinct PMTs with ≥m hits within 200 ns of NC)"
    )
    ax.legend()
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Statistics
    hom_mean = np.mean(hom.multiplicity_counts) if hom.multiplicity_counts else 0
    opt_mean = np.mean(opt.multiplicity_counts) if opt.multiplicity_counts else 0
    stats_text = (
        f"Mean multiplicity — {hom.label}: {hom_mean:.2f}, "
        f"{opt.label}: {opt_mean:.2f}"
    )
    ax.text(
        0.5,
        -0.1,
        stats_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        fontstyle="italic",
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "02_nc_multiplicity_histogram.png"), dpi=150
    )
    plt.close(fig)
    print("  Saved 02_nc_multiplicity_histogram.png")


def plot_ge77_nc_detection(
    hom: DetectionResult,
    opt: DetectionResult,
    output_dir: str,
    M: int,
    m: int,
) -> None:
    """Plot 3: Ge77-muon NC detection overview."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    categories = [
        "Total Ge77\nmuon NCs",
        "Detectable\n(≥1 photon)",
        "Detectable\n(within 200ns)",
        "Detectable\n(only >200ns)",
        f"Detected\n(M≥{M}, m≥{m})",
    ]
    hom_vals = [
        hom.ge77_nc_total,
        hom.ge77_nc_any_photon,
        hom.ge77_nc_photon_within_200ns,
        hom.ge77_nc_photon_only_outside_200ns,
        hom.ge77_nc_detected,
    ]
    opt_vals = [
        opt.ge77_nc_total,
        opt.ge77_nc_any_photon,
        opt.ge77_nc_photon_within_200ns,
        opt.ge77_nc_photon_only_outside_200ns,
        opt.ge77_nc_detected,
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars_h = ax.bar(x - width / 2, hom_vals, width, label=hom.label, color=COLOR_HOM)
    bars_o = ax.bar(x + width / 2, opt_vals, width, label=opt.label, color=COLOR_OPT)

    for bar, val in zip(bars_h, hom_vals):
        _annotate_bar(ax, bar, val, hom.ge77_nc_total)
    for bar, val in zip(bars_o, opt_vals):
        _annotate_bar(ax, bar, val, opt.ge77_nc_total)

    ax.set_ylabel("Number of NCs")
    ax.set_title(
        "Ge-77 Muon NC Detection Overview\n"
        "(NCs belonging to muons with ≥1 Ge-77 producing NC; "
        "only muons with ≥1 NC)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    delta = opt.ge77_nc_detected - hom.ge77_nc_detected
    sign = "+" if delta >= 0 else ""
    summary = (
        f"Δ detected Ge77 NCs (opt−hom): {sign}{delta:,}  "
        f"({sign}{100 * delta / max(hom.ge77_nc_detected, 1):.1f}%)"
    )
    ax.text(
        0.5,
        -0.12,
        summary,
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        fontstyle="italic",
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "03_ge77_nc_detection_overview.png"), dpi=150
    )
    plt.close(fig)
    print("  Saved 03_ge77_nc_detection_overview.png")


def plot_confusion_matrix(
    hom: DetectionResult,
    opt: DetectionResult,
    output_dir: str,
    W: int,
    M: int,
) -> None:
    """Plot 4: Ge77 classification confusion matrix comparison."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    categories = [
        "True Positive\n(Ge77 → classified)",
        "False Negative\n(Ge77 → missed)",
        "True Negative\n(non-Ge77 → correct)",
        "False Positive\n(non-Ge77 → misclass.)",
    ]
    hom_vals = [hom.tp, hom.fn, hom.tn, hom.fp]
    opt_vals = [opt.tp, opt.fn, opt.tn, opt.fp]

    x = np.arange(len(categories))
    width = 0.35

    bars_h = ax.bar(x - width / 2, hom_vals, width, label=hom.label, color=COLOR_HOM)
    bars_o = ax.bar(x + width / 2, opt_vals, width, label=opt.label, color=COLOR_OPT)

    for bar, val in zip(bars_h, hom_vals):
        _annotate_bar(ax, bar, val)
    for bar, val in zip(bars_o, opt_vals):
        _annotate_bar(ax, bar, val)

    ax.set_ylabel("Number of Muons")
    ax.set_title(
        f"Ge-77 Muon Classification (W≥{W}, M≥{M})\n"
        f"(only muons with ≥1 NC)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Compute metrics
    def _metrics(tp, fp, tn, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return prec, rec, f1, fpr

    hp, hr, hf, hfpr = _metrics(hom.tp, hom.fp, hom.tn, hom.fn)
    op, orr, of, ofpr = _metrics(opt.tp, opt.fp, opt.tn, opt.fn)

    stats = (
        f"{hom.label}:  Precision={hp:.3f}  Recall={hr:.3f}  "
        f"F1={hf:.3f}  FPR={hfpr:.4f}\n"
        f"{opt.label}:  Precision={op:.3f}  Recall={orr:.3f}  "
        f"F1={of:.3f}  FPR={ofpr:.4f}"
    )
    ax.text(
        0.5,
        -0.15,
        stats,
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        fontstyle="italic",
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "04_ge77_confusion_matrix.png"), dpi=150
    )
    plt.close(fig)
    print("  Saved 04_ge77_confusion_matrix.png")


def plot_w_histogram(
    hom: DetectionResult,
    opt: DetectionResult,
    output_dir: str,
    W: int,
    M: int,
    ge77: bool,
) -> None:
    """Plot 5/6: W histogram (detected NCs per muon in [1µs, 200µs])."""
    suffix = "ge77" if ge77 else "non_ge77"
    title_label = "Ge-77" if ge77 else "Non-Ge-77"
    fname = f"05_w_histogram_{suffix}.png" if ge77 else f"06_w_histogram_{suffix}.png"

    hom_data = hom.w_hist_ge77 if ge77 else hom.w_hist_non_ge77
    opt_data = opt.w_hist_ge77 if ge77 else opt.w_hist_non_ge77

    fig, ax = plt.subplots(figsize=FIGSIZE)

    max_w = max(
        max(hom_data) if hom_data else 0,
        max(opt_data) if opt_data else 0,
    )
    bins = np.arange(0, max_w + 2) - 0.5

    ax.hist(
        hom_data,
        bins=bins,
        alpha=0.6,
        label=hom.label,
        color=COLOR_HOM,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.hist(
        opt_data,
        bins=bins,
        alpha=0.6,
        label=opt.label,
        color=COLOR_OPT,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.axvline(
        W - 0.5,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"W threshold = {W}",
    )

    ax.set_xlabel(f"Detected NCs per Muon in [1µs, 200µs] (M≥{M})")
    ax.set_ylabel("Number of Muons")
    ax.set_title(
        f"Detected NC Count per {title_label} Muon\n"
        f"(time window [1µs, 200µs], detection threshold M≥{M}; "
        f"only muons with ≥1 NC)"
    )
    ax.legend()
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Statistics
    hom_above = sum(1 for v in hom_data if v >= W)
    opt_above = sum(1 for v in opt_data if v >= W)
    n_hom = len(hom_data)
    n_opt = len(opt_data)
    stats = (
        f"{hom.label}: {hom_above}/{n_hom} muons above W={W} "
        f"({100 * hom_above / max(n_hom, 1):.1f}%)    "
        f"{opt.label}: {opt_above}/{n_opt} muons above W={W} "
        f"({100 * opt_above / max(n_opt, 1):.1f}%)"
    )
    ax.text(
        0.5,
        -0.1,
        stats,
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        fontstyle="italic",
    )

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PMT coverage: homogeneous vs. optimized setup."
    )
    parser.add_argument(
        "--optimized-path",
        required=True,
        help="Path to optimized PMT simulation data.",
    )
    parser.add_argument(
        "--muon-path",
        default=DEFAULT_MUON_PATH,
        help="Path to muon→NC simulation data.",
    )
    parser.add_argument(
        "--homogeneous-path",
        default=DEFAULT_HOMOGENEOUS_PATH,
        help="Path to homogeneous PMT simulation data.",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=6,
        help="Min detected NCs in [1µs,200µs] for Ge77 classification (default: 6).",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=6,
        help="Min PMT multiplicity per NC for detection (default: 6).",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=1,
        help="Min hits per PMT for it to count as firing (default: 1).",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PMT COVERAGE COMPARISON")
    print("=" * 60)
    print(f"  Muon path:       {args.muon_path}")
    print(f"  Homogeneous:     {args.homogeneous_path}")
    print(f"  Optimized:       {args.optimized_path}")
    print(f"  W={args.W}, M={args.M}, m={args.m}")
    print()

    # Load data
    runs = load_all_runs(args.muon_path, args.homogeneous_path, args.optimized_path)

    # Validate
    validate_runs(runs)

    # Analyze
    print("Analyzing homogeneous setup...")
    hom_result = analyze_setup(runs, "hom", args.M, args.m, args.W)
    print("Analyzing optimized setup...")
    opt_result = analyze_setup(runs, "opt", args.M, args.m, args.W)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in [hom_result, opt_result]:
        print(f"\n--- {r.label} ---")
        print(f"  Total NCs:            {r.nc_total:,}")
        print(f"  Detectable (any):     {r.nc_any_photon:,} ({100*r.nc_any_photon/max(r.nc_total,1):.1f}%)")
        print(f"  Detectable (200ns):   {r.nc_photon_within_200ns:,}")
        print(f"  Detectable (>200ns):  {r.nc_photon_only_outside_200ns:,}")
        print(f"  Detected:             {r.nc_detected:,} ({100*r.nc_detected/max(r.nc_total,1):.1f}%)")
        print(f"  Ge77 muons:           {r.n_ge77_muons:,} / {r.n_muons_with_nc:,}")
        print(f"  TP={r.tp}  FN={r.fn}  TN={r.tn}  FP={r.fp}")

    # Create output directory
    output_dir = os.path.join(args.optimized_path, "coverage_comparison")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nPlots will be saved to: {output_dir}")

    # Generate plots
    print("\nGenerating plots...")
    plot_nc_detection_overview(hom_result, opt_result, output_dir, args.M, args.m)
    plot_multiplicity_histogram(hom_result, opt_result, output_dir, args.M)
    plot_ge77_nc_detection(hom_result, opt_result, output_dir, args.M, args.m)
    plot_confusion_matrix(hom_result, opt_result, output_dir, args.W, args.M)
    plot_w_histogram(hom_result, opt_result, output_dir, args.W, args.M, ge77=True)
    plot_w_histogram(hom_result, opt_result, output_dir, args.W, args.M, ge77=False)

    print("\nDone.")


if __name__ == "__main__":
    main()