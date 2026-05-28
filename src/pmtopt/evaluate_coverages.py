#!/usr/bin/env python3
"""
PMT Configuration Evaluator
============================

Evaluates multiple PMT configurations (given as JSON voxel lists)
against a common SSD dataset. Produces:

  - NC coverage line plots (M=1..10, log + linear scale) per config
  - Ge77 muon heatmaps (Accuracy, Precision) for W x Config, per M
  - Confusion matrix text file for all (config, M, W) combinations
  - Summary text file with NC statistics

Usage:
    python -m pmtopt.evaluate_coverages \\
        --hdf5 data.hdf5 \\
        --baseline baseline.json \\
        --configs opt1.json opt2.json random1.json \\
        --labels "Baseline" "ncM1m1" "random" \\
        --output-dir eval_results/ \\
        --seed 42

Author: Thomas Buerger (University of Tübingen)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import sparse
from scipy import stats as scipy_stats

from pmtopt.geometry import (
    DEFAULT_AREA_RATIOS,
    MUON_TIME_WINDOW_MIN_NS,
    MUON_TIME_WINDOW_MAX_NS,
    calc_fom_confusion,
    calc_ge_survival_confusion,
    calc_deadtime_confusion,
    calc_veto_fraction,
    figure_of_merit,
    MUSUN_RATE,
    MUONS_PER_RUN_DIR,
    VETO_DURATION_H,
)
from pmtopt.data_loading import (
    load_raw_sparse,
    binarize_from_raw,
    load_muon_data,
    build_muon_index,
    count_hdf5_runs,
)
from pmtopt.homogeneous import (
    compute_wasserstein_homogeneity,
    get_w2_ref,
)


# ===================================================================
# Constants
# ===================================================================

M_VALUES = list(range(1, 11))   # M = 1..10
W_VALUES = list(range(1, 21))   # W = 1..20

# Scatter plot selection — always include M=4,5,6
_SCATTER_M_VALUES = [1, 2, 4, 5, 6, 8, 10]

# Name of the "all voxels" reference config — excluded from all statistical
# evaluations (regression, Spearman, Pearson, correlation matrices).
# It may appear only as a visual reference line in profile/line plots.
_ALL_VOXELS_NAME = "all_voxels"


# ===================================================================
# Data structures
# ===================================================================

class ConfigResult:
    """Stores evaluation results for one PMT configuration."""

    def __init__(self, name: str, voxel_ids: list[str], label: str):
        self.name = name
        self.label = label
        self.voxel_ids = voxel_ids
        self.voxel_dicts: list[dict] = []          # full voxel objects from JSON (center, layer, …)
        self.w2: float | None = None               # global W2 homogeneity (all voxels)
        self.col_indices: np.ndarray | None = None

        # NC coverage: coverage_counts[nc] = #selected voxels seeing NC
        self.coverage_counts: np.ndarray | None = None

        # NC statistics
        self.num_ncs: int = 0
        self.num_visible: int = 0       # >= 1 hit in B (after binarization)
        self.num_detected: dict[int, int] = {}  # M -> count(coverage >= M)

        # Muon confusion: (M, W) -> {"TP", "FP", "TN", "FN"}
        self.confusion: dict[tuple[int, int], dict[str, int]] = {}


class EvalData:
    """Shared simulation data loaded once."""

    def __init__(self):
        self.B: sparse.csc_matrix | None = None
        self.voxel_ids: np.ndarray | None = None
        self.voxel_id_to_col: dict[str, int] = {}
        self.num_ncs: int = 0
        self.num_primaries: int = 0
        self.num_runs: int = 0        # unique run IDs in event_ids

        # Muon data
        self.global_muon_id: np.ndarray | None = None
        self.nc_time_ns: np.ndarray | None = None
        self.nc_flag_ge77: np.ndarray | None = None
        self.nc_to_muon_local: np.ndarray | None = None
        self.eligible_nc_mask: np.ndarray | None = None
        self.num_ge77_muons: int = 0
        self.ge77_muon_global_ids: np.ndarray | None = None
        self.total_muons: int = 0

        # Precomputed for muon evaluation (all muons)
        self.all_unique_muons: np.ndarray | None = None
        self.global_to_all_local: np.ndarray | None = None
        self.ge77_mask_all: np.ndarray | None = None
        self.nc_is_veto_candidate: np.ndarray | None = None

        # Per-muon Ge77 NC counts (bincount indexed by all_unique_muons)
        self.ge77_nc_per_muon: np.ndarray | None = None

        # "All voxels" reference coverage (computed from all HDF5 voxels)
        self.coverage_counts_all: np.ndarray | None = None
        self.num_all_voxels: int = 0


# ===================================================================
# Loading
# ===================================================================

def load_config_json(json_path: str) -> tuple[list[str], list[dict], dict]:
    """Load voxel IDs, voxel dicts, and config metadata from a JSON file.

    Supports two formats:
    - Greedy result: dict with ``selected_voxels`` list of objects with
      ``index``, ``center``, ``layer`` keys, and optional ``config`` metadata.
    - Plain list: a JSON array of voxel dicts with ``index``, ``center``,
      ``layer`` (e.g. homogeneous output).
    Returns (voxel_ids, voxel_dicts, metadata_dict).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        raw_list = data
        voxel_ids = [v["index"] if isinstance(v, dict) else v for v in raw_list]
        voxel_dicts = [v for v in raw_list if isinstance(v, dict)]
        data = {}
    else:
        voxel_dicts = data.get("selected_voxels", [])
        voxel_ids = [v["index"] for v in voxel_dicts]

    if len(voxel_ids) == 0:
        raise ValueError(f"No voxels found in {json_path}")

    return voxel_ids, voxel_dicts, data


def load_shared_data(
    hdf5_path: str,
    all_voxel_ids: set[str],
    area_ratios: dict[str, float],
    m: int,
    seed: int,
    verbose: bool = True,
) -> EvalData:
    """
    Load HDF5 data once, build B matrix for the union of all
    voxels needed by any configuration.
    """
    ed = EvalData()

    if verbose:
        print("\n" + "=" * 65)
        print("Loading shared simulation data")
        print("=" * 65)

    # Load raw sparse for ALL voxels (no validity filter)
    (raw_rows, raw_cols, raw_vals,
     full_voxel_ids, centers, layers,
     num_ncs, num_primaries) = load_raw_sparse(
        hdf5_path, verbose=verbose, skip_validity=True,
    )

    ed.num_ncs = num_ncs
    ed.num_primaries = num_primaries
    num_all_voxels = len(full_voxel_ids)
    ed.num_all_voxels = num_all_voxels

    full_id_to_col = {vid: i for i, vid in enumerate(full_voxel_ids)}

    # Validate all requested voxels exist
    missing = all_voxel_ids - set(full_id_to_col.keys())
    if missing:
        raise RuntimeError(
            f"{len(missing)} voxel(s) from configs not found in HDF5. "
            f"First 5: {list(missing)[:5]}"
        )

    # Binarize ALL voxels once (consistent stochastic rounding seed)
    if verbose:
        print(f"\nBinarizing all {num_all_voxels} voxels for reference ...")
    B_all = binarize_from_raw(
        raw_rows, raw_cols, raw_vals,
        num_ncs=num_ncs,
        num_voxels=num_all_voxels,
        layers=layers,
        area_ratios=area_ratios,
        m=m,
        seed=seed,
    )

    # Per-NC coverage count with ALL voxels (the reference upper bound)
    ed.coverage_counts_all = np.asarray(
        B_all.sum(axis=1)
    ).ravel().astype(np.int32)

    if verbose:
        nnz_all = B_all.nnz
        mem_all = (B_all.data.nbytes + B_all.indices.nbytes
                   + B_all.indptr.nbytes) / 1e6
        print(f"All-voxels B: {num_ncs} x {num_all_voxels}, "
              f"nnz={nnz_all:,}, {mem_all:.1f} MB")

    # Subset B_all to only the columns needed by the JSON configs
    needed_old_cols = sorted(full_id_to_col[vid] for vid in all_voxel_ids)
    num_sub_voxels = len(needed_old_cols)
    sub_voxel_ids = full_voxel_ids[needed_old_cols]

    B = B_all[:, needed_old_cols]  # CSC column slicing — efficient

    ed.B = B
    ed.voxel_ids = sub_voxel_ids
    ed.voxel_id_to_col = {vid: i for i, vid in enumerate(sub_voxel_ids)}

    if verbose:
        nnz = B.nnz
        density = nnz / (num_ncs * num_sub_voxels) * 100
        mem_mb = (B.data.nbytes + B.indices.nbytes + B.indptr.nbytes) / 1e6
        print(f"Config B: {num_ncs} x {num_sub_voxels} "
              f"(of {num_all_voxels} total), "
              f"nnz={nnz:,} ({density:.3f}%), {mem_mb:.1f} MB")

    # Load muon data
    (ed.global_muon_id, ed.nc_time_ns,
     ed.nc_flag_ge77) = load_muon_data(
        hdf5_path, num_ncs=num_ncs, verbose=verbose,
    )
    ed.num_runs = count_hdf5_runs(hdf5_path)

    (ed.nc_to_muon_local, _, ed.ge77_muon_global_ids,
     ed.eligible_nc_mask, ed.num_ge77_muons) = build_muon_index(
        ed.global_muon_id, ed.nc_time_ns, ed.nc_flag_ge77,
        verbose=verbose,
    )

    ed.all_unique_muons = np.unique(ed.global_muon_id)
    ed.total_muons = len(ed.all_unique_muons)
    ed.global_to_all_local = np.searchsorted(
        ed.all_unique_muons, ed.global_muon_id
    )
    ed.ge77_mask_all = np.isin(
        ed.all_unique_muons, ed.ge77_muon_global_ids
    )

    # Veto candidate NCs: in time window, not Ge77 themselves
    in_time = (
        (ed.nc_time_ns >= MUON_TIME_WINDOW_MIN_NS)
        & (ed.nc_time_ns <= MUON_TIME_WINDOW_MAX_NS)
    )
    ed.nc_is_veto_candidate = in_time & (~ed.nc_flag_ge77)

    # Per-muon Ge77 NC counts (no time window — flag_ge77 is a physics flag)
    ge77_nc_rows = np.where(ed.nc_flag_ge77.astype(bool))[0]
    if len(ge77_nc_rows) > 0:
        ed.ge77_nc_per_muon = np.bincount(
            ed.global_to_all_local[ge77_nc_rows],
            minlength=ed.total_muons,
        ).astype(np.int32)
    else:
        ed.ge77_nc_per_muon = np.zeros(ed.total_muons, dtype=np.int32)

    if verbose:
        print(f"\nTotal muons: {ed.total_muons:,}")
        print(f"Ge77 muons: {ed.num_ge77_muons:,}")

    return ed


# ===================================================================
# W2 homogeneity computation
# ===================================================================



def compute_config_w2(voxel_dicts: list[dict]) -> float | None:
    """Global W2 homogeneity (2-Wasserstein vs uniform detector surface)."""
    if len(voxel_dicts) < 2:
        return None
    centers = np.array([v["center"] for v in voxel_dicts], dtype=float)
    return compute_wasserstein_homogeneity(centers, reference=get_w2_ref())["w2"]


# ===================================================================
# Evaluation
# ===================================================================

def map_voxels_to_columns(config: ConfigResult, ed: EvalData) -> None:
    """Map config voxel IDs to B-matrix column indices."""
    col_indices = []
    for vid in config.voxel_ids:
        if vid not in ed.voxel_id_to_col:
            raise RuntimeError(
                f"Voxel '{vid}' from config '{config.name}' "
                f"not found in B matrix."
            )
        col_indices.append(ed.voxel_id_to_col[vid])
    config.col_indices = np.array(col_indices, dtype=np.int32)


def evaluate_nc(
    config: ConfigResult,
    ed: EvalData,
    M_values: list[int],
) -> None:
    """
    Compute NC coverage counts and detection statistics.

    Fills config.coverage_counts, config.num_visible,
    config.num_detected[M] for each M.
    """
    B = ed.B
    col_indices = config.col_indices
    num_ncs = ed.num_ncs

    coverage_counts = np.zeros(num_ncs, dtype=np.int16)
    for col in col_indices:
        s, e = B.indptr[col], B.indptr[col + 1]
        coverage_counts[B.indices[s:e]] += 1

    config.coverage_counts = coverage_counts
    config.num_ncs = num_ncs
    config.num_visible = int(np.sum(coverage_counts >= 1))

    for M in M_values:
        config.num_detected[M] = int(np.sum(coverage_counts >= M))


def evaluate_muon(
    config: ConfigResult,
    ed: EvalData,
    M_values: list[int],
    W_values: list[int],
) -> None:
    """
    Compute Ge77 muon confusion matrix for all (M, W).

    NC detected iff coverage_counts >= M.
    Muon's detected-NC count = #eligible NCs (in time window, non-Ge77)
    that are detected.
    Muon classified positive iff detected_nc_count >= W.
    """
    coverage_counts = config.coverage_counts
    if coverage_counts is None:
        raise RuntimeError("Call evaluate_nc before evaluate_muon")

    num_all_muons = ed.total_muons

    for M in M_values:
        nc_detected_veto = (coverage_counts >= M) & ed.nc_is_veto_candidate

        detected_veto_idx = np.where(nc_detected_veto)[0]
        if len(detected_veto_idx) > 0:
            all_muon_lids = ed.global_to_all_local[detected_veto_idx]
            all_muon_det_counts = np.bincount(
                all_muon_lids, minlength=num_all_muons
            ).astype(np.int32)
        else:
            all_muon_det_counts = np.zeros(num_all_muons, dtype=np.int32)

        for W in W_values:
            classified_pos = (all_muon_det_counts >= W)

            TP = int(np.sum(classified_pos & ed.ge77_mask_all))
            FP = int(np.sum(classified_pos & ~ed.ge77_mask_all))
            FN = int(np.sum(~classified_pos & ed.ge77_mask_all))
            TN = int(np.sum(~classified_pos & ~ed.ge77_mask_all))

            entry: dict = {"TP": TP, "FP": FP, "TN": TN, "FN": FN}
            if ed.ge77_nc_per_muon is not None:
                entry["tp_ge77_nc_counts"] = ed.ge77_nc_per_muon[
                    classified_pos & ed.ge77_mask_all
                ].astype(np.int32)
                entry["fn_ge77_nc_counts"] = ed.ge77_nc_per_muon[
                    ~classified_pos & ed.ge77_mask_all
                ].astype(np.int32)
            config.confusion[(M, W)] = entry


# ===================================================================
# Color helper
# ===================================================================

_SETUP_PALETTE: list[str] = [
    # ── 0–9: tab10 (standard categorical palette) ────────────────────
    "#1f77b4",  # 0  blue
    "#d62728",  # 1  red
    "#2ca02c",  # 2  green
    "#ff7f0e",  # 3  orange
    "#9467bd",  # 4  purple
    "#17becf",  # 5  cyan
    "#8c564b",  # 6  brown
    "#e377c2",  # 7  pink
    "#bcbd22",  # 8  yellow-green
    "#7f7f7f",  # 9  grey
    # ── 10–19: perceptually distinct additions ───────────────────────
    "#1b9e77",  # 10 dark teal        (darker/greener than cyan)
    "#e7298a",  # 11 hot magenta      (more saturated than pink)
    "#e6ab02",  # 12 amber            (more yellow than orange)
    "#006d2c",  # 13 dark forest      (much darker than green)
    "#d95f02",  # 14 burnt sienna     (more reddish than orange)
    "#7570b3",  # 15 slate blue       (more grey-blue than purple)
    "#f46d43",  # 16 coral            (warm red-orange)
    "#74c476",  # 17 mint green       (lighter/cooler than green)
    "#b15928",  # 18 rust brown       (reddish-brown, distinct from brown)
    "#313695",  # 19 dark navy        (much darker than blue)
]


def _get_colors(n: int) -> list[str]:
    """Return n visually distinguishable colors, cycling _SETUP_PALETTE if n > 20."""
    return [_SETUP_PALETTE[i % len(_SETUP_PALETTE)] for i in range(n)]


def _config_label(cfg: ConfigResult) -> str:
    """Label with W2 appended if available."""
    if cfg.w2 is not None:
        return f"{cfg.label} (W2={cfg.w2:.1f})"
    return cfg.label


def _sorted_by_w2(configs: list[ConfigResult]) -> list[ConfigResult]:
    """Return configs sorted by W2 descending (None W2 goes last)."""
    return sorted(
        configs,
        key=lambda c: (c.w2 is None, -(c.w2 or 0.0)),
    )


# ===================================================================
# Pearson correlation significance helpers
# ===================================================================

def _pearson_rcrit(n: int, sigma: float = 3.0) -> float:
    """Critical Pearson |r| for sigma-level two-sided significance, n samples."""
    p_two = 2.0 * scipy_stats.norm.sf(sigma)
    t_crit = scipy_stats.t.ppf(1.0 - p_two / 2.0, df=max(n - 2, 1))
    dof = max(n - 2, 1)
    return float(t_crit / np.sqrt(dof + t_crit ** 2))


def _draw_pearson_rcrit(
    ax: plt.Axes,
    n: int,
    sigma: float = 3.0,
    color: str = "black",
    linestyle: str = "--",
    linewidth: float = 1.2,
    draw_label: bool = True,
) -> float:
    """Draw ±r_crit horizontal lines on ax. Returns r_crit."""
    r_crit = _pearson_rcrit(n, sigma)
    lbl = f"{sigma:.0f}σ threshold  |r| = {r_crit:.2f}" if draw_label else "_nolegend_"
    ax.axhline( r_crit, color=color, linestyle=linestyle, linewidth=linewidth,
                label=lbl, alpha=0.75)
    ax.axhline(-r_crit, color=color, linestyle=linestyle, linewidth=linewidth,
                label="_nolegend_", alpha=0.75)
    return r_crit


_P_3SIGMA: float = 2.0 * scipy_stats.norm.sf(3.0)


def _scatter_corr_panel(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    colors: list[str],
    labels: list[str],
    x_label: str = "",
    y_label: str = "",
    title: str = "",
) -> None:
    """Scatter with OLS line and Pearson r / Spearman ρ text box (upper-right).

    Annotation includes 3σ and 5σ Pearson significance thresholds.
    """
    mask = np.isfinite(xs) & np.isfinite(ys)
    for x, y, c, lbl in zip(xs, ys, colors, labels):
        if np.isfinite(x) and np.isfinite(y):
            ax.scatter([x], [y], color=c, s=55, zorder=3)
            ax.annotate(lbl, xy=(x, y), xytext=(3, 3),
                        textcoords="offset points", fontsize=6, color=c)
    if mask.sum() >= 3 and np.std(xs[mask]) > 0 and np.std(ys[mask]) > 0:
        slope, intercept, *_ = scipy_stats.linregress(xs[mask], ys[mask])
        x_fit = np.linspace(xs[mask].min(), xs[mask].max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "k--", linewidth=1.0, zorder=2)
        r_val,   p_r   = scipy_stats.pearsonr(xs[mask], ys[mask])
        rho_val, p_rho = scipy_stats.spearmanr(xs[mask], ys[mask])
        r_crit_3 = _pearson_rcrit(int(mask.sum()), sigma=3.0)
        r_crit_5 = _pearson_rcrit(int(mask.sum()), sigma=5.0)
        sig_marker = "**" if abs(r_val) >= r_crit_5 else ("*" if abs(r_val) >= r_crit_3 else "")
        ax.text(
            0.96, 0.96,
            f"r = {r_val:+.2f}{sig_marker}  (3σ: {r_crit_3:.2f}  5σ: {r_crit_5:.2f})\nρ = {rho_val:+.2f}  p={p_rho:.2g}",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
        )
    if x_label:
        ax.set_xlabel(x_label, fontsize=10)
    if y_label:
        ax.set_ylabel(y_label, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)


# ===================================================================
# Plotting: NC coverage line plot
# ===================================================================

def plot_nc_coverage(
    configs: list[ConfigResult],
    M_max: int,
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """
    Two-panel NC detection coverage plot:
    - Left: log scale, all configs (incl. all-voxels reference)
    - Right: linear scale, configs without the all-voxels reference
    Configs ordered by W2 descending; labels show W2 value.
    """
    M_range = list(range(1, M_max + 1))

    # Sort all configs by W2 desc; the all-voxels config has no W2
    ordered = _sorted_by_w2(configs)
    no_all = [c for c in ordered if c.name != "all_voxels"]

    if color_map is None:
        colors_all = _get_colors(len(ordered))
        color_map = {cfg.name: colors_all[i] for i, cfg in enumerate(ordered)}

    fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(16, 6))

    for cfg in ordered:
        ys = [cfg.num_detected.get(M, 0) for M in M_range]
        ax_log.plot(
            M_range, ys,
            marker="o", markersize=3, linewidth=1.5,
            color=color_map[cfg.name],
            label=_config_label(cfg),
        )

    ax_log.set_yscale("log")
    ax_log.set_xlabel("Multiplicity threshold M", fontsize=11)
    ax_log.set_ylabel("Detected NCs (log scale)", fontsize=11)
    ax_log.set_title("NC Detection Coverage (log scale)", fontsize=12)
    ax_log.set_xticks(M_range)
    ax_log.legend(fontsize=8, loc="upper right")
    ax_log.grid(axis="y", alpha=0.3)
    ax_log.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    for cfg in no_all:
        ys = [cfg.num_detected.get(M, 0) for M in M_range]
        ax_lin.plot(
            M_range, ys,
            marker="o", markersize=3, linewidth=1.5,
            color=color_map[cfg.name],
            label=_config_label(cfg),
        )

    ax_lin.set_xlabel("Multiplicity threshold M", fontsize=11)
    ax_lin.set_ylabel("Detected NCs (linear scale)", fontsize=11)
    ax_lin.set_title("NC Detection Coverage – without all-voxels (linear scale)", fontsize=12)
    ax_lin.set_xticks(M_range)
    ax_lin.legend(fontsize=8, loc="upper right")
    ax_lin.grid(axis="y", alpha=0.3)
    ax_lin.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    plt.tight_layout()
    out_path = output_dir / "nc_coverage_overview.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plotting: NC coverage bar chart (additional)
# ===================================================================

def plot_nc_coverage_bars(
    configs: list[ConfigResult],
    M_max: int,
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """
    Grouped bar chart of NC detection coverage — one bar per setup per M.

    Two panels (one figure):
    - Left: linear scale, all configs except all-voxels reference.
    - Right: log scale, all configs including all-voxels reference.

    Bar height = number of detected NCs at that M threshold.
    Text above each bar (rotated 90°): count and percentage of total NCs.
    """
    M_range = list(range(1, M_max + 1))

    ordered = _sorted_by_w2(configs)
    no_all  = [c for c in ordered if c.name != "all_voxels"]

    if color_map is None:
        colors_all = _get_colors(len(ordered))
        color_map  = {cfg.name: colors_all[i] for i, cfg in enumerate(ordered)}

    x = np.arange(len(M_range))

    # --- figure sizing ---
    # width: enough room for grouped bars + annotations at all M values
    n_max = max(len(no_all), len(ordered))
    bar_width = min(0.8 / n_max, 0.18)
    fig_w = max(18, len(M_range) * (n_max * bar_width + 0.6) + 3)
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(fig_w, 8))

    def _draw_bars(ax: plt.Axes, cfg_list: list[ConfigResult], scale: str) -> None:
        n = len(cfg_list)
        if n == 0:
            return
        bw = min(0.8 / n, 0.18)
        for i, cfg in enumerate(cfg_list):
            offset = (i - (n - 1) / 2) * bw
            vals  = [cfg.num_detected.get(M, 0) for M in M_range]
            total = cfg.num_ncs if cfg.num_ncs > 0 else 1
            bars  = ax.bar(
                x + offset, vals, bw,
                label=_config_label(cfg),
                color=color_map[cfg.name],
            )
            for bar, val in zip(bars, vals):
                pct   = 100.0 * val / total
                y_pos = max(bar.get_height(), 1) if scale == "log" else bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_pos,
                    f"{val:,}\n({pct:.1f}%)",
                    ha="center", va="bottom",
                    fontsize=5, rotation=90, fontweight="bold",
                )

    _draw_bars(ax_lin, no_all,  "linear")
    _draw_bars(ax_log, ordered, "log")

    for ax, scale, title in [
        (ax_lin, "linear", "NC Detection Coverage (linear scale)"),
        (ax_log, "log",    "NC Detection Coverage (log scale)"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels([f"M≥{M}" for M in M_range], fontsize=9)
        ax.set_xlabel("Multiplicity threshold M", fontsize=11)
        ax.set_ylabel("Detected NCs", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
        )
        if scale == "log":
            ax.set_yscale("log")
        # extra headroom so rotated annotations don't get clipped
        ax.margins(y=0.35)

    ax_lin.set_title("NC Detection Coverage (linear scale,\nexcl. all-voxels reference)", fontsize=11)
    ax_log.set_title("NC Detection Coverage (log scale,\nincl. all-voxels reference)", fontsize=11)

    plt.tight_layout()
    out_path = output_dir / "nc_coverage_bars.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plotting: NC Detectability Overview (Plot 03)
# ===================================================================

def plot_nc_detectability_overview(
    configs: list[ConfigResult],
    M_default: int,
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """
    Two-panel NC detectability overview (Plot 03).

    Left panel — absolute grouped bars:
        "Total NCs" and "Detected (M≥M_default)" for every config.
    Right panel — Δ diverging horizontal bars:
        Difference vs. the Baseline config for all other configs.

    Annotations above each bar (rotated 90°): count and percentage of
    total NCs.  Total NCs is identical for all setups (shared truth), so
    the Δ panel only shows the 'Detected' category.
    """
    # Identify baseline reference (first config with label "Baseline")
    baseline_cfg = next(
        (c for c in configs if c.label == "Baseline"),
        configs[1] if len(configs) > 1 else configs[0],
    )

    ordered = _sorted_by_w2(configs)
    n = len(ordered)
    if color_map is None:
        colors_all = _get_colors(n)
        color_map = {cfg.name: colors_all[i] for i, cfg in enumerate(ordered)}

    cat_labels = [f"Detected\n(M≥{M_default})"]
    x = np.arange(len(cat_labels))
    width = min(0.8 / n, 0.18)

    fig, (ax_abs, ax_delta) = plt.subplots(
        1, 2,
        figsize=(max(14, n * 3 + 4), 8),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    # ── Left panel: absolute grouped bars ────────────────────────────
    for i, cfg in enumerate(ordered):
        offset = (i - (n - 1) / 2) * width
        vals  = [cfg.num_detected.get(M_default, 0)]
        total = cfg.num_ncs if cfg.num_ncs > 0 else 1
        bars  = ax_abs.bar(
            x + offset, vals, width,
            label=_config_label(cfg), color=color_map[cfg.name],
        )
        for bar, val in zip(bars, vals):
            pct = 100.0 * val / total
            ax_abs.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:,}\n({pct:.1f}%)",
                ha="center", va="bottom",
                fontsize=6, rotation=90, fontweight="bold",
            )

    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(cat_labels, fontsize=10)
    ax_abs.set_ylabel("Number of NCs", fontsize=11)
    ax_abs.set_title(
        f"NC Detectability Overview  [M_default = {M_default}]",
        fontsize=11,
    )
    ax_abs.legend(fontsize=8)
    ax_abs.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    ax_abs.margins(y=0.30)
    # Indicate what 100% corresponds to (the omitted Total NCs bar)
    _nc_total_note = ordered[0].num_ncs if ordered else 0
    if _nc_total_note > 0:
        ax_abs.text(
            0.01, 0.99, f"100 % = {_nc_total_note:,} NCs",
            transform=ax_abs.transAxes, fontsize=9, va="top", ha="left",
            color="dimgray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.85),
        )

    # ── Right panel: Δ vs Baseline ────────────────────────────────────
    non_ref = [c for c in ordered if c.name != baseline_cfg.name]

    if not non_ref:
        ax_delta.text(
            0.5, 0.5, "Single setup —\nno Δ to display",
            transform=ax_delta.transAxes, ha="center", va="center",
            fontsize=11, color="gray",
        )
        ax_delta.set_axis_off()
    else:
        n_nr     = len(non_ref)
        bar_h    = 0.8 / n_nr
        y_base   = np.array([0.0])          # one category: 'Detected'
        ref_val  = baseline_cfg.num_detected.get(M_default, 0)
        ref_tot  = baseline_cfg.num_ncs if baseline_cfg.num_ncs > 0 else 1

        for j, cfg in enumerate(non_ref):
            val    = cfg.num_detected.get(M_default, 0)
            delta  = val - ref_val
            offset = (j - (n_nr - 1) / 2) * bar_h

            bars = ax_delta.barh(
                y_base + offset, [delta], bar_h * 0.88,
                label=_config_label(cfg), color=color_map[cfg.name],
            )
            for bar in bars:
                pct  = 100.0 * delta / max(ref_tot, 1)
                sign = "+" if delta >= 0 else ""
                if delta == 0:
                    ax_delta.text(
                        0, bar.get_y() + bar.get_height() / 2,
                        " =ref", va="center", ha="left", fontsize=7,
                        color=color_map[cfg.name], fontstyle="italic",
                    )
                    continue
                txt = f"{sign}{delta:,}\n({sign}{pct:.1f}%)"
                pad = max(abs(delta) * 0.02, 1)
                ax_delta.text(
                    delta + (pad if delta >= 0 else -pad),
                    bar.get_y() + bar.get_height() / 2,
                    txt, va="center",
                    ha="left" if delta >= 0 else "right",
                    fontsize=7, color=color_map[cfg.name], fontweight="bold",
                )

        ax_delta.axvline(0, color="black", linewidth=0.9)
        ax_delta.set_yticks(y_base)
        ax_delta.set_yticklabels([f"Detected (M≥{M_default})"], fontsize=10)
        ax_delta.set_xlabel(
            f"Δ count vs Baseline ({baseline_cfg.label})", fontsize=10
        )
        ax_delta.set_title(
            f"Δ from Baseline: {baseline_cfg.label}", fontsize=11
        )
        ax_delta.legend(fontsize=8, title="vs Baseline")
        ax_delta.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v):+,}")
        )
        ax_delta.grid(True, axis="x", alpha=0.3)
        ax_delta.invert_yaxis()

    plt.tight_layout()
    out_path = output_dir / "nc_detectability_overview.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plotting: Recall / Precision M×W sweep (Plots 10 & 11)
# ===================================================================

def plot_mw_sweep(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
    total_primaries: int = 0,
) -> None:
    """
    One figure per metric (Recall, Precision, and optionally FoM) sweeping
    all (M, W) pairs.

    X-axis order: M1W1, M1W2, …, M1W_max, M2W1, … (M outer, W inner).
    Each config is one line in its palette colour.  Vertical dashed
    separators mark M-group boundaries; the M value is labelled above
    each group.  Y-axis shows percentage values (0 %–100 %) for
    Recall/Precision, and raw FoM values for the FoM sweep.

    Recall    = TP / (TP + FN)
    Precision = TP / (TP + FP)
    FoM       = calc_fom_confusion(TP, FP, FN, total_primaries)  [if total_primaries > 0]
    """
    # Exclude the "all voxels" reference from line plots
    ordered    = [cfg for cfg in _sorted_by_w2(configs)
                  if cfg.name != _ALL_VOXELS_NAME]
    if color_map is None:
        colors_all = _get_colors(len(ordered))
        color_map  = {cfg.name: colors_all[i] for i, cfg in enumerate(ordered)}

    mw_pairs = [(M, W) for M in M_values for W in W_values]
    x_labels = [f"M{M}W{W}" for M, W in mw_pairs]
    x        = np.arange(len(mw_pairs))
    n_w      = len(W_values)

    metrics_list = [
        ("Recall",    "recall"),
        ("Precision", "precision"),
    ]
    if total_primaries > 0:
        metrics_list.append(("FoM", "fom"))

    for metric_name, fname_suffix in metrics_list:
        fig_w = max(30, len(mw_pairs) * 0.18)
        fig, ax = plt.subplots(figsize=(fig_w, 9))

        for cfg in ordered:
            vals = []
            for M, W in mw_pairs:
                cm = cfg.confusion.get((M, W))
                if cm is None:
                    vals.append(np.nan)
                    continue
                TP, FP, FN = cm["TP"], cm["FP"], cm["FN"]
                if metric_name == "Recall":
                    denom = TP + FN
                    vals.append(TP / denom if denom > 0 else 0.0)
                elif metric_name == "Precision":
                    denom = TP + FP
                    vals.append(TP / denom if denom > 0 else 0.0)
                else:  # FoM
                    vals.append(calc_fom_confusion(
                        TP, FP, FN, total_primaries,
                        tp_ge77_nc_counts=cm.get("tp_ge77_nc_counts"),
                        fn_ge77_nc_counts=cm.get("fn_ge77_nc_counts"),
                    ))

            ax.plot(
                x, vals,
                color=color_map[cfg.name],
                label=_config_label(cfg),
                linewidth=1.2, marker=".", markersize=4,
            )

        # Envelope: for each M, the highest value achieved by any setup
        # at any W in that M-group (placed at the x-position of that W)
        env_x, env_y = [], []
        for M in M_values:
            best_idx = None
            best_val = -np.inf
            for W in W_values:
                idx = next(
                    (i for i, (m, w) in enumerate(mw_pairs) if m == M and w == W),
                    None,
                )
                if idx is None:
                    continue
                for cfg in ordered:
                    cm = cfg.confusion.get((M, W))
                    if cm is None:
                        continue
                    TP, FP, FN = cm["TP"], cm["FP"], cm["FN"]
                    if metric_name == "Recall":
                        denom = TP + FN
                        v = TP / denom if denom > 0 else 0.0
                    elif metric_name == "Precision":
                        denom = TP + FP
                        v = TP / denom if denom > 0 else 0.0
                    else:  # FoM
                        v = calc_fom_confusion(
                            TP, FP, FN, total_primaries,
                            tp_ge77_nc_counts=cm.get("tp_ge77_nc_counts"),
                            fn_ge77_nc_counts=cm.get("fn_ge77_nc_counts"),
                        )
                        if not np.isfinite(v):
                            continue
                    if v > best_val:
                        best_val = v
                        best_idx = idx
            if best_idx is not None and np.isfinite(best_val):
                env_x.append(best_idx)
                env_y.append(best_val)
        if env_x:
            ax.plot(
                env_x, env_y,
                color="black", linestyle="--", linewidth=2.0,
                marker="D", markersize=6,
                label="Envelope (best at W=1)",
                zorder=5,
            )

        # Vertical separators + M-group labels above the axes
        for gi, M in enumerate(M_values):
            group_start = gi * n_w
            group_mid   = group_start + (n_w - 1) / 2
            if gi > 0:
                ax.axvline(
                    group_start - 0.5, color="gray",
                    linewidth=0.6, linestyle="--", alpha=0.5,
                )
            ax.text(
                group_mid, 1.02, f"M={M}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color="dimgray",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
        ax.set_ylabel(metric_name, fontsize=12)
        if metric_name == "FoM":
            ax.yaxis.set_major_locator(mticker.AutoLocator())
        else:
            ax.set_ylim(-0.02, 1.08)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
            )
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.set_title(
            f"Ge-77 Muon Classification — {metric_name} across all (M, W) combinations",
            fontsize=13, pad=20,
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlim(-0.5, len(mw_pairs) - 0.5)

        fig.tight_layout()
        out_path = output_dir / f"mw_sweep_{fname_suffix}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        if verbose:
            print(f"  Saved: {out_path}")



# ===================================================================
# Plotting: Figure of Merit
# ===================================================================

def _fom_grid(
    cfg: "ConfigResult",
    M_values: list[int],
    W_values: list[int],
    total_muons: int,
) -> dict[tuple[int, int], float]:
    """Return {(M, W): fom} for all combinations. Missing entries → nan."""
    result: dict[tuple[int, int], float] = {}
    for M in M_values:
        for W in W_values:
            cm = cfg.confusion.get((M, W))
            if cm is None:
                result[(M, W)] = np.nan
            else:
                result[(M, W)] = calc_fom_confusion(
                    cm["TP"], cm["FP"], cm["FN"], total_muons,
                    tp_ge77_nc_counts=cm.get("tp_ge77_nc_counts"),
                    fn_ge77_nc_counts=cm.get("fn_ge77_nc_counts"),
                )
    return result


def plot_fom_summary(
    configs: list,
    M_values: list[int],
    W_values: list[int],
    total_muons: int,
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """Horizontal bar chart: max FoM per config, annotated with optimal (M, W).

    Configs are shown in W2-sorted order (consistent with all other plots).
    """
    ordered = _sorted_by_w2(configs)
    if color_map is None:
        _pal = _get_colors(len(ordered))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(ordered)}
    colors = [color_map.get(cfg.name, "gray") for cfg in ordered]

    max_foms: list[float] = []
    best_mw:  list[str]   = []

    for cfg in ordered:
        grid  = _fom_grid(cfg, M_values, W_values, total_muons)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best = max(valid, key=valid.__getitem__)
            max_foms.append(valid[best])
            best_mw.append(f"M{best[0]}W{best[1]}")
        else:
            max_foms.append(np.nan)
            best_mw.append("N/A")

    labels = [_config_label(cfg) for cfg in ordered]
    y      = np.arange(len(ordered))
    y_max  = max((f for f in max_foms if np.isfinite(f)), default=1.0)

    fig, ax = plt.subplots(figsize=(8, max(4, len(ordered) * 0.55)))
    bars = ax.barh(y, max_foms, color=colors, height=0.6)

    for bar, mw, fom in zip(bars, best_mw, max_foms):
        if np.isfinite(fom):
            ax.text(
                fom + 0.01 * y_max,
                bar.get_y() + bar.get_height() / 2,
                f"{fom:.4g}  [{mw}]",
                va="center", ha="left", fontsize=8,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    # Extra right margin so annotations don't get clipped
    ax.set_xlim(right=y_max * 1.35)
    ax.set_xlabel("Figure of Merit  (max over all M, W)", fontsize=11)
    ax.set_title(
        "Ge-77 Muon Figure of Merit — Best (M, W) per Configuration",
        fontsize=12, pad=10,
    )
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "fom_summary.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


def plot_fom_per_setup(
    configs: list,
    M_values: list[int],
    W_values: list[int],
    total_muons: int,
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """Per-config FoM heatmap (M × W grid).

    Produces one PNG file per config:
      fom_heatmap_{name}.png  — 2-D colour map of FoM over (M, W)

    The cell with the maximum FoM is highlighted.
    """
    ordered  = _sorted_by_w2(configs)
    mw_pairs = [(M, W) for M in M_values for W in W_values]
    x_labels = [f"M{M}W{W}" for M, W in mw_pairs]
    x_arr    = np.arange(len(mw_pairs))
    n_w      = len(W_values)

    for cfg in ordered:
        grid      = _fom_grid(cfg, M_values, W_values, total_muons)
        title_sfx = _config_label(cfg)
        safe_name = cfg.name.replace(" ", "_").replace("/", "_")
        finite    = {k: v for k, v in grid.items() if np.isfinite(v)}

        # ── 1. Heatmap ────────────────────────────────────────────────
        heat = np.array(
            [[grid.get((M, W), np.nan) for M in M_values] for W in W_values]
        )  # shape (n_W, n_M)

        fig_h, ax_h = plt.subplots(
            figsize=(max(6, len(M_values) * 0.9), max(5, len(W_values) * 0.45))
        )
        im = ax_h.imshow(
            heat, aspect="auto", origin="lower",
            extent=[M_values[0] - 0.5, M_values[-1] + 0.5,
                    W_values[0] - 0.5, W_values[-1] + 0.5],
            cmap="viridis",
        )
        plt.colorbar(im, ax=ax_h, label="Figure of Merit")

        if finite:
            best = max(finite, key=finite.__getitem__)
            ax_h.scatter(
                best[0], best[1],
                marker="*", s=250, color="red", zorder=5,
                label=f"Best: M{best[0]}W{best[1]} = {finite[best]:.4g}",
            )
            ax_h.legend(fontsize=9, loc="upper right")

        ax_h.set_xlabel("M  (NC threshold)", fontsize=11)
        ax_h.set_ylabel("W  (muon threshold)", fontsize=11)
        ax_h.set_xticks(M_values)
        ax_h.set_yticks(W_values)
        ax_h.set_title(f"Figure of Merit — {title_sfx}", fontsize=12, pad=10)
        fig_h.tight_layout()
        out_h = output_dir / f"fom_heatmap_{safe_name}.png"
        plt.savefig(out_h, dpi=200, bbox_inches="tight")
        plt.close(fig_h)
        if verbose:
            print(f"  Saved: {out_h}")



# ===================================================================
# Plotting: W2 correlation — FoM and Recall
# ===================================================================

def plot_w2_fom_best(
    configs: list,
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """Scatter: best FoM (max over all M, W) vs W2 homogeneity.

    One labelled point per config with W2.  OLS regression line and
    Spearman ρ are shown when ≥ 2 configs have W2 values.
    """
    w2_cfgs = sorted(
        [cfg for cfg in configs if cfg.w2 is not None],
        key=lambda c: c.w2,
    )
    if len(w2_cfgs) < 2:
        if verbose:
            print("  [SKIP] w2_fom_best.png: fewer than 2 configs have W2.")
        return

    if color_map is None:
        _pal = _get_colors(len(w2_cfgs))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(w2_cfgs)}
    w2_vals  = np.array([c.w2 for c in w2_cfgs])
    fom_best = np.array([
        max(
            (v for v in _fom_grid(c, M_values, W_values, total_primaries).values()
             if np.isfinite(v)),
            default=np.nan,
        )
        for c in w2_cfgs
    ])
    mask = np.isfinite(fom_best)

    fig, ax = plt.subplots(figsize=(8, 6))
    for c, w2v, fom in zip(w2_cfgs, w2_vals, fom_best):
        if np.isfinite(fom):
            ax.scatter(w2v, fom, color=color_map.get(c.name, "gray"), s=65, zorder=3)
            ax.annotate(c.label, xy=(w2v, fom), xytext=(4, 3),
                        textcoords="offset points", fontsize=7)

    if mask.sum() >= 2:
        slope, intercept, _, _, _ = scipy_stats.linregress(
            w2_vals[mask], fom_best[mask]
        )
        rho, p_rho = scipy_stats.spearmanr(w2_vals[mask], fom_best[mask])
        x_line = np.linspace(w2_vals[mask].min(), w2_vals[mask].max(), 200)
        ax.plot(x_line, slope * x_line + intercept,
                color="black", linestyle="--", linewidth=1.3,
                label=f"OLS   ρ_s = {rho:+.3f}   p = {p_rho:.3f}")
        ax.legend(fontsize=9)

    ax.set_xlabel("Global W2 (mm) — lower = more uniform", fontsize=11)
    ax.set_ylabel("Best FoM  (max over all M, W)", fontsize=11)
    ax.set_title("W2 Homogeneity vs Best Achievable FoM", fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "w2_fom_best.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


def plot_w2_sorted_heatmaps(
    configs: list,
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """W2-sorted heatmaps of Recall and FoM vs (setup × W) for M ∈ {1,3,5,10}.

    Setups are sorted left-to-right by ascending W2 (most uniform first).
    The W2 value is appended to each x-axis label.
    Only setups with a W2 value are included.

    Produces 4 Recall figures and 4 FoM figures (one per M in {1,3,5,10}).
      w2_heatmap_recall_M{M:02d}.png
      w2_heatmap_fom_M{M:02d}.png
    """
    w2_cfgs = sorted(
        [cfg for cfg in configs if cfg.w2 is not None],
        key=lambda c: c.w2,
    )
    if len(w2_cfgs) < 2:
        if verbose:
            print("  [SKIP] w2_sorted_heatmaps: fewer than 2 configs have W2.")
        return

    heatmap_ms = [m for m in [1, 3, 5, 10] if m in M_values]
    n_cfg    = len(w2_cfgs)
    n_w      = len(W_values)
    x_labels = [f"{c.label}\nW2={c.w2:.1f}" for c in w2_cfgs]

    for M in heatmap_ms:
        for metric, fname_pfx, cmap, fixed_vmin, fixed_vmax in [
            ("Recall", "w2_heatmap_recall", "YlOrRd",  0.0,  1.0),
            ("FoM",    "w2_heatmap_fom",    "viridis", None, None),
        ]:
            data = np.full((n_w, n_cfg), np.nan)
            for wi, W in enumerate(W_values):
                for ci, cfg in enumerate(w2_cfgs):
                    cm = cfg.confusion.get((M, W))
                    if cm is None:
                        continue
                    if metric == "Recall":
                        denom = cm["TP"] + cm["FN"]
                        data[wi, ci] = cm["TP"] / denom if denom > 0 else 0.0
                    else:
                        data[wi, ci] = calc_fom_confusion(
                            cm["TP"], cm["FP"], cm["FN"], total_primaries,
                            tp_ge77_nc_counts=cm.get("tp_ge77_nc_counts"),
                            fn_ge77_nc_counts=cm.get("fn_ge77_nc_counts"),
                        )

            finite_vals = data[np.isfinite(data)]
            _vmin = fixed_vmin if fixed_vmin is not None else (
                finite_vals.min() if len(finite_vals) else 0.0
            )
            _vmax = fixed_vmax if fixed_vmax is not None else (
                finite_vals.max() if len(finite_vals) else 1.0
            )
            mid = (_vmin + _vmax) / 2

            fig_w = max(8, n_cfg * 2.4 + 2)
            fig_h = max(5, n_w * 0.45 + 2)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            im = ax.imshow(
                data, aspect="auto", cmap=cmap,
                vmin=_vmin, vmax=_vmax, origin="upper",
            )
            plt.colorbar(im, ax=ax, label=metric)

            for wi in range(n_w):
                for ci in range(n_cfg):
                    val = data[wi, ci]
                    if not np.isfinite(val):
                        continue
                    txt     = f"{val:.3f}" if metric == "Recall" else f"{val:.3g}"
                    txt_col = "white" if val > mid else "black"
                    ax.text(ci, wi, txt, ha="center", va="center",
                            fontsize=6, color=txt_col)

            ax.set_xticks(range(n_cfg))
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n_w))
            ax.set_yticklabels([f"W={W}" for W in W_values], fontsize=8)
            metric_title = "Recall" if metric == "Recall" else "Figure of Merit"
            ax.set_title(
                f"{metric_title} at M={M}  —  setups sorted by W2 ascending",
                fontsize=11, pad=8,
            )
            ax.set_xlabel("Setup  (sorted by W2, most uniform → left)", fontsize=10)
            ax.set_ylabel("W  (muon threshold)", fontsize=10)

            fig.tight_layout()
            out_path = output_dir / f"{fname_pfx}_M{M:02d}.png"
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close()
            if verbose:
                print(f"  Saved: {out_path}")


# ===================================================================
# Plotting: Muon Ge77 heatmaps
# ===================================================================

def plot_muon_heatmaps(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: Path,
    num_ge77_muons: int,
    total_muons: int,
    verbose: bool = True,
) -> None:
    """
    Per M: one figure with 2 heatmaps (Recall, Precision).
    Axes: W (y) x Config (x). Cell text = value.
    """
    config_labels = [c.label for c in configs]
    num_configs = len(configs)

    for M in M_values:
        fig, axes = plt.subplots(
            1, 2, figsize=(max(7, num_configs * 2.4 + 2), len(W_values) * 0.8 + 3),
            sharey=True,
        )

        metric_names = ["Recall", "Precision"]
        cmaps = ["YlOrRd", "PuBu"]

        for mi, (metric_name, cmap) in enumerate(
            zip(metric_names, cmaps)
        ):
            ax = axes[mi]
            data_matrix = np.full((len(W_values), num_configs), np.nan)

            for wi, W in enumerate(W_values):
                for ci, cfg in enumerate(configs):
                    cm = cfg.confusion.get((M, W))
                    if cm is None:
                        continue

                    TP, FP, TN, FN = cm["TP"], cm["FP"], cm["TN"], cm["FN"]

                    if metric_name == "Recall":
                        denom = TP + FN
                        val = TP / denom if denom > 0 else 0.0
                    elif metric_name == "Precision":
                        denom = TP + FP
                        val = TP / denom if denom > 0 else 0.0
                    else:
                        val = 0.0

                    data_matrix[wi, ci] = val

            # Plot heatmap
            im = ax.imshow(
                data_matrix, aspect="auto", cmap=cmap,
                vmin=0, vmax=1,
            )

            # Cell text
            for wi in range(len(W_values)):
                for ci in range(num_configs):
                    val = data_matrix[wi, ci]
                    if np.isnan(val):
                        continue
                    text_color = "white" if val > 0.7 else "black"
                    ax.text(
                        ci, wi, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=7, color=text_color,
                    )

            ax.set_xticks(range(num_configs))
            ax.set_xticklabels(config_labels, rotation=45, ha="right",
                               fontsize=8)
            ax.set_yticks(range(len(W_values)))
            ax.set_yticklabels([f"W={W}" for W in W_values], fontsize=9)
            ax.set_title(f"{metric_name}", fontsize=11)

            if mi == 0:
                ax.set_ylabel("W threshold", fontsize=10)

        fig.suptitle(
            f"Ge77 Muon Classification (M={M})\n"
            f"Ge77 muons: {num_ge77_muons:,} / {total_muons:,} total  "
            f"| Recall = TP/(TP+FN)  Precision = TP/(TP+FP)",
            fontsize=10, y=1.02,
        )
        plt.tight_layout()

        out_path = output_dir / f"muon_heatmap_M{M}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        if verbose:
            print(f"  Saved: {out_path}")


# ===================================================================
# Text output
# ===================================================================

def write_confusion_txt(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: Path,
    num_ge77_muons: int,
    total_muons: int,
    verbose: bool = True,
    ratio_override_warning: Optional[str] = None,
) -> None:
    """Write confusion matrices to text file."""
    out_path = output_dir / "confusion_matrices.txt"
    with open(out_path, "w") as f:
        if ratio_override_warning is not None:
            f.write(ratio_override_warning + "\n")
        f.write(f"# Ge77 Muon Confusion Matrices\n")
        f.write(f"# Total muons: {total_muons:,}\n")
        f.write(f"# Ge77 muons: {num_ge77_muons:,}\n")
        f.write(f"# Non-Ge77 muons: {total_muons - num_ge77_muons:,}\n\n")

        header = (f"{'Config':<25} {'M':>3} {'W':>3} "
                  f"{'TP':>8} {'FP':>8} {'TN':>10} {'FN':>8} "
                  f"{'Acc':>8} {'Prec':>8}\n")
        f.write(header)
        f.write("-" * len(header) + "\n")

        for cfg in configs:
            for M in M_values:
                for W in W_values:
                    cm = cfg.confusion.get((M, W))
                    if cm is None:
                        continue
                    TP, FP, TN, FN = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
                    total = TP + FP + TN + FN
                    acc = (TP + TN) / total if total > 0 else 0
                    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
                    f.write(
                        f"{cfg.label:<25} {M:>3} {W:>3} "
                        f"{TP:>8} {FP:>8} {TN:>10} {FN:>8} "
                        f"{acc:>8.4f} {prec:>8.4f}\n"
                    )
            f.write("\n")

    if verbose:
        print(f"  Saved: {out_path}")


def write_nc_summary_txt(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
    verbose: bool = True,
    ratio_override_warning: Optional[str] = None,
) -> None:
    """Write NC detection summary to text file."""
    out_path = output_dir / "nc_summary.txt"
    with open(out_path, "w") as f:
        if ratio_override_warning is not None:
            f.write(ratio_override_warning + "\n")
        f.write(f"# NC Detection Summary\n\n")

        for cfg in configs:
            f.write(f"--- {cfg.label} ---\n")
            f.write(f"  Total NCs:    {cfg.num_ncs:>12,}\n")
            f.write(f"  Visible:      {cfg.num_visible:>12,} "
                    f"({cfg.num_visible / cfg.num_ncs:.4%})\n")
            for M in M_values:
                n = cfg.num_detected.get(M, 0)
                pct_total = n / cfg.num_ncs if cfg.num_ncs > 0 else 0
                pct_vis = n / cfg.num_visible if cfg.num_visible > 0 else 0
                f.write(f"  Detected M≥{M}: {n:>12,} "
                        f"({pct_total:.4%} total, {pct_vis:.4%} of visible)\n")
            f.write("\n")

    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plotting: W2 vs coverage
# ===================================================================


def plot_w2_scatter(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
    M_fixed: int = 1,
    W_fixed: int = 1,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """Three-panel W2 scatter figure at fixed (M_fixed, W_fixed).

    Panel 1 — W2 vs NC coverage fraction for multiple M thresholds
               (colored by M value; individual setups annotated).
    Panel 2 — W2 vs Recall at (M_fixed, W_fixed), per-setup colours.
    Panel 3 — W2 vs Precision at (M_fixed, W_fixed), per-setup colours.

    The "all_voxels" reference config is excluded from all panels
    (it has no W2 and would bias any visual trend).  OLS regression
    lines are overlaid on panels 2 and 3.
    """
    w2_cfgs = [cfg for cfg in configs
               if cfg.w2 is not None and cfg.name != _ALL_VOXELS_NAME]
    if len(w2_cfgs) < 2:
        if verbose:
            print("  w2_scatter: fewer than 2 configs with W2 — skipped.")
        return

    w2_vals = np.array([cfg.w2 for cfg in w2_cfgs])
    if color_map is None:
        setup_colors = _get_colors(len(w2_cfgs))
        color_map = {cfg.name: setup_colors[i] for i, cfg in enumerate(w2_cfgs)}

    # M values shown in panel 1 (NC coverage across multiple thresholds)
    panel1_ms = sorted({M for M in [1, 2, 4, 5, 10] if M in M_values})
    m_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(panel1_ms)))

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(
        f"W2 Homogeneity vs Performance  (M={M_fixed}, W={W_fixed})",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: W2 vs NC fraction for multiple M thresholds ─────────
    ax = axes[0]
    for sM, cm in zip(panel1_ms, m_colors):
        fracs = np.array([_nc_frac(cfg, sM) for cfg in w2_cfgs])
        ax.scatter(w2_vals, fracs, color=cm, s=55, zorder=3, label=f"M={sM}")
        for cfg, w2v, frac in zip(w2_cfgs, w2_vals, fracs):
            ax.annotate(cfg.label, xy=(w2v, frac), xytext=(4, 2),
                        textcoords="offset points", fontsize=6, color=cm)
    ax.set_xlabel("Global W2 (mm) — lower = more uniform", fontsize=10)
    ax.set_ylabel("NC detection fraction", fontsize=10)
    ax.set_title("W2 vs NC Coverage", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.legend(title="M threshold", fontsize=8, title_fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panels 2 & 3: W2 vs Recall / Precision with OLS line ─────────
    for ax, metric_fn, ylabel, title in [
        (axes[1], _recall,    "Recall",    f"W2 vs Recall  (M={M_fixed}, W={W_fixed})"),
        (axes[2], _precision, "Precision", f"W2 vs Precision  (M={M_fixed}, W={W_fixed})"),
    ]:
        y_vals = np.array([metric_fn(cfg, M_fixed, W_fixed) for cfg in w2_cfgs])
        for cfg, w2v, yv in zip(w2_cfgs, w2_vals, y_vals):
            c = color_map[cfg.name]
            ax.scatter([w2v], [yv], color=c, s=70, zorder=3)
            ax.annotate(cfg.label, xy=(w2v, yv), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color=c)

        # OLS regression line (if enough variance)
        if len(w2_vals) >= 3 and np.std(w2_vals) > 0 and np.std(y_vals) > 0:
            slope, intercept, *_ = scipy_stats.linregress(w2_vals, y_vals)
            x_fit = np.linspace(w2_vals.min(), w2_vals.max(), 200)
            ax.plot(x_fit, slope * x_fit + intercept,
                    color="black", linewidth=1.2, linestyle="--", zorder=2,
                    label="OLS fit")

        ax.set_xlabel("Global W2 (mm)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / f"w2_scatter_M{M_fixed}_W{W_fixed}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


def plot_muon_w2_scatter(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: Path,
    num_ge77_muons: int,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """One subplot per (M, W) pair where any W2-plotted setup achieves
    TP > 20% of all Ge77 muons (recall > 0.20).  Precision per setup is
    printed below each panel.  Shared legend."""
    w2_cfgs = [cfg for cfg in configs if cfg.w2 is not None]

    if len(w2_cfgs) < 2 or num_ge77_muons == 0:
        return

    threshold = 0.05 * num_ge77_muons

    # Find all (M, W) pairs where at least one w2-plotted config exceeds threshold
    selected_mw = []
    for M in M_values:
        for W in W_values:
            for cfg in w2_cfgs:
                cm = cfg.confusion.get((M, W))
                if cm is not None and cm["TP"] > threshold:
                    selected_mw.append((M, W))
                    break  # one setup suffices

    if not selected_mw:
        if verbose:
            print(f"  muon_w2_scatter: no (M,W) pair reaches TP > 5% of Ge77 muons "
                  f"(>{0.05 * num_ge77_muons:.0f}) — skipped.")
        return

    if color_map is None:
        _pal = _get_colors(len(w2_cfgs))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(w2_cfgs)}

    ncols = 3
    nrows = (len(selected_mw) + ncols - 1) // ncols
    # hspace scales with number of configs so the per-panel text box
    # (one line per config) does not overlap the next row's axes.
    hspace = max(0.55, 0.09 * (len(w2_cfgs) + 2))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5, nrows * 5.5),
        gridspec_kw={"hspace": hspace},
    )
    axes_flat = np.array(axes).flatten()

    legend_handles = []

    for pi, (M, W) in enumerate(selected_mw):
        ax = axes_flat[pi]

        precision_lines = []
        for cfg in w2_cfgs:
            color = color_map[cfg.name]
            w2 = cfg.w2
            cm = cfg.confusion.get((M, W))
            tp = cm["TP"] if cm is not None else 0
            fp = cm["FP"] if cm is not None else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            ax.plot([w2, w2], [0, tp], color=color, linewidth=1.8, alpha=0.7)
            ax.scatter([w2], [tp], color=color, s=60, zorder=3)
            precision_lines.append(f"{cfg.label}: {prec:.3f}")

            if pi == 0:
                legend_handles.append(
                    plt.Line2D([0], [0], color=color, marker="o", markersize=6,
                               label=_config_label(cfg), linewidth=1.5)
                )

        ax.set_xlabel("Global W2 (mm)", fontsize=9)
        ax.set_ylabel("Detected Ge77 muons (TP)", fontsize=9)
        ax.set_title(f"M ≥ {M},  W ≥ {W}", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

        # Precision text below each subplot — one config per line, inside a bbox
        prec_text = "Precision:\n" + "\n".join(f"  {line}" for line in precision_lines)
        ax.text(
            0.5, -0.02, prec_text,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=7, family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="lightyellow",
                edgecolor="gray",
                alpha=0.85,
                linewidth=0.6,
            ),
        )

    for pi in range(len(selected_mw), len(axes_flat)):
        axes_flat[pi].set_visible(False)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(w2_cfgs), 4),
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
        framealpha=0.9,
    )
    fig.suptitle(
        f"Ge77 Muon Detection (TP) vs Global W2 Homogeneity\n"
        f"(Ge77 muons: {num_ge77_muons:,}  |  panels shown: TP > 5% of Ge77 muons  |  "
        f"{len(selected_mw)} (M,W) pair(s))",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out_path = output_dir / "muon_w2_scatter.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}  [{len(selected_mw)} panel(s)]")


# ===================================================================
# W2 correlation analysis helpers
# ===================================================================

# Core regression/residual overlay is shared with compare_coverages.py
# via src/pmtopt/w2_plot_helpers.py to ensure identical visual style.
from pmtopt.w2_plot_helpers import regression_overlay as _regression_overlay


def _nc_frac(cfg: ConfigResult, M: int) -> float:
    """NC detection fraction for config at threshold M."""
    return cfg.num_detected.get(M, 0) / cfg.num_ncs if cfg.num_ncs > 0 else 0.0


def _recall(cfg: ConfigResult, M: int, W: int) -> float:
    cm = cfg.confusion.get((M, W))
    if cm is None:
        return 0.0
    denom = cm["TP"] + cm["FN"]
    return cm["TP"] / denom if denom > 0 else 0.0


def _precision(cfg: ConfigResult, M: int, W: int) -> float:
    cm = cfg.confusion.get((M, W))
    if cm is None:
        return 0.0
    denom = cm["TP"] + cm["FP"]
    return cm["TP"] / denom if denom > 0 else 0.0


def _signal_survival(cfg: ConfigResult, M: int, W: int) -> float:
    """Signal survival at (M, W): 1 − (TP+FP)/(TP+FP+TN+FN)."""
    cm = cfg.confusion.get((M, W))
    if cm is None:
        return 0.0
    return 1.0 - calc_veto_fraction(cm["TP"], cm["FP"], cm["TN"], cm["FN"])


# ===================================================================
# Plot A — W2 × NC coverage correlation scatter
# ===================================================================

def plot_w2_nc_correlation(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """
    Plot A — W2 vs NC detection fraction for every M threshold.

    Layout: 4 rows × 3 cols (one panel per M=1..10, 2 empty).
    Each panel has a scatter subplot (top) and a residual subplot (bottom),
    sharing the x-axis.  OLS regression line with 95 % CI is overlaid.
    Pearson r and Spearman ρ with p-values are annotated.
    """
    w2_cfgs = [cfg for cfg in configs
               if cfg.w2 is not None and cfg.name != _ALL_VOXELS_NAME]
    if len(w2_cfgs) < 2:
        if verbose:
            print("  w2_nc_correlation: fewer than 2 configs have W2 — skipped.")
        return

    if color_map is None:
        colors_all = _get_colors(len(w2_cfgs))
        color_map  = {cfg.name: colors_all[i] for i, cfg in enumerate(w2_cfgs)}
    w2_arr     = np.array([cfg.w2 for cfg in w2_cfgs])
    labels     = [cfg.label for cfg in w2_cfgs]
    color_pts  = [color_map.get(cfg.name, "gray") for cfg in w2_cfgs]

    ncols = 3
    nrows = (len(M_values) + ncols - 1) // ncols   # e.g. 4 for M=1..10

    # Each M panel = 2 matplotlib rows (scatter + residual)
    fig = plt.figure(figsize=(ncols * 5, nrows * 6))
    fig.suptitle(
        "W2 Homogeneity vs NC Detection Fraction\n"
        "(OLS fit · 95 % CI · Pearson r · Spearman ρ)",
        fontsize=13,
    )

    for pi, M in enumerate(M_values):
        col = pi % ncols
        row = pi // ncols

        # Two gridspec rows per logical panel row (scatter + residual)
        gs_row_top    = row * 2
        gs_row_bottom = row * 2 + 1

        ax_scatter = fig.add_subplot(nrows * 2, ncols,
                                     gs_row_top * ncols + col + 1)
        ax_resid   = fig.add_subplot(nrows * 2, ncols,
                                     gs_row_bottom * ncols + col + 1,
                                     sharex=ax_scatter)

        y_arr = np.array([_nc_frac(cfg, M) for cfg in w2_cfgs])

        _regression_overlay(
            ax_scatter, ax_resid,
            w2_arr, y_arr,
            color_pts, labels,
            y_label=f"NC fraction (M≥{M})",
        )
        ax_scatter.set_title(f"M ≥ {M}", fontsize=9)
        ax_scatter.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%")
        )
        ax_resid.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%")
        )
        plt.setp(ax_scatter.get_xticklabels(), visible=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / "w2_nc_correlation.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot B — W2 × Muon metrics correlation scatter
# ===================================================================

def plot_w2_muon_correlation(
    configs: list[ConfigResult],
    M_values: list[int],
    W_default: int,
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """
    Plot B — W2 vs Ge-77 Recall and Precision at W=W_default for every M.

    Layout: 4 rows × 6 cols (Recall on left 3 cols, Precision on right 3 cols).
    Each cell = scatter (top) + residual (bottom).
    """
    w2_cfgs = [cfg for cfg in configs
               if cfg.w2 is not None and cfg.name != _ALL_VOXELS_NAME]
    if len(w2_cfgs) < 2:
        if verbose:
            print("  w2_muon_correlation: fewer than 2 configs have W2 — skipped.")
        return

    if color_map is None:
        colors_all = _get_colors(len(w2_cfgs))
        color_map  = {cfg.name: colors_all[i] for i, cfg in enumerate(w2_cfgs)}
    w2_arr     = np.array([cfg.w2 for cfg in w2_cfgs])
    labels     = [cfg.label for cfg in w2_cfgs]
    color_pts  = [color_map.get(cfg.name, "gray") for cfg in w2_cfgs]

    ncols_half = 3                                   # cols per metric
    nrows_M    = (len(M_values) + ncols_half - 1) // ncols_half

    # 2 matplotlib sub-rows per logical row, 2*ncols_half cols total
    total_rows = nrows_M * 2
    total_cols = ncols_half * 2

    fig = plt.figure(figsize=(total_cols * 4.5, total_rows * 3))
    fig.suptitle(
        f"W2 Homogeneity vs Ge-77 Recall / Precision  (W = {W_default})\n"
        "(OLS fit · 95 % CI · Pearson r · Spearman ρ)",
        fontsize=13,
    )

    for pi, M in enumerate(M_values):
        logical_col = pi % ncols_half
        logical_row = pi // ncols_half

        for mi, (metric_fn, metric_name, col_offset) in enumerate([
            (_recall,    "Recall",    0),
            (_precision, "Precision", ncols_half),
        ]):
            col = logical_col + col_offset
            scatter_row = logical_row * 2
            resid_row   = logical_row * 2 + 1

            ax_scatter = fig.add_subplot(
                total_rows, total_cols,
                scatter_row * total_cols + col + 1,
            )
            ax_resid = fig.add_subplot(
                total_rows, total_cols,
                resid_row * total_cols + col + 1,
                sharex=ax_scatter,
            )

            y_arr = np.array([metric_fn(cfg, M, W_default) for cfg in w2_cfgs])

            _regression_overlay(
                ax_scatter, ax_resid,
                w2_arr, y_arr,
                color_pts, labels,
                y_label=f"{metric_name} (M≥{M}, W≥{W_default})",
            )
            ax_scatter.set_title(f"{metric_name}  M≥{M}", fontsize=8)
            ax_scatter.set_ylim(-0.05, 1.05)
            ax_scatter.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
            )
            ax_resid.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%")
            )
            plt.setp(ax_scatter.get_xticklabels(), visible=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / "w2_muon_correlation.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot C — Correlation matrix (Pearson + Spearman)
# ===================================================================

def plot_w2_correlation_matrix(
    configs: list[ConfigResult],
    M_values: list[int],
    M_default: int,
    W_default: int,
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """
    Plot C — Pearson and Spearman correlation matrices.

    Variables: W2, NC_frac at each M, Recall and Precision at
    (M_default, W_default).  Two heatmap subplots side-by-side.
    """
    w2_cfgs = [cfg for cfg in configs
               if cfg.w2 is not None and cfg.name != _ALL_VOXELS_NAME]
    if len(w2_cfgs) < 3:
        if verbose:
            print("  w2_correlation_matrix: fewer than 3 configs have W2 — skipped.")
        return

    # Build variable matrix
    var_names = ["W2"] + [f"NC_M{M}" for M in M_values] + [
        f"Recall\nM{M_default}W{W_default}",
        f"Precision\nM{M_default}W{W_default}",
    ]
    data_rows = []
    for cfg in w2_cfgs:
        row = [cfg.w2]
        row += [_nc_frac(cfg, M) for M in M_values]
        row += [_recall(cfg, M_default, W_default),
                _precision(cfg, M_default, W_default)]
        data_rows.append(row)

    X = np.array(data_rows)   # shape (n_cfgs, n_vars)
    nv = len(var_names)

    pearson_mat  = np.full((nv, nv), np.nan)
    spearman_mat = np.full((nv, nv), np.nan)
    pval_p_mat   = np.full((nv, nv), np.nan)
    pval_s_mat   = np.full((nv, nv), np.nan)

    for i in range(nv):
        for j in range(nv):
            if i == j:
                pearson_mat[i, j] = spearman_mat[i, j] = 1.0
                pval_p_mat[i, j]  = pval_s_mat[i, j]  = 0.0
                continue
            if np.std(X[:, i]) == 0 or np.std(X[:, j]) == 0:
                continue  # constant column — leave as NaN
            try:
                r,   pr  = scipy_stats.pearsonr(X[:, i],  X[:, j])
                rho, prs = scipy_stats.spearmanr(X[:, i], X[:, j])
            except ValueError:
                continue
            pearson_mat[i, j]  = r
            spearman_mat[i, j] = rho
            pval_p_mat[i, j]   = pr
            pval_s_mat[i, j]   = prs

    fig, (ax_p, ax_s) = plt.subplots(
        1, 2,
        figsize=(max(14, nv * 1.1) * 2, max(10, nv * 1.0)),
    )
    fig.suptitle(
        f"Correlation Matrices  (n = {len(w2_cfgs)} configs)\n"
        f"Muon metrics at M={M_default}, W={W_default}",
        fontsize=12,
    )

    for ax, mat, pmat, title in [
        (ax_p, pearson_mat,  pval_p_mat,  "Pearson r"),
        (ax_s, spearman_mat, pval_s_mat,  "Spearman ρ"),
    ]:
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        for i in range(nv):
            for j in range(nv):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                p   = pmat[i, j]
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                txt = f"{val:+.2f}{sig}"
                tc  = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=tc, fontweight="bold")

        ax.set_xticks(range(nv))
        ax.set_yticks(range(nv))
        ax.set_xticklabels(var_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(var_names, fontsize=8)
        ax.set_title(title, fontsize=11)

    fig.text(0.5, 0.01, "* p<0.05   ** p<0.01",
             ha="center", fontsize=8, fontstyle="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = output_dir / "w2_correlation_matrix.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot D — Coverage profile colored by W2
# ===================================================================

def plot_w2_coverage_profile(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """
    Plot D — NC detection fraction vs M, one line per config colored by W2.

    Lines are colored on a continuous blue→red colormap (blue = low W2 =
    uniform; red = high W2 = clustered).  Configs without W2 are drawn
    in gray with a dashed line.  A colorbar on the right shows the W2
    scale.
    """
    # "all_voxels" is drawn as a distinct black reference line — not included
    # in the colormap scaling or any statistical computation.
    ref_cfg   = next((cfg for cfg in configs if cfg.name == _ALL_VOXELS_NAME), None)
    w2_cfgs   = [cfg for cfg in configs
                 if cfg.w2 is not None and cfg.name != _ALL_VOXELS_NAME]
    gray_cfgs = [cfg for cfg in configs
                 if cfg.w2 is None and cfg.name != _ALL_VOXELS_NAME]

    if not w2_cfgs:
        if verbose:
            print("  w2_coverage_profile: no configs have W2 — skipped.")
        return

    w2_vals = np.array([cfg.w2 for cfg in w2_cfgs])
    w2_min, w2_max = w2_vals.min(), w2_vals.max()
    cmap = plt.cm.coolwarm_r   # blue=low W2, red=high W2

    fig, ax = plt.subplots(figsize=(10, 6))

    norm = plt.Normalize(vmin=w2_min, vmax=w2_max if w2_max > w2_min else w2_min + 1)

    # All-voxels reference line (visual upper bound only)
    if ref_cfg is not None:
        ys_ref = [_nc_frac(ref_cfg, M) for M in M_values]
        ax.plot(M_values, ys_ref, color="black", linewidth=2.0,
                linestyle="--", alpha=0.7, label="All voxels (max. reference)", zorder=5)

    for cfg in gray_cfgs:
        ys = [_nc_frac(cfg, M) for M in M_values]
        ax.plot(M_values, ys, color="gray", linewidth=1.0,
                linestyle="--", alpha=0.6, label=cfg.label)

    for cfg in w2_cfgs:
        color = cmap(norm(cfg.w2))
        ys = [_nc_frac(cfg, M) for M in M_values]
        ax.plot(M_values, ys, color=color, linewidth=1.8,
                marker="o", markersize=4, label=cfg.label)
        # Label at rightmost point
        ax.annotate(
            cfg.label, xy=(M_values[-1], ys[-1]),
            xytext=(4, 0), textcoords="offset points",
            fontsize=6, color=color, va="center",
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Global W2 (mm)  [blue=uniform, red=clustered]", fontsize=9)

    ax.set_xlabel("Multiplicity threshold M", fontsize=11)
    ax.set_ylabel("NC detection fraction", fontsize=11)
    ax.set_title(
        "NC Coverage Profile Colored by W2 Homogeneity\n"
        "(systematic shift reveals W2–coverage relationship)",
        fontsize=12,
    )
    ax.set_xticks(M_values)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%")
    )
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "w2_coverage_profile.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot E — Spearman ρ vs M
# ===================================================================

def plot_w2_spearman_vs_m(
    configs: list[ConfigResult],
    M_values: list[int],
    W_default: int,
    output_dir: Path,
    verbose: bool = True,
    total_primaries: int = 0,
    W_values: list[int] | None = None,
) -> None:
    """
    Plot E — Spearman ρ between W2 and each metric as a function of M.

    Lines on one axes:
      - ρ(W2, NC_fraction_M)
      - ρ(W2, Recall at (M, W_default))
      - ρ(W2, Precision at (M, W_default))
      - ρ(W2, best FoM at M)  [only when total_primaries > 0]

    A shaded band marks the ±0.3 'weak correlation' zone.
    Markers are filled for p<0.05, hollow for p≥0.05.
    """
    w2_cfgs = [cfg for cfg in configs
               if cfg.w2 is not None and cfg.name != _ALL_VOXELS_NAME]
    if len(w2_cfgs) < 3:
        if verbose:
            print("  w2_spearman_vs_m: fewer than 3 configs have W2 — skipped.")
        return

    w2_arr = np.array([cfg.w2 for cfg in w2_cfgs])

    rho_nc   = []
    rho_rec  = []
    rho_prec = []
    rho_fom  = []
    p_nc   = []
    p_rec  = []
    p_prec = []
    p_fom  = []

    for M in M_values:
        nc_arr   = np.array([_nc_frac(cfg, M) for cfg in w2_cfgs])
        rec_arr  = np.array([_recall(cfg, M, W_default) for cfg in w2_cfgs])
        prec_arr = np.array([_precision(cfg, M, W_default) for cfg in w2_cfgs])

        r1, p1 = scipy_stats.spearmanr(w2_arr, nc_arr)
        r2, p2 = scipy_stats.spearmanr(w2_arr, rec_arr)
        r3, p3 = scipy_stats.spearmanr(w2_arr, prec_arr)

        rho_nc.append(r1);   p_nc.append(p1)
        rho_rec.append(r2);  p_rec.append(p2)
        rho_prec.append(r3); p_prec.append(p3)

        if total_primaries > 0 and W_values:
            fom_arr = []
            for cfg in w2_cfgs:
                best = np.nan
                for W in W_values:
                    cm = cfg.confusion.get((M, W))
                    if cm is None:
                        continue
                    v = calc_fom_confusion(
                        cm["TP"], cm["FP"], cm["FN"], total_primaries,
                        tp_ge77_nc_counts=cm.get("tp_ge77_nc_counts"),
                        fn_ge77_nc_counts=cm.get("fn_ge77_nc_counts"),
                    )
                    if np.isfinite(v) and (np.isnan(best) or v > best):
                        best = v
                fom_arr.append(best)
            fom_arr = np.array(fom_arr)
            mask = np.isfinite(fom_arr)
            if mask.sum() >= 3:
                r4, p4 = scipy_stats.spearmanr(w2_arr[mask], fom_arr[mask])
            else:
                r4, p4 = np.nan, np.nan
            rho_fom.append(r4)
            p_fom.append(p4)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.array(M_values)
    ax.axhline(0,  color="black", linewidth=0.8)
    ax.axhspan(-0.3, 0.3, color="gray", alpha=0.08, label="weak |ρ|<0.3")

    series = [
        (rho_nc,   p_nc,   "NC fraction",               "#1f77b4", "o"),
        (rho_rec,  p_rec,  f"Recall (W={W_default})",   "#d62728", "s"),
        (rho_prec, p_prec, f"Precision (W={W_default})", "#2ca02c", "^"),
    ]
    if total_primaries > 0 and rho_fom:
        series.append((rho_fom, p_fom, "Best FoM (over W)", "#ff7f0e", "D"))

    for rhos, ps, label, color, marker in series:
        rhos = np.array(rhos)
        ps   = np.array(ps)
        sig_mask  = ps < _P_3SIGMA
        nsig_mask = ~sig_mask & np.isfinite(rhos)
        # Filled markers for 3σ significant, hollow for non-significant
        ax.plot(x, rhos, color=color, linewidth=1.5, label=label)
        if sig_mask.any():
            ax.scatter(x[sig_mask],  rhos[sig_mask],
                       color=color, s=60, marker=marker,
                       zorder=4)
        if nsig_mask.any():
            ax.scatter(x[nsig_mask], rhos[nsig_mask],
                       facecolors="none", edgecolors=color,
                       s=60, marker=marker, linewidth=1.2,
                       zorder=4)

    ax.set_xlabel("Multiplicity threshold M", fontsize=11)
    ax.set_ylabel("Spearman ρ  (W2 vs metric)", fontsize=11)
    ax.set_title(
        f"Spearman Correlation between W2 and Coverage Metrics vs M\n"
        f"(filled = 3σ significant | W_default = {W_default})",
        fontsize=12,
    )
    ax.set_xticks(M_values)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "w2_spearman_vs_m.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot F — NC coverage vs Ge-77 Recall correlation scatter
# ===================================================================

def plot_nc_recall_correlation(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    W_fixed: int = 1,
    verbose: bool = True,
) -> None:
    """
    Plot F — NC detection fraction vs Ge-77 Recall at W=W_fixed for each M.

    Each point is one PMT configuration (all_voxels reference excluded).
    Layout: one panel per M value, each with a scatter+OLS (top) and
    residual (bottom) subplot sharing the x-axis.
    Pearson r and Spearman ρ with p-values are annotated per panel.
    """
    plot_cfgs = [cfg for cfg in configs if cfg.name != _ALL_VOXELS_NAME]
    if len(plot_cfgs) < 2:
        if verbose:
            print("  nc_recall_correlation: fewer than 2 configs — skipped.")
        return

    if color_map is None:
        _pal = _get_colors(len(plot_cfgs))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(plot_cfgs)}

    color_pts = [color_map.get(cfg.name, "gray") for cfg in plot_cfgs]
    labels    = [cfg.label for cfg in plot_cfgs]

    ncols = min(len(M_values), 4)
    nrows = (len(M_values) + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 5, nrows * 6))
    fig.suptitle(
        f"NC Detection Fraction vs Ge-77 Recall  (W = {W_fixed})\n"
        "(OLS fit · 95 % CI · Pearson r · Spearman ρ)",
        fontsize=13,
    )

    for pi, M in enumerate(M_values):
        col = pi % ncols
        row = pi // ncols

        ax_scatter = fig.add_subplot(
            nrows * 2, ncols, row * 2 * ncols + col + 1,
        )
        ax_resid = fig.add_subplot(
            nrows * 2, ncols, (row * 2 + 1) * ncols + col + 1,
            sharex=ax_scatter,
        )

        x_arr = np.array([_nc_frac(cfg, M) for cfg in plot_cfgs])
        y_arr = np.array([_recall(cfg, M, W_fixed) for cfg in plot_cfgs])

        _regression_overlay(
            ax_scatter, ax_resid,
            x_arr, y_arr,
            color_pts, labels,
            y_label=f"Recall (M≥{M}, W≥{W_fixed})",
            x_label=f"NC fraction (M≥{M})",
        )
        ax_scatter.set_title(f"M ≥ {M}", fontsize=9)
        ax_scatter.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%")
        )
        ax_scatter.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%")
        )
        ax_resid.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%")
        )
        plt.setp(ax_scatter.get_xticklabels(), visible=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / f"nc_recall_correlation_W{W_fixed}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 20 — Recall at W=1 across M thresholds
# ===================================================================

def plot_recall_w1_vs_m(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """Line plot: Ge-77 Recall at W=1 for all configs across M thresholds.

    Addresses: 'if recall is best at M=1, is it also best for larger M?'
    W is kept fixed at 1 throughout.  The all_voxels reference is excluded.
    """
    plot_cfgs = [cfg for cfg in _sorted_by_w2(configs)
                 if cfg.name != _ALL_VOXELS_NAME]
    if not plot_cfgs:
        return

    if color_map is None:
        _pal = _get_colors(len(plot_cfgs))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(plot_cfgs)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for cfg in plot_cfgs:
        recalls = [_recall(cfg, M, 1) for M in M_values]
        ax.plot(M_values, recalls, marker="o", linewidth=1.5, markersize=5,
                color=color_map.get(cfg.name, "gray"),
                label=_config_label(cfg))

    ax.set_xlabel("M — minimum firing PMTs per NC", fontsize=11)
    ax.set_ylabel("Ge-77 Recall  (W = 1)", fontsize=11)
    ax.set_title(
        "Ge-77 Muon Recall at W=1 across M Thresholds\n"
        "(does the best-recall config at M=1 remain the best at higher M?)",
        fontsize=12,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_xticks(M_values)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "recall_w1_vs_m.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 21 — Recall at each config's FoM-optimal (M, W)
# ===================================================================

def plot_recall_at_best_fom(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
    min_m: int | None = None,
) -> None:
    """Two-panel horizontal bar: Recall (left) and Precision (right) at FoM-optimal (M, W).

    Each bar is annotated with the optimal (M, W) pair and metric value.
    The all_voxels reference is excluded.
    ``min_m`` restricts the search to M >= min_m.
    """
    plot_cfgs = [cfg for cfg in _sorted_by_w2(configs)
                 if cfg.name != _ALL_VOXELS_NAME]
    if not plot_cfgs:
        return
    if color_map is None:
        _pal = _get_colors(len(plot_cfgs))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(plot_cfgs)}

    m_search = [M for M in M_values if (min_m is None or M >= min_m)]
    if not m_search:
        return

    recalls:    list[float] = []
    precisions: list[float] = []
    opt_mw_labels: list[str] = []
    for cfg in plot_cfgs:
        grid = _fom_grid(cfg, m_search, W_values, total_primaries)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best_mw = max(valid, key=valid.__getitem__)
            recalls.append(_recall(cfg, best_mw[0], best_mw[1]))
            precisions.append(_precision(cfg, best_mw[0], best_mw[1]))
            opt_mw_labels.append(f"M{best_mw[0]}W{best_mw[1]}")
        else:
            recalls.append(float("nan"))
            precisions.append(float("nan"))
            opt_mw_labels.append("N/A")

    labels = [_config_label(cfg) for cfg in plot_cfgs]
    colors = [color_map.get(cfg.name, "gray") for cfg in plot_cfgs]
    y = np.arange(len(plot_cfgs))

    fig, (ax_rec, ax_prec) = plt.subplots(
        1, 2, figsize=(16, max(4, len(plot_cfgs) * 0.55))
    )

    for ax, vals, metric in [
        (ax_rec,  recalls,    "Recall"),
        (ax_prec, precisions, "Precision"),
    ]:
        bars  = ax.barh(y, vals, color=colors, height=0.6)
        x_max = max((v for v in vals if np.isfinite(v)), default=1.0)
        for bar, mw, val in zip(bars, opt_mw_labels, vals):
            if np.isfinite(val):
                ax.text(
                    val + 0.01 * x_max,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%  [{mw}]",
                    va="center", ha="left", fontsize=8,
                )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(right=x_max * 1.4)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_title(metric, fontsize=11)
        if min_m is not None:
            ax.set_xlabel(f"Ge-77 {metric} at FoM-optimal (M≥{min_m}, W)", fontsize=11)
        else:
            ax.set_xlabel(f"Ge-77 {metric} at FoM-optimal (M, W)", fontsize=11)

    if min_m is not None:
        fig.suptitle(
            f"Ge-77 Recall & Precision at FoM-Optimal (M≥{min_m}, W)\n"
            f"(brackets: (M, W) maximising FoM with M≥{min_m})",
            fontsize=12, y=1.01,
        )
        fname = f"21_recall_precision_at_best_fom_Mge{min_m}.png"
    else:
        fig.suptitle(
            "Ge-77 Recall & Precision at FoM-Optimal (M, W)\n"
            "(brackets: (M, W) maximising FoM for each setup)",
            fontsize=12, y=1.01,
        )
        fname = "21_recall_precision_at_best_fom.png"

    fig.tight_layout()
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 22 — NC fraction (M=1) vs Recall at best FoM (scatter + OLS)
# ===================================================================

def plot_nc_recall_at_best_fom(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """Scatter: NC detection fraction (M=1) vs Recall at each config's FoM-optimal (M, W).

    OLS regression + 95 % CI overlay.  The all_voxels reference is excluded.
    """
    plot_cfgs = [cfg for cfg in configs if cfg.name != _ALL_VOXELS_NAME]
    if len(plot_cfgs) < 2:
        if verbose:
            print("  nc_recall_at_best_fom: fewer than 2 configs — skipped.")
        return

    if color_map is None:
        _pal = _get_colors(len(plot_cfgs))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(plot_cfgs)}
    color_pts = [color_map.get(cfg.name, "gray") for cfg in plot_cfgs]
    labels = [cfg.label for cfg in plot_cfgs]

    nc_fracs = np.array([_nc_frac(cfg, 1) for cfg in plot_cfgs])
    recalls_best: list[float] = []
    for cfg in plot_cfgs:
        grid = _fom_grid(cfg, M_values, W_values, total_primaries)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best_mw = max(valid, key=valid.__getitem__)
            recalls_best.append(_recall(cfg, best_mw[0], best_mw[1]))
        else:
            recalls_best.append(float("nan"))
    recalls_arr = np.array(recalls_best)

    mask = np.isfinite(recalls_arr)
    if mask.sum() < 2:
        if verbose:
            print("  nc_recall_at_best_fom: fewer than 2 finite values — skipped.")
        return

    fig, (ax_scatter, ax_resid) = plt.subplots(
        2, 1, figsize=(8, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.suptitle(
        "NC Detection Fraction (M=1) vs Ge-77 Recall at FoM-Optimal (M, W)\n"
        "(OLS fit · 95 % CI · Pearson r · Spearman ρ)",
        fontsize=12,
    )

    _regression_overlay(
        ax_scatter, ax_resid,
        nc_fracs[mask], recalls_arr[mask],
        [c for c, m in zip(color_pts, mask) if m],
        [lb for lb, m in zip(labels, mask) if m],
        y_label="Recall at FoM-optimal (M, W)",
        x_label="NC detection fraction (M=1)",
    )
    ax_scatter.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax_resid.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax_resid.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    plt.setp(ax_scatter.get_xticklabels(), visible=False)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = output_dir / "nc_recall_at_best_fom.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 23 — Ge-77 survival at FoM-optimal (M, W) per config
# ===================================================================

def plot_ge77_survival_at_best_fom(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """Scatter: Ge-77 survival (Recall) vs setup index at each config's FoM-optimal (M, W).

    Ge77 survival = TP / (TP + FN) = Recall.
    The all_voxels reference is excluded.
    """
    plot_cfgs = [cfg for cfg in configs if cfg.name != _ALL_VOXELS_NAME]
    if not plot_cfgs:
        return
    if color_map is None:
        _pal = _get_colors(len(plot_cfgs))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(plot_cfgs)}

    values: list[float] = []
    mw_labels: list[str] = []
    for cfg in plot_cfgs:
        grid = _fom_grid(cfg, M_values, W_values, total_primaries)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best_mw = max(valid, key=valid.__getitem__)
            values.append(_recall(cfg, best_mw[0], best_mw[1]))
            mw_labels.append(f"M{best_mw[0]}W{best_mw[1]}")
        else:
            values.append(float("nan"))
            mw_labels.append("N/A")

    labels = [_config_label(cfg) for cfg in plot_cfgs]
    colors = [color_map.get(cfg.name, "gray") for cfg in plot_cfgs]
    x = np.arange(len(plot_cfgs))

    fig, ax = plt.subplots(figsize=(max(6, len(plot_cfgs) * 0.9), 5))
    for i, (c, val, mw) in enumerate(zip(colors, values, mw_labels)):
        if np.isfinite(val):
            ax.scatter([i], [val], color=c, s=80, zorder=3)
            ax.annotate(
                mw, xy=(i, val), xytext=(0, 8),
                textcoords="offset points", fontsize=7, ha="center", color=c,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Setup", fontsize=11)
    ax.set_ylabel("Ge-77 Survival  (Recall = TP / (TP+FN))", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.set_ylim(bottom=0)
    ax.set_title(
        "Ge-77 Muon Survival at Each Config's FoM-Optimal (M, W)\n"
        "(labels show the (M, W) that maximises FoM for that config)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = output_dir / "ge77_survival_at_best_fom.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 24 — Signal survival at FoM-optimal (M, W) per config
# ===================================================================

def plot_signal_survival_at_best_fom(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """Scatter: signal survival 1-(TP+FP)/(TP+FP+TN+FN) vs setup index at each config's FoM-optimal (M, W).

    Signal survival = (TN+FN) / (TP+FP+TN+FN).
    The all_voxels reference is excluded.
    """
    plot_cfgs = [cfg for cfg in configs if cfg.name != _ALL_VOXELS_NAME]
    if not plot_cfgs:
        return
    if color_map is None:
        _pal = _get_colors(len(plot_cfgs))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(plot_cfgs)}

    values: list[float] = []
    mw_labels: list[str] = []
    for cfg in plot_cfgs:
        grid = _fom_grid(cfg, M_values, W_values, total_primaries)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best_mw = max(valid, key=valid.__getitem__)
            values.append(_signal_survival(cfg, best_mw[0], best_mw[1]))
            mw_labels.append(f"M{best_mw[0]}W{best_mw[1]}")
        else:
            values.append(float("nan"))
            mw_labels.append("N/A")

    labels = [_config_label(cfg) for cfg in plot_cfgs]
    colors = [color_map.get(cfg.name, "gray") for cfg in plot_cfgs]
    x = np.arange(len(plot_cfgs))

    fig, ax = plt.subplots(figsize=(max(6, len(plot_cfgs) * 0.9), 5))
    for i, (c, val, mw) in enumerate(zip(colors, values, mw_labels)):
        if np.isfinite(val):
            ax.scatter([i], [val], color=c, s=80, zorder=3)
            ax.annotate(
                mw, xy=(i, val), xytext=(0, 8),
                textcoords="offset points", fontsize=7, ha="center", color=c,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Setup", fontsize=11)
    ax.set_ylabel("Signal Survival = (TN+FN) / (TP+FP+TN+FN)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.3f}%"))
    ax.set_ylim(bottom=0)
    ax.set_title(
        "Signal Survival at Each Config's FoM-Optimal (M, W)\n"
        "(1 − veto fraction; labels show optimal (M, W))",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = output_dir / "signal_survival_at_best_fom.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Text output — survival table
# ===================================================================

def write_survival_table(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """Write survival_at_best_fom.txt: Ge77 and signal survival at FoM-optimal (M,W).

    Ge77 survival   = TP / (TP+FN)          = Recall
    Signal survival = (TP+FP) / total_primary_muons
    The all_voxels reference is excluded.
    """
    plot_cfgs = [cfg for cfg in configs if cfg.name != _ALL_VOXELS_NAME]
    out_path = output_dir / "survival_at_best_fom.txt"
    W = 80
    with open(out_path, "w") as f:
        f.write("Survival at FoM-Optimal (M, W) per Config\n")
        f.write("=" * W + "\n\n")
        f.write("  Ge77 survival   = TP / (TP + FN)            = Recall\n")
        f.write("  Signal survival = (TP + FP) / all_muons\n")
        f.write(f"  all_muons       = {total_primaries:,}\n\n")

        hdr = (
            f"  {'Config':<25}  {'(M,W)':>8}  "
            f"{'Ge77 Survival':>15}  {'Signal Survival':>16}  "
            f"{'TP':>8}  {'FP':>8}  {'FN':>8}\n"
        )
        f.write(hdr)
        f.write("  " + "-" * (len(hdr) - 3) + "\n")

        for cfg in plot_cfgs:
            grid = _fom_grid(cfg, M_values, W_values, total_primaries)
            valid = {k: v for k, v in grid.items() if np.isfinite(v)}
            if not valid:
                f.write(f"  {cfg.label:<25}  {'N/A':>8}  {'N/A':>15}  {'N/A':>16}\n")
                continue
            best_mw = max(valid, key=valid.__getitem__)
            M_b, W_b = best_mw
            cm = cfg.confusion.get((M_b, W_b), {})
            tp = cm.get("TP", 0)
            fp = cm.get("FP", 0)
            fn = cm.get("FN", 0)
            ge77_surv = _recall(cfg, M_b, W_b)
            sig_surv  = _signal_survival(cfg, M_b, W_b)
            f.write(
                f"  {cfg.label:<25}  {f'M{M_b}W{W_b}':>8}  "
                f"{ge77_surv*100:>14.2f}%  {sig_surv*100:>15.4f}%  "
                f"{tp:>8,}  {fp:>8,}  {fn:>8,}\n"
            )
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 01b — NC coverage rank Spearman stability vs M
# ===================================================================

def plot_nc_rank_spearman(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
    verbose: bool = True,
    M_ref: int = 1,
) -> None:
    """Spearman rank stability of NC coverage rankings vs M (vs reference M_ref).

    Filled markers = 3σ significant.  Saved as 01b_nc_rank_spearman.png.
    """
    plot_cfgs = [cfg for cfg in configs if cfg.name != _ALL_VOXELS_NAME]
    if len(plot_cfgs) < 3:
        if verbose:
            print("  [SKIP] 01b_nc_rank_spearman: fewer than 3 configs.")
        return
    if M_ref not in M_values:
        M_ref = M_values[0]

    fracs = {M: np.array([_nc_frac(cfg, M) for cfg in plot_cfgs]) for M in M_values}
    ref_ranks = scipy_stats.rankdata(-fracs[M_ref])

    rho_vs_ref, p_vs_ref = [], []
    for M in M_values:
        rho, p = scipy_stats.spearmanr(ref_ranks, scipy_stats.rankdata(-fracs[M]))
        rho_vs_ref.append(rho)
        p_vs_ref.append(p)

    rho_vs_ref = np.array(rho_vs_ref)
    p_vs_ref   = np.array(p_vs_ref)
    x = np.array(M_values)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle=":")
    ax.axvline(M_ref, color="black", linewidth=1.0, linestyle="--",
               label=f"Reference M = {M_ref}", alpha=0.5)

    sig  = p_vs_ref < _P_3SIGMA
    nsig = ~sig & np.isfinite(rho_vs_ref)
    color = "#1f77b4"
    ax.plot(x, rho_vs_ref, color=color, linewidth=1.8,
            label=f"ρ(rank@M={M_ref}, rank@M)")
    if sig.any():
        ax.scatter(x[sig], rho_vs_ref[sig], color=color, s=65, marker="o", zorder=4)
    if nsig.any():
        ax.scatter(x[nsig], rho_vs_ref[nsig], facecolors="none", edgecolors=color,
                   s=65, marker="o", linewidth=1.3, zorder=4)

    ax.set_xlabel("Multiplicity threshold M", fontsize=11)
    ax.set_ylabel("Spearman ρ of NC coverage rankings", fontsize=11)
    ax.set_title(
        f"NC Coverage Rank Stability vs M\n"
        f"(filled = 3σ significant; reference M = {M_ref})",
        fontsize=12,
    )
    ax.set_xticks(M_values)
    ax.set_ylim(-0.15, 1.15)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "01b_nc_rank_spearman.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 04 — Ge77 muon detection funnel
# ===================================================================

def plot_ge77_muon_overview(
    configs: list[ConfigResult],
    M_default: int,
    W_default: int,
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
    total_primary_muons: int = 0,
) -> None:
    """Two-panel figure: Ge77 TP bar grouped per setup + Δ from reference.

    SSD data does not have photon timing info, so only the TP (Detected)
    category is shown.  Saved as 04_ge77_muon_overview.png.
    """
    ordered = _sorted_by_w2(configs)
    n = len(ordered)
    if color_map is None:
        _pal = _get_colors(n)
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(ordered)}
    colors = [color_map.get(cfg.name, "gray") for cfg in ordered]
    ref_cfg = next((c for c in ordered if c.label == "Baseline"), ordered[0])

    cat_labels = [
        f"Detected\n(M≥{M_default}, W≥{W_default})\n= Recall TP",
    ]
    x = np.arange(len(cat_labels))
    width = min(0.8 / n, 0.30)

    fig, (ax_abs, ax_delta) = plt.subplots(
        1, 2, figsize=(22, 9), gridspec_kw={"width_ratios": [3, 2]},
    )

    _ge77_total = int(np.sum(ordered[0].confusion.get((M_default, W_default), {}).get("TP", 0))
                      + np.sum(ordered[0].confusion.get((M_default, W_default), {}).get("FN", 0)))

    for i, (cfg, c) in enumerate(zip(ordered, colors)):
        cm = cfg.confusion.get((M_default, W_default), {})
        vals   = [cm.get("TP", 0)]
        offset = (i - (n - 1) / 2) * width
        bars   = ax_abs.bar(x + offset, vals, width, label=_config_label(cfg), color=c)
        for bar, val in zip(bars, vals):
            pct = 100.0 * val / max(_ge77_total, 1)
            ax_abs.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=6, rotation=90, fontweight="bold",
            )

    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(cat_labels, fontsize=9)
    ax_abs.set_ylabel("Number of Ge77 muons", fontsize=11)
    ax_abs.set_title(
        f"Ge-77 Muon Detection  [M={M_default}, W={W_default}]\n"
        "(Detected = ≥W NCs with ≥M firing PMTs in [1 µs, 200 µs])",
        fontsize=11,
    )
    ax_abs.legend(fontsize=9)
    ax_abs.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    _all_muons = total_primary_muons
    _note_lines = f"100 % = {_ge77_total:,} Ge77 muons"
    if _all_muons > 0 and _all_muons != _ge77_total:
        _note_lines += f"\n(total primary muons: {_all_muons:,})"
    ax_abs.text(
        0.01, 0.99, _note_lines,
        transform=ax_abs.transAxes, fontsize=9, va="top", ha="left",
        color="dimgray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.85),
    )

    if n < 2:
        ax_delta.text(0.5, 0.5, "Single setup —\nno Δ to display",
                      transform=ax_delta.transAxes, ha="center", va="center",
                      fontsize=11, color="gray")
        ax_delta.set_axis_off()
    else:
        ref_cm = ref_cfg.confusion.get((M_default, W_default), {})
        ref_val = ref_cm.get("TP", 0)
        ref_total = int(ref_val + ref_cm.get("FN", 0))
        non_ref = [(cfg, color_map.get(cfg.name, "gray"))
                   for cfg in ordered if cfg is not ref_cfg]
        n_non_ref = len(non_ref)
        bar_h = 0.8 / max(n_non_ref, 1)
        y_base = np.array([0.0])

        for j, (cfg, c) in enumerate(non_ref):
            val   = cfg.confusion.get((M_default, W_default), {}).get("TP", 0)
            delta = val - ref_val
            offset = (j - (n_non_ref - 1) / 2) * bar_h
            bars = ax_delta.barh(y_base + offset, [delta], bar_h * 0.88,
                                  label=_config_label(cfg), color=c)
            for bar in bars:
                pct  = 100.0 * delta / max(ref_total, 1)
                sign = "+" if delta >= 0 else ""
                if delta == 0:
                    ax_delta.text(0, bar.get_y() + bar.get_height() / 2,
                                   " =ref", va="center", ha="left", fontsize=6,
                                   color=c, fontstyle="italic")
                    continue
                txt = f"{sign}{delta:,}\n({sign}{pct:.1f}%)"
                pad = max(abs(delta) * 0.02, 1)
                ax_delta.text(
                    delta + (pad if delta >= 0 else -pad),
                    bar.get_y() + bar.get_height() / 2,
                    txt, va="center", ha="left" if delta >= 0 else "right",
                    fontsize=7, color=c, fontweight="bold",
                )

        ax_delta.axvline(0, color="black", linewidth=0.9)
        ax_delta.set_yticks(y_base)
        ax_delta.set_yticklabels(cat_labels, fontsize=9)
        ax_delta.set_xlabel(f"Δ muon count vs reference ({ref_cfg.label})", fontsize=11)
        ax_delta.set_title(f"Δ from reference: {ref_cfg.label}", fontsize=10)
        ax_delta.legend(fontsize=9, title="vs reference")
        ax_delta.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):+,}"))
        ax_delta.grid(True, axis="x", alpha=0.3)
        ax_delta.invert_yaxis()

    fig.tight_layout()
    out_path = output_dir / "04_ge77_muon_overview.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 06 — Confusion bar (2×2 TP/FN/TN/FP)
# ===================================================================

def plot_confusion_bar(
    configs: list[ConfigResult],
    M_default: int,
    W_default: int,
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
    total_primaries: int = 0,
) -> None:
    """Four sub-figures (2×2): TP, FN, TN, FP at (M_default, W_default).

    Values as % of total_primaries.  Saved as 06_confusion_bar.png.
    """
    ordered = _sorted_by_w2(configs)
    n = len(ordered)
    if color_map is None:
        _pal = _get_colors(n)
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(ordered)}
    categories = [
        ("True Positive\n(Ge77 → classified)", "TP"),
        ("False Negative\n(Ge77 → missed)", "FN"),
        ("True Negative\n(non-Ge77 → correct)", "TN"),
        ("False Positive\n(non-Ge77 → misclass.)", "FP"),
    ]
    colors = [color_map.get(cfg.name, "gray") for cfg in ordered]
    _denom = max(total_primaries, 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()

    for ax, (cat_label, key) in zip(axes_flat, categories):
        all_pcts = []
        for i, (cfg, c) in enumerate(zip(ordered, colors)):
            conf = cfg.confusion.get((M_default, W_default), {})
            tn_actual = total_primaries - conf.get("TP", 0) - conf.get("FP", 0) - conf.get("FN", 0)
            counts = {"TP": conf.get("TP", 0), "FN": conf.get("FN", 0),
                      "TN": tn_actual, "FP": conf.get("FP", 0)}
            val  = counts[key]
            pct = 100.0 * val / _denom
            all_pcts.append(pct)
            bar = ax.bar([i], [pct], 0.6, label=cfg.label, color=c)
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2,
                bar[0].get_height(),
                f"{val:,}\n({pct:.3f}%)",
                ha="center", va="bottom", fontsize=7, rotation=90,
            )

        vmin = min(all_pcts) if all_pcts else 0.0
        vmax = max(all_pcts) if all_pcts else 1.0
        span = vmax - vmin
        if span > 0:
            margin = span * 0.4
            ax.set_ylim(max(0.0, vmin - margin), vmax + margin)
        else:
            ax.set_ylim(0.0, vmax * 1.2 if vmax > 0 else 1.0)

        ax.set_xticks(range(n))
        ax.set_xticklabels([cfg.label for cfg in ordered], rotation=45,
                           ha="right", fontsize=9)
        ax.set_ylabel("% of total simulated muons", fontsize=10)
        ax.set_title(cat_label, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}%"))
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Ge-77 Muon Classification (W≥{W_default}, M≥{M_default})\n"
        f"(values as % of {total_primaries:,} total simulated muons)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = output_dir / "06_confusion_bar.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 07b — TP vs FP scatter across all (M, W)
# ===================================================================

def plot_tp_fp_scatter(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """Scatter: TP vs FP across all (M, W) combinations.  Saved as 07b_tp_fp_scatter.png."""
    if color_map is None:
        _pal = _get_colors(len(configs))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(configs)}

    fig, ax = plt.subplots(figsize=(9, 7))

    all_tp: list[float] = []
    all_fp: list[float] = []

    for cfg in configs:
        c = color_map.get(cfg.name, "gray")
        tp_vals, fp_vals = [], []
        for M in M_values:
            for W in W_values:
                cm = cfg.confusion.get((M, W))
                if cm is None:
                    continue
                tp_vals.append(float(cm["TP"]))
                fp_vals.append(float(cm["FP"]))
        if tp_vals:
            ax.scatter(tp_vals, fp_vals, color=c, s=10, alpha=0.35, zorder=3)
            all_tp.extend(tp_vals)
            all_fp.extend(fp_vals)

    all_tp_arr = np.array(all_tp)
    all_fp_arr = np.array(all_fp)

    if len(all_tp_arr) >= 3:
        r_val,   p_r   = scipy_stats.pearsonr(all_tp_arr, all_fp_arr)
        rho_val, p_rho = scipy_stats.spearmanr(all_tp_arr, all_fp_arr)
        ax.text(
            0.05, 0.95,
            f"Pearson  r = {r_val:+.3f}   (p = {p_r:.2g})\n"
            f"Spearman ρ = {rho_val:+.3f}   (p = {p_rho:.2g})",
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color_map.get(cfg.name, "gray"),
                       markersize=7, label=cfg.label)
            for cfg in configs
        ],
        fontsize=9, loc="upper right",
    )
    ax.set_xlabel("True Positives (TP) — Ge-77 muons correctly classified", fontsize=11)
    ax.set_ylabel("False Positives (FP) — non-Ge-77 muons misclassified", fontsize=11)
    ax.set_title(
        "TP vs FP — all (M, W) combinations\n"
        "(each point = one (M, W) per setup; colour = setup)",
        fontsize=12,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "07b_tp_fp_scatter.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 08 — W2 vs NC coverage (single panel, multi-M)
# ===================================================================

def plot_w2_nc_scatter_single(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
    M_fixed: int = 1,
    W_fixed: int = 1,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """Single-panel W2 vs NC coverage fraction for multiple M thresholds.

    Saved as 08_w2_nc_scatter_M{M_fixed:02d}_W{W_fixed:02d}.png.
    """
    w2_cfgs = [cfg for cfg in configs if cfg.w2 is not None
               and cfg.name != _ALL_VOXELS_NAME]
    if len(w2_cfgs) < 2:
        if verbose:
            print("  [SKIP] 08_w2_nc_scatter: fewer than 2 configs have W2.")
        return

    if color_map is None:
        _pal = _get_colors(len(w2_cfgs))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(w2_cfgs)}
    w2_vals = np.array([cfg.w2 for cfg in w2_cfgs])

    panel_ms = sorted({M for M in [1, 2, 4, 5, 10] if M in M_values})
    m_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(panel_ms)))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("W2 Homogeneity vs NC Coverage", fontsize=12, fontweight="bold")

    for sM, mc in zip(panel_ms, m_colors):
        fracs = np.array([_nc_frac(cfg, sM) for cfg in w2_cfgs])
        ax.scatter(w2_vals, fracs, color=mc, s=55, zorder=3, label=f"M={sM}")
        for cfg, w2v, frac in zip(w2_cfgs, w2_vals, fracs):
            ax.annotate(cfg.label, xy=(w2v, frac), xytext=(4, 2),
                        textcoords="offset points", fontsize=6, color=mc)

    ax.set_xlabel("Global W2 (mm) — lower = more uniform", fontsize=10)
    ax.set_ylabel("NC detection fraction", fontsize=10)
    ax.set_title("W2 vs NC Coverage", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.legend(title="M threshold", fontsize=9, title_fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / f"08_w2_nc_scatter_M{M_fixed:02d}_W{W_fixed:02d}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 09 — W2 vs Recall at best FoM (global W2 only, two M constraints)
# ===================================================================

def _plot_09_w2_recall_bestfom(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: Path,
    min_m: int | None,
    total_primaries: int,
    color_map: dict[str, str] | None,
    verbose: bool,
) -> None:
    """W2 vs Recall at best-FoM scatter.  One file per M constraint."""
    w2_cfgs = [cfg for cfg in configs
               if cfg.w2 is not None and cfg.name != _ALL_VOXELS_NAME]
    if len(w2_cfgs) < 2:
        return

    if color_map is None:
        _pal = _get_colors(len(w2_cfgs))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(w2_cfgs)}

    m_search = [M for M in M_values if (min_m is None or M >= min_m)]
    if not m_search:
        return

    w2_arr = np.array([cfg.w2 for cfg in w2_cfgs])
    recalls: list[float] = []
    for cfg in w2_cfgs:
        grid  = _fom_grid(cfg, m_search, W_values, total_primaries)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best = max(valid, key=valid.__getitem__)
            recalls.append(_recall(cfg, best[0], best[1]))
        else:
            recalls.append(float("nan"))
    rec_arr   = np.array(recalls)
    colors_pt = [color_map.get(cfg.name, "gray") for cfg in w2_cfgs]
    labels_pt = [cfg.label for cfg in w2_cfgs]

    m_tag  = f"M≥{min_m}" if min_m is not None else "all M"
    m_file = f"_Mge{min_m}" if min_m is not None else ""

    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter_corr_panel(
        ax, w2_arr, rec_arr, colors_pt, labels_pt,
        x_label="W2_global (mm)  [lower = more uniform]",
        y_label=f"Ge-77 Recall  ({m_tag})",
        title=f"W2_global vs Recall at Best FoM  ({m_tag})",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    fig.tight_layout()
    out_path = output_dir / f"09_w2_global_recall_bestfom{m_file}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


def plot_w2_recall_best_fom_variants(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: Path,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """Plot 09: W2_global vs Recall at best-FoM for two M constraints (all M and M≥6)."""
    for min_m in (None, 6):
        _plot_09_w2_recall_bestfom(
            configs, M_values, W_values, output_dir, min_m,
            total_primaries, color_map, verbose,
        )


# ===================================================================
# Plot 09b — W2 vs FoM at best FoM (global W2 only, two M constraints)
# ===================================================================

def _plot_09b_w2_fom_bestfom(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: Path,
    min_m: int | None,
    total_primaries: int,
    color_map: dict[str, str] | None,
    verbose: bool,
) -> None:
    """W2 vs best FoM scatter.  One file per M constraint."""
    w2_cfgs = [cfg for cfg in configs
               if cfg.w2 is not None and cfg.name != _ALL_VOXELS_NAME]
    if len(w2_cfgs) < 2:
        return

    if color_map is None:
        _pal = _get_colors(len(w2_cfgs))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(w2_cfgs)}

    m_search = [M for M in M_values if (min_m is None or M >= min_m)]
    if not m_search:
        return

    w2_arr = np.array([cfg.w2 for cfg in w2_cfgs])
    foms: list[float] = []
    for cfg in w2_cfgs:
        grid  = _fom_grid(cfg, m_search, W_values, total_primaries)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        foms.append(max(valid.values()) if valid else float("nan"))
    fom_arr   = np.array(foms)
    colors_pt = [color_map.get(cfg.name, "gray") for cfg in w2_cfgs]
    labels_pt = [cfg.label for cfg in w2_cfgs]

    m_tag  = f"M≥{min_m}" if min_m is not None else "all M"
    m_file = f"_Mge{min_m}" if min_m is not None else ""

    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter_corr_panel(
        ax, w2_arr, fom_arr, colors_pt, labels_pt,
        x_label="W2_global (mm)  [lower = more uniform]",
        y_label=f"Best FoM  ({m_tag})",
        title=f"W2_global vs Best FoM  ({m_tag})",
    )
    fig.tight_layout()
    out_path = output_dir / f"09b_w2_global_fom_bestfom{m_file}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


def plot_w2_fom_best_variants(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: Path,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """Plot 09b: W2_global vs best FoM for two M constraints (all M and M≥6)."""
    for min_m in (None, 6):
        _plot_09b_w2_fom_bestfom(
            configs, M_values, W_values, output_dir, min_m,
            total_primaries, color_map, verbose,
        )


# ===================================================================
# Plot 12b — FoM summary restricted to M ≥ min_M
# ===================================================================

def plot_fom_summary_min_m(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: Path,
    min_M: int = 6,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """Horizontal bar: max FoM per config restricted to M ≥ min_M.

    Saved as 12b_fom_summary_M{min_M}plus.png.
    """
    eligible_M = [M for M in M_values if M >= min_M]
    if not eligible_M:
        if verbose:
            print(f"  [SKIP] 12b_fom_summary_M{min_M}plus: no M ≥ {min_M}.")
        return

    ordered = _sorted_by_w2(configs)
    if color_map is None:
        _pal = _get_colors(len(ordered))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(ordered)}
    colors = [color_map.get(cfg.name, "gray") for cfg in ordered]

    max_foms: list[float] = []
    best_mw:  list[str]   = []
    for cfg in ordered:
        grid  = _fom_grid(cfg, eligible_M, W_values, total_primaries)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best = max(valid, key=valid.__getitem__)
            max_foms.append(valid[best])
            best_mw.append(f"M{best[0]}W{best[1]}")
        else:
            max_foms.append(float("nan"))
            best_mw.append("N/A")

    y_max = max((f for f in max_foms if np.isfinite(f)), default=1.0)
    y     = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(8, max(4, len(ordered) * 0.55)))
    bars = ax.barh(y, max_foms, color=colors, height=0.6)

    for bar, mw, fom in zip(bars, best_mw, max_foms):
        if np.isfinite(fom):
            ax.text(
                fom + 0.01 * y_max,
                bar.get_y() + bar.get_height() / 2,
                f"{fom:.4g}  [{mw}]", va="center", ha="left", fontsize=8,
            )

    ax.set_yticks(y)
    ax.set_yticklabels([_config_label(cfg) for cfg in ordered], fontsize=9)
    ax.set_xlim(right=y_max * 1.35)
    ax.set_xlabel(f"Figure of Merit  (max over M≥{min_M}, all W)", fontsize=11)
    ax.set_title(
        f"Ge-77 Muon Figure of Merit — Best (M≥{min_M}, W) per Configuration",
        fontsize=12, pad=10,
    )
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / f"12b_fom_summary_M{min_M}plus.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 17a — W2 vs NC coverage grid (one panel per M)
# ===================================================================

def plot_w2_nc_coverage_scatter_grid(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """17a — Grid of panels (one per M): W2 vs NC detection fraction.

    Each panel has OLS fit and Pearson r / Spearman ρ annotated.
    Saved as 17a_w2_nc_coverage_scatter.png.
    """
    w2_cfgs = [cfg for cfg in configs
               if cfg.w2 is not None and cfg.name != _ALL_VOXELS_NAME]
    if len(w2_cfgs) < 2:
        if verbose:
            print("  [SKIP] 17a_w2_nc_coverage_scatter: fewer than 2 configs have W2.")
        return

    ordered = _sorted_by_w2(w2_cfgs)
    if color_map is None:
        _pal = _get_colors(len(ordered))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(ordered)}
    w2_arr    = np.array([cfg.w2 for cfg in ordered])
    labels    = [cfg.label for cfg in ordered]
    color_pts = [color_map.get(cfg.name, "gray") for cfg in ordered]

    ncols = min(len(M_values), 4)
    nrows = (len(M_values) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.2),
                              squeeze=False)
    fig.suptitle(
        "W2 Homogeneity vs NC Detection Fraction\n"
        "(OLS fit · Pearson r · Spearman ρ per M)",
        fontsize=12,
    )

    for pi, M in enumerate(M_values):
        row, col = divmod(pi, ncols)
        ax = axes[row][col]
        y_arr = np.array([_nc_frac(cfg, M) for cfg in ordered])
        _scatter_corr_panel(
            ax, w2_arr, y_arr, color_pts, labels,
            x_label="W2 (mm)",
            y_label=f"NC fraction (M≥{M})",
            title=f"M ≥ {M}",
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))

    for pi in range(len(M_values), nrows * ncols):
        row, col = divmod(pi, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_dir / "17a_w2_nc_coverage_scatter.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 17b — W2 vs FoM correlation vs M (M ≥ min_m)
# ===================================================================

def plot_w2_fom_corr_mge(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: Path,
    min_m: int = 6,
    verbose: bool = True,
) -> None:
    """17b — Pearson r and Spearman ρ between W2 and best FoM (M ≥ min_m) vs M.

    Saved as 17b_w2_fom_corr_Mge{min_m}.png.
    """
    w2_cfgs = [cfg for cfg in configs
               if cfg.w2 is not None and cfg.name != _ALL_VOXELS_NAME]
    if len(w2_cfgs) < 3:
        if verbose:
            print("  [SKIP] 17b_w2_fom_corr: fewer than 3 configs have W2.")
        return
    eligible_M = [M for M in M_values if M >= min_m]
    if not eligible_M:
        if verbose:
            print(f"  [SKIP] 17b_w2_fom_corr: no M values >= {min_m}.")
        return

    w2_arr = np.array([cfg.w2 for cfg in w2_cfgs])
    _n_w2  = len(w2_cfgs)
    pearson_r, spearman_rho = [], []
    p_pearson, p_spearman   = [], []

    for M in eligible_M:
        fom_arr = []
        for cfg in w2_cfgs:
            best = np.nan
            for W in W_values:
                cm = cfg.confusion.get((M, W))
                if cm is None:
                    continue
                v = calc_fom_confusion(
                    cm["TP"], cm["FP"], cm["FN"], total_primaries,
                    tp_ge77_nc_counts=cm.get("tp_ge77_nc_counts"),
                    fn_ge77_nc_counts=cm.get("fn_ge77_nc_counts"),
                )
                if np.isfinite(v) and (np.isnan(best) or v > best):
                    best = v
            fom_arr.append(best)
        fom_np = np.array(fom_arr)
        msk = np.isfinite(fom_np) & np.isfinite(w2_arr)
        if msk.sum() < 3 or np.std(w2_arr[msk]) == 0 or np.std(fom_np[msk]) == 0:
            pearson_r.append(np.nan);    p_pearson.append(np.nan)
            spearman_rho.append(np.nan); p_spearman.append(np.nan)
            continue
        r_val, p_r   = scipy_stats.pearsonr(w2_arr[msk], fom_np[msk])
        rho,   p_rho = scipy_stats.spearmanr(w2_arr[msk], fom_np[msk])
        pearson_r.append(r_val);   p_pearson.append(p_r)
        spearman_rho.append(rho);  p_spearman.append(p_rho)

    x            = np.array(eligible_M)
    pearson_r    = np.array(pearson_r)
    p_pearson    = np.array(p_pearson)
    spearman_rho = np.array(spearman_rho)
    p_spearman   = np.array(p_spearman)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="black", linewidth=0.8)
    _draw_pearson_rcrit(ax, _n_w2, sigma=3.0)
    _draw_pearson_rcrit(ax, _n_w2, sigma=5.0, linestyle="-.", draw_label=True)

    for vals, ps, label, color, marker in [
        (pearson_r,    p_pearson,  "Pearson r",   "#1f77b4", "o"),
        (spearman_rho, p_spearman, "Spearman ρ",  "#d62728", "s"),
    ]:
        sig  = ps < _P_3SIGMA
        nsig = ~sig & np.isfinite(vals)
        ax.plot(x, vals, color=color, linewidth=1.8, label=label)
        if sig.any():
            ax.scatter(x[sig],  vals[sig],  color=color, s=60, marker=marker, zorder=4)
        if nsig.any():
            ax.scatter(x[nsig], vals[nsig], facecolors="none", edgecolors=color,
                       s=60, marker=marker, linewidth=1.2, zorder=4)

    ax.set_xlabel("Multiplicity threshold M", fontsize=11)
    ax.set_ylabel("Correlation coefficient (W2 vs best FoM)", fontsize=11)
    ax.set_title(
        f"W2 vs FoM  (M ≥ {min_m})  — Pearson r and Spearman ρ vs M\n"
        "(filled = 3σ significant; dashed = 3σ / dash-dot = 5σ Pearson threshold)",
        fontsize=12,
    )
    ax.set_xticks(eligible_M)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / f"17b_w2_fom_corr_Mge{min_m}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 17c — W2 vs Recall at best-FoM: two scatter panels
# ===================================================================

def plot_w2_recall_corr_split(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: Path,
    min_m_constrained: int = 6,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """17c — W2 vs Recall: left = all M, right = M ≥ min_m_constrained.

    Saved as 17c_w2_recall_corr.png.
    """
    w2_cfgs = [cfg for cfg in configs
               if cfg.w2 is not None and cfg.name != _ALL_VOXELS_NAME]
    if len(w2_cfgs) < 2:
        if verbose:
            print("  [SKIP] 17c_w2_recall_corr: fewer than 2 configs have W2.")
        return

    ordered = _sorted_by_w2(w2_cfgs)
    if color_map is None:
        _pal = _get_colors(len(ordered))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(ordered)}
    w2_arr    = np.array([cfg.w2 for cfg in ordered])
    labels    = [cfg.label for cfg in ordered]
    color_pts = [color_map.get(cfg.name, "gray") for cfg in ordered]

    eligible_ge = [M for M in M_values if M >= min_m_constrained]

    def _recall_at_fom(cfg, m_search):
        grid  = _fom_grid(cfg, m_search, W_values, total_primaries)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best = max(valid, key=valid.__getitem__)
            return _recall(cfg, best[0], best[1])
        return float("nan")

    recalls_all = np.array([_recall_at_fom(cfg, M_values) for cfg in ordered])
    recalls_ge  = (np.array([_recall_at_fom(cfg, eligible_ge) for cfg in ordered])
                   if eligible_ge else np.full(len(ordered), np.nan))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        "W2 vs Muon Recall — Pearson r and Spearman ρ\n"
        "(OLS fit · left: all M · right: M ≥ " + str(min_m_constrained) + ")",
        fontsize=12,
    )

    for ax, recalls, title in [
        (ax_l, recalls_all, "Recall at best FoM  (all M)"),
        (ax_r, recalls_ge,  f"Recall at best FoM  (M ≥ {min_m_constrained})"),
    ]:
        _scatter_corr_panel(
            ax, w2_arr, recalls, color_pts, labels,
            x_label="W2 (mm)  [lower = more uniform]",
            y_label="Ge-77 Muon Recall",
            title=title,
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = output_dir / "17c_w2_recall_corr.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 17d — W2 vs NC coverage: Pearson r and Spearman ρ vs M (all M)
# ===================================================================

def plot_w2_nc_coverage_corr_all_m(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """17d — W2 vs NC coverage: Pearson r and Spearman ρ vs M (all M).

    Saved as 17d_w2_nc_coverage_corr.png.
    """
    w2_cfgs = [cfg for cfg in configs
               if cfg.w2 is not None and cfg.name != _ALL_VOXELS_NAME]
    if len(w2_cfgs) < 3:
        if verbose:
            print("  [SKIP] 17d_w2_nc_coverage_corr: fewer than 3 configs have W2.")
        return

    w2_arr = np.array([cfg.w2 for cfg in w2_cfgs])
    _n_w2  = len(w2_cfgs)
    pearson_r, spearman_rho = [], []
    p_pearson, p_spearman   = [], []

    for M in M_values:
        nc_arr = np.array([_nc_frac(cfg, M) for cfg in w2_cfgs])
        msk = np.isfinite(nc_arr) & np.isfinite(w2_arr)
        if msk.sum() < 3 or np.std(w2_arr[msk]) == 0 or np.std(nc_arr[msk]) == 0:
            pearson_r.append(np.nan);    p_pearson.append(np.nan)
            spearman_rho.append(np.nan); p_spearman.append(np.nan)
            continue
        r_val, p_r   = scipy_stats.pearsonr(w2_arr[msk], nc_arr[msk])
        rho,   p_rho = scipy_stats.spearmanr(w2_arr[msk], nc_arr[msk])
        pearson_r.append(r_val);   p_pearson.append(p_r)
        spearman_rho.append(rho);  p_spearman.append(p_rho)

    x            = np.array(M_values)
    pearson_r    = np.array(pearson_r)
    p_pearson    = np.array(p_pearson)
    spearman_rho = np.array(spearman_rho)
    p_spearman   = np.array(p_spearman)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="black", linewidth=0.8)
    _draw_pearson_rcrit(ax, _n_w2, sigma=3.0)
    _draw_pearson_rcrit(ax, _n_w2, sigma=5.0, linestyle="-.", draw_label=True)

    for vals, ps, label, color, marker in [
        (pearson_r,    p_pearson,  "Pearson r",   "#1f77b4", "o"),
        (spearman_rho, p_spearman, "Spearman ρ",  "#d62728", "s"),
    ]:
        sig  = ps < _P_3SIGMA
        nsig = ~sig & np.isfinite(vals)
        ax.plot(x, vals, color=color, linewidth=1.8, label=label)
        if sig.any():
            ax.scatter(x[sig],  vals[sig],  color=color, s=60, marker=marker, zorder=4)
        if nsig.any():
            ax.scatter(x[nsig], vals[nsig], facecolors="none", edgecolors=color,
                       s=60, marker=marker, linewidth=1.2, zorder=4)

    ax.set_xlabel("Multiplicity threshold M", fontsize=11)
    ax.set_ylabel("Correlation coefficient (W2 vs NC coverage)", fontsize=11)
    ax.set_title(
        "W2 vs NC Coverage  (all M)  — Pearson r and Spearman ρ vs M\n"
        "(filled = 3σ significant; dashed = 3σ / dash-dot = 5σ Pearson threshold)",
        fontsize=12,
    )
    ax.set_xticks(M_values)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "17d_w2_nc_coverage_corr.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 17e — W2 vs Recall: Pearson r and Spearman ρ vs M (3 panels)
# ===================================================================

def plot_w2_recall_corr_all_m(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: Path,
    min_m_constrained: int = 6,
    verbose: bool = True,
) -> None:
    """17e — W2 vs Recall: Pearson r and Spearman ρ vs M (3 side-by-side panels).

    Panels: (1) best recall over all W at each M,
            (2) recall at W=1,
            (3) best recall M ≥ min_m_constrained.
    Saved as 17e_w2_recall_corr.png.
    """
    w2_cfgs = [cfg for cfg in configs
               if cfg.w2 is not None and cfg.name != _ALL_VOXELS_NAME]
    if len(w2_cfgs) < 3:
        if verbose:
            print("  [SKIP] 17e_w2_recall_corr: fewer than 3 configs have W2.")
        return

    w2_arr = np.array([cfg.w2 for cfg in w2_cfgs])
    _n_w2  = len(w2_cfgs)

    def _best_recall(cfg, M):
        return max((_recall(cfg, M, W) for W in W_values
                    if np.isfinite(_recall(cfg, M, W))),
                   default=float("nan"))

    def _corr_line(m_list, get_val):
        pr, sr, pp, sp = [], [], [], []
        for M in m_list:
            y_arr = np.array([get_val(cfg, M) for cfg in w2_cfgs])
            msk = np.isfinite(y_arr) & np.isfinite(w2_arr)
            if msk.sum() < 3 or np.std(w2_arr[msk]) == 0 or np.std(y_arr[msk]) == 0:
                pr.append(np.nan);  pp.append(np.nan)
                sr.append(np.nan);  sp.append(np.nan)
                continue
            r_val, p_r   = scipy_stats.pearsonr(w2_arr[msk], y_arr[msk])
            rho,   p_rho = scipy_stats.spearmanr(w2_arr[msk], y_arr[msk])
            pr.append(r_val);  pp.append(p_r)
            sr.append(rho);    sp.append(p_rho)
        return np.array(pr), np.array(sr), np.array(pp), np.array(sp)

    eligible_ge = [M for M in M_values if M >= min_m_constrained]

    panels = [
        (M_values,    _best_recall,
         "Best Recall over all W  (all M)"),
        (M_values,    lambda cfg, M: _recall(cfg, M, 1),
         "Recall at W = 1  (all M)"),
        (eligible_ge, _best_recall,
         f"Best Recall over all W  (M ≥ {min_m_constrained})"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "W2 vs Ge-77 Muon Recall  — Pearson r and Spearman ρ vs M\n"
        "(filled = 3σ significant; dashed = 3σ / dash-dot = 5σ Pearson threshold)",
        fontsize=12,
    )

    for ax, (m_list, get_val, title) in zip(axes, panels):
        if not m_list:
            ax.set_visible(False)
            continue
        x = np.array(m_list)
        pr, sr, pp, sp = _corr_line(m_list, get_val)

        ax.axhline(0, color="black", linewidth=0.8)
        _draw_pearson_rcrit(ax, _n_w2, sigma=3.0, draw_label=(ax is axes[0]))
        _draw_pearson_rcrit(ax, _n_w2, sigma=5.0, linestyle="-.",
                            draw_label=(ax is axes[0]))

        for vals, ps, label, color, marker in [
            (pr, pp, "Pearson r",   "#1f77b4", "o"),
            (sr, sp, "Spearman ρ",  "#d62728", "s"),
        ]:
            sig  = ps < _P_3SIGMA
            nsig = ~sig & np.isfinite(vals)
            ax.plot(x, vals, color=color, linewidth=1.8, label=label)
            if sig.any():
                ax.scatter(x[sig],  vals[sig],  color=color, s=60,
                           marker=marker, zorder=4)
            if nsig.any():
                ax.scatter(x[nsig], vals[nsig], facecolors="none",
                           edgecolors=color, s=60, marker=marker,
                           linewidth=1.2, zorder=4)

        ax.set_xlabel("Multiplicity threshold M", fontsize=10)
        ax.set_ylabel("Correlation (W2 vs Recall)", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(m_list)
        ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "17e_w2_recall_corr.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# Plot 19b — NC recall correlation summary (Pearson r vs W, per M)
# ===================================================================

def plot_nc_recall_correlation_summary(
    configs: list[ConfigResult],
    M_values: list[int],
    W_fixed_values: list[int],
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """19b — Pearson r(NC fraction, Recall) vs W, one curve per M.

    Saved as 19b_nc_recall_correlation_summary.png.
    """
    plot_cfgs = [cfg for cfg in configs if cfg.name != _ALL_VOXELS_NAME]
    if len(plot_cfgs) < 3:
        if verbose:
            print("  [SKIP] 19b_nc_recall_correlation_summary: fewer than 3 configs.")
        return

    n = len(plot_cfgs)
    r_crit_3 = _pearson_rcrit(n, sigma=3.0)
    r_crit_5 = _pearson_rcrit(n, sigma=5.0)

    _pal = _get_colors(len(M_values))
    color_map_m = {M: _pal[i] for i, M in enumerate(M_values)}

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_pearson_rcrit(ax, n, sigma=3.0, draw_label=True)
    _draw_pearson_rcrit(ax, n, sigma=5.0, linestyle="-.", draw_label=True)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")

    for M in M_values:
        nc_arr = np.array([_nc_frac(cfg, M) for cfg in plot_cfgs])
        rs = []
        for W in W_fixed_values:
            rec_arr = np.array([_recall(cfg, M, W) for cfg in plot_cfgs])
            mask = np.isfinite(nc_arr) & np.isfinite(rec_arr)
            if mask.sum() >= 3 and np.std(nc_arr[mask]) > 0 and np.std(rec_arr[mask]) > 0:
                r_val, _ = scipy_stats.pearsonr(nc_arr[mask], rec_arr[mask])
                rs.append(r_val)
            else:
                rs.append(float("nan"))
        rs = np.array(rs)
        ax.plot(W_fixed_values, rs, marker="o", color=color_map_m[M],
                label=f"M={M}", linewidth=1.5)

    ax.set_xlabel("W — minimum NC detections per muon", fontsize=11)
    ax.set_ylabel("Pearson r  (NC fraction vs Ge-77 Recall)", fontsize=11)
    ax.set_title(
        "NC Detection Fraction vs Ge-77 Recall — Pearson r\n"
        f"(n={n} setups; dashed = 3σ |r|={r_crit_3:.2f};  dash-dot = 5σ |r|={r_crit_5:.2f})",
        fontsize=12,
    )
    ax.set_xticks(W_fixed_values)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=9, loc="upper right", ncol=max(1, len(M_values) // 5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "19b_nc_recall_correlation_summary.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# FoM colormap background helper + Plot 25 / 25d / 25e
# ===================================================================

def _fom_colormap_background(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    n_grid: int = 300,
    cmap: str = "viridis",
    alpha: float = 0.35,
    x_range: "tuple[float, float] | None" = None,
    y_range: "tuple[float, float] | None" = None,
):
    """Draw FoM(signal_surv, ge_surv) colormap + contours on ax.

    Returns the pcolormesh artist (for colorbar attachment), or None.
    """
    if x_range is not None:
        xg = np.linspace(x_range[0], x_range[1], n_grid)
    else:
        if not xs:
            return None
        mx, mx2 = min(xs), max(xs)
        dx = max((mx2 - mx) * 0.05, 1e-4)
        xg = np.linspace(mx - dx, mx2 + dx, n_grid)
    if y_range is not None:
        yg = np.linspace(y_range[0], y_range[1], n_grid)
    else:
        if not ys:
            return None
        my, my2 = min(ys), max(ys)
        dy = max((my2 - my) * 0.05, 1e-4)
        yg = np.linspace(my - dy, my2 + dy, n_grid)
    XX, YY = np.meshgrid(xg, yg)
    _fom_vec = np.vectorize(figure_of_merit)
    ZZ = _fom_vec(YY, XX)   # ge_surv=YY, signal_surv=XX
    ZZ = np.where(np.isfinite(ZZ), ZZ, np.nan)
    pcm = ax.pcolormesh(XX, YY, ZZ, cmap=cmap, alpha=alpha, shading="auto", zorder=0)
    finite_z = ZZ[np.isfinite(ZZ)]
    if finite_z.size > 0:
        n_levels = 8
        levels = np.linspace(finite_z.min(), finite_z.max(), n_levels + 2)[1:-1]
        cs = ax.contour(XX, YY, ZZ, levels=levels, colors="gray",
                        linewidths=0.6, alpha=0.8, zorder=1)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
    return pcm


def plot_ge_surv_vs_livetime(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """25 — Ge77 survival vs 1−deadtime scatter for all (M, W) combinations.

    Saved as 25_ge_surv_vs_livetime.png.
    """
    if color_map is None:
        _pal = _get_colors(len(configs))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(configs)}

    fig, ax = plt.subplots(figsize=(10, 7))

    all_xs: list[float] = []
    all_ys: list[float] = []
    per_cfg: list[tuple[list[float], list[float]]] = []

    for cfg in configs:
        xs, ys = [], []
        for M in M_values:
            for W in W_values:
                cm = cfg.confusion.get((M, W))
                if cm is None:
                    continue
                TN = total_primaries - cm["TP"] - cm["FP"] - cm["FN"]
                ge_surv  = calc_ge_survival_confusion(
                    cm.get("tp_ge77_nc_counts",
                           np.ones(cm["TP"], dtype=np.int32)),
                    cm.get("fn_ge77_nc_counts",
                           np.ones(cm["FN"], dtype=np.int32)),
                )
                deadtime = calc_deadtime_confusion(cm["TP"], cm["FP"], TN, cm["FN"])
                xs.append(1.0 - deadtime)
                ys.append(ge_surv)
        per_cfg.append((xs, ys))
        all_xs.extend(xs)
        all_ys.extend(ys)

    pcm = _fom_colormap_background(ax, all_xs, all_ys)
    if pcm is not None:
        fig.colorbar(pcm, ax=ax, label="FoM", pad=0.01)

    for cfg, (xs, ys) in zip(configs, per_cfg):
        c = color_map.get(cfg.name, "gray")
        if xs:
            ax.scatter(xs, ys, color=c, s=18, alpha=0.7, zorder=3)

    ax.set_xlabel("1 − Deadtime  (signal livetime fraction)", fontsize=11)
    ax.set_ylabel("Ge77 survival  (Σ FN Ge77 NCs / Σ all Ge77 NCs)", fontsize=11)
    ax.set_title(
        "Ge77 Survival vs Signal Livetime Trade-off\n"
        "(each point = one (M, W) combination; bottom-right = optimal)",
        fontsize=12,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color_map.get(cfg.name, "gray"),
                       markersize=8, label=cfg.label)
            for cfg in configs
        ],
        fontsize=9, loc="upper left",
    )
    ax.grid(True, alpha=0.3)
    if all_xs and all_ys:
        _ax_dx = max((max(all_xs) - min(all_xs)) * 0.05, 1e-4)
        _ax_dy = max((max(all_ys) - min(all_ys)) * 0.05, 1e-4)
        ax.set_xlim(min(all_xs) - _ax_dx, max(all_xs) + _ax_dx)
        ax.set_ylim(min(all_ys) - _ax_dy, max(all_ys) + _ax_dy)
    fig.tight_layout()
    out_path = output_dir / "25_ge_surv_vs_livetime.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


def plot_ge_surv_best_fom(
    configs: list[ConfigResult],
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: Path,
    color_map: dict[str, str] | None = None,
    verbose: bool = True,
    m_min: int = 1,
) -> None:
    """25d/25e — One point per setup at FoM-optimal (M, W).

    m_min restricts M search (use 6 for 25e).
    Saved as 25d_ge_surv_best_fom.png or 25e_ge_surv_best_fom_M_ge6.png.
    """
    if color_map is None:
        _pal = _get_colors(len(configs))
        color_map = {cfg.name: _pal[i] for i, cfg in enumerate(configs)}

    m_search = [M for M in M_values if M >= m_min]
    if not m_search:
        if verbose:
            print(f"  [SKIP] plot_ge_surv_best_fom(m_min={m_min}): no M values >= {m_min}.")
        return

    pts_x: list[float] = []
    pts_y: list[float] = []
    pt_labels: list[str] = []

    for cfg in configs:
        grid = _fom_grid(cfg, m_search, W_values, total_primaries)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if not valid:
            pts_x.append(float("nan"))
            pts_y.append(float("nan"))
            pt_labels.append("N/A")
            continue
        best_M, best_W = max(valid, key=valid.__getitem__)
        cm = cfg.confusion.get((best_M, best_W))
        TN = total_primaries - cm["TP"] - cm["FP"] - cm["FN"]
        ge_surv  = calc_ge_survival_confusion(
            cm.get("tp_ge77_nc_counts", np.ones(cm["TP"], dtype=np.int32)),
            cm.get("fn_ge77_nc_counts", np.ones(cm["FN"], dtype=np.int32)),
        )
        deadtime = calc_deadtime_confusion(cm["TP"], cm["FP"], TN, cm["FN"])
        pts_x.append(1.0 - deadtime)
        pts_y.append(ge_surv)
        pt_labels.append(f"M{best_M}W{best_W}")

    all_xs = [x for x in pts_x if np.isfinite(x)]
    all_ys = [y for y in pts_y if np.isfinite(y)]

    fig, ax = plt.subplots(figsize=(10, 7))
    pcm = _fom_colormap_background(ax, all_xs, all_ys)
    if pcm is not None:
        fig.colorbar(pcm, ax=ax, label="FoM", pad=0.01)

    for cfg, x, y, lbl in zip(configs, pts_x, pts_y, pt_labels):
        c = color_map.get(cfg.name, "gray")
        if np.isfinite(x):
            ax.scatter([x], [y], color=c, s=60, zorder=4)
            ax.annotate(lbl, xy=(x, y), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color=c)

    m_desc = f"M ≥ {m_min}" if m_min > 1 else "all M"
    ax.set_xlabel("1 − Deadtime  (signal livetime fraction)", fontsize=11)
    ax.set_ylabel("Ge77 survival  (Σ FN Ge77 NCs / Σ all Ge77 NCs)", fontsize=11)
    ax.set_title(
        f"Ge77 Survival vs Signal Livetime — Best FoM per Setup  ({m_desc})\n"
        "(each point = FoM-optimal (M, W); bottom-right = optimal)",
        fontsize=12,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color_map.get(cfg.name, "gray"),
                       markersize=8, label=cfg.label)
            for cfg in configs
        ],
        fontsize=9, loc="upper left",
    )
    ax.grid(True, alpha=0.3)
    if all_xs and all_ys:
        _ax_dx = max((max(all_xs) - min(all_xs)) * 0.05, 1e-4)
        _ax_dy = max((max(all_ys) - min(all_ys)) * 0.05, 1e-4)
        ax.set_xlim(min(all_xs) - _ax_dx, max(all_xs) + _ax_dx)
        ax.set_ylim(min(all_ys) - _ax_dy, max(all_ys) + _ax_dy)
    fig.tight_layout()
    fname = f"25{'d' if m_min <= 1 else 'e'}_ge_surv_best_fom{'_M_ge6' if m_min > 1 else ''}.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


# ===================================================================
# CLI
# ===================================================================

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare multiple PMT configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--hdf5", type=str, required=True,
                        help="Path to the raw SSD HDF5 data file. "
                             "Area ratios are applied here via stochastic "
                             "rounding (from the baseline JSON or CLI flags). "
                             "Do NOT pass a ratio-adjusted HDF5 "
                             "(from main.py --write-hdf5) — that would "
                             "double-apply the scaling.")
    parser.add_argument("--baseline", type=str, required=True,
                        help="JSON file for the baseline (homogeneous) config. "
                             "This is mandatory.")
    parser.add_argument("--configs", type=str, nargs="+", required=True,
                        help="JSON files for additional PMT configs.")
    parser.add_argument("--labels", type=str, nargs="*", default=None,
                        help="Labels for configs (excluding baseline). "
                             "If omitted, filenames are used.")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Directory for output files.")
    parser.add_argument("-m", type=int, default=1,
                        help="Hit threshold per voxel for binarization.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for stochastic rounding.")
    parser.add_argument("--M-max", type=int, default=10,
                        help="Maximum M value to evaluate.")
    parser.add_argument("--M-default", type=int, default=1,
                        help="Default M threshold used in the NC detectability "
                             "overview plot (Plot 03).")
    parser.add_argument("--W-default", type=int, default=6,
                        help="Default W threshold used in muon correlation "
                             "and matrix plots.")
    parser.add_argument("--W-max", type=int, default=20,
                        help="Maximum W value to evaluate.")

    # Area ratios
    parser.add_argument("--pit", type=float, default=None)
    parser.add_argument("--bot", type=float, default=None)
    parser.add_argument("--top", type=float, default=None)
    parser.add_argument("--wall", type=float, default=None)

    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)
    return args


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    verbose = not args.quiet

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    M_values = list(range(1, args.M_max + 1))
    W_values = list(range(1, args.W_max + 1))

    # ------------------------------------------------------------------
    # Load all configs
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 65)
        print("Loading PMT configurations")
        print("=" * 65)

    # Baseline (always labeled "Baseline")
    bl_voxels, bl_voxel_dicts, bl_data = load_config_json(args.baseline)

    # Area ratios — use baseline JSON ratios if present, else CLI/defaults
    _ALL_LAYERS = ("pit", "bot", "top", "wall")
    cli_ratio_map = {
        k: v for k, v in {"pit": args.pit, "bot": args.bot,
                           "top": args.top, "wall": args.wall}.items()
        if v is not None
    }
    cli_ratio_flags = set(cli_ratio_map)
    json_ratios = bl_data.get("config", {}).get("area_ratios", {})
    if json_ratios:
        if cli_ratio_flags:
            # Require all four layers when overriding JSON ratios via CLI
            missing_layers = [l for l in _ALL_LAYERS if l not in cli_ratio_flags]
            if missing_layers:
                raise RuntimeError(
                    f"Baseline JSON '{args.baseline}' already contains "
                    f"area_ratios. To override them via CLI flags you must "
                    f"specify ALL four layers, but "
                    f"{[f'--{l}' for l in missing_layers]} are missing."
                )
            # Full CLI override — use CLI values and emit a prominent warning
            area_ratios = dict(cli_ratio_map)
            ratio_override_warning = (
                "\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "!!                        WARNING                           !!\n"
                "!!  CLI area ratios are overriding the ratios stored in     !!\n"
                f"!!  the baseline JSON ({Path(args.baseline).name!r:^38s})  !!\n"
                "!!  The binarization will NOT match the ratios used during  !!\n"
                "!!  greedy optimisation.  Remove --pit/--bot/--top/--wall   !!\n"
                "!!  to use the JSON ratios.                                 !!\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            )
            print(ratio_override_warning, file=sys.stderr)
            if verbose:
                print("  Area ratios (CLI override — see WARNING above):")
                for layer in _ALL_LAYERS:
                    json_val = json_ratios.get(layer, DEFAULT_AREA_RATIOS[layer])
                    cli_val = area_ratios[layer]
                    changed = " <-- overrides JSON" if cli_val != json_val else ""
                    print(f"    {layer:>6}: {cli_val}  (JSON had {json_val}){changed}")
        else:
            ratio_override_warning = None
            area_ratios = dict(DEFAULT_AREA_RATIOS)
            area_ratios.update(json_ratios)
            if verbose:
                print(f"  Area ratios from baseline JSON (applied via stochastic "
                      f"rounding to raw HDF5):")
                for layer, ratio in area_ratios.items():
                    src = "(from JSON)" if layer in json_ratios else "(default)"
                    print(f"    {layer:>6}: {ratio}  {src}")
    else:
        ratio_override_warning = None
        area_ratios = dict(DEFAULT_AREA_RATIOS)
        if args.pit is not None:
            area_ratios["pit"] = args.pit
        if args.bot is not None:
            area_ratios["bot"] = args.bot
        if args.top is not None:
            area_ratios["top"] = args.top
        if args.wall is not None:
            area_ratios["wall"] = args.wall
        if verbose:
            print(f"  Area ratios (CLI/defaults):")
            for layer, ratio in area_ratios.items():
                src = "(CLI)" if layer in cli_ratio_flags else "(default)"
                print(f"    {layer:>6}: {ratio}  {src}")

    baseline = ConfigResult(
        name=Path(args.baseline).stem,
        voxel_ids=bl_voxels,
        label="Baseline",
    )
    baseline.voxel_dicts = bl_voxel_dicts
    baseline.w2 = compute_config_w2(bl_voxel_dicts)
    if verbose:
        print(f"  Baseline: {args.baseline} ({len(bl_voxels)} voxels)")

    # Other configs
    if args.labels and len(args.labels) != len(args.configs):
        raise RuntimeError(
            f"Number of --labels ({len(args.labels)}) must match "
            f"number of --configs ({len(args.configs)}). "
            f"Do not include a label for the baseline."
        )

    configs_extra: list[ConfigResult] = []
    for i, cfg_path in enumerate(args.configs):
        voxel_ids, voxel_dicts, cfg_data = load_config_json(cfg_path)
        if args.labels:
            label = args.labels[i]
        else:
            label = Path(cfg_path).stem
        cr = ConfigResult(
            name=Path(cfg_path).stem,
            voxel_ids=voxel_ids,
            label=label,
        )
        cr.voxel_dicts = voxel_dicts
        cr.w2 = compute_config_w2(voxel_dicts)
        configs_extra.append(cr)
        if verbose:
            print(f"  Config {i+1}: {cfg_path} ({len(voxel_ids)} voxels) "
                  f"-> \"{label}\"")

    all_configs = [baseline] + configs_extra

    # Collect all voxel IDs
    all_voxel_ids: set[str] = set()
    for cfg in all_configs:
        all_voxel_ids.update(cfg.voxel_ids)

    if verbose:
        print(f"\n  Total unique voxels across all configs: "
              f"{len(all_voxel_ids)}")

    # ------------------------------------------------------------------
    # Load shared data
    # ------------------------------------------------------------------
    ed = load_shared_data(
        hdf5_path=args.hdf5,
        all_voxel_ids=all_voxel_ids,
        area_ratios=area_ratios,
        m=args.m,
        seed=args.seed,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # Build "All voxels" reference config from pre-computed coverage
    # ------------------------------------------------------------------
    all_voxels_cfg = ConfigResult(
        name="all_voxels",
        voxel_ids=[],
        label=f"All voxels (N={ed.num_all_voxels})",
    )
    all_voxels_cfg.coverage_counts = ed.coverage_counts_all
    all_voxels_cfg.num_ncs = ed.num_ncs
    all_voxels_cfg.num_visible = int(np.sum(ed.coverage_counts_all >= 1))
    for M in M_values:
        all_voxels_cfg.num_detected[M] = int(
            np.sum(ed.coverage_counts_all >= M)
        )
    evaluate_muon(all_voxels_cfg, ed, M_values, W_values)
    all_configs = [all_voxels_cfg] + all_configs

    # ------------------------------------------------------------------
    # Evaluate all configs
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 65)
        print("Evaluating configurations")
        print("=" * 65)

    for cfg in all_configs:
        t0 = time.time()
        if cfg.coverage_counts is not None:
            # Pre-computed (e.g. "All voxels") — muon eval already done above
            pass
        else:
            map_voxels_to_columns(cfg, ed)
            evaluate_nc(cfg, ed, M_values)
            evaluate_muon(cfg, ed, M_values, W_values)
        dt = time.time() - t0
        if verbose:
            det1 = cfg.num_detected.get(1, 0)
            print(f"  {cfg.label:<30} | visible={cfg.num_visible:>10,} | "
                  f"det(M=1)={det1:>10,} | {dt:.1f}s")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 65)
        print("Generating output")
        print("=" * 65)

    # Canonical color map: sorted by W2 desc (all_voxels last).
    # Built once and passed to every plot so the same config always uses
    # the same color, regardless of which subset each plot displays.
    _sorted_for_colors = _sorted_by_w2(all_configs)
    _pal = _get_colors(len(_sorted_for_colors))
    color_map = {cfg.name: _pal[i] for i, cfg in enumerate(_sorted_for_colors)}

    _heatmap_ms = [m for m in [1, 3, 5, 10] if m in M_values]

    # ── 01 NC coverage line plot (log + linear) ──────────────────────
    plot_nc_coverage(all_configs, args.M_max, output_dir,
                     color_map=color_map, verbose=verbose)

    # ── 01b NC rank Spearman stability ───────────────────────────────
    plot_nc_rank_spearman(all_configs, M_values, output_dir, verbose=verbose)

    # ── 03 NC detectability overview (absolute + Δ vs Baseline) ──────
    plot_nc_detectability_overview(
        all_configs, args.M_default, output_dir,
        color_map=color_map, verbose=verbose,
    )

    # Text files
    write_confusion_txt(
        all_configs, M_values, W_values, output_dir,
        num_ge77_muons=ed.num_ge77_muons,
        total_muons=ed.total_muons,
        verbose=verbose,
        ratio_override_warning=ratio_override_warning,
    )
    write_nc_summary_txt(
        all_configs, M_values, output_dir,
        verbose=verbose,
        ratio_override_warning=ratio_override_warning,
    )

    # Figure of Merit — use total primary muons (all muons, not just NC-producing)
    _total_primaries = (
        ed.num_primaries if ed.num_primaries > 0
        else MUONS_PER_RUN_DIR * ed.num_runs
    )
    if verbose:
        _runtime_h = _total_primaries / MUSUN_RATE
        _runtime_yr = _runtime_h / (24 * 365.25)
        _source = "HDF5 /primaries" if ed.num_primaries > 0 else f"{ed.num_runs} runs × {MUONS_PER_RUN_DIR:,}"
        print(f"\n  FoM: total primary muons = {_total_primaries:,}  ({_source})")
        print(f"  FoM: simulated livetime  = {_runtime_h:,.0f} h  =  {_runtime_yr:.2f} yr  (at {MUSUN_RATE} µ/h)")

    # ── 04 Ge77 muon detection funnel ────────────────────────────────
    plot_ge77_muon_overview(
        all_configs, args.M_default, args.W_default, output_dir,
        color_map=color_map, verbose=verbose,
        total_primary_muons=_total_primaries,
    )

    # ── 06 Confusion bar (2×2 TP/FN/TN/FP) ──────────────────────────
    plot_confusion_bar(
        all_configs, args.M_default, args.W_default, output_dir,
        color_map=color_map, verbose=verbose,
        total_primaries=_total_primaries,
    )

    # ── 07b TP vs FP scatter ─────────────────────────────────────────
    plot_tp_fp_scatter(all_configs, M_values, W_values, output_dir,
                       color_map=color_map, verbose=verbose)

    # ── 08 W2 vs NC coverage single-panel ────────────────────────────
    plot_w2_nc_scatter_single(all_configs, M_values, output_dir,
                               M_fixed=1, W_fixed=1,
                               color_map=color_map, verbose=verbose)

    # ── 09/09b W2 vs Recall/FoM at best FoM (two M constraints) ─────
    plot_w2_recall_best_fom_variants(
        all_configs, M_values, W_values, output_dir,
        total_primaries=_total_primaries, color_map=color_map, verbose=verbose,
    )
    plot_w2_fom_best_variants(
        all_configs, M_values, W_values, output_dir,
        total_primaries=_total_primaries, color_map=color_map, verbose=verbose,
    )

    # ── 10/11/12 M×W sweep + FoM summary ────────────────────────────
    plot_mw_sweep(all_configs, M_values, W_values, output_dir,
                  color_map=color_map, verbose=verbose,
                  total_primaries=_total_primaries)
    plot_fom_summary(all_configs, M_values, W_values, _total_primaries, output_dir,
                     color_map=color_map, verbose=verbose)
    plot_fom_summary_min_m(all_configs, M_values, W_values, _total_primaries, output_dir,
                           min_M=6, color_map=color_map, verbose=verbose)

    # ── W2 correlation analysis ──────────────────────────────────────
    plot_w2_fom_best(all_configs, M_values, W_values, _total_primaries, output_dir,
                     color_map=color_map, verbose=verbose)

    # W2 vs performance scatter (three-panel, M=1/W=1 only)
    plot_w2_scatter(all_configs, M_values, output_dir, M_fixed=1, W_fixed=1,
                    color_map=color_map, verbose=verbose)

    # nc_correlation: only M=1..4 where the relationship is statistically significant.
    _corr_ms = [m for m in M_values if m <= 4]
    plot_w2_nc_correlation(
        all_configs, _corr_ms, output_dir, color_map=color_map, verbose=verbose,
    )
    plot_w2_correlation_matrix(
        all_configs, M_values, args.M_default, args.W_default,
        output_dir, verbose=verbose,
    )
    plot_w2_coverage_profile(all_configs, M_values, output_dir, verbose=verbose)
    plot_w2_spearman_vs_m(
        all_configs, M_values, args.W_default, output_dir, verbose=verbose,
        total_primaries=_total_primaries, W_values=W_values,
    )

    # ── 17a–e W2 correlation scatter/line plots ──────────────────────
    plot_w2_nc_coverage_scatter_grid(
        all_configs, M_values, output_dir, color_map=color_map, verbose=verbose,
    )
    plot_w2_fom_corr_mge(
        all_configs, M_values, W_values, _total_primaries, output_dir,
        min_m=6, verbose=verbose,
    )
    plot_w2_recall_corr_split(
        all_configs, M_values, W_values, _total_primaries, output_dir,
        min_m_constrained=6, color_map=color_map, verbose=verbose,
    )
    plot_w2_nc_coverage_corr_all_m(
        all_configs, M_values, output_dir, verbose=verbose,
    )
    plot_w2_recall_corr_all_m(
        all_configs, M_values, W_values, output_dir,
        min_m_constrained=6, verbose=verbose,
    )

    # ── 19 NC-Recall correlation (W = 1, 2, 3, 5, 10) ────────────────
    for _W_fixed in [1, 2, 3, 5, 10]:
        if _W_fixed in W_values:
            plot_nc_recall_correlation(
                all_configs, _heatmap_ms, output_dir,
                color_map=color_map, verbose=verbose,
                W_fixed=_W_fixed,
            )

    # ── 19b NC-Recall correlation summary (Pearson r vs W, per M) ────
    _w_summary = [W for W in W_values if W <= 20]
    plot_nc_recall_correlation_summary(
        all_configs, M_values, _w_summary, output_dir, verbose=verbose,
    )

    # ── 20/21 Recall / Precision at FoM-optimal ──────────────────────
    plot_recall_w1_vs_m(all_configs, M_values, output_dir,
                        color_map=color_map, verbose=verbose)
    plot_recall_at_best_fom(
        all_configs, M_values, W_values, _total_primaries, output_dir,
        color_map=color_map, verbose=verbose,
    )
    plot_recall_at_best_fom(
        all_configs, M_values, W_values, _total_primaries, output_dir,
        color_map=color_map, verbose=verbose, min_m=6,
    )

    # ── 22/23/24 NC-recall, Ge77 survival, signal survival scatter ───
    plot_nc_recall_at_best_fom(all_configs, M_values, W_values, _total_primaries, output_dir,
                               color_map=color_map, verbose=verbose)
    plot_ge77_survival_at_best_fom(all_configs, M_values, W_values, _total_primaries,
                                   output_dir, color_map=color_map, verbose=verbose)
    plot_signal_survival_at_best_fom(all_configs, M_values, W_values, _total_primaries,
                                     output_dir, color_map=color_map, verbose=verbose)
    write_survival_table(all_configs, M_values, W_values, _total_primaries,
                         output_dir, verbose=verbose)

    # ── 25/25d/25e Ge77 survival vs livetime ─────────────────────────
    plot_ge_surv_vs_livetime(
        all_configs, M_values, W_values, _total_primaries, output_dir,
        color_map=color_map, verbose=verbose,
    )
    plot_ge_surv_best_fom(
        all_configs, M_values, W_values, _total_primaries, output_dir,
        color_map=color_map, verbose=verbose, m_min=1,
    )
    plot_ge_surv_best_fom(
        all_configs, M_values, W_values, _total_primaries, output_dir,
        color_map=color_map, verbose=verbose, m_min=6,
    )

    if verbose:
        print("\nDone.")


if __name__ == "__main__":
    main()