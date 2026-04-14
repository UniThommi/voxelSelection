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
    MUSUN_RATE,
    MUONS_PER_RUN_DIR,
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

            config.confusion[(M, W)] = {
                "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            }


# ===================================================================
# Color helper
# ===================================================================

def _get_colors(n: int) -> list:
    """Return n visually distinguishable colors."""
    if n <= 10:
        return list(plt.cm.tab10.colors[:n])
    return [plt.cm.tab20(i / n) for i in range(n)]


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
# Plotting: NC coverage line plot
# ===================================================================

def plot_nc_coverage(
    configs: list[ConfigResult],
    M_max: int,
    output_dir: Path,
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
    colors_all = _get_colors(n)
    color_map = {cfg.name: colors_all[i] for i, cfg in enumerate(ordered)}

    cat_labels = ["Total NCs", f"Detected\n(M≥{M_default})"]
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
        vals  = [cfg.num_ncs, cfg.num_detected.get(M_default, 0)]
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
        f"NC Detectability Overview  [M_default = {M_default}]\n"
        "(Total NCs is identical for all setups — shared NC truth)",
        fontsize=11,
    )
    ax_abs.legend(fontsize=8)
    ax_abs.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    ax_abs.margins(y=0.30)

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
    ordered    = _sorted_by_w2(configs)
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
                    vals.append(calc_fom_confusion(TP, FP, FN, total_primaries))

            ax.plot(
                x, vals,
                color=color_map[cfg.name],
                label=_config_label(cfg),
                linewidth=1.2, marker=".", markersize=4,
            )

        # Envelope: max across all configs at W=1 for each M (connected)
        env_x, env_y = [], []
        for M in M_values:
            idx = next((i for i, (m, w) in enumerate(mw_pairs) if m == M and w == 1), None)
            if idx is None:
                continue
            max_val = -np.inf
            for cfg in ordered:
                cm = cfg.confusion.get((M, 1))
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
                    v = calc_fom_confusion(TP, FP, FN, total_primaries)
                    if not np.isfinite(v):
                        continue
                max_val = max(max_val, v)
            if np.isfinite(max_val):
                env_x.append(idx)
                env_y.append(max_val)
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
                    cm["TP"], cm["FP"], cm["FN"], total_muons
                )
    return result


def plot_fom_summary(
    configs: list,
    M_values: list[int],
    W_values: list[int],
    total_muons: int,
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """Horizontal bar chart: max FoM per config, annotated with optimal (M, W).

    Configs are shown in W2-sorted order (consistent with all other plots).
    """
    ordered = _sorted_by_w2(configs)
    colors  = _get_colors(len(ordered))

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
                mw,
                va="center", ha="left", fontsize=8,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
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
    """Per-config FoM figures: heatmap (M × W grid) and MW-sweep line.

    Produces two PNG files per config:
      fom_heatmap_{name}.png  — 2-D colour map of FoM over (M, W)
      fom_line_{name}.png     — line sweep of FoM across all (M, W) pairs

    The cell / point with the maximum FoM is highlighted in both panels.
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

        # ── 2. Line sweep ─────────────────────────────────────────────
        fom_vals = [grid.get((M, W), np.nan) for M, W in mw_pairs]
        fig_w    = max(30, len(mw_pairs) * 0.18)
        fig_l, ax_l = plt.subplots(figsize=(fig_w, 6))

        ax_l.plot(x_arr, fom_vals, color="steelblue",
                  linewidth=1.4, marker=".", markersize=4)

        finite_idx = [(i, v) for i, v in enumerate(fom_vals) if np.isfinite(v)]
        if finite_idx:
            best_i, best_v = max(finite_idx, key=lambda t: t[1])
            best_pair = mw_pairs[best_i]
            ax_l.scatter(
                best_i, best_v,
                marker="*", s=250, color="red", zorder=5,
                label=f"Best: M{best_pair[0]}W{best_pair[1]} = {best_v:.4g}",
            )
            ax_l.legend(fontsize=9, loc="upper right")

        for gi, M in enumerate(M_values):
            group_start = gi * n_w
            group_mid   = group_start + (n_w - 1) / 2
            if gi > 0:
                ax_l.axvline(group_start - 0.5, color="gray",
                             linewidth=0.6, linestyle="--", alpha=0.5)
            ax_l.text(
                group_mid, 1.02, f"M={M}",
                transform=ax_l.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color="dimgray",
            )

        ax_l.set_xticks(x_arr)
        ax_l.set_xticklabels(x_labels, rotation=90, fontsize=7)
        ax_l.set_ylabel("Figure of Merit", fontsize=11)
        ax_l.set_xlim(-0.5, len(mw_pairs) - 0.5)
        ax_l.set_title(f"Figure of Merit — {title_sfx}", fontsize=12, pad=20)
        ax_l.grid(True, axis="y", alpha=0.3)
        fig_l.tight_layout()
        out_l = output_dir / f"fom_line_{safe_name}.png"
        plt.savefig(out_l, dpi=200, bbox_inches="tight")
        plt.close(fig_l)
        if verbose:
            print(f"  Saved: {out_l}")



# ===================================================================
# Plotting: W2 correlation — FoM and Recall
# ===================================================================

def plot_w2_fom_best(
    configs: list,
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: Path,
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

    colors   = _get_colors(len(w2_cfgs))
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
    for c, w2v, fom, col in zip(w2_cfgs, w2_vals, fom_best, colors):
        if np.isfinite(fom):
            ax.scatter(w2v, fom, color=col, s=65, zorder=3)
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
                            cm["TP"], cm["FP"], cm["FN"], total_primaries
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

    colors = _get_colors(len(w2_cfgs))
    color_map = {cfg.name: colors[i] for i, cfg in enumerate(w2_cfgs)}

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


# ===================================================================
# Plot A — W2 × NC coverage correlation scatter
# ===================================================================

def plot_w2_nc_correlation(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
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

    colors_all = _get_colors(len(w2_cfgs))
    color_map  = {cfg.name: colors_all[i] for i, cfg in enumerate(w2_cfgs)}
    w2_arr     = np.array([cfg.w2 for cfg in w2_cfgs])
    labels     = [cfg.label for cfg in w2_cfgs]
    color_pts  = [color_map[cfg.name] for cfg in w2_cfgs]

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

    colors_all = _get_colors(len(w2_cfgs))
    color_map  = {cfg.name: colors_all[i] for i, cfg in enumerate(w2_cfgs)}
    w2_arr     = np.array([cfg.w2 for cfg in w2_cfgs])
    labels     = [cfg.label for cfg in w2_cfgs]
    color_pts  = [color_map[cfg.name] for cfg in w2_cfgs]

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
) -> None:
    """
    Plot E — Spearman ρ between W2 and each metric as a function of M.

    Three lines on one axes:
      - ρ(W2, NC_fraction_M)
      - ρ(W2, Recall at (M, W_default))
      - ρ(W2, Precision at (M, W_default))

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
    p_nc   = []
    p_rec  = []
    p_prec = []

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

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.array(M_values)
    ax.axhline(0,  color="black", linewidth=0.8)
    ax.axhspan(-0.3, 0.3, color="gray", alpha=0.08, label="weak |ρ|<0.3")

    for rhos, ps, label, color, marker in [
        (rho_nc,   p_nc,   "NC fraction",              "#1f77b4", "o"),
        (rho_rec,  p_rec,  f"Recall (W={W_default})",  "#d62728", "s"),
        (rho_prec, p_prec, f"Precision (W={W_default})","#2ca02c", "^"),
    ]:
        rhos = np.array(rhos)
        ps   = np.array(ps)
        sig_mask  = ps < 0.05
        # Filled markers for significant, hollow for non-significant
        ax.plot(x, rhos, color=color, linewidth=1.5, label=label)
        if sig_mask.any():
            ax.scatter(x[sig_mask],  rhos[sig_mask],
                       color=color, s=60, marker=marker,
                       zorder=4, label=f"{label} (p<0.05)")
        if (~sig_mask).any():
            ax.scatter(x[~sig_mask], rhos[~sig_mask],
                       facecolors="none", edgecolors=color,
                       s=60, marker=marker, linewidth=1.2,
                       zorder=4)

    ax.set_xlabel("Multiplicity threshold M", fontsize=11)
    ax.set_ylabel("Spearman ρ  (W2 vs metric)", fontsize=11)
    ax.set_title(
        f"Spearman Correlation between W2 and Coverage Metrics vs M\n"
        f"(filled = p<0.05 significant | W_default = {W_default})",
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

    # NC coverage line plot (log + linear, two panels)
    plot_nc_coverage(all_configs, args.M_max, output_dir, verbose=verbose)

    # NC coverage bar chart — redundant with line plot, omitted:
    # plot_nc_coverage_bars(all_configs, args.M_max, output_dir, verbose=verbose)

    # NC detectability overview — absolute + Δ vs Baseline (Plot 03)
    plot_nc_detectability_overview(
        all_configs, args.M_default, output_dir, verbose=verbose
    )

    # Muon heatmaps: only M=1,3,5,10 — sufficient to show the (M,W) trade-off.
    _heatmap_ms = [m for m in [1, 3, 5, 10] if m in M_values]
    plot_muon_heatmaps(
        all_configs, _heatmap_ms, W_values, output_dir,
        num_ge77_muons=ed.num_ge77_muons,
        total_muons=ed.total_muons,
        verbose=verbose,
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

    # M×W sweep
    plot_mw_sweep(all_configs, M_values, W_values, output_dir, verbose=verbose,
                  total_primaries=_total_primaries)
    plot_fom_summary(all_configs, M_values, W_values, _total_primaries, output_dir, verbose=verbose)
    plot_fom_per_setup(all_configs, M_values, W_values, _total_primaries, output_dir, verbose=verbose)
    plot_w2_fom_best(all_configs, M_values, W_values, _total_primaries, output_dir, verbose=verbose)
    plot_w2_sorted_heatmaps(all_configs, M_values, W_values, _total_primaries, output_dir, verbose=verbose)

    # W2 vs performance scatter (three-panel, M=1/W=1 only)
    plot_w2_scatter(all_configs, M_values, output_dir, M_fixed=1, W_fixed=1,
                    verbose=verbose)

    # W2 correlation analysis
    # nc_correlation: only M=1..4 where the relationship is statistically significant.
    _corr_ms = [m for m in M_values if m <= 4]
    plot_w2_nc_correlation(
        all_configs, _corr_ms, output_dir, verbose=verbose,
    )
    # muon_correlation: large grid — omitted; Spearman summary covers it.
    # plot_w2_muon_correlation(
    #     all_configs, M_values, args.W_default, output_dir, verbose=verbose,
    # )
    plot_w2_correlation_matrix(
        all_configs, M_values, args.M_default, args.W_default,
        output_dir, verbose=verbose,
    )
    plot_w2_coverage_profile(
        all_configs, M_values, output_dir, verbose=verbose,
    )
    plot_w2_spearman_vs_m(
        all_configs, M_values, args.W_default, output_dir, verbose=verbose,
    )

    if verbose:
        print("\nDone.")


if __name__ == "__main__":
    main()