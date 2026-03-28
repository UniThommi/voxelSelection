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

from pmtopt.geometry import (
    DEFAULT_AREA_RATIOS,
    MUON_TIME_WINDOW_MIN_NS,
    MUON_TIME_WINDOW_MAX_NS,
)
from pmtopt.homogeneous import (
    compute_wasserstein_homogeneity,
    get_w2_ref,
)
from pmtopt.data_loading import (
    load_raw_sparse,
    binarize_from_raw,
    load_muon_data,
    build_muon_index,
)


# ===================================================================
# Constants
# ===================================================================

M_VALUES = list(range(1, 11))   # M = 1..10
W_VALUES = list(range(1, 21))   # W = 1..20

# Scatter plot selection — always include M=4,5,6
_SCATTER_M_VALUES = [1, 2, 4, 5, 6, 8, 10]


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
    Per M: one figure with 3 heatmaps (Recall, Precision, F2-score).
    Axes: W (y) x Config (x). Cell text = value.
    """
    config_labels = [c.label for c in configs]
    num_configs = len(configs)

    for M in M_values:
        fig, axes = plt.subplots(
            1, 3, figsize=(max(9, num_configs * 2.4 + 2), len(W_values) * 0.8 + 3),
            sharey=True,
        )

        metric_names = ["Recall", "Precision", "F2-score"]
        cmaps = ["YlOrRd", "PuBu", "Greens"]

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
                    elif metric_name == "F2-score":
                        # F_beta with beta=2: weights recall twice as much as precision
                        denom = 5 * TP + 4 * FN + FP
                        val = 5 * TP / denom if denom > 0 else 0.0
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
            f"| Recall = TP/(TP+FN)  Precision = TP/(TP+FP)  F2 = 5·TP/(5·TP+4·FN+FP)",
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


def plot_w2_coverage_scatter(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """One subplot per selected M — global W2 (x) vs detected NCs (y).
    Each setup shown as a vertical stem + dot. Shared legend with setup names."""
    w2_cfgs = [cfg for cfg in configs if cfg.w2 is not None]

    if len(w2_cfgs) < 2:
        return

    colors = _get_colors(len(w2_cfgs))
    color_map = {cfg.name: colors[i] for i, cfg in enumerate(w2_cfgs)}

    scatter_ms = [M for M in _SCATTER_M_VALUES if M in set(M_values)]
    if not scatter_ms:
        scatter_ms = M_values[:6]

    ncols = 3
    nrows = (len(scatter_ms) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes_flat = np.array(axes).flatten()

    legend_handles = []

    for pi, M in enumerate(scatter_ms):
        ax = axes_flat[pi]

        for cfg in w2_cfgs:
            color = color_map[cfg.name]
            w2 = cfg.w2
            count = cfg.num_detected.get(M, 0)
            ax.plot([w2, w2], [0, count], color=color, linewidth=1.8, alpha=0.7)
            ax.scatter([w2], [count], color=color, s=60, zorder=3)
            if pi == 0:
                legend_handles.append(
                    plt.Line2D([0], [0], color=color, marker="o", markersize=6,
                               label=_config_label(cfg), linewidth=1.5)
                )

        ax.set_xlabel("Global W2 (mm)", fontsize=9)
        ax.set_ylabel("Detected NCs", fontsize=9)
        ax.set_title(f"M ≥ {M}", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    for pi in range(len(scatter_ms), len(axes_flat)):
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
        "NC Coverage vs Global W2 Homogeneity\n"
        "(vertical stem per setup, one panel per M threshold)",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = output_dir / "w2_coverage_scatter.png"
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
    # Extra vertical space per panel for precision text below axes
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5, nrows * 5.5),
        gridspec_kw={"hspace": 0.55},
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

        # Precision text below each subplot
        prec_text = "Precision:  " + "   ".join(precision_lines)
        ax.text(
            0.5, -0.22, prec_text,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=7, family="monospace",
            wrap=True,
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

    # Muon heatmaps
    plot_muon_heatmaps(
        all_configs, M_values, W_values, output_dir,
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

    # W2 vs coverage plots
    plot_w2_coverage_scatter(all_configs, M_values, output_dir, verbose=verbose)
    plot_muon_w2_scatter(
        all_configs, M_values, W_values, output_dir,
        num_ge77_muons=ed.num_ge77_muons,
        verbose=verbose,
    )

    if verbose:
        print("\nDone.")


if __name__ == "__main__":
    main()