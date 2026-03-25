#!/usr/bin/env python3
"""
PMT Configuration Evaluator
============================

Evaluates multiple PMT configurations (given as JSON voxel lists)
against a common SSD dataset. Produces:

  - NC coverage histograms (cumulative, M=0..8) per config
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
    compute_nn_homogeneity,
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

M_VALUES = list(range(1, 21))  # M = 1..20
W_VALUES = list(range(1, 21))  # W = 1..20

# CV plot helpers
_LAYERS_ORDERED = ["pit", "bot", "top", "wall"]
_LAYER_COLORS = {"pit": "#4C72B0", "bot": "#DD8452", "top": "#55A868", "wall": "#C44E52"}
_SCATTER_M_VALUES = [1, 2, 4, 6, 8, 10, 12, 15, 20]


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
        self.cv_per_layer: dict[str, float | None] = {}  # CV of NN spacing per layer
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
# CV computation
# ===================================================================

def compute_config_cv(voxel_dicts: list[dict]) -> dict[str, float | None]:
    """CV of NN distances per layer from voxel dicts (need 'center' and 'layer')."""
    result: dict[str, float | None] = {}
    for layer in _LAYERS_ORDERED:
        layer_voxels = [v for v in voxel_dicts if v.get("layer") == layer]
        if len(layer_voxels) < 2:
            result[layer] = None
            continue
        centers = np.array([v["center"] for v in layer_voxels], dtype=float)
        hom = compute_nn_homogeneity(centers, layer)
        result[layer] = hom["cv"] if hom is not None else None
    return result


def _aggregate_cv(cfg: "ConfigResult") -> float | None:
    """N-weighted mean CV across layers. Returns None if no CV data."""
    if not cfg.cv_per_layer or not cfg.voxel_dicts:
        return None
    total_n = 0
    weighted_sum = 0.0
    for layer, cv in cfg.cv_per_layer.items():
        if cv is None:
            continue
        n = sum(1 for v in cfg.voxel_dicts if v.get("layer") == layer)
        weighted_sum += cv * n
        total_n += n
    return weighted_sum / total_n if total_n > 0 else None


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
# Plotting: NC coverage histogram
# ===================================================================

def plot_nc_coverage(
    configs: list[ConfigResult],
    M_max: int,
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """
    Overview bar chart: for each M (1..M_max), number of NCs with
    coverage >= M. Grouped bars per config.

    Annotations below: total NCs, visible, detected (M=1).
    """
    M_range = list(range(1, M_max + 1))
    num_configs = len(configs)
    num_bars = len(M_range)

    fig, ax = plt.subplots(figsize=(max(12, num_bars * 1.5), 7))

    bar_width = 0.8 / num_configs
    colors = plt.cm.tab10.colors

    for ci, cfg in enumerate(configs):
        counts = [cfg.num_detected.get(M, 0) for M in M_range]

        x_pos = np.arange(num_bars) + ci * bar_width
        bars = ax.bar(
            x_pos, counts, bar_width,
            label=cfg.label,
            color=colors[ci % len(colors)],
            edgecolor="white", linewidth=0.5,
        )

        # Annotate bars with percentage
        for bi, (bar, count) in enumerate(zip(bars, counts)):
            if count == 0:
                continue
            pct_total = count / cfg.num_ncs * 100
            pct_vis = (count / cfg.num_visible * 100
                       if cfg.num_visible > 0 else 0)
            txt = f"{pct_total:.1f}%\n({pct_vis:.1f}%)"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                txt,
                ha="center", va="bottom",
                fontsize=6, rotation=0,
            )

    # X-axis
    ax.set_xticks(np.arange(num_bars) + bar_width * (num_configs - 1) / 2)
    ax.set_xticklabels(
        [f"M ≥ {M}" for M in M_range],
        fontsize=10,
    )
    ax.set_ylabel("Number of NCs", fontsize=12)
    ax.set_title("NC Detection Coverage by Multiplicity Threshold", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")

    # Annotation box below plot
    anno_lines = []
    for cfg in configs:
        line = (
            f"{cfg.label}: "
            f"total={cfg.num_ncs:,}, "
            f"visible={cfg.num_visible:,} "
            f"({cfg.num_visible / cfg.num_ncs:.2%}), "
            f"detected(M=1)={cfg.num_detected.get(1, 0):,} "
            f"({cfg.num_detected.get(1, 0) / cfg.num_ncs:.2%})"
        )
        anno_lines.append(line)

    anno_text = "\n".join(anno_lines)
    fig.text(
        0.5, -0.02, anno_text,
        ha="center", va="top",
        fontsize=8, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                  alpha=0.8),
    )

    plt.tight_layout()
    out_path = output_dir / "nc_coverage_histogram.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


def plot_nc_coverage_per_m(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """
    Per-M bar chart: for each M, one bar per config showing the number
    of NCs with coverage >= M. Annotated with % of total and % of visible.
    """
    num_configs = len(configs)
    colors = plt.cm.tab10.colors
    bar_width = 0.8 / num_configs

    for M in M_values:
        fig, ax = plt.subplots(figsize=(max(6, num_configs * 1.2 + 2), 5))

        for ci, cfg in enumerate(configs):
            count = cfg.num_detected.get(M, 0)
            bar = ax.bar(
                ci * bar_width,
                count,
                bar_width,
                label=cfg.label,
                color=colors[ci % len(colors)],
                edgecolor="white", linewidth=0.5,
            )
            if count > 0:
                pct_total = count / cfg.num_ncs * 100
                pct_vis = (count / cfg.num_visible * 100
                           if cfg.num_visible > 0 else 0)
                ax.text(
                    ci * bar_width + bar_width / 2,
                    count,
                    f"{count:,}\n{pct_total:.1f}%\n({pct_vis:.1f}%)",
                    ha="center", va="bottom",
                    fontsize=8,
                )

        ax.set_xticks(np.arange(num_configs) * bar_width + bar_width / 2)
        ax.set_xticklabels([cfg.label for cfg in configs],
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Number of NCs", fontsize=11)
        ax.set_title(f"NC Detection Coverage  M ≥ {M}", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        out_path = output_dir / f"nc_coverage_M{M}.png"
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
    Per M: one figure with 3 heatmaps (Accuracy, Precision).
    Axes: W (y) x Config (x). Cell text = value.
    """
    config_labels = [c.label for c in configs]
    num_configs = len(configs)

    for M in M_values:
        fig, axes = plt.subplots(
            1, 3, figsize=(max(9, num_configs * 2.4 + 2), len(W_values) * 0.8 + 3),
            sharey=True,
        )

        metric_names = ["Accuracy", "Precision", "Specificity"]
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

                    if metric_name == "Accuracy":
                        denom = TP + FP + TN + FN
                        val = (TP + TN) / denom if denom > 0 else 0.0
                    elif metric_name == "Precision":
                        denom = TP + FP
                        val = TP / denom if denom > 0 else 0.0
                    elif metric_name == "Specificity":
                        denom = TN + FP
                        val = TN / denom if denom > 0 else 0.0
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
            f"Ge77 muons: {num_ge77_muons:,} / {total_muons:,} total",
            fontsize=12, y=1.02,
        )
        fig.colorbar(im, ax=axes, shrink=0.8, label="Value")
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
) -> None:
    """Write confusion matrices to text file."""
    out_path = output_dir / "confusion_matrices.txt"
    with open(out_path, "w") as f:
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
) -> None:
    """Write NC detection summary to text file."""
    out_path = output_dir / "nc_summary.txt"
    with open(out_path, "w") as f:
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
# Plotting: CV vs coverage (Options 1 / 1b / 2 / 3)
# ===================================================================

def plot_coverage_cv_overview(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """Option 1: Two-panel — coverage(M) line curves (top) + CV grouped bars per layer (bottom)."""
    tab_colors = plt.cm.tab10.colors
    cv_configs = [(ci, cfg) for ci, cfg in enumerate(configs) if cfg.cv_per_layer]

    fig, (ax_cov, ax_cv) = plt.subplots(
        2, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # --- Top panel: coverage curves ---
    for ci, cfg in enumerate(configs):
        ys = [cfg.num_detected.get(M, 0) for M in M_values]
        agg_cv = _aggregate_cv(cfg)
        lbl = cfg.label + (f" (CV={agg_cv:.3f})" if agg_cv is not None else "")
        ax_cov.plot(
            M_values, ys, marker="o", markersize=3, linewidth=1.5,
            color=tab_colors[ci % len(tab_colors)], label=lbl,
        )
    ax_cov.set_yscale("log")
    ax_cov.set_xlabel("Multiplicity threshold M", fontsize=11)
    ax_cov.set_ylabel("Detected NCs", fontsize=11)
    ax_cov.set_title("NC Detection Coverage vs Multiplicity Threshold", fontsize=12)
    ax_cov.set_xticks(M_values)
    ax_cov.legend(fontsize=8, loc="upper right")
    ax_cov.grid(axis="y", alpha=0.3)
    ax_cov.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # --- Bottom panel: CV grouped bars per layer ---
    if not cv_configs:
        ax_cv.text(0.5, 0.5, "No CV data available", transform=ax_cv.transAxes, ha="center")
    else:
        n_cfg = len(cv_configs)
        n_lay = len(_LAYERS_ORDERED)
        bar_width = 0.8 / n_lay
        x_pos = np.arange(n_cfg)

        for li, layer in enumerate(_LAYERS_ORDERED):
            cv_vals = [
                cfg.cv_per_layer.get(layer) or 0.0
                for _, cfg in cv_configs
            ]
            ax_cv.bar(
                x_pos + li * bar_width - (n_lay - 1) * bar_width / 2,
                cv_vals, bar_width,
                label=layer, color=_LAYER_COLORS[layer],
                edgecolor="white", linewidth=0.5, alpha=0.85,
            )

        ax_cv.set_xticks(x_pos)
        ax_cv.set_xticklabels(
            [cfg.label for _, cfg in cv_configs], rotation=20, ha="right", fontsize=9,
        )
        ax_cv.set_ylabel("CV (NN homogeneity)", fontsize=10)
        ax_cv.set_title("Nearest-Neighbour Spacing CV per Layer", fontsize=11)
        ax_cv.legend(title="Layer", fontsize=8, loc="upper right")
        ax_cv.grid(axis="y", alpha=0.3)
        ax_cv.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = output_dir / "cv_coverage_overview.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


def plot_coverage_cv_per_layer(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """Option 1b: 2×2 subplots, one per layer — coverage curves (left axis, log)
    with that layer's CV as a horizontal dashed line per config (right axis)."""
    tab_colors = plt.cm.tab10.colors
    cv_configs = [(ci, cfg) for ci, cfg in enumerate(configs) if cfg.cv_per_layer]

    if not cv_configs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes_flat = axes.flatten()

    for li, layer in enumerate(_LAYERS_ORDERED):
        ax = axes_flat[li]
        ax2 = ax.twinx()

        # Coverage curves for ALL configs (left axis)
        for ci, cfg in enumerate(configs):
            color = tab_colors[ci % len(tab_colors)]
            ys = [cfg.num_detected.get(M, 0) for M in M_values]
            ax.plot(
                M_values, ys, color=color, linewidth=1.5,
                marker="o", markersize=2,
                label=cfg.label,
            )

        # CV horizontal lines — only configs with CV (right axis)
        cv_vals_layer = []
        for ci, cfg in cv_configs:
            cv = cfg.cv_per_layer.get(layer)
            if cv is None:
                continue
            color = tab_colors[ci % len(tab_colors)]
            ax2.axhline(cv, color=color, linestyle="--", linewidth=1.4, alpha=0.85)
            cv_vals_layer.append(cv)

        ax.set_yscale("log")
        ax.set_xlabel("M", fontsize=9)
        ax.set_ylabel("Detected NCs (log)", fontsize=9)
        ax.set_title(f"Layer: {layer.upper()}", fontsize=11)
        ax.set_xticks(M_values[::2])
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.25)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

        ax2.set_ylabel("CV (NN spacing)  — dashed", color="gray", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="gray")
        ax2.set_ylim(bottom=0, top=(max(cv_vals_layer) * 1.6 + 0.02) if cv_vals_layer else 1.0)

    # Shared legend from first subplot
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=min(len(configs), 5),
        fontsize=8, bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(
        "NC Coverage vs M per Layer  "
        "(solid lines = detected NCs [left axis];  dashed lines = CV [right axis])",
        fontsize=11, y=1.04,
    )
    plt.tight_layout()
    out_path = output_dir / "cv_coverage_per_layer.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


def plot_coverage_ratio_by_cv(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """Option 2: Coverage retention ratio detected(M)/detected(M=1) vs M,
    lines colored by aggregate N-weighted CV."""
    cv_configs = [(ci, cfg) for ci, cfg in enumerate(configs) if cfg.cv_per_layer]
    valid_cvs = [c for c in (_aggregate_cv(cfg) for _, cfg in cv_configs) if c is not None]

    if not valid_cvs:
        return

    cv_min, cv_max = min(valid_cvs), max(valid_cvs)
    norm = plt.Normalize(vmin=cv_min, vmax=cv_max if cv_max > cv_min else cv_min + 1e-6)
    cmap = plt.cm.viridis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(13, 7))

    # Configs without CV in light gray (e.g. all_voxels reference)
    for ci, cfg in enumerate(configs):
        if cfg.cv_per_layer:
            continue
        det1 = cfg.num_detected.get(1, 1) or 1
        ratios = [cfg.num_detected.get(M, 0) / det1 for M in M_values]
        ax.plot(M_values, ratios, color="lightgray", linewidth=1.0,
                linestyle=":", label=cfg.label, zorder=1)

    for ci, cfg in cv_configs:
        agg_cv = _aggregate_cv(cfg)
        if agg_cv is None:
            continue
        det1 = cfg.num_detected.get(1, 1) or 1
        ratios = [cfg.num_detected.get(M, 0) / det1 for M in M_values]
        color = cmap(norm(agg_cv))
        ax.plot(
            M_values, ratios, color=color, linewidth=2.0,
            marker="o", markersize=3,
            label=f"{cfg.label}  (CV={agg_cv:.3f})",
            zorder=2,
        )

    ax.set_xlabel("Multiplicity threshold M", fontsize=11)
    ax.set_ylabel("Coverage retention  detected(M) / detected(M=1)", fontsize=11)
    ax.set_title(
        "NC Coverage Retention vs M — colored by Aggregate NN-Spacing CV\n"
        "(CV = N-weighted mean of per-layer std/mean of nearest-neighbour distances)",
        fontsize=11,
    )
    ax.set_xticks(M_values)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(sm, ax=ax, label="Aggregate CV (N-weighted)")

    plt.tight_layout()
    out_path = output_dir / "cv_coverage_ratio_by_cv.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved: {out_path}")


def plot_cv_coverage_scatter(
    configs: list[ConfigResult],
    M_values: list[int],
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """Option 3: One subplot per selected M — aggregate CV (x) vs detected NCs (y).
    Each setup shown as a vertical stem (line from 0 up to coverage) + labelled dot."""
    tab_colors = plt.cm.tab10.colors
    cv_cfg_pairs = [
        (ci, cfg) for ci, cfg in enumerate(configs)
        if cfg.cv_per_layer and _aggregate_cv(cfg) is not None
    ]

    if len(cv_cfg_pairs) < 2:
        return

    scatter_ms = [M for M in _SCATTER_M_VALUES if M in set(M_values)]
    if not scatter_ms:
        scatter_ms = M_values[:9]

    ncols = 3
    nrows = (len(scatter_ms) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes_flat = np.array(axes).flatten()

    for pi, M in enumerate(scatter_ms):
        ax = axes_flat[pi]

        for ci, cfg in cv_cfg_pairs:
            color = tab_colors[ci % len(tab_colors)]
            cv = _aggregate_cv(cfg)
            count = cfg.num_detected.get(M, 0)
            # Vertical stem from y=0 to coverage
            ax.plot([cv, cv], [0, count], color=color, linewidth=1.8, alpha=0.7)
            ax.scatter([cv], [count], color=color, s=50, zorder=3)
            ax.annotate(
                cfg.label, (cv, count),
                textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=7, color=color,
            )

        ax.set_xlabel("Aggregate CV", fontsize=9)
        ax.set_ylabel("Detected NCs", fontsize=9)
        ax.set_title(f"M ≥ {M}", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    for pi in range(len(scatter_ms), len(axes_flat)):
        axes_flat[pi].set_visible(False)

    fig.suptitle(
        "NC Coverage vs Aggregate NN-Spacing CV  "
        "(vertical stem per setup, one panel per M threshold)",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = output_dir / "cv_coverage_scatter.png"
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
    parser.add_argument("--M-max", type=int, default=20,
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
    cli_ratio_flags = {
        k for k, v in {"pit": args.pit, "bot": args.bot,
                        "top": args.top, "wall": args.wall}.items()
        if v is not None
    }
    json_ratios = bl_data.get("config", {}).get("area_ratios", {})
    if json_ratios:
        if cli_ratio_flags:
            raise RuntimeError(
                f"Baseline JSON '{args.baseline}' already contains "
                f"area_ratios — do not also pass "
                f"--{'/--'.join(sorted(cli_ratio_flags))}. "
                f"Remove the CLI ratio flags to use the JSON values."
            )
        area_ratios = dict(DEFAULT_AREA_RATIOS)
        area_ratios.update(json_ratios)
        if verbose:
            print(f"  Area ratios from baseline JSON (applied via stochastic "
                  f"rounding to raw HDF5):")
            for layer, ratio in area_ratios.items():
                src = "(from JSON)" if layer in json_ratios else "(default)"
                print(f"    {layer:>6}: {ratio}  {src}")
    else:
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
    baseline.cv_per_layer = compute_config_cv(bl_voxel_dicts)
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
        cr.cv_per_layer = compute_config_cv(voxel_dicts)
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

    # NC histogram (overview)
    plot_nc_coverage(all_configs, args.M_max, output_dir, verbose=verbose)

    # NC histogram per M
    plot_nc_coverage_per_m(all_configs, M_values, output_dir, verbose=verbose)

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
    )
    write_nc_summary_txt(all_configs, M_values, output_dir, verbose=verbose)

    # CV vs coverage plots
    plot_coverage_cv_overview(all_configs, M_values, output_dir, verbose=verbose)
    plot_coverage_cv_per_layer(all_configs, M_values, output_dir, verbose=verbose)
    plot_coverage_ratio_by_cv(all_configs, M_values, output_dir, verbose=verbose)
    plot_cv_coverage_scatter(all_configs, M_values, output_dir, verbose=verbose)

    if verbose:
        print("\nDone.")


if __name__ == "__main__":
    main()