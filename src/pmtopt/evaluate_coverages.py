#!/usr/bin/env python3
"""
PMT Configuration Evaluator
============================

Evaluates multiple PMT configurations (given as JSON voxel lists)
against a common SSD dataset. Produces:

  - NC coverage histograms (cumulative, M=0..8) per config
  - Ge77 muon heatmaps (Accuracy, Precision, Recall) for W x Config, per M
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
from pmtopt.data_loading import (
    load_raw_sparse,
    binarize_from_raw,
    load_muon_data,
    build_muon_index,
)


# ===================================================================
# Constants
# ===================================================================

M_VALUES = list(range(1, 9))  # M = 1..8
W_VALUES = list(range(1, 9))  # W = 1..8


# ===================================================================
# Data structures
# ===================================================================

class ConfigResult:
    """Stores evaluation results for one PMT configuration."""

    def __init__(self, name: str, voxel_ids: list[str], label: str):
        self.name = name
        self.label = label
        self.voxel_ids = voxel_ids
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

def load_config_json(json_path: str) -> tuple[list[str], dict]:
    """Load voxel IDs and full config from a greedy result JSON.

    Supports two formats:
    - Greedy result: dict with ``selected_voxels`` list of objects with
      ``index`` keys, and optional ``config`` metadata.
    - Plain list: a JSON array of voxel ID strings.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        voxel_ids = [
            v["index"] if isinstance(v, dict) else v for v in data
        ]
        data = {}
    else:
        voxels = data.get("selected_voxels", [])
        voxel_ids = [v["index"] for v in voxels]

    if len(voxel_ids) == 0:
        raise ValueError(f"No voxels found in {json_path}")

    return voxel_ids, data


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
    Bar chart: for each M (0..M_max), number of NCs with coverage >= M.
    M=0 shows total NCs (= all NCs). Grouped bars per config.

    Annotations below: total NCs, visible, detected (M=1).
    """
    M_range = list(range(0, M_max + 1))
    num_configs = len(configs)
    num_bars = len(M_range)

    fig, ax = plt.subplots(figsize=(max(12, num_bars * 1.5), 7))

    bar_width = 0.8 / num_configs
    colors = plt.cm.tab10.colors

    for ci, cfg in enumerate(configs):
        counts = []
        for M in M_range:
            if M == 0:
                counts.append(cfg.num_ncs)
            else:
                counts.append(cfg.num_detected.get(M, 0))

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
            M_val = M_range[bi]
            pct_total = count / cfg.num_ncs * 100
            if M_val == 0:
                txt = f"{count:,}"
            else:
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
        ["All NCs"] + [f"M ≥ {M}" for M in M_range[1:]],
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
    Per M: one figure with 3 heatmaps (Accuracy, Precision, Recall).
    Axes: W (y) x Config (x). Cell text = value.
    """
    config_labels = [c.label for c in configs]
    num_configs = len(configs)

    for M in M_values:
        fig, axes = plt.subplots(
            1, 3, figsize=(max(6, num_configs * 1.8 + 2), len(W_values) * 0.8 + 3),
            sharey=True,
        )

        metric_names = ["Accuracy", "Precision", "Recall"]
        cmaps = ["YlGn", "YlOrRd", "PuBu"]

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
                    elif metric_name == "Recall":
                        denom = TP + FN
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
                  f"{'Acc':>8} {'Prec':>8} {'Rec':>8}\n")
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
                    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
                    f.write(
                        f"{cfg.label:<25} {M:>3} {W:>3} "
                        f"{TP:>8} {FP:>8} {TN:>10} {FN:>8} "
                        f"{acc:>8.4f} {prec:>8.4f} {rec:>8.4f}\n"
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
    parser.add_argument("--M-max", type=int, default=8,
                        help="Maximum M value to evaluate.")
    parser.add_argument("--W-max", type=int, default=8,
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
    bl_voxels, bl_data = load_config_json(args.baseline)

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
        voxel_ids, cfg_data = load_config_json(cfg_path)
        if args.labels:
            label = args.labels[i]
        else:
            label = Path(cfg_path).stem
        cr = ConfigResult(
            name=Path(cfg_path).stem,
            voxel_ids=voxel_ids,
            label=label,
        )
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

    # NC histogram
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
    )
    write_nc_summary_txt(all_configs, M_values, output_dir, verbose=verbose)

    if verbose:
        print("\nDone.")


if __name__ == "__main__":
    main()