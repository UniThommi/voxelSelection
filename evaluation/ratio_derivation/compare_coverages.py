#!/usr/bin/env python3
"""
PMT Coverage Comparison — N Configurations
===========================================
Compares NC detection efficiency and Ge-77 muon classification across
N PMT configurations using raw LGDO HDF5 simulation data.

This script fully replaces evaluation/comparePMTCoverage.py and extends
it to an arbitrary number of configurations with full M/W sweeps,
heatmaps, and W2 homogeneity analysis.

Usage
-----
    python compare_coverages.py \\
        --muon-dir  /path/to/muon_sim \\
        --sim-dirs  /path/sim_hom /path/sim_opt1 /path/sim_opt2 \\
        --labels    Homogeneous Optimized1 Optimized2 \\
        [--configs  hom.json opt1.json opt2.json]  # optional, for W2 \\
        [--m 1] [--M-max 10] [--W-max 20] \\
        [--M-default 1] [--W-default 1] \\
        [--output-dir ./coverage_results]

Author: Thomas Buerger (University of Tübingen)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Ensure ratio_analysis package is importable when run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ratio_analysis.raw_loading import (
    build_nc_truth,
    build_pmt_matrix,
    check_all_files_integrity,
    count_vertices_by_run,
)
from ratio_analysis.coverage_analysis import (
    evaluate_nc,
    evaluate_muon,
    compute_metrics,
    MUON_WINDOW_LO_NS,
    MUON_WINDOW_HI_NS,
)

# ──────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────
@dataclass
class SetupResult:
    """All analysis results for one PMT configuration."""
    label:    str
    nc:       dict[str, Any]           # from evaluate_nc()
    muon:     dict[str, Any]           # from evaluate_muon()
    pmt_uids: np.ndarray               # det_uid per B column
    w2:       Optional[float] = None   # Wasserstein homogeneity (if config JSON given)


# ──────────────────────────────────────────────────────────────────────
# W2 computation (optional — requires POT and pmtopt package)
# ──────────────────────────────────────────────────────────────────────
def _try_compute_w2(config_json: str) -> Optional[float]:
    """Compute W2 homogeneity from a voxel JSON file. Returns None on failure."""
    try:
        from pmtopt.homogeneous import compute_wasserstein_homogeneity, get_w2_ref
    except ImportError:
        print("  [WARN] pmtopt not importable; W2 will not be computed.")
        return None
    try:
        with open(config_json) as f:
            data = json.load(f)
        voxel_dicts = (
            data if isinstance(data, list)
            else data.get("selected_voxels", [])
        )
        voxel_dicts = [v for v in voxel_dicts if isinstance(v, dict) and "center" in v]
        if len(voxel_dicts) < 2:
            return None
        centers = np.array([v["center"] for v in voxel_dicts], dtype=float)
        return float(compute_wasserstein_homogeneity(centers, reference=get_w2_ref())["w2"])
    except Exception as exc:
        print(f"  [WARN] W2 computation failed for {config_json!r}: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────
_MAX_DET_UIDS = 300


def validate_vertex_counts(sim_dirs: list[str], labels: list[str]) -> None:
    """Verify all setups were simulated from the same primaries.

    Checks that vertex counts per run match across all N setups.
    Raises RuntimeError on any mismatch (same behaviour as
    comparePMTCoverage.py validate_runs()).
    """
    print("Validating vertex counts across setups ...")
    all_counts: list[dict[str, int]] = []
    for sim_dir, label in zip(sim_dirs, labels):
        try:
            all_counts.append(count_vertices_by_run(sim_dir))
        except FileNotFoundError as exc:
            print(f"  [WARN] {label}: {exc}")
            all_counts.append({})

    all_run_labels = sorted(set().union(*(set(c.keys()) for c in all_counts)))
    mismatches = []
    for run_label in all_run_labels:
        counts = {labels[i]: all_counts[i].get(run_label) for i in range(len(labels))}
        present = [v for v in counts.values() if v is not None]
        if len(set(present)) > 1:
            mismatches.append((run_label, counts))

    if mismatches:
        for run_label, counts in mismatches:
            detail = "  ".join(f"{lbl}={v}" for lbl, v in counts.items())
            print(f"  [FAIL] {run_label}: {detail}")
        raise RuntimeError(
            f"{len(mismatches)} run(s) have mismatched vertex counts across setups."
        )
    print(
        f"  [PASS] Vertex counts match across all {len(labels)} setups "
        f"({len(all_run_labels)} run(s))."
    )


def validate_pmt_uids(results: list[SetupResult]) -> None:
    """Warn if any setup has an unexpected PMT count."""
    for r in results:
        n = len(r.pmt_uids)
        if n > _MAX_DET_UIDS:
            print(
                f"  [WARN] {r.label}: {n} unique det_uids > {_MAX_DET_UIDS}. "
                "Possible data mismatch."
            )
        elif n < _MAX_DET_UIDS:
            print(
                f"  [WARN] {r.label}: only {n} unique det_uids "
                f"(expected ~{_MAX_DET_UIDS}). Some PMTs may have detected nothing."
            )
        else:
            print(f"  [PASS] {r.label}: {n} unique det_uids.")


# ──────────────────────────────────────────────────────────────────────
# Shared plot helpers
# ──────────────────────────────────────────────────────────────────────
def _colors(n: int) -> list:
    return [plt.cm.tab10(i % 10) for i in range(n)]


def _annotate_bar(
    ax: plt.Axes, bar, value: int, total: Optional[int] = None, fontsize: int = 8
) -> None:
    text = f"{value:,}"
    if total and total > 0:
        text += f"\n({100 * value / total:.1f}%)"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        text,
        ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
    )


# ──────────────────────────────────────────────────────────────────────
# Plot 01 — NC coverage line (M sweep)
# ──────────────────────────────────────────────────────────────────────
def plot_nc_coverage_line(
    results: list[SetupResult],
    M_values: list[int],
    output_dir: str,
) -> None:
    """Line plot: NC detection fraction vs M for all configs (linear + log)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = _colors(len(results))

    for r, c in zip(results, colors):
        nc_total = r.nc["nc_total"]
        fracs  = [r.nc["nc_detected"][M] / max(nc_total, 1) for M in M_values]
        counts = [r.nc["nc_detected"][M] for M in M_values]
        for ax, y in zip(axes, [fracs, counts]):
            ax.plot(M_values, y, marker="o", color=c, label=r.label)

    # Reference lines for detectability (if available)
    for r, c in zip(results, colors):
        if r.nc["nc_any_photon"] >= 0:
            frac_any = r.nc["nc_any_photon"] / max(r.nc["nc_total"], 1)
            axes[0].axhline(
                frac_any, color=c, linestyle=":", linewidth=1,
                label=f"{r.label} (any photon)",
            )

    for ax, ylabel, title in zip(
        axes,
        ["NC detection fraction", "NC detected (count)"],
        ["NC Coverage vs M threshold (fraction)", "NC Coverage vs M threshold (count)"],
    ):
        ax.set_xlabel("M — minimum firing PMTs per NC")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(M_values)

    fig.tight_layout()
    fname = "01_nc_coverage_line.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 02 — NC multiplicity histogram
# ──────────────────────────────────────────────────────────────────────
def plot_nc_multiplicity_histogram(
    results: list[SetupResult],
    M_default: int,
    output_dir: str,
) -> None:
    """Overlaid histogram of per-NC PMT multiplicity for all configs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = _colors(len(results))

    all_mults = [r.nc["multiplicity_counts"] for r in results]
    max_mult = max(
        (int(m.max()) for m in all_mults if len(m) > 0 and m.max() > 0),
        default=1,
    )
    bins = np.arange(1, max_mult + 2) - 0.5

    for r, c, mult in zip(results, colors, all_mults):
        nonzero = mult[mult > 0]
        ax.hist(
            nonzero, bins=bins, alpha=0.5, label=r.label,
            color=c, edgecolor="black", linewidth=0.4,
        )

    ax.axvline(
        M_default - 0.5, color="red", linestyle="--", linewidth=1.5,
        label=f"M default = {M_default}",
    )
    ax.set_xlabel("PMT multiplicity (# firing PMTs per NC)")
    ax.set_ylabel("Number of NCs")
    ax.set_title(
        "PMT Multiplicity Distribution per NC\n"
        "(# distinct PMTs with ≥m hits within 200 ns of NC; NCs with 0 excluded from bars)"
    )
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Mean and zero-multiplicity stats
    means = ", ".join(
        f"{r.label}: mean={np.mean(m):.2f}" for r, m in zip(results, all_mults)
    )
    zeros = ", ".join(
        f"{r.label}: {int((m == 0).sum()):,}" for r, m in zip(results, all_mults)
    )
    ax.text(
        0.5, -0.12,
        f"Mean multiplicity (all NCs): {means}\n"
        f"NCs with 0 firing PMTs: {zeros}",
        transform=ax.transAxes, ha="center", fontsize=9, fontstyle="italic",
    )

    fig.tight_layout()
    fname = "02_nc_multiplicity_histogram.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plots 03 & 04 — Detectability overview: absolute + Δ from reference
# ──────────────────────────────────────────────────────────────────────
def _draw_detectability_panels(
    ax_abs: plt.Axes,
    ax_delta: plt.Axes,
    results: list[SetupResult],
    nc_key: str,
    M_default: int,
    title: str,
    total_key: str,
) -> None:
    """Draw the two-panel detectability figure on pre-created axes.

    Left panel (ax_abs): grouped vertical bars with absolute counts and
    percentage annotations, one bar group per category for all N setups.

    Right panel (ax_delta): horizontal diverging bars showing the count
    difference Δ = setup_i − setup_0 relative to the first setup (the
    reference).  Each non-reference setup gets its own row of bars per
    category, coloured consistently with the left panel.  Works for any
    N ≥ 1; shows a placeholder message for N = 1.

    The "Total NCs" category is excluded from the Δ panel because it is
    identical across all setups (shared NC truth).
    """
    n = len(results)
    colors = _colors(n)

    only_outside_key = (
        nc_key + "only_outside_200ns" if nc_key == "nc_"
        else nc_key + "only_outside"
    )
    detected_key = (nc_key + "detected", M_default)
    has_detect = results[0].nc.get(nc_key + "any_photon", -1) >= 0

    if has_detect:
        abs_cat_labels = [
            "Total NCs",
            "≥1 photon\n(any time)",
            "≥1 photon\n(within 200 ns)",
            "Photons only\noutside 200 ns",
            f"Detected\n(M≥{M_default})",
        ]
        abs_val_keys = [
            total_key,
            nc_key + "any_photon",
            nc_key + "within_200ns",
            only_outside_key,
            detected_key,
        ]
        delta_cat_labels = [
            "≥1 photon (any time)",
            "≥1 photon (within 200 ns)",
            "Photons only outside 200 ns",
            f"Detected (M≥{M_default})",
        ]
        delta_val_keys = [
            nc_key + "any_photon",
            nc_key + "within_200ns",
            only_outside_key,
            detected_key,
        ]
    else:
        abs_cat_labels = ["Total NCs", f"Detected\n(M≥{M_default})"]
        abs_val_keys   = [total_key, detected_key]
        delta_cat_labels = [f"Detected (M≥{M_default})"]
        delta_val_keys   = [detected_key]

    def _get(r: SetupResult, vk) -> int:
        return r.nc[vk[0]][vk[1]] if isinstance(vk, tuple) else r.nc[vk]

    # ── Left panel: absolute grouped bars ────────────────────────────
    x     = np.arange(len(abs_cat_labels))
    width = min(0.8 / n, 0.30)

    for i, (r, c) in enumerate(zip(results, colors)):
        vals   = [_get(r, vk) for vk in abs_val_keys]
        total  = r.nc[total_key]
        offset = (i - (n - 1) / 2) * width
        bars   = ax_abs.bar(x + offset, vals, width, label=r.label, color=c)
        for bar, val in zip(bars, vals):
            _annotate_bar(ax_abs, bar, val, total, fontsize=7)

    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(abs_cat_labels, fontsize=9)
    ax_abs.set_ylabel("Number of NCs")
    ax_abs.set_title(title)
    ax_abs.legend(fontsize=8)
    ax_abs.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )

    # ── Right panel: Δ horizontal diverging bars ──────────────────────
    if n < 2:
        ax_delta.text(
            0.5, 0.5, "Single setup —\nno Δ to display",
            transform=ax_delta.transAxes, ha="center", va="center",
            fontsize=11, color="gray",
        )
        ax_delta.set_axis_off()
        return

    n_non_ref = n - 1
    n_cats    = len(delta_cat_labels)
    bar_h     = 0.8 / n_non_ref
    y_base    = np.arange(n_cats, dtype=float)
    ref_total = results[0].nc[total_key]
    ref_vals  = [_get(results[0], vk) for vk in delta_val_keys]

    for j, (r, c) in enumerate(zip(results[1:], colors[1:])):
        vals   = [_get(r, vk) for vk in delta_val_keys]
        deltas = [v - rv for v, rv in zip(vals, ref_vals)]
        offset = (j - (n_non_ref - 1) / 2) * bar_h

        bars = ax_delta.barh(
            y_base + offset, deltas, bar_h * 0.88,
            label=r.label, color=c,
        )
        for bar, delta in zip(bars, deltas):
            if delta == 0:
                continue
            pct  = 100.0 * delta / max(ref_total, 1)
            sign = "+" if pct >= 0 else ""
            txt  = f"{sign}{delta:,}\n({sign}{pct:.1f}%)"
            pad  = max(abs(delta) * 0.02, 1)
            ax_delta.text(
                delta + (pad if delta >= 0 else -pad),
                bar.get_y() + bar.get_height() / 2,
                txt,
                va="center",
                ha="left" if delta >= 0 else "right",
                fontsize=7,
                color=c,
                fontweight="bold",
            )

    ax_delta.axvline(0, color="black", linewidth=0.9)
    ax_delta.set_yticks(y_base)
    ax_delta.set_yticklabels(delta_cat_labels, fontsize=9)
    ax_delta.set_xlabel(f"Δ count vs reference ({results[0].label})", fontsize=9)
    ax_delta.set_title(
        f"Δ from reference: {results[0].label}", fontsize=10
    )
    ax_delta.legend(fontsize=8, title="vs reference")
    ax_delta.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):+,}")
    )
    ax_delta.grid(True, axis="x", alpha=0.3)
    ax_delta.invert_yaxis()  # top category at top, matching the left panel order


def _detectability_figure(
    results: list[SetupResult],
    nc_key: str,
    M_default: int,
    title: str,
    total_key: str,
    fname: str,
    output_dir: str,
) -> None:
    """Create and save a two-panel detectability figure."""
    fig, (ax_abs, ax_delta) = plt.subplots(
        1, 2,
        figsize=(20, 7),
        gridspec_kw={"width_ratios": [3, 2]},
    )
    _draw_detectability_panels(
        ax_abs, ax_delta, results, nc_key, M_default, title, total_key
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_nc_detectability_overview(
    results: list[SetupResult],
    M_default: int,
    output_dir: str,
) -> None:
    """Two-panel figure: absolute NC detectability counts + Δ from reference."""
    _detectability_figure(
        results, nc_key="nc_", M_default=M_default,
        title=(
            "NC Detection Overview\n"
            "(total NCs identical for all setups; only muons with ≥1 NC)"
        ),
        total_key="nc_total",
        fname="03_nc_detectability_overview.png",
        output_dir=output_dir,
    )


# ──────────────────────────────────────────────────────────────────────
# Plot 04 — Ge77-muon NC detectability overview
# ──────────────────────────────────────────────────────────────────────
def plot_ge77_nc_detectability_overview(
    results: list[SetupResult],
    M_default: int,
    output_dir: str,
) -> None:
    """Two-panel figure: Ge77-NC detectability counts + Δ from reference."""
    _detectability_figure(
        results, nc_key="ge77_nc_", M_default=M_default,
        title=(
            "Ge-77 Muon NC Detection Overview\n"
            "(NCs belonging to muons with ≥1 Ge-77 producing NC; "
            "only muons with ≥1 NC)"
        ),
        total_key="ge77_nc_total",
        fname="04_ge77_nc_detectability_overview.png",
        output_dir=output_dir,
    )


# ──────────────────────────────────────────────────────────────────────
# Plot 05 — NC coverage line (M sweep) — already plot 01; keeping numbering
# Plot 05 — Multiplicity histogram  — plot 02
# Now: Plot 05 — Muon heatmaps
# ──────────────────────────────────────────────────────────────────────
def _heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    M_values: list[int],
    W_values: list[int],
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
) -> None:
    """Draw a single (M rows, W cols) heatmap on ax."""
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[
            W_values[0] - 0.5, W_values[-1] + 0.5,
            M_values[0] - 0.5, M_values[-1] + 0.5,
        ],
    )
    ax.set_xlabel("W (min detected NCs per muon in [1µs, 200µs])")
    ax.set_ylabel("M (min firing PMTs per NC)")
    ax.set_title(title)
    ax.set_xticks(W_values[::max(1, len(W_values) // 10)])
    ax.set_yticks(M_values)
    plt.colorbar(im, ax=ax)


def plot_muon_heatmaps(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
) -> None:
    """Per-config 3-panel heatmap: Recall, Precision, F2 over (M, W)."""
    metrics_to_plot = ["Recall", "Precision", "F2"]

    for r in results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Ge-77 Classification — {r.label}", fontsize=13)

        for col, metric in enumerate(metrics_to_plot):
            grid = np.zeros((len(M_values), len(W_values)))
            for mi, M in enumerate(M_values):
                for wi, W in enumerate(W_values):
                    conf = r.muon["confusion"][(M, W)]
                    m = compute_metrics(conf["TP"], conf["FP"], conf["TN"], conf["FN"])
                    grid[mi, wi] = m[metric]
            _heatmap(axes[col], grid, M_values, W_values, title=metric)

        fig.tight_layout()
        safe_label = r.label.replace(" ", "_").replace("/", "-")
        fname = f"05_muon_heatmap_{safe_label}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 06 — Confusion bar at fixed (M, W)
# ──────────────────────────────────────────────────────────────────────
def plot_confusion_bar(
    results: list[SetupResult],
    M_default: int,
    W_default: int,
    output_dir: str,
) -> None:
    """Grouped bar: TP, FN, TN, FP at (M_default, W_default) for all configs."""
    n = len(results)
    categories = [
        "True Positive\n(Ge77 → classified)",
        "False Negative\n(Ge77 → missed)",
        "True Negative\n(non-Ge77 → correct)",
        "False Positive\n(non-Ge77 → misclass.)",
    ]
    keys = ("TP", "FN", "TN", "FP")
    x = np.arange(len(categories))
    width = 0.8 / n
    colors = _colors(n)

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (r, c) in enumerate(zip(results, colors)):
        conf = r.muon["confusion"][(M_default, W_default)]
        vals = [conf[k] for k in keys]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=r.label, color=c)
        for bar, val in zip(bars, vals):
            _annotate_bar(ax, bar, val)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Number of Muons")
    ax.set_title(
        f"Ge-77 Muon Classification (W≥{W_default}, M≥{M_default})\n"
        "(only muons with ≥1 NC)"
    )
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Metrics per config
    lines = []
    for r in results:
        conf = r.muon["confusion"][(M_default, W_default)]
        m = compute_metrics(conf["TP"], conf["FP"], conf["TN"], conf["FN"])
        lines.append(
            f"{r.label}: Precision={m['Precision']:.3f}  Recall={m['Recall']:.3f}  "
            f"F2={m['F2']:.3f}"
        )
    ax.text(
        0.5, -0.15, "\n".join(lines),
        transform=ax.transAxes, ha="center", fontsize=9,
        fontstyle="italic", family="monospace",
    )

    fig.tight_layout()
    fname = "06_confusion_bar.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 07 — W histogram at fixed M
# ──────────────────────────────────────────────────────────────────────
def _build_log_bins(max_val: int) -> list[int]:
    """Log-spaced bin edges: 0,1,...,9, 10,20,...,90, 100,... up to max_val."""
    edges: list[int] = list(range(min(max_val + 1, 10)))
    if max_val < 10:
        return edges
    decade = 10
    while decade <= max_val:
        for mult in range(1, 10):
            val = mult * decade
            if val > max_val:
                break
            edges.append(val)
        decade *= 10
    return sorted(set(edges))


def _bin_data(data: list[int], bin_edges: list[int]) -> list[int]:
    edges_arr = np.array(bin_edges)
    data_arr  = np.array(data)
    counts    = [0] * len(bin_edges)
    if len(data_arr) == 0:
        return counts
    indices = np.clip(
        np.searchsorted(edges_arr, data_arr, side="right") - 1,
        0, len(bin_edges) - 1,
    )
    for idx in indices:
        counts[idx] += 1
    return counts


def _draw_w_panel(
    ax: plt.Axes,
    results: list[SetupResult],
    M: int,
    W: int,
    ge77: bool,
) -> None:
    """Draw one W-histogram panel (Ge77 or non-Ge77 muons)."""
    key = "ge77" if ge77 else "non_ge77"
    label = "Ge-77" if ge77 else "Non-Ge-77"
    colors = _colors(len(results))

    all_data = [r.muon["w_hist"][M][key] for r in results]
    max_w = max((max(d) for d in all_data if d), default=0)
    if max_w == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        ax.set_title(f"Detected NC Count per {label} Muon")
        return

    bin_edges = _build_log_bins(max_w)
    x = np.arange(len(bin_edges))
    width = 0.8 / len(results)

    for i, (r, c, d) in enumerate(zip(results, colors, all_data)):
        counts = _bin_data(d, bin_edges)
        offset = (i - (len(results) - 1) / 2) * width
        ax.bar(x + offset, counts, width, label=r.label, color=c,
               edgecolor="black", linewidth=0.4)

    w_bin_idx = np.searchsorted(bin_edges, W, side="right") - 1
    ax.axvline(w_bin_idx - 0.5, color="red", linestyle="--", linewidth=1.5,
               label=f"W threshold = {W}")

    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in bin_edges], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel(f"Detected NCs per Muon in [1µs, 200µs] (M≥{M})")
    ax.set_ylabel("Number of Muons")
    ax.set_title(
        f"Detected NC Count per {label} Muon\n"
        f"(time window [1µs, 200µs], M≥{M}; only muons with ≥1 NC)"
    )
    ax.legend()
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    stats = "    ".join(
        f"{r.label}: {sum(1 for v in d if v >= W)}/{max(len(d), 1)} "
        f"≥W ({100*sum(1 for v in d if v >= W)/max(len(d), 1):.1f}%)"
        for r, d in zip(results, all_data)
    )
    ax.text(0.5, -0.16, stats, transform=ax.transAxes,
            ha="center", fontsize=8, fontstyle="italic")


def plot_w_histogram(
    results: list[SetupResult],
    M_default: int,
    W_default: int,
    output_dir: str,
) -> None:
    """W-histogram at M_default: Ge77 and non-Ge77 muons side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 7))
    _draw_w_panel(axes[0], results, M_default, W_default, ge77=True)
    _draw_w_panel(axes[1], results, M_default, W_default, ge77=False)
    fig.tight_layout()
    fname = "07_w_histogram.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 08 — W2 vs NC coverage scatter (optional)
# ──────────────────────────────────────────────────────────────────────
def plot_w2_scatter(
    results: list[SetupResult],
    M_values: list[int],
    M_default: int,
    W_default: int,
    output_dir: str,
) -> None:
    """Scatter plots: W2 vs NC coverage and W2 vs Recall/F2."""
    w2_results = [r for r in results if r.w2 is not None]
    if len(w2_results) < 2:
        print("  [SKIP] 09_w2_scatter.png: fewer than 2 configs have W2.")
        return

    scatter_M = [M for M in [1, 2, 4, M_default, max(M_values)] if M in M_values]
    scatter_M = sorted(set(scatter_M))

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    colors_m = plt.cm.plasma(np.linspace(0.1, 0.9, len(scatter_M)))
    w2_vals  = np.array([r.w2 for r in w2_results])

    # Panel 1: W2 vs NC coverage at multiple M
    ax = axes[0]
    for M, cm in zip(scatter_M, colors_m):
        cov = np.array([
            r.nc["nc_detected"][M] / max(r.nc["nc_total"], 1)
            for r in w2_results
        ])
        ax.scatter(w2_vals, cov, color=cm, label=f"M={M}", zorder=3, s=60)
        for r, w2, c in zip(w2_results, w2_vals, cov):
            ax.annotate(r.label, xy=(w2, c), xytext=(4, 2),
                        textcoords="offset points", fontsize=7, color=cm)
    ax.set_xlabel("W2 (mm) — lower = more uniform")
    ax.set_ylabel("NC detection fraction")
    ax.set_title("W2 vs NC Coverage")
    ax.legend(title="M")
    ax.grid(True, alpha=0.3)

    # Panel 2: W2 vs Recall at (M_default, W_default)
    ax = axes[1]
    recalls = [
        compute_metrics(
            r.muon["confusion"][(M_default, W_default)]["TP"],
            r.muon["confusion"][(M_default, W_default)]["FP"],
            r.muon["confusion"][(M_default, W_default)]["TN"],
            r.muon["confusion"][(M_default, W_default)]["FN"],
        )["Recall"]
        for r in w2_results
    ]
    ax.scatter(w2_vals, recalls, color="steelblue", s=80, zorder=3)
    for r, w2, rec in zip(w2_results, w2_vals, recalls):
        ax.annotate(r.label, xy=(w2, rec), xytext=(4, 2),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("W2 (mm)")
    ax.set_ylabel("Recall")
    ax.set_title(f"W2 vs Recall (M={M_default}, W={W_default})")
    ax.grid(True, alpha=0.3)

    # Panel 3: W2 vs F2 at (M_default, W_default)
    ax = axes[2]
    f2s = [
        compute_metrics(
            r.muon["confusion"][(M_default, W_default)]["TP"],
            r.muon["confusion"][(M_default, W_default)]["FP"],
            r.muon["confusion"][(M_default, W_default)]["TN"],
            r.muon["confusion"][(M_default, W_default)]["FN"],
        )["F2"]
        for r in w2_results
    ]
    ax.scatter(w2_vals, f2s, color="darkorange", s=80, zorder=3)
    for r, w2, f2 in zip(w2_results, w2_vals, f2s):
        ax.annotate(r.label, xy=(w2, f2), xytext=(4, 2),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("W2 (mm)")
    ax.set_ylabel("F2 score")
    ax.set_title(f"W2 vs F2 (M={M_default}, W={W_default})")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = "09_w2_scatter.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Text output
# ──────────────────────────────────────────────────────────────────────
def write_nc_summary(
    results: list[SetupResult],
    M_values: list[int],
    output_dir: str,
) -> None:
    """Write nc_summary.txt with per-config NC detection statistics."""
    fpath = os.path.join(output_dir, "nc_summary.txt")
    with open(fpath, "w") as f:
        f.write("NC Detection Summary\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            nc = r.nc
            ms = r.muon["muon_stats"]
            f.write(f"--- {r.label} ---\n")
            f.write(f"  Total NCs                  : {nc['nc_total']:,}\n")
            if nc["nc_any_photon"] >= 0:
                f.write(
                    f"  ≥1 photon (any time)       : {nc['nc_any_photon']:,}  "
                    f"({100*nc['nc_any_photon']/max(nc['nc_total'],1):.1f}%)\n"
                )
                f.write(
                    f"  ≥1 photon (within 200 ns)  : {nc['nc_within_200ns']:,}  "
                    f"({100*nc['nc_within_200ns']/max(nc['nc_total'],1):.1f}%)\n"
                )
                f.write(
                    f"  Photons only >200 ns       : {nc['nc_only_outside_200ns']:,}\n"
                )
            f.write(f"  Ge77 NC total              : {nc['ge77_nc_total']:,}\n")
            f.write("\n  NC detected by M threshold:\n")
            for M in M_values:
                det  = nc["nc_detected"][M]
                gdet = nc["ge77_nc_detected"][M]
                frac = 100 * det / max(nc["nc_total"], 1)
                f.write(
                    f"    M={M:2d}:  {det:,}  ({frac:.1f}%)  "
                    f"[Ge77: {gdet:,} / {nc['ge77_nc_total']:,}]\n"
                )
            f.write(
                f"\n  Muons: {ms['total']:,}  "
                f"Ge77: {ms['n_ge77']:,}  "
                f"non-Ge77: {ms['n_non_ge77']:,}\n"
            )
            if r.w2 is not None:
                f.write(f"  W2 homogeneity: {r.w2:.2f} mm\n")
            f.write("\n")
    print("  Saved nc_summary.txt")


def write_confusion_matrices(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
) -> None:
    """Write confusion_matrices.txt for all (config, M, W)."""
    fpath = os.path.join(output_dir, "confusion_matrices.txt")
    with open(fpath, "w") as f:
        f.write("Ge-77 Classification Confusion Matrices\n")
        f.write("=" * 90 + "\n\n")
        hdr = (
            f"{'Config':<20} {'M':>3} {'W':>3}  "
            f"{'TP':>7} {'FP':>7} {'TN':>7} {'FN':>7}  "
            f"{'Recall':>7} {'Prec':>7} {'F2':>7}\n"
        )
        f.write(hdr)
        f.write("-" * len(hdr) + "\n")
        for r in results:
            for M in M_values:
                for W in W_values:
                    conf = r.muon["confusion"][(M, W)]
                    m = compute_metrics(
                        conf["TP"], conf["FP"], conf["TN"], conf["FN"]
                    )
                    f.write(
                        f"{r.label:<20} {M:>3} {W:>3}  "
                        f"{conf['TP']:>7} {conf['FP']:>7} "
                        f"{conf['TN']:>7} {conf['FN']:>7}  "
                        f"{m['Recall']:>7.4f} {m['Precision']:>7.4f} "
                        f"{m['F2']:>7.4f}\n"
                    )
            f.write("\n")
    print("  Saved confusion_matrices.txt")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare NC detection efficiency and Ge-77 classification "
            "across N PMT configurations using raw LGDO HDF5 data. "
            "Fully replaces evaluation/comparePMTCoverage.py."
        )
    )
    parser.add_argument(
        "--muon-dir", required=True,
        help="Directory with Sim 1 NC truth (contains run_NNN subdirs).",
    )
    parser.add_argument(
        "--sim-dirs", nargs="+", required=True, metavar="DIR",
        help="One Sim 2 directory per PMT configuration (contains run_NNN subdirs).",
    )
    parser.add_argument(
        "--labels", nargs="+", required=True, metavar="LABEL",
        help="One label per sim-dir (same count as --sim-dirs).",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None, metavar="JSON",
        help=(
            "Optional: one voxel JSON file per setup, used for W2 computation. "
            "Must match the count of --sim-dirs if provided."
        ),
    )
    parser.add_argument("--m", type=int, default=1,
                        help="Min photon hits per PMT per NC (default: 1).")
    parser.add_argument("--M-max", type=int, default=10,
                        help="Max M for sweep (default: 10).")
    parser.add_argument("--W-max", type=int, default=20,
                        help="Max W for sweep (default: 20).")
    parser.add_argument("--M-default", type=int, default=1,
                        help="Fixed M for confusion/W plots (default: 1).")
    parser.add_argument("--W-default", type=int, default=1,
                        help="Fixed W for confusion plot (default: 1).")
    parser.add_argument(
        "--output-dir", default="./coverage_results",
        help="Output directory (default: ./coverage_results).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── validate argument counts ──────────────────────────────────────
    if len(args.sim_dirs) != len(args.labels):
        print(
            f"ERROR: --sim-dirs ({len(args.sim_dirs)}) and "
            f"--labels ({len(args.labels)}) must have the same count.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.configs is not None and len(args.configs) != len(args.sim_dirs):
        print(
            f"ERROR: --configs ({len(args.configs)}) and "
            f"--sim-dirs ({len(args.sim_dirs)}) must have the same count.",
            file=sys.stderr,
        )
        sys.exit(1)

    M_values  = list(range(1, args.M_max + 1))
    W_values  = list(range(1, args.W_max + 1))
    M_default = min(max(args.M_default, 1), args.M_max)
    W_default = min(max(args.W_default, 1), args.W_max)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("PMT COVERAGE COMPARISON — N CONFIGURATIONS")
    print("=" * 70)
    print(f"  Muon dir : {args.muon_dir}")
    for i, (d, lbl) in enumerate(zip(args.sim_dirs, args.labels)):
        print(f"  Setup {i+1}  : {lbl}  ({d})")
    print(f"  m={args.m}, M=1..{args.M_max}, W=1..{args.W_max}")
    print(f"  M_default={M_default}, W_default={W_default}")
    print(f"  Output   : {args.output_dir}")
    print()

    # ── 0. Integrity pre-flight check ────────────────────────────────
    check_all_files_integrity(args.muon_dir, args.sim_dirs, args.labels)
    print()

    # ── 1. Load shared NC truth ───────────────────────────────────────
    print("Loading NC truth ...")
    nc_truth = build_nc_truth(args.muon_dir, verbose=True)
    print()

    # ── 2. Vertex count validation ────────────────────────────────────
    validate_vertex_counts(args.sim_dirs, args.labels)
    print()

    # ── 3. Process each setup ─────────────────────────────────────────
    results: list[SetupResult] = []

    for i, (sim_dir, label) in enumerate(zip(args.sim_dirs, args.labels)):
        print(f"[{i+1}/{len(args.sim_dirs)}] Processing: {label}")

        B, pmt_uids, detect_info = build_pmt_matrix(
            sim_dir, nc_truth, m_threshold=args.m, verbose=True
        )

        print("  Evaluating NC coverage ...")
        nc_res = evaluate_nc(B, nc_truth, M_values, detect_info=detect_info)

        print("  Evaluating muon classification ...")
        muon_res = evaluate_muon(B, nc_truth, M_values, W_values)

        w2 = None
        if args.configs is not None:
            print("  Computing W2 ...")
            w2 = _try_compute_w2(args.configs[i])
            if w2 is not None:
                print(f"  W2 = {w2:.2f} mm")

        results.append(SetupResult(
            label=label, nc=nc_res, muon=muon_res, pmt_uids=pmt_uids, w2=w2,
        ))

        del B, pmt_uids, detect_info
        gc.collect()
        print()

    # ── 4. Validate PMT uid counts ────────────────────────────────────
    print("Validating PMT uid counts ...")
    validate_pmt_uids(results)
    print()

    # ── 5. Console summary ────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        nc   = r.nc
        ms   = r.muon["muon_stats"]
        conf = r.muon["confusion"][(M_default, W_default)]
        m    = compute_metrics(conf["TP"], conf["FP"], conf["TN"], conf["FN"])
        print(f"\n--- {r.label} ---")
        print(f"  Total NCs:          {nc['nc_total']:,}")
        if nc["nc_any_photon"] >= 0:
            print(f"  Any photon:         {nc['nc_any_photon']:,}  ({100*nc['nc_any_photon']/max(nc['nc_total'],1):.1f}%)")
            print(f"  Within 200 ns:      {nc['nc_within_200ns']:,}  ({100*nc['nc_within_200ns']/max(nc['nc_total'],1):.1f}%)")
            print(f"  Only >200 ns:       {nc['nc_only_outside_200ns']:,}")
        print(f"  Detected (M≥{M_default}):     {nc['nc_detected'][M_default]:,}  ({100*nc['nc_detected'][M_default]/max(nc['nc_total'],1):.1f}%)")
        print(f"  Muons: {ms['total']:,}  Ge77: {ms['n_ge77']:,}  non-Ge77: {ms['n_non_ge77']:,}")
        print(f"  At (M={M_default}, W={W_default}): TP={conf['TP']}  FN={conf['FN']}  TN={conf['TN']}  FP={conf['FP']}")
        print(f"  Recall={m['Recall']:.3f}  Precision={m['Precision']:.3f}  F2={m['F2']:.3f}")
        if r.w2 is not None:
            print(f"  W2: {r.w2:.2f} mm")

    print()

    # ── 6. Generate plots ─────────────────────────────────────────────
    print("Generating plots ...")
    plot_nc_coverage_line(results, M_values, args.output_dir)
    plot_nc_multiplicity_histogram(results, M_default, args.output_dir)
    plot_nc_detectability_overview(results, M_default, args.output_dir)
    plot_ge77_nc_detectability_overview(results, M_default, args.output_dir)
    plot_muon_heatmaps(results, M_values, W_values, args.output_dir)
    plot_confusion_bar(results, M_default, W_default, args.output_dir)
    plot_w_histogram(results, M_default, W_default, args.output_dir)
    plot_w2_scatter(results, M_values, M_default, W_default, args.output_dir)

    # ── 7. Write text files ───────────────────────────────────────────
    print("Writing text summaries ...")
    write_nc_summary(results, M_values, args.output_dir)
    write_confusion_matrices(results, M_values, W_values, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
