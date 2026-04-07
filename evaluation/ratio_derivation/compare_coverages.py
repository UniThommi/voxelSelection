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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Ensure ratio_analysis package is importable when run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Also make src/pmtopt importable (needed for the shared W2 plotting helper).
_pmtopt_src = Path(__file__).resolve().parents[2] / "src"
if _pmtopt_src.is_dir() and str(_pmtopt_src) not in sys.path:
    sys.path.insert(0, str(_pmtopt_src))

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

from scipy import stats as scipy_stats

# Shared W2-correlation plotting helper (identical across both pipelines).
from pmtopt.w2_plot_helpers import regression_overlay as _regression_overlay

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


def validate_vertex_counts(sim_dirs: list[str], labels: list[str], omit_runs: set[str] | None = None) -> None:
    """Verify all setups were simulated from the same primaries.

    Checks that vertex counts per run match across all N setups.
    Raises RuntimeError on any mismatch (same behaviour as
    comparePMTCoverage.py validate_runs()).
    """
    print("Validating vertex counts across setups ...")
    all_counts: list[dict[str, int]] = []
    for sim_dir, label in zip(sim_dirs, labels):
        try:
            all_counts.append(count_vertices_by_run(sim_dir, omit_runs=omit_runs))
        except FileNotFoundError as exc:
            print(f"  [WARN] {label}: {exc}")
            all_counts.append({})

    all_run_labels = sorted(set().union(*(set(c.keys()) for c in all_counts)))
    mismatches = []
    for run_label in all_run_labels:
        counts = {labels[i]: all_counts[i].get(run_label) for i in range(len(labels))}
        present = [v for v in counts.values() if v is not None]
        # Flag if any setup is missing the run OR if counts differ across setups.
        if len(present) < len(counts) or len(set(present)) > 1:
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

# Qualitative palette: 10 clearly distinguishable colours ordered so that
# adjacent entries have maximum hue contrast.  Used for ALL non-heatmap
# plots so every setup is always represented by the same colour.
_SETUP_PALETTE: list[str] = [
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
]


def _colors(n: int) -> list[str]:
    """Return a list of n setup colours, cycling _SETUP_PALETTE if n > 10."""
    return [_SETUP_PALETTE[i % len(_SETUP_PALETTE)] for i in range(n)]


def _setup_color(results: list, r) -> str:
    """Return the palette colour for SetupResult r relative to the full list."""
    return _SETUP_PALETTE[results.index(r) % len(_SETUP_PALETTE)]


def _annotate_bar(
    ax: plt.Axes, bar, value: int, total: Optional[int] = None,
    fontsize: int = 8, rotation: int = 0,
) -> None:
    text = f"{value:,}"
    if total and total > 0:
        text += f"\n({100 * value / total:.1f}%)"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        text,
        ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
        rotation=rotation,
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
    """Overlaid histogram of per-NC PMT multiplicity: linear (left) + log (right)."""
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(20, 6))
    colors = _colors(len(results))

    all_mults = [r.nc["multiplicity_counts"] for r in results]
    max_mult = max(
        (int(m.max()) for m in all_mults if len(m) > 0 and m.max() > 0),
        default=1,
    )
    bins = np.arange(1, max_mult + 2) - 0.5

    for r, c, mult in zip(results, colors, all_mults):
        nonzero = mult[mult > 0]
        for ax in (ax_lin, ax_log):
            ax.hist(
                nonzero, bins=bins, alpha=0.5, label=r.label,
                color=c, edgecolor="black", linewidth=0.4,
            )

    for ax, scale, scale_label in zip(
        (ax_lin, ax_log), ("linear", "log"), ("Linear scale", "Log scale")
    ):
        ax.axvline(
            M_default - 0.5, color="red", linestyle="--", linewidth=1.5,
            label=f"M default = {M_default}",
        )
        ax.set_xlabel("PMT multiplicity (# firing PMTs per NC)")
        ax.set_ylabel("Number of NCs")
        ax.set_title(
            f"PMT Multiplicity Distribution per NC — {scale_label}\n"
            "(# distinct PMTs with ≥m hits within 200 ns of NC; NCs with 0 excluded)"
        )
        ax.legend(fontsize=8)
        ax.set_yscale(scale)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Mean and zero-multiplicity stats beneath the figure
    means = ", ".join(
        f"{r.label}: mean={np.mean(m[m > 0]):.2f}" if (m > 0).any() else f"{r.label}: mean=N/A"
        for r, m in zip(results, all_mults)
    )
    zeros = ", ".join(
        f"{r.label}: {int((m == 0).sum()):,}" for r, m in zip(results, all_mults)
    )
    fig.text(
        0.5, 0.01,
        f"Mean multiplicity (non-zero NCs only): {means}\n"
        f"NCs with 0 firing PMTs: {zeros}",
        ha="center", fontsize=8, fontstyle="italic",
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])
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
    ylabel: str = "Number of NCs",
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
    has_detect = all(r.nc.get(nc_key + "any_photon", -1) >= 0 for r in results)

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
            _annotate_bar(ax_abs, bar, val, total, fontsize=6, rotation=90)

    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(abs_cat_labels, fontsize=9)
    ax_abs.set_ylabel(ylabel)
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
            pct  = 100.0 * delta / max(ref_total, 1)
            sign = "+" if pct >= 0 else ""
            if delta == 0:
                # Bar has zero width; annotate at x=0 to make it visible
                ax_delta.text(
                    0, bar.get_y() + bar.get_height() / 2,
                    " =ref",
                    va="center", ha="left", fontsize=6,
                    color=c, fontstyle="italic",
                )
                continue
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
    ylabel: str = "Number of NCs",
    note: str = "",
) -> None:
    """Create and save a two-panel detectability figure."""
    fig, (ax_abs, ax_delta) = plt.subplots(
        1, 2,
        figsize=(22, 9),
        gridspec_kw={"width_ratios": [3, 2]},
    )
    _draw_detectability_panels(
        ax_abs, ax_delta, results, nc_key, M_default, title, total_key,
        ylabel=ylabel,
    )
    if note:
        fig.text(
            0.5, 0.01, note,
            ha="center", fontsize=8, fontstyle="italic",
            color="#444444", wrap=True,
        )
        fig.tight_layout(rect=[0, 0.06, 1, 1])
    else:
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
# Plot 04 — Ge77-muon detection funnel (muon-level, consistent with heatmap)
# ──────────────────────────────────────────────────────────────────────
def plot_ge77_muon_overview(
    results: list[SetupResult],
    M_default: int,
    W_default: int,
    output_dir: str,
) -> None:
    """Two-panel figure: Ge77-muon detection funnel + Δ from reference.

    Uses the same muon-level definition as the heatmap Recall:
      - A muon is Ge77 if ANY of its NCs has flag_ge77 == 1.
      - A muon is "detected" if it has ≥W NCs with ≥M firing PMTs
        in the [1 µs, 200 µs] time window.

    Bar categories (left to right):
      Total Ge77 muons
      → ≥1 NC has any photon   (upper bound, if detect_info available)
      → ≥1 NC has photon ≤200 ns after NC (tighter bound)
      → Detected (TP at M_default, W_default)  ← identical to heatmap TP
    """
    n = len(results)
    colors = _colors(n)
    has_photon_info = all(
        r.muon["ge77_muon_detectability"]["any_photon"] >= 0 for r in results
    )

    if has_photon_info:
        cat_labels = [
            "Total\nGe77 muons",
            "≥1 NC:\nany photon",
            "≥1 NC:\nphoton ≤200 ns\nafter NC",
            f"Detected\n(M≥{M_default}, W≥{W_default})\n= Recall TP",
        ]
        delta_cat_labels = cat_labels[1:]  # skip Total in delta panel

        def _vals(r: SetupResult) -> list[int]:
            gd = r.muon["ge77_muon_detectability"]
            tp = r.muon["confusion"][(M_default, W_default)]["TP"]
            return [r.muon["muon_stats"]["n_ge77"], gd["any_photon"], gd["within_200ns"], tp]
    else:
        cat_labels = [
            "Total\nGe77 muons",
            f"Detected\n(M≥{M_default}, W≥{W_default})\n= Recall TP",
        ]
        delta_cat_labels = cat_labels[1:]

        def _vals(r: SetupResult) -> list[int]:
            tp = r.muon["confusion"][(M_default, W_default)]["TP"]
            return [r.muon["muon_stats"]["n_ge77"], tp]

    x = np.arange(len(cat_labels))
    width = min(0.8 / n, 0.30)

    fig, (ax_abs, ax_delta) = plt.subplots(
        1, 2, figsize=(22, 9), gridspec_kw={"width_ratios": [3, 2]},
    )

    # ── Left panel: absolute grouped bars ────────────────────────────
    for i, (r, c) in enumerate(zip(results, colors)):
        vals   = _vals(r)
        total  = r.muon["muon_stats"]["n_ge77"]
        offset = (i - (n - 1) / 2) * width
        bars   = ax_abs.bar(x + offset, vals, width, label=r.label, color=c)
        for bar, val in zip(bars, vals):
            _annotate_bar(ax_abs, bar, val, total, fontsize=6, rotation=90)

    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(cat_labels, fontsize=9)
    ax_abs.set_ylabel("Number of Ge77 muons")
    ax_abs.set_title(
        f"Ge-77 Muon Detection Funnel  [M={M_default}, W={W_default}]\n"
        "(muon is Ge77 if any NC has flag_ge77==1; "
        "'Detected' = ≥W NCs with ≥M firing PMTs in [1 µs, 200 µs])"
    )
    ax_abs.legend(fontsize=8)
    ax_abs.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # ── Right panel: Δ horizontal diverging bars ──────────────────────
    if n < 2:
        ax_delta.text(
            0.5, 0.5, "Single setup —\nno Δ to display",
            transform=ax_delta.transAxes, ha="center", va="center",
            fontsize=11, color="gray",
        )
        ax_delta.set_axis_off()
    else:
        n_non_ref  = n - 1
        n_cats     = len(delta_cat_labels)
        bar_h      = 0.8 / n_non_ref
        y_base     = np.arange(n_cats, dtype=float)
        ref_vals   = _vals(results[0])[1:]   # skip Total
        ref_total  = results[0].muon["muon_stats"]["n_ge77"]

        for j, (r, c) in enumerate(zip(results[1:], colors[1:])):
            vals   = _vals(r)[1:]   # skip Total
            deltas = [v - rv for v, rv in zip(vals, ref_vals)]
            offset = (j - (n_non_ref - 1) / 2) * bar_h

            bars = ax_delta.barh(y_base + offset, deltas, bar_h * 0.88, label=r.label, color=c)
            for bar, delta in zip(bars, deltas):
                pct  = 100.0 * delta / max(ref_total, 1)
                sign = "+" if delta >= 0 else ""
                if delta == 0:
                    ax_delta.text(
                        0, bar.get_y() + bar.get_height() / 2,
                        " =ref", va="center", ha="left", fontsize=6,
                        color=c, fontstyle="italic",
                    )
                    continue
                txt = f"{sign}{delta:,}\n({sign}{pct:.1f}%)"
                pad = max(abs(delta) * 0.02, 1)
                ax_delta.text(
                    delta + (pad if delta >= 0 else -pad),
                    bar.get_y() + bar.get_height() / 2,
                    txt, va="center",
                    ha="left" if delta >= 0 else "right",
                    fontsize=7, color=c, fontweight="bold",
                )

        ax_delta.axvline(0, color="black", linewidth=0.9)
        ax_delta.set_yticks(y_base)
        ax_delta.set_yticklabels(delta_cat_labels, fontsize=9)
        ax_delta.set_xlabel(f"Δ muon count vs reference ({results[0].label})", fontsize=9)
        ax_delta.set_title(f"Δ from reference: {results[0].label}", fontsize=10)
        ax_delta.legend(fontsize=8, title="vs reference")
        ax_delta.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):+,}"))
        ax_delta.grid(True, axis="x", alpha=0.3)
        ax_delta.invert_yaxis()

    fig.tight_layout()
    fname = "04_ge77_muon_overview.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 05 — NC coverage line (M sweep) — already plot 01; keeping numbering
# Plot 05 — Multiplicity histogram  — plot 02
# Now: Plot 05 — Muon heatmaps
# ──────────────────────────────────────────────────────────────────────
def plot_muon_heatmaps(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
) -> None:
    """One figure per M: 2-panel heatmap (Recall, Precision) with x=setup, y=W.

    Both metric panels share the same logarithmic colour scale.
    """
    metrics_to_plot = ["Recall", "Precision"]
    n_setups = len(results)
    labels = [r.label for r in results]

    # Shared LogNorm across all M values and all metrics (0–1 range)
    shared_norm = mcolors.LogNorm(vmin=1e-4, vmax=1.0)

    for M in M_values:
        # Build grids: shape (len(W_values), n_setups) for each metric
        grids: dict[str, np.ndarray] = {}
        for metric in metrics_to_plot:
            grid = np.zeros((len(W_values), n_setups))
            for wi, W in enumerate(W_values):
                for si, r in enumerate(results):
                    conf = r.muon["confusion"][(M, W)]
                    m = compute_metrics(conf["TP"], conf["FP"], conf["TN"], conf["FN"])
                    grid[wi, si] = max(m[metric], 1e-4)
            grids[metric] = grid

        fig_w = max(6, 2 + 2 * n_setups)
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, 6))
        fig.suptitle(f"Ge-77 Classification  M={M}", fontsize=13)

        for col, metric in enumerate(metrics_to_plot):
            ax = axes[col]
            grid = grids[metric]
            im = ax.imshow(
                grid,
                origin="lower",
                aspect="auto",
                cmap="viridis",
                norm=shared_norm,
                extent=[
                    -0.5, n_setups - 0.5,
                    W_values[0] - 0.5, W_values[-1] + 0.5,
                ],
            )
            ax.set_xticks(range(n_setups))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(W_values[::max(1, len(W_values) // 10)])
            ax.set_xlabel("Setup")
            ax.set_ylabel("W (min detected NCs per muon)")
            ax.set_title(metric)
            plt.colorbar(im, ax=ax, label=metric)

        fig.tight_layout()
        fname = f"05_muon_heatmap_M{M:02d}.png"
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

    fig, ax = plt.subplots(figsize=(16, 8))
    for i, (r, c) in enumerate(zip(results, colors)):
        conf = r.muon["confusion"][(M_default, W_default)]
        vals = [conf[k] for k in keys]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=r.label, color=c)
        for bar, val in zip(bars, vals):
            _annotate_bar(ax, bar, val, fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Number of Muons")
    ax.set_title(
        f"Ge-77 Muon Classification (W≥{W_default}, M≥{M_default})\n"
        "(only muons with ≥1 NC)"
    )
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Metrics per config — placed below the axis with extra room
    lines = []
    for r in results:
        conf = r.muon["confusion"][(M_default, W_default)]
        m = compute_metrics(conf["TP"], conf["FP"], conf["TN"], conf["FN"])
        lines.append(
            f"{r.label}: Recall={m['Recall']:.3f}  Precision={m['Precision']:.3f}"
        )
    n_lines = len(lines)
    fig.text(
        0.5, 0.01, "\n".join(lines),
        ha="center", fontsize=8,
        fontstyle="italic", family="monospace",
    )

    fig.tight_layout(rect=[0, 0.04 + 0.025 * n_lines, 1, 1])
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

    # Show at most ~20 tick labels so the x axis stays readable
    step = max(1, len(bin_edges) // 20)
    shown = list(range(0, len(bin_edges), step))
    ax.set_xticks([x[i] for i in shown])
    ax.set_xticklabels([str(bin_edges[i]) for i in shown], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(f"Detected NCs per Muon in [1µs, 200µs] (M≥{M})", fontsize=9)
    ax.set_ylabel("Number of Muons")
    ax.set_title(
        f"Detected NC Count per {label} Muon\n"
        f"(time window [1µs, 200µs], M≥{M}; only muons with ≥1 NC)"
    )
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    stats = "    ".join(
        f"{r.label}: {sum(1 for v in d if v >= W)}/{max(len(d), 1)} "
        f"≥W ({100*sum(1 for v in d if v >= W)/max(len(d), 1):.1f}%)"
        for r, d in zip(results, all_data)
    )
    ax.text(0.5, -0.20, stats, transform=ax.transAxes,
            ha="center", fontsize=8, fontstyle="italic")


def plot_w_histogram(
    results: list[SetupResult],
    M_default: int,
    W_default: int,
    output_dir: str,
) -> None:
    """W-histogram at M_default: Ge77 and non-Ge77 muons side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(26, 8))
    _draw_w_panel(axes[0], results, M_default, W_default, ge77=True)
    _draw_w_panel(axes[1], results, M_default, W_default, ge77=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fname = "07_w_histogram.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plots 10/11 — Recall / Precision line across all (M, W) combinations
# ──────────────────────────────────────────────────────────────────────
def plot_mw_sweep(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
) -> None:
    """One line plot per metric (Recall, Precision) across all (M, W) pairs.

    X-axis order: M1W1, M1W2, …, M1W_max, M2W1, … (M outer, W inner).
    Each setup is a line in its consistent palette colour.
    Vertical separators mark M-group boundaries; the M value is annotated
    above each group.  Y-axis shows percentage values (0 %–100 %).
    """
    colors  = _colors(len(results))
    mw_pairs = [(M, W) for M in M_values for W in W_values]
    x_labels = [f"M{M}W{W}" for M, W in mw_pairs]
    x        = np.arange(len(mw_pairs))
    n_w      = len(W_values)

    for metric, fname_part, plot_num in [
        ("Recall",    "recall",    "10"),
        ("Precision", "precision", "11"),
    ]:
        # Wide enough to give each tick ~0.18 inches; minimum 30 inches
        fig_w = max(30, len(mw_pairs) * 0.18)
        fig, ax = plt.subplots(figsize=(fig_w, 9))

        for r, c in zip(results, colors):
            vals = [
                compute_metrics(
                    r.muon["confusion"][(M, W)]["TP"],
                    r.muon["confusion"][(M, W)]["FP"],
                    r.muon["confusion"][(M, W)]["TN"],
                    r.muon["confusion"][(M, W)]["FN"],
                )[metric]
                for M, W in mw_pairs
            ]
            ax.plot(x, vals, color=c, label=r.label,
                    linewidth=1.2, marker=".", markersize=4)

        # Vertical separators + M-group labels at the top of the axes
        for gi, M in enumerate(M_values):
            group_start = gi * n_w
            group_mid   = group_start + (n_w - 1) / 2
            if gi > 0:
                ax.axvline(group_start - 0.5, color="gray",
                           linewidth=0.6, linestyle="--", alpha=0.5)
            ax.text(
                group_mid, 1.02, f"M={M}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color="dimgray",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_ylim(-0.02, 1.08)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
        )
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.set_title(
            f"Ge-77 Muon Classification — {metric} across all (M, W) combinations",
            fontsize=13, pad=20,
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlim(-0.5, len(mw_pairs) - 0.5)

        fig.tight_layout()
        fname = f"{plot_num}_mw_sweep_{fname_part}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 12 — W2 vs NC coverage scatter (optional)
# ──────────────────────────────────────────────────────────────────────
def plot_w2_scatter(
    results: list[SetupResult],
    M_values: list[int],
    M_default: int,
    W_default: int,
    W_values: list[int],
    output_dir: str,
    M_fixed: int = 1,
    W_fixed: int = 1,
) -> None:
    """Three-panel W2 scatter figure at fixed (M_fixed, W_fixed) — one file.

    Panel 1 — W2 vs NC coverage fraction for multiple M thresholds
               (colored by M; per-setup point annotations).
    Panel 2 — W2 vs Recall at (M_fixed, W_fixed), per-setup colours + OLS.
    Panel 3 — W2 vs Precision at (M_fixed, W_fixed), per-setup colours + OLS.
    """
    w2_results = [r for r in results if r.w2 is not None]
    if len(w2_results) < 2:
        print("  [SKIP] 09_w2_scatter*.png: fewer than 2 setups have W2.")
        return

    w2_vals = np.array([r.w2 for r in w2_results])
    setup_colors = [_setup_color(results, r) for r in w2_results]

    # M values shown in panel 1 (NC coverage across multiple thresholds)
    panel1_ms = sorted({M for M in [1, 2, 4, 5, 10] if M in M_values})
    m_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(panel1_ms)))

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(
        f"W2 Homogeneity vs Performance  (M={M_fixed}, W={W_fixed})",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: W2 vs NC fraction for multiple M thresholds ──────────
    ax = axes[0]
    for sM, cm in zip(panel1_ms, m_colors):
        fracs = np.array([
            r.nc["nc_detected"][sM] / max(r.nc["nc_total"], 1)
            for r in w2_results
        ])
        ax.scatter(w2_vals, fracs, color=cm, s=55, zorder=3, label=f"M={sM}")
        for r, w2v, frac in zip(w2_results, w2_vals, fracs):
            ax.annotate(r.label, xy=(w2v, frac), xytext=(4, 2),
                        textcoords="offset points", fontsize=6, color=cm)
    ax.set_xlabel("Global W2 (mm) — lower = more uniform", fontsize=10)
    ax.set_ylabel("NC detection fraction", fontsize=10)
    ax.set_title("W2 vs NC Coverage", fontsize=11)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.legend(title="M threshold", fontsize=8, title_fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panels 2 & 3: W2 vs Recall / Precision with OLS line ──────────
    for ax, metric_key, ylabel, title in [
        (axes[1], "Recall",    "Recall",
         f"W2 vs Recall  (M={M_fixed}, W={W_fixed})"),
        (axes[2], "Precision", "Precision",
         f"W2 vs Precision  (M={M_fixed}, W={W_fixed})"),
    ]:
        y_vals = np.array([
            compute_metrics(
                r.muon["confusion"][(M_fixed, W_fixed)]["TP"],
                r.muon["confusion"][(M_fixed, W_fixed)]["FP"],
                r.muon["confusion"][(M_fixed, W_fixed)]["TN"],
                r.muon["confusion"][(M_fixed, W_fixed)]["FN"],
            )[metric_key]
            for r in w2_results
        ])

        for r, w2v, yv, c in zip(w2_results, w2_vals, y_vals, setup_colors):
            ax.scatter([w2v], [yv], color=c, s=70, zorder=3)
            ax.annotate(r.label, xy=(w2v, yv), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color=c)

        # OLS regression line
        if len(w2_vals) >= 3 and np.std(w2_vals) > 0 and np.std(y_vals) > 0:
            slope, intercept, *_ = scipy_stats.linregress(w2_vals, y_vals)
            x_fit = np.linspace(w2_vals.min(), w2_vals.max(), 200)
            ax.plot(x_fit, slope * x_fit + intercept,
                    color="black", linewidth=1.2, linestyle="--", zorder=2,
                    label="OLS fit")

        ax.set_xlabel("Global W2 (mm)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = f"09_w2_scatter_M{M_fixed:02d}_W{W_fixed:02d}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# W2 correlation analysis (Plots A–E)  — mirrors evaluate_coverages.py
# ──────────────────────────────────────────────────────────────────────

# ── data-extraction helpers (SetupResult equivalents of _nc_frac etc.) ─

def _cc_nc_frac(r: SetupResult, M: int) -> float:
    return r.nc["nc_detected"][M] / max(r.nc["nc_total"], 1)


def _cc_recall(r: SetupResult, M: int, W: int) -> float:
    cm = r.muon["confusion"].get((M, W))
    if cm is None:
        return 0.0
    return compute_metrics(cm["TP"], cm["FP"], cm["TN"], cm["FN"])["Recall"]


def _cc_precision(r: SetupResult, M: int, W: int) -> float:
    cm = r.muon["confusion"].get((M, W))
    if cm is None:
        return 0.0
    return compute_metrics(cm["TP"], cm["FP"], cm["TN"], cm["FN"])["Precision"]


def _sorted_by_w2_cc(results: list[SetupResult]) -> list[SetupResult]:
    """Return results sorted by W2 descending (None W2 last)."""
    return sorted(results, key=lambda r: (r.w2 is None, -(r.w2 or 0.0)))


# ── Plot A — W2 × NC coverage correlation scatter ──────────────────────

def plot_w2_nc_correlation(
    results: list[SetupResult],
    M_values: list[int],
    output_dir: str,
) -> None:
    """
    Plot A — W2 vs NC detection fraction for every M threshold.

    4 rows × 3 cols grid (one panel per M, two extra panels hidden).
    Each panel: scatter + OLS + 95 % CI (top), residuals (bottom).
    Pearson r and Spearman ρ with p-values annotated per panel.
    """
    w2_res = [r for r in results if r.w2 is not None]
    if len(w2_res) < 2:
        print("  [SKIP] w2_nc_correlation: fewer than 2 setups have W2.")
        return

    ordered    = _sorted_by_w2_cc(w2_res)
    colors_all = _colors(len(ordered))
    color_map  = {r.label: colors_all[i] for i, r in enumerate(ordered)}
    w2_arr     = np.array([r.w2 for r in ordered])
    labels     = [r.label for r in ordered]
    color_pts  = [color_map[r.label] for r in ordered]

    ncols = 3
    nrows = (len(M_values) + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 5, nrows * 6))
    fig.suptitle(
        "W2 Homogeneity vs NC Detection Fraction\n"
        "(OLS fit · 95 % CI · Pearson r · Spearman ρ)",
        fontsize=13,
    )

    for pi, M in enumerate(M_values):
        col = pi % ncols
        row = pi // ncols
        ax_scatter = fig.add_subplot(nrows * 2, ncols,
                                     row * 2 * ncols + col + 1)
        ax_resid   = fig.add_subplot(nrows * 2, ncols,
                                     (row * 2 + 1) * ncols + col + 1,
                                     sharex=ax_scatter)

        y_arr = np.array([_cc_nc_frac(r, M) for r in ordered])
        _regression_overlay(ax_scatter, ax_resid, w2_arr, y_arr,
                            color_pts, labels,
                            y_label=f"NC fraction (M≥{M})")
        ax_scatter.set_title(f"M ≥ {M}", fontsize=9)
        ax_scatter.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
        ax_resid.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
        plt.setp(ax_scatter.get_xticklabels(), visible=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = "13_w2_nc_correlation.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Plot B — W2 × Muon metrics correlation scatter ─────────────────────

def plot_w2_muon_correlation(
    results: list[SetupResult],
    M_values: list[int],
    W_default: int,
    output_dir: str,
) -> None:
    """
    Plot B — W2 vs Ge-77 Recall and Precision at W=W_default for every M.

    Left 3 cols = Recall, right 3 cols = Precision.
    Each cell: scatter + regression (top) + residuals (bottom).
    """
    w2_res = [r for r in results if r.w2 is not None]
    if len(w2_res) < 2:
        print("  [SKIP] w2_muon_correlation: fewer than 2 setups have W2.")
        return

    ordered    = _sorted_by_w2_cc(w2_res)
    colors_all = _colors(len(ordered))
    color_map  = {r.label: colors_all[i] for i, r in enumerate(ordered)}
    w2_arr     = np.array([r.w2 for r in ordered])
    labels     = [r.label for r in ordered]
    color_pts  = [color_map[r.label] for r in ordered]

    ncols_half = 3
    nrows_M    = (len(M_values) + ncols_half - 1) // ncols_half
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

        for metric_fn, metric_name, col_offset in [
            (_cc_recall,    "Recall",    0),
            (_cc_precision, "Precision", ncols_half),
        ]:
            col = logical_col + col_offset
            ax_scatter = fig.add_subplot(
                total_rows, total_cols,
                logical_row * 2 * total_cols + col + 1,
            )
            ax_resid = fig.add_subplot(
                total_rows, total_cols,
                (logical_row * 2 + 1) * total_cols + col + 1,
                sharex=ax_scatter,
            )

            y_arr = np.array([metric_fn(r, M, W_default) for r in ordered])
            _regression_overlay(ax_scatter, ax_resid, w2_arr, y_arr,
                                color_pts, labels,
                                y_label=f"{metric_name} (M≥{M}, W≥{W_default})")
            ax_scatter.set_title(f"{metric_name}  M≥{M}", fontsize=8)
            ax_scatter.set_ylim(-0.05, 1.05)
            ax_scatter.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
            ax_resid.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
            plt.setp(ax_scatter.get_xticklabels(), visible=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = "14_w2_muon_correlation.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Plot C — Correlation matrix (Pearson + Spearman) ───────────────────

def plot_w2_correlation_matrix(
    results: list[SetupResult],
    M_values: list[int],
    M_default: int,
    W_default: int,
    output_dir: str,
) -> None:
    """
    Plot C — Pearson and Spearman correlation matrices.

    Variables: W2, NC_frac_M1..M_max, Recall and Precision at
    (M_default, W_default).  Two heatmaps side-by-side.
    Cell annotations include significance stars (* p<0.05, ** p<0.01).
    """
    w2_res = [r for r in results if r.w2 is not None]
    if len(w2_res) < 3:
        print("  [SKIP] w2_correlation_matrix: fewer than 3 setups have W2.")
        return

    var_names = ["W2"] + [f"NC_M{M}" for M in M_values] + [
        f"Recall\nM{M_default}W{W_default}",
        f"Precision\nM{M_default}W{W_default}",
    ]
    data_rows = []
    for r in w2_res:
        row = [r.w2]
        row += [_cc_nc_frac(r, M) for M in M_values]
        row += [_cc_recall(r, M_default, W_default),
                _cc_precision(r, M_default, W_default)]
        data_rows.append(row)

    X  = np.array(data_rows)
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
                r_val, pr  = scipy_stats.pearsonr(X[:, i],  X[:, j])
                rho,   prs = scipy_stats.spearmanr(X[:, i], X[:, j])
            except ValueError:
                continue
            pearson_mat[i, j]  = r_val
            spearman_mat[i, j] = rho
            pval_p_mat[i, j]   = pr
            pval_s_mat[i, j]   = prs

    fig, (ax_p, ax_s) = plt.subplots(
        1, 2,
        figsize=(max(14, nv * 1.1) * 2, max(10, nv * 1.0)),
    )
    fig.suptitle(
        f"Correlation Matrices  (n = {len(w2_res)} setups)\n"
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
                tc  = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:+.2f}{sig}",
                        ha="center", va="center",
                        fontsize=7, color=tc, fontweight="bold")
        ax.set_xticks(range(nv))
        ax.set_yticks(range(nv))
        ax.set_xticklabels(var_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(var_names, fontsize=8)
        ax.set_title(title, fontsize=11)

    fig.text(0.5, 0.01, "* p<0.05   ** p<0.01",
             ha="center", fontsize=8, fontstyle="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = "15_w2_correlation_matrix.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Plot D — Coverage profile colored by W2 ────────────────────────────

def plot_w2_coverage_profile(
    results: list[SetupResult],
    M_values: list[int],
    output_dir: str,
) -> None:
    """
    Plot D — NC detection fraction vs M, one line per setup colored by W2.

    Blue = low W2 (uniform); red = high W2 (clustered).
    Setups without W2 are drawn in gray dashed.
    """
    w2_res   = [r for r in results if r.w2 is not None]
    gray_res = [r for r in results if r.w2 is None]

    if not w2_res:
        print("  [SKIP] w2_coverage_profile: no setups have W2.")
        return

    w2_vals = np.array([r.w2 for r in w2_res])
    w2_min, w2_max = w2_vals.min(), w2_vals.max()
    cmap = plt.cm.coolwarm_r
    norm = plt.Normalize(vmin=w2_min,
                         vmax=w2_max if w2_max > w2_min else w2_min + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in gray_res:
        ys = [_cc_nc_frac(r, M) for M in M_values]
        ax.plot(M_values, ys, color="gray", linewidth=1.0,
                linestyle="--", alpha=0.6, label=r.label)

    for r in w2_res:
        color = cmap(norm(r.w2))
        ys = [_cc_nc_frac(r, M) for M in M_values]
        ax.plot(M_values, ys, color=color, linewidth=1.8,
                marker="o", markersize=4, label=r.label)
        ax.annotate(r.label, xy=(M_values[-1], ys[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=6, color=color, va="center")

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
        mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fname = "16_w2_coverage_profile.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Plot E — Spearman ρ vs M ────────────────────────────────────────────

def plot_w2_spearman_vs_m(
    results: list[SetupResult],
    M_values: list[int],
    W_default: int,
    output_dir: str,
) -> None:
    """
    Plot E — Spearman ρ(W2, metric) as a function of M threshold.

    Three lines: NC fraction, Recall (W_default), Precision (W_default).
    Filled markers = p<0.05; hollow = not significant.
    Gray band marks weak |ρ|<0.3 zone.
    """
    w2_res = [r for r in results if r.w2 is not None]
    if len(w2_res) < 3:
        print("  [SKIP] w2_spearman_vs_m: fewer than 3 setups have W2.")
        return

    w2_arr = np.array([r.w2 for r in w2_res])

    rho_nc, rho_rec, rho_prec = [], [], []
    p_nc,   p_rec,   p_prec   = [], [], []

    for M in M_values:
        nc_arr   = np.array([_cc_nc_frac(r, M) for r in w2_res])
        rec_arr  = np.array([_cc_recall(r, M, W_default) for r in w2_res])
        prec_arr = np.array([_cc_precision(r, M, W_default) for r in w2_res])

        r1, p1 = scipy_stats.spearmanr(w2_arr, nc_arr)
        r2, p2 = scipy_stats.spearmanr(w2_arr, rec_arr)
        r3, p3 = scipy_stats.spearmanr(w2_arr, prec_arr)

        rho_nc.append(r1);   p_nc.append(p1)
        rho_rec.append(r2);  p_rec.append(p2)
        rho_prec.append(r3); p_prec.append(p3)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.array(M_values)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhspan(-0.3, 0.3, color="gray", alpha=0.08, label="weak |ρ|<0.3")

    for rhos, ps, label, color, marker in [
        (rho_nc,   p_nc,   "NC fraction",              "#1f77b4", "o"),
        (rho_rec,  p_rec,  f"Recall (W={W_default})",  "#d62728", "s"),
        (rho_prec, p_prec, f"Precision (W={W_default})","#2ca02c", "^"),
    ]:
        rhos = np.array(rhos)
        ps   = np.array(ps)
        sig  = ps < 0.05
        ax.plot(x, rhos, color=color, linewidth=1.5, label=label)
        if sig.any():
            ax.scatter(x[sig],  rhos[sig],  color=color, s=60,
                       marker=marker, zorder=4, label=f"{label} (p<0.05)")
        if (~sig).any():
            ax.scatter(x[~sig], rhos[~sig], facecolors="none",
                       edgecolors=color, s=60, marker=marker,
                       linewidth=1.2, zorder=4)

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
    fname = "17_w2_spearman_vs_m.png"
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
            f.write(
                f"  Ge77-producing NCs (flag_ge77=1): {nc['ge77_nc_total']:,}\n"
                f"  NOTE: These are NCs inside Ge detectors, NOT all NCs of Ge77 muons.\n"
                f"        A low detection fraction here does NOT imply low muon recall.\n"
            )
            f.write("\n  NC detected by M threshold:\n")
            for M in M_values:
                det  = nc["nc_detected"][M]
                gdet = nc["ge77_nc_detected"][M]
                frac = 100 * det / max(nc["nc_total"], 1)
                gfrac = 100 * gdet / max(nc["ge77_nc_total"], 1)
                f.write(
                    f"    M={M:2d}:  all NCs: {det:,}  ({frac:.1f}%)  "
                    f"| Ge77-producing NCs: {gdet:,} / {nc['ge77_nc_total']:,}"
                    f"  ({gfrac:.1f}%)\n"
                )
            f.write(
                f"\n  Muons: {ms['total']:,}  "
                f"Ge77: {ms['n_ge77']:,}  "
                f"non-Ge77: {ms['n_non_ge77']:,}\n"
            )
            gd = r.muon["ge77_muon_detectability"]
            if gd["any_photon"] >= 0:
                f.write(
                    f"\n  Ge77-muon detection funnel (consistent with heatmap Recall):\n"
                    f"    Total Ge77 muons       : {ms['n_ge77']:,}\n"
                    f"    ≥1 NC any photon       : {gd['any_photon']:,}  "
                    f"({100*gd['any_photon']/max(ms['n_ge77'],1):.1f}%)\n"
                    f"    ≥1 NC within 200 ns    : {gd['within_200ns']:,}  "
                    f"({100*gd['within_200ns']/max(ms['n_ge77'],1):.1f}%)\n"
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
            f"{'Recall':>7} {'Prec':>7}\n"
        )
        f.write(hdr)
        f.write("-" * (len(hdr) - 1) + "\n")
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
                        f"{m['Recall']:>7.4f} {m['Precision']:>7.4f}\n"
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
    parser.add_argument(
        "--omit-runs", nargs="+", default=[], metavar="RUN",
        help=(
            "Run directory names to skip across all setups and the NC dir "
            "(e.g. --omit-runs run_002 run_003)."
        ),
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

    M_values   = list(range(1, args.M_max + 1))
    W_values   = list(range(1, args.W_max + 1))
    M_default  = min(max(args.M_default, 1), args.M_max)
    W_default  = min(max(args.W_default, 1), args.W_max)
    omit_runs  = set(args.omit_runs) if args.omit_runs else None

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("PMT COVERAGE COMPARISON — N CONFIGURATIONS")
    print("=" * 70)
    print(f"  Muon dir : {args.muon_dir}")
    for i, (d, lbl) in enumerate(zip(args.sim_dirs, args.labels)):
        print(f"  Setup {i+1}  : {lbl}  ({d})")
    print(f"  m={args.m}, M=1..{args.M_max}, W=1..{args.W_max}")
    print(f"  M_default={M_default}, W_default={W_default}")
    if omit_runs:
        print(f"  Omitting : {sorted(omit_runs)}")
    print(f"  Output   : {args.output_dir}")
    print()

    # ── 0. Integrity pre-flight check ────────────────────────────────
    check_all_files_integrity(args.muon_dir, args.sim_dirs, args.labels, omit_runs=omit_runs)
    print()

    # ── 1. Load shared NC truth ───────────────────────────────────────
    print("Loading NC truth ...")
    nc_truth = build_nc_truth(args.muon_dir, verbose=True, omit_runs=omit_runs)
    print()

    # ── 2. Vertex count validation ────────────────────────────────────
    validate_vertex_counts(args.sim_dirs, args.labels, omit_runs=omit_runs)
    print()

    # ── 3. Process each setup ─────────────────────────────────────────
    results: list[SetupResult] = []

    for i, (sim_dir, label) in enumerate(zip(args.sim_dirs, args.labels)):
        print(f"[{i+1}/{len(args.sim_dirs)}] Processing: {label}")

        B, pmt_uids, detect_info = build_pmt_matrix(
            sim_dir, nc_truth, m_threshold=args.m, verbose=True, omit_runs=omit_runs
        )

        print("  Evaluating NC coverage ...")
        nc_res = evaluate_nc(B, nc_truth, M_values, detect_info=detect_info)

        print("  Evaluating muon classification ...")
        muon_res = evaluate_muon(B, nc_truth, M_values, W_values, detect_info=detect_info)

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
        gd = r.muon["ge77_muon_detectability"]
        if gd["any_photon"] >= 0:
            print(
                f"  Ge77 muon funnel: "
                f"any photon={gd['any_photon']:,}  "
                f"within 200ns={gd['within_200ns']:,}  "
                f"detected (TP)={conf['TP']:,}"
            )
        print(f"  At (M={M_default}, W={W_default}): TP={conf['TP']}  FN={conf['FN']}  TN={conf['TN']}  FP={conf['FP']}")
        print(f"  Recall={m['Recall']:.3f}  Precision={m['Precision']:.3f}")
        if r.w2 is not None:
            print(f"  W2: {r.w2:.2f} mm")

    print()

    # ── 6. Generate plots ─────────────────────────────────────────────
    print("Generating plots ...")
    plot_nc_coverage_line(results, M_values, args.output_dir)
    plot_nc_multiplicity_histogram(results, M_default, args.output_dir)
    plot_nc_detectability_overview(results, M_default, args.output_dir)
    plot_ge77_muon_overview(results, M_default, W_default, args.output_dir)
    plot_muon_heatmaps(results, M_values, W_values, args.output_dir)
    plot_confusion_bar(results, M_default, W_default, args.output_dir)
    plot_w_histogram(results, M_default, W_default, args.output_dir)
    # M×W sweep (omitted — heatmaps provide a clearer 2D view):
    # plot_mw_sweep(results, M_values, W_values, args.output_dir)
    plot_w2_scatter(results, M_values, M_default, W_default, W_values, args.output_dir)

    # W2 correlation analysis (Plots A–E)
    plot_w2_nc_correlation(results, M_values, args.output_dir)
    plot_w2_muon_correlation(results, M_values, W_default, args.output_dir)
    plot_w2_correlation_matrix(results, M_values, M_default, W_default, args.output_dir)
    plot_w2_coverage_profile(results, M_values, args.output_dir)
    plot_w2_spearman_vs_m(results, M_values, W_default, args.output_dir)

    # ── 7. Write text files ───────────────────────────────────────────
    print("Writing text summaries ...")
    write_nc_summary(results, M_values, args.output_dir)
    write_confusion_matrices(results, M_values, W_values, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
