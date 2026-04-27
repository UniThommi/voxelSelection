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
from pmtopt.geometry import (
    calc_fom_confusion,
    calc_ge_survival_confusion,
    calc_deadtime_confusion,
    calc_veto_fraction,
    figure_of_merit,
    MUSUN_RATE,
    MUONS_PER_RUN_DIR,
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

# Qualitative palette: 20 clearly distinguishable colours ordered so that
# adjacent entries have maximum hue contrast.  Covers up to 20 setups
# without repeating; cycles only beyond that.  Used for ALL non-heatmap
# plots so every setup is always represented by the same colour.
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


def _colors(n: int) -> list[str]:
    """Return a list of n setup colours, cycling _SETUP_PALETTE if n > 20."""
    return [_SETUP_PALETTE[i % len(_SETUP_PALETTE)] for i in range(n)]


def _setup_color(results: list, r) -> str:
    """Return the palette colour for SetupResult r relative to the full list."""
    return _SETUP_PALETTE[results.index(r) % len(_SETUP_PALETTE)]


def _w2_sorted(results: list) -> list:
    """Return results sorted by ascending W2 (None values last)."""
    return sorted(results, key=lambda r: (r.w2 is None, r.w2 or 0.0))


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
    color_map: dict[str, str] | None = None,
) -> None:
    """Line plot: NC detection fraction vs M for all configs (linear + log)."""
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = [color_map.get(r.label, "gray") for r in results]

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
                label="_nolegend_",
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
    color_map: dict[str, str] | None = None,
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
    ref_result = results[0]
    if color_map is None:
        _pal = _colors(n)
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    results = _w2_sorted(results)
    colors  = [color_map.get(r.label, "gray") for r in results]

    only_outside_key = (
        nc_key + "only_outside_200ns" if nc_key == "nc_"
        else nc_key + "only_outside"
    )
    detected_key = (nc_key + "detected", M_default)
    has_detect = all(r.nc.get(nc_key + "any_photon", -1) >= 0 for r in results)

    if has_detect:
        abs_cat_labels = [
            "≥1 photon\n(any time)",
            "≥1 photon\n(within 200 ns)",
            "Photons only\noutside 200 ns",
            f"Detected\n(M≥{M_default})",
        ]
        abs_val_keys = [
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
        abs_cat_labels = [f"Detected\n(M≥{M_default})"]
        abs_val_keys   = [detected_key]
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
    # Indicate what 100% corresponds to (the omitted Total bar)
    _nc_total_note = results[0].nc[total_key] if results else 0
    if _nc_total_note > 0:
        ax_abs.text(
            0.01, 0.99, f"100 % = {_nc_total_note:,} NCs",
            transform=ax_abs.transAxes, fontsize=9, va="top", ha="left",
            color="dimgray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.85),
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
    ref_total = ref_result.nc[total_key]
    ref_vals  = [_get(ref_result, vk) for vk in delta_val_keys]
    non_ref   = [(r, color_map.get(r.label, "gray")) for r in results if r is not ref_result]

    for j, (r, c) in enumerate(non_ref):
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
    ax_delta.set_xlabel(f"Δ count vs reference ({ref_result.label})", fontsize=9)
    ax_delta.set_title(
        f"Δ from reference: {ref_result.label}", fontsize=10
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
    color_map: dict[str, str] | None = None,
) -> None:
    """Create and save a two-panel detectability figure."""
    fig, (ax_abs, ax_delta) = plt.subplots(
        1, 2,
        figsize=(22, 9),
        gridspec_kw={"width_ratios": [3, 2]},
    )
    _draw_detectability_panels(
        ax_abs, ax_delta, results, nc_key, M_default, title, total_key,
        ylabel=ylabel, color_map=color_map,
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
    color_map: dict[str, str] | None = None,
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
        color_map=color_map,
    )


# ──────────────────────────────────────────────────────────────────────
# Plot 04 — Ge77-muon detection funnel (muon-level, consistent with heatmap)
# ──────────────────────────────────────────────────────────────────────
def plot_ge77_muon_overview(
    results: list[SetupResult],
    M_default: int,
    W_default: int,
    output_dir: str,
    color_map: dict[str, str] | None = None,
    total_primary_muons: int = 0,
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
    ref_result = results[0]
    if color_map is None:
        _pal = _colors(n)
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    results = _w2_sorted(results)
    colors  = [color_map.get(r.label, "gray") for r in results]
    has_photon_info = all(
        r.muon["ge77_muon_detectability"]["any_photon"] >= 0 for r in results
    )

    if has_photon_info:
        cat_labels = [
            "≥1 NC:\nany photon",
            "≥1 NC:\nphoton ≤200 ns\nafter NC",
            f"Detected\n(M≥{M_default}, W≥{W_default})\n= Recall TP",
        ]
        delta_cat_labels = cat_labels  # all included (Total removed)

        def _vals(r: SetupResult) -> list[int]:
            gd = r.muon["ge77_muon_detectability"]
            tp = r.muon["confusion"][(M_default, W_default)]["TP"]
            return [gd["any_photon"], gd["within_200ns"], tp]
    else:
        cat_labels = [
            f"Detected\n(M≥{M_default}, W≥{W_default})\n= Recall TP",
        ]
        delta_cat_labels = cat_labels

        def _vals(r: SetupResult) -> list[int]:
            tp = r.muon["confusion"][(M_default, W_default)]["TP"]
            return [tp]

    x = np.arange(len(cat_labels))
    width = min(0.8 / n, 0.30)

    fig, (ax_abs, ax_delta) = plt.subplots(
        1, 2, figsize=(22, 9), gridspec_kw={"width_ratios": [3, 2]},
    )

    # ── Left panel: absolute grouped bars ────────────────────────────
    _ge77_total = ref_result.muon["muon_stats"]["n_ge77"] if results else 0
    for i, (r, c) in enumerate(zip(results, colors)):
        vals   = _vals(r)
        offset = (i - (n - 1) / 2) * width
        bars   = ax_abs.bar(x + offset, vals, width, label=r.label, color=c)
        for bar, val in zip(bars, vals):
            _annotate_bar(ax_abs, bar, val, _ge77_total, fontsize=6, rotation=90)

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
    # Indicate what 100% corresponds to
    _all_muons = total_primary_muons if total_primary_muons > 0 else (results[0].muon["muon_stats"]["total"] if results else 0)
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
        ref_vals   = _vals(ref_result)
        ref_total  = ref_result.muon["muon_stats"]["n_ge77"]
        non_ref    = [(r, color_map.get(r.label, "gray")) for r in results if r is not ref_result]

        for j, (r, c) in enumerate(non_ref):
            vals   = _vals(r)
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
        ax_delta.set_xlabel(f"Δ muon count vs reference ({ref_result.label})", fontsize=9)
        ax_delta.set_title(f"Δ from reference: {ref_result.label}", fontsize=10)
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
    sorted_results = _w2_sorted(results)
    n_setups = len(sorted_results)
    labels = [r.label for r in sorted_results]

    # Shared LogNorm across all M values and all metrics (0–1 range)
    shared_norm = mcolors.LogNorm(vmin=1e-4, vmax=1.0)

    for M in M_values:
        # Build grids: shape (len(W_values), n_setups) for each metric
        grids: dict[str, np.ndarray] = {}
        for metric in metrics_to_plot:
            grid = np.zeros((len(W_values), n_setups))
            for wi, W in enumerate(W_values):
                for si, r in enumerate(sorted_results):
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
    color_map: dict[str, str] | None = None,
) -> None:
    """Four sub-figures (2×2): TP, FN, TN, FP at (M_default, W_default).

    Each panel shows one confusion metric with linear scale and y-axis zoomed
    to the data range so differences across setups are clearly visible.
    """
    results = _w2_sorted(results)
    n = len(results)
    if color_map is None:
        _pal = _colors(n)
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    categories = [
        ("True Positive\n(Ge77 → classified)", "TP"),
        ("False Negative\n(Ge77 → missed)", "FN"),
        ("True Negative\n(non-Ge77 → correct)", "TN"),
        ("False Positive\n(non-Ge77 → misclass.)", "FP"),
    ]
    colors = [color_map.get(r.label, "gray") for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()

    for ax, (cat_label, key) in zip(axes_flat, categories):
        all_vals = []
        for i, (r, c) in enumerate(zip(results, colors)):
            conf = r.muon["confusion"][(M_default, W_default)]
            val = conf[key]
            all_vals.append(val)
            bar = ax.bar([i], [val], 0.6, label=r.label, color=c)
            _annotate_bar(ax, bar[0], val, fontsize=8)

        vmin = min(all_vals) if all_vals else 0
        vmax = max(all_vals) if all_vals else 1
        span = vmax - vmin
        if span > 0:
            margin = span * 0.4
            ax.set_ylim(max(0, vmin - margin), vmax + margin)
        else:
            ax.set_ylim(0, vmax * 1.2 if vmax > 0 else 1)

        ax.set_xticks(range(n))
        ax.set_xticklabels([r.label for r in results], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Number of Muons")
        ax.set_title(cat_label, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Ge-77 Muon Classification (W≥{W_default}, M≥{M_default})\n"
        "(linear scale · y-axis zoomed to data range per panel)",
        fontsize=13,
    )

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

    fig.tight_layout(rect=[0, 0.04 + 0.025 * n_lines, 1, 0.96])
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
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """One line plot per metric (Recall, Precision, and optionally FoM) across all (M, W) pairs.

    X-axis order: M1W1, M1W2, …, M1W_max, M2W1, … (M outer, W inner).
    Each setup is a line in its consistent palette colour.
    Vertical separators mark M-group boundaries; the M value is annotated
    above each group.  Y-axis shows percentage values (0 %–100 %) for
    Recall/Precision; raw FoM values for the FoM sweep.
    """
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    colors = [color_map.get(r.label, "gray") for r in results]
    mw_pairs = [(M, W) for M in M_values for W in W_values]
    x_labels = [f"M{M}W{W}" for M, W in mw_pairs]
    x        = np.arange(len(mw_pairs))
    n_w      = len(W_values)

    metrics_list = [
        ("Recall",    "recall",    "10"),
        ("Precision", "precision", "11"),
    ]
    if total_primaries > 0:
        metrics_list.append(("FoM", "fom", "12"))

    for metric, fname_part, plot_num in metrics_list:
        # Wide enough to give each tick ~0.18 inches; minimum 30 inches
        fig_w = max(30, len(mw_pairs) * 0.18)
        fig, ax = plt.subplots(figsize=(fig_w, 9))

        for r, c in zip(results, colors):
            vals = []
            for M, W in mw_pairs:
                cm = r.muon["confusion"].get((M, W))
                if cm is None:
                    vals.append(np.nan)
                    continue
                if metric == "Recall":
                    vals.append(compute_metrics(
                        cm["TP"], cm["FP"], cm["TN"], cm["FN"],
                    )[metric])
                elif metric == "Precision":
                    vals.append(compute_metrics(
                        cm["TP"], cm["FP"], cm["TN"], cm["FN"],
                    )[metric])
                else:  # FoM
                    vals.append(calc_fom_confusion(
                        cm["TP"], cm["FP"], cm["FN"], total_primaries,
                        tp_ge77_nc_counts=cm.get("tp_ge77_nc_counts"),
                        fn_ge77_nc_counts=cm.get("fn_ge77_nc_counts"),
                    ))
            ax.plot(x, vals, color=c, label=r.label,
                    linewidth=1.2, marker=".", markersize=4)

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
                for r in results:
                    cm = r.muon["confusion"].get((M, W))
                    if cm is None:
                        continue
                    if metric == "Recall":
                        v = compute_metrics(cm["TP"], cm["FP"], cm["TN"], cm["FN"])[metric]
                    elif metric == "Precision":
                        v = compute_metrics(cm["TP"], cm["FP"], cm["TN"], cm["FN"])[metric]
                    else:  # FoM
                        v = calc_fom_confusion(
                            cm["TP"], cm["FP"], cm["FN"], total_primaries,
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
        if metric == "FoM":
            ax.yaxis.set_major_locator(mticker.AutoLocator())
        else:
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
# Plots 12 — Figure of Merit
# ──────────────────────────────────────────────────────────────────────

def _cc_fom_grid(
    r: "SetupResult",
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
) -> dict[tuple[int, int], float]:
    """Return {(M, W): fom} for all combinations. Missing entries → nan.

    ``total_primaries`` is the total number of simulated primary muons
    (all muons, not just those that produced neutron captures).
    """
    result: dict[tuple[int, int], float] = {}
    for M in M_values:
        for W in W_values:
            cm = r.muon["confusion"].get((M, W))
            if cm is None:
                result[(M, W)] = float("nan")
            else:
                result[(M, W)] = calc_fom_confusion(
                    cm["TP"], cm["FP"], cm["FN"], total_primaries,
                    tp_ge77_nc_counts=cm.get("tp_ge77_nc_counts"),
                    fn_ge77_nc_counts=cm.get("fn_ge77_nc_counts"),
                )
    return result


def plot_fom_summary(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """Horizontal bar chart: max FoM per setup, annotated with optimal (M, W).

    Setups are shown in their original order.
    ``total_primaries`` is the total number of simulated primary muons
    (all muons).  Pass 0 to fall back to the per-setup NC-muon count.
    """
    results = _w2_sorted(results)
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    colors = [color_map.get(r.label, "gray") for r in results]
    max_foms: list[float] = []
    best_mw:  list[str]   = []

    for r in results:
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        grid  = _cc_fom_grid(r, M_values, W_values, _tp)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best = max(valid, key=valid.__getitem__)
            max_foms.append(valid[best])
            best_mw.append(f"M{best[0]}W{best[1]}")
        else:
            max_foms.append(float("nan"))
            best_mw.append("N/A")

    y_max  = max((f for f in max_foms if np.isfinite(f)), default=1.0)
    y      = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(8, max(4, len(results) * 0.55)))
    bars = ax.barh(y, max_foms, color=colors, height=0.6)

    for bar, mw, fom in zip(bars, best_mw, max_foms):
        if np.isfinite(fom):
            ax.text(
                fom + 0.01 * y_max,
                bar.get_y() + bar.get_height() / 2,
                f"{fom:.4g}  [{mw}]", va="center", ha="left", fontsize=8,
            )

    ax.set_yticks(y)
    ax.set_yticklabels([r.label for r in results], fontsize=9)
    ax.set_xlim(right=y_max * 1.35)
    ax.set_xlabel("Figure of Merit  (max over all M, W)", fontsize=11)
    ax.set_title(
        "Ge-77 Muon Figure of Merit — Best (M, W) per Configuration",
        fontsize=12, pad=10,
    )
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fname = "12_fom_summary.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_fom_summary_min_m(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    min_M: int = 6,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """Horizontal bar chart: max FoM per setup restricted to M ≥ min_M.

    Identical to plot_fom_summary but only considers (M, W) pairs where
    M >= min_M.  Saved as 12b_fom_summary_M{min_M}plus.png.
    """
    eligible_M = [M for M in M_values if M >= min_M]
    if not eligible_M:
        print(f"  [SKIP] 12b_fom_summary_M{min_M}plus.png: no M values ≥ {min_M} in M_values.")
        return

    results = _w2_sorted(results)
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    colors = [color_map.get(r.label, "gray") for r in results]
    max_foms: list[float] = []
    best_mw:  list[str]   = []

    for r in results:
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        grid  = _cc_fom_grid(r, eligible_M, W_values, _tp)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best = max(valid, key=valid.__getitem__)
            max_foms.append(valid[best])
            best_mw.append(f"M{best[0]}W{best[1]}")
        else:
            max_foms.append(float("nan"))
            best_mw.append("N/A")

    y_max = max((f for f in max_foms if np.isfinite(f)), default=1.0)
    y     = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(8, max(4, len(results) * 0.55)))
    bars = ax.barh(y, max_foms, color=colors, height=0.6)

    for bar, mw, fom in zip(bars, best_mw, max_foms):
        if np.isfinite(fom):
            ax.text(
                fom + 0.01 * y_max,
                bar.get_y() + bar.get_height() / 2,
                f"{fom:.4g}  [{mw}]", va="center", ha="left", fontsize=8,
            )

    ax.set_yticks(y)
    ax.set_yticklabels([r.label for r in results], fontsize=9)
    ax.set_xlim(right=y_max * 1.35)
    ax.set_xlabel(f"Figure of Merit  (max over M≥{min_M}, all W)", fontsize=11)
    ax.set_title(
        f"Ge-77 Muon Figure of Merit — Best (M≥{min_M}, W) per Configuration",
        fontsize=12, pad=10,
    )
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fname = f"12b_fom_summary_M{min_M}plus.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_fom_per_setup(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
) -> None:
    """Per-setup FoM heatmap (M × W grid).

    Produces one PNG file per setup:
      12_fom_heatmap_{label}.png  — 2-D colour map of FoM over (M, W)

    The cell with the maximum FoM is highlighted.
    ``total_primaries`` is the total number of simulated primary muons.
    """
    mw_pairs = [(M, W) for M in M_values for W in W_values]
    x_labels = [f"M{M}W{W}" for M, W in mw_pairs]
    x_arr    = np.arange(len(mw_pairs))
    n_w      = len(W_values)

    for r in results:
        _tp  = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        grid = _cc_fom_grid(r, M_values, W_values, _tp)
        safe_name = r.label.replace(" ", "_").replace("/", "_")
        finite    = {k: v for k, v in grid.items() if np.isfinite(v)}

        # ── 1. Heatmap ────────────────────────────────────────────────
        heat = np.array(
            [[grid.get((M, W), float("nan")) for M in M_values] for W in W_values]
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
        ax_h.set_title(f"Figure of Merit — {r.label}", fontsize=12, pad=10)
        fig_h.tight_layout()
        fname_h = f"12_fom_heatmap_{safe_name}.png"
        fig_h.savefig(os.path.join(output_dir, fname_h), dpi=150)
        plt.close(fig_h)
        print(f"  Saved {fname_h}")



# ──────────────────────────────────────────────────────────────────────
# Plots 18 — W2 correlation: FoM and Recall
# ──────────────────────────────────────────────────────────────────────

def plot_w2_fom_best(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """Scatter: best FoM (max over all M, W) vs W2.

    One labelled point per setup with W2.  OLS + Spearman ρ overlay.
    """
    w2_res = sorted(
        [r for r in results if r.w2 is not None],
        key=lambda r: r.w2,
    )
    if len(w2_res) < 2:
        print("  [SKIP] 18_w2_fom_best.png: fewer than 2 setups have W2.")
        return

    if color_map is None:
        _pal = _colors(len(w2_res))
        color_map = {r.label: _pal[i] for i, r in enumerate(w2_res)}
    w2_vals  = np.array([r.w2 for r in w2_res])
    fom_best = np.array([
        max(
            (v for v in _cc_fom_grid(
                r, M_values, W_values,
                total_primaries if total_primaries > 0
                else r.muon["muon_stats"]["total"]
             ).values() if np.isfinite(v)),
            default=float("nan"),
        )
        for r in w2_res
    ])
    mask = np.isfinite(fom_best)

    fig, ax = plt.subplots(figsize=(8, 6))
    for r, w2v, fom in zip(w2_res, w2_vals, fom_best):
        if np.isfinite(fom):
            col = color_map.get(r.label, "gray")
            ax.scatter(w2v, fom, color=col, s=65, zorder=3)
            ax.annotate(r.label, xy=(w2v, fom), xytext=(4, 3),
                        textcoords="offset points", fontsize=7)

    if mask.sum() >= 2:
        from scipy import stats as _st
        slope, intercept, _, _, _ = _st.linregress(w2_vals[mask], fom_best[mask])
        rho, p_rho = _st.spearmanr(w2_vals[mask], fom_best[mask])
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
    fname = "18_w2_fom_best.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_w2_sorted_heatmaps(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
) -> None:
    """W2-sorted heatmaps of Recall and FoM vs (setup × W) for M ∈ {1,3,5,10}.

    Setups sorted left-to-right by ascending W2 (most uniform first).
    W2 value is appended to each x-axis label.
    Only setups with a W2 value are included.

    Produces 4 Recall figures and 4 FoM figures:
      18_w2_heatmap_recall_M{M:02d}.png
      18_w2_heatmap_fom_M{M:02d}.png
    """
    w2_res = sorted(
        [r for r in results if r.w2 is not None],
        key=lambda r: r.w2,
    )
    if len(w2_res) < 2:
        print("  [SKIP] 18_w2_sorted_heatmaps: fewer than 2 setups have W2.")
        return

    heatmap_ms = [m for m in [1, 3, 5, 10] if m in M_values]
    n_res    = len(w2_res)
    n_w      = len(W_values)
    x_labels = [f"{r.label}\nW2={r.w2:.1f}" for r in w2_res]

    for M in heatmap_ms:
        for metric, fname_pfx, cmap, fixed_vmin, fixed_vmax in [
            ("Recall", "18_w2_heatmap_recall", "YlOrRd",  0.0,  1.0),
            ("FoM",    "18_w2_heatmap_fom",    "viridis", None, None),
        ]:
            data = np.full((n_w, n_res), np.nan)
            for wi, W in enumerate(W_values):
                for ri, r in enumerate(w2_res):
                    cm = r.muon["confusion"].get((M, W))
                    if cm is None:
                        continue
                    if metric == "Recall":
                        data[wi, ri] = compute_metrics(
                            cm["TP"], cm["FP"], cm["TN"], cm["FN"]
                        )["Recall"]
                    else:
                        _tp = (total_primaries if total_primaries > 0
                               else r.muon["muon_stats"]["total"])
                        data[wi, ri] = calc_fom_confusion(
                            cm["TP"], cm["FP"], cm["FN"], _tp,
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

            fig_w = max(8, n_res * 2.4 + 2)
            fig_h = max(5, n_w * 0.45 + 2)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            im = ax.imshow(
                data, aspect="auto", cmap=cmap,
                vmin=_vmin, vmax=_vmax, origin="upper",
            )
            plt.colorbar(im, ax=ax, label=metric)

            for wi in range(n_w):
                for ri in range(n_res):
                    val = data[wi, ri]
                    if not np.isfinite(val):
                        continue
                    txt     = f"{val:.3f}" if metric == "Recall" else f"{val:.3g}"
                    txt_col = "white" if val > mid else "black"
                    ax.text(ri, wi, txt, ha="center", va="center",
                            fontsize=6, color=txt_col)

            ax.set_xticks(range(n_res))
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
            fname = f"{fname_pfx}_M{M:02d}.png"
            fig.savefig(os.path.join(output_dir, fname), dpi=150)
            plt.close(fig)
            print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 08 — W2 vs NC coverage scatter (standalone)
# ──────────────────────────────────────────────────────────────────────
def plot_w2_nc_scatter(
    results: list[SetupResult],
    M_values: list[int],
    output_dir: str,
    M_fixed: int = 1,
    W_fixed: int = 1,
    color_map: dict[str, str] | None = None,
) -> None:
    """W2 vs NC coverage fraction for multiple M thresholds — standalone plot.

    Saved as 08_w2_nc_scatter_M{M_fixed:02d}_W{W_fixed:02d}.png.
    """
    w2_results = [r for r in results if r.w2 is not None]
    if len(w2_results) < 2:
        print("  [SKIP] 08_w2_nc_scatter*.png: fewer than 2 setups have W2.")
        return

    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    w2_vals = np.array([r.w2 for r in w2_results])

    panel_ms = sorted({M for M in [1, 2, 4, 5, 10] if M in M_values})
    m_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(panel_ms)))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("W2 Homogeneity vs NC Coverage", fontsize=13, fontweight="bold")

    for sM, cm in zip(panel_ms, m_colors):
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
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.legend(title="M threshold", fontsize=8, title_fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = f"08_w2_nc_scatter_M{M_fixed:02d}_W{W_fixed:02d}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 09 — W2 vs Recall / Precision scatter
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
    color_map: dict[str, str] | None = None,
) -> None:
    """Two-panel W2 scatter figure at fixed (M_fixed, W_fixed) — one file.

    Panel 1 — W2 vs Recall at (M_fixed, W_fixed), per-setup colours + OLS.
    Panel 2 — W2 vs Precision at (M_fixed, W_fixed), per-setup colours + OLS.

    W2 vs NC coverage is saved separately as 08_w2_nc_scatter_*.png.
    """
    w2_results = [r for r in results if r.w2 is not None]
    if len(w2_results) < 2:
        print("  [SKIP] 09_w2_scatter*.png: fewer than 2 setups have W2.")
        return

    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    w2_vals = np.array([r.w2 for r in w2_results])
    setup_colors = [color_map.get(r.label, "gray") for r in w2_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f"W2 Homogeneity vs Performance  (M={M_fixed}, W={W_fixed})",
        fontsize=13, fontweight="bold",
    )

    # ── Panels 1 & 2: W2 vs Recall / Precision with OLS line ──────────
    for ax, metric_key, ylabel, title in [
        (axes[0], "Recall",    "Recall",
         f"W2 vs Recall  (M={M_fixed}, W={W_fixed})"),
        (axes[1], "Precision", "Precision",
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


def _cc_signal_survival(r: SetupResult, M: int, W: int) -> float:
    """Signal survival at (M, W): 1 − (TP+FP)/(TP+FP+TN+FN)."""
    cm = r.muon["confusion"].get((M, W))
    if cm is None:
        return 0.0
    return 1.0 - calc_veto_fraction(cm["TP"], cm["FP"], cm["TN"], cm["FN"])


def _sorted_by_w2_cc(results: list[SetupResult]) -> list[SetupResult]:
    """Return results sorted by W2 descending (None W2 last)."""
    return sorted(results, key=lambda r: (r.w2 is None, -(r.w2 or 0.0)))


# ── Plot A — W2 × NC coverage correlation scatter ──────────────────────

def plot_w2_nc_correlation(
    results: list[SetupResult],
    M_values: list[int],
    output_dir: str,
    color_map: dict[str, str] | None = None,
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

    ordered = _sorted_by_w2_cc(w2_res)
    if color_map is None:
        colors_all = _colors(len(ordered))
        color_map  = {r.label: colors_all[i] for i, r in enumerate(ordered)}
    w2_arr    = np.array([r.w2 for r in ordered])
    labels    = [r.label for r in ordered]
    color_pts = [color_map.get(r.label, "gray") for r in ordered]

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
    color_map: dict[str, str] | None = None,
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

    ordered = _sorted_by_w2_cc(w2_res)
    if color_map is None:
        colors_all = _colors(len(ordered))
        color_map  = {r.label: colors_all[i] for i, r in enumerate(ordered)}
    w2_arr    = np.array([r.w2 for r in ordered])
    labels    = [r.label for r in ordered]
    color_pts = [color_map.get(r.label, "gray") for r in ordered]

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
    total_primaries: int = 0,
    W_values: list[int] | None = None,
) -> None:
    """
    Plot E — Spearman ρ(W2, metric) as a function of M threshold.

    Lines: NC fraction, Recall (W_default), Precision (W_default),
    and optionally best FoM (max over W) when total_primaries > 0.
    Filled markers = p<0.05; hollow = not significant.
    Gray band marks weak |ρ|<0.3 zone.
    """
    w2_res = [r for r in results if r.w2 is not None]
    if len(w2_res) < 3:
        print("  [SKIP] w2_spearman_vs_m: fewer than 3 setups have W2.")
        return

    w2_arr = np.array([r.w2 for r in w2_res])

    rho_nc, rho_rec, rho_prec, rho_fom = [], [], [], []
    p_nc,   p_rec,   p_prec,   p_fom   = [], [], [], []

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

        if total_primaries > 0:
            _w_vals = W_values if W_values is not None else []
            fom_arr = []
            for r in w2_res:
                best = np.nan
                for W in _w_vals:
                    cm = r.muon["confusion"].get((M, W))
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
    ax.axhline(0, color="black", linewidth=0.8)
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
# Plot F — NC fraction vs Ge-77 Recall correlation
# ──────────────────────────────────────────────────────────────────────

def plot_nc_recall_correlation(
    results: list[SetupResult],
    M_values: list[int],
    output_dir: str,
    color_map: dict[str, str] | None = None,
    W_fixed: int = 1,
) -> None:
    """Plot F — NC detection fraction vs Ge-77 Recall at W=W_fixed for each M.

    Each point is one PMT configuration.
    Layout: one panel per M value, scatter+OLS+95%CI (top), residuals (bottom).
    Pearson r and Spearman ρ with p-values annotated per panel.
    Saved as 19_nc_recall_correlation.png.
    """
    if len(results) < 2:
        print("  [SKIP] 19_nc_recall_correlation.png: fewer than 2 setups.")
        return

    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    color_pts = [color_map.get(r.label, "gray") for r in results]
    labels    = [r.label for r in results]

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
        ax_scatter = fig.add_subplot(nrows * 2, ncols,
                                     row * 2 * ncols + col + 1)
        ax_resid   = fig.add_subplot(nrows * 2, ncols,
                                     (row * 2 + 1) * ncols + col + 1,
                                     sharex=ax_scatter)

        nc_arr  = np.array([_cc_nc_frac(r, M) for r in results])
        rec_arr = np.array([_cc_recall(r, M, W_fixed) for r in results])

        _regression_overlay(
            ax_scatter, ax_resid, nc_arr, rec_arr,
            color_pts, labels,
            y_label=f"Recall (M≥{M}, W≥{W_fixed})",
            x_label=f"NC fraction (M≥{M})",
        )
        ax_scatter.set_title(f"M ≥ {M}", fontsize=9)
        ax_scatter.set_ylim(-0.05, 1.05)
        ax_scatter.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
        ax_scatter.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax_resid.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
        ax_resid.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
        plt.setp(ax_scatter.get_xticklabels(), visible=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = "19_nc_recall_correlation.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 20 — Recall at W=1 across M thresholds
# ──────────────────────────────────────────────────────────────────────
def plot_recall_w1_vs_m(
    results: list[SetupResult],
    M_values: list[int],
    output_dir: str,
    color_map: dict[str, str] | None = None,
) -> None:
    """Two-panel: Ge-77 Recall at W=1 (left, zoomed y) and setup rank at each M (right).

    The rank panel immediately shows whether the best setup at M=1 stays on top
    as M increases — a stable rank-1 line means the ordering is preserved.
    """
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    colors = [color_map.get(r.label, "gray") for r in results]

    all_recalls = {r.label: [_cc_recall(r, M, 1) for M in M_values] for r in results}

    fig, (ax_recall, ax_rank) = plt.subplots(1, 2, figsize=(18, 6))

    # ── Left panel: recall, zoomed y-axis ────────────────────────────
    for r, c in zip(results, colors):
        ax_recall.plot(M_values, all_recalls[r.label], marker="o", color=c,
                       label=r.label, linewidth=1.5, markersize=5)

    all_vals = [v for recalls in all_recalls.values() for v in recalls if np.isfinite(v)]
    if all_vals:
        vmin, vmax = min(all_vals), max(all_vals)
        margin = max((vmax - vmin) * 0.3, 0.005)
        ax_recall.set_ylim(max(0.0, vmin - margin), min(1.0, vmax + margin))

    ax_recall.set_xlabel("M — minimum firing PMTs per NC", fontsize=11)
    ax_recall.set_ylabel("Ge-77 Recall  (W = 1)", fontsize=11)
    ax_recall.set_title(
        "Ge-77 Muon Recall at W=1 across M Thresholds",
        fontsize=12,
    )
    ax_recall.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax_recall.set_xticks(M_values)
    ax_recall.legend(fontsize=8, loc="upper right")
    ax_recall.grid(True, alpha=0.3)

    # ── Right panel: rank at each M (rank 1 = highest recall) ────────
    for r, c in zip(results, colors):
        ranks = []
        for mi, M in enumerate(M_values):
            vals_at_m = [(results[j].label, all_recalls[results[j].label][mi])
                         for j in range(len(results))]
            vals_sorted = sorted(vals_at_m, key=lambda x: -x[1])
            rank = next(i + 1 for i, (lbl, _) in enumerate(vals_sorted) if lbl == r.label)
            ranks.append(rank)
        ax_rank.plot(M_values, ranks, marker="o", color=c, label=r.label,
                     linewidth=1.5, markersize=5)

    ax_rank.set_xlabel("M — minimum firing PMTs per NC", fontsize=11)
    ax_rank.set_ylabel("Recall rank  (1 = highest)", fontsize=11)
    ax_rank.set_title(
        "Setup Rank by Recall at W=1 across M\n"
        "(rank 1 = best; flat line → ordering is preserved)",
        fontsize=12,
    )
    ax_rank.set_xticks(M_values)
    ax_rank.set_yticks(range(1, len(results) + 1))
    ax_rank.invert_yaxis()
    ax_rank.legend(fontsize=8)
    ax_rank.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = "20_recall_w1_vs_m.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 21 — Recall at each setup's FoM-optimal (M, W)
# ──────────────────────────────────────────────────────────────────────
def plot_recall_at_best_fom(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """Horizontal bar: Recall at each setup's FoM-optimal (M, W).

    Each bar is annotated with its optimal (M, W) pair and Recall value.
    """
    results = _w2_sorted(results)
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    colors = [color_map.get(r.label, "gray") for r in results]

    recalls: list[float] = []
    opt_mw_labels: list[str] = []
    for r in results:
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        grid = _cc_fom_grid(r, M_values, W_values, _tp)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best_mw = max(valid, key=valid.__getitem__)
            recalls.append(_cc_recall(r, best_mw[0], best_mw[1]))
            opt_mw_labels.append(f"M{best_mw[0]}W{best_mw[1]}")
        else:
            recalls.append(float("nan"))
            opt_mw_labels.append("N/A")

    y = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(9, max(4, len(results) * 0.55)))
    bars = ax.barh(y, recalls, color=colors, height=0.6)

    x_max = max((r for r in recalls if np.isfinite(r)), default=1.0)
    for bar, mw, rec in zip(bars, opt_mw_labels, recalls):
        if np.isfinite(rec):
            ax.text(
                rec + 0.01 * x_max,
                bar.get_y() + bar.get_height() / 2,
                f"{rec*100:.1f}%  [{mw}]",
                va="center", ha="left", fontsize=8,
            )

    ax.set_yticks(y)
    ax.set_yticklabels([r.label for r in results], fontsize=9)
    ax.set_xlim(right=x_max * 1.4)
    ax.set_xlabel("Ge-77 Recall at FoM-optimal (M, W)", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_title(
        "Ge-77 Muon Recall at Each Setup's FoM-Optimal (M, W)\n"
        "(brackets show the (M, W) that maximises FoM for that setup)",
        fontsize=12, pad=10,
    )
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fname = "21_recall_at_best_fom.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 22 — NC fraction (M=1) vs Recall at best FoM (scatter + OLS)
# ──────────────────────────────────────────────────────────────────────
def plot_nc_recall_at_best_fom(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """Scatter: NC detection fraction (M=1) vs Recall at each setup's FoM-optimal (M, W).

    OLS regression + 95 % CI overlay via _regression_overlay.
    Saved as 22_nc_recall_at_best_fom.png.
    """
    if len(results) < 2:
        print("  [SKIP] 22_nc_recall_at_best_fom.png: fewer than 2 setups.")
        return

    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    color_pts = [color_map.get(r.label, "gray") for r in results]
    labels = [r.label for r in results]

    nc_fracs = np.array([_cc_nc_frac(r, 1) for r in results])
    recalls_best: list[float] = []
    for r in results:
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        grid = _cc_fom_grid(r, M_values, W_values, _tp)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best_mw = max(valid, key=valid.__getitem__)
            recalls_best.append(_cc_recall(r, best_mw[0], best_mw[1]))
        else:
            recalls_best.append(float("nan"))
    recalls_arr = np.array(recalls_best)

    mask = np.isfinite(recalls_arr)
    if mask.sum() < 2:
        print("  [SKIP] 22_nc_recall_at_best_fom.png: fewer than 2 finite values.")
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
    fname = "22_nc_recall_at_best_fom.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 23 — Ge-77 survival at FoM-optimal (M, W) per setup
# ──────────────────────────────────────────────────────────────────────
def plot_ge77_survival_at_best_fom(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """Scatter: Ge-77 survival (Recall) at each setup's FoM-optimal (M, W) vs setup index.

    Ge77 survival = TP / (TP + FN) = Recall.
    Each point is annotated with the (M, W) pair that maximises FoM for that setup.
    """
    results = _w2_sorted(results)
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    colors = [color_map.get(r.label, "gray") for r in results]

    values: list[float] = []
    mw_labels: list[str] = []
    for r in results:
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        grid = _cc_fom_grid(r, M_values, W_values, _tp)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best_mw = max(valid, key=valid.__getitem__)
            values.append(_cc_recall(r, best_mw[0], best_mw[1]))
            mw_labels.append(f"M{best_mw[0]}W{best_mw[1]}")
        else:
            values.append(float("nan"))
            mw_labels.append("N/A")

    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(max(6, len(results) * 0.9), 5))

    for i, (r, c, val, mw) in enumerate(zip(results, colors, values, mw_labels)):
        if np.isfinite(val):
            ax.scatter([i], [val], color=c, s=80, zorder=3)
            ax.annotate(
                mw, xy=(i, val), xytext=(0, 8),
                textcoords="offset points", fontsize=7, ha="center", color=c,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([r.label for r in results], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Setup", fontsize=11)
    ax.set_ylabel("Ge-77 Survival  (Recall = TP / (TP+FN))", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    finite_vals_23 = [v for v in values if np.isfinite(v)]
    if finite_vals_23:
        vmin23, vmax23 = min(finite_vals_23), max(finite_vals_23)
        margin23 = max((vmax23 - vmin23) * 0.4, 0.002)
        ax.set_ylim(max(0.0, vmin23 - margin23), min(1.0, vmax23 + margin23))
    ax.set_title(
        "Ge-77 Muon Survival at Each Setup's FoM-Optimal (M, W)\n"
        "(labels show the (M, W) that maximises FoM for that setup)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fname = "23_ge77_survival_at_best_fom.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 24 — Signal survival at FoM-optimal (M, W) per setup
# ──────────────────────────────────────────────────────────────────────
def plot_signal_survival_at_best_fom(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """Scatter: signal survival 1-(TP+FP)/(TP+FP+TN+FN) at each setup's FoM-optimal (M, W) vs setup index.

    Signal survival = (TN+FN) / (TP+FP+TN+FN) — fraction of muons that did NOT
    trigger a veto at the optimal operating point.
    """
    results = _w2_sorted(results)
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    colors = [color_map.get(r.label, "gray") for r in results]

    values: list[float] = []
    mw_labels: list[str] = []
    for r in results:
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        grid = _cc_fom_grid(r, M_values, W_values, _tp)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best_mw = max(valid, key=valid.__getitem__)
            values.append(_cc_signal_survival(r, best_mw[0], best_mw[1]))
            mw_labels.append(f"M{best_mw[0]}W{best_mw[1]}")
        else:
            values.append(float("nan"))
            mw_labels.append("N/A")

    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(max(6, len(results) * 0.9), 5))

    for i, (r, c, val, mw) in enumerate(zip(results, colors, values, mw_labels)):
        if np.isfinite(val):
            ax.scatter([i], [val], color=c, s=80, zorder=3)
            ax.annotate(
                mw, xy=(i, val), xytext=(0, 8),
                textcoords="offset points", fontsize=7, ha="center", color=c,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([r.label for r in results], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Setup", fontsize=11)
    ax.set_ylabel("Signal Survival = (TN+FN) / (TP+FP+TN+FN)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.3f}%"))
    finite_vals_24 = [v for v in values if np.isfinite(v)]
    if finite_vals_24:
        vmin24, vmax24 = min(finite_vals_24), max(finite_vals_24)
        margin24 = max((vmax24 - vmin24) * 0.4, 0.00005)
        ax.set_ylim(max(0.0, vmin24 - margin24), min(1.0, vmax24 + margin24))
    ax.set_title(
        "Signal Survival at Each Setup's FoM-Optimal (M, W)\n"
        "(1 − veto fraction; labels show optimal (M, W))",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fname = "24_signal_survival_at_best_fom.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Shared helper — FoM background colormap + contours
# ──────────────────────────────────────────────────────────────────────
def _fom_colormap_background(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    n_grid: int = 300,
) -> "matplotlib.cm.ScalarMappable":
    """Draw a FoM(signal_surv, ge_surv) colormap + labelled contours on ax.

    The grid is clipped to the convex hull of the visible data range (with
    a 5 % margin on each side).  Returns the pcolormesh artist so the
    caller can attach a colorbar.
    """
    if not xs or not ys:
        return None
    mx, mx2 = min(xs), max(xs)
    my, my2 = min(ys), max(ys)
    dx = max((mx2 - mx) * 0.05, 1e-4)
    dy = max((my2 - my) * 0.05, 1e-4)
    xg = np.linspace(mx - dx, mx2 + dx, n_grid)
    yg = np.linspace(my - dy, my2 + dy, n_grid)
    XX, YY = np.meshgrid(xg, yg)
    _fom_vec = np.vectorize(figure_of_merit)
    ZZ = _fom_vec(YY, XX)  # ge_surv=YY, signal_surv=XX
    ZZ = np.where(np.isfinite(ZZ), ZZ, np.nan)
    pcm = ax.pcolormesh(XX, YY, ZZ, cmap="viridis", alpha=0.35, shading="auto", zorder=0)
    finite_z = ZZ[np.isfinite(ZZ)]
    if finite_z.size > 0:
        n_levels = 8
        levels = np.linspace(finite_z.min(), finite_z.max(), n_levels + 2)[1:-1]
        cs = ax.contour(XX, YY, ZZ, levels=levels, colors="gray",
                        linewidths=0.6, alpha=0.8, zorder=1)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
    return pcm


def _parse_advisor_csv(path: str) -> list[dict]:
    """Parse advisor fom_data.csv (key=value, comma-separated per line)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = {}
            for item in line.split(","):
                k, v = item.strip().split("=")
                d[k.strip()] = float(v.strip())
            rows.append(d)
    return rows


# ──────────────────────────────────────────────────────────────────────
# Plot 25 — ge_surv vs (1 − deadtime) trade-off scatter
# ──────────────────────────────────────────────────────────────────────
def plot_ge_surv_vs_livetime(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """Scatter: Ge77-NC-weighted ge_surv (y) vs 1 − deadtime (x) per setup.

    Each point corresponds to one (M, W) combination for one setup.
    Lower ge_surv = fewer Ge77 isotopes survive (better background rejection).
    Higher 1 − deadtime = more signal livetime (better).
    Good operating points sit in the bottom-right corner.
    """
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    colors = [color_map.get(r.label, "gray") for r in results]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Collect all points first so the colormap spans the visible data range.
    all_xs: list[float] = []
    all_ys: list[float] = []
    per_setup: list[tuple[list[float], list[float]]] = []
    for r in results:
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        xs, ys = [], []
        for M in M_values:
            for W in W_values:
                cm = r.muon["confusion"].get((M, W))
                if cm is None:
                    continue
                TN = _tp - cm["TP"] - cm["FP"] - cm["FN"]
                ge_surv  = calc_ge_survival_confusion(
                    cm.get("tp_ge77_nc_counts", np.ones(cm["TP"], dtype=np.int32)),
                    cm.get("fn_ge77_nc_counts", np.ones(cm["FN"], dtype=np.int32)),
                )
                deadtime = calc_deadtime_confusion(cm["TP"], cm["FP"], TN, cm["FN"])
                xs.append(1.0 - deadtime)
                ys.append(ge_surv)
        per_setup.append((xs, ys))
        all_xs.extend(xs)
        all_ys.extend(ys)

    pcm = _fom_colormap_background(ax, all_xs, all_ys)
    if pcm is not None:
        fig.colorbar(pcm, ax=ax, label="FoM", pad=0.01)

    for r, c, (xs, ys) in zip(results, colors, per_setup):
        if xs:
            ax.scatter(xs, ys, color=c, s=18, alpha=0.7, zorder=3)
            ax.annotate(
                r.label,
                xy=(float(np.median(xs)), float(np.median(ys))),
                xytext=(4, 3), textcoords="offset points",
                fontsize=7, color=c, fontweight="bold",
            )

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
                       markerfacecolor=color_map.get(r.label, "gray"),
                       markersize=7, label=r.label)
            for r in results
        ],
        fontsize=8, loc="best",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "25_ge_surv_vs_livetime.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 25b — ge_surv vs livetime: advisor's data + user setups at M=6
# ──────────────────────────────────────────────────────────────────────
def plot_ge_surv_vs_livetime_advisor(
    results: list[SetupResult],
    W_values: list[int],
    output_dir: str,
    advisor_csv: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
    M_fixed: int = 6,
) -> None:
    """Ge77 survival vs signal livetime at M=M_fixed, overlaying advisor data.

    Advisor's CSV (threshold=W, signal_surv, ge_surv, FOM) is shown as a
    distinct series.  Each user setup is shown at M=M_fixed for all W in
    W_values.  The FoM colormap spans the combined data range.
    """
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    colors = [color_map.get(r.label, "gray") for r in results]

    advisor_rows = _parse_advisor_csv(advisor_csv)
    adv_xs = [row["signal_surv"] for row in advisor_rows]
    adv_ys = [row["ge_surv"]     for row in advisor_rows]
    adv_ws = [int(row["threshold"]) for row in advisor_rows]

    per_setup: list[tuple[list[float], list[float], list[int]]] = []
    all_xs = list(adv_xs)
    all_ys = list(adv_ys)
    for r in results:
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        xs, ys, ws = [], [], []
        for W in W_values:
            cm = r.muon["confusion"].get((M_fixed, W))
            if cm is None:
                continue
            TN = _tp - cm["TP"] - cm["FP"] - cm["FN"]
            ge_surv  = calc_ge_survival_confusion(
                cm.get("tp_ge77_nc_counts", np.ones(cm["TP"], dtype=np.int32)),
                cm.get("fn_ge77_nc_counts", np.ones(cm["FN"], dtype=np.int32)),
            )
            deadtime = calc_deadtime_confusion(cm["TP"], cm["FP"], TN, cm["FN"])
            xs.append(1.0 - deadtime)
            ys.append(ge_surv)
            ws.append(W)
        per_setup.append((xs, ys, ws))
        all_xs.extend(xs)
        all_ys.extend(ys)

    fig, ax = plt.subplots(figsize=(10, 7))
    pcm = _fom_colormap_background(ax, all_xs, all_ys)
    if pcm is not None:
        fig.colorbar(pcm, ax=ax, label="FoM", pad=0.01)

    # Advisor data
    ax.scatter(adv_xs, adv_ys, color="black", s=40, marker="D", zorder=4,
               label=f"Advisor (M={M_fixed})")
    for x, y, w in zip(adv_xs, adv_ys, adv_ws):
        ax.annotate(f"W={w}", xy=(x, y), xytext=(3, 3),
                    textcoords="offset points", fontsize=6, color="black")

    # User setups at M=M_fixed
    for r, c, (xs, ys, ws) in zip(results, colors, per_setup):
        if xs:
            ax.scatter(xs, ys, color=c, s=22, alpha=0.8, zorder=3)
            ax.annotate(
                r.label,
                xy=(float(np.median(xs)), float(np.median(ys))),
                xytext=(4, 3), textcoords="offset points",
                fontsize=7, color=c, fontweight="bold",
            )

    ax.set_xlabel("1 − Deadtime  (signal livetime fraction)", fontsize=11)
    ax.set_ylabel("Ge77 survival  (Σ FN Ge77 NCs / Σ all Ge77 NCs)", fontsize=11)
    ax.set_title(
        f"Ge77 Survival vs Signal Livetime — M={M_fixed}, W sweep\n"
        "(each point = one W value; bottom-right = optimal)",
        fontsize=12,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    handles = [
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="black",
                   markersize=7, label=f"Advisor (M={M_fixed})"),
    ] + [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map.get(r.label, "gray"),
                   markersize=7, label=f"{r.label} (M={M_fixed})")
        for r in results
    ]
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "25b_ge_surv_vs_livetime_advisor.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 25c — ge_surv vs livetime: one plot per setup, all (M,W) labelled
# ──────────────────────────────────────────────────────────────────────
def plot_ge_surv_vs_livetime_per_setup(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """One PNG per setup: scatter of all (M, W) combinations, labelled.

    Points are coloured by M (using a discrete colormap) and annotated
    with the W value so every operating point is identifiable.
    """
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}

    m_cmap = plt.cm.get_cmap("tab10", len(M_values))
    m_colors = {M: m_cmap(i) for i, M in enumerate(M_values)}

    safe_label = str.maketrans(" /\\:*?\"<>|", "__________")

    for r in results:
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        points: list[tuple[float, float, int, int]] = []
        for M in M_values:
            for W in W_values:
                cm = r.muon["confusion"].get((M, W))
                if cm is None:
                    continue
                TN = _tp - cm["TP"] - cm["FP"] - cm["FN"]
                ge_surv  = calc_ge_survival_confusion(
                    cm.get("tp_ge77_nc_counts", np.ones(cm["TP"], dtype=np.int32)),
                    cm.get("fn_ge77_nc_counts", np.ones(cm["FN"], dtype=np.int32)),
                )
                deadtime = calc_deadtime_confusion(cm["TP"], cm["FP"], TN, cm["FN"])
                points.append((1.0 - deadtime, ge_surv, M, W))

        if not points:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        fig, ax = plt.subplots(figsize=(10, 7))
        pcm = _fom_colormap_background(ax, xs, ys)
        if pcm is not None:
            fig.colorbar(pcm, ax=ax, label="FoM", pad=0.01)

        for x, y, M, W in points:
            ax.scatter([x], [y], color=m_colors[M], s=30, alpha=0.85, zorder=3)
            ax.annotate(f"W={W}", xy=(x, y), xytext=(3, 2),
                        textcoords="offset points", fontsize=6,
                        color=m_colors[M], alpha=0.9)

        ax.set_xlabel("1 − Deadtime  (signal livetime fraction)", fontsize=11)
        ax.set_ylabel("Ge77 survival  (Σ FN Ge77 NCs / Σ all Ge77 NCs)", fontsize=11)
        ax.set_title(
            f"Ge77 Survival vs Signal Livetime — {r.label}\n"
            "(each point = one (M, W) combination; colour = M; label = W)",
            fontsize=12,
        )
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
        ax.legend(
            handles=[
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=m_colors[M], markersize=7,
                           label=f"M={M}")
                for M in M_values
            ],
            fontsize=8, loc="best", title="M threshold",
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe = r.label.translate(safe_label)
        fname = f"25c_ge_surv_vs_livetime_{safe}.png"
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


def write_survival_table(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
) -> None:
    """Write survival_at_best_fom.txt: Ge77 and signal survival at FoM-optimal (M,W) per setup.

    Ge77 survival   = TP / (TP+FN)          = Recall
    Signal survival = (TP+FP) / all_muons
    """
    fpath = os.path.join(output_dir, "survival_at_best_fom.txt")
    W = 80
    with open(fpath, "w") as f:
        f.write("Survival at FoM-Optimal (M, W) per Setup\n")
        f.write("=" * W + "\n\n")
        f.write("  Ge77 survival   = TP / (TP + FN)            = Recall\n")
        f.write("  Signal survival = (TP + FP) / all_muons\n")
        _denom_note = total_primaries if total_primaries > 0 else "per-setup muon count"
        f.write(f"  all_muons       = {_denom_note:,}\n\n")

        hdr = (
            f"  {'Setup':<25}  {'(M,W)':>8}  "
            f"{'Ge77 Survival':>15}  {'Signal Survival':>16}  "
            f"{'TP':>8}  {'FP':>8}  {'FN':>8}\n"
        )
        f.write(hdr)
        f.write("  " + "-" * (len(hdr) - 3) + "\n")

        for r in results:
            _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
            grid = _cc_fom_grid(r, M_values, W_values, _tp)
            valid = {k: v for k, v in grid.items() if np.isfinite(v)}
            if not valid:
                f.write(f"  {r.label:<25}  {'N/A':>8}  {'N/A':>15}  {'N/A':>16}\n")
                continue
            best_mw = max(valid, key=valid.__getitem__)
            M_b, W_b = best_mw
            cm = r.muon["confusion"].get((M_b, W_b), {})
            tp = cm.get("TP", 0)
            fp = cm.get("FP", 0)
            fn = cm.get("FN", 0)
            ge77_surv = _cc_recall(r, M_b, W_b)
            sig_surv  = _cc_signal_survival(r, M_b, W_b)
            f.write(
                f"  {r.label:<25}  {f'M{M_b}W{W_b}':>8}  "
                f"{ge77_surv*100:>14.2f}%  {sig_surv*100:>15.4f}%  "
                f"{tp:>8,}  {fp:>8,}  {fn:>8,}\n"
            )
    print("  Saved survival_at_best_fom.txt")


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
    parser.add_argument(
        "--advisor-csv", default=None, metavar="CSV",
        help=(
            "Path to advisor's fom_data.csv for plot 25b overlay "
            "(key=value format: threshold, signal_surv, ge_surv, FOM)."
        ),
    )
    parser.add_argument(
        "--advisor-M", type=int, default=6,
        help="Fixed M value used in the advisor comparison plot (default: 6).",
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

    # Build a global color map keyed by label so every setup uses the same
    # colour in all plots regardless of which subsets each function receives.
    _pal_global = _colors(len(results))
    color_map   = {r.label: _pal_global[i] for i, r in enumerate(results)}

    # Figure of Merit — use total primary muons (all muons, not just NC-producing)
    _n_runs          = len(count_vertices_by_run(args.sim_dirs[0], omit_runs=omit_runs))
    _total_primaries = _n_runs * MUONS_PER_RUN_DIR
    _runtime_h    = _total_primaries / MUSUN_RATE
    _runtime_yr   = _runtime_h / (24 * 365.25)
    print(f"\n  FoM: total primary muons = {_total_primaries:,}  ({_n_runs} runs × {MUONS_PER_RUN_DIR:,})")
    print(f"  FoM: simulated livetime  = {_runtime_h:,.0f} h  =  {_runtime_yr:.2f} yr  (at {MUSUN_RATE} µ/h)")

    plot_nc_coverage_line(results, M_values, args.output_dir,
                          color_map=color_map)
    # Multiplicity histogram — secondary diagnostic, omitted from main output:
    # plot_nc_multiplicity_histogram(results, M_default, args.output_dir)
    plot_nc_detectability_overview(results, M_default, args.output_dir,
                                   color_map=color_map)
    plot_ge77_muon_overview(results, M_default, W_default, args.output_dir,
                            color_map=color_map,
                            total_primary_muons=_total_primaries)
    # Heatmaps deactivated — not useful for comparison at this scale.
    _heatmap_ms = [m for m in [1, 3, 5, 10] if m in M_values]
    # plot_muon_heatmaps(results, _heatmap_ms, W_values, args.output_dir)
    plot_confusion_bar(results, M_default, W_default, args.output_dir,
                       color_map=color_map)
    # W histogram — too noisy at this scale, omitted:
    # plot_w_histogram(results, M_default, W_default, args.output_dir)

    # M×W sweep (Recall, Precision, FoM)
    plot_mw_sweep(results, M_values, W_values, args.output_dir,
                  total_primaries=_total_primaries, color_map=color_map)
    plot_fom_summary(results, M_values, W_values, args.output_dir,
                     total_primaries=_total_primaries, color_map=color_map)
    plot_fom_summary_min_m(results, M_values, W_values, args.output_dir,
                           min_M=6, total_primaries=_total_primaries, color_map=color_map)
    # plot_fom_per_setup: per-setup FoM heatmaps deactivated.
    # plot_fom_per_setup(results, M_values, W_values, args.output_dir, total_primaries=_total_primaries)
    plot_w2_fom_best(results, M_values, W_values, args.output_dir,
                     total_primaries=_total_primaries, color_map=color_map)
    # plot_w2_sorted_heatmaps: W2-sorted heatmaps deactivated.
    # plot_w2_sorted_heatmaps(results, M_values, W_values, args.output_dir, total_primaries=_total_primaries)

    plot_w2_nc_scatter(results, M_values, args.output_dir, color_map=color_map)
    plot_w2_scatter(results, M_values, M_default, W_default, W_values, args.output_dir,
                    color_map=color_map)

    # W2 correlation analysis
    # nc_correlation: only M=1..4 where the relationship is statistically significant.
    _corr_ms = [m for m in M_values if m <= 4]
    plot_w2_nc_correlation(results, _corr_ms, args.output_dir,
                           color_map=color_map)
    # muon_correlation: large 30-panel grid — omitted; Spearman summary covers it.
    # plot_w2_muon_correlation(results, M_values, W_default, args.output_dir,
    #                          color_map=color_map)
    plot_w2_correlation_matrix(results, M_values, M_default, W_default, args.output_dir)
    plot_w2_coverage_profile(results, M_values, args.output_dir)
    plot_w2_spearman_vs_m(results, M_values, W_default, args.output_dir,
                          total_primaries=_total_primaries, W_values=W_values)

    plot_nc_recall_correlation(results, _heatmap_ms, args.output_dir,
                               color_map=color_map, W_fixed=W_default)

    # New plots: FoM-optimal analysis
    plot_recall_w1_vs_m(results, M_values, args.output_dir, color_map=color_map)
    plot_recall_at_best_fom(results, M_values, W_values, args.output_dir,
                            total_primaries=_total_primaries, color_map=color_map)
    plot_nc_recall_at_best_fom(results, M_values, W_values, args.output_dir,
                               total_primaries=_total_primaries, color_map=color_map)
    plot_ge77_survival_at_best_fom(results, M_values, W_values, args.output_dir,
                                   total_primaries=_total_primaries, color_map=color_map)
    plot_signal_survival_at_best_fom(results, M_values, W_values, args.output_dir,
                                     total_primaries=_total_primaries, color_map=color_map)
    plot_ge_surv_vs_livetime(results, M_values, W_values, args.output_dir,
                             total_primaries=_total_primaries, color_map=color_map)
    if args.advisor_csv:
        plot_ge_surv_vs_livetime_advisor(
            results, W_values, args.output_dir,
            advisor_csv=args.advisor_csv,
            total_primaries=_total_primaries,
            color_map=color_map,
            M_fixed=args.advisor_M,
        )
    plot_ge_surv_vs_livetime_per_setup(results, M_values, W_values, args.output_dir,
                                       total_primaries=_total_primaries, color_map=color_map)

    # ── 7. Write text files ───────────────────────────────────────────
    print("Writing text summaries ...")
    write_nc_summary(results, M_values, args.output_dir)
    write_confusion_matrices(results, M_values, W_values, args.output_dir)
    write_survival_table(results, M_values, W_values, args.output_dir,
                         total_primaries=_total_primaries)

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
