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
import pandas as pd

# ── Publication-quality global style ──────────────────────────────────
plt.rcParams.update({
    "font.size":              13,
    "axes.titlesize":         14,
    "axes.labelsize":         13,
    "xtick.labelsize":        11,
    "ytick.labelsize":        11,
    "legend.fontsize":        11,
    "legend.title_fontsize":  12,
    "lines.linewidth":        1.5,
    "patch.linewidth":        0.8,
    "axes.grid":              True,
    "grid.alpha":             0.3,
    "axes.axisbelow":         True,
})

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

# ──────────────────────────────────────────────────────────────────────
# Pearson correlation significance (3-sigma hypothesis test)
# ──────────────────────────────────────────────────────────────────────
def _pearson_rcrit(n: int, sigma: float = 3.0) -> float:
    """Critical Pearson |r| for 3-sigma two-sided significance, n samples.

    Under H0: r=0 the statistic t = r*sqrt((n-2)/(1-r^2)) follows Student-t
    with dof = n-2.  We invert this to get r_crit from t_crit.
    Three-sigma is the Gaussian-equivalent two-sided probability.
    """
    p_two = 2.0 * scipy_stats.norm.sf(sigma)   # two-sided p for 3σ
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
    """Draw ±r_crit horizontal lines on ax. Returns r_crit.

    Call this on any line plot whose y-axis is the Pearson correlation
    coefficient.  Do NOT call it on scatter plots or Spearman-only plots.
    """
    r_crit = _pearson_rcrit(n, sigma)
    lbl = f"{sigma:.0f}σ threshold  |r| = {r_crit:.2f}" if draw_label else "_nolegend_"
    ax.axhline( r_crit, color=color, linestyle=linestyle, linewidth=linewidth,
                label=lbl, alpha=0.75)
    ax.axhline(-r_crit, color=color, linestyle=linestyle, linewidth=linewidth,
                label="_nolegend_", alpha=0.75)
    return r_crit


# Two-sided p-value for 3-sigma significance (used for marker filling)
_P_3SIGMA: float = 2.0 * scipy_stats.norm.sf(3.0)

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
    VETO_DURATION_H,
    R_PIT,
    R_ZYL_BOT,
    R_ZYL_TOP,
    R_ZYLINDER,
    Z_BASE_GLOBAL,
)

# ── W2 uniform-reference geometry constants ───────────────────────────
# Z range: full cylinder height [Z_BASE, Z_BASE + 8900 mm].
# 8900 mm = H_CYLINDER in homogeneous.py (full wall span, pit at bottom,
# top ring at top).  Z_BASE_GLOBAL = Z_ORIGIN + Z_OFFSET = 20 - 5000 = -4980 mm.
_W2_Z_MIN: float = float(Z_BASE_GLOBAL)          # -4980 mm
_W2_Z_MAX: float = float(Z_BASE_GLOBAL + 8900)   #  3920 mm
# Number of quantile levels shared by W2_z and W2_r (same count → comparable).
_W2_NQUANT: int = 10_001
_W2_R_MAX: float = float(R_ZYLINDER)  # 4300 mm

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
    w2_global: Optional[float] = None  # 3-D W2 homogeneity (if config JSON given)
    w2_z:      Optional[float] = None  # 1-D z-marginal W2 (mm)
    w2_phi:    Optional[float] = None  # circular azimuthal W2 (rad)
    w2_r:      Optional[float] = None  # 1-D radial W2  r=sqrt(x²+y²) (mm)
    per_area_n: dict[str, int] = None  # PMT count per layer (pit/bot/top/wall)
    stat_limit_rows: list = None       # NC-truth stat-limit curve rows for Plot 25/25b


# ──────────────────────────────────────────────────────────────────────
# W2 computation (optional — requires POT and pmtopt package)
# ──────────────────────────────────────────────────────────────────────
def _count_areas_from_json(config_json: str) -> dict[str, int]:
    """Return {layer: count} from a voxel JSON config file. Returns {} on failure."""
    try:
        with open(config_json) as f:
            data = json.load(f)
        voxel_dicts = data if isinstance(data, list) else data.get("selected_voxels", [])
        counts: dict[str, int] = {}
        for v in voxel_dicts:
            if isinstance(v, dict) and "layer" in v:
                layer = v["layer"]
                counts[layer] = counts.get(layer, 0) + 1
        return counts
    except Exception as exc:
        print(f"  [WARN] Area count failed for {config_json!r}: {exc}")
        return {}


def _compute_w2_z(centers: np.ndarray) -> float:
    """1-D W2 between z-marginal of config and Uniform([_W2_Z_MIN, _W2_Z_MAX]).

    Reference: _W2_NQUANT equally-spaced quantile levels of a uniform distribution
    over the full cylinder height [Z_BASE, Z_BASE + 8900 mm].
    The quantile function is Q_ref(q) = Z_MIN + (Z_MAX - Z_MIN) * q.

    This replaces the old approach that used the z-marginal of the 3-D
    surface-area reference, which was NOT uniform in z (it had spikes at the
    flat-surface z-values plus a uniform band from the wall).
    """
    q = np.linspace(0.0, 1.0, _W2_NQUANT)
    q_cfg = np.quantile(centers[:, 2], q)
    q_ref = _W2_Z_MIN + (_W2_Z_MAX - _W2_Z_MIN) * q   # uniform quantile function
    return float(np.sqrt(np.mean((q_cfg - q_ref) ** 2)))


def _compute_w2_phi(centers: np.ndarray) -> float:
    """1-D W2 between phi-marginal of config and Uniform([0, 2π)).

    Reference: Q_ref(q) = 2π × q (fixed linear-uniform, no rotation search).
    W2_phi = 0 only if PMT azimuths are perfectly uniformly spaced starting at 0.
    """
    phi = (np.arctan2(centers[:, 1], centers[:, 0]) + 2 * np.pi) % (2 * np.pi)
    q = np.linspace(0.0, 1.0, _W2_NQUANT)
    q_cfg = np.quantile(phi, q)
    q_ref = 2.0 * np.pi * q
    return float(np.sqrt(np.mean((q_cfg - q_ref) ** 2)))


def _compute_w2_r(centers: np.ndarray) -> float:
    """1-D W2 between radial marginal and Uniform([0, R_ZYLINDER]).

    Reference: Q_ref(q) = R_ZYLINDER × q (linear-uniform on [0, R_ZYLINDER]).
    W2_r = 0 only if PMTs are uniformly spaced in r from 0 to R_ZYLINDER.
    The homogeneous (surface-proportional) setup concentrates 69 % at r = R_ZYLINDER
    and therefore has W2_r >> 0.
    """
    r_cfg = np.sqrt(centers[:, 0] ** 2 + centers[:, 1] ** 2)
    q = np.linspace(0.0, 1.0, _W2_NQUANT)
    q_cfg = np.quantile(r_cfg, q)
    q_ref = _W2_R_MAX * q
    return float(np.sqrt(np.mean((q_cfg - q_ref) ** 2)))


def _try_compute_w2(config_json: str) -> dict[str, Optional[float]]:
    """Compute W2_global, W2_z, W2_phi, and W2_r from a voxel JSON file.

    W2_global — 3-D EMD vs surface-area-proportional reference (unchanged).
    W2_z      — 1-D quantile W2 vs Uniform([Z_BASE, Z_BASE+8900 mm]).
    W2_phi    — circular W2 vs Uniform([0, 2π)) via cyclic-shift search.
    W2_r      — 1-D quantile W2 vs area-uniform distribution on [0, R_ZYLINDER].
    """
    empty: dict[str, Optional[float]] = {"w2_global": None, "w2_z": None, "w2_phi": None, "w2_r": None}
    try:
        from pmtopt.homogeneous import compute_wasserstein_homogeneity, get_w2_ref
    except ImportError:
        print("  [WARN] pmtopt not importable; W2 will not be computed.")
        return empty
    try:
        with open(config_json) as f:
            data = json.load(f)
        voxel_dicts = (
            data if isinstance(data, list)
            else data.get("selected_voxels", [])
        )
        voxel_dicts = [v for v in voxel_dicts if isinstance(v, dict) and "center" in v]
        if len(voxel_dicts) < 2:
            return empty
        centers = np.array([v["center"] for v in voxel_dicts], dtype=float)
        r_cfg = np.sqrt(centers[:, 0] ** 2 + centers[:, 1] ** 2)
        # Diagnostic: show r statistics to verify input to W2_r
        print(
            f"  [W2_r diag] N={len(r_cfg)}  "
            f"r: min={r_cfg.min():.1f}  median={np.median(r_cfg):.1f}  "
            f"max={r_cfg.max():.1f}  "
            f"frac@r>{R_ZYLINDER-200:.0f}={np.mean(r_cfg > R_ZYLINDER-200):.3f}  "
            f"center[0] range=[{centers[:,0].min():.1f}, {centers[:,0].max():.1f}]"
        )
        ref = get_w2_ref()
        w2_global = float(compute_wasserstein_homogeneity(centers, reference=ref)["w2"])
        w2_z      = _compute_w2_z(centers)    # uniform z reference
        w2_phi    = _compute_w2_phi(centers)   # uniform phi reference (unchanged)
        w2_r      = _compute_w2_r(centers)    # area-uniform r reference
        return {"w2_global": w2_global, "w2_z": w2_z, "w2_phi": w2_phi, "w2_r": w2_r}
    except Exception as exc:
        print(f"  [WARN] W2 computation failed for {config_json!r}: {exc}")
        return empty


def plot_w2_uniform_ref_validation(output_dir: str) -> None:
    """Generate validation plots for the W2 uniform reference distributions.

    Produces ``00_w2_ref_validation.png`` with three panels:

    - W2_z: CDF of the uniform z reference vs the expected straight line.
    - W2_phi: histogram of the N-midpoint uniform phi reference (should be flat).
    - W2_r: CDF of the area-uniform r reference vs expected sqrt curve.

    Also prints a short diagnostics table confirming:
    - N_ref is identical for z and r (_W2_NQUANT).
    - Phi midpoints lie strictly inside (0, 2π) — no boundary duplication.
    - z and r spans match the detector geometry.
    """
    n = _W2_NQUANT
    q = np.linspace(0.0, 1.0, n)

    # Reference quantile arrays (all three now linear-uniform on their range)
    ref_z   = _W2_Z_MIN + (_W2_Z_MAX - _W2_Z_MIN) * q   # Uniform z
    ref_r   = _W2_R_MAX * q                               # Uniform r (linear)
    ref_phi = 2.0 * np.pi * q                             # Uniform phi

    # ── Diagnostics ──────────────────────────────────────────────────
    print("  [W2 Ref Validation]")
    print(f"    N_ref (z, r, phi): {n}")
    print(f"    z   ∈ [{ref_z[0]:.1f}, {ref_z[-1]:.1f}] mm  "
          f"(span = {ref_z[-1] - ref_z[0]:.0f} mm) ✓")
    print(f"    r   ∈ [{ref_r[0]:.4f}, {ref_r[-1]:.1f}] mm  "
          f"(max = R_ZYLINDER = {_W2_R_MAX:.0f} mm) ✓")
    print(f"    phi ∈ [{ref_phi[0]:.6f}, {ref_phi[-1]:.6f}] rad  "
          f"(span = {ref_phi[-1]:.4f} ≈ 2π = {2*np.pi:.4f}) ✓")

    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1 — W2_z: CDF of uniform z reference
    ax = axes[0]
    ax.plot(ref_z, q, color="#1f77b4", linewidth=1.8, label="Uniform ref CDF")
    ax.plot([_W2_Z_MIN, _W2_Z_MAX], [0.0, 1.0], "k--", linewidth=1.0,
            label="Expected (linear)")
    ax.set_xlabel("z  (mm)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title(
        f"W2_z reference\nUniform([{_W2_Z_MIN:.0f}, {_W2_Z_MAX:.0f}] mm)\n"
        f"N_ref = {n:,}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2 — W2_phi: CDF of linear-uniform phi reference
    ax = axes[1]
    ax.plot(np.degrees(ref_phi), q, color="#2ca02c", linewidth=1.8, label="Uniform ref CDF")
    ax.plot([0.0, 360.0], [0.0, 1.0], "k--", linewidth=1.0, label="Expected (linear)")
    ax.set_xlabel("φ  (degrees)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title(
        f"W2_phi reference\nUniform([0, 2π))  — fixed (no rotation search)\n"
        f"N_ref = {n:,}  |  Q(q) = 2π·q",
        fontsize=11,
    )
    ax.set_xlim(0, 360)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3 — W2_r: CDF of linear-uniform r reference
    ax = axes[2]
    ax.plot(ref_r, q, color="#d62728", linewidth=1.8, label="Linear-uniform ref CDF")
    ax.plot([0.0, _W2_R_MAX], [0.0, 1.0], "k--", linewidth=1.0,
            label=r"Expected: $F(r)=r/R_{max}$")
    ax.set_xlabel("r  (mm)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title(
        f"W2_r reference\nUniform([0, {_W2_R_MAX:.0f} mm])\n"
        f"N_ref = {n:,}  |  Q(q) = R_max·q",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "W2 Uniform Reference Validation\n"
        "All three metrics use Uniform references with N_ref = {:,} quantile levels".format(n),
        fontsize=12,
    )
    fig.tight_layout()
    fname = "00_w2_ref_validation.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


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
    return sorted(results, key=lambda r: (r.w2_global is None, r.w2_global or 0.0))


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
    """Line plot: NC detection fraction vs M for all configs."""
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    colors = [color_map.get(r.label, "gray") for r in results]

    fig, ax = plt.subplots(figsize=(9, 6))
    for r, c in zip(results, colors):
        nc_total = r.nc["nc_total"]
        fracs = [r.nc["nc_detected"][M] / max(nc_total, 1) for M in M_values]
        ax.plot(M_values, fracs, marker="o", color=c, label=r.label)

    ax.set_xlabel("M — minimum firing PMTs per NC")
    ax.set_ylabel("NC detection fraction")
    ax.set_title("NC Coverage vs M threshold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(M_values)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    fig.tight_layout()
    fname = "01_nc_coverage_line.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 01b — NC Coverage rank stability (Spearman rank correlation vs M)
# ──────────────────────────────────────────────────────────────────────
def plot_nc_rank_spearman(
    results: list["SetupResult"],
    M_values: list[int],
    output_dir: str,
    M_ref: int = 1,
) -> None:
    """Spearman rank stability of NC coverage rankings vs M (vs reference M=M_ref).

    High ρ ≈ 1 means the relative ordering of setups is preserved at that M.
    Filled markers indicate 3-sigma significance.
    Saved as 01b_nc_rank_spearman.png.
    """
    if len(results) < 3:
        print("  [SKIP] 01b_nc_rank_spearman: fewer than 3 setups.")
        return
    if M_ref not in M_values:
        M_ref = M_values[0]

    fracs = {M: np.array([_cc_nc_frac(r, M) for r in results]) for M in M_values}
    ref_ranks = scipy_stats.rankdata(-fracs[M_ref])

    rho_vs_ref, p_vs_ref = [], []

    for M in M_values:
        ranks_m = scipy_stats.rankdata(-fracs[M])
        rho, p = scipy_stats.spearmanr(ref_ranks, ranks_m)
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

    ax.set_xlabel("Multiplicity threshold M", fontsize=13)
    ax.set_ylabel("Spearman ρ of NC coverage rankings", fontsize=13)
    ax.set_title(
        f"NC Coverage Rank Stability vs M\n"
        f"(filled = 3σ significant; reference M = {M_ref})",
        fontsize=14,
    )
    ax.set_xticks(M_values)
    ax.set_ylim(-0.15, 1.15)
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = "01b_nc_rank_spearman.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


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
    ax_abs.set_xticklabels(abs_cat_labels, fontsize=11)
    ax_abs.set_ylabel(ylabel)
    ax_abs.set_title(title)
    ax_abs.legend(fontsize=11)
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
    ax_delta.set_yticklabels(delta_cat_labels, fontsize=11)
    ax_delta.set_xlabel(f"Δ count vs reference ({ref_result.label})", fontsize=13)
    ax_delta.set_title(
        f"Δ from reference: {ref_result.label}", fontsize=10
    )
    ax_delta.legend(fontsize=11, title="vs reference")
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
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
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
    ax_abs.legend(fontsize=11)
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
        ax_delta.set_yticklabels(delta_cat_labels, fontsize=11)
        ax_delta.set_xlabel(f"Δ muon count vs reference ({ref_result.label})", fontsize=13)
        ax_delta.set_title(f"Δ from reference: {ref_result.label}", fontsize=10)
        ax_delta.legend(fontsize=11, title="vs reference")
        ax_delta.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):+,}"))
        ax_delta.grid(True, axis="x", alpha=0.3)
        ax_delta.invert_yaxis()

    fig.tight_layout()
    fname = "04_ge77_muon_overview.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 05 — NC coverage line (M sweep) — already plot 01; keeping numbering
# Plot 05 — Multiplicity histogram  — plot 02
# Now: Plot 05 — Muon heatmaps
def plot_confusion_bar(
    results: list[SetupResult],
    M_default: int,
    W_default: int,
    output_dir: str,
    color_map: dict[str, str] | None = None,
    total_primaries: int = 0,
) -> None:
    """Four sub-figures (2×2): TP, FN, TN, FP at (M_default, W_default).

    Values are expressed as percentages of the total simulated muon count
    (total_primaries when given; falls back to TP+FP+TN+FN per setup).
    Absolute counts are shown as bar annotations.
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
        all_pcts = []
        for i, (r, c) in enumerate(zip(results, colors)):
            conf = r.muon["confusion"][(M_default, W_default)]
            tn_actual = total_primaries - conf["TP"] - conf["FP"] - conf["FN"]
            counts = {"TP": conf["TP"], "FN": conf["FN"], "TN": tn_actual, "FP": conf["FP"]}
            val  = counts[key]
            pct = 100.0 * val / max(total_primaries, 1)
            all_pcts.append(pct)
            bar = ax.bar([i], [pct], 0.6, label=r.label, color=c)
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2,
                bar[0].get_height(),
                f"{pct:.2f}%\n({val:,})",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
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
        ax.set_xticklabels([r.label for r in results], rotation=45, ha="right", fontsize=11)
        ax.set_ylabel("% of total simulated muons")
        ax.set_title(cat_label, fontsize=13)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}%"))
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Ge-77 Muon Classification (W≥{W_default}, M≥{M_default})\n"
        f"(values as % of {total_primaries:,} total simulated muons)",
        fontsize=13,
    )

    fig.tight_layout()
    fname = "06_confusion_bar.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_tp_fp_scatter(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    color_map: dict[str, str] | None = None,
) -> None:
    """Scatter plot of TP vs FP across all (M, W) combinations and setups.

    Tests the hypothesis: more TP → more FP, which would indicate that setups
    detect more light globally rather than improving discrimination.
    Pearson r and Spearman ρ are annotated; an OLS regression line is overlaid.
    Saved as 07b_tp_fp_scatter.png.
    """
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}

    fig, ax = plt.subplots(figsize=(9, 7))

    all_tp: list[float] = []
    all_fp: list[float] = []

    for r in results:
        c = color_map.get(r.label, "gray")
        tp_vals, fp_vals = [], []
        for M in M_values:
            for W in W_values:
                cm = r.muon["confusion"].get((M, W))
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
            transform=ax.transAxes, fontsize=10, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color_map.get(r.label, "gray"),
                       markersize=7, label=r.label)
            for r in results
        ],
        fontsize=10, loc="upper right",
    )
    ax.set_xlabel("True Positives (TP) — Ge-77 muons correctly classified", fontsize=13)
    ax.set_ylabel("False Positives (FP) — non-Ge-77 muons misclassified", fontsize=13)
    ax.set_title(
        "TP vs FP — all (M, W) combinations\n"
        "(each point = one (M, W) per setup; colour = setup)",
        fontsize=14,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "07b_tp_fp_scatter.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
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
                        cm["TP"], cm["FP"], cm["FN"],
                    )[metric])
                elif metric == "Precision":
                    vals.append(compute_metrics(
                        cm["TP"], cm["FP"], cm["FN"],
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
                        v = compute_metrics(cm["TP"], cm["FP"], cm["FN"])[metric]
                    elif metric == "Precision":
                        v = compute_metrics(cm["TP"], cm["FP"], cm["FN"])[metric]
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
        ax.set_ylabel(metric, fontsize=13)
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
        ax.legend(fontsize=11, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlim(-0.5, len(mw_pairs) - 0.5)

        fig.tight_layout()
        fname = f"{plot_num}_mw_sweep_{fname_part}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=300)
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
    ax.set_yticklabels([r.label for r in results], fontsize=11)
    ax.set_xlim(right=y_max * 1.35)
    ax.set_xlabel("Figure of Merit  (max over all M, W)", fontsize=13)
    ax.set_title(
        "Ge-77 Muon Figure of Merit — Best (M, W) per Configuration",
        fontsize=14, pad=10,
    )
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fname = "12_fom_summary.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
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
    ax.set_yticklabels([r.label for r in results], fontsize=11)
    ax.set_xlim(right=y_max * 1.35)
    ax.set_xlabel(f"Figure of Merit  (max over M≥{min_M}, all W)", fontsize=13)
    ax.set_title(
        f"Ge-77 Muon Figure of Merit — Best (M≥{min_M}, W) per Configuration",
        fontsize=14, pad=10,
    )
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fname = f"12b_fom_summary_M{min_M}plus.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


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
    w2_results = [r for r in results if r.w2_global is not None]
    if len(w2_results) < 2:
        print("  [SKIP] 08_w2_nc_scatter*.png: fewer than 2 setups have W2.")
        return

    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    w2_vals = np.array([r.w2_global for r in w2_results])

    panel_ms = sorted({M for M in [1, 2, 4, 5, 10] if M in M_values})
    m_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(panel_ms)))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("W2 Homogeneity vs NC Coverage", fontsize=14, fontweight="bold")

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
    ax.set_title("W2 vs NC Coverage", fontsize=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.legend(title="M threshold", fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = f"08_w2_nc_scatter_M{M_fixed:02d}_W{W_fixed:02d}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 09 — W2 vs Recall at best-FoM (M,W): 4 W2 variants × 2 M sets
# ──────────────────────────────────────────────────────────────────────
def _plot_09_w2_recall_bestfom(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    w2_getter,
    w2_label: str,
    fname_suffix: str,
    min_m: int | None = None,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """W2 vs Recall at best-FoM (M,W) scatter.  One file per W2 variant × M constraint."""
    w2_res = [r for r in results if w2_getter(r) is not None]
    if len(w2_res) < 2:
        return

    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}

    m_search = [M for M in M_values if (min_m is None or M >= min_m)]
    if not m_search:
        return

    w2_arr = np.array([w2_getter(r) for r in w2_res])
    recalls: list[float] = []
    for r in w2_res:
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        grid  = _cc_fom_grid(r, m_search, W_values, _tp)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best = max(valid, key=valid.__getitem__)
            recalls.append(_cc_recall(r, best[0], best[1]))
        else:
            recalls.append(float("nan"))
    rec_arr = np.array(recalls)

    colors = [color_map.get(r.label, "gray") for r in w2_res]
    labels = [r.label for r in w2_res]

    m_tag  = f"M≥{min_m}" if min_m is not None else "all M"
    m_file = f"_Mge{min_m}" if min_m is not None else ""

    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter_corr_panel(
        ax, w2_arr, rec_arr, colors, labels,
        x_label=w2_label,
        y_label=f"Ge-77 Recall  ({m_tag})",
        title=f"{w2_label} vs Recall at Best FoM  ({m_tag})",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    fig.tight_layout()
    fname = f"09_w2_{fname_suffix}_recall_bestfom{m_file}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_w2_recall_best_fom_all_variants(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """Plot 09: W2_x vs Recall at best-FoM for all 4 W2 variants × 2 M constraints."""
    variants = [
        (lambda r: r.w2_global, "W2_global (mm)", "global"),
        (lambda r: r.w2_z,      "W2_z (mm)",      "z"),
        (lambda r: r.w2_phi,    "W2_φ (rad)",      "phi"),
        (lambda r: r.w2_r,      "W2_r (mm)",       "r"),
    ]
    for getter, label, suffix in variants:
        for min_m in (None, 6):
            _plot_09_w2_recall_bestfom(
                results, M_values, W_values, output_dir,
                getter, label, suffix,
                min_m=min_m,
                total_primaries=total_primaries,
                color_map=color_map,
            )


# ──────────────────────────────────────────────────────────────────────
# Plot 09b — W2 vs FoM at best-FoM (M,W): 4 W2 variants × 2 M sets
# ──────────────────────────────────────────────────────────────────────
def _plot_09b_w2_fom_bestfom(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    w2_getter,
    w2_label: str,
    fname_suffix: str,
    min_m: int | None = None,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """W2 vs FoM (best over M,W) scatter. One file per W2 variant × M constraint."""
    w2_res = [r for r in results if w2_getter(r) is not None]
    if len(w2_res) < 2:
        return

    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}

    m_search = [M for M in M_values if (min_m is None or M >= min_m)]
    if not m_search:
        return

    w2_arr = np.array([w2_getter(r) for r in w2_res])
    foms: list[float] = []
    for r in w2_res:
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        grid  = _cc_fom_grid(r, m_search, W_values, _tp)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        foms.append(max(valid.values()) if valid else float("nan"))
    fom_arr = np.array(foms)

    colors = [color_map.get(r.label, "gray") for r in w2_res]
    labels = [r.label for r in w2_res]

    m_tag  = f"M≥{min_m}" if min_m is not None else "all M"
    m_file = f"_Mge{min_m}" if min_m is not None else ""

    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter_corr_panel(
        ax, w2_arr, fom_arr, colors, labels,
        x_label=w2_label,
        y_label=f"Best FoM  ({m_tag})",
        title=f"{w2_label} vs Best FoM  ({m_tag})",
    )
    fig.tight_layout()
    fname = f"09b_w2_{fname_suffix}_fom_bestfom{m_file}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_w2_fom_best_fom_all_variants(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """Plot 09b: W2_x vs FoM at best-FoM for all 4 W2 variants × 2 M constraints."""
    variants = [
        (lambda r: r.w2_global, "W2_global (mm)", "global"),
        (lambda r: r.w2_z,      "W2_z (mm)",      "z"),
        (lambda r: r.w2_phi,    "W2_φ (rad)",      "phi"),
        (lambda r: r.w2_r,      "W2_r (mm)",       "r"),
    ]
    for getter, label, suffix in variants:
        for min_m in (None, 6):
            _plot_09b_w2_fom_bestfom(
                results, M_values, W_values, output_dir,
                getter, label, suffix,
                min_m=min_m,
                total_primaries=total_primaries,
                color_map=color_map,
            )


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
    return compute_metrics(cm["TP"], cm["FP"], cm["FN"])["Recall"]


def _cc_precision(r: SetupResult, M: int, W: int) -> float:
    cm = r.muon["confusion"].get((M, W))
    if cm is None:
        return 0.0
    return compute_metrics(cm["TP"], cm["FP"], cm["FN"])["Precision"]


def _cc_signal_survival(r: SetupResult, M: int, W: int, total_primaries: int) -> float:
    """Signal survival at (M, W): 1 − (TP+FP) / total_primaries."""
    cm = r.muon["confusion"].get((M, W))
    if cm is None:
        return 0.0
    tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]
    tn = total_primaries - tp - fp - fn
    return 1.0 - calc_veto_fraction(tp, fp, tn, fn)


def _sorted_by_w2_cc(results: list[SetupResult]) -> list[SetupResult]:
    """Return results sorted by W2 descending (None W2 last)."""
    return sorted(results, key=lambda r: (r.w2_global is None, -(r.w2_global or 0.0)))


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
    w2_res = [r for r in results if r.w2_global is not None]
    if len(w2_res) < 2:
        print("  [SKIP] w2_nc_correlation: fewer than 2 setups have W2.")
        return

    ordered = _sorted_by_w2_cc(w2_res)
    if color_map is None:
        colors_all = _colors(len(ordered))
        color_map  = {r.label: colors_all[i] for i, r in enumerate(ordered)}
    w2_arr    = np.array([r.w2_global for r in ordered])
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
        ax_scatter.set_title(f"M ≥ {M}", fontsize=12)
        ax_scatter.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
        ax_resid.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
        plt.setp(ax_scatter.get_xticklabels(), visible=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = "13_w2_nc_correlation.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


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
    w2_res = [r for r in results if r.w2_global is not None]
    if len(w2_res) < 3:
        print("  [SKIP] w2_correlation_matrix: fewer than 3 setups have W2.")
        return

    var_names = ["W2"] + [f"NC_M{M}" for M in M_values] + [
        f"Recall\nM{M_default}W{W_default}",
        f"Precision\nM{M_default}W{W_default}",
    ]
    data_rows = []
    for r in w2_res:
        row = [r.w2_global]
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
        fontsize=14,
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
        ax.set_xticklabels(var_names, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(var_names, fontsize=10)
        ax.set_title(title, fontsize=13)

    fig.text(0.5, 0.01, "* p<0.05   ** p<0.01",
             ha="center", fontsize=8, fontstyle="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = "15_w2_correlation_matrix.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
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
    w2_res   = [r for r in results if r.w2_global is not None]
    gray_res = [r for r in results if r.w2_global is None]

    if not w2_res:
        print("  [SKIP] w2_coverage_profile: no setups have W2.")
        return

    w2_vals = np.array([r.w2_global for r in w2_res])
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
        color = cmap(norm(r.w2_global))
        ys = [_cc_nc_frac(r, M) for M in M_values]
        ax.plot(M_values, ys, color=color, linewidth=1.8,
                marker="o", markersize=4, label=r.label)
        ax.annotate(r.label, xy=(M_values[-1], ys[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=6, color=color, va="center")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Global W2 (mm)  [blue=clustered, red=uniform]", fontsize=11)

    ax.set_xlabel("Multiplicity threshold M", fontsize=13)
    ax.set_ylabel("NC detection fraction", fontsize=13)
    ax.set_title(
        "NC Coverage Profile Colored by W2 Homogeneity",
        fontsize=14,
    )
    ax.set_xticks(M_values)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fname = "16_w2_coverage_profile.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
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
    Filled markers = 3σ significant; hollow = not significant.
    """
    w2_res = [r for r in results if r.w2_global is not None]
    if len(w2_res) < 3:
        print("  [SKIP] w2_spearman_vs_m: fewer than 3 setups have W2.")
        return

    w2_arr = np.array([r.w2_global for r in w2_res])

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
        sig  = ps < _P_3SIGMA
        ax.plot(x, rhos, color=color, linewidth=1.5, label=label)
        if sig.any():
            ax.scatter(x[sig],  rhos[sig],  color=color, s=60,
                       marker=marker, zorder=4, label=f"{label} (3σ)")
        if (~sig).any():
            ax.scatter(x[~sig], rhos[~sig], facecolors="none",
                       edgecolors=color, s=60, marker=marker,
                       linewidth=1.2, zorder=4)

    ax.set_xlabel("Multiplicity threshold M", fontsize=13)
    ax.set_ylabel("Spearman ρ  (W2 vs metric)", fontsize=13)
    ax.set_title(
        f"Spearman Correlation between W2 and Coverage Metrics vs M\n"
        f"(filled = 3σ significant | W_default = {W_default})",
        fontsize=14,
    )
    ax.set_xticks(M_values)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fname = "17_w2_spearman_vs_m.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Shared helper — compact scatter with Pearson / Spearman annotation
# ──────────────────────────────────────────────────────────────────────
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

    The annotation includes the 3-sigma Pearson significance threshold so the
    reader can judge significance without referring to a separate legend.
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


# ──────────────────────────────────────────────────────────────────────
# Plot 17a — W2 vs NC Coverage: Pearson r and Spearman ρ per M panel
# ──────────────────────────────────────────────────────────────────────
def plot_w2_nc_coverage_scatter(
    results: list["SetupResult"],
    M_values: list[int],
    output_dir: str,
    color_map: dict[str, str] | None = None,
) -> None:
    """17a — Scatter of W2 vs NC detection fraction for each M threshold.

    Grid of panels (one per M value) each showing W2 on x and NC fraction
    on y, with OLS fit and Pearson r / Spearman ρ annotated.
    Saved as 17a_w2_nc_coverage_scatter.png.
    """
    w2_res = [r for r in results if r.w2_global is not None]
    if len(w2_res) < 2:
        print("  [SKIP] 17a_w2_nc_coverage_scatter: fewer than 2 setups have W2.")
        return

    ordered = _sorted_by_w2_cc(w2_res)
    if color_map is None:
        cols_all  = _colors(len(ordered))
        color_map = {r.label: cols_all[i] for i, r in enumerate(ordered)}
    w2_arr    = np.array([r.w2_global for r in ordered])
    labels    = [r.label for r in ordered]
    color_pts = [color_map.get(r.label, "gray") for r in ordered]

    ncols = min(len(M_values), 4)
    nrows = (len(M_values) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.2),
                              squeeze=False)
    fig.suptitle(
        "W2 Homogeneity vs NC Detection Fraction\n"
        "(OLS fit · Pearson r · Spearman ρ per M)",
        fontsize=13,
    )

    for pi, M in enumerate(M_values):
        row, col = divmod(pi, ncols)
        ax = axes[row][col]
        y_arr = np.array([_cc_nc_frac(r, M) for r in ordered])
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
    fname = "17a_w2_nc_coverage_scatter.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 17b — W2 vs FoM for M ≥ min_m: Pearson r and Spearman ρ vs M
# ──────────────────────────────────────────────────────────────────────
def plot_w2_fom_corr_mge(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    min_m: int = 6,
    total_primaries: int = 0,
) -> None:
    """17b — W2 vs best-FoM-over-W at each M (M ≥ min_m).

    For every eligible M, computes the best FoM over all W per setup, then
    Pearson r and Spearman ρ between W2 values and those FoMs.
    Both correlations are plotted as lines vs M with significance markers.
    Saved as 17b_w2_fom_corr_Mge{min_m}.png.
    """
    w2_res = [r for r in results if r.w2_global is not None]
    if len(w2_res) < 3:
        print("  [SKIP] 17b_w2_fom_corr: fewer than 3 setups have W2.")
        return
    eligible_M = [M for M in M_values if M >= min_m]
    if not eligible_M:
        print(f"  [SKIP] 17b_w2_fom_corr: no M values >= {min_m}.")
        return

    w2_arr = np.array([r.w2_global for r in w2_res])
    _n_w2 = len(w2_res)
    pearson_r, spearman_rho = [], []
    p_pearson, p_spearman   = [], []

    for M in eligible_M:
        fom_arr = []
        for r in w2_res:
            _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
            best = np.nan
            for W in W_values:
                cm = r.muon["confusion"].get((M, W))
                if cm is None:
                    continue
                v = calc_fom_confusion(
                    cm["TP"], cm["FP"], cm["FN"], _tp,
                    tp_ge77_nc_counts=cm.get("tp_ge77_nc_counts"),
                    fn_ge77_nc_counts=cm.get("fn_ge77_nc_counts"),
                )
                if np.isfinite(v) and (np.isnan(best) or v > best):
                    best = v
            fom_arr.append(best)
        fom_arr_np = np.array(fom_arr)
        msk = np.isfinite(fom_arr_np) & np.isfinite(w2_arr)
        if msk.sum() < 3 or np.std(w2_arr[msk]) == 0 or np.std(fom_arr_np[msk]) == 0:
            pearson_r.append(np.nan);   p_pearson.append(np.nan)
            spearman_rho.append(np.nan); p_spearman.append(np.nan)
            continue
        r_val, p_r   = scipy_stats.pearsonr(w2_arr[msk], fom_arr_np[msk])
        rho,   p_rho = scipy_stats.spearmanr(w2_arr[msk], fom_arr_np[msk])
        pearson_r.append(r_val);   p_pearson.append(p_r)
        spearman_rho.append(rho);  p_spearman.append(p_rho)

    x          = np.array(eligible_M)
    pearson_r  = np.array(pearson_r)
    p_pearson  = np.array(p_pearson)
    spearman_rho = np.array(spearman_rho)
    p_spearman   = np.array(p_spearman)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="black", linewidth=0.8)
    _draw_pearson_rcrit(ax, _n_w2, sigma=3.0)
    _draw_pearson_rcrit(ax, _n_w2, sigma=5.0, linestyle="-.")

    for vals, ps, label, color, marker in [
        (pearson_r,   p_pearson,  "Pearson r",   "#1f77b4", "o"),
        (spearman_rho, p_spearman, "Spearman ρ", "#d62728", "s"),
    ]:
        sig  = ps < _P_3SIGMA
        nsig = ~sig & np.isfinite(vals)
        ax.plot(x, vals, color=color, linewidth=1.8, label=label)
        if sig.any():
            ax.scatter(x[sig],  vals[sig],  color=color, s=60, marker=marker, zorder=4)
        if nsig.any():
            ax.scatter(x[nsig], vals[nsig], facecolors="none", edgecolors=color,
                       s=60, marker=marker, linewidth=1.2, zorder=4)

    ax.set_xlabel("Multiplicity threshold M", fontsize=13)
    ax.set_ylabel("Correlation coefficient (W2 vs best FoM)", fontsize=13)
    ax.set_title(
        f"W2 Homogeneity vs FoM  (M ≥ {min_m})  — Pearson r and Spearman ρ vs M\n"
        "(best FoM over all W at each M; filled = 3σ significant; dashed = 3σ / dash-dot = 5σ Pearson threshold)",
        fontsize=14,
    )
    ax.set_xticks(eligible_M)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = f"17b_w2_fom_corr_Mge{min_m}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 17c — W2 vs Muon Recall: scatter for best-FoM and best-FoM-M>=6
# ──────────────────────────────────────────────────────────────────────
def plot_w2_recall_corr_split(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    min_m_constrained: int = 6,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """17c — W2 vs Muon Recall: two scatter panels for direct comparison.

    Left panel  — Recall at best FoM across all M.
    Right panel — Recall at best FoM restricted to M >= min_m_constrained.
    Both panels show OLS fit and Pearson r / Spearman ρ annotations.
    Saved as 17c_w2_recall_corr.png.
    """
    w2_res = [r for r in results if r.w2_global is not None]
    if len(w2_res) < 2:
        print("  [SKIP] 17c_w2_recall_corr: fewer than 2 setups have W2.")
        return

    ordered = _sorted_by_w2_cc(w2_res)
    if color_map is None:
        cols_all  = _colors(len(ordered))
        color_map = {r.label: cols_all[i] for i, r in enumerate(ordered)}
    w2_arr    = np.array([r.w2_global for r in ordered])
    labels    = [r.label for r in ordered]
    color_pts = [color_map.get(r.label, "gray") for r in ordered]

    eligible_all = M_values
    eligible_ge  = [M for M in M_values if M >= min_m_constrained]

    def _recall_at_fom(r, m_search):
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        grid  = _cc_fom_grid(r, m_search, W_values, _tp)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best = max(valid, key=valid.__getitem__)
            return _cc_recall(r, best[0], best[1])
        return float("nan")

    recalls_all = np.array([_recall_at_fom(r, eligible_all) for r in ordered])
    recalls_ge  = (np.array([_recall_at_fom(r, eligible_ge)  for r in ordered])
                   if eligible_ge else np.full(len(ordered), np.nan))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        "W2 Homogeneity vs Muon Recall — Pearson r and Spearman ρ\n"
        "(OLS fit · left: all M · right: M ≥ " + str(min_m_constrained) + ")",
        fontsize=13,
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
    fname = "17c_w2_recall_corr.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 17d — W2 global vs NC coverage: Pearson r and Spearman ρ vs M (all M)
# ──────────────────────────────────────────────────────────────────────
def plot_w2_nc_coverage_corr_all_m(
    results: list["SetupResult"],
    M_values: list[int],
    output_dir: str,
) -> None:
    """17d — W2 global vs NC coverage: Pearson r and Spearman ρ vs M (all M).

    Delegates to _plot_17d_w2_variant with the global W2 getter.
    Saved as 17d_w2_nc_coverage_corr.png.
    """
    _plot_17d_w2_variant(
        results, M_values, output_dir,
        lambda r: r.w2_global, "W2_global (mm)", "17d_w2_nc_coverage_corr.png",
    )


# ──────────────────────────────────────────────────────────────────────
# Plot 17e — W2 global vs Recall: 4-curve Pearson and Spearman correlation
# ──────────────────────────────────────────────────────────────────────
def plot_w2_recall_corr_all_m(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    min_m_constrained: int = 6,
    total_primaries: int = 0,
) -> None:
    """17e — W2 global vs Recall: 4-curve Pearson and Spearman correlation vs M.

    Delegates to _plot_17e_w2_variant with the global W2 getter.
    Saved as 17e_w2_recall_corr.png.
    """
    _plot_17e_w2_variant(
        results, M_values, W_values, output_dir,
        lambda r: r.w2_global, "W2_global (mm)", "17e_w2_recall_corr.png",
        min_m_constrained=min_m_constrained,
        total_primaries=total_primaries,
    )


# ──────────────────────────────────────────────────────────────────────
# Private helpers for parametric 17d (NC coverage) and 17e (4-curve recall)
# ──────────────────────────────────────────────────────────────────────

def _plot_17d_w2_variant(
    results: list["SetupResult"],
    M_values: list[int],
    output_dir: str,
    w2_getter,
    w2_label: str,
    fname: str,
) -> None:
    """Parametric NC coverage correlation for 17d variants (global/z/phi/r).

    Pearson r and Spearman ρ between the selected W2 component and NC detection
    fraction, plotted vs M (all M).  Includes 3σ and 5σ significance lines.
    """
    w2_res = [r for r in results if w2_getter(r) is not None]
    if len(w2_res) < 3:
        print(f"  [SKIP] {fname}: fewer than 3 setups have W2.")
        return

    w2_arr = np.array([w2_getter(r) for r in w2_res])
    _n_w2  = len(w2_res)
    pearson_r, spearman_rho = [], []
    p_pearson, p_spearman   = [], []

    for M in M_values:
        nc_arr = np.array([_cc_nc_frac(r, M) for r in w2_res])
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
    _draw_pearson_rcrit(ax, _n_w2, sigma=5.0, linestyle="-.")

    for vals, ps, label, color, marker in [
        (pearson_r,    p_pearson,  "Pearson r",  "#1f77b4", "o"),
        (spearman_rho, p_spearman, "Spearman ρ", "#d62728", "s"),
    ]:
        sig  = ps < _P_3SIGMA
        nsig = ~sig & np.isfinite(vals)
        ax.plot(x, vals, color=color, linewidth=1.8, label=label)
        if sig.any():
            ax.scatter(x[sig],  vals[sig],  color=color, s=60, marker=marker, zorder=4)
        if nsig.any():
            ax.scatter(x[nsig], vals[nsig], facecolors="none", edgecolors=color,
                       s=60, marker=marker, linewidth=1.2, zorder=4)

    ax.set_xlabel("Multiplicity threshold M", fontsize=13)
    ax.set_ylabel(f"Correlation ({w2_label} vs NC coverage)", fontsize=13)
    ax.set_title(
        f"{w2_label} vs NC Coverage  (all M)  — Pearson r and Spearman ρ vs M\n"
        "(filled = 3σ significant; dashed = 3σ / dash-dot = 5σ Pearson threshold)",
        fontsize=14,
    )
    ax.set_xticks(M_values)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


def _plot_17e_w2_variant(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    w2_getter,
    w2_label: str,
    fname: str,
    min_m_constrained: int = 6,
    total_primaries: int = 0,
) -> None:
    """Parametric 4-curve recall correlation for 17e variants (global/z/phi/r).

    Two-panel figure (Pearson r left, Spearman ρ right), each with four curves:
      1. MW at best FoM   — W* = argmax_W FoM(M,W) at each M (all M)
      2. Recall at best FoM — Recall(M, W*_FoM) for same W* (all M)
      3. MW at best MW    — W* = argmax_W Recall(M,W) at each M (M ≥ min_m_constrained)
      4. Recall at best MW — Recall(M, W*_recall) for same W* (M ≥ min_m_constrained)
    Both panels include 3σ and 5σ Pearson significance threshold lines.
    """
    w2_res = [r for r in results if w2_getter(r) is not None]
    if len(w2_res) < 3:
        print(f"  [SKIP] {fname}: fewer than 3 setups have W2.")
        return

    w2_arr = np.array([w2_getter(r) for r in w2_res])
    _n_w2  = len(w2_res)
    eligible_m = [M for M in M_values if M >= min_m_constrained]

    def _w_at_best_fom(r: "SetupResult", M: int) -> float:
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        fom_grid = _cc_fom_grid(r, [M], W_values, _tp)
        best_fom, best_w = np.nan, np.nan
        for W in W_values:
            fom = fom_grid.get((M, W), np.nan)
            if np.isfinite(fom) and (np.isnan(best_fom) or fom > best_fom):
                best_fom = fom
                best_w = float(W)
        return best_w

    def _recall_at_best_fom(r: "SetupResult", M: int) -> float:
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        fom_grid = _cc_fom_grid(r, [M], W_values, _tp)
        best_fom, best_rec = np.nan, np.nan
        for W in W_values:
            fom = fom_grid.get((M, W), np.nan)
            if np.isfinite(fom) and (np.isnan(best_fom) or fom > best_fom):
                best_fom = fom
                best_rec = _cc_recall(r, M, W)
        return best_rec

    def _w_at_best_recall(r: "SetupResult", M: int) -> float:
        best_rec, best_w = np.nan, np.nan
        for W in W_values:
            rec = _cc_recall(r, M, W)
            if np.isfinite(rec) and (np.isnan(best_rec) or rec > best_rec):
                best_rec = rec
                best_w = float(W)
        return best_w

    def _recall_at_best_recall(r: "SetupResult", M: int) -> float:
        best_rec = np.nan
        for W in W_values:
            rec = _cc_recall(r, M, W)
            if np.isfinite(rec) and (np.isnan(best_rec) or rec > best_rec):
                best_rec = rec
        return best_rec

    def _corr_line(m_list, get_val):
        pr, sr, pp, sp = [], [], [], []
        for M in m_list:
            y = np.array([get_val(r, M) for r in w2_res])
            msk = np.isfinite(y) & np.isfinite(w2_arr)
            if msk.sum() < 3 or np.std(w2_arr[msk]) == 0 or np.std(y[msk]) == 0:
                pr.append(np.nan); pp.append(np.nan)
                sr.append(np.nan); sp.append(np.nan)
                continue
            r_val, p_r = scipy_stats.pearsonr(w2_arr[msk], y[msk])
            rho,  p_rho = scipy_stats.spearmanr(w2_arr[msk], y[msk])
            pr.append(r_val); pp.append(p_r)
            sr.append(rho);   sp.append(p_rho)
        return np.array(pr), np.array(sr), np.array(pp), np.array(sp)

    # (m_list, metric_fn, label, color, marker, linestyle)
    curves = [
        (M_values,   _w_at_best_fom,        "MW at best FoM",     "#ff7f0e", "o", "-"),
        (M_values,   _recall_at_best_fom,   "Recall at best FoM", "#1f77b4", "s", "-"),
        (eligible_m, _w_at_best_recall,     "MW at best MW",      "#2ca02c", "^", "--"),
        (eligible_m, _recall_at_best_recall,"Recall at best MW",  "#d62728", "D", "--"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        f"{w2_label} vs Ge-77 Recall  —  Pearson r (left) / Spearman ρ (right) vs M\n"
        f"Dashed curves restricted to M ≥ {min_m_constrained}  •  "
        "filled markers = 3σ significant  •  horizontal lines = 3σ / 5σ threshold",
        fontsize=11,
    )

    for col_idx, (ax, corr_name) in enumerate(zip(axes, ["Pearson r", "Spearman ρ"])):
        ax.axhline(0, color="black", linewidth=0.8)
        _draw_pearson_rcrit(ax, _n_w2, sigma=3.0)
        _draw_pearson_rcrit(ax, _n_w2, sigma=5.0, linestyle="-.")

        for m_list, get_val, label, color, marker, ls in curves:
            if not m_list:
                continue
            pr, sr, pp, sp = _corr_line(m_list, get_val)
            vals = pr if col_idx == 0 else sr
            ps   = pp if col_idx == 0 else sp
            x    = np.array(m_list)

            sig  = ps < _P_3SIGMA
            nsig = ~sig & np.isfinite(vals)
            ax.plot(x, vals, color=color, linewidth=1.8, label=label, linestyle=ls)
            if sig.any():
                ax.scatter(x[sig],  vals[sig],  color=color, s=60,
                           marker=marker, zorder=4)
            if nsig.any():
                ax.scatter(x[nsig], vals[nsig], facecolors="none", edgecolors=color,
                           s=60, marker=marker, linewidth=1.2, zorder=4)

        ax.set_xlabel("Multiplicity threshold M", fontsize=11)
        ax.set_ylabel(f"Correlation ({w2_label} vs metric)", fontsize=11)
        ax.set_title(corr_name, fontsize=12)
        ax.set_xticks(M_values)
        ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plots 17a_z / 17b_z / 17c_z — same analysis for W2_z component
# Plots 17a_phi / 17b_phi / 17c_phi — same analysis for W2_phi component
# ──────────────────────────────────────────────────────────────────────

def _plot_17a_w2_variant(
    results: list["SetupResult"],
    M_values: list[int],
    output_dir: str,
    w2_getter: "Callable",
    metric_label: str,
    fname: str,
    color_map: dict[str, str] | None = None,
) -> None:
    """Shared scatter grid for 17a_z and 17a_phi variants."""
    w2_res = [r for r in results if w2_getter(r) is not None]
    if len(w2_res) < 2:
        print(f"  [SKIP] {fname}: fewer than 2 setups have {metric_label}.")
        return

    ordered = sorted(w2_res, key=lambda r: -(w2_getter(r) or 0.0))
    if color_map is None:
        cols_all  = _colors(len(ordered))
        color_map = {r.label: cols_all[i] for i, r in enumerate(ordered)}
    w2_arr    = np.array([w2_getter(r) for r in ordered])
    labels    = [r.label for r in ordered]
    color_pts = [color_map.get(r.label, "gray") for r in ordered]

    ncols = min(len(M_values), 4)
    nrows = (len(M_values) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.2), squeeze=False)
    fig.suptitle(
        f"{metric_label} vs NC Detection Fraction\n"
        "(OLS fit · Pearson r · Spearman ρ per M)",
        fontsize=13,
    )

    for pi, M in enumerate(M_values):
        row, col = divmod(pi, ncols)
        ax = axes[row][col]
        y_arr = np.array([_cc_nc_frac(r, M) for r in ordered])
        _scatter_corr_panel(
            ax, w2_arr, y_arr, color_pts, labels,
            x_label=metric_label,
            y_label=f"NC fraction (M≥{M})",
            title=f"M ≥ {M}",
        )
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))

    for pi in range(len(M_values), nrows * ncols):
        row, col = divmod(pi, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


def _plot_17b_w2_variant(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    w2_getter: "Callable",
    metric_label: str,
    fname: str,
    min_m: int = 6,
    total_primaries: int = 0,
) -> None:
    """Shared correlation-line plot for 17b_z and 17b_phi variants."""
    w2_res = [r for r in results if w2_getter(r) is not None]
    if len(w2_res) < 3:
        print(f"  [SKIP] {fname}: fewer than 3 setups have {metric_label}.")
        return
    eligible_M = [M for M in M_values if M >= min_m]
    if not eligible_M:
        print(f"  [SKIP] {fname}: no M values >= {min_m}.")
        return

    w2_arr = np.array([w2_getter(r) for r in w2_res])
    _n_w2 = len(w2_res)
    pearson_r, spearman_rho = [], []
    p_pearson, p_spearman   = [], []

    for M in eligible_M:
        fom_arr = []
        for r in w2_res:
            _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
            best = np.nan
            for W in W_values:
                cm = r.muon["confusion"].get((M, W))
                if cm is None:
                    continue
                v = calc_fom_confusion(
                    cm["TP"], cm["FP"], cm["FN"], _tp,
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

    x          = np.array(eligible_M)
    pearson_r  = np.array(pearson_r)
    p_pearson  = np.array(p_pearson)
    spearman_rho = np.array(spearman_rho)
    p_spearman   = np.array(p_spearman)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="black", linewidth=0.8)
    _draw_pearson_rcrit(ax, _n_w2, sigma=3.0)
    _draw_pearson_rcrit(ax, _n_w2, sigma=5.0, linestyle="-.")

    for vals, ps, lbl, color, marker in [
        (pearson_r,    p_pearson,  "Pearson r",   "#1f77b4", "o"),
        (spearman_rho, p_spearman, "Spearman ρ",  "#d62728", "s"),
    ]:
        sig  = ps < _P_3SIGMA
        nsig = ~sig & np.isfinite(vals)
        ax.plot(x, vals, color=color, linewidth=1.8, label=lbl)
        if sig.any():
            ax.scatter(x[sig],  vals[sig],  color=color, s=60, marker=marker, zorder=4)
        if nsig.any():
            ax.scatter(x[nsig], vals[nsig], facecolors="none", edgecolors=color,
                       s=60, marker=marker, linewidth=1.2, zorder=4)

    ax.set_xlabel("Multiplicity threshold M", fontsize=13)
    ax.set_ylabel(f"Correlation ({metric_label} vs best FoM)", fontsize=13)
    ax.set_title(
        f"{metric_label} vs FoM  (M ≥ {min_m})  — Pearson r and Spearman ρ vs M\n"
        "(best FoM over all W at each M; filled = 3σ significant; dashed = 3σ / dash-dot = 5σ Pearson threshold)",
        fontsize=14,
    )
    ax.set_xticks(eligible_M)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


def _plot_17c_w2_variant(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    w2_getter: "Callable",
    metric_label: str,
    fname: str,
    min_m_constrained: int = 6,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """Shared recall-scatter for 17c_z and 17c_phi variants."""
    w2_res = [r for r in results if w2_getter(r) is not None]
    if len(w2_res) < 2:
        print(f"  [SKIP] {fname}: fewer than 2 setups have {metric_label}.")
        return

    ordered = sorted(w2_res, key=lambda r: -(w2_getter(r) or 0.0))
    if color_map is None:
        cols_all  = _colors(len(ordered))
        color_map = {r.label: cols_all[i] for i, r in enumerate(ordered)}
    w2_arr    = np.array([w2_getter(r) for r in ordered])
    labels    = [r.label for r in ordered]
    color_pts = [color_map.get(r.label, "gray") for r in ordered]

    eligible_all = M_values
    eligible_ge  = [M for M in M_values if M >= min_m_constrained]

    def _recall_at_fom(r, m_search):
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        grid  = _cc_fom_grid(r, m_search, W_values, _tp)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best = max(valid, key=valid.__getitem__)
            return _cc_recall(r, best[0], best[1])
        return float("nan")

    recalls_all = np.array([_recall_at_fom(r, eligible_all) for r in ordered])
    recalls_ge  = (np.array([_recall_at_fom(r, eligible_ge)  for r in ordered])
                   if eligible_ge else np.full(len(ordered), np.nan))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        f"{metric_label} vs Muon Recall — Pearson r and Spearman ρ\n"
        "(OLS fit · left: all M · right: M ≥ " + str(min_m_constrained) + ")",
        fontsize=13,
    )

    for ax, recalls, title in [
        (ax_l, recalls_all, "Recall at best FoM  (all M)"),
        (ax_r, recalls_ge,  f"Recall at best FoM  (M ≥ {min_m_constrained})"),
    ]:
        _scatter_corr_panel(
            ax, w2_arr, recalls, color_pts, labels,
            x_label=f"{metric_label}  [lower = more uniform]",
            y_label="Ge-77 Muon Recall",
            title=title,
        )
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_w2_z_nc_coverage_scatter(
    results: list["SetupResult"],
    M_values: list[int],
    output_dir: str,
    color_map: dict[str, str] | None = None,
) -> None:
    """17a_z — W2_z (z-marginal) vs NC detection fraction scatter grid."""
    _plot_17a_w2_variant(
        results, M_values, output_dir,
        lambda r: r.w2_z, "W2_z (mm)", "17a_z_nc_coverage_scatter.png",
        color_map=color_map,
    )


def plot_w2_phi_nc_coverage_scatter(
    results: list["SetupResult"],
    M_values: list[int],
    output_dir: str,
    color_map: dict[str, str] | None = None,
) -> None:
    """17a_phi — W2_phi (circular azimuthal) vs NC detection fraction scatter grid."""
    _plot_17a_w2_variant(
        results, M_values, output_dir,
        lambda r: r.w2_phi, "W2_φ (rad)", "17a_phi_nc_coverage_scatter.png",
        color_map=color_map,
    )


def plot_w2_z_fom_corr_mge(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    min_m: int = 6,
    total_primaries: int = 0,
) -> None:
    """17b_z — W2_z vs best-FoM-over-W correlation lines vs M (M >= min_m)."""
    _plot_17b_w2_variant(
        results, M_values, W_values, output_dir,
        lambda r: r.w2_z, "W2_z (mm)", f"17b_z_fom_corr_Mge{min_m}.png",
        min_m=min_m, total_primaries=total_primaries,
    )


def plot_w2_phi_fom_corr_mge(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    min_m: int = 6,
    total_primaries: int = 0,
) -> None:
    """17b_phi — W2_phi vs best-FoM-over-W correlation lines vs M (M >= min_m)."""
    _plot_17b_w2_variant(
        results, M_values, W_values, output_dir,
        lambda r: r.w2_phi, "W2_φ (rad)", f"17b_phi_fom_corr_Mge{min_m}.png",
        min_m=min_m, total_primaries=total_primaries,
    )


def plot_w2_z_recall_corr_split(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    min_m_constrained: int = 6,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """17c_z — W2_z vs Muon Recall scatter (all M and M >= min_m_constrained)."""
    _plot_17c_w2_variant(
        results, M_values, W_values, output_dir,
        lambda r: r.w2_z, "W2_z (mm)", "17c_z_recall_corr.png",
        min_m_constrained=min_m_constrained,
        total_primaries=total_primaries,
        color_map=color_map,
    )


def plot_w2_phi_recall_corr_split(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    min_m_constrained: int = 6,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """17c_phi — W2_phi vs Muon Recall scatter (all M and M >= min_m_constrained)."""
    _plot_17c_w2_variant(
        results, M_values, W_values, output_dir,
        lambda r: r.w2_phi, "W2_φ (rad)", "17c_phi_recall_corr.png",
        min_m_constrained=min_m_constrained,
        total_primaries=total_primaries,
        color_map=color_map,
    )


def plot_w2_r_nc_coverage_scatter(
    results: list["SetupResult"],
    M_values: list[int],
    output_dir: str,
    color_map: dict[str, str] | None = None,
) -> None:
    """17a_r — W2_r (radial) vs NC detection fraction scatter grid."""
    _plot_17a_w2_variant(
        results, M_values, output_dir,
        lambda r: r.w2_r, "W2_r (mm)", "17a_r_nc_coverage_scatter.png",
        color_map=color_map,
    )


def plot_w2_r_fom_corr_mge(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    min_m: int = 6,
    total_primaries: int = 0,
) -> None:
    """17b_r — W2_r vs best-FoM-over-W correlation lines vs M (M >= min_m)."""
    _plot_17b_w2_variant(
        results, M_values, W_values, output_dir,
        lambda r: r.w2_r, "W2_r (mm)", f"17b_r_fom_corr_Mge{min_m}.png",
        min_m=min_m, total_primaries=total_primaries,
    )


def plot_w2_r_recall_corr_split(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    min_m_constrained: int = 6,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """17c_r — W2_r vs Muon Recall scatter (all M and M >= min_m_constrained)."""
    _plot_17c_w2_variant(
        results, M_values, W_values, output_dir,
        lambda r: r.w2_r, "W2_r (mm)", "17c_r_recall_corr.png",
        min_m_constrained=min_m_constrained,
        total_primaries=total_primaries, color_map=color_map,
    )


# ──────────────────────────────────────────────────────────────────────
# Plot 17a FoM variant — W2 vs FoM scatter grid (per M panel, all W2 variants)
# ──────────────────────────────────────────────────────────────────────

def _plot_17a_fom_w2_variant(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    w2_getter: "Callable",
    w2_label: str,
    fname: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """Scatter grid: W2 vs best-FoM-over-W for each M panel.

    Analogous to _plot_17a_w2_variant (NC coverage) but uses the best FoM
    optimised over all W thresholds at each fixed M.  Each panel shows one M.
    """
    w2_res = [r for r in results if w2_getter(r) is not None]
    if len(w2_res) < 2:
        print(f"  [SKIP] {fname}: fewer than 2 setups have {w2_label}.")
        return

    ordered = sorted(w2_res, key=lambda r: -(w2_getter(r) or 0.0))
    if color_map is None:
        cols_all  = _colors(len(ordered))
        color_map = {r.label: cols_all[i] for i, r in enumerate(ordered)}
    w2_arr    = np.array([w2_getter(r) for r in ordered])
    labels    = [r.label for r in ordered]
    color_pts = [color_map.get(r.label, "gray") for r in ordered]

    ncols = min(len(M_values), 4)
    nrows = (len(M_values) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.2), squeeze=False)
    fig.suptitle(
        f"{w2_label} vs FoM (best over W)\n"
        "(OLS fit · Pearson r · Spearman ρ per M)",
        fontsize=13,
    )

    for pi, M in enumerate(M_values):
        row, col = divmod(pi, ncols)
        ax = axes[row][col]
        fom_arr: list[float] = []
        for r in ordered:
            _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
            best = np.nan
            for W in W_values:
                cm = r.muon["confusion"].get((M, W))
                if cm is None:
                    continue
                v = calc_fom_confusion(
                    cm["TP"], cm["FP"], cm["FN"], _tp,
                    tp_ge77_nc_counts=cm.get("tp_ge77_nc_counts"),
                    fn_ge77_nc_counts=cm.get("fn_ge77_nc_counts"),
                )
                if np.isfinite(v) and (np.isnan(best) or v > best):
                    best = v
            fom_arr.append(best)
        y_arr = np.array(fom_arr)
        _scatter_corr_panel(
            ax, w2_arr, y_arr, color_pts, labels,
            x_label=w2_label,
            y_label=f"Best FoM (M={M})",
            title=f"M = {M}",
        )

    for pi in range(len(M_values), nrows * ncols):
        row, col = divmod(pi, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_w2_fom_scatter_all_m(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """17a_fom — W2_global vs FoM (best over W) scatter grid (one panel per M)."""
    _plot_17a_fom_w2_variant(
        results, M_values, W_values, output_dir,
        lambda r: r.w2_global, "W2_global (mm)", "17a_fom_scatter.png",
        total_primaries=total_primaries, color_map=color_map,
    )


def plot_w2_z_fom_scatter(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """17a_z_fom — W2_z vs FoM (best over W) scatter grid (one panel per M)."""
    _plot_17a_fom_w2_variant(
        results, M_values, W_values, output_dir,
        lambda r: r.w2_z, "W2_z (mm)", "17a_z_fom_scatter.png",
        total_primaries=total_primaries, color_map=color_map,
    )


def plot_w2_phi_fom_scatter(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """17a_phi_fom — W2_phi vs FoM (best over W) scatter grid (one panel per M)."""
    _plot_17a_fom_w2_variant(
        results, M_values, W_values, output_dir,
        lambda r: r.w2_phi, "W2_φ (rad)", "17a_phi_fom_scatter.png",
        total_primaries=total_primaries, color_map=color_map,
    )


def plot_w2_r_fom_scatter(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
) -> None:
    """17a_r_fom — W2_r vs FoM (best over W) scatter grid (one panel per M)."""
    _plot_17a_fom_w2_variant(
        results, M_values, W_values, output_dir,
        lambda r: r.w2_r, "W2_r (mm)", "17a_r_fom_scatter.png",
        total_primaries=total_primaries, color_map=color_map,
    )


# ──────────────────────────────────────────────────────────────────────
# Plots 17d_z / 17d_phi / 17d_r — NC coverage correlation for W2 variants
# Plots 17e_z / 17e_phi / 17e_r — 4-curve recall correlation for W2 variants
# ──────────────────────────────────────────────────────────────────────

def plot_w2_z_nc_coverage_corr_all_m(
    results: list["SetupResult"],
    M_values: list[int],
    output_dir: str,
) -> None:
    """17d_z — W2_z vs NC coverage: Pearson r and Spearman ρ vs M (all M)."""
    _plot_17d_w2_variant(
        results, M_values, output_dir,
        lambda r: r.w2_z, "W2_z (mm)", "17d_z_nc_coverage_corr.png",
    )


def plot_w2_phi_nc_coverage_corr_all_m(
    results: list["SetupResult"],
    M_values: list[int],
    output_dir: str,
) -> None:
    """17d_phi — W2_phi vs NC coverage: Pearson r and Spearman ρ vs M (all M)."""
    _plot_17d_w2_variant(
        results, M_values, output_dir,
        lambda r: r.w2_phi, "W2_φ (rad)", "17d_phi_nc_coverage_corr.png",
    )


def plot_w2_r_nc_coverage_corr_all_m(
    results: list["SetupResult"],
    M_values: list[int],
    output_dir: str,
) -> None:
    """17d_r — W2_r vs NC coverage: Pearson r and Spearman ρ vs M (all M)."""
    _plot_17d_w2_variant(
        results, M_values, output_dir,
        lambda r: r.w2_r, "W2_r (mm)", "17d_r_nc_coverage_corr.png",
    )


def plot_w2_z_recall_corr_all_m(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    min_m_constrained: int = 6,
    total_primaries: int = 0,
) -> None:
    """17e_z — W2_z vs Recall: 4-curve Pearson and Spearman correlation vs M."""
    _plot_17e_w2_variant(
        results, M_values, W_values, output_dir,
        lambda r: r.w2_z, "W2_z (mm)", "17e_z_recall_corr.png",
        min_m_constrained=min_m_constrained,
        total_primaries=total_primaries,
    )


def plot_w2_phi_recall_corr_all_m(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    min_m_constrained: int = 6,
    total_primaries: int = 0,
) -> None:
    """17e_phi — W2_phi vs Recall: 4-curve Pearson and Spearman correlation vs M."""
    _plot_17e_w2_variant(
        results, M_values, W_values, output_dir,
        lambda r: r.w2_phi, "W2_φ (rad)", "17e_phi_recall_corr.png",
        min_m_constrained=min_m_constrained,
        total_primaries=total_primaries,
    )


def plot_w2_r_recall_corr_all_m(
    results: list["SetupResult"],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    min_m_constrained: int = 6,
    total_primaries: int = 0,
) -> None:
    """17e_r — W2_r vs Recall: 4-curve Pearson and Spearman correlation vs M."""
    _plot_17e_w2_variant(
        results, M_values, W_values, output_dir,
        lambda r: r.w2_r, "W2_r (mm)", "17e_r_recall_corr.png",
        min_m_constrained=min_m_constrained,
        total_primaries=total_primaries,
    )


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
        ax_scatter.set_title(f"M ≥ {M}", fontsize=12)
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
    fname = f"19_nc_recall_correlation_W{W_fixed}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 19b — NC fraction vs Recall Pearson r summary across W and M
# ──────────────────────────────────────────────────────────────────────
def plot_nc_recall_correlation_summary(
    results: list[SetupResult],
    M_values: list[int],
    W_fixed_values: list[int],
    output_dir: str,
    color_map: dict[str, str] | None = None,
) -> None:
    """Summary: Pearson r(NC fraction, Recall) vs W, one curve per M.

    Draws the 3-sigma Pearson significance threshold so the reader can
    judge which (M, W) combinations show a statistically significant
    NC-Recall relationship.
    Saved as 19b_nc_recall_correlation_summary.png.
    """
    if len(results) < 3:
        print("  [SKIP] 19b: fewer than 3 setups.")
        return

    if color_map is None:
        _pal = _colors(len(M_values))
        color_map_m = {M: _pal[i] for i, M in enumerate(M_values)}
    else:
        _pal = _colors(len(M_values))
        color_map_m = {M: _pal[i] for i, M in enumerate(M_values)}

    W_vals = [W for W in W_fixed_values]
    n = len(results)
    r_crit_3 = _pearson_rcrit(n, sigma=3.0)
    r_crit_5 = _pearson_rcrit(n, sigma=5.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_pearson_rcrit(ax, n, sigma=3.0, draw_label=True)
    _draw_pearson_rcrit(ax, n, sigma=5.0, linestyle="-.", draw_label=True)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")

    for M in M_values:
        nc_arr = np.array([_cc_nc_frac(r, M) for r in results])
        rs = []
        for W in W_vals:
            rec_arr = np.array([_cc_recall(r, M, W) for r in results])
            mask = np.isfinite(nc_arr) & np.isfinite(rec_arr)
            if mask.sum() >= 3 and np.std(nc_arr[mask]) > 0 and np.std(rec_arr[mask]) > 0:
                r_val, _ = scipy_stats.pearsonr(nc_arr[mask], rec_arr[mask])
                rs.append(r_val)
            else:
                rs.append(float("nan"))
        rs = np.array(rs)
        c = color_map_m[M]
        ax.plot(W_vals, rs, marker="o", color=c, label=f"M={M}", linewidth=1.5)

    ax.set_xlabel("W — minimum NC detections per muon", fontsize=13)
    ax.set_ylabel("Pearson r  (NC fraction vs Ge-77 Recall)", fontsize=13)
    ax.set_title(
        "NC Detection Fraction vs Ge-77 Recall — Pearson r\n"
        f"(n={n} setups; dashed = 3σ |r|={r_crit_3:.2f};  dash-dot = 5σ |r|={r_crit_5:.2f})",
        fontsize=14,
    )
    ax.set_xticks(W_vals)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=10, loc="upper right", ncol=max(1, len(M_values) // 5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = "19b_nc_recall_correlation_summary.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
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
    """Ge-77 Recall at W=1 across M thresholds (single panel)."""
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    colors = [color_map.get(r.label, "gray") for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    for r, c in zip(results, colors):
        recalls = [_cc_recall(r, M, 1) for M in M_values]
        ax.plot(M_values, recalls, marker="o", color=c, label=r.label,
                linewidth=1.5, markersize=5)

    all_vals = [_cc_recall(r, M, 1) for r in results for M in M_values
                if np.isfinite(_cc_recall(r, M, 1))]
    if all_vals:
        vmin, vmax = min(all_vals), max(all_vals)
        margin = max((vmax - vmin) * 0.3, 0.005)
        ax.set_ylim(max(0.0, vmin - margin), min(1.0, vmax + margin))

    ax.set_xlabel("M — minimum firing PMTs per NC", fontsize=13)
    ax.set_ylabel("Ge-77 Recall  (W = 1)", fontsize=13)
    ax.set_title("Ge-77 Muon Recall at W=1 across M Thresholds", fontsize=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.set_xticks(M_values)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "20_recall_w1_vs_m.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
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
    fixed_m: int | None = None,
    min_m: int | None = None,
) -> None:
    """Two-panel horizontal bar: Recall (left) and Precision (right) at FoM-optimal (M, W).

    Each bar is annotated with its optimal (M, W) pair and metric value.
    ``fixed_m``: lock M to this value, only optimise W.
    ``min_m``:   restrict search to M >= min_m.
    """
    results = _w2_sorted(results)
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    colors = [color_map.get(r.label, "gray") for r in results]

    if fixed_m is not None:
        m_search = [fixed_m]
    elif min_m is not None:
        m_search = [M for M in M_values if M >= min_m]
    else:
        m_search = M_values

    recalls:    list[float] = []
    precisions: list[float] = []
    opt_mw_labels: list[str] = []
    for r in results:
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        grid = _cc_fom_grid(r, m_search, W_values, _tp)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if valid:
            best_mw = max(valid, key=valid.__getitem__)
            recalls.append(_cc_recall(r, best_mw[0], best_mw[1]))
            precisions.append(_cc_precision(r, best_mw[0], best_mw[1]))
            if fixed_m is not None:
                opt_mw_labels.append(f"W{best_mw[1]}")
            else:
                opt_mw_labels.append(f"M{best_mw[0]}W{best_mw[1]}")
        else:
            recalls.append(float("nan"))
            precisions.append(float("nan"))
            opt_mw_labels.append("N/A")

    y = np.arange(len(results))
    fig, (ax_rec, ax_prec) = plt.subplots(
        1, 2, figsize=(16, max(4, len(results) * 0.55))
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
        ax.set_yticklabels([r.label for r in results], fontsize=11)
        ax.set_xlim(right=x_max * 1.4)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_title(metric, fontsize=13)

        if fixed_m is not None:
            ax.set_xlabel(f"Ge-77 {metric} at FoM-optimal W  (M={fixed_m})", fontsize=13)
        elif min_m is not None:
            ax.set_xlabel(f"Ge-77 {metric} at FoM-optimal (M≥{min_m}, W)", fontsize=13)
        else:
            ax.set_xlabel(f"Ge-77 {metric} at FoM-optimal (M, W)", fontsize=13)

    if fixed_m is not None:
        fig.suptitle(
            f"Ge-77 Recall & Precision at FoM-Optimal W  (M={fixed_m} fixed)\n"
            f"(brackets: W maximising FoM at M={fixed_m})",
            fontsize=14, y=1.01,
        )
        fname = f"21_recall_precision_at_best_fom_for_M{fixed_m}.png"
    elif min_m is not None:
        fig.suptitle(
            f"Ge-77 Recall & Precision at FoM-Optimal (M≥{min_m}, W)\n"
            f"(brackets: (M, W) maximising FoM with M≥{min_m})",
            fontsize=14, y=1.01,
        )
        fname = f"21_recall_precision_at_best_fom_Mge{min_m}.png"
    else:
        fig.suptitle(
            "Ge-77 Recall & Precision at FoM-Optimal (M, W)\n"
            "(brackets: (M, W) maximising FoM for each setup)",
            fontsize=14, y=1.01,
        )
        fname = "21_recall_precision_at_best_fom.png"

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
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
    normalize: bool = False,
    cmap: str = "viridis",
    alpha: float = 0.35,
    x_range: "tuple[float, float] | None" = None,
    y_range: "tuple[float, float] | None" = None,
) -> "matplotlib.cm.ScalarMappable":
    """Draw a FoM(signal_surv, ge_surv) colormap + labelled contours on ax.

    When ``x_range`` / ``y_range`` are given the grid covers exactly those
    bounds (no data-driven extent needed).  Otherwise the grid is built from
    the data in ``xs`` / ``ys`` with a 5 % margin.
    Returns the pcolormesh artist so the caller can attach a colorbar.
    If ``normalize`` is True the FoM values are rescaled to [0, 1].
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
    ZZ = _fom_vec(YY, XX)  # ge_surv=YY, signal_surv=XX
    ZZ = np.where(np.isfinite(ZZ), ZZ, np.nan)
    if normalize:
        finite_z = ZZ[np.isfinite(ZZ)]
        if finite_z.size > 0:
            z_min, z_max = finite_z.min(), finite_z.max()
            span = z_max - z_min if z_max > z_min else 1.0
            ZZ = (ZZ - z_min) / span
    pcm = ax.pcolormesh(XX, YY, ZZ, cmap=cmap, alpha=alpha, shading="auto", zorder=0)
    finite_z = ZZ[np.isfinite(ZZ)]
    if finite_z.size > 0:
        n_levels = 8
        levels = np.linspace(finite_z.min(), finite_z.max(), n_levels + 2)[1:-1]
        cs = ax.contour(XX, YY, ZZ, levels=levels, colors="gray",
                        linewidths=0.6, alpha=0.8, zorder=1)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
    return pcm


def _parse_w_cut_csv(path: str) -> list[dict]:
    """Parse CSV with format per line:
    ``x cut <W>, ge_77_surv: <val> +-<unc>, sig_surv: <val> +-<unc>``

    Returns list of dicts with keys:
        ``x_cut``, ``ge_77_surv``, ``ge_77_surv_unc``, ``sig_surv``, ``sig_surv_unc``.

    Rows with sig_surv < 0 are skipped: they are unphysical because the dead-time
    exceeds the measurement time (veto_fraction × VETO_DURATION_H × MUSUN_RATE > 1).
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                x_cut = float(parts[0].split()[-1])

                ge_part = parts[1].split(":")[-1].strip()
                if "+-" in ge_part:
                    ge_val, ge_unc = ge_part.split("+-", 1)
                else:
                    ge_val, ge_unc = ge_part, "0"
                ge_77_surv     = float(ge_val.strip())
                ge_77_surv_unc = abs(float(ge_unc.strip()))

                sig_part = parts[2].split(":")[-1].strip()
                if "+-" in sig_part:
                    sig_val, sig_unc = sig_part.split("+-", 1)
                else:
                    sig_val, sig_unc = sig_part, "0"
                sig_surv     = float(sig_val.strip())
                sig_surv_unc = abs(float(sig_unc.strip()))
            except (ValueError, IndexError):
                continue

            if not np.isfinite(sig_surv) or sig_surv < 0.0:
                continue  # unphysical: dead-time > measurement time

            rows.append({
                "x_cut":        x_cut,
                "ge_77_surv":   ge_77_surv,
                "ge_77_surv_unc": ge_77_surv_unc,
                "sig_surv":     sig_surv,
                "sig_surv_unc": sig_surv_unc,
            })
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
    stat_limit_rows: "list[dict] | None" = None,
    stat_limit_color: str = "tab:blue",
) -> None:
    """Scatter: Ge77-NC-weighted ge_surv (y) vs 1 − deadtime (x) per setup.

    Each point corresponds to one (M, W) combination for one setup.
    Lower ge_surv = fewer Ge77 isotopes survive (better background rejection).
    Higher 1 − deadtime = more signal livetime (better).
    Good operating points sit in the bottom-right corner.

    If ``stat_limit_rows`` is given, the shared NC-truth stat limit is overlaid
    as 'Full Captures - stat. limit'.
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

    sl_filt: list[dict] = []
    if stat_limit_rows:
        sl_filt = [r for r in stat_limit_rows if np.isfinite(r["sig_surv"])]
        all_xs.extend(r["sig_surv"]   for r in sl_filt)
        all_ys.extend(r["ge_77_surv"] for r in sl_filt)

    pcm = _fom_colormap_background(ax, all_xs, all_ys)
    if pcm is not None:
        fig.colorbar(pcm, ax=ax, label="FoM", pad=0.01)

    for r, c, (xs, ys) in zip(results, colors, per_setup):
        if xs:
            ax.scatter(xs, ys, color=c, s=18, alpha=0.7, zorder=3)

    if sl_filt:
        xs_sl = np.array([r["sig_surv"]       for r in sl_filt])
        ys_sl = np.array([r["ge_77_surv"]     for r in sl_filt])
        si_sl = np.array([r["ge_77_surv_unc"] for r in sl_filt])
        co_sl = np.array([_combined_unc_25b(s, v) for s, v in zip(si_sl, ys_sl)])
        _plot_curve_with_bands(ax, xs_sl, ys_sl, si_sl, co_sl,
                               color=stat_limit_color,
                               label="Full Captures - stat. limit",
                               linestyle=":", linewidth=1.5, zorder=4,
                               alpha_inner=0.15, alpha_outer=0.07)

    ax.set_xlabel("1 − Deadtime  (signal livetime fraction)", fontsize=13)
    ax.set_ylabel("Ge77 survival  (Σ FN Ge77 NCs / Σ all Ge77 NCs)", fontsize=13)
    ax.set_title(
        "Ge77 Survival vs Signal Livetime Trade-off\n"
        "(each point = one (M, W) combination; bottom-right = optimal)",
        fontsize=14,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map.get(r.label, "gray"),
                   markersize=8, label=r.label)
        for r in results
    ]
    if sl_filt:
        legend_handles.append(
            plt.Line2D([0], [0], color=stat_limit_color, linestyle=":",
                       linewidth=1.5, label="Full Captures - stat. limit")
        )
    ax.legend(handles=legend_handles, fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    # Lock axis limits to heatmap extent so no blank background areas appear.
    if all_xs and all_ys:
        _ax_dx = max((max(all_xs) - min(all_xs)) * 0.05, 1e-4)
        _ax_dy = max((max(all_ys) - min(all_ys)) * 0.05, 1e-4)
        ax.set_xlim(min(all_xs) - _ax_dx, max(all_xs) + _ax_dx)
        ax.set_ylim(min(all_ys) - _ax_dy, max(all_ys) + _ax_dy)
    fig.tight_layout()
    fname = "25_ge_surv_vs_livetime.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 25b — ge_surv vs livetime: advisor's data + user setups at M=6
# (with two-level uncertainty bands: statistical and stat ⊕ systematic)
# ──────────────────────────────────────────────────────────────────────

# 35% relative systematic uncertainty (Virtual Depth paper, arXiv:1802.05040)
_SYS_REL_25B: float = 0.35
# Dead-time scale: VETO_DURATION_H × MUSUN_RATE
_DT_SCALE: float = VETO_DURATION_H * MUSUN_RATE


def _combined_unc_25b(stat: float, value: float) -> float:
    """Combined statistical + systematic uncertainty (35% relative systematic)."""
    return float(np.sqrt(stat**2 + (_SYS_REL_25B * abs(value))**2))


def _plot_curve_with_bands(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    stat_dy: np.ndarray,
    comb_dy: np.ndarray,
    color: str,
    label: str,
    linestyle: str = "-",
    linewidth: float = 1.5,
    zorder: int = 3,
    alpha_inner: float = 0.25,
    alpha_outer: float = 0.12,
    point_labels: "np.ndarray | None" = None,
) -> None:
    """Plot a 2D parametric curve with nested uncertainty bands.

    Points are sorted by x (sig_surv) so fill_between renders cleanly.
    Only the main line gets a legend entry; band fills are unlabelled.
    Inner band = statistical; outer band = stat ⊕ systematic.

    ``point_labels`` — optional array of W-value labels (same order as xs
    before sorting).  When provided, discrete markers are drawn at every
    point and the W-value is annotated at the first and last points.
    """
    if len(xs) == 0:
        return
    order = np.argsort(xs)
    xs_s  = xs[order];       ys_s  = ys[order]
    si_dy = stat_dy[order];  co_dy = comb_dy[order]

    ax.fill_between(xs_s, ys_s - co_dy, ys_s + co_dy,
                    color=color, alpha=alpha_outer, linewidth=0, zorder=zorder - 1)
    ax.fill_between(xs_s, ys_s - si_dy, ys_s + si_dy,
                    color=color, alpha=alpha_inner, linewidth=0, zorder=zorder - 1)
    ax.plot(xs_s, ys_s, color=color, linestyle=linestyle,
            linewidth=linewidth, zorder=zorder, label=label)

    if point_labels is not None and len(point_labels) > 0:
        pl_s = np.asarray(point_labels)[order]
        # Distinct marker at every data point
        ax.scatter(xs_s, ys_s, color=color, s=22, zorder=zorder + 1,
                   marker="o", linewidths=0)
        # Annotate first point (smallest x = highest W)
        ax.annotate(
            f"W = {int(pl_s[0])}",
            xy=(xs_s[0], ys_s[0]),
            xytext=(-5, 0), textcoords="offset points",
            color=color, fontsize=8, ha="right", va="center",
            fontweight="bold",
        )
        # Annotate last point (largest x = lowest W)
        if len(xs_s) > 1:
            ax.annotate(
                f"W = {int(pl_s[-1])}",
                xy=(xs_s[-1], ys_s[-1]),
                xytext=(5, 0), textcoords="offset points",
                color=color, fontsize=8, ha="left", va="center",
                fontweight="bold",
            )


def _rows_to_band_arrays(rows: list[dict]) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Extract (xs, ys, stat_dy, comb_dy) arrays from a list of curve dicts."""
    xs     = np.array([r["sig_surv"]       for r in rows])
    ys     = np.array([r["ge_77_surv"]     for r in rows])
    si_dy  = np.array([r["ge_77_surv_unc"] for r in rows])
    co_dy  = np.array([_combined_unc_25b(s, v) for s, v in zip(si_dy, ys)])
    return xs, ys, si_dy, co_dy


def _compute_setup_curve_25b(
    r: SetupResult,
    W_values: list[int],
    M_fixed: int,
    total_primaries: int,
) -> list[dict]:
    """Compute ge_77_surv and sig_surv with Poisson uncertainties for one setup."""
    _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
    rows: list[dict] = []
    for W in W_values:
        cm = r.muon["confusion"].get((M_fixed, W))
        if cm is None:
            continue
        TN    = _tp - cm["TP"] - cm["FP"] - cm["FN"]
        tp_gc = cm.get("tp_ge77_nc_counts", np.ones(cm["TP"], dtype=np.int32))
        fn_gc = cm.get("fn_ge77_nc_counts", np.ones(cm["FN"], dtype=np.int32))

        ge_surv  = calc_ge_survival_confusion(tp_gc, fn_gc)
        deadtime = calc_deadtime_confusion(cm["TP"], cm["FP"], TN, cm["FN"])
        sig_surv = 1.0 - deadtime

        if not np.isfinite(sig_surv) or sig_surv < 0.0:
            continue

        fn_sum         = float(np.sum(fn_gc))
        total_ge77_nc  = float(np.sum(tp_gc)) + fn_sum
        n_vetoed       = cm["TP"] + cm["FP"]

        ge_unc  = np.sqrt(max(fn_sum,    0.0)) / max(total_ge77_nc, 1.0)
        sig_unc = _DT_SCALE * np.sqrt(max(n_vetoed, 0)) / max(_tp, 1)

        rows.append({
            "x_cut":          float(W),
            "ge_77_surv":     ge_surv,
            "ge_77_surv_unc": ge_unc,
            "sig_surv":       sig_surv,
            "sig_surv_unc":   sig_unc,
        })
    return rows


# ──────────────────────────────────────────────────────────────────────
# In-water box filter (for advisor plot 25b comparison)
# ──────────────────────────────────────────────────────────────────────
_WATER_BOX_CRYO_R_M  = 3.200   # cryostat inner radius (m)
_WATER_BOX_Z_LO_M    = -2.280  # cryostat lower straight boundary (m)
_WATER_BOX_Z_HI_M    =  1.720  # cryostat upper straight boundary (m)
_WATER_BOX_TANK_R_M  =  6.000  # water tank radius (m)
_WATER_BOX_MATERIALS: frozenset[str] = frozenset(
    {"Water", "metal_steel", "tyvek", "my_tyvek"}
)


def _in_water_box_mask(nc_truth: pd.DataFrame) -> np.ndarray:
    """Boolean mask for NCs inside the water tank but outside the cryostat box.

    Mirrors the advisor's ``in_water`` spatial + material selection:
      - r < water_tank_radius (inside tank)
      - r > cryo_inner_radius  OR  z outside cryostat straight part (outside cryo box)
      - nc_material_name in {Water, metal_steel, tyvek, my_tyvek}

    If nc_material_name is absent, only the spatial filter is applied.
    """
    r_xy = np.sqrt(nc_truth["nc_x"].values**2 + nc_truth["nc_y"].values**2)
    z    = nc_truth["nc_z"].values
    outside_cryo = (
        (r_xy >= _WATER_BOX_CRYO_R_M) |
        (z    <  _WATER_BOX_Z_LO_M)   |
        (z    >  _WATER_BOX_Z_HI_M)
    )
    spatial = (r_xy < _WATER_BOX_TANK_R_M) & outside_cryo
    if "nc_material_name" in nc_truth.columns:
        mat_ok = nc_truth["nc_material_name"].isin(_WATER_BOX_MATERIALS).values
        return spatial & mat_ok
    return spatial


def _compute_nc_truth_stat_limit_in_water(
    nc_truth_full: pd.DataFrame,
    W_values: list[int],
    total_primaries: int,
) -> list[dict]:
    """NC-truth stat limit counting only in-water NCs (for advisor plot 25b).

    Applies _in_water_box_mask so the W-threshold counting matches the
    advisor's spatial + material filter.  Ge77 muon classification uses the
    FULL nc_truth so that Ge77 muons whose NCs are all outside the water
    region are still treated as Ge77 (FN when not tagged).

    Parameters
    ----------
    nc_truth_full : full NC truth DataFrame (including nc_x, nc_y, nc_z cols).
    W_values      : W thresholds to sweep (use W up to 50).
    total_primaries : total simulated muons for TN denominator.
    """
    # Step 1: Ge77 muon classification from FULL nc_truth
    muon_is_ge77 = (
        nc_truth_full.groupby(["run_id", "muon_id"])["flag_ge77"]
        .apply(lambda flags: bool((flags == 1).any()))
    )
    unique_muon_idx = muon_is_ge77.index
    ge77_truth      = muon_is_ge77.values.astype(bool)

    # Step 2: Apply in-water filter
    mask        = _in_water_box_mask(nc_truth_full)
    nc_filtered = nc_truth_full[mask]

    if nc_filtered.empty:
        print("  [WARN] _compute_nc_truth_stat_limit_in_water: no in-water NCs found.")
        return []

    # Step 3: In-window in-water NC counts per muon
    in_window = (
        (nc_filtered["nc_time_ns"].values >= MUON_WINDOW_LO_NS) &
        (nc_filtered["nc_time_ns"].values <= MUON_WINDOW_HI_NS)
    )
    nc_w_df = pd.DataFrame({
        "run_id":    nc_filtered["run_id"].values,
        "muon_id":   nc_filtered["muon_id"].values,
        "in_window": in_window.astype(np.int8),
    })
    w_per_muon = nc_w_df.groupby(["run_id", "muon_id"])["in_window"].sum()
    w_counts   = w_per_muon.reindex(unique_muon_idx, fill_value=0).values.astype(np.int32)

    # Step 4: Per-muon in-water Ge77 NC counts (no time cut) for ge_surv
    ge77_nc_per_muon: np.ndarray = (
        nc_filtered.groupby(["run_id", "muon_id"])["flag_ge77"]
        .sum()
        .reindex(unique_muon_idx, fill_value=0)
        .values.astype(np.int32)
    )
    total_ge77_nc = float(ge77_nc_per_muon[ge77_truth].sum())
    if total_ge77_nc == 0:
        print("  [WARN] _compute_nc_truth_stat_limit_in_water: no in-water Ge77 NCs.")
        return []

    _tp_denom = max(total_primaries, 1)
    rows: list[dict] = []
    for W in W_values:
        classified_ge77 = w_counts >= W
        tp_mask = ge77_truth  &  classified_ge77
        fn_mask = ge77_truth  & ~classified_ge77
        tp = int(tp_mask.sum())
        fp = int((~ge77_truth &  classified_ge77).sum())
        fn = int(fn_mask.sum())
        tn = total_primaries - tp - fp - fn

        deadtime = calc_deadtime_confusion(tp, fp, tn, fn)
        sig_surv = 1.0 - deadtime
        if not np.isfinite(sig_surv) or sig_surv < 0.0:
            continue

        fn_gc_sum = float(ge77_nc_per_muon[fn_mask].sum())
        ge_surv   = fn_gc_sum / total_ge77_nc

        ge_unc   = np.sqrt(max(fn_gc_sum, 0.0)) / total_ge77_nc
        n_vetoed = tp + fp
        sig_unc  = _DT_SCALE * np.sqrt(max(n_vetoed, 0)) / max(_tp_denom, 1)

        rows.append({
            "x_cut":          float(W),
            "ge_77_surv":     ge_surv,
            "ge_77_surv_unc": ge_unc,
            "sig_surv":       sig_surv,
            "sig_surv_unc":   sig_unc,
        })
    return rows


def _compute_nc_truth_stat_limit(
    nc_truth: pd.DataFrame,
    W_values: list[int],
    total_primaries: int,
) -> list[dict]:
    """Statistical limit from NC truth: perfect detection, no optical threshold.

    For each muon, counts NCs in [1µs, 200µs] directly from NC truth (no M
    threshold, no optical-simulation timing cut).  This is the theoretical
    upper bound achievable by any detector configuration sharing the same NC
    truth.  Both Ge77 and non-Ge77 muons use NC truth counts.

    Parameters
    ----------
    nc_truth : DataFrame with columns run_id, muon_id, flag_ge77, nc_time_ns.
    W_values : W thresholds to sweep.
    total_primaries : total simulated primary muons (for correct TN denominator).

    Returns
    -------
    List of row dicts: x_cut, ge_77_surv, ge_77_surv_unc, sig_surv, sig_surv_unc.
    """
    # Muon ground truth using composite (run_id, muon_id) key
    muon_is_ge77 = (
        nc_truth.groupby(["run_id", "muon_id"])["flag_ge77"]
        .apply(lambda flags: bool((flags == 1).any()))
    )
    unique_muon_idx = muon_is_ge77.index   # MultiIndex (run_id, muon_id)
    ge77_truth      = muon_is_ge77.values.astype(bool)

    # In-window NC counts per muon — truth level, no M cut
    in_window = (
        (nc_truth["nc_time_ns"].values >= MUON_WINDOW_LO_NS)
        & (nc_truth["nc_time_ns"].values <= MUON_WINDOW_HI_NS)
    )
    nc_w = pd.DataFrame({
        "run_id":    nc_truth["run_id"].values,
        "muon_id":   nc_truth["muon_id"].values,
        "in_window": in_window.astype(np.int8),
    })
    w_per_muon = nc_w.groupby(["run_id", "muon_id"])["in_window"].sum()
    w_counts   = w_per_muon.reindex(unique_muon_idx, fill_value=0).values.astype(np.int32)

    # Per-muon Ge77 NC counts for ge_surv weighting (no time-window cut)
    ge77_nc_per_muon: np.ndarray = (
        nc_truth.groupby(["run_id", "muon_id"])["flag_ge77"]
        .sum()
        .reindex(unique_muon_idx, fill_value=0)
        .values.astype(np.int32)
    )
    total_ge77_nc = float(ge77_nc_per_muon[ge77_truth].sum())

    _tp_denom = max(total_primaries, 1)
    rows: list[dict] = []

    for W in W_values:
        classified_ge77 = w_counts >= W
        tp_mask = ge77_truth  &  classified_ge77
        fn_mask = ge77_truth  & ~classified_ge77
        tp = int(tp_mask.sum())
        fp = int((~ge77_truth &  classified_ge77).sum())
        fn = int(fn_mask.sum())
        tn = total_primaries - tp - fp - fn   # correct: all simulated muons

        deadtime = calc_deadtime_confusion(tp, fp, tn, fn)
        sig_surv = 1.0 - deadtime

        if not np.isfinite(sig_surv) or sig_surv < 0.0:
            continue

        fn_gc_sum = float(ge77_nc_per_muon[fn_mask].sum())
        ge_surv   = fn_gc_sum / max(total_ge77_nc, 1.0)

        ge_unc   = np.sqrt(max(fn_gc_sum, 0.0)) / max(total_ge77_nc, 1.0)
        n_vetoed = tp + fp
        sig_unc  = _DT_SCALE * np.sqrt(max(n_vetoed, 0)) / max(_tp_denom, 1)

        rows.append({
            "x_cut":          float(W),
            "ge_77_surv":     ge_surv,
            "ge_77_surv_unc": ge_unc,
            "sig_surv":       sig_surv,
            "sig_surv_unc":   sig_unc,
        })
    return rows


def _find_baseline_result(results: list[SetupResult]) -> SetupResult:
    """Find the baseline setup by label (case-insensitive 'baseline' check).

    Search order:
      1. First setup has 'baseline' in label → use it (expected case).
      2. Any other setup has 'baseline' in label → use it with an info message.
      3. No 'baseline' label found → use first setup with a clear warning.
    """
    if "baseline" in results[0].label.lower():
        return results[0]
    for r in results[1:]:
        if "baseline" in r.label.lower():
            print(
                f"  [INFO] 25b_baseline: baseline setup found as '{r.label}' "
                f"(not the first setup)."
            )
            return r
    print(
        f"  [WARN] 25b_baseline: no setup with 'baseline' in its label was found. "
        f"Using first setup '{results[0].label}' as baseline. "
        f"Consider naming the baseline setup with 'Baseline' for unambiguous detection."
    )
    return results[0]


def _draw_advisor_plot(
    ax: plt.Axes,
    results_to_show: list[SetupResult],
    W_range: list[int],
    M_fixed: int,
    total_primaries: int,
    advisor_rows: list[dict],
    stat_limit_rows: list[dict],
    color_map: dict[str, str],
    sig_surv_min: float = 0.0,
    heatmap_cmap: str = "YlOrBr",
    heatmap_alpha: float = 0.30,
    label_overrides: "dict[str, str] | None" = None,
    nc_stat_limit_override: "list[dict] | None" = None,
) -> "matplotlib.cm.ScalarMappable | None":
    """Draw all advisor-comparison curves onto *ax*.

    Four curve types, each with inner (stat) and outer (stat⊕sys) bands:
      1. L1000 CDR Baseline (advisor CSV data)
      2. L1000 CDR Baseline — stat. limit (from stat-limit CSV, ``stat_limit_rows`` param)
      3. Each setup in results_to_show at M_fixed
      4. Single shared NC-truth stat limit ('Full Captures - stat. limit')

    Note: ``stat_limit_rows`` (parameter) = advisor CSV stat limit (curve 2).
          ``r.stat_limit_rows`` (per-setup field) = NC-truth stat limit (curve 4).
    ``nc_stat_limit_override`` — when provided, replaces r.stat_limit_rows as the
          source for curve 4 (used by the advisor plot to apply the in-water box
          filter so both stat limits are computed over the same NC population).

    ``sig_surv_min`` restricts all curves to signal survival >= sig_surv_min.
    ``label_overrides`` maps r.label.lower() → display label for user setups.
    Returns the pcolormesh artist for the FoM background colorbar.
    """
    W_lo = float(W_range[0])  if W_range else -np.inf
    W_hi = float(W_range[-1]) if W_range else  np.inf

    def _collect_filtered(rows: list[dict]) -> list[dict]:
        return [r for r in rows
                if W_lo <= r["x_cut"] <= W_hi and r["sig_surv"] >= sig_surv_min]

    def _collect_stat_limit(rows: list[dict]) -> list[dict]:
        """Filter stat-limit rows by sig_surv only — no W upper cap."""
        return [r for r in rows if r["sig_surv"] >= sig_surv_min]

    def _w_labels_sorted(rows: list[dict]) -> np.ndarray:
        """Return x_cut (W) values sorted in the same order as _plot_curve_with_bands."""
        xs_tmp = np.array([r["sig_surv"] for r in rows])
        order  = np.argsort(xs_tmp)
        return np.array([rows[i]["x_cut"] for i in order])

    # Pre-compute all filtered rows so we can build the FoM heatmap extent first
    adv_filt = _collect_filtered(advisor_rows)
    sl_filt  = _collect_filtered(stat_limit_rows)

    all_setup_rows: list[list[dict]] = []
    for r in results_to_show:
        sr = _collect_filtered(
            _compute_setup_curve_25b(r, W_range, M_fixed, total_primaries))
        all_setup_rows.append(sr)

    # Shared NC-truth stat limit: use in-water override when given (advisor plot),
    # otherwise fall back to the pre-computed full stat limit from results.
    if nc_stat_limit_override is not None:
        _shared_sl_rows_raw = nc_stat_limit_override
    else:
        _shared_sl_rows_raw = next(
            (r.stat_limit_rows for r in results_to_show if r.stat_limit_rows), None
        )
    shared_sl_filt = _collect_stat_limit(_shared_sl_rows_raw) if _shared_sl_rows_raw else []

    # Baseline color for the shared stat limit
    _bl_result = next(
        (r for r in results_to_show if "baseline" in r.label.lower()),
        results_to_show[0] if results_to_show else None,
    )
    _stat_limit_color = color_map.get(_bl_result.label, "gray") if _bl_result else "gray"

    # ── FoM background heatmap (normalised 0–1, full axes range) ─────
    pcm = _fom_colormap_background(
        ax, [], [],
        normalize=True, cmap=heatmap_cmap, alpha=heatmap_alpha,
        x_range=(0.80, 1.0), y_range=(0.0, 1.0),
    )

    # ── 1. L1000 CDR Baseline ─────────────────────────────────────────
    if adv_filt:
        xs, ys, si, co = _rows_to_band_arrays(adv_filt)
        _plot_curve_with_bands(ax, xs, ys, si, co,
                               color="black", label="L1000 CDR Baseline",
                               linestyle="-", linewidth=2.0, zorder=5,
                               point_labels=_w_labels_sorted(adv_filt))

    # ── 2. L1000 CDR Baseline — stat. limit ───────────────────────────
    if sl_filt:
        xs, ys, si, co = _rows_to_band_arrays(sl_filt)
        _plot_curve_with_bands(ax, xs, ys, si, co,
                               color="dimgray", label="L1000 CDR Baseline — stat. limit",
                               linestyle="--", linewidth=1.8, zorder=4,
                               point_labels=_w_labels_sorted(sl_filt))

    # ── 3. User setup curves ──────────────────────────────────────────
    _overrides = label_overrides or {}
    for r, sr in zip(results_to_show, all_setup_rows):
        c = color_map.get(r.label, "gray")
        disp_label = _overrides.get(r.label.lower(), r.label)

        if sr:
            xs, ys, si, co = _rows_to_band_arrays(sr)
            _plot_curve_with_bands(ax, xs, ys, si, co,
                                   color=c, label=disp_label,
                                   linestyle="-", linewidth=1.5, zorder=3,
                                   point_labels=_w_labels_sorted(sr))

    # ── 4. Single shared NC-truth stat limit (W up to 50) ────────────
    if shared_sl_filt:
        xs, ys, si, co = _rows_to_band_arrays(shared_sl_filt)
        _plot_curve_with_bands(ax, xs, ys, si, co,
                               color=_stat_limit_color,
                               label="Full Captures - stat. limit",
                               linestyle=":", linewidth=1.2, zorder=3,
                               alpha_inner=0.15, alpha_outer=0.07,
                               point_labels=_w_labels_sorted(shared_sl_filt))

    # ── Legend: add reference patches for band explanation ────────────
    from matplotlib.patches import Patch as _Patch
    handles, _ = ax.get_legend_handles_labels()
    handles += [
        _Patch(facecolor="gray", alpha=0.25, edgecolor="none",
               label="Statistical uncertainty"),
        _Patch(facecolor="gray", alpha=0.12, edgecolor="none",
               label="Stat. ⊕ 35 % systematic"),
    ]
    ax.legend(handles=handles, fontsize=10, loc="upper left")
    return pcm


def plot_ge_surv_vs_livetime_advisor(
    results: list[SetupResult],
    W_values: list[int],
    output_dir: str,
    advisor_csv: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
    M_fixed: int = 6,
    statistical_limit_csv: str | None = None,
    baseline_display_label: str | None = None,
    nc_stat_limit_override: "list[dict] | None" = None,
) -> None:
    """Plot 25b: Ge-77 survival vs signal livetime at M_fixed, all setups.

    Overlays L1000 CDR Baseline, its stat. limit (from stat-limit CSV),
    all user setups, and a single shared NC-truth stat limit.
    Every curve carries an inner (statistical) and outer (stat ⊕ 35 % systematic)
    uncertainty band. All W values are included.

    ``baseline_display_label`` overrides the display name of the setup identified
    as the baseline (label containing 'baseline', case-insensitive).
    ``nc_stat_limit_override`` — when set, replaces the per-setup stat limit with
    this pre-computed in-water box filtered stat limit for a fair comparison with
    the advisor's stat limit.
    """
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}

    advisor_rows    = _parse_w_cut_csv(advisor_csv)
    stat_limit_rows = _parse_w_cut_csv(statistical_limit_csv) if statistical_limit_csv else []

    W_range = W_values

    _label_overrides: dict[str, str] | None = None
    if baseline_display_label:
        _bl = _find_baseline_result(results)
        _label_overrides = {_bl.label.lower(): baseline_display_label}

    fig, ax = plt.subplots(figsize=(12, 9))
    pcm = _draw_advisor_plot(
        ax, results, W_range, M_fixed, total_primaries,
        advisor_rows, stat_limit_rows, color_map,
        sig_surv_min=0.80,
        label_overrides=_label_overrides,
        nc_stat_limit_override=nc_stat_limit_override,
    )
    if pcm is not None:
        fig.colorbar(pcm, ax=ax, label="FoM (normalised 0–1)", pad=0.01)

    ax.set_xlabel("Signal survival  (1 − deadtime)", fontsize=13)
    ax.set_ylabel("Ge-77 survival  (Σ FN NCs / Σ all Ge-77 NCs)", fontsize=13)
    ax.set_title(
        f"Ge-77 Survival vs Signal Livetime  [M = {M_fixed}]\n"
        f"M = min. firing PMTs per NC  ·  W = min. detected NCs per muon to tag as Ge-77\n"
        f"(inner band: stat.  outer band: stat. ⊕ 35 % syst.  ·  signal survival ≥ 80 %)",
        fontsize=12,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.set_xlim(0.80, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "25b_ge_surv_vs_livetime_advisor.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_ge_surv_vs_livetime_advisor_baseline(
    results: list[SetupResult],
    W_values: list[int],
    output_dir: str,
    advisor_csv: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
    M_fixed: int = 6,
    statistical_limit_csv: str | None = None,
    baseline_display_label: str | None = None,
    nc_stat_limit_override: "list[dict] | None" = None,
) -> None:
    """Plot 25b_baseline: same as plot 25b but shows only the baseline setup.

    The baseline setup is identified by a case-insensitive 'baseline' match
    in the setup label.  Falls back to the first setup with a warning if none
    is found.

    ``baseline_display_label`` overrides the baseline's legend label.
    ``nc_stat_limit_override`` — in-water box filtered stat limit for advisor
    plot comparison (see plot_ge_surv_vs_livetime_advisor).
    """
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}

    advisor_rows    = _parse_w_cut_csv(advisor_csv)
    stat_limit_rows = _parse_w_cut_csv(statistical_limit_csv) if statistical_limit_csv else []

    W_range = W_values

    baseline = _find_baseline_result(results)
    _bl_disp = baseline_display_label if baseline_display_label else baseline.label
    _label_overrides: dict[str, str] | None = None
    if baseline_display_label:
        _label_overrides = {baseline.label.lower(): baseline_display_label}

    fig, ax = plt.subplots(figsize=(12, 9))
    pcm = _draw_advisor_plot(
        ax, [baseline], W_range, M_fixed, total_primaries,
        advisor_rows, stat_limit_rows, color_map,
        sig_surv_min=0.80,
        label_overrides=_label_overrides,
        nc_stat_limit_override=nc_stat_limit_override,
    )
    if pcm is not None:
        fig.colorbar(pcm, ax=ax, label="FoM (normalised 0–1)", pad=0.01)

    ax.set_xlabel("Signal survival  (1 − deadtime)", fontsize=13)
    ax.set_ylabel("Ge-77 survival  (Σ FN NCs / Σ all Ge-77 NCs)", fontsize=13)
    ax.set_title(
        f"Ge-77 Survival vs Signal Livetime  [M = {M_fixed}]  — {_bl_disp} only\n"
        f"M = min. firing PMTs per NC  ·  W = min. detected NCs per muon to tag as Ge-77\n"
        f"(inner band: stat.  outer band: stat. ⊕ 35 % syst.  ·  signal survival ≥ 80 %)",
        fontsize=12,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.set_xlim(0.80, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "25b_baseline_ge_surv_vs_livetime_advisor.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 25 — NC-truth baseline: setup curve + NC-truth stat limit only
# ──────────────────────────────────────────────────────────────────────
def plot_ge_surv_vs_livetime_nc_truth_baseline(
    results: list[SetupResult],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
    M_fixed: int = 6,
    baseline_display_label: str | None = None,
) -> None:
    """Plot 25: baseline setup curve + its NC-truth statistical limit.

    Exactly two curves (no advisor data):
      1. Baseline setup at M_fixed (optical-simulation result)
      2. NC-truth stat limit for the baseline setup (pre-computed,
         stored in SetupResult.stat_limit_rows)

    The baseline is identified by a case-insensitive 'baseline' match in the
    setup label; falls back to the first setup with a warning.

    ``baseline_display_label`` overrides the baseline's legend label if given.
    """
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}

    baseline = _find_baseline_result(results)
    c = color_map.get(baseline.label, "tab:blue")

    _tp = total_primaries if total_primaries > 0 else baseline.muon["muon_stats"]["total"]

    setup_rows = _compute_setup_curve_25b(baseline, W_values, M_fixed, _tp)
    sl_rows    = baseline.stat_limit_rows or []

    def _collect_setup(rows: list[dict]) -> list[dict]:
        return [r for r in rows if r["sig_surv"] >= 0.80]

    def _collect_sl(rows: list[dict]) -> list[dict]:
        """Stat-limit filter: sig_surv only — no W upper cap."""
        return [r for r in rows if r["sig_surv"] >= 0.80]

    setup_filt = _collect_setup(setup_rows)
    sl_filt    = _collect_sl(sl_rows)

    disp_label = baseline_display_label if baseline_display_label else baseline.label

    fig, ax = plt.subplots(figsize=(10, 8))

    # FoM background heatmap — always covers full axes range
    pcm = _fom_colormap_background(
        ax, [], [], normalize=True, cmap="YlOrBr", alpha=0.30,
        x_range=(0.80, 1.0), y_range=(0.0, 1.0),
    )
    if pcm is not None:
        fig.colorbar(pcm, ax=ax, label="FoM (normalised 0–1)", pad=0.01)

    def _w_labels(rows: list[dict]) -> np.ndarray:
        xs_tmp = np.array([r["sig_surv"] for r in rows])
        order  = np.argsort(xs_tmp)
        return np.array([rows[i]["x_cut"] for i in order])

    if setup_filt:
        xs, ys, si, co = _rows_to_band_arrays(setup_filt)
        _plot_curve_with_bands(ax, xs, ys, si, co,
                               color=c, label=disp_label,
                               linestyle="-", linewidth=2.0, zorder=4,
                               point_labels=_w_labels(setup_filt))

    if sl_filt:
        xs, ys, si, co = _rows_to_band_arrays(sl_filt)
        _plot_curve_with_bands(ax, xs, ys, si, co,
                               color=c, label="Full Captures - stat. limit",
                               linestyle=":", linewidth=1.5, zorder=3,
                               alpha_inner=0.15, alpha_outer=0.07,
                               point_labels=_w_labels(sl_filt))

    from matplotlib.patches import Patch as _Patch
    handles, _ = ax.get_legend_handles_labels()
    handles += [
        _Patch(facecolor="gray", alpha=0.25, edgecolor="none",
               label="Statistical uncertainty"),
        _Patch(facecolor="gray", alpha=0.12, edgecolor="none",
               label="Stat. ⊕ 35 % systematic"),
    ]
    ax.legend(handles=handles, fontsize=10, loc="upper left")

    ax.set_xlabel("Signal survival  (1 − deadtime)", fontsize=13)
    ax.set_ylabel("Ge-77 survival  (Σ FN NCs / Σ all Ge-77 NCs)", fontsize=13)
    ax.set_title(
        f"Ge-77 Survival vs Signal Livetime  [M = {M_fixed}]  — {disp_label} + NC-truth limit\n"
        f"M = min. firing PMTs per NC  ·  W = min. detected NCs per muon to tag as Ge-77\n"
        f"(inner band: stat.  outer band: stat. ⊕ 35 % syst.  ·  signal survival ≥ 80 %)",
        fontsize=12,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.set_xlim(0.80, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "25_ge_surv_vs_livetime_nc_truth.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 25c — all user setups + shared stat limit, no advisor data
# ──────────────────────────────────────────────────────────────────────
def plot_ge_surv_setups_only(
    results: list[SetupResult],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
    M_fixed: int = 6,
    baseline_display_label: str | None = None,
) -> None:
    """Plot 25c: all user setups + single shared stat limit, no advisor data.

    x: 80–100% (signal survival), y: 0–100% (Ge-77 survival).
    FoM colormap covers the full axes range.
    All setups are shown; the shared NC-truth stat limit is drawn once as
    'Full Captures - stat. limit' using the baseline setup's colour.
    """
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}

    W_range = W_values

    _label_overrides: dict[str, str] | None = None
    if baseline_display_label:
        _bl = _find_baseline_result(results)
        _label_overrides = {_bl.label.lower(): baseline_display_label}

    # Shared stat limit (same for all setups — use first available)
    _shared_sl_raw = next(
        (r.stat_limit_rows for r in results if r.stat_limit_rows), None
    )
    shared_sl_filt = (
        [r for r in _shared_sl_raw if r["sig_surv"] >= 0.80]
        if _shared_sl_raw else []
    )

    # Baseline color for the stat limit line
    _bl_r = next(
        (r for r in results if "baseline" in r.label.lower()),
        results[0] if results else None,
    )
    _sl_color = color_map.get(_bl_r.label, "gray") if _bl_r else "gray"

    def _w_labels_sorted(rows: list[dict]) -> np.ndarray:
        xs_tmp = np.array([r["sig_surv"] for r in rows])
        order  = np.argsort(xs_tmp)
        return np.array([rows[i]["x_cut"] for i in order])

    fig, ax = plt.subplots(figsize=(12, 9))

    # FoM background — full axes
    pcm = _fom_colormap_background(
        ax, [], [], normalize=True, cmap="YlOrBr", alpha=0.30,
        x_range=(0.80, 1.0), y_range=(0.0, 1.0),
    )
    if pcm is not None:
        fig.colorbar(pcm, ax=ax, label="FoM (normalised 0–1)", pad=0.01)

    _overrides = _label_overrides or {}
    _tp = total_primaries
    for r in results:
        if _tp == 0:
            _tp = r.muon["muon_stats"]["total"]
        c = color_map.get(r.label, "gray")
        disp_label = _overrides.get(r.label.lower(), r.label)
        sr = [row for row in _compute_setup_curve_25b(r, W_range, M_fixed, _tp)
              if row["sig_surv"] >= 0.80]
        if sr:
            xs, ys, si, co = _rows_to_band_arrays(sr)
            _plot_curve_with_bands(ax, xs, ys, si, co,
                                   color=c, label=disp_label,
                                   linestyle="-", linewidth=1.5, zorder=3,
                                   point_labels=_w_labels_sorted(sr))

    if shared_sl_filt:
        xs, ys, si, co = _rows_to_band_arrays(shared_sl_filt)
        _plot_curve_with_bands(ax, xs, ys, si, co,
                               color=_sl_color,
                               label="Full Captures - stat. limit",
                               linestyle=":", linewidth=1.5, zorder=4,
                               alpha_inner=0.15, alpha_outer=0.07,
                               point_labels=_w_labels_sorted(shared_sl_filt))

    from matplotlib.patches import Patch as _Patch
    handles, _ = ax.get_legend_handles_labels()
    handles += [
        _Patch(facecolor="gray", alpha=0.25, edgecolor="none",
               label="Statistical uncertainty"),
        _Patch(facecolor="gray", alpha=0.12, edgecolor="none",
               label="Stat. ⊕ 35 % systematic"),
    ]
    ax.legend(handles=handles, fontsize=10, loc="upper left")

    ax.set_xlabel("Signal survival  (1 − deadtime)", fontsize=13)
    ax.set_ylabel("Ge-77 survival  (Σ FN NCs / Σ all Ge-77 NCs)", fontsize=13)
    ax.set_title(
        f"Ge-77 Survival vs Signal Livetime  [M = {M_fixed}]  — all setups\n"
        f"M = min. firing PMTs per NC  ·  W = min. detected NCs per muon to tag as Ge-77\n"
        f"(inner band: stat.  outer band: stat. ⊕ 35 % syst.  ·  signal survival ≥ 80 %)",
        fontsize=12,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.set_xlim(0.80, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "25c_ge_surv_vs_livetime_setups.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plot 25_stat_limit_only — only the shared NC-truth stat limit
# ──────────────────────────────────────────────────────────────────────
def plot_stat_limit_only(
    stat_limit_rows: list[dict],
    output_dir: str,
    stat_limit_color: str = "tab:blue",
) -> None:
    """Plot 25_stat_limit_only: only the NC-truth statistical limit curve.

    Shows the theoretical upper bound (W up to 50) with uncertainty bands.
    x: 80–100%, y: 0–100%. FoM colormap covers full axes.
    """
    sl_filt = [r for r in stat_limit_rows if r["sig_surv"] >= 0.80]
    if not sl_filt:
        print("  [SKIP] plot_stat_limit_only: no stat-limit rows with sig_surv ≥ 80 %.")
        return

    def _w_labels_sorted(rows: list[dict]) -> np.ndarray:
        xs_tmp = np.array([r["sig_surv"] for r in rows])
        order  = np.argsort(xs_tmp)
        return np.array([rows[i]["x_cut"] for i in order])

    fig, ax = plt.subplots(figsize=(10, 8))

    pcm = _fom_colormap_background(
        ax, [], [], normalize=True, cmap="YlOrBr", alpha=0.30,
        x_range=(0.80, 1.0), y_range=(0.0, 1.0),
    )
    if pcm is not None:
        fig.colorbar(pcm, ax=ax, label="FoM (normalised 0–1)", pad=0.01)

    xs, ys, si, co = _rows_to_band_arrays(sl_filt)
    _plot_curve_with_bands(ax, xs, ys, si, co,
                           color=stat_limit_color,
                           label="Full Captures - stat. limit",
                           linestyle=":", linewidth=2.0, zorder=4,
                           alpha_inner=0.15, alpha_outer=0.07,
                           point_labels=_w_labels_sorted(sl_filt))

    from matplotlib.patches import Patch as _Patch
    handles, _ = ax.get_legend_handles_labels()
    handles += [
        _Patch(facecolor="gray", alpha=0.25, edgecolor="none",
               label="Statistical uncertainty"),
        _Patch(facecolor="gray", alpha=0.12, edgecolor="none",
               label="Stat. ⊕ 35 % systematic"),
    ]
    ax.legend(handles=handles, fontsize=10, loc="upper left")

    ax.set_xlabel("Signal survival  (1 − deadtime)", fontsize=13)
    ax.set_ylabel("Ge-77 survival  (Σ FN NCs / Σ all Ge-77 NCs)", fontsize=13)
    ax.set_title(
        "NC-Truth Statistical Limit\n"
        "W = min. detected NCs per muon (truth level, up to W = 50)\n"
        "(inner band: stat.  outer band: stat. ⊕ 35 % syst.  ·  signal survival ≥ 80 %)",
        fontsize=12,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.set_xlim(0.80, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "25_stat_limit_only.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Plots 25d / 25e — ge_surv vs livetime: one best-FoM point per setup
# ──────────────────────────────────────────────────────────────────────
def plot_ge_surv_best_fom(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int = 0,
    color_map: dict[str, str] | None = None,
    m_min: int = 1,
    stat_limit_rows: "list[dict] | None" = None,
    stat_limit_color: str = "tab:blue",
) -> None:
    """Scatter: one point per setup at the FoM-optimal (M, W).

    ``m_min`` restricts the M search space to M >= m_min (use 6 for plot 25e).
    Each point is annotated with its optimal (M, W) pair.
    If ``stat_limit_rows`` is given, the shared NC-truth stat limit is overlaid.
    """
    if color_map is None:
        _pal = _colors(len(results))
        color_map = {r.label: _pal[i] for i, r in enumerate(results)}
    colors = [color_map.get(r.label, "gray") for r in results]

    m_search = [M for M in M_values if M >= m_min]
    if not m_search:
        print(f"  [SKIP] plot_ge_surv_best_fom(m_min={m_min}): no M values >= {m_min}.")
        return

    pts_x: list[float] = []
    pts_y: list[float] = []
    pt_labels: list[str] = []

    for r in results:
        _tp = total_primaries if total_primaries > 0 else r.muon["muon_stats"]["total"]
        grid = _cc_fom_grid(r, m_search, W_values, _tp)
        valid = {k: v for k, v in grid.items() if np.isfinite(v)}
        if not valid:
            pts_x.append(float("nan"))
            pts_y.append(float("nan"))
            pt_labels.append("N/A")
            continue
        best_M, best_W = max(valid, key=valid.__getitem__)
        cm = r.muon["confusion"].get((best_M, best_W))
        TN = _tp - cm["TP"] - cm["FP"] - cm["FN"]
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

    sl_filt: list[dict] = []
    if stat_limit_rows:
        sl_filt = [r for r in stat_limit_rows if np.isfinite(r["sig_surv"])]
        all_xs = list(all_xs) + [r["sig_surv"]   for r in sl_filt]
        all_ys = list(all_ys) + [r["ge_77_surv"] for r in sl_filt]

    fig, ax = plt.subplots(figsize=(10, 7))
    pcm = _fom_colormap_background(ax, all_xs, all_ys)
    if pcm is not None:
        fig.colorbar(pcm, ax=ax, label="FoM", pad=0.01)

    for r, c, x, y, lbl in zip(results, colors, pts_x, pts_y, pt_labels):
        if np.isfinite(x):
            ax.scatter([x], [y], color=c, s=60, zorder=4)
            ax.annotate(lbl, xy=(x, y), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color=c)

    if sl_filt:
        xs_sl = np.array([r["sig_surv"]       for r in sl_filt])
        ys_sl = np.array([r["ge_77_surv"]     for r in sl_filt])
        si_sl = np.array([r["ge_77_surv_unc"] for r in sl_filt])
        co_sl = np.array([_combined_unc_25b(s, v) for s, v in zip(si_sl, ys_sl)])
        _plot_curve_with_bands(ax, xs_sl, ys_sl, si_sl, co_sl,
                               color=stat_limit_color,
                               label="Full Captures - stat. limit",
                               linestyle=":", linewidth=1.5, zorder=4,
                               alpha_inner=0.15, alpha_outer=0.07)

    m_desc = f"M ≥ {m_min}" if m_min > 1 else "all M"
    ax.set_xlabel("1 − Deadtime  (signal livetime fraction)", fontsize=13)
    ax.set_ylabel("Ge77 survival  (Σ FN Ge77 NCs / Σ all Ge77 NCs)", fontsize=13)
    ax.set_title(
        f"Ge77 Survival vs Signal Livetime — Best FoM Point per Setup  ({m_desc})\n"
        "(each point = FoM-optimal (M, W); bottom-right = optimal)",
        fontsize=14,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map.get(r.label, "gray"),
                   markersize=8, label=r.label)
        for r in results
    ]
    if sl_filt:
        legend_handles.append(
            plt.Line2D([0], [0], color=stat_limit_color, linestyle=":",
                       linewidth=1.5, label="Full Captures - stat. limit")
        )
    ax.legend(handles=legend_handles, fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    # Lock axis limits to heatmap extent so no blank background areas appear.
    if all_xs and all_ys:
        _ax_dx = max((max(all_xs) - min(all_xs)) * 0.05, 1e-4)
        _ax_dy = max((max(all_ys) - min(all_ys)) * 0.05, 1e-4)
        ax.set_xlim(min(all_xs) - _ax_dx, max(all_xs) + _ax_dx)
        ax.set_ylim(min(all_ys) - _ax_dy, max(all_ys) + _ax_dy)
    fig.tight_layout()
    fname = f"25{'d' if m_min <= 1 else 'e'}_ge_surv_best_fom{'_M_ge6' if m_min > 1 else ''}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


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
            if r.w2_global is not None:
                f.write(f"  W2 homogeneity: {r.w2_global:.2f} mm\n")
            f.write("\n")
    print("  Saved nc_summary.txt")


def write_confusion_matrices(
    results: list[SetupResult],
    M_values: list[int],
    W_values: list[int],
    output_dir: str,
    total_primaries: int,
) -> None:
    """Write confusion_matrices.txt for all (config, M, W).

    TN = total_primaries - TP - FP - FN (all simulated muons as denominator).
    """
    fpath = os.path.join(output_dir, "confusion_matrices.txt")
    with open(fpath, "w") as f:
        f.write("Ge-77 Classification Confusion Matrices\n")
        f.write(f"total_primaries = {total_primaries:,}\n")
        f.write("=" * 90 + "\n\n")
        hdr = (
            f"{'Config':<20} {'M':>3} {'W':>3}  "
            f"{'TP':>7} {'FP':>7} {'TN':>12} {'FN':>7}  "
            f"{'Recall':>7} {'Prec':>7}\n"
        )
        f.write(hdr)
        f.write("-" * (len(hdr) - 1) + "\n")
        for r in results:
            for M in M_values:
                for W in W_values:
                    conf = r.muon["confusion"][(M, W)]
                    tn   = total_primaries - conf["TP"] - conf["FP"] - conf["FN"]
                    m    = compute_metrics(conf["TP"], conf["FP"], conf["FN"])
                    f.write(
                        f"{r.label:<20} {M:>3} {W:>3}  "
                        f"{conf['TP']:>7} {conf['FP']:>7} "
                        f"{tn:>12} {conf['FN']:>7}  "
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
        if total_primaries > 0:
            f.write(f"  all_muons       = {total_primaries:,}\n\n")
        else:
            f.write(f"  all_muons       = per-setup muon count\n\n")

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
            sig_surv  = _cc_signal_survival(r, M_b, W_b, total_primaries=_tp)
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
    parser.add_argument("--M-max", type=int, default=15,
                        help="Max M for sweep (default: 15).")
    parser.add_argument("--W-max", type=int, default=30,
                        help="Max W for sweep (default: 30).")
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
    parser.add_argument(
        "--statistical-limit-csv", default=None, metavar="CSV",
        help=(
            "Path to statistical-limit CSV for plot 25b overlay "
            "(format per line: 'x cut <val>, ge_77_surv: <val>, sig_surv: <val>')."
        ),
    )
    parser.add_argument(
        "--baseline-display-label",
        default="Full Captures Homogeneous PMT Distribution",
        metavar="LABEL",
        help=(
            "Display label for the baseline setup in Plot 25/25b comparison plots "
            "(default: 'Full Captures Homogeneous PMT Distribution')."
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
    # Load material names when an advisor CSV is provided so the in-water
    # box filter (needed for a fair stat-limit comparison) is available.
    _load_material = bool(args.advisor_csv)
    print("Loading NC truth ...")
    nc_truth = build_nc_truth(
        args.muon_dir, verbose=True, omit_runs=omit_runs,
        include_material=_load_material,
    )
    print()

    # ── 1b. Total primaries (needed for stat limit and FoM) ───────────
    _n_runs_pre      = len(count_vertices_by_run(args.muon_dir, omit_runs=omit_runs))
    _total_primaries = _n_runs_pre * MUONS_PER_RUN_DIR
    _runtime_h       = _total_primaries / MUSUN_RATE
    _runtime_yr      = _runtime_h / (24 * 365.25)
    print(f"  Total primary muons: {_total_primaries:,}  "
          f"({_n_runs_pre} runs × {MUONS_PER_RUN_DIR:,})")
    print(f"  Simulated livetime : {_runtime_h:,.0f} h  =  {_runtime_yr:.2f} yr  "
          f"(at {MUSUN_RATE} µ/h)")
    print()

    # ── 1c. NC-truth statistical limit (shared across all setups) ────
    # Always computed with W up to 50 regardless of args.W_max.
    W_values_stat = list(range(1, 51))
    print("Computing NC-truth statistical limit ...")
    _stat_limit_rows = _compute_nc_truth_stat_limit(nc_truth, W_values_stat, _total_primaries)
    print(f"  Stat-limit rows computed: {len(_stat_limit_rows)}")

    # In-water box filtered stat limit — only computed when an advisor CSV is
    # provided, so the two stat limits are comparable (same NC population).
    _stat_limit_rows_advisor: list[dict] = []
    if args.advisor_csv:
        print("Computing in-water box stat limit for advisor plot comparison ...")
        _stat_limit_rows_advisor = _compute_nc_truth_stat_limit_in_water(
            nc_truth, W_values_stat, _total_primaries
        )
        print(f"  In-water stat-limit rows: {len(_stat_limit_rows_advisor)}")
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

        _w2_dict: dict[str, Optional[float]] = {}
        per_area_n: dict[str, int] = {}
        if args.configs is not None:
            print("  Computing W2 metrics and area counts ...")
            _w2_dict   = _try_compute_w2(args.configs[i])
            per_area_n = _count_areas_from_json(args.configs[i])
            if _w2_dict.get("w2_global") is not None:
                print(
                    f"  W2_global = {_w2_dict['w2_global']:.2f} mm  "
                    f"W2_z = {_w2_dict['w2_z']:.2f} mm  "
                    f"W2_phi = {_w2_dict['w2_phi']:.4f} rad  "
                    f"W2_r = {_w2_dict['w2_r']:.2f} mm"
                )
            if per_area_n:
                print(f"  Areas: {per_area_n}")

        results.append(SetupResult(
            label=label, nc=nc_res, muon=muon_res, pmt_uids=pmt_uids,
            w2_global=_w2_dict.get("w2_global"),
            w2_z=_w2_dict.get("w2_z"),
            w2_phi=_w2_dict.get("w2_phi"),
            w2_r=_w2_dict.get("w2_r"),
            per_area_n=per_area_n,
            stat_limit_rows=_stat_limit_rows,
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
        m    = compute_metrics(conf["TP"], conf["FP"], conf["FN"])
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
        _tn = _total_primaries - conf['TP'] - conf['FP'] - conf['FN']
        print(f"  At (M={M_default}, W={W_default}): TP={conf['TP']}  FN={conf['FN']}  TN={_tn:,}  FP={conf['FP']}")
        print(f"  Recall={m['Recall']:.3f}  Precision={m['Precision']:.3f}")
        if r.w2_global is not None:
            print(
                f"  W2_global={r.w2_global:.2f} mm  "
                f"W2_z={r.w2_z:.2f} mm  "
                f"W2_phi={r.w2_phi:.4f} rad  "
                f"W2_r={r.w2_r:.2f} mm"
            )

    print()

    # ── 6. Generate plots ─────────────────────────────────────────────
    print("Generating plots ...")

    # W2 reference validation (always generated — documents the uniform references)
    print("  Generating W2 reference validation ...")
    plot_w2_uniform_ref_validation(args.output_dir)

    # Build a global color map keyed by label so every setup uses the same
    # colour in all plots regardless of which subsets each function receives.
    _pal_global = _colors(len(results))
    color_map   = {r.label: _pal_global[i] for i, r in enumerate(results)}

    # _total_primaries and _runtime_h/_runtime_yr already computed in step 1b.
    print(f"\n  FoM: total primary muons = {_total_primaries:,}  "
          f"({_n_runs_pre} runs × {MUONS_PER_RUN_DIR:,})")
    print(f"  FoM: simulated livetime  = {_runtime_h:,.0f} h  =  {_runtime_yr:.2f} yr  "
          f"(at {MUSUN_RATE} µ/h)")

    plot_nc_coverage_line(results, M_values, args.output_dir,
                          color_map=color_map)
    plot_nc_rank_spearman(results, M_values, args.output_dir, M_ref=1)
    plot_nc_detectability_overview(results, M_default, args.output_dir,
                                   color_map=color_map)
    plot_ge77_muon_overview(results, M_default, W_default, args.output_dir,
                            color_map=color_map,
                            total_primary_muons=_total_primaries)
    _heatmap_ms = [m for m in [1, 3, 5, 10] if m in M_values]
    plot_confusion_bar(results, M_default, W_default, args.output_dir,
                       color_map=color_map, total_primaries=_total_primaries)

    # Plot 07b — TP vs FP scatter (test: more TP → more FP?)
    plot_tp_fp_scatter(results, M_values, W_values, args.output_dir,
                       color_map=color_map)

    # M×W sweep (Recall, Precision, FoM)
    plot_mw_sweep(results, M_values, W_values, args.output_dir,
                  total_primaries=_total_primaries, color_map=color_map)
    plot_fom_summary(results, M_values, W_values, args.output_dir,
                     total_primaries=_total_primaries, color_map=color_map)
    plot_fom_summary_min_m(results, M_values, W_values, args.output_dir,
                           min_M=6, total_primaries=_total_primaries, color_map=color_map)

    plot_w2_nc_scatter(results, M_values, args.output_dir, color_map=color_map)
    plot_w2_recall_best_fom_all_variants(results, M_values, W_values, args.output_dir,
                                          total_primaries=_total_primaries, color_map=color_map)
    plot_w2_fom_best_fom_all_variants(results, M_values, W_values, args.output_dir,
                                       total_primaries=_total_primaries, color_map=color_map)

    # W2 correlation analysis
    # nc_correlation: only M=1..4 where the relationship is statistically significant.
    _corr_ms = [m for m in M_values if m <= 4]
    plot_w2_nc_correlation(results, _corr_ms, args.output_dir,
                           color_map=color_map)
    plot_w2_correlation_matrix(results, M_values, M_default, W_default, args.output_dir)
    plot_w2_coverage_profile(results, M_values, args.output_dir)

    # W2 correlation plots: only keep global NC-coverage scatter; focus on correlation lines.
    plot_w2_nc_coverage_scatter(results, M_values, args.output_dir,
                                color_map=color_map)

    # Plot 17a_fom — W2 vs FoM scatter grid (one panel per M, all W2 variants)
    plot_w2_fom_scatter_all_m(results, M_values, W_values, args.output_dir,
                              total_primaries=_total_primaries, color_map=color_map)
    plot_w2_z_fom_scatter(results, M_values, W_values, args.output_dir,
                          total_primaries=_total_primaries, color_map=color_map)
    plot_w2_phi_fom_scatter(results, M_values, W_values, args.output_dir,
                            total_primaries=_total_primaries, color_map=color_map)
    plot_w2_r_fom_scatter(results, M_values, W_values, args.output_dir,
                          total_primaries=_total_primaries, color_map=color_map)

    # Plot 17b — FoM correlation for all W2 variants (all M)
    plot_w2_fom_corr_mge(results, M_values, W_values, args.output_dir,
                         min_m=1, total_primaries=_total_primaries)
    plot_w2_z_fom_corr_mge(results, M_values, W_values, args.output_dir,
                            min_m=1, total_primaries=_total_primaries)
    plot_w2_phi_fom_corr_mge(results, M_values, W_values, args.output_dir,
                              min_m=1, total_primaries=_total_primaries)
    plot_w2_r_fom_corr_mge(results, M_values, W_values, args.output_dir,
                             min_m=1, total_primaries=_total_primaries)

    # Plot 17d — NC coverage correlation for all W2 variants (all M)
    plot_w2_nc_coverage_corr_all_m(results, M_values, args.output_dir)
    plot_w2_z_nc_coverage_corr_all_m(results, M_values, args.output_dir)
    plot_w2_phi_nc_coverage_corr_all_m(results, M_values, args.output_dir)
    plot_w2_r_nc_coverage_corr_all_m(results, M_values, args.output_dir)

    # Plot 17e — 4-curve recall correlation for all W2 variants (all M)
    plot_w2_recall_corr_all_m(results, M_values, W_values, args.output_dir,
                               min_m_constrained=6,
                               total_primaries=_total_primaries)
    plot_w2_z_recall_corr_all_m(results, M_values, W_values, args.output_dir,
                                 min_m_constrained=6,
                                 total_primaries=_total_primaries)
    plot_w2_phi_recall_corr_all_m(results, M_values, W_values, args.output_dir,
                                   min_m_constrained=6,
                                   total_primaries=_total_primaries)
    plot_w2_r_recall_corr_all_m(results, M_values, W_values, args.output_dir,
                                  min_m_constrained=6,
                                  total_primaries=_total_primaries)

    # Combined Spearman summary kept for backwards compatibility:
    plot_w2_spearman_vs_m(results, M_values, W_default, args.output_dir,
                          total_primaries=_total_primaries, W_values=W_values)

    # NC-recall correlation for multiple W thresholds
    for _W_fixed in [1, 2, 3, 5, 10]:
        if _W_fixed in W_values:
            plot_nc_recall_correlation(results, _heatmap_ms, args.output_dir,
                                       color_map=color_map, W_fixed=_W_fixed)
    _w19_fixed = [W for W in [1, 2, 3, 5, 10] if W in W_values]
    plot_nc_recall_correlation_summary(results, _heatmap_ms, _w19_fixed, args.output_dir,
                                        color_map=color_map)

    # New plots: FoM-optimal analysis
    plot_recall_w1_vs_m(results, M_values, args.output_dir, color_map=color_map)
    # Plot 21 — Recall + Precision at best FoM (all M)
    plot_recall_at_best_fom(results, M_values, W_values, args.output_dir,
                            total_primaries=_total_primaries, color_map=color_map)
    # Plot 21c — Recall + Precision at best FoM (M ≥ 6)
    plot_recall_at_best_fom(results, M_values, W_values, args.output_dir,
                            total_primaries=_total_primaries, color_map=color_map,
                            min_m=6)
    # Stat limit colour: use the baseline setup's colour so it's visually consistent.
    _bl_for_sl = _find_baseline_result(results)
    _sl_color  = color_map.get(_bl_for_sl.label, "tab:blue")

    plot_ge_surv_vs_livetime(results, M_values, W_values, args.output_dir,
                             total_primaries=_total_primaries, color_map=color_map,
                             stat_limit_rows=_stat_limit_rows,
                             stat_limit_color=_sl_color)
    # Plot 25: baseline + NC-truth stat limit (always generated, no advisor CSV needed)
    plot_ge_surv_vs_livetime_nc_truth_baseline(
        results, W_values, args.output_dir,
        total_primaries=_total_primaries,
        color_map=color_map,
        M_fixed=args.advisor_M,
        baseline_display_label=args.baseline_display_label,
    )
    # Plot 25c: all setups + shared stat limit, no advisor data
    plot_ge_surv_setups_only(
        results, W_values, args.output_dir,
        total_primaries=_total_primaries,
        color_map=color_map,
        M_fixed=args.advisor_M,
        baseline_display_label=args.baseline_display_label,
    )
    # Plot 25_stat_limit_only: only the shared stat limit
    plot_stat_limit_only(_stat_limit_rows, args.output_dir, stat_limit_color=_sl_color)
    if args.advisor_csv:
        plot_ge_surv_vs_livetime_advisor(
            results, W_values, args.output_dir,
            advisor_csv=args.advisor_csv,
            total_primaries=_total_primaries,
            color_map=color_map,
            M_fixed=args.advisor_M,
            statistical_limit_csv=args.statistical_limit_csv,
            baseline_display_label=args.baseline_display_label,
            nc_stat_limit_override=_stat_limit_rows_advisor or None,
        )
        plot_ge_surv_vs_livetime_advisor_baseline(
            results, W_values, args.output_dir,
            advisor_csv=args.advisor_csv,
            total_primaries=_total_primaries,
            color_map=color_map,
            M_fixed=args.advisor_M,
            statistical_limit_csv=args.statistical_limit_csv,
            baseline_display_label=args.baseline_display_label,
            nc_stat_limit_override=_stat_limit_rows_advisor or None,
        )
    plot_ge_surv_best_fom(results, M_values, W_values, args.output_dir,
                          total_primaries=_total_primaries, color_map=color_map, m_min=1,
                          stat_limit_rows=_stat_limit_rows, stat_limit_color=_sl_color)
    plot_ge_surv_best_fom(results, M_values, W_values, args.output_dir,
                          total_primaries=_total_primaries, color_map=color_map, m_min=6,
                          stat_limit_rows=_stat_limit_rows, stat_limit_color=_sl_color)

    # ── 7. Write text files ───────────────────────────────────────────
    print("Writing text summaries ...")
    write_nc_summary(results, M_values, args.output_dir)
    write_confusion_matrices(results, M_values, W_values, args.output_dir,
                             total_primaries=_total_primaries)
    write_survival_table(results, M_values, W_values, args.output_dir,
                         total_primaries=_total_primaries)

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
