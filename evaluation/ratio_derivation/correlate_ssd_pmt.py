#!/usr/bin/env python3
"""
SSD–PMT Correlation Analysis
=============================
Sweeps the SSD area ratio and identifies the value that maximises agreement
between SSD-derived metrics and real PMT simulation results across N setups.

For each experimental setup the script loads:
  - PMT simulation data  (LGDO format, Sim 2 optical + NC truth)
  - SSD simulation data  (shared HDF5 target_matrix + phi_matrix)

The ratio is swept uniformly across all detector layers first; then
independently per layer while holding all others at the global optimum.
Pearson and Spearman correlations between SSD and PMT metrics are computed
at every ratio step, and the optimal ratio is identified from mean |Pearson r|
for NC coverage at M = 1–4.

Additionally, at the optimal ratio, the script produces:
  - NC coverage ranking consistency plots (Spearman ρ between SSD and PMT
    ranking of setups, vs M threshold)
  - FoM comparison bars and scatter per M
  - FoM comparison under M ≥ min_M constraint (both PMT-anchor and
    independent-best methods)
  - Recall comparison at PMT's best (M,W)
  - Best-(M,W) agreement table between PMT and SSD

Usage
-----
Fixed-ratio mode (default) — evaluate at explicitly given per-layer ratios:

    python correlate_ssd_pmt.py \\
        --muon-dir  /path/to/nc_truth/ \\
        --ssd-hdf5  /path/to/data.hdf5 \\
        --setup-dirs /path/A /path/B /path/C \\
        --pit 2.0731 --bot 2.3843 --top 2.2004 --wall 1.8776 \\
        [--m 1] [--M-max 10] [--W-max 20] [--W-default 1] \\
        [--min-M-fom 6] [--output-dir ./correlation_results] \\
        [--seed 42]

Ratio-sweep mode — sweep the ratio to maximise SSD–PMT correlation:

    python correlate_ssd_pmt.py \\
        --ratio-sweep \\
        --muon-dir  /path/to/nc_truth/ \\
        --ssd-hdf5  /path/to/data.hdf5 \\
        --setup-dirs /path/A /path/B /path/C \\
        [--m 1] [--M-max 10] [--W-max 20] [--W-default 1] \\
        [--ratio-min 1.0] [--ratio-max 6.0] [--ratio-step 0.1] \\
        [--min-M-fom 6] [--output-dir ./correlation_results] \\
        [--seed 42] [--skip-per-layer]

Author: Thomas Buerger (University of Tübingen)
"""
from __future__ import annotations

import argparse
import gc
import glob
import h5py
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── sys.path setup ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
_pmtopt_src = Path(__file__).resolve().parents[2] / "src"
if _pmtopt_src.is_dir() and str(_pmtopt_src) not in sys.path:
    sys.path.insert(0, str(_pmtopt_src))

from ratio_analysis.raw_loading import build_nc_truth, build_pmt_matrix
from ratio_analysis.coverage_analysis import (
    evaluate_nc  as _pmt_evaluate_nc,
    evaluate_muon as _pmt_evaluate_muon,
    compute_metrics,
)
from pmtopt.data_loading import (
    load_raw_sparse,
    binarize_from_raw,
    build_muon_index,
)
from pmtopt.geometry import (
    DEFAULT_AREA_RATIOS,
    MUON_TIME_WINDOW_MIN_NS,
    MUON_TIME_WINDOW_MAX_NS,
    MUSUN_RATE,
    MUONS_PER_RUN_DIR,
    calc_fom_confusion,
)

# ── Publication-quality global style (matches compare_coverages.py) ───
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


# ─────────────────────────────────────────────────────────────────────
# SECTION 2 — Constants & helpers
# ─────────────────────────────────────────────────────────────────────

_SSD_RUN_COL  = "run_id"
_SSD_MUON_COL = "local_muon_id"
_SSD_NC_COL   = "nc_id"

_DEFAULT_RATIO_RANGE: tuple[float, float] = (
    min(DEFAULT_AREA_RATIOS.values()),
    max(DEFAULT_AREA_RATIOS.values()),
)

_SETUP_PALETTE: list[str] = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#17becf", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f",
    "#1b9e77", "#e7298a", "#e6ab02", "#006d2c", "#d95f02",
    "#7570b3", "#f46d43", "#74c476", "#b15928", "#313695",
]

# Two-sided p-value for 3-sigma significance (used for marker filling)
_P_3SIGMA: float = 2.0 * scipy_stats.norm.sf(3.0)


def _pearson_rcrit(n: int, sigma: float = 3.0) -> float:
    """Critical Pearson |r| for sigma-sigma two-sided significance, n samples."""
    p_two  = 2.0 * scipy_stats.norm.sf(sigma)
    t_crit = scipy_stats.t.ppf(1.0 - p_two / 2.0, df=max(n - 2, 1))
    dof    = max(n - 2, 1)
    return float(t_crit / np.sqrt(dof + t_crit ** 2))


def _colors(n: int) -> list[str]:
    return [_SETUP_PALETTE[i % len(_SETUP_PALETTE)] for i in range(n)]


@dataclass
class SSDResult:
    """SSD evaluation results at one area ratio for one setup."""
    nc_detected: dict[int, int]
    num_ncs:     int
    confusion:   dict[tuple[int, int], dict]   # (M,W) -> {TP,FP,TN,FN,tp/fn_ge77_nc_counts}


@dataclass
class SetupData:
    """All data for one experimental setup."""
    label:       str
    w2:          Optional[float]
    pmt_nc:      dict
    pmt_muon:    dict
    ssd_results: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# SECTION 3 — JSON loader
# ─────────────────────────────────────────────────────────────────────

def _load_json_voxel_ids(json_path: str) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [v["index"] if isinstance(v, dict) and "index" in v else str(v) for v in data]
    voxel_dicts = data.get("selected_voxels", [])
    return [v["index"] for v in voxel_dicts if isinstance(v, dict) and "index" in v]


# ─────────────────────────────────────────────────────────────────────
# SECTION 4 — W2 helper
# ─────────────────────────────────────────────────────────────────────

def compute_setup_w2(json_path: str) -> Optional[float]:
    try:
        from pmtopt.homogeneous import compute_wasserstein_homogeneity, get_w2_ref
    except ImportError:
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        voxel_dicts = (data if isinstance(data, list) else data.get("selected_voxels", []))
        voxel_dicts = [v for v in voxel_dicts if isinstance(v, dict) and "center" in v]
        if len(voxel_dicts) < 2:
            return None
        centers = np.array([v["center"] for v in voxel_dicts], dtype=float)
        return float(compute_wasserstein_homogeneity(centers, reference=get_w2_ref())["w2"])
    except Exception as exc:
        print(f"  [WARN] W2 failed for {json_path!r}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────
# SECTION 5 — NC alignment
# ─────────────────────────────────────────────────────────────────────

def align_ssd_to_pmt(hdf5_path: str, nc_truth: pd.DataFrame) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        phi_columns = [c.decode() if isinstance(c, bytes) else str(c) for c in f["phi_columns"][:]]
        phi_col_idx = {name: i for i, name in enumerate(phi_columns)}
        phi = f["phi_matrix"]
        nc_time_ns   = phi[:, phi_col_idx["nC_time_in_ns"]].astype(np.float64)
        nc_flag_ge77 = phi[:, phi_col_idx["nC_flag_Ge77"]].astype(bool)
        event_id_cols = [c.decode() if isinstance(c, bytes) else str(c)
                         for c in f["event_id_columns"][:]]
        event_ids = f["event_ids"][:]
        run_col  = event_id_cols.index("run_id")
        muon_col = event_id_cols.index("muon_id")
        pairs = np.stack([event_ids[:, run_col], event_ids[:, muon_col]], axis=1)
        _, global_muon_id = np.unique(pairs, axis=0, return_inverse=True)
        global_muon_id = global_muon_id.astype(np.int64)
        has_alignment_cols = all(c in phi_col_idx for c in [_SSD_RUN_COL, _SSD_MUON_COL, _SSD_NC_COL])

    num_ncs = len(global_muon_id)
    if has_alignment_cols:
        warnings.warn("SSD phi_matrix alignment columns found but key-based alignment not yet "
                      "implemented — falling back to nc_time_ns ordering (approximate).")
    else:
        warnings.warn("SSD phi_matrix missing alignment columns — using nc_time_ns ordering (approximate).")

    ssd_row_order = np.arange(num_ncs, dtype=np.int64)
    (_, _, ge77_muon_global_ids, _, _) = build_muon_index(
        global_muon_id, nc_time_ns, nc_flag_ge77, verbose=False,
    )
    all_unique_muons    = np.unique(global_muon_id)
    total_muons         = len(all_unique_muons)
    global_to_all_local = np.searchsorted(all_unique_muons, global_muon_id).astype(np.int32)
    ge77_mask_all       = np.isin(all_unique_muons, ge77_muon_global_ids)
    in_time = (nc_time_ns >= MUON_TIME_WINDOW_MIN_NS) & (nc_time_ns <= MUON_TIME_WINDOW_MAX_NS)
    nc_is_veto_candidate = in_time & (~nc_flag_ge77)

    return {
        "ssd_row_order":        ssd_row_order,
        "global_muon_id":       global_muon_id,
        "nc_time_ns":           nc_time_ns,
        "nc_flag_ge77":         nc_flag_ge77,
        "all_unique_muons":     all_unique_muons,
        "ge77_muon_global_ids": ge77_muon_global_ids,
        "global_to_all_local":  global_to_all_local,
        "ge77_mask_all":        ge77_mask_all,
        "nc_is_veto_candidate": nc_is_veto_candidate,
        "total_muons":          total_muons,
    }


# ─────────────────────────────────────────────────────────────────────
# SECTION 6 — PMT data loading (per setup, called once)
# ─────────────────────────────────────────────────────────────────────

def load_pmt_setup(
    setup_dir: str,
    nc_truth: pd.DataFrame,
    M_values: list[int],
    W_values: list[int],
    m_threshold: int = 1,
) -> tuple[dict, dict, str]:
    json_files = sorted(glob.glob(os.path.join(setup_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No *.json config found in {setup_dir!r}")
    json_path = json_files[0]
    B, _pmt_uids, detect_info = build_pmt_matrix(setup_dir, nc_truth, m_threshold)
    pmt_nc   = _pmt_evaluate_nc(B, nc_truth, M_values, detect_info)
    pmt_muon = _pmt_evaluate_muon(B, nc_truth, M_values, W_values, detect_info)
    del B
    gc.collect()
    return pmt_nc, pmt_muon, json_path


# ─────────────────────────────────────────────────────────────────────
# SECTION 7 — SSD evaluation at one ratio
# ─────────────────────────────────────────────────────────────────────

def evaluate_ssd_at_ratio(
    raw_rows: np.ndarray,
    raw_cols: np.ndarray,
    raw_vals: np.ndarray,
    num_ncs: int,
    num_voxels: int,
    layers: np.ndarray,
    area_ratios: dict[str, float],
    m: int,
    M_values: list[int],
    W_values: list[int],
    alignment: dict,
    voxel_col_subset: np.ndarray,
    seed: int = 42,
) -> SSDResult:
    """Apply stochastic rounding, subset to setup voxels, compute metrics.

    Computes tp_ge77_nc_counts / fn_ge77_nc_counts per (M,W) to match the
    NC-weighted FoM definition used in compare_coverages.py.
    """
    B_full = binarize_from_raw(
        raw_rows, raw_cols, raw_vals,
        num_ncs, num_voxels, layers,
        area_ratios, m, seed=seed,
    )

    rows  = alignment["ssd_row_order"]
    B_sub = B_full.tocsr()[rows, :][:, voxel_col_subset].tocsc()

    # NC coverage counts
    coverage_counts = np.zeros(len(rows), dtype=np.int16)
    for col in range(B_sub.shape[1]):
        s, e = B_sub.indptr[col], B_sub.indptr[col + 1]
        coverage_counts[B_sub.indices[s:e]] += 1

    nc_detected = {M: int(np.sum(coverage_counts >= M)) for M in M_values}

    # Shared muon arrays
    nc_is_veto  = alignment["nc_is_veto_candidate"][rows]
    g2al        = alignment["global_to_all_local"][rows]
    ge77_mask   = alignment["ge77_mask_all"]
    total_muons = alignment["total_muons"]

    # Per-muon Ge77-NC counts (all flag_ge77==1 NCs, no time-window restriction).
    # Matches coverage_analysis.py: nc_truth.groupby([...])["flag_ge77"].sum()
    ge77_nc_flags          = alignment["nc_flag_ge77"][rows]
    ge77_nc_counts_per_muon = np.bincount(
        g2al[ge77_nc_flags], minlength=total_muons
    ).astype(np.int32)

    confusion: dict[tuple[int, int], dict] = {}
    for M in M_values:
        nc_det_veto = (coverage_counts >= M) & nc_is_veto
        det_idx     = np.where(nc_det_veto)[0]
        if len(det_idx) > 0:
            muon_det = np.bincount(g2al[det_idx], minlength=total_muons).astype(np.int32)
        else:
            muon_det = np.zeros(total_muons, dtype=np.int32)

        for W in W_values:
            cls     = muon_det >= W
            tp_mask = ge77_mask &  cls
            fn_mask = ge77_mask & ~cls
            confusion[(M, W)] = {
                "TP": int(tp_mask.sum()),
                "FP": int((cls & ~ge77_mask).sum()),
                "FN": int(fn_mask.sum()),
                "TN": int((~cls & ~ge77_mask).sum()),
                "tp_ge77_nc_counts": ge77_nc_counts_per_muon[tp_mask],
                "fn_ge77_nc_counts": ge77_nc_counts_per_muon[fn_mask],
            }

    del B_full, B_sub
    gc.collect()
    return SSDResult(nc_detected=nc_detected, num_ncs=int(len(rows)), confusion=confusion)


# ─────────────────────────────────────────────────────────────────────
# SECTION 8 — Metric extraction helpers
# ─────────────────────────────────────────────────────────────────────

def _pmt_nc_frac(setup: SetupData, M: int) -> float:
    total = setup.pmt_nc.get("nc_total", 0)
    return setup.pmt_nc["nc_detected"].get(M, 0) / total if total else 0.0


def _ssd_nc_frac(result: SSDResult, M: int) -> float:
    return result.nc_detected.get(M, 0) / result.num_ncs if result.num_ncs else 0.0


def _pmt_recall(setup: SetupData, M: int, W: int) -> float:
    conf = setup.pmt_muon["confusion"].get((M, W))
    if conf is None:
        return 0.0
    return compute_metrics(conf["TP"], conf["FP"], conf["FN"])["Recall"]


def _ssd_recall(result: SSDResult, M: int, W: int) -> float:
    conf = result.confusion.get((M, W))
    if conf is None:
        return 0.0
    return compute_metrics(conf["TP"], conf["FP"], conf["FN"])["Recall"]


def _pmt_fom(setup: SetupData, M: int, W: int, total_primaries: int) -> float:
    """FoM from PMT confusion matrix — passes NC-count arrays for weighted computation."""
    conf = setup.pmt_muon["confusion"].get((M, W))
    if conf is None:
        return float("nan")
    return calc_fom_confusion(
        conf["TP"], conf["FP"], conf["FN"], total_primaries,
        tp_ge77_nc_counts=conf.get("tp_ge77_nc_counts"),
        fn_ge77_nc_counts=conf.get("fn_ge77_nc_counts"),
    )


def _ssd_fom(result: SSDResult, M: int, W: int, total_primaries: int) -> float:
    """FoM from SSD confusion matrix — passes NC-count arrays for weighted computation."""
    conf = result.confusion.get((M, W))
    if conf is None:
        return float("nan")
    return calc_fom_confusion(
        conf["TP"], conf["FP"], conf["FN"], total_primaries,
        tp_ge77_nc_counts=conf.get("tp_ge77_nc_counts"),
        fn_ge77_nc_counts=conf.get("fn_ge77_nc_counts"),
    )


def _pmt_best_mw(
    setup: SetupData,
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    min_M: int = 1,
) -> tuple[int, int]:
    """Return the (M, W) pair that maximises PMT FoM (optionally M ≥ min_M)."""
    best_mw  = (M_values[0], W_values[0])
    best_fom = float("-inf")
    for M in M_values:
        if M < min_M:
            continue
        for W in W_values:
            v = _pmt_fom(setup, M, W, total_primaries)
            if np.isfinite(v) and v > best_fom:
                best_fom = v
                best_mw  = (M, W)
    return best_mw


def _pmt_best_w_at_m(
    setup: SetupData,
    M: int,
    W_values: list[int],
    total_primaries: int,
) -> int:
    best_w   = W_values[0]
    best_fom = float("-inf")
    for W in W_values:
        v = _pmt_fom(setup, M, W, total_primaries)
        if np.isfinite(v) and v > best_fom:
            best_fom = v
            best_w   = W
    return best_w


def _ssd_best_mw(
    result: SSDResult,
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    min_M: int = 1,
) -> tuple[int, int]:
    """Return the (M, W) pair that maximises SSD FoM (optionally M ≥ min_M)."""
    best_mw  = (M_values[0], W_values[0])
    best_fom = float("-inf")
    for M in M_values:
        if M < min_M:
            continue
        for W in W_values:
            v = _ssd_fom(result, M, W, total_primaries)
            if np.isfinite(v) and v > best_fom:
                best_fom = v
                best_mw  = (M, W)
    return best_mw


def _ssd_fom_grid(
    result: SSDResult,
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
) -> dict[tuple[int, int], float]:
    """Return {(M, W): fom} for all combinations."""
    return {
        (M, W): _ssd_fom(result, M, W, total_primaries)
        for M in M_values for W in W_values
    }


def _count_runs(muon_dir: str) -> int:
    pattern = os.path.join(muon_dir, "run_*")
    n = len([d for d in glob.glob(pattern) if os.path.isdir(d)])
    return max(1, n)


# ─────────────────────────────────────────────────────────────────────
# SECTION 9 — Correlation computation
# ─────────────────────────────────────────────────────────────────────

def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    try:
        r, p = scipy_stats.pearsonr(x, y)
        return float(r), float(p)
    except ValueError:
        return float("nan"), float("nan")


def _safe_spearmanr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    try:
        rho, p = scipy_stats.spearmanr(x, y)
        return float(rho), float(p)
    except Exception:
        return float("nan"), float("nan")


def compute_correlations(
    setups: list[SetupData],
    ratio_key,
    M_values: list[int],
    W_default: int,
    ratio_label: Optional[float] = None,
    total_primaries: int = 0,
) -> list[dict]:
    """Pearson/Spearman between SSD and PMT metrics for each M."""
    if ratio_label is None:
        ratio_label = ratio_key if isinstance(ratio_key, float) else float("nan")

    rows: list[dict] = []
    for M in M_values:
        ssd_nc = np.array([_ssd_nc_frac(s.ssd_results[ratio_key], M) for s in setups])
        pmt_nc = np.array([_pmt_nc_frac(s, M) for s in setups])
        pr, pp = _safe_pearsonr(ssd_nc, pmt_nc)
        sr, sp = _safe_spearmanr(ssd_nc, pmt_nc)
        rows.append({"ratio": ratio_label, "M": M, "metric": "nc_coverage",
                     "pearson_r": pr, "pearson_p": pp,
                     "spearman_rho": sr, "spearman_p": sp, "n": len(setups)})

        ssd_rec = np.array([_ssd_recall(s.ssd_results[ratio_key], M, W_default) for s in setups])
        pmt_rec = np.array([_pmt_recall(s, M, W_default) for s in setups])
        pr, pp = _safe_pearsonr(ssd_rec, pmt_rec)
        sr, sp = _safe_spearmanr(ssd_rec, pmt_rec)
        rows.append({"ratio": ratio_label, "M": M, "metric": "recall",
                     "pearson_r": pr, "pearson_p": pp,
                     "spearman_rho": sr, "spearman_p": sp, "n": len(setups)})

        if total_primaries > 0:
            ssd_fv = np.array([_ssd_fom(s.ssd_results[ratio_key], M, W_default, total_primaries)
                               for s in setups])
            pmt_fv = np.array([_pmt_fom(s, M, W_default, total_primaries) for s in setups])
            mask = np.isfinite(ssd_fv) & np.isfinite(pmt_fv)
            if mask.sum() >= 3:
                pr, pp = _safe_pearsonr(ssd_fv[mask], pmt_fv[mask])
                sr, sp = _safe_spearmanr(ssd_fv[mask], pmt_fv[mask])
            else:
                pr = pp = sr = sp = float("nan")
            rows.append({"ratio": ratio_label, "M": M, "metric": "fom",
                         "pearson_r": pr, "pearson_p": pp,
                         "spearman_rho": sr, "spearman_p": sp, "n": int(mask.sum())})

    return rows


# ─────────────────────────────────────────────────────────────────────
# SECTION 10 — Global ratio sweep
# ─────────────────────────────────────────────────────────────────────

def global_ratio_sweep(
    setups: list[SetupData],
    ssd_coo_data: tuple,
    setup_voxel_subsets: list[np.ndarray],
    alignment: dict,
    ratio_factors: np.ndarray,
    m: int,
    M_values: list[int],
    W_values: list[int],
    W_default: int,
    early_stop_patience: int = 3,
    seed: int = 42,
    total_primaries: int = 0,
) -> tuple[pd.DataFrame, float]:
    raw_rows, raw_cols, raw_vals, voxel_ids, _, layers, num_ncs, _ = ssd_coo_data
    num_voxels = len(voxel_ids)
    all_rows: list[dict] = []
    recent_mean_abs: list[float] = []
    patience_count = 0

    for ratio_factor in ratio_factors:
        ratio_factor = float(ratio_factor)
        area_ratios  = {lyr: ratio_factor for lyr in DEFAULT_AREA_RATIOS}
        print(f"  ratio={ratio_factor:.2f}", end=" ... ", flush=True)

        for i, setup in enumerate(setups):
            setup.ssd_results[ratio_factor] = evaluate_ssd_at_ratio(
                raw_rows, raw_cols, raw_vals, num_ncs, num_voxels, layers,
                area_ratios, m, M_values, W_values, alignment,
                setup_voxel_subsets[i], seed=seed,
            )

        corr_rows = compute_correlations(setups, ratio_factor, M_values, W_default,
                                         ratio_label=ratio_factor,
                                         total_primaries=total_primaries)
        all_rows.extend(corr_rows)

        nc_rows  = [r for r in corr_rows if r["metric"] == "nc_coverage" and 1 <= r["M"] <= 4]
        abs_rs   = [abs(r["pearson_r"]) for r in nc_rows if not math.isnan(r["pearson_r"])]
        mean_abs = float(np.mean(abs_rs)) if abs_rs else 0.0
        print(f"mean|r|={mean_abs:.3f}")

        if recent_mean_abs and mean_abs < recent_mean_abs[-1]:
            patience_count += 1
        else:
            patience_count = 0
        recent_mean_abs.append(mean_abs)

        if patience_count >= early_stop_patience:
            print(f"  Early stop at ratio={ratio_factor:.2f} (mean|r| down for "
                  f"{early_stop_patience} steps)")
            break

    corr_df = pd.DataFrame(all_rows)
    if corr_df.empty:
        return corr_df, float(ratio_factors[0])

    nc_df  = corr_df[corr_df["metric"] == "nc_coverage"]
    grp    = nc_df.groupby("ratio")["pearson_r"].mean().dropna()
    optimal_ratio = float(grp.idxmax()) if not grp.empty else float(ratio_factors[0])
    print(f"  → Optimal global ratio: {optimal_ratio:.2f}")
    return corr_df, optimal_ratio


# ─────────────────────────────────────────────────────────────────────
# SECTION 11 — Per-layer ratio sweep
# ─────────────────────────────────────────────────────────────────────

def per_layer_ratio_sweep(
    setups: list[SetupData],
    ssd_coo_data: tuple,
    setup_voxel_subsets: list[np.ndarray],
    alignment: dict,
    optimal_global_ratio: float,
    ratio_factors: np.ndarray,
    m: int,
    M_values: list[int],
    W_values: list[int],
    W_default: int,
    seed: int = 42,
    total_primaries: int = 0,
    early_stop_patience: int = 3,
) -> tuple[pd.DataFrame, dict[str, float]]:
    raw_rows, raw_cols, raw_vals, voxel_ids, _, layers, num_ncs, _ = ssd_coo_data
    num_voxels = len(voxel_ids)
    all_rows:         list[dict]       = []
    per_layer_optima: dict[str, float] = {}

    for layer in ["pit", "bot", "top", "wall"]:
        print(f"  Layer: {layer}")
        layer_rows:      list[dict]  = []
        recent_mean_abs: list[float] = []
        patience_count               = 0

        for ratio_factor in ratio_factors:
            ratio_factor = float(ratio_factor)
            area_ratios  = {lyr: optimal_global_ratio for lyr in DEFAULT_AREA_RATIOS}
            area_ratios[layer] = ratio_factor
            temp_key = (layer, ratio_factor)
            print(f"    ratio={ratio_factor:.2f}", end=" ... ", flush=True)

            for i, setup in enumerate(setups):
                setup.ssd_results[temp_key] = evaluate_ssd_at_ratio(
                    raw_rows, raw_cols, raw_vals, num_ncs, num_voxels, layers,
                    area_ratios, m, M_values, W_values, alignment,
                    setup_voxel_subsets[i], seed=seed,
                )

            cr = compute_correlations(setups, temp_key, M_values, W_default,
                                      ratio_label=ratio_factor,
                                      total_primaries=total_primaries)
            for row in cr:
                row["layer"] = layer
            layer_rows.extend(cr)

            for setup in setups:
                del setup.ssd_results[temp_key]
            gc.collect()

            nc_rows  = [r for r in cr if r["metric"] == "nc_coverage" and 1 <= r["M"] <= 4]
            abs_rs   = [abs(r["pearson_r"]) for r in nc_rows if not math.isnan(r["pearson_r"])]
            mean_abs = float(np.mean(abs_rs)) if abs_rs else 0.0
            print(f"mean|r|={mean_abs:.3f}")

            if recent_mean_abs and mean_abs < recent_mean_abs[-1]:
                patience_count += 1
            else:
                patience_count = 0
            recent_mean_abs.append(mean_abs)

            if patience_count >= early_stop_patience:
                print(f"    Early stop at ratio={ratio_factor:.2f}")
                break

        all_rows.extend(layer_rows)
        nc_layer = [r for r in layer_rows if r["metric"] == "nc_coverage" and 1 <= r["M"] <= 4]
        if nc_layer:
            df_l = pd.DataFrame(nc_layer)
            grp  = df_l.groupby("ratio")["pearson_r"].mean().dropna()
            per_layer_optima[layer] = float(grp.idxmax()) if not grp.empty else optimal_global_ratio
        else:
            per_layer_optima[layer] = optimal_global_ratio
        print(f"    → Optimal {layer} ratio: {per_layer_optima[layer]:.2f}")

    return pd.DataFrame(all_rows), per_layer_optima


# ─────────────────────────────────────────────────────────────────────
# SECTION 12 — Shared scatter helper
# ─────────────────────────────────────────────────────────────────────

def _scatter_panel(
    ax: plt.Axes,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    color_pts: list[str],
    labels: list[str],
    x_label: str,
    y_label: str,
    force_origin: bool = False,
) -> None:
    """Scatter + y=x reference + OLS regression + CI + Pearson/Spearman stats box.

    When *force_origin* is True the OLS fit is constrained to pass through the
    origin (zero intercept).  This is appropriate for same-metric SSD-vs-PMT
    comparisons where perfect agreement implies PMT = 1·SSD, i.e. a proportional
    relationship through the origin.  Introducing a free intercept in that context
    can confound a pure scale bias with a spurious offset and distort the
    interpretation of the fit slope.

    For cross-type comparisons (e.g. W2 vs NC fraction, or SSD NC vs PMT Recall)
    the metric range does not include the origin in any physically meaningful way,
    so *force_origin* should be False (the default).
    """
    n = len(x_arr)
    for x, y, c, lbl in zip(x_arr, y_arr, color_pts, labels):
        ax.scatter([x], [y], color=c, s=55, zorder=3)
        ax.annotate(lbl, xy=(x, y), xytext=(4, 3),
                    textcoords="offset points", fontsize=6, color=c)

    all_vals = np.concatenate([x_arr, y_arr])
    if len(all_vals) > 0:
        vmin = float(all_vals.min())
        vmax = float(all_vals.max())
        pad  = 0.05 * max(vmax - vmin, 1e-9)
        ref  = np.array([vmin - pad, vmax + pad])
        ax.plot(ref, ref, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)

    # Pre-compute zero-intercept slope once so it can appear in both the
    # annotation box and the regression overlay without duplicating effort.
    _fit_slope: float | None = None
    if force_origin and n >= 3 and np.std(x_arr) > 0 and np.std(y_arr) > 0:
        xx = float(np.dot(x_arr, x_arr))
        _fit_slope = float(np.dot(x_arr, y_arr) / xx) if xx > 0 else None

    if n >= 3 and np.std(x_arr) > 0 and np.std(y_arr) > 0:
        try:
            r_v, p_r   = scipy_stats.pearsonr(x_arr, y_arr)
            rho, p_rho = scipy_stats.spearmanr(x_arr, y_arr)
            r_crit     = _pearson_rcrit(n, sigma=3.0)
            sig_mark   = "*" if abs(r_v) >= r_crit else ""
            if force_origin and _fit_slope is not None:
                ann = (f"Pearson  r = {r_v:+.3f}{sig_mark}  (p={p_r:.3g})\n"
                       f"Spearman ρ = {rho:+.3f}  (p={p_rho:.3g})\n"
                       f"3σ threshold: |r|≥{r_crit:.2f}\n"
                       f"OLS slope={_fit_slope:.3f}  (origin-forced)")
            else:
                ann = (f"Pearson  r = {r_v:+.3f}{sig_mark}  (p={p_r:.3g})\n"
                       f"Spearman ρ = {rho:+.3f}  (p={p_rho:.3g})\n"
                       f"3σ threshold: |r|≥{r_crit:.2f}")
        except ValueError:
            ann = "constant data — no stats"
    elif n < 3:
        ann = "n < 3 — no stats"
    else:
        ann = "constant data — no stats"

    ax.text(0.03, 0.97, ann, transform=ax.transAxes,
            ha="left", va="top", fontsize=7, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85))

    if n >= 3 and np.std(x_arr) > 0 and np.std(y_arr) > 0:
        x_fit = np.linspace(x_arr.min(), x_arr.max(), 200)
        if force_origin and _fit_slope is not None:
            # Zero-intercept OLS: slope = Σ(x·y) / Σ(x²)
            # SE of predicted value at x_fit: |x_fit| · sqrt(MSE / Σ(x²))
            # where MSE = Σ(residuals²) / (n−1)  [one free parameter: slope]
            y_fit = _fit_slope * x_fit
            ax.plot(x_fit, y_fit, color="black", linewidth=1.2, linestyle="--", zorder=2)
            residuals = y_arr - _fit_slope * x_arr
            dof    = max(n - 1, 1)
            mse    = np.sum(residuals ** 2) / dof
            t_crit = scipy_stats.t.ppf(0.975, df=dof)
            ci = t_crit * np.abs(x_fit) * np.sqrt(mse / np.dot(x_arr, x_arr))
            ax.fill_between(x_fit, y_fit - ci, y_fit + ci, color="black", alpha=0.08)
        else:
            slope, intercept, *_ = scipy_stats.linregress(x_arr, y_arr)
            y_fit  = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, color="black", linewidth=1.2, linestyle="--", zorder=2)
            y_pred    = slope * x_arr + intercept
            residuals = y_arr - y_pred
            se        = np.sqrt(np.sum(residuals ** 2) / max(n - 2, 1))
            x_mean    = x_arr.mean()
            t_crit    = scipy_stats.t.ppf(0.975, df=max(n - 2, 1))
            ci = t_crit * se * np.sqrt(1 / n + (x_fit - x_mean) ** 2
                                       / np.sum((x_arr - x_mean) ** 2))
            ax.fill_between(x_fit, y_fit - ci, y_fit + ci, color="black", alpha=0.08)

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(alpha=0.3)


# ─────────────────────────────────────────────────────────────────────
# SECTION 13a — Correlation vs ratio line plots (existing)
# ─────────────────────────────────────────────────────────────────────

def plot_corr_vs_ratio(
    corr_df: pd.DataFrame,
    metric: str,
    M_values: list[int],
    W_default: int,
    output_dir: str,
    optimal_ratio: float,
) -> None:
    if corr_df.empty or "metric" not in corr_df.columns:
        print(f"  [SKIP] No correlation data (ratio sweep not run)")
        return
    df = corr_df[corr_df["metric"] == metric].copy()
    if df.empty:
        print(f"  [SKIP] No data for metric={metric!r}")
        return

    n_setups = int(df["n"].max()) if not df.empty else 0
    n_cols = min(5, len(M_values))
    n_rows = math.ceil(len(M_values) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.0 * n_cols, 3.2 * n_rows), squeeze=False)

    for idx, M in enumerate(M_values):
        ax  = axes[idx // n_cols][idx % n_cols]
        sub = df[df["M"] == M].sort_values("ratio")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvspan(_DEFAULT_RATIO_RANGE[0], _DEFAULT_RATIO_RANGE[1],
                   color="gray", alpha=0.12, label="Default ratios")
        ax.axvline(optimal_ratio, color="#2ca02c", linewidth=1.2, linestyle="--",
                   alpha=0.85, label=f"Optimal ({optimal_ratio:.2f})")
        ax.plot(sub["ratio"], sub["pearson_r"],
                color="blue", linewidth=1.2, label="Pearson r")
        ax.plot(sub["ratio"], sub["spearman_rho"],
                color="red", linewidth=1.2, linestyle="--", label="Spearman ρ")

        # 3σ significance threshold for Pearson
        if n_setups >= 3:
            r_crit = _pearson_rcrit(n_setups, sigma=3.0)
            ax.axhline( r_crit, color="black", linewidth=0.8, linestyle=":",
                        alpha=0.6, label=f"3σ |r|={r_crit:.2f}")
            ax.axhline(-r_crit, color="black", linewidth=0.8, linestyle=":",
                        alpha=0.6, label="_nolegend_")

        ax.set_title(f"M={M}", fontsize=9)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Area ratio", fontsize=9)
        ax.set_ylabel("Correlation", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=6, loc="lower right")

    for idx in range(len(M_values), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    m_label = {"nc_coverage": "NC Coverage",
               "recall":      f"Recall (W={W_default})",
               "fom":         f"FoM (W={W_default})"}.get(metric, metric)
    fig.suptitle(f"SSD–PMT Correlation vs. Area Ratio — {m_label}", fontsize=12)
    fig.tight_layout()
    fname = f"corr_vs_ratio_{metric}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_scatter_ssd_pmt(
    setups: list[SetupData],
    ratio_factor: float,
    metric: str,
    M_values: list[int],
    W_default: int,
    output_dir: str,
) -> None:
    valid = [s for s in setups if ratio_factor in s.ssd_results]
    if not valid:
        return

    colors  = _colors(len(setups))
    c_valid = [colors[setups.index(s)] for s in valid]
    l_valid = [s.label for s in valid]

    n_cols = min(5, len(M_values))
    n_rows = math.ceil(len(M_values) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 4.0 * n_rows), squeeze=False)

    x_base = "SSD NC fraction" if metric == "nc_coverage" else "SSD Recall"
    y_base = "PMT NC fraction" if metric == "nc_coverage" else "PMT Recall"
    title_metric = "NC Coverage" if metric == "nc_coverage" else f"Recall (W={W_default})"

    for idx, M in enumerate(M_values):
        ax = axes[idx // n_cols][idx % n_cols]
        if metric == "nc_coverage":
            x_arr = np.array([_ssd_nc_frac(s.ssd_results[ratio_factor], M) for s in valid])
            y_arr = np.array([_pmt_nc_frac(s, M) for s in valid])
        else:
            x_arr = np.array([_ssd_recall(s.ssd_results[ratio_factor], M, W_default) for s in valid])
            y_arr = np.array([_pmt_recall(s, M, W_default) for s in valid])
        _scatter_panel(ax, x_arr, y_arr, c_valid, l_valid,
                       f"{x_base} (M={M})", f"{y_base} (M={M})",
                       force_origin=True)
        ax.set_title(f"M={M}", fontsize=9)

    for idx in range(len(M_values), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(f"SSD vs PMT — {title_metric}  |  ratio={ratio_factor:.2f}", fontsize=12)
    fig.tight_layout()
    fname = f"scatter_{metric}_ratio{ratio_factor:.1f}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_w2_comparison(
    setups: list[SetupData],
    ratio_factor: float,
    M_values: list[int],
    W_default: int,
    output_dir: str,
) -> None:
    w2_setups = [s for s in setups if s.w2 is not None and ratio_factor in s.ssd_results]
    if len(w2_setups) < 2:
        print("  [SKIP] W2 comparison: fewer than 2 setups have W2.")
        return

    m_panels = [M for M in [1, 2, 4, 5, 10] if M in M_values] or M_values[:5]
    n_cols = min(5, len(m_panels))
    n_rows = math.ceil(len(m_panels) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 4.0 * n_rows), squeeze=False)
    colors = _colors(len(w2_setups))

    for idx, M in enumerate(m_panels):
        ax     = axes[idx // n_cols][idx % n_cols]
        w2_arr = np.array([s.w2 for s in w2_setups], dtype=float)
        ssd_arr = np.array([_ssd_nc_frac(s.ssd_results[ratio_factor], M) for s in w2_setups])
        pmt_arr = np.array([_pmt_nc_frac(s, M) for s in w2_setups])

        for i, (s, c) in enumerate(zip(w2_setups, colors)):
            ax.scatter([s.w2], [ssd_arr[i]], color=c, marker="^", s=60, zorder=3, alpha=0.9)
            ax.scatter([s.w2], [pmt_arr[i]], color=c, marker="o", s=60, zorder=3, alpha=0.9)
            ax.annotate(s.label, xy=(s.w2, pmt_arr[i]), xytext=(4, 3),
                        textcoords="offset points", fontsize=6, color=c)

        if len(w2_setups) >= 3 and np.std(w2_arr) > 0:
            x_fit = np.linspace(w2_arr.min(), w2_arr.max(), 200)
            for vals, lstyle in [(ssd_arr, "--"), (pmt_arr, "-")]:
                if np.std(vals) > 0:
                    s_l, icpt, *_ = scipy_stats.linregress(w2_arr, vals)
                    ax.plot(x_fit, s_l * x_fit + icpt, color="black",
                            linewidth=1.0, linestyle=lstyle, alpha=0.7)

        ax.set_title(f"M={M}", fontsize=9)
        ax.set_xlabel("W2 (mm)", fontsize=10)
        ax.set_ylabel("NC fraction", fontsize=10)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(handles=[
                Line2D([0], [0], marker="^", color="gray", linestyle="None",
                       markersize=7, label=f"SSD (ratio={ratio_factor:.2f})"),
                Line2D([0], [0], marker="o", color="gray", linestyle="None",
                       markersize=7, label="PMT"),
            ], fontsize=8)

    for idx in range(len(m_panels), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(f"W2 vs NC Fraction — SSD vs PMT  |  ratio={ratio_factor:.2f}", fontsize=12)
    fig.tight_layout()
    fname = "w2_coverage_comparison.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 13b — FoM scatter: SSD vs PMT at optimal ratio
# ─────────────────────────────────────────────────────────────────────

def plot_scatter_ssd_pmt_fom(
    setups: list[SetupData],
    ratio_factor: float,
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: str,
) -> None:
    """Scatter of SSD-FoM vs PMT-FoM; (M,W) chosen per setup to maximise PMT FoM."""
    valid = [s for s in setups if ratio_factor in s.ssd_results]
    if not valid:
        return

    colors  = _colors(len(setups))
    c_valid = [colors[setups.index(s)] for s in valid]
    x_arr, y_arr, labels, mw_labels = [], [], [], []

    for s in valid:
        M_b, W_b = _pmt_best_mw(s, M_values, W_values, total_primaries)
        x_arr.append(_ssd_fom(s.ssd_results[ratio_factor], M_b, W_b, total_primaries))
        y_arr.append(_pmt_fom(s, M_b, W_b, total_primaries))
        labels.append(s.label)
        mw_labels.append(f"M{M_b}W{W_b}")

    x_arr = np.array(x_arr, dtype=float)
    y_arr = np.array(y_arr, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter_panel(ax, x_arr, y_arr, c_valid, labels,
                   f"SSD FoM  (ratio={ratio_factor:.2f})", "PMT FoM",
                   force_origin=True)
    for x, y, mw in zip(x_arr, y_arr, mw_labels):
        if np.isfinite(x) and np.isfinite(y):
            ax.annotate(mw, xy=(x, y), xytext=(4, -9),
                        textcoords="offset points", fontsize=6, color="dimgray")

    ax.set_title(
        f"SSD vs PMT — Figure of Merit  |  ratio={ratio_factor:.2f}\n"
        "(M,W) chosen per setup to maximise PMT FoM", fontsize=11,
    )
    fig.tight_layout()
    fname = f"scatter_fom_ratio{ratio_factor:.1f}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 13c — W2 Spearman for FoM / Recall cross-plots (existing)
# ─────────────────────────────────────────────────────────────────────

def plot_w2_spearman_fom(
    setups: list[SetupData],
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    optimal_ratio: float,
    output_dir: str,
) -> None:
    w2_setups = [s for s in setups if s.w2 is not None and optimal_ratio in s.ssd_results]
    if len(w2_setups) < 3:
        print("  [SKIP] w2_spearman_fom: fewer than 3 setups have W2 + SSD results.")
        return

    w2_arr = np.array([s.w2 for s in w2_setups], dtype=float)
    rho_pmt, rho_ssd, p_pmt, p_ssd = [], [], [], []

    for M in M_values:
        pmt_fom_arr, ssd_fom_arr = [], []
        for s in w2_setups:
            W_best = _pmt_best_w_at_m(s, M, W_values, total_primaries)
            pmt_fom_arr.append(_pmt_fom(s, M, W_best, total_primaries))
            ssd_fom_arr.append(_ssd_fom(s.ssd_results[optimal_ratio], M, W_best, total_primaries))

        pmt_arr = np.array(pmt_fom_arr, dtype=float)
        ssd_arr = np.array(ssd_fom_arr, dtype=float)
        mask_p  = np.isfinite(pmt_arr) & np.isfinite(w2_arr)
        mask_s  = np.isfinite(ssd_arr) & np.isfinite(w2_arr)
        r1, p1  = _safe_spearmanr(w2_arr[mask_p], pmt_arr[mask_p]) if mask_p.sum() >= 3 else (float("nan"), float("nan"))
        r2, p2  = _safe_spearmanr(w2_arr[mask_s], ssd_arr[mask_s]) if mask_s.sum() >= 3 else (float("nan"), float("nan"))
        rho_pmt.append(r1); p_pmt.append(p1)
        rho_ssd.append(r2); p_ssd.append(p2)

    x = np.array(M_values)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhspan(-0.3, 0.3, color="gray", alpha=0.08, label="weak |ρ|<0.3")

    for rhos, ps, label, color, marker in [
        (rho_pmt, p_pmt, "PMT FoM (best W per setup)", "#1f77b4", "o"),
        (rho_ssd, p_ssd, f"SSD FoM (same W, ratio={optimal_ratio:.2f})", "#d62728", "s"),
    ]:
        rhos = np.array(rhos, dtype=float)
        ps   = np.array(ps,   dtype=float)
        sig  = ps < _P_3SIGMA
        ax.plot(x, rhos, color=color, linewidth=1.5, label=label)
        finite = np.isfinite(rhos)
        if (sig & finite).any():
            ax.scatter(x[sig & finite], rhos[sig & finite], color=color,
                       s=60, marker=marker, zorder=4, label=f"{label} (3σ)")
        if (~sig & finite).any():
            ax.scatter(x[~sig & finite], rhos[~sig & finite],
                       facecolors="none", edgecolors=color,
                       s=60, marker=marker, linewidth=1.2, zorder=4)

    ax.set_xlabel("Multiplicity threshold M", fontsize=13)
    ax.set_ylabel("Spearman ρ  (W2 vs FoM)", fontsize=13)
    ax.set_title("Spearman Correlation: W2 vs FoM — PMT vs SSD\n"
                 "(filled = 3σ; W = argmax PMT FoM at that M)", fontsize=13)
    ax.set_xticks(M_values)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = "w2_spearman_fom.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_ssd_nc_vs_pmt_recall_at_best_fom(
    setups: list[SetupData],
    ratio_factor: float,
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: str,
) -> None:
    valid = [s for s in setups if ratio_factor in s.ssd_results]
    if not valid:
        return

    colors  = _colors(len(setups))
    c_valid = [colors[setups.index(s)] for s in valid]
    x_arr, y_arr, labels, mw_labels = [], [], [], []

    for s in valid:
        M_b, W_b = _pmt_best_mw(s, M_values, W_values, total_primaries)
        x_arr.append(_ssd_nc_frac(s.ssd_results[ratio_factor], 1))
        y_arr.append(_pmt_recall(s, M_b, W_b))
        labels.append(s.label)
        mw_labels.append(f"M{M_b}W{W_b}")

    x_arr = np.array(x_arr, dtype=float)
    y_arr = np.array(y_arr, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter_panel(ax, x_arr, y_arr, c_valid, labels,
                   f"SSD NC fraction  (M=1, ratio={ratio_factor:.2f})",
                   "PMT Recall at FoM-optimal (M,W)")
    for x, y, mw in zip(x_arr, y_arr, mw_labels):
        if np.isfinite(x) and np.isfinite(y):
            ax.annotate(mw, xy=(x, y), xytext=(4, -9),
                        textcoords="offset points", fontsize=6, color="dimgray")

    ax.set_title(f"SSD NC fraction (M=1) vs PMT Recall at FoM-optimal (M,W)\n"
                 f"ratio={ratio_factor:.2f}", fontsize=11)
    fig.tight_layout()
    fname = f"scatter_ssd_nc_vs_pmt_recall_best_fom_ratio{ratio_factor:.1f}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_ssd_recall_vs_pmt_recall_at_best_fom(
    setups: list[SetupData],
    ratio_factor: float,
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: str,
) -> None:
    valid = [s for s in setups if ratio_factor in s.ssd_results]
    if not valid:
        return

    colors  = _colors(len(setups))
    c_valid = [colors[setups.index(s)] for s in valid]
    x_arr, y_arr, labels, mw_labels = [], [], [], []

    for s in valid:
        M_b, W_b = _pmt_best_mw(s, M_values, W_values, total_primaries)
        x_arr.append(_ssd_recall(s.ssd_results[ratio_factor], 1, 1))
        y_arr.append(_pmt_recall(s, M_b, W_b))
        labels.append(s.label)
        mw_labels.append(f"M{M_b}W{W_b}")

    x_arr = np.array(x_arr, dtype=float)
    y_arr = np.array(y_arr, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter_panel(ax, x_arr, y_arr, c_valid, labels,
                   f"SSD Recall  (M=1, W=1, ratio={ratio_factor:.2f})",
                   "PMT Recall at FoM-optimal (M,W)")
    for x, y, mw in zip(x_arr, y_arr, mw_labels):
        if np.isfinite(x) and np.isfinite(y):
            ax.annotate(mw, xy=(x, y), xytext=(4, -9),
                        textcoords="offset points", fontsize=6, color="dimgray")

    ax.set_title(f"SSD Recall (M=1,W=1) vs PMT Recall at FoM-optimal (M,W)\n"
                 f"ratio={ratio_factor:.2f}", fontsize=11)
    fig.tight_layout()
    fname = f"scatter_ssd_recall_vs_pmt_recall_best_fom_ratio{ratio_factor:.1f}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 13d — NC Ranking Consistency (Spearman ρ + rank scatter)
# ─────────────────────────────────────────────────────────────────────

def plot_nc_ranking_consistency(
    setups: list[SetupData],
    ratio_factor: float,
    M_values: list[int],
    output_dir: str,
) -> None:
    """Two plots showing whether SSD preserves the PMT setup ranking by NC coverage.

    Plot A (ranking_nc_spearman.png):
        Spearman ρ(PMT ranks, SSD ranks) vs M.  High ρ means SSD correctly
        orders the setups by NC coverage.  Filled markers = 3σ significant.

    Plot B (ranking_nc_scatter.png):
        For representative M values, scatter of PMT rank vs SSD rank (one
        point per setup).  Diagonal = perfect agreement.  Points off-diagonal
        indicate rank inversions.
    """
    valid = [s for s in setups if ratio_factor in s.ssd_results]
    n     = len(valid)
    if n < 3:
        print("  [SKIP] ranking_nc: fewer than 3 setups at this ratio.")
        return

    colors  = _colors(len(setups))
    c_valid = [colors[setups.index(s)] for s in valid]

    # Compute per-M Spearman ρ between PMT and SSD NC-coverage ranks
    rho_vals, p_vals = [], []
    pmt_ranks_per_m  = {}
    ssd_ranks_per_m  = {}

    for M in M_values:
        pmt_nc = np.array([_pmt_nc_frac(s, M) for s in valid])
        ssd_nc = np.array([_ssd_nc_frac(s.ssd_results[ratio_factor], M) for s in valid])
        # rank 1 = highest coverage; use -values so rank 1 = max
        pmt_r  = scipy_stats.rankdata(-pmt_nc)
        ssd_r  = scipy_stats.rankdata(-ssd_nc)
        pmt_ranks_per_m[M] = pmt_r
        ssd_ranks_per_m[M] = ssd_r
        rho, p = _safe_spearmanr(pmt_r, ssd_r)
        rho_vals.append(rho)
        p_vals.append(p)

    rho_arr = np.array(rho_vals, dtype=float)
    p_arr   = np.array(p_vals,   dtype=float)
    x       = np.array(M_values)
    sig     = p_arr < _P_3SIGMA

    # ── Plot A: Spearman ρ vs M ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle=":")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.plot(x, rho_arr, color="#1f77b4", linewidth=1.8,
            label="ρ(PMT ranks, SSD ranks)")
    ax.scatter(x[sig & np.isfinite(rho_arr)],
               rho_arr[sig & np.isfinite(rho_arr)],
               color="#1f77b4", s=65, marker="o", zorder=4, label="3σ significant")
    ax.scatter(x[~sig & np.isfinite(rho_arr)],
               rho_arr[~sig & np.isfinite(rho_arr)],
               facecolors="none", edgecolors="#1f77b4",
               s=65, marker="o", linewidth=1.3, zorder=4)
    ax.set_xlabel("Multiplicity threshold M", fontsize=13)
    ax.set_ylabel("Spearman ρ  (PMT rank vs SSD rank)", fontsize=13)
    ax.set_title(
        f"NC Coverage Ranking Consistency — PMT vs SSD  |  ratio={ratio_factor:.2f}\n"
        f"(ρ=1: SSD perfectly preserves PMT ranking; filled = 3σ significant)",
        fontsize=13,
    )
    ax.set_xticks(M_values)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname_a = "ranking_nc_spearman.png"
    fig.savefig(os.path.join(output_dir, fname_a), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname_a}")

    # ── Plot B: rank scatter for representative M values ─────────────
    panel_ms = [M for M in [1, 2, 4, 6, 10] if M in M_values]
    if not panel_ms:
        panel_ms = M_values[:5]

    n_cols = min(5, len(panel_ms))
    n_rows = math.ceil(len(panel_ms) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 4.5 * n_rows), squeeze=False)

    for pi, M in enumerate(panel_ms):
        ax      = axes[pi // n_cols][pi % n_cols]
        pmt_r   = pmt_ranks_per_m[M]
        ssd_r   = ssd_ranks_per_m[M]
        rho_val = rho_arr[M_values.index(M)]
        p_val   = p_arr[M_values.index(M)]

        # y=x reference (perfect agreement)
        ax.plot([1, n], [1, n], color="gray", linewidth=0.8, linestyle="--",
                alpha=0.5, zorder=1)

        for i, (s, c) in enumerate(zip(valid, c_valid)):
            ax.scatter([pmt_r[i]], [ssd_r[i]], color=c, s=55, zorder=3)
            ax.annotate(s.label, xy=(pmt_r[i], ssd_r[i]), xytext=(3, 3),
                        textcoords="offset points", fontsize=6, color=c)

        # Count inversions (pairs where PMT and SSD rank order disagrees)
        n_inv = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (pmt_r[i] < pmt_r[j]) != (ssd_r[i] < ssd_r[j]):
                    n_inv += 1

        ax.text(0.97, 0.03,
                f"ρ={rho_val:+.3f}  p={p_val:.3g}\ninversions={n_inv}/{n*(n-1)//2}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.85))

        ax.set_title(f"M={M}", fontsize=11)
        ax.set_xlabel("PMT rank (1=best)", fontsize=10)
        ax.set_ylabel("SSD rank (1=best)", fontsize=10)
        ax.set_xticks(range(1, n + 1))
        ax.set_yticks(range(1, n + 1))
        ax.grid(alpha=0.3)

    for pi in range(len(panel_ms), n_rows * n_cols):
        axes[pi // n_cols][pi % n_cols].set_visible(False)

    fig.suptitle(
        f"Setup Ranking Comparison — PMT vs SSD (NC Coverage)\n"
        f"ratio={ratio_factor:.2f}  |  diagonal = perfect agreement",
        fontsize=13,
    )
    fig.tight_layout()
    fname_b = "ranking_nc_scatter.png"
    fig.savefig(os.path.join(output_dir, fname_b), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname_b}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 13e — Ranking Heatmap (setup × M, PMT vs SSD vs Δ)
# ─────────────────────────────────────────────────────────────────────

def plot_ranking_heatmap(
    setups: list[SetupData],
    ratio_factor: float,
    M_values: list[int],
    output_dir: str,
) -> None:
    """3-panel heatmap: PMT ranks | SSD ranks | Δ(SSD−PMT) per (setup × M).

    Rows = setups sorted by PMT rank at M=1 (best first).
    Cells annotated with rank number.  Saved as ranking_heatmap.png.
    """
    valid = [s for s in setups if ratio_factor in s.ssd_results]
    n     = len(valid)
    if n < 2:
        print("  [SKIP] ranking_heatmap: fewer than 2 setups.")
        return

    # Sort setups by PMT NC fraction at M=1 (best first)
    pmt_m1 = [_pmt_nc_frac(s, M_values[0]) for s in valid]
    order   = np.argsort(pmt_m1)[::-1]
    valid_s = [valid[i] for i in order]

    pmt_mat = np.zeros((n, len(M_values)), dtype=float)
    ssd_mat = np.zeros((n, len(M_values)), dtype=float)

    for j, M in enumerate(M_values):
        pmt_nc = np.array([_pmt_nc_frac(s, M) for s in valid_s])
        ssd_nc = np.array([_ssd_nc_frac(s.ssd_results[ratio_factor], M) for s in valid_s])
        pmt_mat[:, j] = scipy_stats.rankdata(-pmt_nc)
        ssd_mat[:, j] = scipy_stats.rankdata(-ssd_nc)

    delta_mat = ssd_mat - pmt_mat  # positive = SSD ranks worse than PMT

    row_labels = [s.label for s in valid_s]
    col_labels = [f"M={M}" for M in M_values]

    fig, axes = plt.subplots(1, 3, figsize=(max(18, len(M_values) * 1.2), max(6, n * 0.7)))

    for ax, mat, title, cmap, vmin, vmax in [
        (axes[0], pmt_mat,   "PMT Rank",      "Blues",   1, n),
        (axes[1], ssd_mat,   "SSD Rank",      "Blues",   1, n),
        (axes[2], delta_mat, "Δ (SSD − PMT)", "RdBu_r", -n, n),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        for i in range(n):
            for j in range(len(M_values)):
                val = mat[i, j]
                tc  = "white" if (cmap == "Blues" and val > n * 0.7) else "black"
                ax.text(j, i, f"{val:+.0f}" if cmap == "RdBu_r" else f"{int(val)}",
                        ha="center", va="center", fontsize=max(6, 9 - n // 4), color=tc,
                        fontweight="bold")
        ax.set_xticks(range(len(M_values)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(title, fontsize=13)

    fig.suptitle(
        f"Setup Ranking Heatmap — NC Coverage  |  ratio={ratio_factor:.2f}\n"
        f"(rank 1 = highest coverage; Δ > 0 means SSD ranks that setup lower)",
        fontsize=13,
    )
    fig.tight_layout()
    fname = "ranking_heatmap.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 13f — FoM comparison bars: PMT vs SSD per setup
# ─────────────────────────────────────────────────────────────────────

def plot_fom_bars_pmt_vs_ssd(
    setups: list[SetupData],
    ratio_factor: float,
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: str,
) -> None:
    """Grouped horizontal bars: PMT best-FoM vs SSD FoM at PMT's best (M,W).

    Each row = one setup.  The left sub-plot shows absolute FoM values;
    the right sub-plot is a scatter (SSD vs PMT) with a diagonal reference.
    Saved as fom_bars_pmt_vs_ssd.png.
    """
    valid = [s for s in setups if ratio_factor in s.ssd_results]
    n     = len(valid)
    if n < 2:
        print("  [SKIP] fom_bars_pmt_vs_ssd: fewer than 2 setups.")
        return

    colors  = _colors(len(setups))
    c_valid = [colors[setups.index(s)] for s in valid]

    pmt_foms, ssd_foms, mw_labels = [], [], []
    for s in valid:
        M_b, W_b = _pmt_best_mw(s, M_values, W_values, total_primaries)
        pmt_foms.append(_pmt_fom(s, M_b, W_b, total_primaries))
        ssd_foms.append(_ssd_fom(s.ssd_results[ratio_factor], M_b, W_b, total_primaries))
        mw_labels.append(f"M{M_b}W{W_b}")

    pmt_arr = np.array(pmt_foms, dtype=float)
    ssd_arr = np.array(ssd_foms, dtype=float)
    labels  = [s.label for s in valid]

    fig, (ax_bar, ax_sc) = plt.subplots(1, 2, figsize=(16, max(5, n * 0.6)))

    # ── Left: grouped horizontal bars ────────────────────────────────
    y      = np.arange(n)
    h      = 0.35
    bars_p = ax_bar.barh(y - h / 2, pmt_arr, h, label="PMT (best M,W)", color=c_valid, alpha=0.9)
    bars_s = ax_bar.barh(y + h / 2, ssd_arr, h, label="SSD (PMT's best M,W)", color=c_valid,
                          alpha=0.5, hatch="//")

    finite_max = max((v for v in np.concatenate([pmt_arr, ssd_arr]) if np.isfinite(v)), default=1.0)
    for bar, fom, mw in zip(bars_p, pmt_arr, mw_labels):
        if np.isfinite(fom):
            ax_bar.text(fom + 0.01 * finite_max, bar.get_y() + bar.get_height() / 2,
                        f"{fom:.4g} [{mw}]", va="center", fontsize=7)

    for bar, fom in zip(bars_s, ssd_arr):
        if np.isfinite(fom):
            ax_bar.text(fom + 0.01 * finite_max, bar.get_y() + bar.get_height() / 2,
                        f"{fom:.4g}", va="center", fontsize=7, color="dimgray")

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=11)
    ax_bar.set_xlabel("Figure of Merit", fontsize=13)
    ax_bar.set_title("FoM: PMT (best M,W) vs SSD (at PMT's best M,W)", fontsize=12)
    ax_bar.legend(fontsize=10)
    ax_bar.grid(True, axis="x", alpha=0.3)

    # ── Right: scatter with diagonal ─────────────────────────────────
    _scatter_panel(ax_sc, ssd_arr, pmt_arr, c_valid, labels,
                   f"SSD FoM  (ratio={ratio_factor:.2f})", "PMT FoM",
                   force_origin=True)
    ax_sc.set_title("SSD vs PMT FoM", fontsize=12)

    fig.suptitle(
        f"FoM Comparison — PMT vs SSD  |  ratio={ratio_factor:.2f}\n"
        f"(M,W) chosen to maximise PMT FoM; same (M,W) used for SSD",
        fontsize=13,
    )
    fig.tight_layout()
    fname = "fom_bars_pmt_vs_ssd.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 13g — FoM scatter per M (best W per M for each side)
# ─────────────────────────────────────────────────────────────────────

def plot_fom_scatter_per_m(
    setups: list[SetupData],
    ratio_factor: float,
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: str,
) -> None:
    """Grid of scatter panels: SSD best-W FoM vs PMT best-W FoM, one panel per M.

    For each M, each side independently optimises W.  This shows whether
    the FoM ordering across setups is preserved by SSD at every M.
    Saved as fom_scatter_per_m.png.
    """
    valid = [s for s in setups if ratio_factor in s.ssd_results]
    n     = len(valid)
    if n < 2:
        print("  [SKIP] fom_scatter_per_m: fewer than 2 setups.")
        return

    colors  = _colors(len(setups))
    c_valid = [colors[setups.index(s)] for s in valid]
    labels  = [s.label for s in valid]

    n_cols = min(5, len(M_values))
    n_rows = math.ceil(len(M_values) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 4.2 * n_rows), squeeze=False)

    for idx, M in enumerate(M_values):
        ax = axes[idx // n_cols][idx % n_cols]

        pmt_arr = np.array([
            max((_pmt_fom(s, M, W, total_primaries) for W in W_values
                 if np.isfinite(_pmt_fom(s, M, W, total_primaries))),
                default=float("nan"))
            for s in valid
        ])
        ssd_arr = np.array([
            max((_ssd_fom(s.ssd_results[ratio_factor], M, W, total_primaries)
                 for W in W_values
                 if np.isfinite(_ssd_fom(s.ssd_results[ratio_factor], M, W, total_primaries))),
                default=float("nan"))
            for s in valid
        ])

        mask = np.isfinite(pmt_arr) & np.isfinite(ssd_arr)
        if mask.sum() < 2:
            ax.set_visible(False)
            continue

        _scatter_panel(ax, ssd_arr, pmt_arr, c_valid, labels,
                       f"SSD FoM (best W, ratio={ratio_factor:.2f})",
                       "PMT FoM (best W)")
        ax.set_title(f"M={M}", fontsize=11)

    for idx in range(len(M_values), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(
        f"SSD vs PMT — FoM per M  (each side optimises W independently)\n"
        f"ratio={ratio_factor:.2f}",
        fontsize=13,
    )
    fig.tight_layout()
    fname = "fom_scatter_per_m.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 13h — FoM M ≥ min_M comparison (two methods)
# ─────────────────────────────────────────────────────────────────────

def plot_fom_mge_comparison(
    setups: list[SetupData],
    ratio_factor: float,
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    min_M: int,
    output_dir: str,
) -> None:
    """FoM comparison under M ≥ min_M constraint — two methods in one figure.

    Panel A (PMT-anchor): PMT's best (M≥min_M, W) applied to both sides.
        Tests whether SSD agrees with PMT at the same operating point.
    Panel B (independent): each side finds its own best (M≥min_M, W).
        Reveals whether the two simulators agree on the optimal operating point.
    Saved as fom_mge{min_M}_comparison.png.
    """
    eligible_M = [M for M in M_values if M >= min_M]
    if not eligible_M:
        print(f"  [SKIP] fom_mge{min_M}: no M ≥ {min_M} in M_values.")
        return

    valid = [s for s in setups if ratio_factor in s.ssd_results]
    n     = len(valid)
    if n < 2:
        print(f"  [SKIP] fom_mge{min_M}: fewer than 2 setups.")
        return

    colors  = _colors(len(setups))
    c_valid = [colors[setups.index(s)] for s in valid]
    labels  = [s.label for s in valid]

    # Panel A: PMT-anchor
    pmt_anch, ssd_anch, mw_a = [], [], []
    for s in valid:
        M_b, W_b = _pmt_best_mw(s, M_values, W_values, total_primaries, min_M=min_M)
        pmt_anch.append(_pmt_fom(s, M_b, W_b, total_primaries))
        ssd_anch.append(_ssd_fom(s.ssd_results[ratio_factor], M_b, W_b, total_primaries))
        mw_a.append(f"M{M_b}W{W_b}")

    # Panel B: independent
    pmt_ind, ssd_ind, mw_p, mw_s = [], [], [], []
    for s in valid:
        M_bp, W_bp = _pmt_best_mw(s, M_values, W_values, total_primaries, min_M=min_M)
        M_bs, W_bs = _ssd_best_mw(s.ssd_results[ratio_factor], M_values, W_values,
                                   total_primaries, min_M=min_M)
        pmt_ind.append(_pmt_fom(s, M_bp, W_bp, total_primaries))
        ssd_ind.append(_ssd_fom(s.ssd_results[ratio_factor], M_bs, W_bs, total_primaries))
        mw_p.append(f"M{M_bp}W{W_bp}")
        mw_s.append(f"M{M_bs}W{W_bs}")

    pmt_a = np.array(pmt_anch, dtype=float)
    ssd_a = np.array(ssd_anch, dtype=float)
    pmt_i = np.array(pmt_ind,  dtype=float)
    ssd_i = np.array(ssd_ind,  dtype=float)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A
    _scatter_panel(ax_a, ssd_a, pmt_a, c_valid, labels,
                   f"SSD FoM  (ratio={ratio_factor:.2f})", "PMT FoM",
                   force_origin=True)
    for x, y, mw in zip(ssd_a, pmt_a, mw_a):
        if np.isfinite(x) and np.isfinite(y):
            ax_a.annotate(mw, xy=(x, y), xytext=(4, -9),
                          textcoords="offset points", fontsize=6, color="dimgray")
    ax_a.set_title(
        f"Panel A — PMT-anchor\n(M,W) = PMT's best with M≥{min_M}, applied to both",
        fontsize=12,
    )

    # Panel B
    _scatter_panel(ax_b, ssd_i, pmt_i, c_valid, labels,
                   f"SSD FoM  (SSD's own best M≥{min_M})", f"PMT FoM  (PMT's best M≥{min_M})")
    for x, y, mp, ms in zip(ssd_i, pmt_i, mw_p, mw_s):
        if np.isfinite(x) and np.isfinite(y):
            ax_b.annotate(f"PMT:{mp}\nSSD:{ms}", xy=(x, y), xytext=(4, -12),
                          textcoords="offset points", fontsize=5, color="dimgray")
    ax_b.set_title(
        f"Panel B — Independent\neach side finds its own best (M≥{min_M}, W)",
        fontsize=12,
    )

    fig.suptitle(
        f"FoM Comparison — M ≥ {min_M} Constraint  |  ratio={ratio_factor:.2f}",
        fontsize=14,
    )
    fig.tight_layout()
    fname = f"fom_mge{min_M}_comparison.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 13i — Recall comparison at PMT's best (M,W)
# ─────────────────────────────────────────────────────────────────────

def plot_recall_comparison(
    setups: list[SetupData],
    ratio_factor: float,
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: str,
) -> None:
    """Grouped horizontal bars + scatter comparing PMT vs SSD Recall.

    For each setup, Recall is evaluated at the (M,W) that maximises PMT FoM.
    Saved as recall_pmt_vs_ssd.png.
    """
    valid = [s for s in setups if ratio_factor in s.ssd_results]
    n     = len(valid)
    if n < 2:
        print("  [SKIP] recall_comparison: fewer than 2 setups.")
        return

    colors  = _colors(len(setups))
    c_valid = [colors[setups.index(s)] for s in valid]
    labels  = [s.label for s in valid]

    pmt_recs, ssd_recs, mw_labels = [], [], []
    for s in valid:
        M_b, W_b = _pmt_best_mw(s, M_values, W_values, total_primaries)
        pmt_recs.append(_pmt_recall(s, M_b, W_b))
        ssd_recs.append(_ssd_recall(s.ssd_results[ratio_factor], M_b, W_b))
        mw_labels.append(f"M{M_b}W{W_b}")

    pmt_arr = np.array(pmt_recs, dtype=float)
    ssd_arr = np.array(ssd_recs, dtype=float)

    fig, (ax_bar, ax_sc) = plt.subplots(1, 2, figsize=(16, max(5, n * 0.6)))

    # ── Left: grouped horizontal bars ────────────────────────────────
    y      = np.arange(n)
    h      = 0.35
    bars_p = ax_bar.barh(y - h / 2, pmt_arr, h, label="PMT Recall", color=c_valid, alpha=0.9)
    bars_s = ax_bar.barh(y + h / 2, ssd_arr, h, label="SSD Recall", color=c_valid,
                          alpha=0.5, hatch="//")

    for bar, rec, mw in zip(bars_p, pmt_arr, mw_labels):
        ax_bar.text(rec + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{rec*100:.1f}% [{mw}]", va="center", fontsize=7)
    for bar, rec in zip(bars_s, ssd_arr):
        ax_bar.text(rec + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{rec*100:.1f}%", va="center", fontsize=7, color="dimgray")

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=11)
    ax_bar.set_xlabel("Recall", fontsize=13)
    ax_bar.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax_bar.set_title("Recall: PMT vs SSD at PMT's best (M,W)", fontsize=12)
    ax_bar.legend(fontsize=10)
    ax_bar.grid(True, axis="x", alpha=0.3)

    # ── Right: scatter ────────────────────────────────────────────────
    _scatter_panel(ax_sc, ssd_arr, pmt_arr, c_valid, labels,
                   f"SSD Recall  (ratio={ratio_factor:.2f})", "PMT Recall",
                   force_origin=True)
    ax_sc.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax_sc.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax_sc.set_title("SSD vs PMT Recall", fontsize=12)

    fig.suptitle(
        f"Recall Comparison — PMT vs SSD  |  ratio={ratio_factor:.2f}\n"
        f"(M,W) chosen to maximise PMT FoM per setup",
        fontsize=13,
    )
    fig.tight_layout()
    fname = "recall_pmt_vs_ssd.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 13j — Best-(M,W) agreement table
# ─────────────────────────────────────────────────────────────────────

def plot_best_mw_agreement(
    setups: list[SetupData],
    ratio_factor: float,
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    min_M: int,
    output_dir: str,
) -> None:
    """Table figure: best (M,W) chosen by PMT vs SSD per setup.

    Columns: unconstrained best and M≥min_M constrained best, for both
    PMT and SSD.  Cells colored green (agree) / red (disagree).
    Saved as best_mw_agreement.png.
    """
    valid = [s for s in setups if ratio_factor in s.ssd_results]
    n     = len(valid)
    if n < 1:
        print("  [SKIP] best_mw_agreement: no setups.")
        return

    rows_data = []
    agree_uncon, agree_con = [], []

    for s in valid:
        M_pu, W_pu = _pmt_best_mw(s, M_values, W_values, total_primaries)
        M_su, W_su = _ssd_best_mw(s.ssd_results[ratio_factor], M_values, W_values, total_primaries)
        M_pc, W_pc = _pmt_best_mw(s, M_values, W_values, total_primaries, min_M=min_M)
        M_sc, W_sc = _ssd_best_mw(s.ssd_results[ratio_factor], M_values, W_values,
                                   total_primaries, min_M=min_M)
        agr_u = (M_pu == M_su and W_pu == W_su)
        agr_c = (M_pc == M_sc and W_pc == W_sc)
        agree_uncon.append(agr_u)
        agree_con.append(agr_c)
        rows_data.append([
            s.label,
            f"M{M_pu}W{W_pu}", f"M{M_su}W{W_su}", "✓" if agr_u else "✗",
            f"M{M_pc}W{W_pc}", f"M{M_sc}W{W_sc}", "✓" if agr_c else "✗",
        ])

    col_labels = [
        "Setup",
        "PMT best\n(unconstrained)", "SSD best\n(unconstrained)", "Agree?",
        f"PMT best\n(M≥{min_M})", f"SSD best\n(M≥{min_M})", "Agree?",
    ]

    fig, ax = plt.subplots(figsize=(max(14, n * 1.0), max(4, n * 0.6 + 2)))
    ax.set_axis_off()

    tbl = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(range(len(col_labels)))

    # Color agree/disagree cells
    for i, (agr_u, agr_c) in enumerate(zip(agree_uncon, agree_con)):
        row_idx = i + 1  # +1 for header row
        for col_idx, agree in [(3, agr_u), (6, agr_c)]:
            color = "#c8f0c8" if agree else "#f0c8c8"
            tbl[row_idx, col_idx].set_facecolor(color)

    n_agree_u = sum(agree_uncon)
    n_agree_c = sum(agree_con)
    ax.set_title(
        f"Best (M,W) Agreement — PMT vs SSD  |  ratio={ratio_factor:.2f}\n"
        f"Unconstrained: {n_agree_u}/{n} agree   |   "
        f"M≥{min_M} constrained: {n_agree_c}/{n} agree",
        fontsize=13, pad=20,
    )
    fig.tight_layout()
    fname = "best_mw_agreement.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 14 — Text and CSV output
# ─────────────────────────────────────────────────────────────────────

def write_summary(
    corr_df: pd.DataFrame,
    per_layer_corr_df: Optional[pd.DataFrame],
    per_layer_optima: Optional[dict[str, float]],
    optimal_ratio: float,
    setups: list[SetupData],
    M_values: list[int],
    W_values: list[int],
    W_default: int,
    total_primaries: int,
    output_dir: str,
    ratio_min: float = 1.0,
    ratio_max: float = 10.0,
    ratio_step: float = 0.1,
    min_M_fom: int = 6,
    area_ratios: Optional[dict[str, float]] = None,
) -> None:
    import datetime

    def _hr(char: str = "─", width: int = 72) -> str:
        return char * width

    def _fmt_p(p: float) -> str:
        if math.isnan(p):
            return "   n/a  "
        return f"<0.001  " if p < 0.001 else f"{p:.4f}  "

    def _fmt_r(r: float) -> str:
        return "   nan " if math.isnan(r) else f"{r:+.4f}"

    L = []
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    L += ["SSD–PMT Correlation Analysis — Full Summary", "=" * 72,
          f"Generated : {ts}", ""]

    # ── 1. Run metadata ───────────────────────────────────────────────
    runtime_h  = total_primaries / MUSUN_RATE if MUSUN_RATE > 0 else 0.0
    runtime_yr = runtime_h / (24 * 365.25)
    n_runs     = total_primaries // max(MUONS_PER_RUN_DIR, 1)
    if area_ratios is not None:
        _ratio_mode_lines: list[str] = ["  Ratio mode         : fixed (no sweep)"]
        for _lyr, _r in area_ratios.items():
            _ratio_mode_lines.append(f"  {_lyr:>6} ratio     : {_r}")
    else:
        _ratio_mode_lines = [
            f"  Ratio sweep        : [{ratio_min:.2f}, {ratio_max:.2f}]  step={ratio_step:.2f}"
        ]
    L += [
        "1. SIMULATION OVERVIEW", _hr(),
        f"  Setups analysed    : {len(setups)}",
        f"  Total primaries    : {total_primaries:,}  ({n_runs} runs × {MUONS_PER_RUN_DIR:,})",
        f"  Simulated livetime : {runtime_h:,.0f} h  =  {runtime_yr:.2f} yr  (@ {MUSUN_RATE:.0f} µ/h)",
        *_ratio_mode_lines,
        f"  M range            : 1 – {max(M_values)}",
        f"  W range            : 1 – {max(W_values)}",
        f"  W_default (recall) : {W_default}",
        f"  min_M_fom          : {min_M_fom}",
        "",
    ]

    # ── 2. Setup overview ─────────────────────────────────────────────
    L += [
        "2. SETUP OVERVIEW", _hr(),
        f"  {'Label':<30} {'W2 (mm)':>10}  {'PMT best FoM':>14}  {'at (M,W)':>9}",
        f"  {_hr('-', 67)}",
    ]
    for s in setups:
        w2_str  = f"{s.w2:.1f}" if s.w2 is not None else "N/A"
        M_b, W_b = _pmt_best_mw(s, M_values, W_values, total_primaries)
        best_fom = _pmt_fom(s, M_b, W_b, total_primaries)
        fom_str  = f"{best_fom:.5g}" if math.isfinite(best_fom) else "N/A"
        L.append(f"  {s.label:<30} {w2_str:>10}  {fom_str:>14}  {'M'+str(M_b)+'W'+str(W_b):>9}")
    L.append("")

    # ── 3. Fixed / optimal ratios ────────────────────────────────────
    if area_ratios is not None:
        L += ["3. FIXED AREA RATIOS", _hr()]
        for _lyr, _r in area_ratios.items():
            L.append(f"  {_lyr:>6}: {_r}")
        L += [f"  Nominal key (mean): {optimal_ratio:.4f}", ""]
    else:
        L += [
            "3. OPTIMAL AREA RATIOS", _hr(),
            f"  Global optimal ratio : {optimal_ratio:.4f}",
            "  (Determined by max mean|Pearson r| for nc_coverage at M=1..4)",
            "",
        ]
        boundary_warnings: list[str] = []
        if abs(optimal_ratio - ratio_min) < 1e-9:
            boundary_warnings.append(f"  *** GLOBAL optimum hit the LOWER boundary ({ratio_min:.2f}). ***")
        if abs(optimal_ratio - ratio_max) < 1e-9:
            boundary_warnings.append(f"  *** GLOBAL optimum hit the UPPER boundary ({ratio_max:.2f}). ***")
        if per_layer_optima:
            L.append("  Per-layer optimal ratios:")
            for lyr, r in per_layer_optima.items():
                flag = "  *** BOUNDARY ***" if abs(r - ratio_min) < 1e-9 or abs(r - ratio_max) < 1e-9 else ""
                L.append(f"    {lyr:>4} : {r:.4f}{flag}")
            L.append("")
        if boundary_warnings:
            L.append("  BOUNDARY WARNINGS:")
            L.extend(boundary_warnings)
            L.append("")

    # ── 4. Correlation at optimal ratio ───────────────────────────────
    if not corr_df.empty:
        L += [
            "4. CORRELATION AT OPTIMAL RATIO", _hr(),
            f"  {'Metric':<14} {'M':>4}  {'Pearson r':>10}  {'p(Pearson)':>12}"
            f"  {'Spearman ρ':>11}  {'p(Spearman)':>12}  {'n':>4}",
            f"  {_hr('-', 69)}",
        ]
        opt_df = corr_df[corr_df["ratio"].apply(lambda v: abs(v - optimal_ratio) < 1e-9)].copy()
        for metric in ["nc_coverage", "recall", "fom"]:
            sub = opt_df[opt_df["metric"] == metric].sort_values("M")
            if sub.empty:
                continue
            metric_label = {"nc_coverage": "NC coverage",
                            "recall": f"Recall W={W_default}",
                            "fom": f"FoM    W={W_default}"}[metric]
            for _, row in sub.iterrows():
                L.append(
                    f"  {metric_label:<14} {int(row['M']):>4}  {_fmt_r(row['pearson_r']):>10}  "
                    f"{_fmt_p(row['pearson_p']):>12}  {_fmt_r(row['spearman_rho']):>11}  "
                    f"{_fmt_p(row['spearman_p']):>12}  {int(row['n']):>4}"
                )
            L.append("")

    # ── 5. Best correlation across full sweep ─────────────────────────
    if not corr_df.empty:
        L += [
            "5. BEST CORRELATION ACROSS FULL RATIO SWEEP", _hr(),
            f"  {'Metric':<14} {'Ratio':>7}  {'M':>4}  {'|Pearson r|':>12}  {'Spearman ρ':>11}",
            f"  {_hr('-', 55)}",
        ]
        for metric in ["nc_coverage", "recall", "fom"]:
            sub = corr_df[corr_df["metric"] == metric].dropna(subset=["pearson_r"])
            if sub.empty:
                continue
            best_row = sub.loc[sub["pearson_r"].abs().idxmax()]
            mlab = {"nc_coverage": "NC coverage",
                    "recall": f"Recall W={W_default}",
                    "fom": f"FoM    W={W_default}"}[metric]
            L.append(f"  {mlab:<14} {best_row['ratio']:>7.2f}  {int(best_row['M']):>4}  "
                     f"{abs(best_row['pearson_r']):>12.4f}  {_fmt_r(best_row['spearman_rho']):>11}")
        L.append("")

    # ── 6. Per-setup values at optimal ratio ──────────────────────────
    opt_results = [s.ssd_results.get(optimal_ratio) for s in setups]
    if any(v is not None for v in opt_results):
        L += [f"6. PER-SETUP VALUES AT OPTIMAL RATIO  (ratio={optimal_ratio:.2f})", _hr()]
        L += [
            f"  A) NC coverage (M=1) and Recall (M=1, W={W_default})",
            f"  {'Setup':<30} {'SSD NC':>8}  {'PMT NC':>8}  {'SSD Rec':>9}  {'PMT Rec':>9}",
            f"  {_hr('-', 69)}",
        ]
        for s, res in zip(setups, opt_results):
            if res is None:
                L.append(f"  {s.label:<30}  (no SSD results)")
                continue
            L.append(f"  {s.label:<30} {_ssd_nc_frac(res, 1):>8.4f}  {_pmt_nc_frac(s, 1):>8.4f}"
                     f"  {_ssd_recall(res, 1, W_default):>9.4f}  {_pmt_recall(s, 1, W_default):>9.4f}")
        L.append("")

        L += [
            "  B) FoM — using PMT-best (M,W) per setup",
            f"  {'Setup':<30} {'Best(M,W)':>10}  {'SSD FoM':>10}  {'PMT FoM':>10}  {'Δ(SSD-PMT)':>12}",
            f"  {_hr('-', 75)}",
        ]
        for s, res in zip(setups, opt_results):
            M_b, W_b = _pmt_best_mw(s, M_values, W_values, total_primaries)
            pmt_f    = _pmt_fom(s, M_b, W_b, total_primaries)
            ssd_f    = _ssd_fom(res, M_b, W_b, total_primaries) if res is not None else float("nan")
            delta    = ssd_f - pmt_f if (math.isfinite(ssd_f) and math.isfinite(pmt_f)) else float("nan")
            L.append(
                f"  {s.label:<30} {'M'+str(M_b)+'W'+str(W_b):>10}  "
                f"{(f'{ssd_f:.5g}' if math.isfinite(ssd_f) else 'N/A'):>10}  "
                f"{(f'{pmt_f:.5g}' if math.isfinite(pmt_f) else 'N/A'):>10}  "
                f"{(f'{delta:+.5g}' if math.isfinite(delta) else 'N/A'):>12}"
            )
        L.append("")

    # ── 7. Ranking consistency at optimal ratio ───────────────────────
    valid_r = [(s, s.ssd_results.get(optimal_ratio)) for s in setups
               if s.ssd_results.get(optimal_ratio) is not None]
    if len(valid_r) >= 3:
        L += [
            "7. NC COVERAGE RANKING CONSISTENCY (PMT vs SSD, at optimal ratio)", _hr(),
            f"  {'M':>4}  {'Spearman ρ':>12}  {'p-value':>10}  {'Inversions':>12}  {'Significant':>12}",
            f"  {_hr('-', 56)}",
        ]
        valid_setups_r = [sv[0] for sv in valid_r]
        valid_results  = [sv[1] for sv in valid_r]
        nv = len(valid_setups_r)
        for M in M_values:
            pmt_nc = np.array([_pmt_nc_frac(s, M) for s in valid_setups_r])
            ssd_nc = np.array([_ssd_nc_frac(res, M) for res in valid_results])
            pmt_r  = scipy_stats.rankdata(-pmt_nc)
            ssd_r  = scipy_stats.rankdata(-ssd_nc)
            rho, p = _safe_spearmanr(pmt_r, ssd_r)
            n_inv  = sum(1 for i in range(nv) for j in range(i+1, nv)
                         if (pmt_r[i] < pmt_r[j]) != (ssd_r[i] < ssd_r[j]))
            sig    = "yes" if (not math.isnan(p) and p < _P_3SIGMA) else "no"
            L.append(f"  {M:>4}  {_fmt_r(rho):>12}  {_fmt_p(p):>10}  {n_inv:>12}  {sig:>12}")
        L.append("")

    # ── 8. FoM best-(M,W) agreement ──────────────────────────────────
    if any(v is not None for v in opt_results):
        L += [
            f"8. BEST (M,W) AGREEMENT — PMT vs SSD  (ratio={optimal_ratio:.2f})", _hr(),
            f"  {'Setup':<30} {'PMT_best':>10} {'SSD_best':>10} {'Agree':>6}"
            f"  {'PMT_M≥'+str(min_M_fom):>12} {'SSD_M≥'+str(min_M_fom):>12} {'Agree':>6}",
            f"  {_hr('-', 80)}",
        ]
        n_agree_u = n_agree_c = 0
        for s, res in zip(setups, opt_results):
            M_pu, W_pu = _pmt_best_mw(s, M_values, W_values, total_primaries)
            M_su = W_su = None
            M_pc, W_pc = _pmt_best_mw(s, M_values, W_values, total_primaries, min_M=min_M_fom)
            M_sc = W_sc = None
            agr_u = agr_c = "N/A"
            if res is not None:
                M_su, W_su = _ssd_best_mw(res, M_values, W_values, total_primaries)
                M_sc, W_sc = _ssd_best_mw(res, M_values, W_values, total_primaries, min_M=min_M_fom)
                agr_u = "yes" if (M_pu == M_su and W_pu == W_su) else "no"
                agr_c = "yes" if (M_pc == M_sc and W_pc == W_sc) else "no"
                if agr_u == "yes": n_agree_u += 1
                if agr_c == "yes": n_agree_c += 1
            pmt_u = f"M{M_pu}W{W_pu}"
            ssd_u = f"M{M_su}W{W_su}" if M_su is not None else "N/A"
            pmt_c = f"M{M_pc}W{W_pc}"
            ssd_c = f"M{M_sc}W{W_sc}" if M_sc is not None else "N/A"
            L.append(f"  {s.label:<30} {pmt_u:>10} {ssd_u:>10} {agr_u:>6}"
                     f"  {pmt_c:>12} {ssd_c:>12} {agr_c:>6}")
        n_setups_with_res = sum(1 for v in opt_results if v is not None)
        L += [f"  Summary: unconstrained {n_agree_u}/{n_setups_with_res} agree; "
              f"M≥{min_M_fom} {n_agree_c}/{n_setups_with_res} agree", ""]

    # ── 9. FoM M≥min_M comparison ─────────────────────────────────────
    eligible_M = [M for M in M_values if M >= min_M_fom]
    if eligible_M and any(v is not None for v in opt_results):
        L += [
            f"9. FoM M≥{min_M_fom} COMPARISON (PMT-anchor vs independent)",
            _hr(),
            f"  {'Setup':<30} {'PMT_FoM':>10} {'SSD(PMT_MW)':>12} {'Δ':>8}"
            f"  {'PMT_FoM_ind':>12} {'SSD_FoM_ind':>12} {'ΔMWP':>8} {'ΔMWS':>8}",
            f"  {_hr('-', 95)}",
        ]
        for s, res in zip(setups, opt_results):
            M_bp, W_bp = _pmt_best_mw(s, M_values, W_values, total_primaries, min_M=min_M_fom)
            pmt_fom_a  = _pmt_fom(s, M_bp, W_bp, total_primaries)
            ssd_fom_a  = _ssd_fom(res, M_bp, W_bp, total_primaries) if res is not None else float("nan")
            delta_a    = ssd_fom_a - pmt_fom_a if (math.isfinite(ssd_fom_a) and math.isfinite(pmt_fom_a)) else float("nan")

            if res is not None:
                M_bs, W_bs = _ssd_best_mw(res, M_values, W_values, total_primaries, min_M=min_M_fom)
                pmt_fom_i  = pmt_fom_a  # PMT uses its own best
                ssd_fom_i  = _ssd_fom(res, M_bs, W_bs, total_primaries)
                delta_mwp  = f"M{M_bp}W{W_bp}"
                delta_mws  = f"M{M_bs}W{W_bs}"
            else:
                pmt_fom_i = ssd_fom_i = float("nan")
                delta_mwp = delta_mws = "N/A"

            def _fs(v): return f"{v:.4g}" if math.isfinite(v) else "N/A"
            def _fd(v): return f"{v:+.4g}" if math.isfinite(v) else "N/A"

            L.append(
                f"  {s.label:<30} {_fs(pmt_fom_a):>10} {_fs(ssd_fom_a):>12} {_fd(delta_a):>8}"
                f"  {_fs(pmt_fom_i):>12} {_fs(ssd_fom_i):>12} {delta_mwp:>8} {delta_mws:>8}"
            )
        L.append("")

    # ── 10. Per-layer sweep summary ───────────────────────────────────
    if per_layer_corr_df is not None and not per_layer_corr_df.empty:
        L += [
            "10. PER-LAYER SWEEP — BEST |PEARSON r| PER LAYER", _hr(),
            f"  {'Layer':<6} {'Metric':<14} {'Ratio':>7}  {'M':>4}  {'|Pearson r|':>12}  {'Spearman ρ':>11}",
            f"  {_hr('-', 58)}",
        ]
        for layer in ["pit", "bot", "top", "wall"]:
            layer_df = per_layer_corr_df[per_layer_corr_df["layer"] == layer]
            for metric in ["nc_coverage", "recall", "fom"]:
                sub = layer_df[layer_df["metric"] == metric].dropna(subset=["pearson_r"])
                if sub.empty:
                    continue
                best_row = sub.loc[sub["pearson_r"].abs().idxmax()]
                mlab = {"nc_coverage": "NC coverage",
                        "recall": f"Recall W={W_default}",
                        "fom": f"FoM    W={W_default}"}[metric]
                L.append(
                    f"  {layer:<6} {mlab:<14} {best_row['ratio']:>7.2f}  "
                    f"{int(best_row['M']):>4}  {abs(best_row['pearson_r']):>12.4f}  "
                    f"{_fmt_r(best_row['spearman_rho']):>11}"
                )
        L.append("")

    # ── 11. Interpretation guide ──────────────────────────────────────
    L += [
        "11. INTERPRETATION GUIDE", _hr(),
        "  Pearson r / Spearman ρ (SSD vs PMT across setups):",
        "    ≥  0.8   strong  — SSD reliably ranks setups the same way as PMT",
        "    0.6–0.8  moderate — ranking broadly preserved; absolute values differ",
        "    0.4–0.6  weak    — partial agreement; SSD may mislead marginal cases",
        "    < 0.4   poor    — SSD and PMT disagree at this ratio",
        "",
        "  Ranking consistency (Section 7):",
        "    Spearman ρ = 1 means SSD perfectly preserves the PMT ordering of setups.",
        "    Inversions count pairs of setups where SSD disagrees with PMT on order.",
        "",
        "  FoM agreement (Section 8):",
        "    'Agree' = PMT and SSD choose identical (M,W) as best.",
        "    Disagreement indicates the two simulators have a different optimal point.",
        "",
        "  Δ (SSD−PMT) in Sections 6B and 9:",
        "    Systematic positive Δ → SSD over-estimates FoM.",
        "    Large |Δ| with high ρ → correct ranking but scale offset.",
        "",
    ]

    with open(os.path.join(output_dir, "correlation_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("  Saved: correlation_summary.txt")

    if not corr_df.empty:
        for metric_name in ["nc_coverage", "recall", "fom"]:
            sub = corr_df[corr_df["metric"] == metric_name]
            if sub.empty:
                continue
            csv_path = os.path.join(output_dir, f"corr_vs_ratio_{metric_name}.csv")
            sub.to_csv(csv_path, index=False)
        print("  Saved: corr_vs_ratio_*.csv")

    if per_layer_corr_df is not None and not per_layer_corr_df.empty:
        per_layer_corr_df.to_csv(
            os.path.join(output_dir, "corr_vs_ratio_per_layer.csv"), index=False
        )
        print("  Saved: corr_vs_ratio_per_layer.csv")


# ─────────────────────────────────────────────────────────────────────
# SECTION 15 — main
# ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Correlate SSD and PMT simulation metrics across experimental setups."
    )
    p.add_argument("--muon-dir",       required=True,
                   help="NC truth directory (contains run_***/ subdirs, Sim 1)")
    p.add_argument("--ssd-hdf5",       required=True,
                   help="Shared SSD HDF5 file (target_matrix + phi_matrix)")
    p.add_argument("--setup-dirs",     nargs="+", required=True,
                   help="Setup directories, each containing run_***/ (Sim 2) + *.json")
    p.add_argument("--m",              type=int, default=1,
                   help="Photon threshold for binarisation (default: 1)")
    p.add_argument("--M-max",          type=int, default=10,
                   help="Maximum multiplicity threshold (default: 10)")
    p.add_argument("--W-max",          type=int, default=20,
                   help="Maximum W threshold (default: 20)")
    p.add_argument("--W-default",      type=int, default=1,
                   help="W used for recall correlation plots (default: 1)")
    p.add_argument("--min-M-fom",      type=int, default=6,
                   help="Minimum M for constrained FoM analysis (default: 6)")
    p.add_argument("--ratio-min",      type=float, default=1.0,
                   help="Start of ratio sweep (default: 1.0)")
    p.add_argument("--ratio-max",      type=float, default=100.0,
                   help="Safety cap for ratio sweep; early stopping applies (default: 100.0)")
    p.add_argument("--ratio-step",     type=float, default=0.1,
                   help="Step size for ratio sweep (default: 0.1)")
    p.add_argument("--output-dir",     default="./correlation_results",
                   help="Output directory (default: ./correlation_results)")
    p.add_argument("--seed",           type=int, default=42,
                   help="Random seed for stochastic rounding (default: 42)")
    p.add_argument("--skip-per-layer", action="store_true",
                   help="Skip the per-layer ratio sweep (only relevant with --ratio-sweep)")
    p.add_argument("--ratio-sweep", action="store_true",
                   help="Enable the global+per-layer ratio sweep to find the optimal "
                        "ratio. When not set (default), --pit/--bot/--top/--wall must "
                        "all be provided and evaluation runs at those fixed ratios.")
    p.add_argument("--pit",  type=float, default=None,
                   help=f"Fixed pit  area ratio (required without --ratio-sweep; "
                        f"physics default: {DEFAULT_AREA_RATIOS['pit']})")
    p.add_argument("--bot",  type=float, default=None,
                   help=f"Fixed bot  area ratio (required without --ratio-sweep; "
                        f"physics default: {DEFAULT_AREA_RATIOS['bot']})")
    p.add_argument("--top",  type=float, default=None,
                   help=f"Fixed top  area ratio (required without --ratio-sweep; "
                        f"physics default: {DEFAULT_AREA_RATIOS['top']})")
    p.add_argument("--wall", type=float, default=None,
                   help=f"Fixed wall area ratio (required without --ratio-sweep; "
                        f"physics default: {DEFAULT_AREA_RATIOS['wall']})")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    M_values = list(range(1, args.M_max + 1))
    W_values = list(range(1, args.W_max + 1))

    # ── 1. NC truth ───────────────────────────────────────────────────
    print("Loading NC truth ...")
    nc_truth = build_nc_truth(args.muon_dir)

    # ── 2. SSD COO data ───────────────────────────────────────────────
    print("\nLoading SSD raw sparse data ...")
    ssd_coo_data = load_raw_sparse(args.ssd_hdf5, skip_validity=True)
    _, _, _, voxel_ids, _, _, _, _ = ssd_coo_data
    ssd_voxel_id_to_col: dict[str, int] = {vid: i for i, vid in enumerate(voxel_ids)}

    # ── 3. SSD muon alignment ─────────────────────────────────────────
    print("\nBuilding SSD muon alignment ...")
    alignment = align_ssd_to_pmt(args.ssd_hdf5, nc_truth)

    # ── 3b. Total primaries ───────────────────────────────────────────
    _n_runs = _count_runs(args.muon_dir)
    total_primaries = _n_runs * MUONS_PER_RUN_DIR
    print(f"\n  FoM: total primary muons = {total_primaries:,}  ({_n_runs} runs × {MUONS_PER_RUN_DIR:,})")
    print(f"  FoM: livetime = {total_primaries / MUSUN_RATE:,.0f} h  "
          f"= {total_primaries / MUSUN_RATE / (24*365.25):.2f} yr")

    if len(args.setup_dirs) < 5:
        warnings.warn(f"Only {len(args.setup_dirs)} setups — correlation statistics may not be meaningful.")

    # ── 4. Load each setup ────────────────────────────────────────────
    setups: list[SetupData]               = []
    setup_voxel_subsets: list[np.ndarray] = []

    for setup_dir in args.setup_dirs:
        label = Path(setup_dir).name
        print(f"\nSetup: {label}")

        json_files = sorted(glob.glob(os.path.join(setup_dir, "*.json")))
        if not json_files:
            raise FileNotFoundError(f"No *.json config in {setup_dir!r}")
        json_path = json_files[0]

        json_ids = _load_json_voxel_ids(json_path)
        missing  = [v for v in json_ids if v not in ssd_voxel_id_to_col]
        if missing:
            warnings.warn(f"{label}: {len(missing)}/{len(json_ids)} voxel IDs missing from SSD HDF5.")
        voxel_col_subset = np.array(
            [ssd_voxel_id_to_col[v] for v in json_ids if v in ssd_voxel_id_to_col],
            dtype=np.int32,
        )

        print("  Loading PMT simulation data ...")
        pmt_nc, pmt_muon, _ = load_pmt_setup(setup_dir, nc_truth, M_values, W_values, m_threshold=args.m)
        w2     = compute_setup_w2(json_path)
        print(f"  W2={'N/A' if w2 is None else f'{w2:.2f}'}, {len(voxel_col_subset)} SSD voxels mapped")

        setups.append(SetupData(label=label, w2=w2, pmt_nc=pmt_nc, pmt_muon=pmt_muon))
        setup_voxel_subsets.append(voxel_col_subset)

    # ── 5. Ratio evaluation — fixed ratios or sweep ───────────────────
    _ALL_LAYERS = ("pit", "bot", "top", "wall")
    cli_ratio_map = {
        lyr: getattr(args, lyr) for lyr in _ALL_LAYERS if getattr(args, lyr) is not None
    }

    corr_df:             pd.DataFrame                = pd.DataFrame()
    per_layer_corr_df:   Optional[pd.DataFrame]      = None
    per_layer_optima:    Optional[dict[str, float]]  = None
    _fixed_area_ratios:  Optional[dict[str, float]]  = None

    if not args.ratio_sweep:
        # ── Fixed ratio mode ──────────────────────────────────────────
        missing_layers = [lyr for lyr in _ALL_LAYERS if lyr not in cli_ratio_map]
        if missing_layers:
            sys.exit(
                f"Error: --ratio-sweep is not set; all four layer ratios must be "
                f"provided.  Missing: {', '.join(f'--{l}' for l in missing_layers)}"
            )
        _fixed_area_ratios = {lyr: cli_ratio_map[lyr] for lyr in _ALL_LAYERS}
        nominal_ratio = float(np.mean(list(_fixed_area_ratios.values())))
        print(f"\nFixed area ratios (no ratio sweep):")
        for _lyr, _r in _fixed_area_ratios.items():
            print(f"  {_lyr:>6}: {_r}")
        print(f"  Nominal key (mean): {nominal_ratio:.4f}")

        (raw_rows, raw_cols, raw_vals, _, _, layers_arr, num_ncs_val, _) = ssd_coo_data
        num_voxels = len(voxel_ids)

        print(f"\nEvaluating {len(setups)} setup(s) at fixed ratios ...")
        for i, setup in enumerate(setups):
            print(f"  {setup.label}", end=" ... ", flush=True)
            setup.ssd_results[nominal_ratio] = evaluate_ssd_at_ratio(
                raw_rows, raw_cols, raw_vals, num_ncs_val, num_voxels, layers_arr,
                _fixed_area_ratios, args.m, M_values, W_values, alignment,
                setup_voxel_subsets[i], seed=args.seed,
            )
            nc_m1 = _ssd_nc_frac(setup.ssd_results[nominal_ratio], 1)
            print(f"NC(M=1)={nc_m1:.3f}")

        optimal_ratio = nominal_ratio

    else:
        # ── Ratio sweep mode ──────────────────────────────────────────
        if cli_ratio_map:
            warnings.warn(
                "--ratio-sweep is active; --pit/--bot/--top/--wall flags are "
                "ignored during the sweep."
            )

        ratio_factors = np.arange(
            args.ratio_min, args.ratio_max + args.ratio_step / 2, args.ratio_step
        )
        print(f"\nGlobal ratio sweep [{args.ratio_min:.2f}, {args.ratio_max:.2f}] "
              f"step={args.ratio_step:.2f} ({len(ratio_factors)} steps) ...")
        corr_df, optimal_ratio = global_ratio_sweep(
            setups, ssd_coo_data, setup_voxel_subsets, alignment,
            ratio_factors, args.m, M_values, W_values, args.W_default,
            seed=args.seed, total_primaries=total_primaries,
        )

        if not args.skip_per_layer:
            print("\nPer-layer ratio sweep ...")
            per_layer_corr_df, per_layer_optima = per_layer_ratio_sweep(
                setups, ssd_coo_data, setup_voxel_subsets, alignment,
                optimal_ratio, ratio_factors, args.m, M_values, W_values,
                args.W_default, seed=args.seed, total_primaries=total_primaries,
            )

    # ── 7. Plots ──────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    odir = args.output_dir

    # Existing correlation-vs-ratio plots
    plot_corr_vs_ratio(corr_df, "nc_coverage", M_values, args.W_default, odir, optimal_ratio)
    plot_corr_vs_ratio(corr_df, "recall",      M_values, args.W_default, odir, optimal_ratio)
    plot_corr_vs_ratio(corr_df, "fom",         M_values, args.W_default, odir, optimal_ratio)

    # Scatter plots at optimal ratio
    plot_scatter_ssd_pmt(setups, optimal_ratio, "nc_coverage", M_values, args.W_default, odir)
    plot_scatter_ssd_pmt(setups, optimal_ratio, "recall",      M_values, args.W_default, odir)
    plot_scatter_ssd_pmt_fom(setups, optimal_ratio, M_values, W_values, total_primaries, odir)

    # W2 plots
    plot_w2_comparison(setups, optimal_ratio, M_values, args.W_default, odir)
    plot_w2_spearman_fom(setups, M_values, W_values, total_primaries, optimal_ratio, odir)

    # Cross-metric scatter
    plot_ssd_nc_vs_pmt_recall_at_best_fom(setups, optimal_ratio, M_values, W_values, total_primaries, odir)
    plot_ssd_recall_vs_pmt_recall_at_best_fom(setups, optimal_ratio, M_values, W_values, total_primaries, odir)

    # ── NEW: ranking and FoM comparison plots ─────────────────────────
    plot_nc_ranking_consistency(setups, optimal_ratio, M_values, odir)
    plot_ranking_heatmap(setups, optimal_ratio, M_values, odir)
    plot_fom_bars_pmt_vs_ssd(setups, optimal_ratio, M_values, W_values, total_primaries, odir)
    plot_fom_scatter_per_m(setups, optimal_ratio, M_values, W_values, total_primaries, odir)
    plot_fom_mge_comparison(setups, optimal_ratio, M_values, W_values, total_primaries,
                             args.min_M_fom, odir)
    plot_recall_comparison(setups, optimal_ratio, M_values, W_values, total_primaries, odir)
    plot_best_mw_agreement(setups, optimal_ratio, M_values, W_values, total_primaries,
                            args.min_M_fom, odir)

    # ── 8. Summary ────────────────────────────────────────────────────
    print("\nWriting summary ...")
    write_summary(
        corr_df, per_layer_corr_df, per_layer_optima, optimal_ratio,
        setups, M_values, W_values, args.W_default,
        total_primaries, odir,
        ratio_min=args.ratio_min, ratio_max=args.ratio_max,
        ratio_step=args.ratio_step, min_M_fom=args.min_M_fom,
        area_ratios=_fixed_area_ratios,
    )
    print(f"\nDone. Results saved to: {odir}")


if __name__ == "__main__":
    main()
