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

Usage
-----
    python correlate_ssd_pmt.py \\
        --muon-dir  /path/to/nc_truth/ \\
        --ssd-hdf5  /path/to/data.hdf5 \\
        --setup-dirs /path/A /path/B /path/C \\
        [--m 1] [--M-max 10] [--W-max 20] [--W-default 1] \\
        [--ratio-min 1.0] [--ratio-max 6.0] [--ratio-step 0.1] \\
        [--output-dir ./correlation_results] \\
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── sys.path setup ────────────────────────────────────────────────────
# Ensure ratio_analysis package is importable when run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Also make src/pmtopt importable.
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


# ─────────────────────────────────────────────────────────────────────
# SECTION 2 — Constants & data containers
# ─────────────────────────────────────────────────────────────────────

# TODO: update column names when HDF5 phi_matrix schema is finalised
_SSD_RUN_COL  = "run_id"         # placeholder (per-NC run identifier)
_SSD_MUON_COL = "local_muon_id"  # placeholder (per-run event ID matching nc_truth.muon_id)
_SSD_NC_COL   = "nc_id"          # placeholder (NC track ID matching nc_truth.nc_id)

_DEFAULT_RATIO_RANGE: tuple[float, float] = (
    min(DEFAULT_AREA_RATIOS.values()),   # 1.8776 (wall)
    max(DEFAULT_AREA_RATIOS.values()),   # 2.3843 (bot)
)

# Qualitative palette — same as compare_coverages.py so colours are
# consistent when cross-referencing plots from both scripts.
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
    """Return n setup colours, cycling _SETUP_PALETTE if n > 10."""
    return [_SETUP_PALETTE[i % len(_SETUP_PALETTE)] for i in range(n)]


@dataclass
class SSDResult:
    """SSD evaluation results at one area ratio for one setup."""
    nc_detected: dict[int, int]                           # M -> count(coverage >= M)
    num_ncs:     int                                       # total NCs evaluated
    confusion:   dict[tuple[int, int], dict[str, int]]    # (M,W) -> {TP,FP,TN,FN}


@dataclass
class SetupData:
    """All data for one experimental setup."""
    label:       str
    w2:          Optional[float]    # 2-Wasserstein homogeneity (ratio-independent)
    pmt_nc:      dict               # from ratio_analysis evaluate_nc()
    pmt_muon:    dict               # from ratio_analysis evaluate_muon()
    ssd_results: dict = field(default_factory=dict)  # ratio_key -> SSDResult


# ─────────────────────────────────────────────────────────────────────
# SECTION 3 — JSON loader
# ─────────────────────────────────────────────────────────────────────

def _load_json_voxel_ids(json_path: str) -> list[str]:
    """Read voxel IDs from a config JSON file.

    Handles two formats:
    - Flat list:  [{"index": "voxel_id", ...}, ...]
    - Dict:       {"selected_voxels": [{"index": "voxel_id", ...}, ...]}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [
            v["index"] if isinstance(v, dict) and "index" in v else str(v)
            for v in data
        ]
    voxel_dicts = data.get("selected_voxels", [])
    return [v["index"] for v in voxel_dicts if isinstance(v, dict) and "index" in v]


# ─────────────────────────────────────────────────────────────────────
# SECTION 4 — W2 helper
# ─────────────────────────────────────────────────────────────────────

def compute_setup_w2(json_path: str) -> Optional[float]:
    """Compute 2-Wasserstein homogeneity from a voxel JSON. Returns None on failure."""
    try:
        from pmtopt.homogeneous import compute_wasserstein_homogeneity, get_w2_ref
    except ImportError:
        print("  [WARN] pmtopt not importable; W2 will not be computed.")
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        voxel_dicts = (
            data if isinstance(data, list)
            else data.get("selected_voxels", [])
        )
        voxel_dicts = [v for v in voxel_dicts if isinstance(v, dict) and "center" in v]
        if len(voxel_dicts) < 2:
            return None
        centers = np.array([v["center"] for v in voxel_dicts], dtype=float)
        return float(
            compute_wasserstein_homogeneity(centers, reference=get_w2_ref())["w2"]
        )
    except Exception as exc:
        print(f"  [WARN] W2 computation failed for {json_path!r}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────
# SECTION 5 — NC alignment
# ─────────────────────────────────────────────────────────────────────

def align_ssd_to_pmt(hdf5_path: str, nc_truth: pd.DataFrame) -> dict:
    """Load SSD phi_matrix and build muon index arrays.

    Returns a dict with all arrays needed for SSD muon-confusion evaluation:

        ssd_row_order        : row indices into SSD B matrix (currently all rows)
        global_muon_id       : per-NC global muon ID (from phi_matrix)
        nc_time_ns           : per-NC NC time [ns]
        nc_flag_ge77         : per-NC Ge77 flag (bool)
        all_unique_muons     : sorted unique global_muon_id values
        ge77_muon_global_ids : global IDs of Ge77 muons
        global_to_all_local  : NC -> muon local index (within all_unique_muons)
        ge77_mask_all        : bool mask for Ge77 muons (shape: total_muons)
        nc_is_veto_candidate : in_time & ~nc_flag_ge77 (shape: num_ncs)
        total_muons          : len(all_unique_muons)
    """
    with h5py.File(hdf5_path, "r") as f:
        phi_columns = [
            c.decode() if isinstance(c, bytes) else str(c)
            for c in f["phi_columns"][:]
        ]
        phi_col_idx = {name: i for i, name in enumerate(phi_columns)}

        phi = f["phi_matrix"]
        nc_time_ns   = phi[:, phi_col_idx["nC_time_in_ns"]].astype(np.float64)
        nc_flag_ge77 = phi[:, phi_col_idx["nC_flag_Ge77"]].astype(bool)

        # Muon IDs live in event_ids, not phi_matrix.
        # Build a globally unique muon ID from (run_id, muon_id) pairs.
        event_id_cols = [c.decode() if isinstance(c, bytes) else str(c)
                         for c in f["event_id_columns"][:]]
        event_ids = f["event_ids"][:]
        run_col  = event_id_cols.index("run_id")
        muon_col = event_id_cols.index("muon_id")
        pairs = np.stack(
            [event_ids[:, run_col], event_ids[:, muon_col]], axis=1
        )
        _, global_muon_id = np.unique(pairs, axis=0, return_inverse=True)
        global_muon_id = global_muon_id.astype(np.int64)

        # Key-based alignment (future): join SSD rows to nc_truth by
        # (run_id, local_muon_id, nc_id) for a bijective 1-to-1 row mapping.
        has_alignment_cols = all(
            c in phi_col_idx for c in [_SSD_RUN_COL, _SSD_MUON_COL, _SSD_NC_COL]
        )

    num_ncs = len(global_muon_id)

    if has_alignment_cols:
        # TODO: implement bijective row mapping to nc_truth when columns are available
        warnings.warn(
            "SSD phi_matrix alignment columns found but key-based alignment is "
            "not yet implemented — falling back to nc_time_ns ordering (approximate)."
        )
    else:
        warnings.warn(
            "SSD phi_matrix missing alignment columns "
            "— using nc_time_ns ordering (approximate)."
        )

    ssd_row_order = np.arange(num_ncs, dtype=np.int64)

    # Muon index (replicate evaluate_coverages.py lines 255–275)
    (_, _, ge77_muon_global_ids, _, _) = build_muon_index(
        global_muon_id, nc_time_ns, nc_flag_ge77, verbose=False,
    )

    all_unique_muons    = np.unique(global_muon_id)
    total_muons         = len(all_unique_muons)
    global_to_all_local = np.searchsorted(
        all_unique_muons, global_muon_id
    ).astype(np.int32)
    ge77_mask_all       = np.isin(all_unique_muons, ge77_muon_global_ids)

    in_time = (
        (nc_time_ns >= MUON_TIME_WINDOW_MIN_NS)
        & (nc_time_ns <= MUON_TIME_WINDOW_MAX_NS)
    )
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
    """Load and evaluate PMT simulation data for one setup.

    Returns (pmt_nc, pmt_muon, json_path).
    """
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

    Replicates evaluate_coverages.py evaluate_nc (lines 330–340) and
    evaluate_muon (lines 363–385) inline using the alignment arrays.
    """
    B_full = binarize_from_raw(
        raw_rows, raw_cols, raw_vals,
        num_ncs, num_voxels, layers,
        area_ratios, m, seed=seed,
    )

    rows  = alignment["ssd_row_order"]
    B_sub = B_full.tocsr()[rows, :][:, voxel_col_subset].tocsc()

    # NC coverage counts — CSC column iteration
    coverage_counts = np.zeros(len(rows), dtype=np.int16)
    for col in range(B_sub.shape[1]):
        s, e = B_sub.indptr[col], B_sub.indptr[col + 1]
        coverage_counts[B_sub.indices[s:e]] += 1

    nc_detected = {M: int(np.sum(coverage_counts >= M)) for M in M_values}

    # Muon confusion (replicate evaluate_coverages.py lines 363–385)
    nc_is_veto  = alignment["nc_is_veto_candidate"][rows]
    g2al        = alignment["global_to_all_local"][rows]
    ge77_mask   = alignment["ge77_mask_all"]
    total_muons = alignment["total_muons"]

    confusion: dict[tuple[int, int], dict[str, int]] = {}
    for M in M_values:
        nc_det_veto  = (coverage_counts >= M) & nc_is_veto
        det_idx      = np.where(nc_det_veto)[0]
        if len(det_idx) > 0:
            muon_lids = g2al[det_idx]
            muon_det  = np.bincount(muon_lids, minlength=total_muons).astype(np.int32)
        else:
            muon_det  = np.zeros(total_muons, dtype=np.int32)

        for W in W_values:
            cls = muon_det >= W
            confusion[(M, W)] = {
                "TP": int(np.sum( cls &  ge77_mask)),
                "FP": int(np.sum( cls & ~ge77_mask)),
                "FN": int(np.sum(~cls &  ge77_mask)),
                "TN": int(np.sum(~cls & ~ge77_mask)),
            }

    del B_full, B_sub
    gc.collect()

    return SSDResult(nc_detected=nc_detected, num_ncs=int(len(rows)), confusion=confusion)


# ─────────────────────────────────────────────────────────────────────
# SECTION 8 — Metric extraction helpers
# ─────────────────────────────────────────────────────────────────────

def _pmt_nc_frac(setup: SetupData, M: int) -> float:
    total = setup.pmt_nc.get("nc_total", 0)
    if total == 0:
        return 0.0
    return setup.pmt_nc["nc_detected"].get(M, 0) / total


def _ssd_nc_frac(result: SSDResult, M: int) -> float:
    if result.num_ncs == 0:
        return 0.0
    return result.nc_detected.get(M, 0) / result.num_ncs


def _pmt_recall(setup: SetupData, M: int, W: int) -> float:
    conf = setup.pmt_muon["confusion"].get((M, W))
    if conf is None:
        return 0.0
    return compute_metrics(conf["TP"], conf["FP"], conf["TN"], conf["FN"])["Recall"]


def _ssd_recall(result: SSDResult, M: int, W: int) -> float:
    conf = result.confusion.get((M, W))
    if conf is None:
        return 0.0
    return compute_metrics(conf["TP"], conf["FP"], conf["TN"], conf["FN"])["Recall"]


def _pmt_fom(setup: SetupData, M: int, W: int, total_primaries: int) -> float:
    """FoM from PMT confusion matrix at (M, W)."""
    conf = setup.pmt_muon["confusion"].get((M, W))
    if conf is None:
        return float("nan")
    return calc_fom_confusion(conf["TP"], conf["FP"], conf["FN"], total_primaries)


def _ssd_fom(result: SSDResult, M: int, W: int, total_primaries: int) -> float:
    """FoM from SSD confusion matrix at (M, W)."""
    conf = result.confusion.get((M, W))
    if conf is None:
        return float("nan")
    return calc_fom_confusion(conf["TP"], conf["FP"], conf["FN"], total_primaries)


def _pmt_best_w_at_m(
    setup: SetupData,
    M: int,
    W_values: list[int],
    total_primaries: int,
) -> int:
    """Return the W that maximises PMT FoM at this M for this setup."""
    best_w   = W_values[0]
    best_fom = float("-inf")
    for W in W_values:
        v = _pmt_fom(setup, M, W, total_primaries)
        if np.isfinite(v) and v > best_fom:
            best_fom = v
            best_w   = W
    return best_w


def _pmt_best_mw(
    setup: SetupData,
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
) -> tuple[int, int]:
    """Return the (M, W) pair that maximises PMT FoM globally for this setup."""
    best_mw  = (M_values[0], W_values[0])
    best_fom = float("-inf")
    for M in M_values:
        for W in W_values:
            v = _pmt_fom(setup, M, W, total_primaries)
            if np.isfinite(v) and v > best_fom:
                best_fom = v
                best_mw  = (M, W)
    return best_mw


def _count_runs(muon_dir: str) -> int:
    """Count run_*/ subdirectories in muon_dir (for total_primaries fallback)."""
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
    """Pearson/Spearman between SSD and PMT metrics for each M.

    Parameters
    ----------
    ratio_key       : key for ssd_results lookup (float for global sweep,
                      (layer, float) tuple for per-layer sweep).
    ratio_label     : float stored in the output rows (defaults to ratio_key
                      when it is already a float).
    total_primaries : total simulated primary muons; when > 0 FoM rows
                      at (M, W_default) are appended.
    """
    if ratio_label is None:
        ratio_label = ratio_key if isinstance(ratio_key, float) else float("nan")

    rows: list[dict] = []
    for M in M_values:
        # NC coverage
        ssd_nc = np.array([_ssd_nc_frac(s.ssd_results[ratio_key], M) for s in setups])
        pmt_nc = np.array([_pmt_nc_frac(s, M) for s in setups])
        pr, pp = _safe_pearsonr(ssd_nc, pmt_nc)
        sr, sp = _safe_spearmanr(ssd_nc, pmt_nc)
        rows.append({
            "ratio": ratio_label, "M": M, "metric": "nc_coverage",
            "pearson_r": pr, "pearson_p": pp,
            "spearman_rho": sr, "spearman_p": sp,
            "n": len(setups),
        })

        # Recall at W_default
        ssd_rec = np.array([_ssd_recall(s.ssd_results[ratio_key], M, W_default)
                            for s in setups])
        pmt_rec = np.array([_pmt_recall(s, M, W_default) for s in setups])
        pr, pp = _safe_pearsonr(ssd_rec, pmt_rec)
        sr, sp = _safe_spearmanr(ssd_rec, pmt_rec)
        rows.append({
            "ratio": ratio_label, "M": M, "metric": "recall",
            "pearson_r": pr, "pearson_p": pp,
            "spearman_rho": sr, "spearman_p": sp,
            "n": len(setups),
        })

        # FoM at (M, W_default) — same W for both SSD and PMT
        if total_primaries > 0:
            ssd_fv = np.array([_ssd_fom(s.ssd_results[ratio_key], M, W_default,
                                        total_primaries) for s in setups])
            pmt_fv = np.array([_pmt_fom(s, M, W_default, total_primaries)
                                for s in setups])
            mask = np.isfinite(ssd_fv) & np.isfinite(pmt_fv)
            if mask.sum() >= 3:
                pr, pp = _safe_pearsonr(ssd_fv[mask], pmt_fv[mask])
                sr, sp = _safe_spearmanr(ssd_fv[mask], pmt_fv[mask])
            else:
                pr = pp = sr = sp = float("nan")
            rows.append({
                "ratio": ratio_label, "M": M, "metric": "fom",
                "pearson_r": pr, "pearson_p": pp,
                "spearman_rho": sr, "spearman_p": sp,
                "n": int(mask.sum()),
            })

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
    """Sweep area ratio uniformly across all layers; return corr_df and optimal ratio."""
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
                raw_rows, raw_cols, raw_vals,
                num_ncs, num_voxels, layers,
                area_ratios, m, M_values, W_values,
                alignment, setup_voxel_subsets[i],
                seed=seed,
            )

        corr_rows = compute_correlations(setups, ratio_factor, M_values, W_default,
                                         ratio_label=ratio_factor,
                                         total_primaries=total_primaries)
        all_rows.extend(corr_rows)

        # Early stop: track mean |Pearson r| for nc_coverage at M = 1..4
        nc_rows  = [r for r in corr_rows
                    if r["metric"] == "nc_coverage" and 1 <= r["M"] <= 4]
        abs_rs   = [abs(r["pearson_r"]) for r in nc_rows
                    if not math.isnan(r["pearson_r"])]
        mean_abs = float(np.mean(abs_rs)) if abs_rs else 0.0
        print(f"mean|r|={mean_abs:.3f}")

        if recent_mean_abs and mean_abs < recent_mean_abs[-1]:
            patience_count += 1
        else:
            patience_count = 0
        recent_mean_abs.append(mean_abs)

        if patience_count >= early_stop_patience:
            print(f"  Early stop at ratio={ratio_factor:.2f} "
                  f"(mean|r| decreased for {early_stop_patience} consecutive steps)")
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
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Sweep one layer's ratio at a time; all others fixed at optimal_global_ratio."""
    raw_rows, raw_cols, raw_vals, voxel_ids, _, layers, num_ncs, _ = ssd_coo_data
    num_voxels = len(voxel_ids)

    all_rows:       list[dict]        = []
    per_layer_optima: dict[str, float] = {}

    for layer in ["pit", "bot", "top", "wall"]:
        print(f"  Layer: {layer}")
        layer_rows: list[dict] = []

        for ratio_factor in ratio_factors:
            ratio_factor = float(ratio_factor)
            area_ratios  = {lyr: optimal_global_ratio for lyr in DEFAULT_AREA_RATIOS}
            area_ratios[layer] = ratio_factor
            temp_key = (layer, ratio_factor)

            for i, setup in enumerate(setups):
                setup.ssd_results[temp_key] = evaluate_ssd_at_ratio(
                    raw_rows, raw_cols, raw_vals,
                    num_ncs, num_voxels, layers,
                    area_ratios, m, M_values, W_values,
                    alignment, setup_voxel_subsets[i],
                    seed=seed,
                )

            cr = compute_correlations(setups, temp_key, M_values, W_default,
                                      ratio_label=ratio_factor,
                                      total_primaries=total_primaries)
            for row in cr:
                row["layer"] = layer
            layer_rows.extend(cr)

            # Free temp results immediately
            for setup in setups:
                del setup.ssd_results[temp_key]
            gc.collect()

        all_rows.extend(layer_rows)

        # Optimal for this layer
        nc_layer = [r for r in layer_rows
                    if r["metric"] == "nc_coverage" and 1 <= r["M"] <= 4]
        if nc_layer:
            df_l = pd.DataFrame(nc_layer)
            grp  = df_l.groupby("ratio")["pearson_r"].mean().dropna()
            per_layer_optima[layer] = (
                float(grp.idxmax()) if not grp.empty else optimal_global_ratio
            )
        else:
            per_layer_optima[layer] = optimal_global_ratio
        print(f"    → Optimal {layer} ratio: {per_layer_optima[layer]:.2f}")

    return pd.DataFrame(all_rows), per_layer_optima


# ─────────────────────────────────────────────────────────────────────
# SECTION 12 — Plot functions
# ─────────────────────────────────────────────────────────────────────

def plot_corr_vs_ratio(
    corr_df: pd.DataFrame,
    metric: str,
    M_values: list[int],
    W_default: int,
    output_dir: str,
    optimal_ratio: float,
) -> None:
    """One panel per M: Pearson r and Spearman ρ vs area ratio."""
    df = corr_df[corr_df["metric"] == metric].copy()
    if df.empty:
        print(f"  [SKIP] No data for metric={metric!r}")
        return

    n_cols = min(5, len(M_values))
    n_rows = math.ceil(len(M_values) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.0 * n_cols, 3.2 * n_rows),
                              squeeze=False)

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

        ax.set_title(f"M={M}", fontsize=9)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Area ratio", fontsize=7)
        ax.set_ylabel("Correlation", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=6, loc="lower right")

    for idx in range(len(M_values), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    if metric == "nc_coverage":
        m_label = "NC Coverage"
    elif metric == "recall":
        m_label = f"Recall (W={W_default})"
    else:
        m_label = f"FoM (W={W_default})"
    fig.suptitle(f"SSD–PMT Correlation vs. Area Ratio — {m_label}", fontsize=11)
    fig.tight_layout()

    fname = f"corr_vs_ratio_{metric}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def _scatter_panel(
    ax: plt.Axes,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    color_pts: list[str],
    labels: list[str],
    x_label: str,
    y_label: str,
) -> None:
    """Scatter + y=x reference + OLS regression + Pearson/Spearman stats box."""
    n = len(x_arr)

    for x, y, c, lbl in zip(x_arr, y_arr, color_pts, labels):
        ax.scatter([x], [y], color=c, s=55, zorder=3)
        ax.annotate(lbl, xy=(x, y), xytext=(4, 3),
                    textcoords="offset points", fontsize=6, color=c)

    # y = x reference line
    all_vals = np.concatenate([x_arr, y_arr])
    if len(all_vals) > 0:
        vmin = float(all_vals.min())
        vmax = float(all_vals.max())
        pad  = 0.05 * max(vmax - vmin, 1e-9)
        ref  = np.array([vmin - pad, vmax + pad])
        ax.plot(ref, ref, color="gray", linewidth=0.8, linestyle="--",
                alpha=0.5, zorder=1)

    # Stats box
    if n >= 3 and np.std(x_arr) > 0 and np.std(y_arr) > 0:
        try:
            r_v, p_r   = scipy_stats.pearsonr(x_arr, y_arr)
            rho, p_rho = scipy_stats.spearmanr(x_arr, y_arr)
            ann = (f"Pearson  r = {r_v:+.3f}  (p={p_r:.3g})\n"
                   f"Spearman ρ = {rho:+.3f}  (p={p_rho:.3g})")
        except ValueError:
            ann = "constant data — no stats"
    elif n < 3:
        ann = "n < 3 — no stats"
    else:
        ann = "constant data — no stats"

    ax.text(0.03, 0.97, ann, transform=ax.transAxes,
            ha="left", va="top", fontsize=7, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.85))

    # OLS line + 95 % CI
    if n >= 3 and np.std(x_arr) > 0 and np.std(y_arr) > 0:
        slope, intercept, *_ = scipy_stats.linregress(x_arr, y_arr)
        x_fit  = np.linspace(x_arr.min(), x_arr.max(), 200)
        y_fit  = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color="black", linewidth=1.2,
                linestyle="--", zorder=2)

        y_pred    = slope * x_arr + intercept
        residuals = y_arr - y_pred
        se        = np.sqrt(np.sum(residuals ** 2) / max(n - 2, 1))
        x_mean    = x_arr.mean()
        t_crit    = scipy_stats.t.ppf(0.975, df=max(n - 2, 1))
        ci = t_crit * se * np.sqrt(
            1 / n + (x_fit - x_mean) ** 2
            / np.sum((x_arr - x_mean) ** 2)
        )
        ax.fill_between(x_fit, y_fit - ci, y_fit + ci, color="black", alpha=0.08)

    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.grid(alpha=0.3)


def plot_scatter_ssd_pmt(
    setups: list[SetupData],
    ratio_factor: float,
    metric: str,
    M_values: list[int],
    W_default: int,
    output_dir: str,
) -> None:
    """SSD vs PMT scatter grid at the given ratio, one panel per M."""
    valid = [s for s in setups if ratio_factor in s.ssd_results]
    if not valid:
        warnings.warn(
            f"No SSD results at ratio={ratio_factor:.2f} — skipping scatter plot."
        )
        return

    colors  = _colors(len(setups))
    c_valid = [colors[setups.index(s)] for s in valid]
    l_valid = [s.label for s in valid]

    n_cols = min(5, len(M_values))
    n_rows = math.ceil(len(M_values) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 4.0 * n_rows),
                              squeeze=False)

    if metric == "nc_coverage":
        x_base       = "SSD NC fraction"
        y_base       = "PMT NC fraction"
        title_metric = "NC Coverage"
    else:
        x_base       = "SSD Recall"
        y_base       = "PMT Recall"
        title_metric = f"Recall (W={W_default})"

    for idx, M in enumerate(M_values):
        ax = axes[idx // n_cols][idx % n_cols]

        if metric == "nc_coverage":
            x_arr = np.array([_ssd_nc_frac(s.ssd_results[ratio_factor], M) for s in valid])
            y_arr = np.array([_pmt_nc_frac(s, M) for s in valid])
        else:
            x_arr = np.array([_ssd_recall(s.ssd_results[ratio_factor], M, W_default)
                              for s in valid])
            y_arr = np.array([_pmt_recall(s, M, W_default) for s in valid])

        _scatter_panel(ax, x_arr, y_arr, c_valid, l_valid,
                       f"{x_base} (M={M})", f"{y_base} (M={M})")
        ax.set_title(f"M={M}", fontsize=9)

    for idx in range(len(M_values), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(
        f"SSD vs PMT — {title_metric}  |  ratio={ratio_factor:.2f}", fontsize=11
    )
    fig.tight_layout()

    fname = f"scatter_{metric}_ratio{ratio_factor:.1f}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_w2_comparison(
    setups: list[SetupData],
    ratio_factor: float,
    M_values: list[int],
    W_default: int,
    output_dir: str,
) -> None:
    """W2 vs NC fraction: dashed markers = SSD, solid markers = PMT."""
    w2_setups = [s for s in setups
                 if s.w2 is not None and ratio_factor in s.ssd_results]
    if len(w2_setups) < 2:
        print("  Skipping W2 comparison: fewer than 2 setups with W2 values.")
        return

    m_panels = [M for M in [1, 2, 4, 5, 10] if M in M_values]
    if not m_panels:
        m_panels = M_values[:5]

    n_cols = min(5, len(m_panels))
    n_rows = math.ceil(len(m_panels) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 4.0 * n_rows),
                              squeeze=False)
    colors = _colors(len(w2_setups))

    for idx, M in enumerate(m_panels):
        ax      = axes[idx // n_cols][idx % n_cols]
        w2_arr  = np.array([s.w2 for s in w2_setups], dtype=float)
        ssd_arr = np.array([_ssd_nc_frac(s.ssd_results[ratio_factor], M)
                            for s in w2_setups])
        pmt_arr = np.array([_pmt_nc_frac(s, M) for s in w2_setups])

        for i, (s, c) in enumerate(zip(w2_setups, colors)):
            ax.scatter([s.w2], [ssd_arr[i]], color=c, marker="^",
                       s=60, zorder=3, alpha=0.9)
            ax.scatter([s.w2], [pmt_arr[i]], color=c, marker="o",
                       s=60, zorder=3, alpha=0.9)
            ax.annotate(s.label, xy=(s.w2, pmt_arr[i]), xytext=(4, 3),
                        textcoords="offset points", fontsize=6, color=c)

        # Regression lines — SSD dashed, PMT solid
        if len(w2_setups) >= 3 and np.std(w2_arr) > 0:
            x_fit = np.linspace(w2_arr.min(), w2_arr.max(), 200)
            for vals, lstyle in [(ssd_arr, "--"), (pmt_arr, "-")]:
                if np.std(vals) > 0:
                    s_l, icpt, *_ = scipy_stats.linregress(w2_arr, vals)
                    ax.plot(x_fit, s_l * x_fit + icpt, color="black",
                            linewidth=1.0, linestyle=lstyle, alpha=0.7)

        ax.set_title(f"M={M}", fontsize=9)
        ax.set_xlabel("W2 (mm)", fontsize=8)
        ax.set_ylabel("NC fraction", fontsize=8)
        ax.grid(alpha=0.3)

        if idx == 0:
            legend_els = [
                Line2D([0], [0], marker="^", color="gray", linestyle="None",
                       markersize=7, label=f"SSD (ratio={ratio_factor:.2f})"),
                Line2D([0], [0], marker="o", color="gray", linestyle="None",
                       markersize=7, label="PMT"),
                Line2D([0], [0], color="black", linestyle="--",
                       linewidth=1, label="SSD regression"),
                Line2D([0], [0], color="black", linestyle="-",
                       linewidth=1, label="PMT regression"),
            ]
            ax.legend(handles=legend_els, fontsize=7)

    for idx in range(len(m_panels), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(
        f"W2 vs NC Fraction — SSD vs PMT  |  ratio={ratio_factor:.2f}", fontsize=11
    )
    fig.tight_layout()

    fname = "w2_coverage_comparison.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 13a — FoM scatter: SSD vs PMT at optimal ratio
# ─────────────────────────────────────────────────────────────────────

def plot_scatter_ssd_pmt_fom(
    setups: list[SetupData],
    ratio_factor: float,
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    output_dir: str,
) -> None:
    """Scatter of SSD-FoM vs PMT-FoM at *ratio_factor*.

    For each setup the best (M, W) is determined by maximising PMT FoM;
    the same (M, W) is then used to extract the SSD FoM — ensuring a
    fair like-for-like comparison.

    One plot: one labelled point per setup.
    """
    valid = [s for s in setups if ratio_factor in s.ssd_results]
    if not valid:
        warnings.warn(
            f"No SSD results at ratio={ratio_factor:.2f} — skipping FoM scatter."
        )
        return

    colors  = _colors(len(setups))
    c_valid = [colors[setups.index(s)] for s in valid]

    x_arr = []
    y_arr = []
    labels = []
    best_mw_labels = []

    for s in valid:
        best_mw = _pmt_best_mw(s, M_values, W_values, total_primaries)
        M_b, W_b = best_mw
        pmt_val = _pmt_fom(s, M_b, W_b, total_primaries)
        ssd_val = _ssd_fom(s.ssd_results[ratio_factor], M_b, W_b, total_primaries)
        x_arr.append(ssd_val)
        y_arr.append(pmt_val)
        labels.append(s.label)
        best_mw_labels.append(f"M{M_b}W{W_b}")

    x_arr = np.array(x_arr, dtype=float)
    y_arr = np.array(y_arr, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter_panel(
        ax, x_arr, y_arr, c_valid, labels,
        f"SSD FoM  (ratio={ratio_factor:.2f})", "PMT FoM",
    )
    # Annotate each point with its best (M,W)
    for x, y, mw in zip(x_arr, y_arr, best_mw_labels):
        if np.isfinite(x) and np.isfinite(y):
            ax.annotate(
                mw, xy=(x, y), xytext=(4, -9),
                textcoords="offset points", fontsize=6, color="dimgray",
            )

    ax.set_title(
        f"SSD vs PMT — Figure of Merit  |  ratio={ratio_factor:.2f}\n"
        f"(M,W) chosen per setup to maximise PMT FoM",
        fontsize=10,
    )
    fig.tight_layout()
    fname = f"scatter_fom_ratio{ratio_factor:.1f}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 13b — W2 Spearman plot for FoM
# ─────────────────────────────────────────────────────────────────────

def plot_w2_spearman_fom(
    setups: list[SetupData],
    M_values: list[int],
    W_values: list[int],
    total_primaries: int,
    optimal_ratio: float,
    output_dir: str,
) -> None:
    """Spearman ρ(W2, FoM) vs M — one line each for PMT and SSD.

    For each M the best W is determined per setup by maximising PMT FoM at
    that M; the same W is then applied to extract SSD FoM, so both series
    use identical (M, W) per setup.

    Only setups that have a W2 value are included.
    """
    w2_setups = [s for s in setups
                 if s.w2 is not None and optimal_ratio in s.ssd_results]
    if len(w2_setups) < 3:
        print("  [SKIP] w2_spearman_fom: fewer than 3 setups have W2 + SSD results.")
        return

    w2_arr = np.array([s.w2 for s in w2_setups], dtype=float)

    rho_pmt, rho_ssd = [], []
    p_pmt,   p_ssd   = [], []

    for M in M_values:
        pmt_fom_arr = []
        ssd_fom_arr = []

        for s in w2_setups:
            W_best = _pmt_best_w_at_m(s, M, W_values, total_primaries)
            pmt_fom_arr.append(_pmt_fom(s, M, W_best, total_primaries))
            ssd_fom_arr.append(
                _ssd_fom(s.ssd_results[optimal_ratio], M, W_best, total_primaries)
            )

        pmt_fom_arr = np.array(pmt_fom_arr, dtype=float)
        ssd_fom_arr = np.array(ssd_fom_arr, dtype=float)

        mask_p = np.isfinite(pmt_fom_arr) & np.isfinite(w2_arr)
        mask_s = np.isfinite(ssd_fom_arr) & np.isfinite(w2_arr)

        if mask_p.sum() >= 3:
            r1, p1 = _safe_spearmanr(w2_arr[mask_p], pmt_fom_arr[mask_p])
        else:
            r1, p1 = float("nan"), float("nan")
        if mask_s.sum() >= 3:
            r2, p2 = _safe_spearmanr(w2_arr[mask_s], ssd_fom_arr[mask_s])
        else:
            r2, p2 = float("nan"), float("nan")

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
        sig  = ps < 0.05
        ax.plot(x, rhos, color=color, linewidth=1.5, label=label)
        finite = np.isfinite(rhos)
        if (sig & finite).any():
            ax.scatter(x[sig & finite], rhos[sig & finite], color=color,
                       s=60, marker=marker, zorder=4, label=f"{label} (p<0.05)")
        if (~sig & finite).any():
            ax.scatter(x[~sig & finite], rhos[~sig & finite],
                       facecolors="none", edgecolors=color,
                       s=60, marker=marker, linewidth=1.2, zorder=4)

    ax.set_xlabel("Multiplicity threshold M", fontsize=11)
    ax.set_ylabel("Spearman ρ  (W2 vs FoM)", fontsize=11)
    ax.set_title(
        "Spearman Correlation: W2 vs FoM — PMT vs SSD\n"
        "(filled = p<0.05; W per setup = argmax PMT FoM at that M)",
        fontsize=11,
    )
    ax.set_xticks(M_values)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fname = "w2_spearman_fom.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 13 — Text and CSV output
# ─────────────────────────────────────────────────────────────────────

def write_summary(
    corr_df: pd.DataFrame,
    per_layer_corr_df: Optional[pd.DataFrame],
    per_layer_optima: Optional[dict[str, float]],
    optimal_ratio: float,
    n_setups: int,
    output_dir: str,
) -> None:
    lines = [
        "SSD–PMT Correlation Analysis Summary",
        "=" * 45,
        f"N setups      : {n_setups}",
        f"Optimal ratio : {optimal_ratio:.4f}",
        "",
    ]

    if per_layer_optima:
        lines.append("Per-layer optimal ratios:")
        for lyr, r in per_layer_optima.items():
            lines.append(f"  {lyr:>4}: {r:.4f}")
        lines.append("")

    if not corr_df.empty:
        nc_df = corr_df[corr_df["metric"] == "nc_coverage"].dropna(subset=["pearson_r"])
        if not nc_df.empty:
            best = nc_df.loc[nc_df["pearson_r"].abs().idxmax()]
            lines += [
                "Best global sweep result (nc_coverage):",
                f"  ratio={best['ratio']:.2f}  M={best['M']}  "
                f"Pearson r={best['pearson_r']:.4f} (p={best['pearson_p']:.4g})  "
                f"Spearman ρ={best['spearman_rho']:.4f}",
                "",
            ]

    with open(os.path.join(output_dir, "correlation_summary.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("  Saved: correlation_summary.txt")

    if not corr_df.empty:
        saved_csvs = []
        for metric_name in ["nc_coverage", "recall", "fom"]:
            sub = corr_df[corr_df["metric"] == metric_name]
            if sub.empty:
                continue
            csv_path = os.path.join(output_dir, f"corr_vs_ratio_{metric_name}.csv")
            sub.to_csv(csv_path, index=False)
            saved_csvs.append(f"corr_vs_ratio_{metric_name}.csv")
        print(f"  Saved: {', '.join(saved_csvs)}")

    if per_layer_corr_df is not None and not per_layer_corr_df.empty:
        per_layer_corr_df.to_csv(
            os.path.join(output_dir, "corr_vs_ratio_per_layer.csv"), index=False
        )
        print("  Saved: corr_vs_ratio_per_layer.csv")


# ─────────────────────────────────────────────────────────────────────
# SECTION 14 — main
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
    p.add_argument("--ratio-min",      type=float, default=1.0,
                   help="Start of ratio sweep (default: 1.0)")
    p.add_argument("--ratio-max",      type=float, default=10.0,
                   help="End of ratio sweep (default: 10.0)")
    p.add_argument("--ratio-step",     type=float, default=0.1,
                   help="Step size for ratio sweep (default: 0.1)")
    p.add_argument("--output-dir",     default="./correlation_results",
                   help="Output directory (default: ./correlation_results)")
    p.add_argument("--seed",           type=int, default=42,
                   help="Random seed for stochastic rounding (default: 42)")
    p.add_argument("--skip-per-layer", action="store_true",
                   help="Skip the per-layer ratio sweep")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    M_values = list(range(1, args.M_max + 1))
    W_values = list(range(1, args.W_max + 1))

    # ── 1. NC truth ───────────────────────────────────────────────────
    print("Loading NC truth ...")
    nc_truth = build_nc_truth(args.muon_dir)

    # ── 2. SSD COO data (loaded once, reused across all ratio steps) ──
    print("\nLoading SSD raw sparse data ...")
    ssd_coo_data = load_raw_sparse(args.ssd_hdf5, skip_validity=True)
    _, _, _, voxel_ids, _, _, _, _ = ssd_coo_data
    ssd_voxel_id_to_col: dict[str, int] = {
        vid: i for i, vid in enumerate(voxel_ids)
    }

    # ── 3. SSD muon alignment ─────────────────────────────────────────
    print("\nBuilding SSD muon alignment ...")
    alignment = align_ssd_to_pmt(args.ssd_hdf5, nc_truth)

    # ── 3b. Total primaries for FoM ───────────────────────────────────
    _n_runs = _count_runs(args.muon_dir)
    total_primaries = _n_runs * MUONS_PER_RUN_DIR
    _runtime_h  = total_primaries / MUSUN_RATE
    _runtime_yr = _runtime_h / (24 * 365.25)
    print(f"\n  FoM: total primary muons = {total_primaries:,}  "
          f"({_n_runs} runs × {MUONS_PER_RUN_DIR:,})")
    print(f"  FoM: simulated livetime  = {_runtime_h:,.0f} h  "
          f"=  {_runtime_yr:.2f} yr  (at {MUSUN_RATE} µ/h)")

    # ── 4. Warn if few setups ─────────────────────────────────────────
    if len(args.setup_dirs) < 5:
        warnings.warn(
            f"Only {len(args.setup_dirs)} setups — "
            "correlation statistics may not be meaningful."
        )

    # ── 5. Load each setup ────────────────────────────────────────────
    setups: list[SetupData]         = []
    setup_voxel_subsets: list[np.ndarray] = []

    for setup_dir in args.setup_dirs:
        label = Path(setup_dir).name
        print(f"\nSetup: {label}")

        json_files = sorted(glob.glob(os.path.join(setup_dir, "*.json")))
        if not json_files:
            raise FileNotFoundError(f"No *.json config found in {setup_dir!r}")
        json_path = json_files[0]

        json_ids = _load_json_voxel_ids(json_path)
        missing  = [v for v in json_ids if v not in ssd_voxel_id_to_col]
        if missing:
            warnings.warn(
                f"{label}: {len(missing)}/{len(json_ids)} voxel IDs not found "
                "in SSD HDF5 — those voxels will be excluded from SSD evaluation."
            )
        voxel_col_subset = np.array(
            [ssd_voxel_id_to_col[v] for v in json_ids if v in ssd_voxel_id_to_col],
            dtype=np.int32,
        )

        print("  Loading PMT simulation data ...")
        pmt_nc, pmt_muon, _ = load_pmt_setup(
            setup_dir, nc_truth, M_values, W_values, m_threshold=args.m
        )
        w2     = compute_setup_w2(json_path)
        w2_str = f"{w2:.2f}" if w2 is not None else "N/A"
        print(f"  W2={w2_str}, {len(voxel_col_subset)} SSD voxels mapped")

        setups.append(
            SetupData(label=label, w2=w2, pmt_nc=pmt_nc, pmt_muon=pmt_muon)
        )
        setup_voxel_subsets.append(voxel_col_subset)

    # ── 6. Global ratio sweep ─────────────────────────────────────────
    ratio_factors = np.arange(
        args.ratio_min,
        args.ratio_max + args.ratio_step / 2,
        args.ratio_step,
    )
    print(
        f"\nGlobal ratio sweep "
        f"[{args.ratio_min:.2f}, {args.ratio_max:.2f}] "
        f"step={args.ratio_step:.2f} ({len(ratio_factors)} steps) ..."
    )
    corr_df, optimal_ratio = global_ratio_sweep(
        setups, ssd_coo_data, setup_voxel_subsets, alignment,
        ratio_factors, args.m, M_values, W_values, args.W_default,
        seed=args.seed, total_primaries=total_primaries,
    )

    # ── 7. Per-layer ratio sweep ──────────────────────────────────────
    per_layer_corr_df: Optional[pd.DataFrame]    = None
    per_layer_optima:  Optional[dict[str, float]] = None
    if not args.skip_per_layer:
        print("\nPer-layer ratio sweep ...")
        per_layer_corr_df, per_layer_optima = per_layer_ratio_sweep(
            setups, ssd_coo_data, setup_voxel_subsets, alignment,
            optimal_ratio, ratio_factors, args.m, M_values, W_values,
            args.W_default, seed=args.seed, total_primaries=total_primaries,
        )

    # ── 8. Plots ──────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_corr_vs_ratio(corr_df, "nc_coverage", M_values,
                       args.W_default, args.output_dir, optimal_ratio)
    plot_corr_vs_ratio(corr_df, "recall", M_values,
                       args.W_default, args.output_dir, optimal_ratio)
    plot_corr_vs_ratio(corr_df, "fom", M_values,
                       args.W_default, args.output_dir, optimal_ratio)
    plot_scatter_ssd_pmt(setups, optimal_ratio, "nc_coverage",
                         M_values, args.W_default, args.output_dir)
    plot_scatter_ssd_pmt(setups, optimal_ratio, "recall",
                         M_values, args.W_default, args.output_dir)
    plot_scatter_ssd_pmt_fom(setups, optimal_ratio, M_values, W_values,
                              total_primaries, args.output_dir)
    plot_w2_comparison(setups, optimal_ratio, M_values,
                       args.W_default, args.output_dir)
    plot_w2_spearman_fom(setups, M_values, W_values, total_primaries,
                         optimal_ratio, args.output_dir)

    # ── 9. Summary ────────────────────────────────────────────────────
    print("\nWriting summary ...")
    write_summary(
        corr_df, per_layer_corr_df, per_layer_optima,
        optimal_ratio, len(setups), args.output_dir,
    )
    print(f"\nDone. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
