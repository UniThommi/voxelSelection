"""NC and muon coverage analysis functions for raw LGDO data.

Works with sparse NC×PMT binary matrices produced by raw_loading.build_pmt_matrix()
and the nc_truth DataFrame from raw_loading.build_nc_truth().

The muon-classification logic mirrors comparePMTCoverage.py exactly:
  - A muon is Ge77 if any of its NCs has flag_ge77 == 1.
  - Muon is classified as Ge77 if it has ≥W detected NCs in [1µs, 200µs].
  - "Detected" means the NC's PMT multiplicity (row-sum of B) ≥ M.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ──────────────────────────────────────────────────────────────────────
# Constants (same as comparePMTCoverage.py)
# ──────────────────────────────────────────────────────────────────────
MUON_WINDOW_LO_NS: float = 1_000.0    # 1 µs
MUON_WINDOW_HI_NS: float = 200_000.0  # 200 µs


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def compute_nc_multiplicities(B: sp.spmatrix) -> np.ndarray:
    """Return per-NC multiplicity array (number of firing PMTs per NC)."""
    return np.asarray(B.sum(axis=1)).ravel().astype(np.int32)


def compute_metrics(
    tp: int, fp: int, fn: int
) -> dict[str, float]:
    """Compute Recall and Precision for Ge-77 muon classification."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "Precision": prec,
        "Recall":    rec,
    }


# ──────────────────────────────────────────────────────────────────────
# NC analysis
# ──────────────────────────────────────────────────────────────────────
def evaluate_nc(
    B: sp.spmatrix,
    nc_truth: pd.DataFrame,
    M_values: list[int],
    detect_info: dict | None = None,
) -> dict[str, Any]:
    """Compute NC detection statistics for each M threshold.

    Parameters
    ----------
    B:
        Sparse NC×PMT binary matrix from build_pmt_matrix().
    nc_truth:
        Shared NC truth DataFrame from build_nc_truth(); rows must align
        with B rows.
    M_values:
        List of PMT multiplicity thresholds to evaluate.
    detect_info:
        Optional dict from build_pmt_matrix() with boolean arrays:
          "nc_any_photon"   — NC has ≥1 photon at any time
          "nc_within_200ns" — NC has ≥1 photon within the time window
        If None, these detectability categories are not computed.

    Returns
    -------
    dict with keys:
        nc_total               : int
        nc_any_photon          : int  (≥1 photon at any time; -1 if unknown)
        nc_within_200ns        : int  (≥1 photon in time window; -1 if unknown)
        nc_only_outside_200ns  : int  (photons exist but all late; -1 if unknown)
        nc_detected            : dict[M → int]
        ge77_nc_total          : int
        ge77_nc_any_photon     : int  (-1 if unknown)
        ge77_nc_within_200ns   : int  (-1 if unknown)
        ge77_nc_only_outside   : int  (-1 if unknown)
        ge77_nc_detected       : dict[M → int]
        multiplicity_counts    : np.ndarray[int32], shape (n_nc,)
    """
    mults    = compute_nc_multiplicities(B)
    ge77_mask = nc_truth["flag_ge77"].values == 1

    # ── detectability categories ──────────────────────────────────────
    if detect_info is not None:
        any_ph   = detect_info["nc_any_photon"]
        within   = detect_info["nc_within_200ns"]
        only_out = any_ph & ~within

        nc_any_photon          = int(any_ph.sum())
        nc_within_200ns        = int(within.sum())
        nc_only_outside_200ns  = int(only_out.sum())
        ge77_nc_any_photon     = int((any_ph   & ge77_mask).sum())
        ge77_nc_within_200ns   = int((within   & ge77_mask).sum())
        ge77_nc_only_outside   = int((only_out & ge77_mask).sum())
    else:
        nc_any_photon = nc_within_200ns = nc_only_outside_200ns = -1
        ge77_nc_any_photon = ge77_nc_within_200ns = ge77_nc_only_outside = -1

    result: dict[str, Any] = {
        "nc_total":              len(nc_truth),
        "nc_any_photon":         nc_any_photon,
        "nc_within_200ns":       nc_within_200ns,
        "nc_only_outside_200ns": nc_only_outside_200ns,
        "nc_detected":           {},
        "ge77_nc_total":         int(ge77_mask.sum()),
        "ge77_nc_any_photon":    ge77_nc_any_photon,
        "ge77_nc_within_200ns":  ge77_nc_within_200ns,
        "ge77_nc_only_outside":  ge77_nc_only_outside,
        "ge77_nc_detected":      {},
        "multiplicity_counts":   mults,
    }

    for M in M_values:
        detected = mults >= M
        result["nc_detected"][M]      = int(detected.sum())
        result["ge77_nc_detected"][M] = int((detected & ge77_mask).sum())

    return result


# ──────────────────────────────────────────────────────────────────────
# Muon analysis
# ──────────────────────────────────────────────────────────────────────
def evaluate_muon(
    B: sp.spmatrix,
    nc_truth: pd.DataFrame,
    M_values: list[int],
    W_values: list[int],
    detect_info: dict | None = None,
) -> dict[str, Any]:
    """Compute muon Ge77 classification results for all (M, W) combinations.

    Classification logic (identical to comparePMTCoverage.py):
      1. A muon is Ge77 (ground truth) if any of its NCs has flag_ge77==1.
      2. For threshold M: a NC is "detected" if its multiplicity (row sum
         of B) ≥ M.
      3. Count detected NCs per muon restricted to the time window
         [1 µs, 200 µs] = [MUON_WINDOW_LO_NS, MUON_WINDOW_HI_NS].
      4. Classify muon as Ge77 if its count ≥ W.

    Parameters
    ----------
    B, nc_truth, M_values, W_values: see evaluate_nc.
    detect_info : optional dict from build_pmt_matrix() with boolean arrays:
        "nc_any_photon"   — NC has ≥1 photon at any time
        "nc_within_200ns" — NC has ≥1 photon within the 200 ns time cut
        When provided, muon-level detectability counts are included in the
        returned dict under the key ``ge77_muon_detectability``.

    Returns
    -------
    dict with keys:
        muon_stats            : dict  — total, n_ge77, n_non_ge77 counts
        confusion             : dict[(M, W) → {"TP", "FP", "TN", "FN"}]
        w_hist                : dict[M → {"ge77": list[int], "non_ge77": list[int]}]
        ge77_muon_detectability : dict with keys:
            "any_photon"   — Ge77 muons where ≥1 NC has any photon (-1 if unknown)
            "within_200ns" — Ge77 muons where ≥1 NC has a photon within 200 ns (-1 if unknown)
            These are the theoretical upper bounds for muon-level detection,
            consistent with the TP metric in ``confusion``.
    """
    mults = compute_nc_multiplicities(B)

    # ── derive muon ground truth ──────────────────────────────────────
    # muon_id (evtid) repeats across runs → use (run_id, muon_id) as composite key.
    # A muon is Ge77 if ANY of its NCs has flag_ge77 == 1.
    muon_is_ge77: pd.Series = (
        nc_truth.groupby(["run_id", "muon_id"])["flag_ge77"]
        .apply(lambda flags: bool((flags == 1).any()))
    )
    unique_muon_idx = muon_is_ge77.index   # MultiIndex of (run_id, muon_id)
    ge77_truth      = muon_is_ge77.values.astype(bool)

    n_ge77     = int(ge77_truth.sum())
    n_non_ge77 = len(unique_muon_idx) - n_ge77

    muon_stats = {
        "total":      len(unique_muon_idx),
        "n_ge77":     n_ge77,
        "n_non_ge77": n_non_ge77,
    }

    # ── precompute in-window mask (shared for all M) ──────────────────
    in_window = (
        (nc_truth["nc_time_ns"].values >= MUON_WINDOW_LO_NS)
        & (nc_truth["nc_time_ns"].values <= MUON_WINDOW_HI_NS)
    )

    nc_work = pd.DataFrame({
        "run_id":    nc_truth["run_id"].values,
        "muon_id":   nc_truth["muon_id"].values,
        "in_window": in_window,
    })

    confusion: dict[tuple[int, int], dict] = {}
    w_hist:    dict[int, dict[str, list[int]]]        = {}

    # Number of Ge77 NCs per muon (all flag_ge77==1 NCs, no time-window cut).
    ge77_nc_counts_per_muon: np.ndarray = (
        nc_truth.groupby(["run_id", "muon_id"])["flag_ge77"]
        .sum()
        .reindex(unique_muon_idx, fill_value=0)
        .values.astype(np.int32)
    )

    for M in M_values:
        detected = (mults >= M).astype(np.int8)
        nc_work["det_in_window"] = detected & in_window

        w_per_muon: pd.Series = (
            nc_work
            .groupby(["run_id", "muon_id"])["det_in_window"]
            .sum()
        )
        w_counts = w_per_muon.reindex(unique_muon_idx, fill_value=0).values.astype(np.int32)

        w_hist[M] = {
            "ge77":     w_counts[ ge77_truth].tolist(),
            "non_ge77": w_counts[~ge77_truth].tolist(),
        }

        for W in W_values:
            classified_ge77 = w_counts >= W
            tp_mask = ge77_truth &  classified_ge77
            fn_mask = ge77_truth & ~classified_ge77
            tp = int(tp_mask.sum())
            fp = int((~ge77_truth &  classified_ge77).sum())
            fn = int(fn_mask.sum())
            confusion[(M, W)] = {
                "TP": tp, "FP": fp, "FN": fn,
                "tp_ge77_nc_counts": ge77_nc_counts_per_muon[tp_mask],
                "fn_ge77_nc_counts": ge77_nc_counts_per_muon[fn_mask],
            }

    # ── muon-level detectability upper bounds (requires detect_info) ─────
    # For each Ge77 muon: does it have at least one NC with a photon?
    # Uses the same muon ordering as the confusion matrix (unique_muon_idx).
    ge77_muon_detectability: dict[str, int] = {"any_photon": -1, "within_200ns": -1}
    if detect_info is not None:
        ph_df = pd.DataFrame({
            "run_id":  nc_truth["run_id"].values,
            "muon_id": nc_truth["muon_id"].values,
            "any_ph":  detect_info["nc_any_photon"].astype(np.int8),
            "within":  detect_info["nc_within_200ns"].astype(np.int8),
        })
        # max over NCs per muon: 1 if ANY NC in that muon has the flag
        muon_ph = (
            ph_df.groupby(["run_id", "muon_id"])[["any_ph", "within"]]
            .max()
            .reindex(unique_muon_idx, fill_value=0)
        )
        any_ph_muon = muon_ph["any_ph"].values.astype(bool)
        within_muon = muon_ph["within"].values.astype(bool)
        ge77_muon_detectability = {
            "any_photon":   int((ge77_truth & any_ph_muon).sum()),
            "within_200ns": int((ge77_truth & within_muon).sum()),
        }

    return {
        "muon_stats":              muon_stats,
        "confusion":               confusion,
        "w_hist":                  w_hist,
        "ge77_muon_detectability": ge77_muon_detectability,
        # Per-Ge77-muon Ge77-flag NC count (same order as w_hist[M]["ge77"]).
        # Used by statistical-limit computation in compare_coverages.py.
        "ge77_nc_counts":          ge77_nc_counts_per_muon[ge77_truth].tolist(),
    }
