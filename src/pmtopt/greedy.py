"""
Greedy voxel selection algorithms.

Implements Maximum k-Coverage and Partial Set Multi-Cover for
NC-level and Ge77-muon-level optimization targets.

Two modes for M>1 / W>1:
  - Priority mode (default): Always optimize for M=1 (W=1 in muon mode),
    but if a voxel can promote NCs from M-1 → M, prioritize it.
    In muon-ge77 mode, further prioritize voxels that simultaneously
    cause a muon to reach W detected NCs at multiplicity M.
    Priority: W-trigger > M-trigger > M=1 baseline.

  - Dynamic mode (--dynamic): Fallback to lower thresholds when no
    voxels can promote at the target level. M>1 and W>1 simultaneously
    is not supported in dynamic mode.
"""

import time

import numpy as np
from scipy import sparse


def muon_weight_delta(d: np.ndarray, k: float) -> np.ndarray:
    """
    Marginal gain of detecting the (d+1)-th NC of a muon under
    exponential saturation weighting.

    f(d) = 1 - exp(-d / k)
    Δf(d) = f(d+1) - f(d) = exp(-d/k) * (1 - exp(-1/k))

    Parameters
    ----------
    d : np.ndarray
        Current number of detected NCs per muon.
    k : float
        Saturation constant.

    Returns
    -------
    np.ndarray, dtype float32
    """
    c = 1.0 - np.exp(-1.0 / k)
    return (np.exp(-d.astype(np.float32) / k) * c).astype(np.float32)


def _apply_spacing(
    centers: np.ndarray,
    layers: np.ndarray,
    available: np.ndarray,
    best_voxel: int,
    min_spacing_sq: float,
    verbose: bool,
) -> None:
    """Apply minimum spacing constraint after selecting a voxel."""
    selected_center = centers[best_voxel]
    selected_layer = layers[best_voxel]
    same_layer_mask = (layers == selected_layer) & available
    same_layer_indices = np.where(same_layer_mask)[0]
    if len(same_layer_indices) > 0:
        diff = centers[same_layer_indices] - selected_center
        dist_sq = np.sum(diff ** 2, axis=1)
        too_close = same_layer_indices[dist_sq < min_spacing_sq]
        available[too_close] = False
        if verbose and len(too_close) > 0:
            print(f"       └─ spacing: excluded {len(too_close)} "
                  f"voxels on layer '{selected_layer}'")


def greedy_select_nc(
    B: sparse.csc_matrix,
    N: int,
    M: int,
    centers: np.ndarray | None = None,
    layers: np.ndarray | None = None,
    min_spacing: float = 0.0,
    muon_weight_k: float | None = None,
    nc_to_muon_local: np.ndarray | None = None,
    num_muons: int = 0,
    dynamic: bool = False,
    verbose: bool = True,
) -> tuple[list[int], list[float], np.ndarray, np.ndarray | None]:
    """
    Greedy voxel selection maximizing NC-level threshold coverage.

    For M>1 (non-dynamic): Always computes gain at M=1. If any voxel
    can promote NCs from M-1 → M, that voxel is prioritized (the one
    promoting the most NCs wins). Otherwise the best M=1 voxel is taken.

    For M>1 (dynamic): Falls back through threshold levels when no
    voxels can promote at the current level.

    Parameters
    ----------
    B : sparse.csc_matrix
        Binary matrix (NCs x voxels) in CSC format.
    N : int
        Number of voxels to select.
    M : int
        Multiplicity threshold for detection.
    centers : np.ndarray or None
        Voxel centers for spacing constraint.
    layers : np.ndarray or None
        Layer labels for spacing constraint.
    min_spacing : float
        Minimum distance between selected voxels on same layer (mm).
    muon_weight_k : float or None
        If set, use muon-level diminishing-returns weighting.
    nc_to_muon_local : np.ndarray or None
        Maps NC -> muon local index (required if muon_weight_k is set).
    num_muons : int
        Number of unique muons (required if muon_weight_k is set).
    dynamic : bool
        If True, use dynamic M fallback instead of priority logic.
    verbose : bool
        Print per-iteration progress.

    Returns
    -------
    selected : list[int]
        Column indices of selected voxels (in selection order).
    efficiencies : list[float]
        Cumulative detection efficiency after each selection step.
    coverage_counts : np.ndarray
        Final coverage count per NC.
    muon_det_counts : np.ndarray or None
        If muon_weight_k is set, number of detected NCs per muon.
    """
    num_ncs, num_voxels = B.shape

    if N > num_voxels:
        raise ValueError(f"N={N} exceeds number of voxels ({num_voxels})")

    enforce_spacing = (min_spacing > 0) and (centers is not None) and (layers is not None)
    min_spacing_sq = min_spacing ** 2

    coverage_counts = np.zeros(num_ncs, dtype=np.int16)
    available = np.ones(num_voxels, dtype=bool)
    selected: list[int] = []
    efficiencies: list[float] = []

    use_muon_weight = (muon_weight_k is not None
                       and nc_to_muon_local is not None
                       and num_muons > 0)
    if use_muon_weight:
        nc_detected_flag = np.zeros(num_ncs, dtype=bool)
        muon_det_counts = np.zeros(num_muons, dtype=np.int32)
    else:
        nc_detected_flag = None
        muon_det_counts = None

    use_priority_M = (M > 1) and not dynamic
    use_dynamic_M = (M > 1) and dynamic

    if verbose:
        spacing_str = f", min_spacing={min_spacing:.0f}mm" if enforce_spacing else ""
        weight_str = f", muon_weight_k={muon_weight_k:.2f}" if use_muon_weight else ""
        if use_dynamic_M:
            mode_str = " (dynamic M: fallback when no NCs at M-1)"
        elif use_priority_M:
            mode_str = " (priority M: M=1 baseline, M-1→M prioritized)"
        else:
            mode_str = ""
        print(f"\nGreedy selection (NC mode): N={N}, M={M}"
              f"{spacing_str}{weight_str}{mode_str}")
        print(f"{'Step':>4} | {'Voxel':>8} | {'Gain':>8} | "
              f"{'Detected':>10} | {'Efficiency':>10}")
        print("-" * 60)

    for step in range(N):
        t0 = time.time()
        phase_tag = ""

        if use_priority_M:
            # --- Priority mode ---
            # Always compute M=1 gain as baseline
            at_m1 = (coverage_counts == 0)
            if use_muon_weight:
                d_per_nc = muon_det_counts[nc_to_muon_local]
                weights_m1 = muon_weight_delta(d_per_nc, muon_weight_k)
                weights_m1[~at_m1] = 0.0
                gains_m1 = B.T.dot(weights_m1)
            else:
                gains_m1 = B.T.dot(at_m1.astype(np.int32))
            gains_m1[~available] = -1

            # Check if any voxel can promote M-1 → M
            at_mthresh = (coverage_counts == (M - 1))
            if np.any(at_mthresh):
                if use_muon_weight:
                    weights_M = muon_weight_delta(d_per_nc, muon_weight_k)
                    weights_M[~at_mthresh] = 0.0
                    gains_M = B.T.dot(weights_M)
                else:
                    gains_M = B.T.dot(at_mthresh.astype(np.int32))
                gains_M[~available] = -1

                best_M_voxel = int(np.argmax(gains_M))
                best_M_gain = gains_M[best_M_voxel]

                if best_M_gain > 0:
                    best_voxel = best_M_voxel
                    best_gain = int(best_M_gain)
                    phase_tag = " [M]"
                else:
                    best_voxel = int(np.argmax(gains_m1))
                    best_gain = int(gains_m1[best_voxel])
                    phase_tag = " [M=1]"
            else:
                best_voxel = int(np.argmax(gains_m1))
                best_gain = int(gains_m1[best_voxel])
                phase_tag = " [M=1]"

        elif use_dynamic_M:
            # --- Dynamic mode (original behavior) ---
            if np.any(coverage_counts == (M - 1)):
                effective_M = M
            else:
                max_c = 0
                for c_level in range(M - 2, -1, -1):
                    if np.any(coverage_counts == c_level):
                        max_c = c_level + 1
                        break
                effective_M = max(max_c, 1)

            at_threshold = (coverage_counts == (effective_M - 1))
            if use_muon_weight:
                d_per_nc = muon_det_counts[nc_to_muon_local]
                weights = muon_weight_delta(d_per_nc, muon_weight_k)
                weights[~at_threshold] = 0.0
                all_gains = B.T.dot(weights)
            else:
                all_gains = B.T.dot(at_threshold.astype(np.int32))
            all_gains[~available] = -1
            best_voxel = int(np.argmax(all_gains))
            best_gain = int(all_gains[best_voxel])
            phase_tag = f" [M=1]" if effective_M == 1 else ""

        else:
            # --- M=1 standard ---
            at_threshold = (coverage_counts == 0)
            if use_muon_weight:
                d_per_nc = muon_det_counts[nc_to_muon_local]
                weights = muon_weight_delta(d_per_nc, muon_weight_k)
                weights[~at_threshold] = 0.0
                all_gains = B.T.dot(weights)
            else:
                all_gains = B.T.dot(at_threshold.astype(np.int32))
            all_gains[~available] = -1
            best_voxel = int(np.argmax(all_gains))
            best_gain = int(all_gains[best_voxel])

        # Update coverage counts
        col_start = B.indptr[best_voxel]
        col_end = B.indptr[best_voxel + 1]
        affected_ncs = B.indices[col_start:col_end]
        coverage_counts[affected_ncs] += 1

        if use_muon_weight:
            newly_seen = affected_ncs[~nc_detected_flag[affected_ncs]]
            nc_detected_flag[affected_ncs] = True
            if len(newly_seen) > 0:
                muon_lids = nc_to_muon_local[newly_seen]
                increments = np.bincount(muon_lids, minlength=num_muons)
                muon_det_counts += increments.astype(np.int32)

        available[best_voxel] = False
        selected.append(best_voxel)

        if enforce_spacing:
            _apply_spacing(centers, layers, available, best_voxel,
                           min_spacing_sq, verbose)

        num_detected = int(np.sum(coverage_counts >= M))
        efficiency = num_detected / num_ncs
        efficiencies.append(efficiency)

        dt = time.time() - t0

        if verbose:
            gain_str = (f"{best_gain:>8.3f}" if use_muon_weight
                        else f"{best_gain:>8}")
            print(f"{step+1:>4} | {best_voxel:>8} | {gain_str} | "
                  f"{num_detected:>10} | {efficiency:>10.4%}  "
                  f"({dt:.2f}s){phase_tag}")

    return selected, efficiencies, coverage_counts, muon_det_counts


def greedy_select_muon(
    B: sparse.csc_matrix,
    N: int,
    W: int,
    nc_to_muon_local: np.ndarray,
    eligible_nc_mask: np.ndarray,
    num_ge77_muons: int,
    M: int = 1,
    centers: np.ndarray | None = None,
    layers: np.ndarray | None = None,
    min_spacing: float = 0.0,
    muon_weight_k: float | None = None,
    dynamic: bool = False,
    verbose: bool = True,
) -> tuple[list[int], list[float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Greedy voxel selection maximizing Ge77 muon detection.

    A NC is "detected" iff it is seen by >= M selected voxels
    (coverage_counts[nc] >= M).
    A Ge77 muon is detected iff >= W of its eligible NCs are detected.

    Priority mode (default, M>1 or W>1):
        Each step computes three candidate voxels:
        1. W-trigger: voxel that promotes a NC from M-1→M AND that NC
           is the W-th detected NC for its muon → highest priority.
        2. M-trigger: voxel that promotes the most NCs from M-1→M.
        3. M=1 baseline: voxel with highest gain at M=1 (new NCs).
        Selection: W-trigger > M-trigger > baseline.

    Dynamic mode (--dynamic, only M=1 supported):
        Falls back through W threshold levels.

    Parameters
    ----------
    B : sparse.csc_matrix
        Binary matrix (NCs x voxels) in CSC format.
    N : int
        Number of voxels to select.
    W : int
        Minimum detected NCs per muon for muon detection.
    nc_to_muon_local : np.ndarray, shape (num_ncs,)
        NC -> local Ge77-muon index (-1 if not eligible).
    eligible_nc_mask : np.ndarray, shape (num_ncs,), dtype bool
    num_ge77_muons : int
    M : int
        NC multiplicity threshold (default 1).
    centers, layers, min_spacing : spacing constraint params.
    muon_weight_k : float or None
    dynamic : bool
        If True, use dynamic W fallback.
    verbose : bool

    Returns
    -------
    selected : list[int]
    efficiencies : list[float]
    nc_detected : np.ndarray, dtype bool
        Per-NC: True if coverage_counts >= M.
    muon_detected_counts : np.ndarray, shape (num_ge77_muons,)
    coverage_counts : np.ndarray, shape (num_ncs,), dtype int16
    """
    num_ncs, num_voxels = B.shape

    if N > num_voxels:
        raise ValueError(f"N={N} exceeds number of voxels ({num_voxels})")

    enforce_spacing = (min_spacing > 0) and (centers is not None) and (layers is not None)
    min_spacing_sq = min_spacing ** 2

    TOP_K = min(50, num_voxels)

    coverage_counts = np.zeros(num_ncs, dtype=np.int16)
    nc_detected = np.zeros(num_ncs, dtype=bool)
    muon_detected_counts = np.zeros(num_ge77_muons, dtype=np.int32)
    available = np.ones(num_voxels, dtype=bool)
    selected: list[int] = []
    efficiencies: list[float] = []

    use_muon_weight = (muon_weight_k is not None)
    use_priority = (not dynamic) and ((M > 1) or (W > 1))
    use_dynamic_W = dynamic and (W > 1) and not use_muon_weight

    if verbose:
        spacing_str = f", min_spacing={min_spacing:.0f}mm" if enforce_spacing else ""
        weight_str = f", muon_weight_k={muon_weight_k:.2f}" if use_muon_weight else ""
        if use_dynamic_W:
            mode_str = " (dynamic W: fallback when no muons at W-1)"
        elif use_priority:
            mode_str = f" (priority: M=1 baseline, M-trigger, W-trigger; M={M}, W={W})"
        else:
            mode_str = ""
        print(f"\nGreedy selection (muon-ge77 mode): N={N}, W={W}, M={M}"
              f"{spacing_str}{weight_str}{mode_str}")
        print(f"{'Step':>4} | {'Voxel':>8} | {'Gain':>8} | "
              f"{'Muons det.':>10} | {'Efficiency':>10}")
        print("-" * 60)

    for step in range(N):
        t0 = time.time()
        phase_tag = ""

        if use_priority:
            # ============================================================
            # PRIORITY MODE
            # ============================================================

            # --- Baseline: M=1 gain (NCs not yet seen by any voxel) ---
            eligible_unseen = eligible_nc_mask & (coverage_counts == 0)
            eligible_unseen_idx = np.where(eligible_unseen)[0]

            if use_muon_weight:
                w_m1 = np.zeros(num_ncs, dtype=np.float32)
                if len(eligible_unseen_idx) > 0:
                    muon_lids = nc_to_muon_local[eligible_unseen_idx]
                    d_vals = muon_detected_counts[muon_lids]
                    w_m1[eligible_unseen_idx] = muon_weight_delta(
                        d_vals, muon_weight_k
                    )
            else:
                w_m1 = np.zeros(num_ncs, dtype=np.int32)
                w_m1[eligible_unseen_idx] = 1

            gains_m1 = B.T.dot(w_m1)
            gains_m1[~available] = -1
            best_m1_voxel = int(np.argmax(gains_m1))
            best_m1_gain = gains_m1[best_m1_voxel]

            # Default: take M=1 baseline
            best_voxel = best_m1_voxel
            best_gain = int(best_m1_gain)
            phase_tag = " [M=1]"

            # --- M-trigger: NCs at M-1 that can be promoted to M ---
            if M > 1:
                at_m_thresh = (coverage_counts == (M - 1)) & eligible_nc_mask
                if np.any(at_m_thresh):
                    gains_M = B.T.dot(at_m_thresh.astype(np.int32))
                    gains_M[~available] = -1
                    best_M_voxel = int(np.argmax(gains_M))
                    best_M_gain = int(gains_M[best_M_voxel])

                    if best_M_gain > 0:
                        best_voxel = best_M_voxel
                        best_gain = best_M_gain
                        phase_tag = " [M]"

            # --- W-trigger ---
            if W > 1 and M > 1:
                # W-trigger with M>1: voxel promotes NC from M-1→M
                # and that causes muon to reach W detected NCs
                at_m_thresh = (coverage_counts == (M - 1)) & eligible_nc_mask
                if np.any(at_m_thresh):
                    gains_W_trigger = B.T.dot(at_m_thresh.astype(np.int32))
                    gains_W_trigger[~available] = -1

                    top_w_indices = np.argsort(gains_W_trigger)[-TOP_K:][::-1]
                    top_w_indices = top_w_indices[
                        gains_W_trigger[top_w_indices] > 0
                    ]

                    best_w_gain = 0
                    best_w_voxel = -1

                    for v in top_w_indices:
                        cs = B.indptr[v]
                        ce = B.indptr[v + 1]
                        v_ncs = B.indices[cs:ce]

                        promoted = v_ncs[
                            (coverage_counts[v_ncs] == (M - 1))
                            & eligible_nc_mask[v_ncs]
                        ]
                        if len(promoted) == 0:
                            continue

                        muon_lids_p = nc_to_muon_local[promoted]
                        unique_muons, promo_counts = np.unique(
                            muon_lids_p, return_counts=True
                        )
                        current_d = muon_detected_counts[unique_muons]
                        crosses_W = (
                            (current_d < W)
                            & ((current_d + promo_counts) >= W)
                        )
                        w_gain = int(crosses_W.sum())

                        if w_gain > best_w_gain:
                            best_w_gain = w_gain
                            best_w_voxel = int(v)

                    if best_w_voxel >= 0 and best_w_gain > 0:
                        best_voxel = best_w_voxel
                        best_gain = best_w_gain
                        phase_tag = " [W]"

            elif W > 1 and M == 1:
                # W>1, M=1: NC detected as soon as 1 voxel sees it.
                eligible_undetected = eligible_nc_mask & (~nc_detected)
                eligible_und_idx = np.where(eligible_undetected)[0]

                if len(eligible_und_idx) > 0:
                    muon_at_w_thresh = (muon_detected_counts == (W - 1))

                    w_weight = np.zeros(num_ncs, dtype=np.int32)
                    muon_lids_und = nc_to_muon_local[eligible_und_idx]
                    at_w = muon_at_w_thresh[muon_lids_und]
                    w_weight[eligible_und_idx[at_w]] = 1

                    gains_W = B.T.dot(w_weight)
                    gains_W[~available] = -1

                    top_w_idx = np.argsort(gains_W)[-TOP_K:][::-1]
                    top_w_idx = top_w_idx[gains_W[top_w_idx] > 0]

                    best_w_gain = 0
                    best_w_voxel = -1

                    for v in top_w_idx:
                        cs = B.indptr[v]
                        ce = B.indptr[v + 1]
                        v_ncs = B.indices[cs:ce]

                        mask_elig = (eligible_nc_mask[v_ncs]
                                     & (~nc_detected[v_ncs]))
                        newly = v_ncs[mask_elig]
                        if len(newly) == 0:
                            continue

                        muon_lids_v = nc_to_muon_local[newly]
                        at_w_mask = muon_at_w_thresh[muon_lids_v]
                        w_gain = len(np.unique(muon_lids_v[at_w_mask]))

                        if w_gain > best_w_gain:
                            best_w_gain = w_gain
                            best_w_voxel = int(v)

                    if best_w_voxel >= 0 and best_w_gain > 0:
                        best_voxel = best_w_voxel
                        best_gain = best_w_gain
                        phase_tag = " [W]"

        elif use_dynamic_W:
            # ============================================================
            # DYNAMIC MODE (original behavior, M=1 enforced)
            # ============================================================
            if np.any(muon_detected_counts == (W - 1)):
                effective_W = W
            else:
                max_d = 0
                for d_level in range(W - 2, -1, -1):
                    if np.any(muon_detected_counts == d_level):
                        max_d = d_level + 1
                        break
                effective_W = max(max_d, 1)
            muon_at_threshold = (muon_detected_counts == (effective_W - 1))

            eligible_undetected = eligible_nc_mask & (~nc_detected)
            eligible_idx = np.where(eligible_undetected)[0]

            if use_muon_weight:
                w = np.zeros(num_ncs, dtype=np.float32)
                if len(eligible_idx) > 0:
                    muon_lids = nc_to_muon_local[eligible_idx]
                    d_vals = muon_detected_counts[muon_lids]
                    w[eligible_idx] = muon_weight_delta(d_vals, muon_weight_k)
            else:
                w = np.zeros(num_ncs, dtype=np.int32)
                if len(eligible_idx) > 0:
                    muon_lids = nc_to_muon_local[eligible_idx]
                    at_thresh = muon_at_threshold[muon_lids]
                    w[eligible_idx[at_thresh]] = 1

            all_gains_upper = B.T.dot(w)
            all_gains_upper[~available] = -1

            top_k_indices = np.argsort(all_gains_upper)[-TOP_K:][::-1]
            top_k_indices = top_k_indices[all_gains_upper[top_k_indices] > 0]

            best_gain = 0.0 if use_muon_weight else 0
            best_voxel = -1

            if len(top_k_indices) == 0:
                avail_idx = np.where(available)[0]
                if len(avail_idx) > 0:
                    best_voxel = int(avail_idx[0])
                else:
                    raise RuntimeError(
                        f"No available voxels left at step {step+1}/{N}. "
                        f"Spacing constraint may be too tight."
                    )
            else:
                for v in top_k_indices:
                    cs = B.indptr[v]
                    ce = B.indptr[v + 1]
                    v_ncs = B.indices[cs:ce]

                    mask_elig = eligible_nc_mask[v_ncs] & (~nc_detected[v_ncs])
                    newly = v_ncs[mask_elig]
                    if len(newly) == 0:
                        continue

                    muon_lids_v = nc_to_muon_local[newly]

                    if use_muon_weight:
                        unique_muons, counts = np.unique(
                            muon_lids_v, return_counts=True
                        )
                        d_vals = muon_detected_counts[unique_muons]
                        gain = 0.0
                        for um_idx in range(len(unique_muons)):
                            d_cur = d_vals[um_idx]
                            n_new = counts[um_idx]
                            for j in range(n_new):
                                gain += float(
                                    muon_weight_delta(
                                        np.array([d_cur + j]), muon_weight_k
                                    )[0]
                                )
                    else:
                        at_thresh_mask = muon_at_threshold[muon_lids_v]
                        gain = len(np.unique(muon_lids_v[at_thresh_mask]))

                    if gain > best_gain:
                        best_gain = gain
                        best_voxel = int(v)

                if best_voxel == -1:
                    best_voxel = int(top_k_indices[0])
                    best_gain = 0

            phase_tag = f" [W=1]" if effective_W == 1 else ""

        else:
            # ============================================================
            # STANDARD MODE (M=1, W=1)
            # ============================================================
            eligible_undetected = eligible_nc_mask & (~nc_detected)
            eligible_idx = np.where(eligible_undetected)[0]

            if use_muon_weight:
                w = np.zeros(num_ncs, dtype=np.float32)
                if len(eligible_idx) > 0:
                    muon_lids = nc_to_muon_local[eligible_idx]
                    d_vals = muon_detected_counts[muon_lids]
                    w[eligible_idx] = muon_weight_delta(d_vals, muon_weight_k)
            else:
                w = np.zeros(num_ncs, dtype=np.int32)
                w[eligible_idx] = 1

            all_gains = B.T.dot(w)
            all_gains[~available] = -1

            top_k_indices = np.argsort(all_gains)[-TOP_K:][::-1]
            top_k_indices = top_k_indices[all_gains[top_k_indices] > 0]

            best_gain = 0.0 if use_muon_weight else 0
            best_voxel = -1

            if len(top_k_indices) == 0:
                avail_idx = np.where(available)[0]
                if len(avail_idx) > 0:
                    best_voxel = int(avail_idx[0])
                else:
                    raise RuntimeError(
                        f"No available voxels left at step {step+1}/{N}.")
            else:
                for v in top_k_indices:
                    cs = B.indptr[v]
                    ce = B.indptr[v + 1]
                    v_ncs = B.indices[cs:ce]
                    mask_elig = eligible_nc_mask[v_ncs] & (~nc_detected[v_ncs])
                    newly = v_ncs[mask_elig]
                    if len(newly) == 0:
                        continue
                    muon_lids_v = nc_to_muon_local[newly]

                    if use_muon_weight:
                        unique_muons, counts = np.unique(
                            muon_lids_v, return_counts=True
                        )
                        d_vals = muon_detected_counts[unique_muons]
                        gain = 0.0
                        for um_idx in range(len(unique_muons)):
                            d_cur = d_vals[um_idx]
                            n_new = counts[um_idx]
                            for j in range(n_new):
                                gain += float(
                                    muon_weight_delta(
                                        np.array([d_cur + j]), muon_weight_k
                                    )[0]
                                )
                    else:
                        gain = len(np.unique(muon_lids_v))

                    if gain > best_gain:
                        best_gain = gain
                        best_voxel = int(v)

                if best_voxel == -1:
                    best_voxel = int(top_k_indices[0])
                    best_gain = 0

        # ================================================================
        # UPDATE STATE
        # ================================================================
        col_start = B.indptr[best_voxel]
        col_end = B.indptr[best_voxel + 1]
        affected_ncs = B.indices[col_start:col_end]

        coverage_counts[affected_ncs] += 1

        # Update nc_detected: NCs that just reached M
        newly_reached_M = affected_ncs[
            (coverage_counts[affected_ncs] == M)
            & eligible_nc_mask[affected_ncs]
            & (~nc_detected[affected_ncs])
        ]
        nc_detected[newly_reached_M] = True

        # Update muon counts for NCs that just became detected
        if len(newly_reached_M) > 0:
            muon_lids_new = nc_to_muon_local[newly_reached_M]
            valid_muon = muon_lids_new >= 0
            if np.any(valid_muon):
                increments = np.bincount(
                    muon_lids_new[valid_muon], minlength=num_ge77_muons
                )
                muon_detected_counts += increments.astype(np.int32)

        available[best_voxel] = False
        selected.append(best_voxel)

        if enforce_spacing:
            _apply_spacing(centers, layers, available, best_voxel,
                           min_spacing_sq, verbose)

        num_muons_detected = int(np.sum(muon_detected_counts >= W))
        efficiency = (num_muons_detected / num_ge77_muons
                      if num_ge77_muons > 0 else 0.0)
        efficiencies.append(efficiency)

        dt = time.time() - t0

        if verbose:
            gain_str = (f"{best_gain:>8.3f}" if use_muon_weight
                        else f"{best_gain:>8}")
            print(f"{step+1:>4} | {best_voxel:>8} | {gain_str} | "
                  f"{num_muons_detected:>10} | {efficiency:>10.4%}  "
                  f"({dt:.2f}s){phase_tag}")

    return selected, efficiencies, nc_detected, muon_detected_counts, coverage_counts