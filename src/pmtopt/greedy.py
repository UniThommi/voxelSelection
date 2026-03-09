"""
Greedy voxel selection algorithms.

Implements Maximum k-Coverage and Partial Set Multi-Cover for
NC-level and Ge77-muon-level optimization targets.
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
    verbose: bool = True,
) -> tuple[list[int], list[float], np.ndarray, np.ndarray | None]:
    """
    Greedy voxel selection maximizing NC-level threshold coverage.

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

    use_dynamic_M = (M > 1)

    if verbose:
        spacing_str = f", min_spacing={min_spacing:.0f}mm" if enforce_spacing else ""
        weight_str = f", muon_weight_k={muon_weight_k:.2f}" if use_muon_weight else ""
        dynamic_str = (" (dynamic M: fallback to M=1 when no NCs at M-1)"
                       if use_dynamic_M else "")
        print(f"\nGreedy selection (NC mode): N={N}, M={M}"
              f"{spacing_str}{weight_str}{dynamic_str}")
        print(f"{'Step':>4} | {'Voxel':>8} | {'Gain':>8} | "
              f"{'Detected':>10} | {'Efficiency':>10}")
        print("-" * 60)

    for step in range(N):
        t0 = time.time()

        if use_dynamic_M:
            if np.any(coverage_counts == (M - 1)):
                effective_M = M
            else:
                max_c = 0
                for c_level in range(M - 2, -1, -1):
                    if np.any(coverage_counts == c_level):
                        max_c = c_level + 1
                        break
                effective_M = max(max_c, 1)
        else:
            effective_M = M
        at_threshold = (coverage_counts == (effective_M - 1))

        if use_muon_weight:
            muon_ids_per_nc = nc_to_muon_local
            d_per_nc = muon_det_counts[muon_ids_per_nc]
            weights = muon_weight_delta(d_per_nc, muon_weight_k)
            weights[~at_threshold] = 0.0
            all_gains = B.T.dot(weights)
        else:
            all_gains = B.T.dot(at_threshold.astype(np.int32))
        all_gains[~available] = -1
        best_voxel = int(np.argmax(all_gains))
        best_gain = int(all_gains[best_voxel])

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

        num_detected = int(np.sum(coverage_counts >= M))
        efficiency = num_detected / num_ncs
        efficiencies.append(efficiency)

        dt = time.time() - t0

        if verbose:
            phase_tag = f" [M=1]" if (use_dynamic_M and effective_M == 1) else ""
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
    centers: np.ndarray | None = None,
    layers: np.ndarray | None = None,
    min_spacing: float = 0.0,
    muon_weight_k: float | None = None,
    verbose: bool = True,
) -> tuple[list[int], list[float], np.ndarray, np.ndarray]:
    """
    Greedy voxel selection maximizing Ge77 muon detection.

    M=1 is enforced: a NC is detected iff any selected voxel sees it.
    A Ge77 muon is detected iff >= W of its eligible NCs are detected.

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
        True for NCs participating in muon optimization.
    num_ge77_muons : int
        Number of Ge77 muons.
    centers, layers, min_spacing : spacing constraint params.
    muon_weight_k : float or None
        If set, use diminishing-returns weighting.
    verbose : bool

    Returns
    -------
    selected : list[int]
    efficiencies : list[float]
    nc_detected : np.ndarray, dtype bool
    muon_detected_counts : np.ndarray, shape (num_ge77_muons,)
    """
    num_ncs, num_voxels = B.shape

    if N > num_voxels:
        raise ValueError(f"N={N} exceeds number of voxels ({num_voxels})")

    enforce_spacing = (min_spacing > 0) and (centers is not None) and (layers is not None)
    min_spacing_sq = min_spacing ** 2

    TOP_K = min(50, num_voxels)

    nc_detected = np.zeros(num_ncs, dtype=bool)
    muon_detected_counts = np.zeros(num_ge77_muons, dtype=np.int32)
    available = np.ones(num_voxels, dtype=bool)
    selected: list[int] = []
    efficiencies: list[float] = []

    use_muon_weight = (muon_weight_k is not None)
    use_dynamic_W = (W > 1) and not use_muon_weight

    if verbose:
        spacing_str = f", min_spacing={min_spacing:.0f}mm" if enforce_spacing else ""
        weight_str = f", muon_weight_k={muon_weight_k:.2f}" if use_muon_weight else ""
        dynamic_str = (" (dynamic W: fallback to W=1 when no muons at W-1)"
                       if use_dynamic_W else "")
        print(f"\nGreedy selection (muon-ge77 mode): N={N}, W={W}"
              f"{spacing_str}{weight_str}{dynamic_str}")
        print(f"{'Step':>4} | {'Voxel':>8} | {'Gain':>8} | "
              f"{'Muons det.':>10} | {'Efficiency':>10}")
        print("-" * 60)

    for step in range(N):
        t0 = time.time()

        if use_dynamic_W:
            if np.any(muon_detected_counts == (W - 1)):
                effective_W = W
            else:
                max_d = 0
                for d_level in range(W - 2, -1, -1):
                    if np.any(muon_detected_counts == d_level):
                        max_d = d_level + 1
                        break
                effective_W = max(max_d, 1)
        else:
            effective_W = W
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
            best_gain = 0
        else:
            for v in top_k_indices:
                col_start = B.indptr[v]
                col_end = B.indptr[v + 1]
                v_ncs = B.indices[col_start:col_end]

                mask_elig = eligible_nc_mask[v_ncs] & (~nc_detected[v_ncs])
                newly_detected_ncs = v_ncs[mask_elig]

                if len(newly_detected_ncs) == 0:
                    continue

                muon_lids_v = nc_to_muon_local[newly_detected_ncs]

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
                    muon_lids_at_thresh = muon_lids_v[at_thresh_mask]
                    gain = len(np.unique(muon_lids_at_thresh))

                if gain > best_gain:
                    best_gain = gain
                    best_voxel = int(v)

            if best_voxel == -1:
                best_voxel = int(top_k_indices[0])
                best_gain = 0

        # Update state
        col_start = B.indptr[best_voxel]
        col_end = B.indptr[best_voxel + 1]
        affected_ncs = B.indices[col_start:col_end]

        newly_detected_mask = ~nc_detected[affected_ncs]
        newly_detected_ncs = affected_ncs[newly_detected_mask]
        nc_detected[affected_ncs] = True

        eligible_new = newly_detected_ncs[eligible_nc_mask[newly_detected_ncs]]
        if len(eligible_new) > 0:
            muon_lids_new = nc_to_muon_local[eligible_new]
            increments = np.bincount(muon_lids_new, minlength=num_ge77_muons)
            muon_detected_counts += increments.astype(np.int32)

        available[best_voxel] = False
        selected.append(best_voxel)

        if enforce_spacing:
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

        num_muons_detected = int(np.sum(muon_detected_counts >= W))
        efficiency = (num_muons_detected / num_ge77_muons
                      if num_ge77_muons > 0 else 0.0)
        efficiencies.append(efficiency)

        dt = time.time() - t0

        if verbose:
            phase_tag = (f" [W=1]"
                         if (use_dynamic_W and effective_W == 1) else "")
            gain_str = (f"{best_gain:>8.3f}" if use_muon_weight
                        else f"{best_gain:>8}")
            print(f"{step+1:>4} | {best_voxel:>8} | {gain_str} | "
                  f"{num_muons_detected:>10} | {efficiency:>10.4%}  "
                  f"({dt:.2f}s){phase_tag}")

    return selected, efficiencies, nc_detected, muon_detected_counts