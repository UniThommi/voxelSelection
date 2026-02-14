#!/usr/bin/env python3
"""
Voxel Importance Map via Randomized Greedy Ensemble
====================================================

Approximates the importance of each voxel for NC detection by running
K randomized greedy selections and aggregating the results.

Theoretical Motivation
----------------------
The Shapley value φ(v) of a voxel v measures its expected marginal
contribution, averaged over all possible orderings of voxel selection:

    φ(v) = (1/|Ω|!) Σ_π [f(S_π(v) ∪ {v}) − f(S_π(v))]

Exact computation requires |Ω|! permutations — intractable for ~10⁴ voxels.
Monte Carlo Shapley [Castro et al., "Polynomial calculation of the Shapley
value based on sampling", Computers & OR 36(5), 2009] approximates this
by sampling random permutations, but each permutation requires a full
sequential pass over all voxels, making it expensive at scale.

Our approach: Instead of uniform random permutations, we sample permutations
induced by a **randomized greedy** — at each step, instead of picking the
voxel with the highest gain deterministically, we sample from the top
candidates weighted by their gain via a softmax distribution:

    P(v) = exp(gain(v) / τ) / Σ_u exp(gain(u) / τ)

where τ (temperature) controls exploration:
    - τ → 0: deterministic greedy (always picks argmax)
    - τ → ∞: uniform random selection
    - τ ≈ median(gains): balanced exploration around good solutions

This is a biased estimator of the Shapley value, but the bias is
meaningful: it weights permutations where good voxels appear early,
reflecting the actual decision process. The approach is related to
the Stochastic Greedy [Mirzasoleiman et al., "Lazier Than Lazy Greedy",
AAAI 2015] but uses full evaluation with stochastic selection rather
than subsampled evaluation with deterministic selection.

Importance Metrics (per voxel)
------------------------------
1. selection_frequency: Fraction of runs in which voxel was selected
   (among the top N). High frequency → robust importance.
2. mean_rank: Average position at which voxel was selected (1 = first
   picked, N = last picked). Low rank → consistently important.
3. mean_marginal_gain: Average gain at time of selection. Measures
   how much detection efficiency the voxel typically contributes.
4. importance_score: Combined score = frequency × (1 / mean_rank).
   This rewards voxels that are both frequently selected AND selected
   early. Analogous to a weighted Shapley approximation.

Usage
-----
    python voxel_importance_map.py <hdf5_file> --N 50 --M 2 --K 500 \\
        --temperature 1.0 --output importance.npz

Author: Ferundo (Thesis project, University of Tübingen)
"""

import argparse
import time
from typing import Optional

import h5py
import numpy as np
from scipy import sparse

# ---------------------------------------------------------------------------
# Detector geometry constants (mm)
# ---------------------------------------------------------------------------
PMT_RADIUS = 131
R_PIT = 3800
R_ZYL_BOT = 3950
R_ZYL_TOP = 1200
R_ZYLINDER = 4300
Z_ORIGIN = 20
Z_OFFSET = -5000
H_ZYLINDER = 8900 - 1
Z_BASE_GLOBAL = Z_ORIGIN + Z_OFFSET


def is_valid_pmt_position(
    center: np.ndarray,
    layer: str,
    pmt_r: float = PMT_RADIUS,
) -> bool:
    """Check whether a PMT fits at the voxel center within layer boundaries."""
    x, y, z = center
    r_center = np.sqrt(x**2 + y**2)

    if layer == "pit":
        return r_center + pmt_r <= R_PIT
    elif layer == "bot":
        return (r_center - pmt_r >= R_ZYL_BOT) and (r_center + pmt_r <= R_ZYLINDER)
    elif layer == "top":
        return (r_center - pmt_r >= R_ZYL_TOP) and (r_center + pmt_r <= R_ZYLINDER)
    elif layer == "wall":
        z_min_allowed = Z_BASE_GLOBAL + pmt_r
        z_max_allowed = Z_BASE_GLOBAL + H_ZYLINDER - pmt_r
        return z_min_allowed <= z <= z_max_allowed
    return False


def get_valid_voxel_mask(
    f: h5py.File,
    voxel_keys: list[str],
    verbose: bool = True,
) -> np.ndarray:
    """Determine which voxels can physically host a PMT."""
    num_voxels = len(voxel_keys)
    valid_mask = np.zeros(num_voxels, dtype=bool)
    layer_counts = {"pit": [0, 0], "bot": [0, 0], "top": [0, 0], "wall": [0, 0]}

    for col_idx, vkey in enumerate(voxel_keys):
        center = f[f"voxels/{vkey}/center"][:]
        layer_raw = f[f"voxels/{vkey}/layer"][()]
        layer = layer_raw.decode() if isinstance(layer_raw, bytes) else str(layer_raw)

        if layer in layer_counts:
            layer_counts[layer][0] += 1
        if is_valid_pmt_position(center, layer):
            valid_mask[col_idx] = True
            if layer in layer_counts:
                layer_counts[layer][1] += 1

    if verbose:
        total = num_voxels
        valid = int(valid_mask.sum())
        print(f"PMT placement filter (r_pmt = {PMT_RADIUS} mm):")
        for ly in ["pit", "bot", "top", "wall"]:
            tot, val = layer_counts[ly]
            print(f"  {ly:>4}: {val}/{tot} valid")
        print(f"  Total: {valid}/{total} valid "
              f"({total - valid} filtered out)")
    return valid_mask


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_binarize(
    filepath: str,
    m: int = 1,
    verbose: bool = True,
) -> tuple[sparse.csc_matrix, np.ndarray, int]:
    """
    Load HDF5 data and construct sparse binary matrix B.

    B[i, j] = 1 iff voxel j registered >= m hits for NC i.
    Only voxels where a PMT can physically be placed are included.

    Returns
    -------
    B : sparse.csc_matrix  (NCs x valid_voxels)
    voxel_ids : np.ndarray of voxel ID strings
    num_primaries : int
    """
    with h5py.File(filepath, "r") as f:
        voxel_keys = sorted(f["target"].keys())

        # --- PMT placement filter ---
        valid_mask = get_valid_voxel_mask(f, voxel_keys, verbose=verbose)
        valid_keys = [k for k, v in zip(voxel_keys, valid_mask) if v]
        num_voxels = len(valid_keys)

        num_ncs = f["target"][valid_keys[0]].shape[0]
        num_primaries = int(f["primaries"][()])

        if verbose:
            print(f"Loading data: {num_ncs} NCs, {num_voxels} valid voxels, "
                  f"{num_primaries} primaries")

        rows_list: list[np.ndarray] = []
        cols_list: list[np.ndarray] = []

        for col_idx, voxel_key in enumerate(valid_keys):
            if verbose and col_idx % 1000 == 0:
                print(f"  Loading voxel {col_idx}/{num_voxels}...", end="\r")
            hits = f["target"][voxel_key][:]
            mask = hits >= m
            nc_indices = np.where(mask)[0]
            if len(nc_indices) > 0:
                rows_list.append(nc_indices)
                cols_list.append(
                    np.full(len(nc_indices), col_idx, dtype=np.int32)
                )

        if verbose:
            print()

        all_rows = np.concatenate(rows_list)
        all_cols = np.concatenate(cols_list)
        all_data = np.ones(len(all_rows), dtype=np.int8)

        B = sparse.coo_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(num_ncs, num_voxels),
            dtype=np.int8,
        ).tocsc()

        if verbose:
            nnz = B.nnz
            density = nnz / (num_ncs * num_voxels) * 100
            mem_mb = (B.data.nbytes + B.indices.nbytes + B.indptr.nbytes) / 1e6
            print(f"Sparse matrix: {num_ncs} x {num_voxels}, "
                  f"nnz = {nnz:,} ({density:.3f}%), memory = {mem_mb:.1f} MB")

        voxel_ids = np.array(valid_keys)

    return B, voxel_ids, num_primaries


# ---------------------------------------------------------------------------
# Randomized Greedy — single run
# ---------------------------------------------------------------------------

def randomized_greedy_single(
    B: sparse.csc_matrix,
    N: int,
    M: int,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single run of the randomized greedy algorithm.

    Instead of deterministically picking argmax(gain), we sample from
    a softmax distribution over gains:

        P(v) ∝ exp(gain(v) / τ)

    This introduces stochasticity: different runs produce different
    selections, enabling ensemble-based importance estimation.

    Parameters
    ----------
    B : sparse.csc_matrix
        Binary (NCs x voxels) matrix.
    N : int
        Number of voxels to select.
    M : int
        Multiplicity threshold.
    temperature : float
        Softmax temperature τ. Controls exploration vs exploitation.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    selected : np.ndarray of int, shape (N,)
        Column indices of selected voxels in selection order.
    gains : np.ndarray of float, shape (N,)
        Marginal gain at each selection step.
    efficiency : np.ndarray of float, shape (N,)
        Cumulative detection efficiency after each step.
    """
    num_ncs, num_voxels = B.shape

    coverage_counts = np.zeros(num_ncs, dtype=np.int16)
    available = np.ones(num_voxels, dtype=bool)

    selected = np.empty(N, dtype=np.int32)
    gains = np.empty(N, dtype=np.float32)
    efficiency = np.empty(N, dtype=np.float32)

    for step in range(N):
        # Compute marginal gains for all available voxels.
        # gain(v) = |{i : c[i] == M-1 AND B[i,v] == 1}|
        at_threshold = (coverage_counts == (M - 1)).astype(np.int32)
        all_gains = B.T.dot(at_threshold).astype(np.float64)

        # Mask out already-selected voxels
        all_gains[~available] = -np.inf

        # ---------------------------------------------------------------
        # Softmax sampling over gains.
        #
        # Numerical stability: subtract max before exp to prevent overflow.
        # Voxels with gain = -inf (already selected) get P = 0.
        #
        # Temperature τ controls the distribution shape:
        #   - Low τ: sharply peaked around the best voxel (near-deterministic)
        #   - High τ: flatter, more exploration of suboptimal voxels
        #
        # For gains that are 0 (voxel contributes nothing right now),
        # we still assign nonzero probability if temperature is high enough,
        # which allows discovering voxels that become useful later (important
        # for M > 1 where the threshold effect creates delayed contributions).
        # ---------------------------------------------------------------
        valid_mask = available.copy()
        valid_gains = all_gains[valid_mask]

        if temperature > 0:
            # Softmax with temperature
            logits = valid_gains / temperature
            logits -= logits.max()  # numerical stability
            probs = np.exp(logits)
            prob_sum = probs.sum()

            if prob_sum > 0:
                probs /= prob_sum
            else:
                # All gains are -inf or 0; fall back to uniform
                probs = np.ones(len(probs)) / len(probs)

            # Sample one voxel from the distribution
            valid_indices = np.where(valid_mask)[0]
            chosen_local = rng.choice(len(valid_indices), p=probs)
            chosen_voxel = int(valid_indices[chosen_local])
            chosen_gain = float(all_gains[chosen_voxel])
        else:
            # τ = 0: deterministic greedy (argmax)
            chosen_voxel = int(np.argmax(all_gains))
            chosen_gain = float(all_gains[chosen_voxel])

        # Update state
        col_start = B.indptr[chosen_voxel]
        col_end = B.indptr[chosen_voxel + 1]
        affected_ncs = B.indices[col_start:col_end]
        coverage_counts[affected_ncs] += 1

        available[chosen_voxel] = False
        selected[step] = chosen_voxel
        gains[step] = chosen_gain

        num_detected = int(np.sum(coverage_counts >= M))
        efficiency[step] = num_detected / num_ncs

    return selected, gains, efficiency


# ---------------------------------------------------------------------------
# Ensemble: K runs → importance map
# ---------------------------------------------------------------------------

def compute_importance_map(
    B: sparse.csc_matrix,
    N: int,
    M: int,
    K: int,
    temperature: float,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Run K randomized greedy selections and aggregate into importance scores.

    The ensemble of randomized greedy runs approximates a distribution over
    "good" voxel subsets. By tracking how often and how early each voxel
    appears across runs, we obtain a continuous importance score that
    captures both individual contribution and robustness.

    Connection to Shapley values:
    The selection frequency weighted by inverse rank approximates a
    restricted Shapley value — restricted to permutations that the
    randomized greedy can generate. This is a biased but computationally
    tractable estimator [cf. Castro et al., 2009 for sampling-based
    Shapley; Mirzasoleiman et al., 2015 for stochastic greedy].

    Parameters
    ----------
    B : sparse.csc_matrix
        Binary (NCs x voxels) matrix.
    N : int
        Number of voxels to select per run.
    M : int
        Multiplicity threshold.
    K : int
        Number of randomized greedy runs.
    temperature : float
        Softmax temperature.
    seed : int
        Base random seed for reproducibility. Run k uses seed + k.
    verbose : bool
        Print progress.

    Returns
    -------
    results : dict with keys:
        - selection_frequency: (num_voxels,) fraction of K runs selected
        - mean_rank: (num_voxels,) mean selection position (1-indexed);
          NaN for voxels never selected
        - std_rank: (num_voxels,) std of selection position
        - mean_gain: (num_voxels,) mean marginal gain at selection
        - importance_score: (num_voxels,) = frequency / mean_rank
        - all_efficiencies: (K, N) efficiency curves for all runs
        - deterministic_selected: (N,) deterministic greedy result for reference
        - deterministic_efficiency: (N,) efficiency curve of deterministic run
    """
    num_ncs, num_voxels = B.shape

    # Accumulators
    selection_count = np.zeros(num_voxels, dtype=np.int32)
    rank_sum = np.zeros(num_voxels, dtype=np.float64)
    rank_sq_sum = np.zeros(num_voxels, dtype=np.float64)
    gain_sum = np.zeros(num_voxels, dtype=np.float64)
    all_efficiencies = np.empty((K, N), dtype=np.float32)

    # --- Deterministic baseline (τ=0) for comparison ---
    if verbose:
        print(f"\nRunning deterministic greedy baseline...")
    rng_det = np.random.default_rng(seed)
    det_selected, det_gains, det_efficiency = randomized_greedy_single(
        B, N, M, temperature=0.0, rng=rng_det,
    )
    if verbose:
        print(f"  Baseline efficiency: {det_efficiency[-1]:.4%}")

    # --- K randomized runs ---
    if verbose:
        print(f"\nRunning {K} randomized greedy runs "
              f"(τ={temperature}, N={N}, M={M})...")
    t_start = time.time()

    for k in range(K):
        rng = np.random.default_rng(seed + k + 1)  # +1 to avoid overlap with baseline

        selected, gains, eff = randomized_greedy_single(
            B, N, M, temperature, rng,
        )
        all_efficiencies[k] = eff

        # Accumulate statistics.
        # rank is 1-indexed: first selected = rank 1, last = rank N.
        for rank_idx in range(N):
            v = selected[rank_idx]
            rank = rank_idx + 1  # 1-indexed
            selection_count[v] += 1
            rank_sum[v] += rank
            rank_sq_sum[v] += rank * rank
            gain_sum[v] += gains[rank_idx]

        if verbose and (k + 1) % max(1, K // 10) == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (k + 1) * (K - k - 1)
            mean_eff = float(np.mean(all_efficiencies[:k+1, -1]))
            print(f"  Run {k+1:>5}/{K}: "
                  f"mean final eff = {mean_eff:.4%}, "
                  f"elapsed = {elapsed:.1f}s, ETA = {eta:.1f}s")

    # --- Compute importance metrics ---
    selection_frequency = selection_count / K

    # Mean and std of rank (only for voxels selected at least once)
    mean_rank = np.full(num_voxels, np.nan)
    std_rank = np.full(num_voxels, np.nan)
    mean_gain = np.zeros(num_voxels, dtype=np.float64)

    mask_selected = selection_count > 0
    mean_rank[mask_selected] = rank_sum[mask_selected] / selection_count[mask_selected]
    mean_gain[mask_selected] = gain_sum[mask_selected] / selection_count[mask_selected]

    # Var = E[X²] - E[X]²
    variance = (
        rank_sq_sum[mask_selected] / selection_count[mask_selected]
        - mean_rank[mask_selected] ** 2
    )
    std_rank[mask_selected] = np.sqrt(np.maximum(variance, 0.0))

    # Importance score: frequency × (1 / mean_rank)
    # Rewards voxels that are both frequently selected AND selected early.
    # Voxels never selected get score 0.
    importance_score = np.zeros(num_voxels, dtype=np.float64)
    importance_score[mask_selected] = (
        selection_frequency[mask_selected] / mean_rank[mask_selected]
    )

    # Normalize to [0, 1] for convenience
    max_score = importance_score.max()
    if max_score > 0:
        importance_score_normalized = importance_score / max_score
    else:
        importance_score_normalized = importance_score

    if verbose:
        t_total = time.time() - t_start
        print(f"\nCompleted {K} runs in {t_total:.1f}s")
        eff_mean = np.mean(all_efficiencies[:, -1])
        eff_std = np.std(all_efficiencies[:, -1])
        print(f"Randomized ensemble efficiency: "
              f"{eff_mean:.4%} ± {eff_std:.4%}")
        print(f"Deterministic baseline:         "
              f"{det_efficiency[-1]:.4%}")

        # Top 20 by importance
        top_idx = np.argsort(importance_score)[::-1][:20]
        print(f"\nTop 20 voxels by importance score:")
        print(f"{'Rank':>4} | {'VoxelCol':>8} | {'Freq':>6} | "
              f"{'MeanRank':>8} | {'StdRank':>7} | {'MeanGain':>8} | "
              f"{'Score':>8} | {'Normalized':>10}")
        print("-" * 85)
        for i, vi in enumerate(top_idx):
            print(f"{i+1:>4} | {vi:>8} | {selection_frequency[vi]:>6.3f} | "
                  f"{mean_rank[vi]:>8.1f} | {std_rank[vi]:>7.1f} | "
                  f"{mean_gain[vi]:>8.1f} | "
                  f"{importance_score[vi]:>8.5f} | "
                  f"{importance_score_normalized[vi]:>10.5f}")

    return {
        "selection_frequency": selection_frequency,
        "mean_rank": mean_rank,
        "std_rank": std_rank,
        "mean_gain": mean_gain,
        "importance_score": importance_score,
        "importance_score_normalized": importance_score_normalized,
        "all_efficiencies": all_efficiencies,
        "deterministic_selected": det_selected,
        "deterministic_efficiency": det_efficiency,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Voxel importance map via randomized greedy ensemble.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("hdf5_file", type=str,
                        help="Path to the HDF5 data file.")
    parser.add_argument("--N", type=int, required=True,
                        help="Number of voxels to select per run.")
    parser.add_argument("--M", type=int, required=True,
                        help="Multiplicity threshold for detection.")
    parser.add_argument("--m", type=int, default=1,
                        help="Hit threshold per voxel.")
    parser.add_argument("--K", type=int, default=500,
                        help="Number of randomized greedy runs.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature τ. Higher = more exploration. "
                             "Rule of thumb: set to ~median(gains) from a "
                             "deterministic run, or start with 1.0.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed for reproducibility.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (npz format).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output.")

    args = parser.parse_args(argv)
    verbose = not args.quiet

    if verbose:
        print("=" * 70)
        print("Voxel Importance Map — Randomized Greedy Ensemble")
        print("=" * 70)

    B, voxel_ids, num_primaries = load_and_binarize(
        args.hdf5_file, m=args.m, verbose=verbose,
    )

    results = compute_importance_map(
        B,
        N=args.N,
        M=args.M,
        K=args.K,
        temperature=args.temperature,
        seed=args.seed,
        verbose=verbose,
    )

    if args.output:
        np.savez(
            args.output,
            voxel_ids=voxel_ids,
            **results,
            N=args.N,
            M=args.M,
            m=args.m,
            K=args.K,
            temperature=args.temperature,
            seed=args.seed,
            num_ncs=B.shape[0],
            num_primaries=num_primaries,
        )
        if verbose:
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()