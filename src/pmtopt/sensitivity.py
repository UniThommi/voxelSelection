"""
Sensitivity analysis for area-ratio perturbation.

Runs the greedy optimization for multiple perturbed ratios and compares
voxel selections via Jaccard similarity and coverage metrics.
"""

import json
import time
from pathlib import Path

import numpy as np

from .geometry import DEFAULT_AREA_RATIOS, PMT_RADIUS, compute_per_area_N
from .data_loading import (
    get_valid_voxel_mask, load_and_binarize,
    load_muon_data, build_muon_index, load_nc_muon_ids,
)
from .greedy import greedy_select_nc, greedy_select_muon
from .plotting import (
    plot_jaccard_curves, plot_coverage_change, plot_per_area_jaccard,
)

import h5py


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    union = len(set_a | set_b)
    if union == 0:
        return 1.0
    return len(set_a & set_b) / union


def compute_jaccard_curve(
    baseline_order: list[int],
    perturbed_order: list[int],
    k_values: list[int],
) -> list[float]:
    """Compute Jaccard similarity for top-k voxels at each k."""
    jaccards = []
    for k in k_values:
        set_base = set(baseline_order[:k])
        set_pert = set(perturbed_order[:k])
        jaccards.append(jaccard(set_base, set_pert))
    return jaccards


def compute_per_area_jaccard(
    baseline_order: list[int],
    perturbed_order: list[int],
    layers: np.ndarray,
    N: int,
) -> dict[str, float]:
    """Compute Jaccard similarity per detector area for top-N voxels."""
    base_set = set(baseline_order[:N])
    pert_set = set(perturbed_order[:N])
    per_area = {}

    for area in ["pit", "bot", "top", "wall"]:
        base_area = {v for v in base_set if layers[v] == area}
        pert_area = {v for v in pert_set if layers[v] == area}
        per_area[area] = jaccard(base_area, pert_area)

    return per_area


# ---------------------------------------------------------------------------
# Single greedy run with custom ratios
# ---------------------------------------------------------------------------

def run_single_greedy(
    filepath: str,
    N: int,
    m: int,
    area_ratios: dict[str, float],
    optimize: str,
    M: int = 1,
    W: int = 1,
    min_spacing: float = 0.0,
    per_area: bool = False,
    muon_weight_k: float | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[list[int], float]:
    """
    Run a single greedy optimization with given area ratios.

    Returns selected voxel column indices (in order) and final efficiency.
    B is constructed and released within this function to avoid OOM.
    """
    B, voxel_ids, centers, layers, num_primaries = load_and_binarize(
        filepath, m=m, area_ratios=area_ratios,
        seed=seed, verbose=verbose,
    )

    # Load muon data if needed
    nc_muon_weight_data = None
    if optimize == "nc" and muon_weight_k is not None:
        nc_to_muon_local_nc, num_muons_nc, _ = load_nc_muon_ids(
            filepath, num_ncs=B.shape[0], verbose=False,
        )
        nc_muon_weight_data = {
            "nc_to_muon_local": nc_to_muon_local_nc,
            "num_muons": num_muons_nc,
        }

    nc_to_muon_local = None
    eligible_nc_mask = None
    num_ge77_muons = 0
    if optimize == "muon-ge77":
        global_muon_id, nc_time_ns, nc_flag_ge77 = load_muon_data(
            filepath, num_ncs=B.shape[0], verbose=False,
        )
        (nc_to_muon_local, _, _,
         eligible_nc_mask, num_ge77_muons) = build_muon_index(
            global_muon_id, nc_time_ns, nc_flag_ge77, verbose=False,
        )

    if per_area:
        allocation = compute_per_area_N(N, verbose=False)
        all_selected: list[int] = []

        if optimize == "muon-ge77":
            shared_nc_detected = np.zeros(B.shape[0], dtype=bool)
            shared_muon_counts = np.zeros(num_ge77_muons, dtype=np.int32)

        for area_name in ["pit", "bot", "top", "wall"]:
            n_area = allocation[area_name]
            if n_area == 0:
                continue

            area_mask = (layers == area_name)
            area_indices = np.where(area_mask)[0]
            if len(area_indices) == 0:
                continue

            B_area = B[:, area_indices]
            centers_area = centers[area_indices]
            layers_area = layers[area_indices]

            if optimize == "nc":
                sel_local, _, _, _ = greedy_select_nc(
                    B_area, N=n_area, M=M,
                    centers=centers_area, layers=layers_area,
                    min_spacing=min_spacing, verbose=False,
                )
                sel_global = [int(area_indices[i]) for i in sel_local]
                all_selected.extend(sel_global)

            elif optimize == "muon-ge77":
                sel_local, _, nc_det, muon_det = greedy_select_muon(
                    B_area, N=n_area, W=W,
                    nc_to_muon_local=nc_to_muon_local,
                    eligible_nc_mask=eligible_nc_mask,
                    num_ge77_muons=num_ge77_muons,
                    centers=centers_area, layers=layers_area,
                    min_spacing=min_spacing,
                    muon_weight_k=muon_weight_k,
                    verbose=False,
                )
                sel_global = [int(area_indices[i]) for i in sel_local]
                all_selected.extend(sel_global)
                shared_nc_detected |= nc_det
                shared_muon_counts += muon_det

            del B_area

        selected_cols = all_selected

        if optimize == "nc":
            coverage_counts = np.zeros(B.shape[0], dtype=np.int16)
            for col in selected_cols:
                s, e = B.indptr[col], B.indptr[col + 1]
                coverage_counts[B.indices[s:e]] += 1
            final_eff = int(np.sum(coverage_counts >= M)) / B.shape[0]
        else:
            n_det = int(np.sum(shared_muon_counts >= W))
            final_eff = n_det / num_ge77_muons if num_ge77_muons > 0 else 0.0

    else:
        if optimize == "nc":
            mw_kwargs = {}
            if nc_muon_weight_data is not None:
                mw_kwargs = {
                    "muon_weight_k": muon_weight_k,
                    "nc_to_muon_local": nc_muon_weight_data["nc_to_muon_local"],
                    "num_muons": nc_muon_weight_data["num_muons"],
                }
            selected_cols, effs, _, _ = greedy_select_nc(
                B, N=N, M=M,
                centers=centers, layers=layers,
                min_spacing=min_spacing, verbose=False,
                **mw_kwargs,
            )
            final_eff = effs[-1] if effs else 0.0

        elif optimize == "muon-ge77":
            selected_cols, effs, _, _ = greedy_select_muon(
                B, N=N, W=W,
                nc_to_muon_local=nc_to_muon_local,
                eligible_nc_mask=eligible_nc_mask,
                num_ge77_muons=num_ge77_muons,
                centers=centers, layers=layers,
                min_spacing=min_spacing,
                muon_weight_k=muon_weight_k,
                verbose=False,
            )
            final_eff = effs[-1] if effs else 0.0

    del B
    return selected_cols, final_eff


# ---------------------------------------------------------------------------
# Full sensitivity analysis
# ---------------------------------------------------------------------------

def run_sensitivity(
    filepath: str,
    N: int,
    m: int,
    area_ratios: dict[str, float],
    optimize: str,
    M: int = 1,
    W: int = 1,
    min_spacing: float = 0.0,
    per_area: bool = False,
    muon_weight_k: float | None = None,
    seed: int = 42,
    deltas: list[float] | None = None,
    output_dir: str = "sensitivity_results",
    verbose: bool = True,
) -> None:
    """
    Run full sensitivity analysis: baseline + perturbed greedy runs.

    Parameters
    ----------
    filepath : str
        Path to HDF5 data file.
    N : int
        Number of voxels to select.
    m : int
        Hit threshold per voxel.
    area_ratios : dict[str, float]
        Nominal area ratios (baseline).
    optimize : str
        "nc" or "muon-ge77".
    M, W : int
        Multiplicity / coincidence thresholds.
    min_spacing : float
        Minimum voxel spacing (mm).
    per_area : bool
        Optimize each area independently.
    muon_weight_k : float or None
        Muon weighting parameter.
    seed : int
        Random seed for stochastic rounding (same for all runs).
    deltas : list[float] or None
        Perturbation values. Default: [-0.20, -0.10, -0.05, +0.05, +0.10, +0.20].
    output_dir : str
        Directory for output files.
    verbose : bool
        Print progress.
    """
    if deltas is None:
        deltas = [-0.20, -0.10, -0.05, +0.05, +0.10, +0.20]

    k_values = list(range(25, N + 1, 25))
    if k_values[-1] != N:
        k_values.append(N)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Sensitivity Analysis — Area Ratio Perturbation")
    print("=" * 65)
    print(f"  Mode:        {optimize}")
    print(f"  N:           {N}")
    print(f"  M:           {M}, m: {m}")
    if optimize == "muon-ge77":
        print(f"  W:           {W}")
    print(f"  Per-area:    {per_area}")
    print(f"  Spacing:     {min_spacing:.0f} mm")
    print(f"  Deltas:      {deltas}")
    print(f"  Nominal ratios: {area_ratios}")
    print(f"  Seed:        {seed}")
    if muon_weight_k is not None:
        print(f"  Muon weight: k = {muon_weight_k:.4f}")
    print(f"  Output dir:  {out_dir}")
    print()

    # Baseline
    print(f"[Baseline] Running nominal greedy (δ = 0)...")
    t0 = time.time()

    baseline_selected, baseline_eff = run_single_greedy(
        filepath=filepath, N=N, m=m,
        area_ratios=area_ratios,
        optimize=optimize, M=M, W=W,
        min_spacing=min_spacing,
        per_area=per_area,
        muon_weight_k=muon_weight_k,
        seed=seed, verbose=True,
    )

    t_baseline = time.time() - t0
    print(f"  Baseline efficiency: {baseline_eff:.4%}")
    print(f"  Baseline time: {t_baseline:.1f}s")

    # Load layers for per-area Jaccard
    with h5py.File(filepath, "r") as f:
        voxel_keys = sorted(
            c.decode() if isinstance(c, bytes) else str(c)
            for c in f["target_columns"][:]
        )
        valid_mask = get_valid_voxel_mask(f, voxel_keys, verbose=False)
        valid_keys = [k for k, v in zip(voxel_keys, valid_mask) if v]
        layers_all = np.empty(len(valid_keys), dtype=object)
        for i, vkey in enumerate(valid_keys):
            layer_raw = f[f"voxels/{vkey}/layer"][()]
            layers_all[i] = (layer_raw.decode()
                             if isinstance(layer_raw, bytes)
                             else str(layer_raw))

    # Perturbed runs
    results: dict[float, dict] = {}

    for idx, delta in enumerate(deltas):
        perturbed_ratios = {
            area: ratio * (1.0 + delta)
            for area, ratio in area_ratios.items()
        }

        print(f"\n[{idx+1}/{len(deltas)}] δ = {delta:+.0%} | "
              f"Ratios: { {a: f'{r:.4f}' for a, r in perturbed_ratios.items()} }")
        t0 = time.time()

        pert_selected, pert_eff = run_single_greedy(
            filepath=filepath, N=N, m=m,
            area_ratios=perturbed_ratios,
            optimize=optimize, M=M, W=W,
            min_spacing=min_spacing,
            per_area=per_area,
            muon_weight_k=muon_weight_k,
            seed=seed, verbose=False,
        )

        dt = time.time() - t0
        delta_cov = ((pert_eff - baseline_eff) / baseline_eff
                     if baseline_eff > 0 else 0.0)

        print(f"  Efficiency: {pert_eff:.4%} (ΔC = {delta_cov:+.4%})")
        print(f"  Time: {dt:.1f}s")

        j_curve = compute_jaccard_curve(
            baseline_selected, pert_selected, k_values,
        )
        per_area_j = compute_per_area_jaccard(
            baseline_selected, pert_selected, layers_all, N,
        )

        results[delta] = {
            "selected_cols": pert_selected,
            "efficiency": pert_eff,
            "delta_coverage": delta_cov,
            "jaccard_curve": j_curve,
            "per_area_jaccard": per_area_j,
            "perturbed_ratios": perturbed_ratios,
            "runtime_s": dt,
        }

        j_50 = j_curve[k_values.index(50)] if 50 in k_values else None
        j_N = j_curve[-1]
        if j_50 is not None:
            print(f"  J_50 = {j_50:.3f}")
        print(f"  J_{N} = {j_N:.3f}")
        print(f"  Per-area J: {per_area_j}")

    # Save JSON
    json_output = out_dir / f"sensitivity_{optimize}_N{N}.json"
    json_data = {
        "config": {
            "optimize": optimize, "N": N, "M": M, "m": m, "W": W,
            "per_area": per_area, "min_spacing": min_spacing,
            "nominal_ratios": area_ratios,
            "muon_weight_k": muon_weight_k,
            "seed": seed, "deltas": deltas, "k_values": k_values,
        },
        "baseline": {
            "efficiency": baseline_eff,
            "selected_cols": baseline_selected,
            "runtime_s": t_baseline,
        },
        "perturbations": {
            f"{d:+.2f}": {
                "efficiency": r["efficiency"],
                "delta_coverage": r["delta_coverage"],
                "jaccard_curve": r["jaccard_curve"],
                "per_area_jaccard": r["per_area_jaccard"],
                "perturbed_ratios": r["perturbed_ratios"],
                "runtime_s": r["runtime_s"],
                "selected_cols": r["selected_cols"],
            }
            for d, r in results.items()
        },
    }
    with open(json_output, "w") as jf:
        json.dump(json_data, jf, indent=2)
    print(f"\nResults saved to {json_output}")

    # Plots
    print("\nGenerating plots...")
    plot_jaccard_curves(
        results, k_values,
        out_dir / f"sensitivity_jaccard_{optimize}_N{N}.png",
        optimize,
    )
    plot_coverage_change(
        results,
        out_dir / f"sensitivity_coverage_{optimize}_N{N}.png",
        optimize,
    )
    plot_per_area_jaccard(
        results,
        out_dir / f"sensitivity_perarea_{optimize}_N{N}.png",
        optimize,
    )

    # Summary table
    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print(f"{'=' * 65}")
    print(f"{'δ':>8} | {'Efficiency':>11} | {'ΔC':>8} | "
          f"{'J_50':>6} | {'J_100':>6} | {'J_200':>6} | "
          f"{'J_' + str(N):>6}")
    print("-" * 65)

    print(f"{'0':>8} | {baseline_eff:>11.4%} | {'—':>8} | "
          f"{'—':>6} | {'—':>6} | {'—':>6} | {'—':>6}")

    for d in sorted(results.keys()):
        r = results[d]
        jc = r["jaccard_curve"]

        def _get_jk(k: int) -> str:
            if k in k_values:
                return f"{jc[k_values.index(k)]:.3f}"
            return "—"

        print(f"{d:>+8.0%} | {r['efficiency']:>11.4%} | "
              f"{r['delta_coverage']:>+8.3%} | "
              f"{_get_jk(50):>6} | {_get_jk(100):>6} | "
              f"{_get_jk(200):>6} | {_get_jk(N):>6}")

    total_time = t_baseline + sum(r["runtime_s"] for r in results.values())
    print(f"\nTotal runtime: {total_time:.0f}s ({total_time/3600:.1f}h)")