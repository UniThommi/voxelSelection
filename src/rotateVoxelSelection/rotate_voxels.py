#!/usr/bin/env python3
"""
Rotate a greedy PMT voxel selection by an azimuthal angle phi.

For each selected voxel:
  1. Rotate its (x, y) coordinates by phi (fraction of 2π, e.g. 0.5 → 180°).
  2. Find the nearest valid voxel on the same sub-surface from a pool of all
     available voxels.
  3. For wall voxels the candidate must share the same z-level (within 1 mm).

After all voxels are mapped, two batch checks are run:
  - Collision: no two originals map to the same target.
  - Self-overlap: no target index appears among the original selected indices
    (same physical PMT placement would remain unchanged).

A spacing-distribution test then compares pairwise distances before/after.
A 3D plot of original (red) vs. rotated (green) positions is always produced.

Usage — single angle:
    python rotate_voxels.py \\
        --all-voxels all_valid.json \\
        --selected greedy_N300.json \\
        --angle 0.25 \\          # fraction of 2π → 90°
        --output-dir plots/

Usage — explore all valid angles interactively (omit --angle):
    python rotate_voxels.py \\
        --all-voxels all_valid.json \\
        --selected greedy_N300.json \\
        --output-dir plots/
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from pmtopt.geometry import (
    R_PIT, R_ZYL_BOT, R_ZYLINDER,
    Z_BASE_GLOBAL, H_ZYLINDER,
    is_valid_pmt_position,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WALL_Z_TOL: float = 1.0   # mm — tolerance for wall z-level matching

# Per-metric relative-change thresholds for the spacing test
SPACING_THRESHOLDS: dict[str, float] = {
    "pw_mean": 0.02,   # mean pairwise distance
    "pw_std":  0.05,   # std of pairwise distances
    "nn_mean": 0.05,   # mean nearest-neighbour distance
    "nn_min":  0.10,   # minimum nearest-neighbour distance
}

LAYER_MARKERS: dict[str, str] = {"pit": "o", "bot": "s", "top": "^", "wall": "D"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_voxel_file(path: str) -> list[dict]:
    """Load a voxel list from JSON.

    Accepts both a bare list ``[...]`` and the greedy output wrapper
    ``{"selected_voxels": [...]}``.
    """
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "selected_voxels" in data:
        return data["selected_voxels"]
    raise ValueError(
        f"{path}: expected a list or a dict with 'selected_voxels', "
        f"got {type(data).__name__}"
    )


def derive_output_stem(selected_path: str, phi_frac: float) -> str:
    """Base filename stem: ``{stem}_rotated_{phi_frac:.4f}x2pi``."""
    return f"{Path(selected_path).stem}_rotated_{phi_frac:.4f}x2pi"


# ---------------------------------------------------------------------------
# Core rotation
# ---------------------------------------------------------------------------

def rotate_point(
    x: float, y: float, z: float, phi_rad: float
) -> tuple[float, float, float]:
    """Rotate (x, y) by phi_rad in the xy-plane; z is unchanged."""
    cos_p = math.cos(phi_rad)
    sin_p = math.sin(phi_rad)
    return (x * cos_p - y * sin_p, x * sin_p + y * cos_p, z)


def build_candidate_pool(voxel: dict, all_voxels: list[dict]) -> list[dict]:
    """Return all voxels on the same layer (and same z for wall).

    Raises ValueError if no candidates exist.
    """
    layer = voxel["layer"]
    pool = [v for v in all_voxels if v["layer"] == layer]
    if layer == "wall":
        z_orig = voxel["center"][2]
        pool = [v for v in pool if abs(v["center"][2] - z_orig) < WALL_Z_TOL]
    if not pool:
        raise ValueError(
            f"No candidates for voxel '{voxel['index']}' "
            f"(layer={layer}, z={voxel['center'][2]:.1f} mm)"
        )
    return pool


def find_nearest_candidate(
    rotated_center: tuple[float, float, float],
    candidates: list[dict],
) -> dict:
    """Return the candidate whose center is closest to rotated_center."""
    centers = np.array([v["center"] for v in candidates])   # (N, 3)
    target = np.array(rotated_center)
    idx = int(np.argmin(np.linalg.norm(centers - target, axis=1)))
    return candidates[idx]


def compute_voxel_mapping(
    selected: list[dict],
    all_voxels: list[dict],
    phi_frac: float,
) -> list[tuple[dict, dict]]:
    """Map each selected voxel to its nearest rotated counterpart.

    Parameters
    ----------
    phi_frac:
        Rotation as a fraction of 2π (e.g. 0.5 → 180°).

    Returns
    -------
    List of (original_voxel, target_voxel) pairs, in the same order as
    *selected*.  No validation is performed here; call check_mapping()
    afterwards.
    """
    phi_rad = phi_frac * 2.0 * math.pi
    mapping: list[tuple[dict, dict]] = []
    for voxel in selected:
        x, y, z = voxel["center"]
        rotated = rotate_point(x, y, z, phi_rad)
        pool = build_candidate_pool(voxel, all_voxels)
        target = find_nearest_candidate(rotated, pool)
        mapping.append((voxel, target))
    return mapping


def check_mapping(mapping: list[tuple[dict, dict]]) -> None:
    """Validate the full mapping after all voxels have been assigned.

    Raises RuntimeError for:

    - **Collision**: two original voxels map to the same target index.
    - **Self-overlap**: the set of target indices intersects the set of
      original indices, meaning the rotated selection occupies positions
      that were already selected (rotation angle likely too small).
    """
    original_indices: set[str] = {orig["index"] for orig, _ in mapping}

    # Build target_index → [originating voxel ids]
    target_map: dict[str, list[str]] = {}
    for orig, tgt in mapping:
        target_map.setdefault(tgt["index"], []).append(orig["index"])

    # Collision check
    collisions = {t: origs for t, origs in target_map.items() if len(origs) > 1}
    if collisions:
        msgs = [f"  target '{t}' ← {origs}" for t, origs in collisions.items()]
        raise RuntimeError(
            "Collision: multiple voxels mapped to the same target:\n"
            + "\n".join(msgs)
        )

    # Self-overlap check
    overlap = original_indices & set(target_map.keys())
    if overlap:
        raise RuntimeError(
            f"Self-overlap: {len(overlap)} target voxel(s) coincide with "
            f"voxels in the original selection — the rotation angle may be "
            f"too small or the grid too symmetric for this angle:\n"
            f"  {sorted(overlap)}"
        )


def check_validity(mapping: list[tuple[dict, dict]]) -> None:
    """Check that every target voxel is a geometrically valid PMT position.

    Raises RuntimeError listing all invalid targets.
    """
    invalid: list[str] = []
    for _, tgt in mapping:
        center = tgt["center"]
        layer  = tgt["layer"]
        if not is_valid_pmt_position(center, layer):
            invalid.append(
                f"  '{tgt['index']}' layer={layer} center={center}"
            )
    if invalid:
        raise RuntimeError(
            f"{len(invalid)} rotated voxel(s) failed the PMT placement "
            f"validity check:\n" + "\n".join(invalid)
        )


def assemble_output_voxels(mapping: list[tuple[dict, dict]]) -> list[dict]:
    """Return the list of target voxels in original selection order."""
    return [tgt for _, tgt in mapping]


# ---------------------------------------------------------------------------
# Explore mode: find all valid rotation angles
# ---------------------------------------------------------------------------

def _precompute_scores(
    source: dict,
    candidates: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute scoring coefficients for one source voxel.

    For rotation angle θ the squared distance from the rotated source to
    candidate c is:

        dist²(θ, c) = K[c] - 2*(A[c]*cos(θ) + B[c]*sin(θ))

    so the nearest candidate is ``argmax_c (A[c]*cos(θ) + B[c]*sin(θ) - K[c]/2)``.

    Returns
    -------
    A, B, K : np.ndarray of shape (len(candidates),)
    """
    x, y, z = source["center"]
    cxyz = np.array([v["center"] for v in candidates])   # (C, 3)
    cx, cy, cz = cxyz[:, 0], cxyz[:, 1], cxyz[:, 2]

    A = x * cx + y * cy
    B = x * cy - y * cx
    K = cx**2 + cy**2 + (z - cz)**2 + (x**2 + y**2)
    return A, B, K


def find_valid_angles(
    selected: list[dict],
    all_voxels: list[dict],
    n_samples: int = 3600,
    skip_validity: bool = False,
) -> list[float]:
    """Scan all rotation angles and return those producing valid mappings.

    Uses dense angular sampling (``n_samples`` evenly-spaced steps from 0
    to 2π exclusive).  Since adjacent distinct voxel-assignment regions are
    separated by >2° on this detector, the default of 3600 steps (0.1°) is
    effectively exact.

    Each source voxel's nearest-candidate assignment is computed via
    vectorised numpy operations over all sample angles at once.

    Parameters
    ----------
    selected : list of voxel dicts (the selection to rotate).
    all_voxels : the full candidate pool.
    n_samples : number of angle steps (default 3600 → every 0.1°).
    skip_validity : if True, skip the ``is_valid_pmt_position`` check.

    Returns
    -------
    Sorted list of phi_frac values (one representative per distinct valid
    mapping configuration).
    """
    V = len(selected)
    original_indices: set[str] = {v["index"] for v in selected}

    # Build per-source-voxel candidate pools and scoring coefficients
    pools: list[list[dict]] = []
    A_list: list[np.ndarray] = []
    B_list: list[np.ndarray] = []
    K_list: list[np.ndarray] = []

    for voxel in selected:
        pool = build_candidate_pool(voxel, all_voxels)
        A, B, K = _precompute_scores(voxel, pool)
        pools.append(pool)
        A_list.append(A)
        B_list.append(B)
        K_list.append(K)

    # Precompute validity mask for every voxel in all_voxels (if needed)
    if not skip_validity:
        validity_mask = np.array(
            [is_valid_pmt_position(v["center"], v["layer"]) for v in all_voxels],
            dtype=bool,
        )
        # Map each pool voxel to its index in all_voxels
        all_index_map: dict[str, int] = {v["index"]: i
                                          for i, v in enumerate(all_voxels)}

    # Sample angles
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    cos_t = np.cos(theta)   # (n_samples,)
    sin_t = np.sin(theta)   # (n_samples,)

    # mapping_matrix[i, j] = local index into pool[i] of nearest candidate
    # for source voxel i at angle sample j
    mapping_matrix = np.empty((V, n_samples), dtype=np.int32)

    print(f"  Scanning {n_samples} angles for {V} voxels ...")
    for i, (A, B, K) in enumerate(zip(A_list, B_list, K_list)):
        # scores shape: (C, n_samples)
        scores = (A[:, None] * cos_t[None, :]
                  + B[:, None] * sin_t[None, :]
                  - K[:, None] / 2.0)
        mapping_matrix[i] = np.argmax(scores, axis=0)

    # Convert local pool indices → global voxel index strings per angle
    # We work with integer IDs to avoid slow Python string operations.
    # Build a (V, n_samples) array of global all_voxels indices.
    global_idx_matrix = np.empty((V, n_samples), dtype=np.int32)
    pool_global_indices: list[np.ndarray] = []
    for i, pool in enumerate(pools):
        gi = np.array([all_index_map[v["index"]] if not skip_validity
                       else 0   # placeholder when not checking
                       for v in pool], dtype=np.int32)
        if skip_validity:
            # Just store pool-local indices as strings via the pool list
            pool_global_indices.append(None)  # not used
        else:
            pool_global_indices.append(gi)
        if not skip_validity:
            global_idx_matrix[i] = gi[mapping_matrix[i]]

    # Per-angle checks — vectorised where possible
    print(f"  Checking collision, self-overlap"
          + ("" if skip_validity else ", validity") + " ...")

    # original_int_indices: set of all_voxels positions of original voxels
    if not skip_validity:
        original_int_set = {all_index_map[v["index"]] for v in selected
                            if v["index"] in all_index_map}

    valid_mask_angles = np.ones(n_samples, dtype=bool)

    for j in range(n_samples):
        targets_local = mapping_matrix[:, j]   # shape (V,)

        # Collision: all target local indices must be distinct
        # (we check per-source-voxel within its own pool, so we need the
        # actual global pool-voxel identity — use the pool object index)
        # Two source voxels can collide only if they share a pool (same layer),
        # so use the pool voxel index tuple as the collision key.
        target_keys = [pools[i][targets_local[i]]["index"] for i in range(V)]
        if len(set(target_keys)) < V:
            valid_mask_angles[j] = False
            continue

        # Self-overlap: no target should be in original selection
        if original_indices & set(target_keys):
            valid_mask_angles[j] = False
            continue

        # Validity
        if not skip_validity:
            gi = global_idx_matrix[:, j]
            if not validity_mask[gi].all():
                valid_mask_angles[j] = False
                continue

    # Find distinct valid mapping configurations via run-length encoding.
    # Two angle samples with identical target_keys arrays are the same config.
    # Build a compact hash per angle (use frozenset of target strings).
    valid_indices = np.where(valid_mask_angles)[0]
    if len(valid_indices) == 0:
        return []

    # Group consecutive valid angles that share the same full mapping tuple
    seen_configs: dict[tuple, float] = {}   # config → representative phi_frac
    for j in valid_indices:
        config_key = tuple(
            pools[i][mapping_matrix[i, j]]["index"] for i in range(V)
        )
        if config_key not in seen_configs:
            seen_configs[config_key] = j / n_samples   # phi_frac representative

    valid_fracs = sorted(seen_configs.values())
    return valid_fracs


def prompt_angle_selection(valid_fracs: list[float]) -> list[float]:
    """Display valid angles and prompt the user to select which to run.

    Accepts:
      - Space- or comma-separated indices: ``0 2 4`` or ``0,2,4``
      - Ranges: ``1-3`` (inclusive)
      - The keyword ``all``
    """
    print(f"\n{'=' * 50}")
    print(f"Found {len(valid_fracs)} valid rotation angle(s):")
    print(f"{'=' * 50}")
    print(f"  {'#':>4}  {'phi_frac':>10}  {'phi_deg':>9}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*9}")
    for idx, frac in enumerate(valid_fracs):
        deg = frac * 360.0
        print(f"  {idx:>4}  {frac:>10.4f}  {deg:>8.2f}°")
    print(f"{'=' * 50}")

    while True:
        raw = input(
            'Select angles to run (e.g. "0 2 4", "1-3", or "all"): '
        ).strip()
        if not raw:
            print("  No input — please enter at least one index.")
            continue
        if raw.lower() == "all":
            return list(valid_fracs)
        try:
            selected_indices: list[int] = []
            for token in raw.replace(",", " ").split():
                if "-" in token:
                    lo, hi = token.split("-", 1)
                    selected_indices.extend(range(int(lo), int(hi) + 1))
                else:
                    selected_indices.append(int(token))
            # Validate
            out_of_range = [i for i in selected_indices
                            if i < 0 or i >= len(valid_fracs)]
            if out_of_range:
                print(f"  Invalid indices: {out_of_range}. "
                      f"Valid range: 0–{len(valid_fracs) - 1}.")
                continue
            return [valid_fracs[i] for i in selected_indices]
        except ValueError:
            print("  Could not parse input. Try e.g. '0 2', '1-3', or 'all'.")


# ---------------------------------------------------------------------------
# Single-angle execution (extracted so both modes can call it)
# ---------------------------------------------------------------------------

def run_for_angle(
    selected: list[dict],
    all_voxels: list[dict],
    phi_frac: float,
    output_dir: Path,
    selected_path: str,
    skip_validity: bool,
    plot: bool,
) -> None:
    """Run the full rotation pipeline for one angle and save outputs.

    In explore mode ``output_dir`` is a subfolder per angle; in single-angle
    mode it is the directory passed directly by the user.
    """
    phi_deg = phi_frac * 360.0
    print(f"\n{'=' * 62}")
    print(f"Running angle: {phi_frac:.4f} × 2π = {phi_deg:.1f}°")
    print(f"{'=' * 62}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = derive_output_stem(selected_path, phi_frac)
    output_path = output_dir / f"{stem}.json"

    # Compute mapping
    print("Computing rotation mapping ...")
    mapping = compute_voxel_mapping(selected, all_voxels, phi_frac)

    # Validation
    print("Validating mapping (collision + self-overlap check) ...")
    check_mapping(mapping)
    print("  OK — no collisions or self-overlaps detected")

    if skip_validity:
        print("PMT placement validity check skipped (--skip-validity).")
    else:
        print("Checking PMT placement validity of rotated voxels ...")
        check_validity(mapping)
        print(f"  OK — all {len(mapping)} rotated voxels are valid PMT positions")

    rotated_voxels = assemble_output_voxels(mapping)

    # Spacing distribution test
    print("\nRunning spacing distribution test ...")
    spacing_results = run_spacing_test(
        selected, rotated_voxels,
        plot=plot,
        output_dir=output_dir if plot else None,
    )
    print_spacing_report(spacing_results, phi_frac, mapping)

    # 3D plot
    plot_3d_path = output_path.with_name(output_path.stem + "_3d.png")
    plot_rotated_voxels(selected, rotated_voxels, plot_3d_path, phi_frac)

    # 2D per-layer arrow plot
    plot_2d_path = output_path.with_name(output_path.stem + "_2d_arrows.png")
    plot_2d_arrows(mapping, plot_2d_path, phi_frac)

    # JSON output
    out_data = {
        "config": {
            "rotation_angle_2pi": phi_frac,
            "source_file": str(selected_path),
            "all_voxels_file": "",   # filled by caller if available
            "n_voxels": len(rotated_voxels),
        },
        "selected_voxels": rotated_voxels,
    }
    with open(output_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Output written to {output_path}")


# ---------------------------------------------------------------------------
# Spacing distribution test
# ---------------------------------------------------------------------------

def compute_pairwise_distances(centers: np.ndarray) -> np.ndarray:
    """Return a flat array of all N*(N-1)/2 pairwise Euclidean distances."""
    n = len(centers)
    diffs = centers[:, None, :] - centers[None, :, :]   # (n, n, 3)
    dists_sq = np.sum(diffs ** 2, axis=2)               # (n, n)
    idx = np.triu_indices(n, k=1)
    return np.sqrt(dists_sq[idx])


def compute_spacing_stats(centers: np.ndarray) -> Optional[dict]:
    """Summary statistics of pairwise and nearest-neighbour distances.

    Returns None if fewer than 2 voxels are given.
    """
    n = len(centers)
    if n < 2:
        return None

    diffs = centers[:, None, :] - centers[None, :, :]   # (n, n, 3)
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))          # (n, n)

    pw = dists[np.triu_indices(n, k=1)]

    dists_nn = dists.copy()
    np.fill_diagonal(dists_nn, np.inf)
    nn = dists_nn.min(axis=1)

    return {
        "pw_min":  float(pw.min()),
        "pw_mean": float(pw.mean()),
        "pw_max":  float(pw.max()),
        "pw_std":  float(pw.std()),
        "nn_min":  float(nn.min()),
        "nn_mean": float(nn.mean()),
        "nn_max":  float(nn.max()),
        "nn_std":  float(nn.std()),
    }


def run_spacing_test(
    before: list[dict],
    after: list[dict],
    plot: bool = False,
    output_dir: Optional[Path] = None,
) -> dict:
    """Compare per-layer spacing distributions before and after rotation."""
    layers = sorted({v["layer"] for v in before})
    results: dict = {}

    for layer in layers:
        b_centers = np.array([v["center"] for v in before if v["layer"] == layer])
        a_centers = np.array([v["center"] for v in after  if v["layer"] == layer])

        b_stats = compute_spacing_stats(b_centers)
        a_stats = compute_spacing_stats(a_centers)

        if b_stats is None or a_stats is None:
            results[layer] = {
                "before": b_stats, "after": a_stats,
                "delta_rel": {}, "passed": None, "count": len(b_centers),
            }
            continue

        delta_rel: dict[str, float] = {}
        for key in SPACING_THRESHOLDS:
            bv, av = b_stats[key], a_stats[key]
            delta_rel[key] = (av - bv) / bv if bv != 0.0 else 0.0

        passed = all(
            abs(delta_rel[k]) <= thr for k, thr in SPACING_THRESHOLDS.items()
        )

        results[layer] = {
            "before": b_stats,
            "after":  a_stats,
            "delta_rel": delta_rel,
            "passed": passed,
            "count": len(b_centers),
        }

        if plot and output_dir is not None:
            _save_spacing_histogram(layer, b_centers, a_centers, output_dir)

    return results


def _save_spacing_histogram(
    layer: str,
    before: np.ndarray,
    after: np.ndarray,
    output_dir: Path,
) -> None:
    """Save an overlaid pairwise-distance histogram for one layer."""
    pw_b = compute_pairwise_distances(before)
    pw_a = compute_pairwise_distances(after)
    fig, ax = plt.subplots(figsize=(8, 5))
    lo = min(pw_b.min(), pw_a.min())
    hi = max(pw_b.max(), pw_a.max())
    bins = np.linspace(lo, hi, 40)
    ax.hist(pw_b, bins=bins, alpha=0.6, color="red",       label="before", density=False)
    ax.hist(pw_a, bins=bins, alpha=0.6, color="limegreen", label="after",  density=False)
    ax.set_xlabel("Pairwise distance (mm)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Pairwise distance distribution — {layer.upper()}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / f"spacing_{layer}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Spacing histogram saved to {out}")


def print_spacing_report(
    results: dict,
    phi_frac: float,
    mapping: list[tuple[dict, dict]],
) -> None:
    """Print a formatted spacing-test summary to stdout."""
    phi_deg = phi_frac * 360.0
    print(f"\n{'=' * 62}")
    print(f"Spacing distribution test  "
          f"(phi = {phi_frac:.4f} × 2π = {phi_deg:.1f}°)")
    print(f"{'=' * 62}")
    print(f"  Total voxels mapped: {len(mapping)}")

    all_passed = True
    for layer, res in sorted(results.items()):
        if res["passed"] is None:
            status = "SKIP"
        elif res["passed"]:
            status = "PASS"
        else:
            status = "FAIL"
            all_passed = False

        print(f"\n  Layer {layer.upper():>4}  ({res['count']} voxels)  [{status}]")
        if res["before"] is None:
            print("    < 2 voxels — skipped")
            continue

        col = f"    {'Metric':<10} {'Before (mm)':>12} {'After (mm)':>12}" \
              f" {'Δ rel':>9}  Threshold"
        print(col)
        print("    " + "-" * (len(col) - 4))
        for key, thr in SPACING_THRESHOLDS.items():
            bv = res["before"][key]
            av = res["after"][key]
            dr = res["delta_rel"][key]
            flag = "✓" if abs(dr) <= thr else "✗"
            print(f"    {key:<10} {bv:>12.1f} {av:>12.1f} {dr:>+8.1%}  "
                  f"≤ {thr:.0%} {flag}")

    print(f"\n{'=' * 62}")
    print(f"Overall spacing test: {'PASS' if all_passed else 'FAIL'}")
    print(f"{'=' * 62}\n")


# ---------------------------------------------------------------------------
# 3D plot
# ---------------------------------------------------------------------------

def plot_rotated_voxels(
    original_voxels: list[dict],
    rotated_voxels: list[dict],
    output_path: Path,
    phi_frac: float,
) -> None:
    """3D scatter of original (red) and rotated (green) voxel positions."""
    Z_BASE = Z_BASE_GLOBAL
    Z_TOP  = Z_BASE + H_ZYLINDER

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection="3d")

    theta = np.linspace(0, 2 * np.pi, 200)
    n_vert = 24
    theta_lines = np.linspace(0, 2 * np.pi, n_vert, endpoint=False)
    for z in [Z_BASE, Z_TOP]:
        ax.plot(R_ZYLINDER * np.cos(theta), R_ZYLINDER * np.sin(theta), z,
                color="gray", alpha=0.3, linewidth=0.5)
    for t in theta_lines:
        ax.plot([R_ZYLINDER * np.cos(t)] * 2, [R_ZYLINDER * np.sin(t)] * 2,
                [Z_BASE, Z_TOP], color="gray", alpha=0.3, linewidth=0.5)

    ax.plot(R_PIT     * np.cos(theta), R_PIT     * np.sin(theta), Z_BASE,
            color="steelblue", alpha=0.6, linewidth=1.0,
            label=f"Pit (r={R_PIT})")
    ax.plot(R_ZYL_BOT * np.cos(theta), R_ZYL_BOT * np.sin(theta), Z_BASE,
            color="seagreen",  alpha=0.6, linewidth=1.0,
            label=f"Bot inner (r={R_ZYL_BOT})")

    for layer in ["pit", "bot", "top", "wall"]:
        marker = LAYER_MARKERS[layer]
        orig_pts = np.array([v["center"] for v in original_voxels
                             if v["layer"] == layer])
        rot_pts  = np.array([v["center"] for v in rotated_voxels
                             if v["layer"] == layer])
        if len(orig_pts):
            ax.scatter(orig_pts[:, 0], orig_pts[:, 1], orig_pts[:, 2],
                       c="red", marker=marker, s=25, alpha=0.7,
                       edgecolors="darkred", linewidths=0.4,
                       label=f"Original {layer} ({len(orig_pts)})")
        if len(rot_pts):
            ax.scatter(rot_pts[:, 0], rot_pts[:, 1], rot_pts[:, 2],
                       c="limegreen", marker=marker, s=25, alpha=0.7,
                       edgecolors="darkgreen", linewidths=0.4,
                       label=f"Rotated {layer} ({len(rot_pts)})")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    phi_deg = phi_frac * 360.0
    ax.set_title(
        f"Voxel Rotation  (phi = {phi_frac:.4f} × 2π = {phi_deg:.1f}°)\n"
        f"Red = original  |  Green = rotated  |  N = {len(original_voxels)}"
    )
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    max_range = max(R_ZYLINDER, (Z_TOP - Z_BASE) / 2)
    mid_z = (Z_BASE + Z_TOP) / 2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"3D plot saved to {output_path}")


# ---------------------------------------------------------------------------
# 2D per-layer arrow plot
# ---------------------------------------------------------------------------

def plot_2d_arrows(
    mapping: list[tuple[dict, dict]],
    output_path: Path,
    phi_frac: float,
) -> None:
    """2x2 figure with one subplot per layer showing original→rotated shifts."""
    phi_deg = phi_frac * 360.0
    layers = ["pit", "bot", "top", "wall"]
    layer_titles = {
        "pit":  "Pit  (x-y plane)",
        "bot":  "Bot  (x-y plane)",
        "top":  "Top  (x-y plane)",
        "wall": "Wall  (φ-z unwrapped)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"Per-layer voxel shifts  (phi = {phi_frac:.4f} × 2π = {phi_deg:.1f}°)\n"
        f"Red = original  |  Green = rotated  |  Arrows = shift",
        fontsize=13,
    )

    for ax, layer in zip(axes.flat, layers):
        pairs = [(orig, tgt) for orig, tgt in mapping if orig["layer"] == layer]

        if not pairs:
            ax.set_title(layer_titles[layer], fontsize=11)
            ax.text(0.5, 0.5, "no voxels", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
            continue

        if layer == "wall":
            orig_phi = np.degrees(
                np.arctan2([o["center"][1] for o, _ in pairs],
                           [o["center"][0] for o, _ in pairs])
            )
            orig_z = np.array([o["center"][2] for o, _ in pairs])
            rot_phi = np.degrees(
                np.arctan2([t["center"][1] for _, t in pairs],
                           [t["center"][0] for _, t in pairs])
            )
            rot_z = np.array([t["center"][2] for _, t in pairs])

            dphi = rot_phi - orig_phi
            dphi = (dphi + 180.0) % 360.0 - 180.0
            dz   = rot_z - orig_z

            ax.scatter(orig_phi, orig_z, c="red",       s=25, alpha=0.8,
                       edgecolors="darkred",   linewidths=0.4, zorder=3,
                       label=f"Original ({len(pairs)})")
            ax.scatter(rot_phi,  rot_z,  c="limegreen", s=25, alpha=0.8,
                       edgecolors="darkgreen", linewidths=0.4, zorder=3,
                       label=f"Rotated ({len(pairs)})")
            for px, pz, dx, dz_ in zip(orig_phi, orig_z, dphi, dz):
                ax.annotate(
                    "", xy=(px + dx, pz + dz_), xytext=(px, pz),
                    arrowprops=dict(arrowstyle="-|>", color="steelblue",
                                   lw=0.8, mutation_scale=8),
                )
            ax.set_xlabel("φ (deg)", fontsize=10)
            ax.set_ylabel("z (mm)", fontsize=10)

        else:
            orig_x = np.array([o["center"][0] for o, _ in pairs])
            orig_y = np.array([o["center"][1] for o, _ in pairs])
            rot_x  = np.array([t["center"][0] for _, t in pairs])
            rot_y  = np.array([t["center"][1] for _, t in pairs])
            dx = rot_x - orig_x
            dy = rot_y - orig_y

            ax.scatter(orig_x, orig_y, c="red",       s=25, alpha=0.8,
                       edgecolors="darkred",   linewidths=0.4, zorder=3,
                       label=f"Original ({len(pairs)})")
            ax.scatter(rot_x,  rot_y,  c="limegreen", s=25, alpha=0.8,
                       edgecolors="darkgreen", linewidths=0.4, zorder=3,
                       label=f"Rotated ({len(pairs)})")
            for px, py, ddx, ddy in zip(orig_x, orig_y, dx, dy):
                ax.annotate(
                    "", xy=(px + ddx, py + ddy), xytext=(px, py),
                    arrowprops=dict(arrowstyle="-|>", color="steelblue",
                                   lw=0.8, mutation_scale=8),
                )
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("x (mm)", fontsize=10)
            ax.set_ylabel("y (mm)", fontsize=10)

        ax.set_title(layer_titles[layer], fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"2D arrow plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rotate a greedy PMT voxel selection by a fraction of 2π.\n\n"
            "Single-angle mode (--angle provided): run immediately and save "
            "outputs to --output-dir.\n"
            "Explore mode (--angle omitted): scan all valid angles, display "
            "them, prompt for selection, then run each chosen angle in its "
            "own subfolder inside --output-dir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all-voxels", required=True, metavar="PATH",
        help="JSON file with all available voxel positions "
             "(bare list or 'selected_voxels' wrapper format).",
    )
    parser.add_argument(
        "--selected", required=True, metavar="PATH",
        help="JSON greedy output (or bare voxel list) to rotate.",
    )
    parser.add_argument(
        "--angle", default=None, type=float, metavar="FRAC",
        help="Rotation as a fraction of 2π (e.g. 0.5 → 180°). "
             "Omit to enter explore mode and discover all valid angles.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=3600, metavar="N",
        help="Angular sampling resolution for explore mode: number of steps "
             "from 0 to 2π (default 3600 = every 0.1°). Increase for finer "
             "resolution; the default is sufficient for all practical voxel "
             "grid spacings on this detector.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save per-layer pairwise-distance histograms.",
    )
    parser.add_argument(
        "--skip-validity", action="store_true",
        help="Skip the PMT placement validity check on rotated voxels "
             "(by default the check is run and invalid targets raise an error).",
    )
    parser.add_argument(
        "--output-dir", required=True, metavar="PATH",
        help="Output directory. In single-angle mode files go directly here; "
             "in explore mode a subfolder phi_{frac:.4f}x2pi/ is created per "
             "selected angle.",
    )
    args = parser.parse_args(argv)

    # Load inputs
    print(f"Loading all voxels from  {args.all_voxels} ...")
    all_voxels = load_voxel_file(args.all_voxels)
    print(f"  {len(all_voxels)} voxels in pool")

    print(f"Loading selected voxels from  {args.selected} ...")
    selected = load_voxel_file(args.selected)
    print(f"  {len(selected)} voxels selected")

    output_dir = Path(args.output_dir)

    # -----------------------------------------------------------------------
    # Single-angle mode
    # -----------------------------------------------------------------------
    if args.angle is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_for_angle(
            selected, all_voxels,
            phi_frac=args.angle,
            output_dir=output_dir,
            selected_path=args.selected,
            skip_validity=args.skip_validity,
            plot=args.plot,
        )
        return

    # -----------------------------------------------------------------------
    # Explore mode: find all valid angles, let user pick
    # -----------------------------------------------------------------------
    print(f"\nExplore mode: scanning {args.n_samples} angles ...")
    valid_fracs = find_valid_angles(
        selected, all_voxels,
        n_samples=args.n_samples,
        skip_validity=args.skip_validity,
    )

    if not valid_fracs:
        print("No valid rotation angles found.")
        return

    chosen_fracs = prompt_angle_selection(valid_fracs)

    for phi_frac in chosen_fracs:
        subfolder = output_dir / f"phi_{phi_frac:.4f}x2pi"
        run_for_angle(
            selected, all_voxels,
            phi_frac=phi_frac,
            output_dir=subfolder,
            selected_path=args.selected,
            skip_validity=args.skip_validity,
            plot=args.plot,
        )

    print(f"\nAll done. Results in: {output_dir}")


if __name__ == "__main__":
    main()
