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

Usage:
    python rotate_voxels.py \\
        --all-voxels all_valid.json \\
        --selected greedy_N300.json \\
        --angle 0.25 \\          # fraction of 2π → 90°
        --output-dir plots/ \\
        [--plot]
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
    """Base filename stem derived from the selected voxel file: ``{stem}_rotated_{phi_frac:.4f}x2pi``."""
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


def assemble_output_voxels(mapping: list[tuple[dict, dict]]) -> list[dict]:
    """Return the list of target voxels in original selection order."""
    return [tgt for _, tgt in mapping]


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

    # Full distance matrix (reused for both pairwise and NN)
    diffs = centers[:, None, :] - centers[None, :, :]   # (n, n, 3)
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))          # (n, n)

    # Pairwise (upper triangle)
    pw = dists[np.triu_indices(n, k=1)]

    # Per-voxel nearest-neighbour distance
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
    """Compare per-layer spacing distributions before and after rotation.

    Returns a dict keyed by layer name, each containing:
    ``before``, ``after`` (stats dicts), ``delta_rel`` (relative changes),
    ``passed`` (bool or None when skipped), ``count`` (number of voxels).
    """
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
    ax.hist(pw_b, bins=bins, alpha=0.6, color="red",    label="before", density=True)
    ax.hist(pw_a, bins=bins, alpha=0.6, color="limegreen", label="after",  density=True)
    ax.set_xlabel("Pairwise distance (mm)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
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
    """3D scatter of original (red) and rotated (green) voxel positions.

    Mirrors the visual style of pmtopt/plotting.py:plot_selected_voxels.
    """
    Z_BASE = Z_BASE_GLOBAL
    Z_TOP  = Z_BASE + H_ZYLINDER

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection="3d")

    # Cylinder wireframe
    theta = np.linspace(0, 2 * np.pi, 200)
    n_vert = 24
    theta_lines = np.linspace(0, 2 * np.pi, n_vert, endpoint=False)
    for z in [Z_BASE, Z_TOP]:
        ax.plot(R_ZYLINDER * np.cos(theta), R_ZYLINDER * np.sin(theta), z,
                color="gray", alpha=0.3, linewidth=0.5)
    for t in theta_lines:
        ax.plot([R_ZYLINDER * np.cos(t)] * 2, [R_ZYLINDER * np.sin(t)] * 2,
                [Z_BASE, Z_TOP], color="gray", alpha=0.3, linewidth=0.5)

    # Reference circles
    ax.plot(R_PIT     * np.cos(theta), R_PIT     * np.sin(theta), Z_BASE,
            color="steelblue", alpha=0.6, linewidth=1.0,
            label=f"Pit (r={R_PIT})")
    ax.plot(R_ZYL_BOT * np.cos(theta), R_ZYL_BOT * np.sin(theta), Z_BASE,
            color="seagreen",  alpha=0.6, linewidth=1.0,
            label=f"Bot inner (r={R_ZYL_BOT})")

    # Scatter: original (red) and rotated (green) per layer
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
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Rotate a greedy PMT voxel selection by a fraction of 2π."
    )
    parser.add_argument(
        "--all-voxels", required=True, metavar="PATH",
        help="JSON file with all valid voxel positions "
             "(same 'selected_voxels' list format as greedy output)",
    )
    parser.add_argument(
        "--selected", required=True, metavar="PATH",
        help="JSON greedy output (or bare voxel list) to rotate",
    )
    parser.add_argument(
        "--angle", required=True, type=float, metavar="FRAC",
        help="Rotation as a fraction of 2π (e.g. 0.5 → 180°, 0.25 → 90°)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save per-layer pairwise-distance histograms",
    )
    parser.add_argument(
        "--output-dir", required=True, metavar="PATH",
        help="Directory where the output JSON and all plots are saved",
    )
    args = parser.parse_args(argv)

    phi_frac: float = args.angle
    phi_deg = phi_frac * 360.0
    print(f"\nRotating voxels by {phi_frac:.4f} × 2π = {phi_deg:.1f}°")

    # Load inputs
    print(f"Loading all voxels from  {args.all_voxels} ...")
    all_voxels = load_voxel_file(args.all_voxels)
    print(f"  {len(all_voxels)} voxels in pool")

    print(f"Loading selected voxels from  {args.selected} ...")
    selected = load_voxel_file(args.selected)
    print(f"  {len(selected)} voxels selected")

    # Output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = derive_output_stem(args.selected, phi_frac)
    output_path = output_dir / f"{stem}.json"

    # Compute mapping (no per-voxel checks)
    print("\nComputing rotation mapping ...")
    mapping = compute_voxel_mapping(selected, all_voxels, phi_frac)

    # Batch validation
    print("Validating mapping (collision + self-overlap check) ...")
    check_mapping(mapping)
    print("  OK — no collisions or self-overlaps detected")

    rotated_voxels = assemble_output_voxels(mapping)

    # Spacing distribution test
    print("\nRunning spacing distribution test ...")
    spacing_results = run_spacing_test(
        selected, rotated_voxels,
        plot=args.plot,
        output_dir=output_dir if args.plot else None,
    )
    print_spacing_report(spacing_results, phi_frac, mapping)

    # 3D plot (always produced)
    plot_3d_path = output_path.with_name(output_path.stem + "_3d.png")
    plot_rotated_voxels(selected, rotated_voxels, plot_3d_path, phi_frac)

    # Write JSON output
    out_data = {
        "config": {
            "rotation_angle_2pi": phi_frac,
            "source_file": str(args.selected),
            "all_voxels_file": str(args.all_voxels),
            "n_voxels": len(rotated_voxels),
        },
        "selected_voxels": rotated_voxels,
    }
    with open(output_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
