#!/usr/bin/env python3
"""
Rotate a greedy PMT voxel selection by an azimuthal angle phi.

For each selected voxel:
  1. Rotate its (x, y) coordinates by phi (fraction of 2π, e.g. 0.5 → 180°).
  2. Find the nearest *valid* voxel on the same sub-surface from a pool of all
     available voxels (candidate pool is pre-filtered to valid PMT positions).
  3. For wall voxels the candidate must share the same z-level (within 1 mm).

After all voxels are mapped, a collision check is run:
  - Collision: no two originals may map to the same target.
  - Self-overlap (targets that coincide with original positions) is counted and
    reported but does NOT cause a discard.

A spacing-distribution test compares pairwise distances before/after.
A 3D plot of original (red) vs. rotated (green) positions is always produced.
A 2D per-layer arrow plot shows individual voxel shifts.

Usage — single angle:
    python rotate_voxels.py \\
        --all-voxels all_valid.json \\
        --selected greedy_N300.json \\
        --angle 0.25 \\          # fraction of 2π → 90°
        --output-dir plots/

Usage — explore all valid angles (omit --angle):
    python rotate_voxels.py \\
        --all-voxels all_valid.json \\
        --selected greedy_N300.json \\
        --output-dir plots/

    Explore mode (bot-split): scans bot voxels for collision-free angles, then
    applies the mean actual bot rotation to all other layers.  All valid
    configurations are saved automatically; no interactive angle selection.
    A comparison figure is created in --output-dir summarising all configs.
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


def build_candidate_pool(
    voxel: dict,
    all_voxels: list[dict],
    skip_validity: bool = False,
) -> list[dict]:
    """Return candidate voxels on the same layer (and same z for wall).

    When ``skip_validity=False`` (default) the pool is pre-filtered to
    positions that pass ``is_valid_pmt_position``, guaranteeing the
    nearest-candidate result is a valid PMT slot.

    When ``skip_validity=True`` all same-layer voxels are included regardless
    of validity; only collision is checked by the caller.

    Raises ValueError if no candidates exist.
    """
    layer = voxel["layer"]
    if skip_validity:
        pool = [v for v in all_voxels if v["layer"] == layer]
    else:
        pool = [
            v for v in all_voxels
            if v["layer"] == layer and is_valid_pmt_position(v["center"], v["layer"])
        ]
    if layer == "wall":
        z_orig = voxel["center"][2]
        pool = [v for v in pool if abs(v["center"][2] - z_orig) < WALL_Z_TOL]
    if not pool:
        label = "any" if skip_validity else "valid"
        raise ValueError(
            f"No {label} candidates for voxel '{voxel['index']}' "
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
    skip_validity: bool = False,
) -> list[tuple[dict, dict]]:
    """Map each selected voxel to its nearest rotated counterpart.

    Parameters
    ----------
    phi_frac:
        Rotation as a fraction of 2π (e.g. 0.5 → 180°).
    skip_validity:
        Passed through to ``build_candidate_pool``; when True, all same-layer
        voxels are candidates (not just valid PMT positions).

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
        pool = build_candidate_pool(voxel, all_voxels, skip_validity=skip_validity)
        target = find_nearest_candidate(rotated, pool)
        mapping.append((voxel, target))
    return mapping


def check_mapping(mapping: list[tuple[dict, dict]]) -> int:
    """Check for collisions; count and return self-overlaps.

    Raises RuntimeError if two original voxels map to the same target
    (collision).  Self-overlap (a target that coincides with an original
    position) is no longer a hard error — the count is returned so the
    caller can report or filter on it.

    Returns
    -------
    self_overlap_count : int
        Number of target indices that appear in the original selection.
    """
    original_indices: set[str] = {orig["index"] for orig, _ in mapping}

    # Build target_index → [originating voxel ids]
    target_map: dict[str, list[str]] = {}
    for orig, tgt in mapping:
        target_map.setdefault(tgt["index"], []).append(orig["index"])

    # Collision check (hard error)
    collisions = {t: origs for t, origs in target_map.items() if len(origs) > 1}
    if collisions:
        msgs = [f"  target '{t}' ← {origs}" for t, origs in collisions.items()]
        raise RuntimeError(
            "Collision: multiple voxels mapped to the same target:\n"
            + "\n".join(msgs)
        )

    # Self-overlap count (soft — returned, not raised)
    overlap = original_indices & set(target_map.keys())
    return len(overlap)


def check_validity(mapping: list[tuple[dict, dict]]) -> None:
    """Check that every target voxel is a geometrically valid PMT position.

    With the pre-filtered candidate pool this should always pass; kept as
    a safety net.  Raises RuntimeError listing all invalid targets.
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
    """Scan all rotation angles and return those producing rigid-grid, collision-free mappings.

    Self-overlap is not a discard criterion.

    When ``skip_validity=False`` (default) the candidate pool for each source
    voxel is pre-filtered to valid PMT positions via ``build_candidate_pool``.
    When ``skip_validity=True`` all same-layer voxels are candidates.

    A snap-error filter enforces that every rotated voxel lands within 50 % of
    the minimum inter-voxel spacing in its candidate pool.  This ensures only
    "rigid grid rotations" — where all voxels jump simultaneously to valid grid
    positions — are accepted, rather than partially-shifted configurations.

    The snap error is derived for free from the scoring coefficients:
        dist²(θ, c) = −2 × score(θ, c)
    so no extra computation is required beyond what is already done for
    nearest-neighbour assignment.

    Parameters
    ----------
    selected : list of voxel dicts (the selection to rotate).
    all_voxels : the full candidate pool.
    n_samples : number of angle steps (default 3600 → every 0.1°).
    skip_validity : whether to include invalid voxels in the candidate pool.

    Returns
    -------
    Sorted list of phi_frac values (one representative per distinct valid
    mapping configuration).
    """
    V = len(selected)

    # Build per-source-voxel candidate pools and scoring coefficients
    pools: list[list[dict]] = []
    A_list: list[np.ndarray] = []
    B_list: list[np.ndarray] = []
    K_list: list[np.ndarray] = []

    for voxel in selected:
        pool = build_candidate_pool(voxel, all_voxels, skip_validity=skip_validity)
        A, B, K = _precompute_scores(voxel, pool)
        pools.append(pool)
        A_list.append(A)
        B_list.append(B)
        K_list.append(K)

    # Per-pool snap-error threshold: (0.5 × min pairwise distance)²
    # dist²(θ, c) = −2 × score(θ, c), so the threshold on score is −thresh_sq / 2.
    snap_thresh_sq = np.empty(V)
    for i, pool in enumerate(pools):
        centers = np.array([v["center"] for v in pool])   # (C, 3)
        n_c = len(centers)
        if n_c < 2:
            snap_thresh_sq[i] = np.inf
        else:
            diffs_sq = np.sum(
                (centers[:, None, :] - centers[None, :, :]) ** 2, axis=2
            )                                                          # (C, C)
            np.fill_diagonal(diffs_sq, np.inf)
            snap_thresh_sq[i] = 0.25 * float(diffs_sq.min())         # (0.5 × d_min)²

    # Sample angles
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    cos_t = np.cos(theta)   # (n_samples,)
    sin_t = np.sin(theta)   # (n_samples,)

    # mapping_matrix[i, j] = local index into pool[i] of nearest candidate
    # max_scores[i, j]      = score of that nearest candidate (≤ 0 always)
    mapping_matrix = np.empty((V, n_samples), dtype=np.int32)
    max_scores     = np.empty((V, n_samples), dtype=np.float64)

    print(f"  Scanning {n_samples} angles for {V} voxels ...")
    arange_j = np.arange(n_samples)
    for i, (A, B, K) in enumerate(zip(A_list, B_list, K_list)):
        # scores shape: (C, n_samples);  dist²(θ,c) = −2 × score(θ,c)
        scores = (A[:, None] * cos_t[None, :]
                  + B[:, None] * sin_t[None, :]
                  - K[:, None] / 2.0)
        mapping_matrix[i] = np.argmax(scores, axis=0)
        max_scores[i]     = scores[mapping_matrix[i], arange_j]

    # Snap-error filter (vectorised): require dist²_i(j) < snap_thresh_sq[i] for all i
    # dist²_i(j) = −2 × max_scores[i, j]
    snap_err_sq = -2.0 * max_scores                             # (V, n_samples)
    snap_ok     = np.all(snap_err_sq < snap_thresh_sq[:, None], axis=0)  # (n_samples,)
    n_snap_pass = int(snap_ok.sum())
    print(f"  Snap-error filter (≤50 %% of min pool spacing): "
          f"{n_snap_pass}/{n_samples} angles pass ...")

    # Collision check on snap-passing angles only
    print("  Checking collision ...")

    valid_mask_angles = snap_ok.copy()

    original_indices_list = [v["index"] for v in selected]

    for j in np.where(snap_ok)[0]:
        targets_local = mapping_matrix[:, j]   # shape (V,)

        target_keys = [pools[i][targets_local[i]]["index"] for i in range(V)]

        # Collision: all target indices must be distinct
        if len(set(target_keys)) < V:
            valid_mask_angles[j] = False
            continue

        # Identity: exclude if every voxel maps to itself
        if target_keys == original_indices_list:
            valid_mask_angles[j] = False

    # Find distinct valid mapping configurations
    valid_indices = np.where(valid_mask_angles)[0]
    if len(valid_indices) == 0:
        return []

    # Group by unique full mapping tuple — one representative phi_frac per config
    seen_configs: dict[tuple, float] = {}
    for j in valid_indices:
        config_key = tuple(
            pools[i][mapping_matrix[i, j]]["index"] for i in range(V)
        )
        if config_key not in seen_configs:
            seen_configs[config_key] = j / n_samples   # phi_frac representative

    valid_fracs = sorted(seen_configs.values())
    return valid_fracs


# ---------------------------------------------------------------------------
# New helpers: mean actual rotation, per-config metrics, comparison plots
# ---------------------------------------------------------------------------

def compute_mean_actual_phi(mapping: list[tuple[dict, dict]]) -> float:
    """Circular mean of per-voxel azimuthal displacements (radians).

    For each (original, target) pair, computes
        Δφ = atan2(y_target, x_target) − atan2(y_orig, x_orig)
    and returns the circular mean angle.
    """
    deltas = np.array([
        math.atan2(tgt["center"][1], tgt["center"][0])
        - math.atan2(orig["center"][1], orig["center"][0])
        for orig, tgt in mapping
    ])
    return float(np.angle(np.mean(np.exp(1j * deltas))))


def compute_config_metrics(mapping: list[tuple[dict, dict]]) -> dict:
    """Compute quality metrics for one rotation configuration.

    Metrics
    -------
    Global:
      self_overlap_total        — targets that coincide with original positions
      mean_delta_phi_deg        — mean |Δφ| per voxel (absolute angular shift)
      std_delta_phi_deg         — std of |Δφ| (uniformity of rotation)
      mean_radial_disp_mm       — mean 3-D Euclidean displacement per voxel

    Per layer (in ``per_layer[layer]``):
      count, self_overlap_count,
      mean_delta_phi_deg, std_delta_phi_deg, mean_radial_disp_mm,
      delta_pw_mean_rel         — relative change in mean pairwise distance
      delta_nn_mean_rel         — relative change in mean nearest-neighbour dist
    """
    original_indices = {orig["index"] for orig, _ in mapping}
    layers = sorted({orig["layer"] for orig, _ in mapping})

    # Global per-voxel quantities
    delta_phi_abs: list[float] = []
    radial_disp:   list[float] = []
    for orig, tgt in mapping:
        phi_o = math.atan2(orig["center"][1], orig["center"][0])
        phi_t = math.atan2(tgt["center"][1],  tgt["center"][0])
        d = (phi_t - phi_o + math.pi) % (2.0 * math.pi) - math.pi
        delta_phi_abs.append(abs(d))
        radial_disp.append(float(np.linalg.norm(
            np.array(tgt["center"]) - np.array(orig["center"])
        )))

    per_layer: dict = {}
    for layer in layers:
        pairs = [(o, t) for o, t in mapping if o["layer"] == layer]
        tgt_ids = [t["index"] for _, t in pairs]
        overlap = sum(1 for ti in tgt_ids if ti in original_indices)

        b_c = np.array([o["center"] for o, _ in pairs])
        a_c = np.array([t["center"] for _, t in pairs])
        b_stats = compute_spacing_stats(b_c) if len(b_c) >= 2 else None
        a_stats = compute_spacing_stats(a_c) if len(a_c) >= 2 else None

        ld: list[float] = []
        lr: list[float] = []
        for orig, tgt in pairs:
            phi_o = math.atan2(orig["center"][1], orig["center"][0])
            phi_t = math.atan2(tgt["center"][1],  tgt["center"][0])
            d = (phi_t - phi_o + math.pi) % (2.0 * math.pi) - math.pi
            ld.append(abs(d))
            lr.append(float(np.linalg.norm(
                np.array(tgt["center"]) - np.array(orig["center"])
            )))

        entry: dict = {
            "count":               len(pairs),
            "self_overlap_count":  overlap,
            "mean_delta_phi_deg":  float(np.degrees(np.mean(ld))) if ld else None,
            "std_delta_phi_deg":   float(np.degrees(np.std(ld)))  if ld else None,
            "mean_radial_disp_mm": float(np.mean(lr))             if lr else None,
        }
        if b_stats and a_stats:
            entry["delta_pw_mean_rel"] = (
                (a_stats["pw_mean"] - b_stats["pw_mean"]) / b_stats["pw_mean"]
                if b_stats["pw_mean"] != 0.0 else 0.0
            )
            entry["delta_nn_mean_rel"] = (
                (a_stats["nn_mean"] - b_stats["nn_mean"]) / b_stats["nn_mean"]
                if b_stats["nn_mean"] != 0.0 else 0.0
            )
        hom_b = compute_nn_homogeneity(b_c, layer)
        hom_a = compute_nn_homogeneity(a_c, layer)
        if hom_b is not None and hom_a is not None:
            entry["cv_before"]     = hom_b["cv"]
            entry["cv_after"]      = hom_a["cv"]
            entry["delta_cv_abs"]  = hom_a["cv"] - hom_b["cv"]
            entry["delta_cv_rel"]  = (
                (hom_a["cv"] - hom_b["cv"]) / hom_b["cv"]
                if hom_b["cv"] != 0.0 else 0.0
            )
        per_layer[layer] = entry

    self_overlap_total = sum(
        1 for _, tgt in mapping if tgt["index"] in original_indices
    )

    return {
        "self_overlap_total":     self_overlap_total,
        "self_overlap_per_layer": {l: per_layer[l]["self_overlap_count"] for l in layers},
        "mean_delta_phi_deg":     float(np.degrees(np.mean(delta_phi_abs))),
        "std_delta_phi_deg":      float(np.degrees(np.std(delta_phi_abs))),
        "mean_radial_disp_mm":    float(np.mean(radial_disp)),
        "per_layer":              per_layer,
    }


def plot_comparison_metrics(
    configs: list[dict],   # each: {"phi_frac", "phi_deg", "metrics"}
    output_dir: Path,
) -> None:
    """Multi-panel figure comparing all valid configs saved in output_dir.

    Panels
    ------
    1. Self-overlap count per layer (grouped bar chart)
    2. Mean angular displacement ± std per config
    3. Mean radial displacement per config
    4. Relative change in mean pairwise distance, per layer
    5. Relative change in mean nearest-neighbour distance, per layer
    6. Summary table (angle, self-overlaps, mean Δφ, mean displacement)
    """
    if not configs:
        return

    n = len(configs)
    angles_deg = [c["phi_deg"] for c in configs]
    x = np.arange(n)
    layers = sorted({l for c in configs for l in c["metrics"]["per_layer"]})
    width_unit = 0.8 / max(len(layers), 1)
    layer_colors = {"pit": "steelblue", "bot": "coral", "top": "mediumseagreen", "wall": "mediumpurple"}

    fig, axes = plt.subplots(3, 2, figsize=(max(16, n * 0.8 + 4), 16))
    fig.suptitle(
        f"Rotation configs comparison  ({n} valid angle(s))",
        fontsize=14, fontweight="bold",
    )

    xlabels = [f"{a:.1f}°" for a in angles_deg]

    # --- Panel 1: self-overlap per layer ---
    ax = axes[0, 0]
    for i, layer in enumerate(layers):
        counts = [c["metrics"]["self_overlap_per_layer"].get(layer, 0) for c in configs]
        bars = ax.bar(
            x + i * width_unit, counts, width_unit,
            label=layer, color=layer_colors.get(layer, f"C{i}"), alpha=0.8,
        )
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    str(cnt), ha="center", va="bottom", fontsize=7,
                )
    ax.set_xticks(x + width_unit * (len(layers) - 1) / 2)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Self-overlap count per layer", fontsize=11)
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 2: mean angular displacement ± std ---
    ax = axes[0, 1]
    vals = [c["metrics"]["mean_delta_phi_deg"] for c in configs]
    errs = [c["metrics"]["std_delta_phi_deg"]  for c in configs]
    bars = ax.bar(x, vals, yerr=errs, capsize=4, color="steelblue", alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errs) * 0.05,
            f"{v:.1f}", ha="center", va="bottom", fontsize=7,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Mean angular displacement ± std  (all voxels)", fontsize=11)
    ax.set_ylabel("Δφ (deg)")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: mean radial displacement ---
    ax = axes[1, 0]
    vals = [c["metrics"]["mean_radial_disp_mm"] for c in configs]
    bars = ax.bar(x, vals, color="coral", alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
            f"{v:.0f}", ha="center", va="bottom", fontsize=7,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Mean 3-D radial displacement  (all voxels)", fontsize=11)
    ax.set_ylabel("Displacement (mm)")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 4: Δ pairwise mean distance per layer ---
    ax = axes[1, 1]
    for i, layer in enumerate(layers):
        vals_pct = []
        xi_list  = []
        for xi, c in zip(x, configs):
            v = c["metrics"]["per_layer"].get(layer, {}).get("delta_pw_mean_rel")
            if v is not None:
                vals_pct.append(v * 100.0)
                xi_list.append(xi)
        if xi_list:
            ax.bar(
                np.array(xi_list) + i * width_unit, vals_pct, width_unit,
                label=layer, color=layer_colors.get(layer, f"C{i}"), alpha=0.8,
            )
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x + width_unit * (len(layers) - 1) / 2)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Δ mean pairwise distance (relative, per layer)", fontsize=11)
    ax.set_ylabel("Δ (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 5: Δ nearest-neighbour mean per layer ---
    ax = axes[2, 0]
    for i, layer in enumerate(layers):
        vals_pct = []
        xi_list  = []
        for xi, c in zip(x, configs):
            v = c["metrics"]["per_layer"].get(layer, {}).get("delta_nn_mean_rel")
            if v is not None:
                vals_pct.append(v * 100.0)
                xi_list.append(xi)
        if xi_list:
            ax.bar(
                np.array(xi_list) + i * width_unit, vals_pct, width_unit,
                label=layer, color=layer_colors.get(layer, f"C{i}"), alpha=0.8,
            )
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x + width_unit * (len(layers) - 1) / 2)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Δ mean nearest-neighbour distance (relative, per layer)", fontsize=11)
    ax.set_ylabel("Δ (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 6: summary table ---
    ax = axes[2, 1]
    ax.axis("off")
    col_labels = ["Angle (°)", "Self-\noverlaps", "Mean Δφ\n(deg)", "Std Δφ\n(deg)", "Mean disp\n(mm)"]
    rows = []
    for c in configs:
        m = c["metrics"]
        rows.append([
            f"{c['phi_deg']:.2f}",
            str(m["self_overlap_total"]),
            f"{m['mean_delta_phi_deg']:.2f}",
            f"{m['std_delta_phi_deg']:.2f}",
            f"{m['mean_radial_disp_mm']:.1f}",
        ])
    table = ax.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    ax.set_title("Summary table", fontsize=11)

    plt.tight_layout()
    out = output_dir / "rotation_configs_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {out}")


def plot_homogeneity_comparison(
    configs: list[dict],   # each: {"phi_frac", "phi_deg", "metrics"}
    output_dir: Path,
) -> None:
    """Figure comparing NN-homogeneity (CV) across all valid rotation configs.

    Panels
    ------
    1. Absolute CV per layer per config — original CV shown as a dashed
       reference line per layer.
    2. ΔCV (absolute change, after − before) per layer per config.
    """
    if not configs:
        return

    n = len(configs)
    angles_deg = [c["phi_deg"] for c in configs]
    x = np.arange(n)
    layers = sorted({l for c in configs for l in c["metrics"]["per_layer"]})
    width_unit = 0.8 / max(len(layers), 1)
    layer_colors = {
        "pit": "steelblue", "bot": "coral",
        "top": "mediumseagreen", "wall": "mediumpurple",
    }
    xlabels = [f"{a:.1f}°" for a in angles_deg]

    fig, axes = plt.subplots(1, 2, figsize=(max(14, n * 0.8 + 4), 6))
    fig.suptitle(
        f"NN-homogeneity (CV) comparison  ({n} valid angle(s))\n"
        "CV = std / mean of nearest-neighbour distances  (lower = more uniform)",
        fontsize=13, fontweight="bold",
    )

    # --- Panel 1: absolute CV, with before-rotation reference lines ---
    ax = axes[0]
    for i, layer in enumerate(layers):
        cv_after_vals = []
        xi_list = []
        for xi, c in zip(x, configs):
            v = c["metrics"]["per_layer"].get(layer, {}).get("cv_after")
            if v is not None:
                cv_after_vals.append(v)
                xi_list.append(xi)
        if xi_list:
            ax.bar(
                np.array(xi_list) + i * width_unit, cv_after_vals, width_unit,
                label=layer, color=layer_colors.get(layer, f"C{i}"), alpha=0.8,
            )
        # Reference line: cv_before (same for all configs since original doesn't change)
        cv_before = next(
            (c["metrics"]["per_layer"].get(layer, {}).get("cv_before")
             for c in configs
             if c["metrics"]["per_layer"].get(layer, {}).get("cv_before") is not None),
            None,
        )
        if cv_before is not None and xi_list:
            center_offset = i * width_unit + width_unit / 2.0
            x_min = min(xi_list) + center_offset - width_unit
            x_max = max(xi_list) + center_offset + width_unit
            ax.hlines(
                cv_before, x_min, x_max,
                colors=layer_colors.get(layer, f"C{i}"),
                linestyles="--", linewidths=1.5, alpha=0.9,
            )

    ax.set_xticks(x + width_unit * (len(layers) - 1) / 2)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_title("CV after rotation (bars) vs. before (dashed lines)", fontsize=11)
    ax.set_ylabel("CV  (std / mean NN dist)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 2: ΔCV per layer ---
    ax = axes[1]
    for i, layer in enumerate(layers):
        delta_vals = []
        xi_list = []
        for xi, c in zip(x, configs):
            v = c["metrics"]["per_layer"].get(layer, {}).get("delta_cv_abs")
            if v is not None:
                delta_vals.append(v)
                xi_list.append(xi)
        if xi_list:
            ax.bar(
                np.array(xi_list) + i * width_unit, delta_vals, width_unit,
                label=layer, color=layer_colors.get(layer, f"C{i}"), alpha=0.8,
            )
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x + width_unit * (len(layers) - 1) / 2)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_title("ΔCV  (after − before rotation, per layer)", fontsize=11)
    ax.set_ylabel("ΔCV")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = output_dir / "rotation_homogeneity_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Homogeneity comparison plot saved to {out}")


# ---------------------------------------------------------------------------
# Single-angle execution
# ---------------------------------------------------------------------------

def run_for_angle(
    selected: list[dict],
    all_voxels: list[dict],
    phi_frac: float,
    output_dir: Path,
    selected_path: str,
    skip_validity: bool,
    plot: bool,
) -> dict:
    """Run the full rotation pipeline for one angle and save outputs.

    Returns a metrics dict suitable for ``plot_comparison_metrics``.
    """
    phi_deg = phi_frac * 360.0
    print(f"\n{'=' * 62}")
    print(f"Running angle: {phi_frac:.4f} × 2π = {phi_deg:.1f}°")
    print(f"{'=' * 62}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = derive_output_stem(selected_path, phi_frac)
    output_path = output_dir / f"{stem}.json"

    # Compute mapping
    if skip_validity:
        print("Computing rotation mapping (ALL voxels as candidates — validity skipped) ...")
    else:
        print("Computing rotation mapping (valid voxels only) ...")
    mapping = compute_voxel_mapping(selected, all_voxels, phi_frac,
                                    skip_validity=skip_validity)

    # Validation
    print("Validating mapping (collision check) ...")
    self_overlap_count = check_mapping(mapping)
    print(f"  OK — no collisions  (self-overlaps: {self_overlap_count})")

    if skip_validity:
        print("WARNING: validity checks skipped — rotated voxels may not be"
              " valid PMT positions.")

    rotated_voxels = assemble_output_voxels(mapping)

    # Spacing distribution test
    print("\nRunning spacing distribution test ...")
    spacing_results = run_spacing_test(
        selected, rotated_voxels,
        plot=plot,
        output_dir=output_dir if plot else None,
    )
    print_spacing_report(spacing_results, phi_frac, mapping)

    # Plots
    plot_3d_path = output_path.with_name(output_path.stem + "_3d.png")
    plot_rotated_voxels(selected, rotated_voxels, plot_3d_path, phi_frac)

    plot_2d_path = output_path.with_name(output_path.stem + "_2d_arrows.png")
    plot_2d_arrows(mapping, plot_2d_path, phi_frac)

    # Metrics
    metrics = compute_config_metrics(mapping)

    # Summary JSON
    summary: dict = {
        "phi_frac":           phi_frac,
        "phi_deg":            phi_deg,
        "validity_skipped":   skip_validity,
        "self_overlap_total": self_overlap_count,
        "metrics":            metrics,
    }
    if skip_validity:
        summary["WARNING"] = (
            "Validity checks were skipped (--skip-validity). "
            "Rotated voxels may not be valid PMT positions."
        )
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_dir / 'summary.json'}")

    # Voxels JSON
    config_block: dict = {
        "rotation_angle_2pi":     phi_frac,
        "source_file":            str(selected_path),
        "all_voxels_file":        "",
        "n_voxels":               len(rotated_voxels),
        "validity_skipped":       skip_validity,
        "self_overlap_total":     self_overlap_count,
        "self_overlap_per_layer": metrics["self_overlap_per_layer"],
    }
    if skip_validity:
        config_block["WARNING"] = (
            "Validity checks were skipped (--skip-validity). "
            "Rotated voxels may not be valid PMT positions."
        )
    out_data = {
        "config":           config_block,
        "selected_voxels":  rotated_voxels,
    }
    with open(output_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Output written to {output_path}")

    return {"phi_frac": phi_frac, "phi_deg": phi_deg, "metrics": metrics}


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


def compute_nn_homogeneity(centers: np.ndarray, layer: str) -> Optional[dict]:
    """Coefficient of variation (std/mean) of nearest-neighbour distances.

    Wall uses geodesic distance sqrt((R_ZYLINDER × Δφ)² + Δz²);
    all other layers use 3-D Euclidean distance.

    Returns None if fewer than 2 voxels are provided.
    """
    n = len(centers)
    if n < 2:
        return None

    if layer == "wall":
        phi = np.arctan2(centers[:, 1], centers[:, 0])
        z = centers[:, 2]
        dphi = phi[:, None] - phi[None, :]
        dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
        arc = R_ZYLINDER * np.abs(dphi)
        dz = z[:, None] - z[None, :]
        dists = np.sqrt(arc ** 2 + dz ** 2)
    else:
        diffs = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))

    np.fill_diagonal(dists, np.inf)
    nn_dists = dists.min(axis=1)

    mean = float(nn_dists.mean())
    std  = float(nn_dists.std())
    cv   = std / mean if mean > 0.0 else 0.0

    return {
        "cv":      cv,
        "nn_mean": mean,
        "nn_std":  std,
        "nn_min":  float(nn_dists.min()),
        "nn_max":  float(nn_dists.max()),
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
# Bot-split explore mode: rotate bot with full checks, apply mean angle
# to all other layers, then check combined collision
# ---------------------------------------------------------------------------

def run_bot_split_angle(
    bot_selected: list[dict],
    non_bot_selected: list[dict],
    all_voxels: list[dict],
    phi_frac_nominal: float,
    output_dir: Path,
    selected_path: str,
    skip_validity: bool,
    plot: bool,
) -> Optional[dict]:
    """Run the bot-split rotation pipeline for one nominal bot angle.

    Steps
    -----
    1. Rotate bot voxels using ``phi_frac_nominal``; compute their mapping.
    2. Compute the circular mean of the actual per-voxel angular displacements
       (the voxels may snap to grid neighbours, so the effective rotation
       differs slightly from the nominal angle).
    3. Apply that mean actual angle to all non-bot voxels.
    4. Check collision across the combined (bot + non-bot) mapping.
    5. Count self-overlaps per layer (not a discard criterion).
    6. Save all outputs to ``output_dir``.

    Returns a metrics dict on success, or None if the combined mapping has a
    collision.
    """
    phi_deg_nominal = phi_frac_nominal * 360.0
    print(f"\n{'=' * 62}")
    print(f"Processing angle: {phi_frac_nominal:.4f} × 2π = {phi_deg_nominal:.1f}°")
    print(f"{'=' * 62}")

    if skip_validity:
        print("  WARNING: validity checks skipped — all voxels used as candidates.")

    # Step 1: bot mapping
    bot_mapping = compute_voxel_mapping(bot_selected, all_voxels, phi_frac_nominal,
                                        skip_validity=skip_validity)

    # Step 2: mean actual rotation derived from bot voxel displacements
    mean_phi_rad  = compute_mean_actual_phi(bot_mapping)
    mean_phi_frac = mean_phi_rad / (2.0 * math.pi) % 1.0
    mean_phi_deg  = mean_phi_rad * 180.0 / math.pi
    print(f"  Bot mean actual rotation: {mean_phi_frac:.4f} × 2π = {mean_phi_deg:.2f}°")

    # Step 3: non-bot mapping using the mean actual bot angle
    non_bot_mapping = compute_voxel_mapping(non_bot_selected, all_voxels, mean_phi_frac,
                                            skip_validity=skip_validity)

    # Step 4: combined collision check
    combined_mapping = bot_mapping + non_bot_mapping
    try:
        self_overlap_count = check_mapping(combined_mapping)
    except RuntimeError as exc:
        print(f"  Discarding — collision in combined mapping:\n  {exc}")
        return None

    print(f"  No collisions.  Total self-overlaps: {self_overlap_count}")

    # Step 5: assemble outputs
    all_selected   = bot_selected + non_bot_selected
    rotated_voxels = assemble_output_voxels(combined_mapping)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem        = derive_output_stem(selected_path, phi_frac_nominal)
    output_path = output_dir / f"{stem}.json"

    # Spacing distribution test
    print("\nRunning spacing distribution test ...")
    spacing_results = run_spacing_test(
        all_selected, rotated_voxels,
        plot=plot,
        output_dir=output_dir if plot else None,
    )
    print_spacing_report(spacing_results, phi_frac_nominal, combined_mapping)

    # 3D plot
    plot_3d_path = output_path.with_name(output_path.stem + "_3d.png")
    plot_rotated_voxels(all_selected, rotated_voxels, plot_3d_path, phi_frac_nominal)

    # 2D per-layer arrow plot
    plot_2d_path = output_path.with_name(output_path.stem + "_2d_arrows.png")
    plot_2d_arrows(combined_mapping, plot_2d_path, phi_frac_nominal)

    # Config metrics
    metrics = compute_config_metrics(combined_mapping)

    # Summary JSON (per-subfolder)
    summary: dict = {
        "phi_frac_nominal":     phi_frac_nominal,
        "phi_deg_nominal":      phi_deg_nominal,
        "mean_actual_phi_frac": mean_phi_frac,
        "mean_actual_phi_deg":  mean_phi_deg,
        "validity_skipped":     skip_validity,
        "self_overlap_total":   self_overlap_count,
        "metrics":              metrics,
    }
    if skip_validity:
        summary["WARNING"] = (
            "Validity checks were skipped (--skip-validity). "
            "Rotated voxels may not be valid PMT positions."
        )
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_dir / 'summary.json'}")

    # Voxels JSON (readable by downstream scripts)
    config_block: dict = {
        "rotation_angle_2pi":     phi_frac_nominal,
        "mean_actual_phi_frac":   mean_phi_frac,
        "source_file":            str(selected_path),
        "n_voxels":               len(rotated_voxels),
        "validity_skipped":       skip_validity,
        "self_overlap_total":     self_overlap_count,
        "self_overlap_per_layer": metrics["self_overlap_per_layer"],
    }
    if skip_validity:
        config_block["WARNING"] = (
            "Validity checks were skipped (--skip-validity). "
            "Rotated voxels may not be valid PMT positions."
        )
    out_data = {
        "config":          config_block,
        "selected_voxels": rotated_voxels,
    }
    with open(output_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Output written to {output_path}")

    return {"phi_frac": phi_frac_nominal, "phi_deg": phi_deg_nominal, "metrics": metrics}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rotate a greedy PMT voxel selection by a fraction of 2π.\n\n"
            "Single-angle mode (--angle provided): run immediately and save "
            "outputs to --output-dir.\n"
            "Explore mode (--angle omitted): scan all collision-free angles "
            "automatically (bot-split: bot voxels drive the scan; mean actual "
            "bot rotation is applied to all other layers), save every valid "
            "config in its own subfolder, and write a comparison figure to "
            "--output-dir."
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
             "from 0 to 2π (default 3600 = every 0.1°).",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save per-layer pairwise-distance histograms.",
    )
    parser.add_argument(
        "--skip-validity", action="store_true",
        help="Skip all PMT placement validity filtering. When set, rotated "
             "voxels are assigned to the nearest voxel from the full pool "
             "(including invalid positions); only the collision check is "
             "applied. A WARNING is written to all summary and voxel JSON "
             "files. Use this to explore configurations without geometric "
             "constraints.",
    )
    parser.add_argument(
        "--output-dir", required=True, metavar="PATH",
        help="Output directory. In single-angle mode files go directly here; "
             "in explore mode a subfolder phi_{frac:.4f}x2pi/ is created per "
             "angle, and a comparison figure is written here.",
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
    # Explore mode
    # -----------------------------------------------------------------------
    bot_selected     = [v for v in selected if v["layer"] == "bot"]
    non_bot_selected = [v for v in selected if v["layer"] != "bot"]

    # --- Fallback: no bot voxels — scan full selection ---
    if not bot_selected:
        print(f"\nExplore mode (no bot voxels): scanning {args.n_samples} angles ...")
        valid_fracs = find_valid_angles(
            selected, all_voxels,
            n_samples=args.n_samples,
            skip_validity=args.skip_validity,
        )
        if not valid_fracs:
            print("No collision-free rotation angles found.")
            return
        print(f"\nFound {len(valid_fracs)} valid angle(s). Running all ...")
        saved_configs: list[dict] = []
        for phi_frac in valid_fracs:
            subfolder = output_dir / f"phi_{phi_frac:.6f}x2pi"
            result = run_for_angle(
                selected, all_voxels,
                phi_frac=phi_frac,
                output_dir=subfolder,
                selected_path=args.selected,
                skip_validity=args.skip_validity,
                plot=args.plot,
            )
            saved_configs.append(result)
        if saved_configs:
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_comparison_metrics(saved_configs, output_dir)
            plot_homogeneity_comparison(saved_configs, output_dir)
        print(f"\nAll done. Results in: {output_dir}")
        return

    # --- Bot-split explore mode ---
    validity_note = "ALL voxels as candidates (validity skipped)" if args.skip_validity \
                    else "valid voxels only"
    print(f"\nBot-split explore mode: scanning {args.n_samples} angles "
          f"for {len(bot_selected)} bot voxels "
          f"(collision check only, {validity_note}) ...")
    valid_fracs = find_valid_angles(
        bot_selected, all_voxels,
        n_samples=args.n_samples,
        skip_validity=args.skip_validity,
    )

    if not valid_fracs:
        print("No collision-free bot rotation angles found.")
        return

    print(f"\nFound {len(valid_fracs)} valid bot angle configuration(s). Running all ...")

    saved_configs = []
    discarded_count = 0

    for phi_frac in valid_fracs:
        subfolder = output_dir / f"phi_{phi_frac:.6f}x2pi"
        result = run_bot_split_angle(
            bot_selected, non_bot_selected, all_voxels,
            phi_frac_nominal=phi_frac,
            output_dir=subfolder,
            selected_path=args.selected,
            skip_validity=args.skip_validity,
            plot=args.plot,
        )
        if result is not None:
            saved_configs.append(result)
        else:
            discarded_count += 1

    # Comparison plots in parent output directory
    if saved_configs:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison_metrics(saved_configs, output_dir)
        plot_homogeneity_comparison(saved_configs, output_dir)

    # Final summary
    print(f"\n{'=' * 62}")
    print(f"Bot-split explore mode complete")
    print(f"  Saved    : {len(saved_configs)} setup(s)")
    for c in saved_configs:
        print(f"    {c['phi_frac']:.6f} × 2π = {c['phi_deg']:.2f}°"
              f"  →  phi_{c['phi_frac']:.6f}x2pi/")
    print(f"  Discarded: {discarded_count} setup(s)  (combined collision)")
    print(f"{'=' * 62}")
    print(f"\nAll done. Results in: {output_dir}")


if __name__ == "__main__":
    main()
