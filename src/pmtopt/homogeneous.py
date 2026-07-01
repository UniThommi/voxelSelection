"""
homogeneous.py — CLI tool for voxel generation and homogeneous PMT selection
on the cylindrical SSD detector.

Usage (via unified CLI):
    python src/pmtopt/main.py homogeneous --mode generate [--output-dir ./output]
    python src/pmtopt/main.py homogeneous --mode select -N 300 [--output-dir ./output]
    python src/pmtopt/main.py homogeneous --mode select -N 100 --areas pit wall

Direct invocation also works:
    python src/pmtopt/homogeneous.py --mode select -N 300
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path
import ot

import numpy as np
import scipy.spatial.distance as _ssd
from matplotlib import patches
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Import geometry constants from pmtopt package
# ---------------------------------------------------------------------------
from pmtopt.geometry import (
    PMT_RADIUS,
    R_PIT,
    R_ZYL_BOT,
    R_ZYL_TOP,
    R_ZYLINDER,
    Z_OFFSET,
    Z_ORIGIN,
    is_valid_pmt_position,
)

# ---------------------------------------------------------------------------
# Voxel-specific local constants (not in geometry.py)
# ---------------------------------------------------------------------------
L_VOXEL: int = 195      # mm — grid cell side length (≈ sqrt(110² × π))
T_ZYLINDER: int = 1     # mm — radial/z thickness for bot/top/wall voxels
DZ_PIT: int = 1         # mm — z-thickness of pit voxels (historical sim value)
H_CYLINDER: int = 8900  # mm — full cylinder height for wall generation/area

# Z coordinate of the cylinder bottom (global frame)
Z_BASE: int = Z_ORIGIN + Z_OFFSET   # = 20 + (-5000) = -4980 mm

VALID_GEOMETRIES = ["currentDist"]
VALID_AREAS = ["pit", "bot", "top", "wall"]
AREA_COLORS = {"pit": "red", "bot": "blue", "top": "green", "wall": "orange"}


# ---------------------------------------------------------------------------
# Area surface formulas
# ---------------------------------------------------------------------------

def compute_area_surfaces(areas: list) -> dict:
    """Return surface area in mm² for each requested area."""
    all_surfaces = {
        "pit":  np.pi * R_PIT**2,
        "bot":  np.pi * (R_ZYLINDER**2 - R_ZYL_BOT**2),
        "top":  np.pi * (R_ZYLINDER**2 - R_ZYL_TOP**2),
        "wall": 2 * np.pi * R_ZYLINDER * H_CYLINDER,
    }
    return {a: all_surfaces[a] for a in areas}


def allocate_N_per_area(N: int, areas: list) -> dict:
    """Distribute N across areas proportional to surface (largest-remainder method)."""
    surfaces = compute_area_surfaces(areas)
    total = sum(surfaces.values())
    fractions = {a: N * s / total for a, s in surfaces.items()}
    floors = {a: int(np.floor(f)) for a, f in fractions.items()}
    remainders = {a: fractions[a] - floors[a] for a in fractions}
    leftover = N - sum(floors.values())
    sorted_areas = sorted(remainders, key=lambda a: remainders[a], reverse=True)
    allocation = dict(floors)
    for i in range(leftover):
        allocation[sorted_areas[i]] += 1

    target_density = N / total
    print(f"\nPer-area PMT allocation (N={N}):")
    print(f"  {'Area':<6} {'N':>6} {'Area (M mm²)':>14} {'Density':>14} {'Dev from target':>16}")
    print(f"  {'-' * 60}")
    for a in areas:
        n_a = allocation[a]
        s = surfaces[a]
        dens = n_a / s
        dev = (dens - target_density) / target_density * 100
        print(f"  {a:<6} {n_a:>6} {s/1e6:>14.2f} {dens:>14.6e} {dev:>+15.1f}%")
    return allocation


# ---------------------------------------------------------------------------
# Voxel generators
# ---------------------------------------------------------------------------

def generate_pit_voxels() -> list:
    """Grid voxels inside circle r <= R_PIT on the bottom pit plane."""
    diameter = 2 * R_PIT
    grid_size = math.ceil(diameter / L_VOXEL) * L_VOXEL
    half_grid = grid_size / 2
    steps = int(grid_size / L_VOXEL)

    z_mid = Z_BASE + DZ_PIT / 2
    z_min_v = z_mid - DZ_PIT / 2
    z_max_v = z_mid + DZ_PIT / 2

    voxels = []
    for y_idx in range(steps):
        for x_idx in range(steps):
            x = -half_grid + x_idx * L_VOXEL
            y = -half_grid + y_idx * L_VOXEL
            corners_2d = [
                (x,           y),
                (x + L_VOXEL, y),
                (x,           y + L_VOXEL),
                (x + L_VOXEL, y + L_VOXEL),
            ]
            if not any(cx**2 + cy**2 <= R_PIT**2 for cx, cy in corners_2d):
                continue
            mid_x = x + L_VOXEL / 2
            mid_y = y + L_VOXEL / 2
            corners_3d = [
                [mid_x - L_VOXEL/2, mid_y - L_VOXEL/2, z_min_v],
                [mid_x + L_VOXEL/2, mid_y - L_VOXEL/2, z_min_v],
                [mid_x + L_VOXEL/2, mid_y + L_VOXEL/2, z_min_v],
                [mid_x - L_VOXEL/2, mid_y + L_VOXEL/2, z_min_v],
                [mid_x - L_VOXEL/2, mid_y - L_VOXEL/2, z_max_v],
                [mid_x + L_VOXEL/2, mid_y - L_VOXEL/2, z_max_v],
                [mid_x + L_VOXEL/2, mid_y + L_VOXEL/2, z_max_v],
                [mid_x - L_VOXEL/2, mid_y + L_VOXEL/2, z_max_v],
            ]
            voxels.append({
                "index": f"00{y_idx:02d}{x_idx:02d}",
                "center": [mid_x, mid_y, z_mid],
                "corners": corners_3d,
                "layer": "pit",
            })
    return voxels


def generate_bot_voxels() -> list:
    """Grid voxels in annulus R_ZYL_BOT <= r <= R_ZYLINDER on the bottom disk."""
    diameter = 2 * R_ZYLINDER
    grid_size = math.ceil(diameter / L_VOXEL) * L_VOXEL
    half_grid = grid_size / 2
    steps = int(grid_size / L_VOXEL)

    z_mid = Z_BASE + T_ZYLINDER / 2
    z_min_v = z_mid - T_ZYLINDER / 2
    z_max_v = z_mid + T_ZYLINDER / 2

    voxels = []
    for y_idx in range(steps):
        for x_idx in range(steps):
            x = -half_grid + x_idx * L_VOXEL
            y = -half_grid + y_idx * L_VOXEL
            corners_2d = [
                (x,           y),
                (x + L_VOXEL, y),
                (x,           y + L_VOXEL),
                (x + L_VOXEL, y + L_VOXEL),
            ]
            if not any(cx**2 + cy**2 <= R_ZYLINDER**2
                       and cx**2 + cy**2 > R_ZYL_BOT**2
                       for cx, cy in corners_2d):
                continue
            mid_x = x + L_VOXEL / 2
            mid_y = y + L_VOXEL / 2
            corners_3d = [
                [mid_x - L_VOXEL/2, mid_y - L_VOXEL/2, z_min_v],
                [mid_x + L_VOXEL/2, mid_y - L_VOXEL/2, z_min_v],
                [mid_x - L_VOXEL/2, mid_y + L_VOXEL/2, z_min_v],
                [mid_x + L_VOXEL/2, mid_y + L_VOXEL/2, z_min_v],
                [mid_x - L_VOXEL/2, mid_y - L_VOXEL/2, z_max_v],
                [mid_x + L_VOXEL/2, mid_y - L_VOXEL/2, z_max_v],
                [mid_x - L_VOXEL/2, mid_y + L_VOXEL/2, z_max_v],
                [mid_x + L_VOXEL/2, mid_y + L_VOXEL/2, z_max_v],
            ]
            voxels.append({
                "index": f"01{y_idx:02d}{x_idx:02d}",
                "center": [mid_x, mid_y, z_mid],
                "corners": corners_3d,
                "layer": "bot",
            })
    return voxels


def generate_top_voxels() -> list:
    """Grid voxels in annulus R_ZYL_TOP <= r <= R_ZYLINDER on the top disk."""
    diameter = 2 * R_ZYLINDER
    grid_size = math.ceil(diameter / L_VOXEL) * L_VOXEL
    half_grid = grid_size / 2
    steps = int(grid_size / L_VOXEL)

    z_top_plane = Z_BASE + H_CYLINDER
    z_mid = z_top_plane + T_ZYLINDER / 2
    z_min_v = z_mid - T_ZYLINDER / 2
    z_max_v = z_mid + T_ZYLINDER / 2

    voxels = []
    for y_idx in range(steps):
        for x_idx in range(steps):
            x = -half_grid + x_idx * L_VOXEL
            y = -half_grid + y_idx * L_VOXEL
            corners_2d = [
                (x,           y),
                (x + L_VOXEL, y),
                (x,           y + L_VOXEL),
                (x + L_VOXEL, y + L_VOXEL),
            ]
            if not any(cx**2 + cy**2 <= R_ZYLINDER**2
                       and cx**2 + cy**2 > R_ZYL_TOP**2
                       for cx, cy in corners_2d):
                continue
            mid_x = x + L_VOXEL / 2
            mid_y = y + L_VOXEL / 2
            corners_3d = [
                [mid_x - L_VOXEL/2, mid_y - L_VOXEL/2, z_min_v],
                [mid_x + L_VOXEL/2, mid_y - L_VOXEL/2, z_min_v],
                [mid_x + L_VOXEL/2, mid_y + L_VOXEL/2, z_min_v],
                [mid_x - L_VOXEL/2, mid_y + L_VOXEL/2, z_min_v],
                [mid_x - L_VOXEL/2, mid_y - L_VOXEL/2, z_max_v],
                [mid_x + L_VOXEL/2, mid_y - L_VOXEL/2, z_max_v],
                [mid_x + L_VOXEL/2, mid_y + L_VOXEL/2, z_max_v],
                [mid_x - L_VOXEL/2, mid_y + L_VOXEL/2, z_max_v],
            ]
            voxels.append({
                "index": f"99{y_idx:02d}{x_idx:02d}",
                "center": [mid_x, mid_y, z_mid],
                "corners": corners_3d,
                "layer": "top",
            })
    return voxels


def generate_wall_voxels() -> list:
    """Parametric voxels on the cylinder wall surface."""
    n_theta = int(round(2 * np.pi * R_ZYLINDER / L_VOXEL))
    angle_per_segment = 2 * np.pi / n_theta
    end_zz = int(np.ceil(H_CYLINDER / L_VOXEL))
    w_index = 30

    r_in = float(R_ZYLINDER)
    r_out = float(R_ZYLINDER + T_ZYLINDER)

    voxels = []
    for i in range(n_theta):
        theta1 = i * angle_per_segment
        theta2 = (i + 1) * angle_per_segment

        for zz in range(end_zz):
            z_bot = Z_BASE + zz * L_VOXEL
            z_top_v = z_bot + L_VOXEL

            # Only keep voxels that overlap with the cylinder height range
            z_overlap_min = max(z_bot, Z_BASE)
            z_overlap_max = min(z_top_v, Z_BASE + H_CYLINDER)
            if z_overlap_min >= z_overlap_max:
                continue

            cos1, sin1 = np.cos(theta1), np.sin(theta1)
            cos2, sin2 = np.cos(theta2), np.sin(theta2)
            all_corners = [
                [r_in  * cos1, r_in  * sin1, z_bot],
                [r_in  * cos2, r_in  * sin2, z_bot],
                [r_in  * cos2, r_in  * sin2, z_top_v],
                [r_in  * cos1, r_in  * sin1, z_top_v],
                [r_out * cos1, r_out * sin1, z_bot],
                [r_out * cos2, r_out * sin2, z_bot],
                [r_out * cos2, r_out * sin2, z_top_v],
                [r_out * cos1, r_out * sin1, z_top_v],
            ]
            center = np.mean(all_corners, axis=0).tolist()
            voxels.append({
                "index": f"{w_index:02d}{zz:02d}{i:02d}",
                "center": center,
                "corners": all_corners,
                "layer": "wall",
            })
    return voxels


def generate_voxels_for_area(area: str) -> list:
    """Dispatch voxel generation to the appropriate area generator."""
    generators = {
        "pit":  generate_pit_voxels,
        "bot":  generate_bot_voxels,
        "top":  generate_top_voxels,
        "wall": generate_wall_voxels,
    }
    return generators[area]()


# ---------------------------------------------------------------------------
# Fibonacci sampling
# ---------------------------------------------------------------------------

def fibonacci_disk(n_points: int, r_inner: float, r_outer: float) -> np.ndarray:
    """Fibonacci spiral points in an annular disk. Returns (n, 2) array [x, y]."""
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_points)
    theta = 2 * np.pi * indices / golden_ratio
    r = np.sqrt(r_inner**2 + (r_outer**2 - r_inner**2) * (indices + 0.5) / n_points)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def fibonacci_cylinder_wall(n_points: int, radius: float,
                             z_min: float, z_max: float) -> np.ndarray:
    """Fibonacci spiral points on a cylinder wall. Returns (n, 3) array [x, y, z]."""
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_points)
    theta = 2 * np.pi * indices / golden_ratio
    z = z_min + (z_max - z_min) * (indices + 0.5) / n_points
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])


def select_homogeneous_for_area(n: int, area: str, voxels: list) -> list:
    """Fibonacci-guided greedy NN selection of n voxels for a given area."""
    valid = [v for v in voxels if is_valid_pmt_position(v["center"], v["layer"])]

    if area == "pit":
        fib_2d = fibonacci_disk(n, 0, R_PIT - PMT_RADIUS)
        z_ref = voxels[0]["center"][2] if voxels else float(Z_BASE + DZ_PIT / 2)
        targets = np.column_stack([fib_2d, np.full(n, z_ref)])
        candidates = valid

    elif area == "bot":
        fib_2d = fibonacci_disk(n, R_ZYL_BOT + PMT_RADIUS, R_ZYLINDER - PMT_RADIUS)
        z_ref = voxels[0]["center"][2] if voxels else float(Z_BASE + T_ZYLINDER / 2)
        targets = np.column_stack([fib_2d, np.full(n, z_ref)])
        if len(valid) >= n:
            candidates = valid
        elif len(voxels) >= n:
            candidates = list(voxels)
            print(f"  Warning: bot has only {len(valid)} valid voxels; "
                  f"using all {len(voxels)} (incl. invalid positions)")
        else:
            raise RuntimeError(
                f"Bot needs {n} voxels but only {len(voxels)} total "
                f"({len(valid)} valid) exist."
            )

    elif area == "top":
        fib_2d = fibonacci_disk(n, R_ZYL_TOP + PMT_RADIUS, R_ZYLINDER - PMT_RADIUS)
        z_ref = voxels[0]["center"][2] if voxels else float(Z_BASE + H_CYLINDER + T_ZYLINDER / 2)
        targets = np.column_stack([fib_2d, np.full(n, z_ref)])
        candidates = valid

    elif area == "wall":
        z_min_wall = float(Z_BASE + PMT_RADIUS)
        z_max_wall = float(Z_BASE + H_CYLINDER - PMT_RADIUS)
        targets = fibonacci_cylinder_wall(n, float(R_ZYLINDER), z_min_wall, z_max_wall)
        candidates = valid

    else:
        raise ValueError(f"Unknown area: {area!r}")

    if not candidates:
        raise RuntimeError(f"No valid voxels for area '{area}'.")
    if len(candidates) < n:
        raise RuntimeError(
            f"Area '{area}': need {n} voxels but only {len(candidates)} available."
        )

    # Greedy nearest-neighbour matching (no reuse)
    used: set = set()
    selected = []
    for target in targets:
        best_dist = float("inf")
        best_idx = None
        for i, v in enumerate(candidates):
            if i in used:
                continue
            d = float(np.linalg.norm(np.array(v["center"]) - target))
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx is not None:
            selected.append(candidates[best_idx])
            used.add(best_idx)
    return selected


# ---------------------------------------------------------------------------
# Wasserstein homogeneity
# ---------------------------------------------------------------------------
def sample_reference_distribution(
    M: int = 3000,
    seed: int = 42,
    areas: list[str] | None = None,
    return_layers: bool = False,
) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
    """Place M reference points on the detector surface via Fibonacci spirals.

    Points are distributed across the requested areas proportionally to
    surface area using the largest-remainder method, then placed
    deterministically with area-uniform Fibonacci spirals (the same scheme
    as the homogeneous PMT setup, :func:`fibonacci_disk` /
    :func:`fibonacci_cylinder_wall`).  The spirals span the *full physical
    surface* of each area (no ``PMT_RADIUS`` inset).

    Parameters
    ----------
    M : int
        Total number of reference points.
    seed : int
        Accepted for API compatibility but unused: Fibonacci placement is
        deterministic.
    areas : list of str or None
        Subset of ``["pit", "bot", "top", "wall"]``.  If None, all four
        areas are used.
    return_layers : bool
        If True, also return a per-point area-label array aligned with the
        returned points.

    Returns
    -------
    ref : np.ndarray, shape (M, 3)
        3-D coordinates of the reference points in mm.
    layers : np.ndarray, shape (M,)
        Per-point area label ("pit"/"bot"/"top"/"wall"). Only returned when
        ``return_layers`` is True.
    """
    _all_areas = ["pit", "bot", "top", "wall"]
    areas_to_use = areas if areas is not None else _all_areas

    del seed  # Fibonacci placement is deterministic; kept only for API compat.

    # z-coordinates of the flat surfaces (use homogeneous.py voxel conventions)
    z_pit = float(Z_BASE + DZ_PIT / 2)
    z_bot = float(Z_BASE + T_ZYLINDER / 2)
    z_top = float(Z_BASE + H_CYLINDER + T_ZYLINDER / 2)
    z_wall_min = float(Z_BASE)
    z_wall_max = float(Z_BASE + H_CYLINDER)

    surface_areas: dict[str, float] = {
        "pit":  np.pi * R_PIT**2,
        "bot":  np.pi * (R_ZYLINDER**2 - R_ZYL_BOT**2),
        "top":  np.pi * (R_ZYLINDER**2 - R_ZYL_TOP**2),
        "wall": 2.0 * np.pi * R_ZYLINDER * (z_wall_max - z_wall_min),
    }
    selected = {a: surface_areas[a] for a in areas_to_use}
    total_area = sum(selected.values())

    # Proportional allocation with largest-remainder rounding
    raw = {a: M * s / total_area for a, s in selected.items()}
    floors = {a: int(np.floor(v)) for a, v in raw.items()}
    remainders = {a: raw[a] - floors[a] for a in raw}
    leftover = M - sum(floors.values())
    sorted_by_rem = sorted(remainders, key=lambda a: remainders[a], reverse=True)
    m_per_area = dict(floors)
    for i in range(leftover):
        m_per_area[sorted_by_rem[i]] += 1

    parts: list[np.ndarray] = []
    layer_parts: list[np.ndarray] = []

    if "pit" in areas_to_use:
        m = m_per_area["pit"]
        # Full physical disk (r_in = 0 .. R_PIT)
        fib_2d = fibonacci_disk(m, 0.0, float(R_PIT))
        parts.append(np.column_stack([fib_2d, np.full(m, z_pit)]))
        layer_parts.append(np.full(m, "pit", dtype=object))

    if "bot" in areas_to_use:
        m = m_per_area["bot"]
        fib_2d = fibonacci_disk(m, float(R_ZYL_BOT), float(R_ZYLINDER))
        parts.append(np.column_stack([fib_2d, np.full(m, z_bot)]))
        layer_parts.append(np.full(m, "bot", dtype=object))

    if "top" in areas_to_use:
        m = m_per_area["top"]
        fib_2d = fibonacci_disk(m, float(R_ZYL_TOP), float(R_ZYLINDER))
        parts.append(np.column_stack([fib_2d, np.full(m, z_top)]))
        layer_parts.append(np.full(m, "top", dtype=object))

    if "wall" in areas_to_use:
        m = m_per_area["wall"]
        parts.append(fibonacci_cylinder_wall(m, float(R_ZYLINDER), z_wall_min, z_wall_max))
        layer_parts.append(np.full(m, "wall", dtype=object))

    points = np.vstack(parts)
    if return_layers:
        layers = (np.concatenate(layer_parts) if layer_parts
                  else np.empty(0, dtype=object))
        return points, layers
    return points


def compute_wasserstein_homogeneity(
    centers: np.ndarray,
    reference: np.ndarray | None = None,
    M: int = 3000,
    seed: int = 42,
) -> dict:
    """2-Wasserstein distance between a PMT configuration and the uniform
    detector surface.

    Uses exact Earth Mover's Distance (``ot.emd2``) on a squared-Euclidean
    cost matrix.  Lower W2 means better spatial homogeneity.

    If ``reference`` is pre-computed, it is reused directly (pass this when
    calling many times to avoid redundant sampling).  Otherwise M reference
    points are sampled via ``sample_reference_distribution``.

    Parameters
    ----------
    centers : np.ndarray, shape (N, 3)
        3-D positions of the PMT configuration in mm.
    reference : np.ndarray, shape (M, 3) or None
        Pre-computed reference sample.  If None, computed from
        ``sample_reference_distribution(M, seed)``.
    M : int
        Reference sample size (used only when ``reference`` is None).
    seed : int
        Seed for reference sampling (used only when ``reference`` is None).

    Returns
    -------
    dict with keys:
        ``w2``         — W2 distance in mm (sqrt of OT cost); primary metric.
        ``ot_cost``    — raw OT cost (W2²); for debugging.
        ``n_config``   — number of configuration points N.
        ``m_reference``— number of reference points used.

    Notes
    -----
    For N=300, M=3000 the cost matrix is ~7 MB and ``ot.emd2`` runs in
    < 1 s.  For tight loops (> 1000 calls) consider the faster
    ``ot.sliced_wasserstein_distance`` approximation instead.
    """

    if len(centers) < 2:
        raise ValueError(f"compute_wasserstein_homogeneity requires at least 2 points, got {len(centers)}")

    if reference is None:
        reference = sample_reference_distribution(M=M, seed=seed)

    N = len(centers)
    M_ref = len(reference)

    a = np.ones(N, dtype=np.float64) / N
    b = np.ones(M_ref, dtype=np.float64) / M_ref
    cost = _ssd.cdist(centers.astype(np.float64), reference.astype(np.float64), metric="sqeuclidean")

    ot_cost = float(ot.emd2(a, b, cost))
    return {
        "w2":          float(np.sqrt(ot_cost)),
        "ot_cost":     ot_cost,
        "n_config":    N,
        "m_reference": M_ref,
    }


# ---------------------------------------------------------------------------
# Process-wide W2 reference cache (M=3000, seed=42, all areas).
# Shared by all callers in the same process via import — computed at most once.
# ---------------------------------------------------------------------------
_W2_REF: np.ndarray | None = None


def get_w2_ref() -> np.ndarray:
    """Return the cached global W2 reference distribution, computing it once."""
    global _W2_REF
    if _W2_REF is None:
        _W2_REF = sample_reference_distribution(M=3000, seed=42)
    return _W2_REF


def compute_cv_nnd(centers: np.ndarray) -> dict:
    """Coefficient of variation of the nearest-neighbour distances (NND).

    For a point set ``P = {p_1, ..., p_n}`` the nearest-neighbour distance of
    point ``p_i`` is ``NND(p_i) = min_{j != i} ||p_i - p_j||`` (Euclidean,
    detector coordinates). The coefficient of variation is then

        CV = sigma_NND / mu_NND,

    the ratio of the standard deviation to the mean of the NNDs of all points.
    If all NNDs are equal the point density is constant — a perfectly
    homogeneous distribution gives ``CV = 0``; larger CV means less uniform.

    Parameters
    ----------
    centers : np.ndarray, shape (n, 3)
        3-D positions of the points in mm.

    Returns
    -------
    dict with keys:
        ``cv``        — coefficient of variation (None if n < 2).
        ``mean_nnd``  — mean nearest-neighbour distance in mm (None if n < 2).
        ``std_nnd``   — std of nearest-neighbour distances in mm (None if n < 2).
        ``n``         — number of points.
    """
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    if n < 2:
        return {"cv": None, "mean_nnd": None, "std_nnd": None, "n": n}

    d = _ssd.cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    nnd = d.min(axis=1)

    mu = float(nnd.mean())
    sigma = float(nnd.std())  # population std (ddof=0)
    cv = sigma / mu if mu > 0 else float("inf")
    return {"cv": cv, "mean_nnd": mu, "std_nnd": sigma, "n": n}


def compute_wasserstein_homogeneity_with_baseline(
    centers: np.ndarray,
    fibonacci_centers: np.ndarray,
    M: int = 3000,
    seed: int = 42,
) -> dict:
    """Like ``compute_wasserstein_homogeneity`` but also reports a Fibonacci
    baseline and a normalised score.

    Computes W2 for both ``centers`` and ``fibonacci_centers`` against the
    *same* reference sample (same seed), enabling a normalised score:
    ``w2_normalized = w2 / w2_fibonacci_baseline``.  A value of 1.0 means the
    configuration is as homogeneous as the Fibonacci ideal.

    Parameters
    ----------
    centers : np.ndarray, shape (N, 3)
        Configuration to evaluate.
    fibonacci_centers : np.ndarray, shape (N, 3)
        Fibonacci reference configuration of the same size.
    M : int
        Reference sample size.
    seed : int
        Seed for reference sampling.

    Returns
    -------
    dict with keys:
        ``w2``                   — W2 of the input configuration.
        ``w2_fibonacci_baseline``— W2 of the Fibonacci configuration.
        ``w2_normalized``        — ``w2 / w2_fibonacci_baseline``.
        ``ot_cost``              — raw OT cost for the input configuration.
        ``n_config``             — N.
        ``m_reference``          — M.
    """
    reference = sample_reference_distribution(M=M, seed=seed)
    result = compute_wasserstein_homogeneity(centers, reference=reference)
    fib_result = compute_wasserstein_homogeneity(fibonacci_centers, reference=reference)
    w2_fib = fib_result["w2"]
    return {
        **result,
        "w2_fibonacci_baseline": w2_fib,
        "w2_normalized": result["w2"] / w2_fib if w2_fib > 0.0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all_voxels_2d(voxels_by_area: dict, output_path: str) -> None:
    """2×2 figure: one subplot per area showing voxel grid in 2D."""
    area_order = [a for a in VALID_AREAS if a in voxels_by_area]
    ncols = 2
    nrows = math.ceil(len(area_order) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7 * nrows), squeeze=False)

    for idx, area in enumerate(area_order):
        ax = axes[idx // ncols][idx % ncols]
        voxels = voxels_by_area[area]

        if area == "wall":
            if voxels:
                centers = np.array([v["center"] for v in voxels])
                phi = np.arctan2(centers[:, 1], centers[:, 0])
                ax.scatter(phi, centers[:, 2], s=2, alpha=0.5, color="orange")
            ax.set_xlabel("φ [rad]")
            ax.set_ylabel("z [mm]")
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(Z_BASE, Z_BASE + H_CYLINDER)
        else:
            for v in voxels:
                cx, cy = v["center"][0], v["center"][1]
                ax.add_patch(patches.Rectangle(
                    (cx - L_VOXEL / 2, cy - L_VOXEL / 2), L_VOXEL, L_VOXEL,
                    linewidth=0.3, edgecolor=AREA_COLORS[area], facecolor="none"
                ))
            if area == "pit":
                ax.add_patch(patches.Circle(
                    (0, 0), R_PIT, fill=False, edgecolor="black", linewidth=1.2))
                lim = R_PIT * 1.1
            elif area in ("bot", "top"):
                r_inner = R_ZYL_BOT if area == "bot" else R_ZYL_TOP
                for r in [r_inner, R_ZYLINDER]:
                    ax.add_patch(patches.Circle(
                        (0, 0), r, fill=False, edgecolor="black", linewidth=1.2))
                lim = R_ZYLINDER * 1.1
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")

        ax.set_title(f"{area.capitalize()} — all voxels (N={len(voxels)})")
        ax.grid(True, alpha=0.3)

    for idx in range(len(area_order), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.suptitle("All generated voxels — 2D view per area", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_all_voxels_3d(voxels_by_area: dict, output_path: str) -> None:
    """3D scatter of all voxel centers, colored by area."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    for area in VALID_AREAS:
        voxels = voxels_by_area.get(area, [])
        if not voxels:
            continue
        centers = np.array([v["center"] for v in voxels])
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   s=2, alpha=0.4, color=AREA_COLORS[area], label=f"{area} ({len(voxels)})")

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title("All voxels — 3D view (colored by area)")
    ax.legend(markerscale=4)
    ax.set_xlim(-R_ZYLINDER * 1.2, R_ZYLINDER * 1.2)
    ax.set_ylim(-R_ZYLINDER * 1.2, R_ZYLINDER * 1.2)
    ax.set_zlim(Z_BASE - 100, Z_BASE + H_CYLINDER + 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_selected_3d(selected_by_area: dict, all_by_area: dict, output_path: str) -> None:
    """3D: all voxel centers in light gray + selected voxels colored by area."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # All voxels as faint background
    for voxels in all_by_area.values():
        if voxels:
            centers = np.array([v["center"] for v in voxels])
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                       s=1, alpha=0.1, color="lightgray", zorder=1)

    # Selected voxels
    total = 0
    for area in VALID_AREAS:
        voxels = selected_by_area.get(area, [])
        if not voxels:
            continue
        centers = np.array([v["center"] for v in voxels])
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   s=40, alpha=0.85, color=AREA_COLORS[area],
                   edgecolors="black", linewidths=0.3,
                   label=f"{area} ({len(voxels)})", zorder=2)
        total += len(voxels)

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title(
        f"Selected PMT positions (total={total})\n"
        f"Red=Pit, Blue=Bot, Green=Top, Orange=Wall"
    )
    ax.legend()
    ax.set_xlim(-R_ZYLINDER * 1.1, R_ZYLINDER * 1.1)
    ax.set_ylim(-R_ZYLINDER * 1.1, R_ZYLINDER * 1.1)
    ax.set_zlim(Z_BASE - 100, Z_BASE + H_CYLINDER + 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_homogeneity(
    selected_by_area: dict,
    w2_data: dict[str, float | None],
    output_path: str,
) -> None:
    """2D homogeneity plots per area: XY for pit/bot/top, φ-z for wall, with W2."""
    area_order = [a for a in VALID_AREAS if a in selected_by_area and selected_by_area[a]]
    if not area_order:
        return
    ncols = 2
    nrows = math.ceil(len(area_order) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6 * nrows), squeeze=False)

    for idx, area in enumerate(area_order):
        ax = axes[idx // ncols][idx % ncols]
        voxels = selected_by_area[area]
        w2_val = w2_data.get(area)
        w2_str = f"W2={w2_val:.1f} mm" if w2_val is not None else "n/a"

        if area == "wall":
            centers = np.array([v["center"] for v in voxels])
            phi = np.arctan2(centers[:, 1], centers[:, 0])
            ax.scatter(phi, centers[:, 2], s=15, color="orange", alpha=0.8)
            for p, zv in zip(phi, centers[:, 2]):
                ax.add_patch(plt.Circle(
                    (p, zv), PMT_RADIUS / R_ZYLINDER,
                    fill=False, edgecolor="orange", linewidth=0.4, alpha=0.5
                ))
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(Z_BASE, Z_BASE + H_CYLINDER)
            ax.set_xlabel("φ [rad]")
            ax.set_ylabel("z [mm]")
        else:
            centers = np.array([v["center"] for v in voxels])
            color = AREA_COLORS[area]
            ax.scatter(centers[:, 0], centers[:, 1], s=15, color=color, alpha=0.8)
            for c in centers:
                ax.add_patch(plt.Circle(
                    (c[0], c[1]), PMT_RADIUS,
                    fill=False, edgecolor=color, linewidth=0.4, alpha=0.5
                ))
            if area == "pit":
                ax.add_patch(plt.Circle(
                    (0, 0), R_PIT, fill=False, edgecolor="black",
                    linewidth=1.5, linestyle="--"
                ))
                lim = R_PIT * 1.15
            elif area in ("bot", "top"):
                r_inner = R_ZYL_BOT if area == "bot" else R_ZYL_TOP
                for r in [r_inner, R_ZYLINDER]:
                    ax.add_patch(plt.Circle(
                        (0, 0), r, fill=False, edgecolor="black",
                        linewidth=1.5, linestyle="--"
                    ))
                lim = R_ZYLINDER * 1.15
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")

        ax.set_title(f"{area.capitalize()} (N={len(voxels)}, {w2_str})")
        ax.grid(True, alpha=0.3)

    for idx in range(len(area_order), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.suptitle(
        "PMT positions per area — homogeneity check\n"
        "W2 = 2-Wasserstein distance vs uniform reference (lower = more uniform)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_homogeneity_from_selection(
    centers: np.ndarray,
    layers: np.ndarray,
    output_path: str,
    label: str = "",
    global_w2: float | None = None,
) -> dict[str, dict]:
    """Per-area 2D PMT-position figure with per-area CV and a global W2.

    Renders the four detector areas (pit / bot / top / wall) as 2D position
    panels in a single figure: x-y scatter with PMT-radius circles for the flat
    surfaces, and an unrolled φ-z scatter for the wall. Each panel title carries
    the per-area coefficient of variation of nearest-neighbour distances
    (:func:`compute_cv_nnd`); the figure suptitle carries the global W2
    homogeneity value for the whole setup.

    Parameters
    ----------
    centers : np.ndarray, shape (N, 3)
        PMT center coordinates in mm.
    layers : np.ndarray, shape (N,)
        Area label per PMT ("pit" / "bot" / "top" / "wall").
    output_path : str
        PNG output path.
    label : str
        Optional label for the figure suptitle (e.g. the config name).
    global_w2 : float or None
        Pre-computed global W2 (mm). If None, it is computed here from the
        cached uniform-surface reference via :func:`get_w2_ref`.

    Returns
    -------
    dict
        ``{area: compute_cv_nnd(...)}`` for each populated area, plus a
        ``"_global"`` entry holding ``{"w2": global_w2}``.
    """
    centers = np.asarray(centers, dtype=np.float64)
    layers = np.asarray(layers)

    if global_w2 is None and len(centers) >= 2:
        global_w2 = compute_wasserstein_homogeneity(
            centers, reference=get_w2_ref())["w2"]

    area_order = [a for a in VALID_AREAS if np.any(layers == a)]
    if not area_order:
        print("  [SKIP] plot_homogeneity_from_selection: no voxels to plot.")
        return {}

    cv_by_area: dict[str, dict] = {}

    ncols = 2
    nrows = math.ceil(len(area_order) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6 * nrows),
                             squeeze=False)

    for idx, area in enumerate(area_order):
        ax = axes[idx // ncols][idx % ncols]
        area_centers = centers[layers == area]

        cv_res = compute_cv_nnd(area_centers)
        cv_by_area[area] = cv_res
        cv_str = (f"CV={cv_res['cv']:.3f}" if cv_res["cv"] is not None
                  else "CV n/a")

        color = AREA_COLORS[area]
        if area == "wall":
            phi = np.arctan2(area_centers[:, 1], area_centers[:, 0])
            ax.scatter(phi, area_centers[:, 2], s=15, color=color, alpha=0.8)
            for p, zv in zip(phi, area_centers[:, 2]):
                ax.add_patch(plt.Circle(
                    (p, zv), PMT_RADIUS / R_ZYLINDER,
                    fill=False, edgecolor=color, linewidth=0.4, alpha=0.5,
                ))
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(Z_BASE, Z_BASE + H_CYLINDER)
            ax.set_xlabel("φ [rad]")
            ax.set_ylabel("z [mm]")
        else:
            ax.scatter(area_centers[:, 0], area_centers[:, 1],
                       s=15, color=color, alpha=0.8)
            for c in area_centers:
                ax.add_patch(plt.Circle(
                    (c[0], c[1]), PMT_RADIUS,
                    fill=False, edgecolor=color, linewidth=0.4, alpha=0.5,
                ))
            if area == "pit":
                ax.add_patch(plt.Circle(
                    (0, 0), R_PIT, fill=False, edgecolor="black",
                    linewidth=1.5, linestyle="--"))
                lim = R_PIT * 1.15
            else:  # bot / top
                r_inner = R_ZYL_BOT if area == "bot" else R_ZYL_TOP
                for r in [r_inner, R_ZYLINDER]:
                    ax.add_patch(plt.Circle(
                        (0, 0), r, fill=False, edgecolor="black",
                        linewidth=1.5, linestyle="--"))
                lim = R_ZYLINDER * 1.15
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")

        ax.set_title(f"{area.capitalize()} (N={len(area_centers)}, {cv_str})")
        ax.grid(True, alpha=0.3)

    for idx in range(len(area_order), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    w2_str = (f"Global W2 = {global_w2:.1f} mm"
              if global_w2 is not None else "Global W2 = n/a")
    suptitle = f"PMT homogeneity per area"
    if label:
        suptitle += f" — {label}"
    suptitle += (f"\n{w2_str}   ·   "
                 f"CV = σ/μ of nearest-neighbour distances (per area, lower = more uniform)")
    plt.suptitle(suptitle, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    cv_by_area["_global"] = {"w2": global_w2}
    return cv_by_area


# ---------------------------------------------------------------------------
# Output filename helper
# ---------------------------------------------------------------------------

def build_output_stem(mode: str, geometry: str,
                      N: int = None, areas: list = None) -> str:
    if mode == "generate":
        return f"allVoxels_{geometry}"
    areas_str = "_".join(sorted(areas, key=VALID_AREAS.index))
    return f"homogeneous{N}PMTs_{areas_str}_{geometry}"


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------

def run_generate(geometry: str, output_dir: str) -> None:
    print(f"Mode: generate | geometry: {geometry}")
    os.makedirs(output_dir, exist_ok=True)
    stem = build_output_stem("generate", geometry)

    voxels_by_area: dict = {}
    for area in VALID_AREAS:
        print(f"  Generating {area} voxels...", end="", flush=True)
        voxels_by_area[area] = generate_voxels_for_area(area)
        print(f" {len(voxels_by_area[area])} voxels")

    total = sum(len(v) for v in voxels_by_area.values())
    print(f"  Total voxels: {total}")

    # JSON — all voxels ordered pit → bot → top → wall
    all_voxels = [v for area in VALID_AREAS for v in voxels_by_area[area]]
    json_path = os.path.join(output_dir, f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_voxels, f, indent=2)
    print(f"  Saved: {json_path}")

    print("  Generating 2D plot...")
    plot_all_voxels_2d(voxels_by_area, os.path.join(output_dir, f"{stem}_2d.png"))
    print("  Generating 3D plot...")
    plot_all_voxels_3d(voxels_by_area, os.path.join(output_dir, f"{stem}_3d.png"))
    print("Done.")


def load_all_voxels_json(path: str) -> list[dict]:
    """Load voxels from JSON (bare list or greedy/homogeneous wrapper)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "selected_voxels" in data:
        return data["selected_voxels"]
    raise ValueError(
        f"{path}: expected a list or a dict with 'selected_voxels', "
        f"got {type(data).__name__}"
    )


def run_select(geometry: str, N: int, areas: list, output_dir: str,
               all_voxels_path: str) -> None:
    print(f"Mode: select | geometry: {geometry} | N={N} | areas={areas}")
    print(f"  All-voxels file: {all_voxels_path}")
    os.makedirs(output_dir, exist_ok=True)
    stem = build_output_stem("select", geometry, N, areas)

    # Load candidate pool from the provided JSON, grouped by area
    all_voxels_list = load_all_voxels_json(all_voxels_path)
    voxels_by_area: dict = {}
    for area in areas:
        voxels_by_area[area] = [v for v in all_voxels_list if v.get("layer") == area]
        print(f"  Loaded {len(voxels_by_area[area])} {area} voxels from file")

    # Area-proportional allocation
    allocation = allocate_N_per_area(N, areas)

    # Fibonacci-guided greedy selection
    print()
    selected_by_area: dict = {}
    for area in areas:
        n_area = allocation[area]
        print(f"  Selecting {n_area} voxels for {area}...", end="", flush=True)
        selected_by_area[area] = select_homogeneous_for_area(
            n_area, area, voxels_by_area[area]
        )
        print(f" done ({len(selected_by_area[area])} selected)")

    total_selected = sum(len(v) for v in selected_by_area.values())
    print(f"\n  Total selected: {total_selected}")

    # Wasserstein homogeneity
    print("\n" + "=" * 72)
    print("HOMOGENEITY CHECK: Wasserstein Distance Analysis")
    print("=" * 72)

    # Pre-compute per-area reference distributions once (seed=42, M=3000)
    area_refs = {
        area: sample_reference_distribution(M=3000, seed=42, areas=[area])
        for area in areas
        if selected_by_area.get(area)
    }
    # Global reference across all selected areas
    global_ref = sample_reference_distribution(M=3000, seed=42, areas=areas)

    all_centers = np.array([
        v["center"] for area in areas for v in selected_by_area.get(area, [])
    ])
    global_w2_result = compute_wasserstein_homogeneity(all_centers, reference=global_ref)

    print(f"\n  Global W2 = {global_w2_result['w2']:.1f} mm  "
          f"(N={global_w2_result['n_config']}, M_ref={global_w2_result['m_reference']})")
    print(f"\n  {'Area':<6} {'N':>4} {'W2 (mm)':>12}")
    print(f"  " + "-" * 26)
    w2_data: dict[str, float | None] = {}
    for area in areas:
        voxels = selected_by_area.get(area, [])
        if len(voxels) >= 2:
            centers = np.array([v["center"] for v in voxels])
            result = compute_wasserstein_homogeneity(centers, reference=area_refs[area])
            w2_data[area] = result["w2"]
            print(f"  {area:<6} {len(voxels):>4} {result['w2']:>12.1f}")
        else:
            w2_data[area] = None
            print(f"  {area:<6} {len(voxels):>4}   (too few points for stats)")
    print(f"\n  W2 = 2-Wasserstein distance vs uniform reference  |  lower = more uniform")

    # Plots
    print("\n  Generating plots...")
    plot_selected_3d(
        selected_by_area, voxels_by_area,
        os.path.join(output_dir, f"{stem}_3d.png")
    )
    plot_homogeneity(
        selected_by_area, w2_data,
        os.path.join(output_dir, f"{stem}_homogeneity.png")
    )

    # JSON — selected voxels in VALID_AREAS order
    all_selected = [v for area in areas for v in selected_by_area[area]]
    json_path = os.path.join(output_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(all_selected, f, indent=2)
    print(f"  Saved: {json_path}")
    print("Done.")


def run_reference(output_dir: str = ".", M: int = 3000, seed: int = 42) -> None:
    """Plot the homogeneously distributed W2 reference points themselves.

    Samples the same uniform-surface reference used by the W2 metric
    (``sample_reference_distribution(M, seed)`` — the points behind
    :func:`get_w2_ref`) and renders them with the per-area CV + global-W2
    figure used for PMT selections. The global W2 is the *self* W2 of the
    reference against ``get_w2_ref()`` (≈ 0 by construction); the per-area
    panel titles report the number of reference points and their CV.
    """
    print(f"Mode: reference | M={M} | seed={seed}")
    os.makedirs(output_dir, exist_ok=True)

    ref, layers = sample_reference_distribution(
        M=M, seed=seed, return_layers=True)

    # Self W2: reference vs the cached W2 reference (identical sample) ≈ 0.
    self_w2 = compute_wasserstein_homogeneity(ref, reference=get_w2_ref())["w2"]

    out_path = os.path.join(
        output_dir, f"reference_distribution_M{M}_seed{seed}_homogeneity.png")
    cv_by_area = plot_homogeneity_from_selection(
        ref, layers, out_path,
        label=f"W2 reference points (M={M}, seed={seed})",
        global_w2=self_w2,
    )

    # Console summary
    print(f"\n  Self W2 (vs get_w2_ref) = {self_w2:.3f} mm")
    print(f"  {'Area':<6} {'N_ref':>6} {'CV':>8} {'mean NND':>12} {'std NND':>12}")
    print(f"  " + "-" * 48)
    for area in VALID_AREAS:
        res = cv_by_area.get(area)
        if res is None:
            continue
        if res["cv"] is None:
            print(f"  {area:<6} {res['n']:>6}   (too few points)")
        else:
            print(f"  {area:<6} {res['n']:>6} {res['cv']:>8.3f} "
                  f"{res['mean_nnd']:>10.1f}mm {res['std_nnd']:>10.1f}mm")
    print("Done.")


# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Voxel generation and homogeneous PMT selection for the SSD detector.\n"
            "Geometry constants from src/pmtopt/geometry.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (via unified CLI):
  python main.py homogeneous --mode generate --output-dir ./output
  python main.py homogeneous --mode select -N 300 --output-dir ./output
  python main.py homogeneous --mode select -N 100 --areas pit wall --output-dir ./output
        """,
    )
    parser.add_argument(
        "--mode", required=True, choices=["generate", "select"],
        help=(
            "'generate': build all voxels for all areas + 2D/3D plots + JSON. "
            "'select': homogeneously distribute N PMTs over chosen areas + plots + JSON."
        ),
    )
    parser.add_argument(
        "--geometry", default="currentDist", choices=VALID_GEOMETRIES,
        help="Detector geometry to use. Default: currentDist.",
    )
    parser.add_argument(
        "-N", "--num-voxels", type=int, default=300,
        help="Number of PMT voxels to select (select mode only). Default: 300.",
    )
    parser.add_argument(
        "--areas", nargs="+", default=list(VALID_AREAS),
        choices=VALID_AREAS, metavar="AREA",
        help=(
            "Areas to distribute PMTs over (select mode). "
            "Choices: pit bot top wall. Default: all four. "
            "N is distributed proportional to each area's surface."
        ),
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory to write output files to. Created if it does not exist. Default: '.'",
    )
    parser.add_argument(
        "--all-voxels", default=None,
        help=(
            "Path to the all-voxels JSON file used as the candidate pool "
            "(required for select mode). Voxels are grouped by their 'layer' "
            "field; for each area the Fibonacci ideal positions are snapped to "
            "the nearest voxel center from this file. Use 'generate' mode first "
            "to create this file, or supply the HDF5-derived voxel list."
        ),
    )

    args = parser.parse_args(argv)

    if args.mode == "select":
        if args.num_voxels < 1:
            parser.error("-N must be a positive integer.")
        if args.all_voxels is None:
            parser.error("--all-voxels is required for select mode.")
        areas = sorted(set(args.areas), key=VALID_AREAS.index)
        run_select(args.geometry, args.num_voxels, areas, args.output_dir,
                   args.all_voxels)
    else:
        run_generate(args.geometry, args.output_dir)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Sanity check: W2 detects global region imbalance
    #   Run with:  python homogeneous.py --sanity-check
    # ------------------------------------------------------------------
    if len(sys.argv) == 2 and sys.argv[1] == "--sanity-check":
        print("Running Wasserstein sanity check ...")
        _N = 300
        _areas = ["pit", "bot", "top", "wall"]

        # 1. Fibonacci homogeneous configuration
        _alloc = allocate_N_per_area(_N, _areas)
        _fib_voxels: list[dict] = []
        for _a in _areas:
            _n_a = _alloc[_a]
            if _a == "pit":
                _pts = fibonacci_disk(_n_a, 0, R_PIT - PMT_RADIUS)
                _z_a = float(Z_BASE + DZ_PIT / 2)
                for _p in _pts:
                    _fib_voxels.append({"center": [_p[0], _p[1], _z_a], "layer": _a})
            elif _a == "bot":
                _pts = fibonacci_disk(_n_a, R_ZYL_BOT + PMT_RADIUS, R_ZYLINDER - PMT_RADIUS)
                _z_a = float(Z_BASE + T_ZYLINDER / 2)
                for _p in _pts:
                    _fib_voxels.append({"center": [_p[0], _p[1], _z_a], "layer": _a})
            elif _a == "top":
                _pts = fibonacci_disk(_n_a, R_ZYL_TOP + PMT_RADIUS, R_ZYLINDER - PMT_RADIUS)
                _z_a = float(Z_BASE + H_CYLINDER + T_ZYLINDER / 2)
                for _p in _pts:
                    _fib_voxels.append({"center": [_p[0], _p[1], _z_a], "layer": _a})
            elif _a == "wall":
                _z_min_w = float(Z_BASE + PMT_RADIUS)
                _z_max_w = float(Z_BASE + H_CYLINDER - PMT_RADIUS)
                _pts = fibonacci_cylinder_wall(_n_a, float(R_ZYLINDER), _z_min_w, _z_max_w)
                for _p in _pts:
                    _fib_voxels.append({"center": _p.tolist(), "layer": _a})

        _fib_centers = np.array([v["center"] for v in _fib_voxels])

        # 2. Clustered configuration: all 300 points on the wall only
        _wall_pts = fibonacci_cylinder_wall(
            _N, float(R_ZYLINDER),
            float(Z_BASE + PMT_RADIUS), float(Z_BASE + H_CYLINDER - PMT_RADIUS)
        )
        _clustered_centers = _wall_pts

        # 3. Compute W2 for both against the full reference
        _ref = sample_reference_distribution(M=3000, seed=42)
        _w2_fib = compute_wasserstein_homogeneity(_fib_centers, reference=_ref)["w2"]
        _w2_clust = compute_wasserstein_homogeneity(_clustered_centers, reference=_ref)["w2"]

        print(f"  W2 (Fibonacci homogeneous) = {_w2_fib:.1f} mm")
        print(f"  W2 (all on wall, clustered) = {_w2_clust:.1f} mm")
        assert _w2_clust > _w2_fib, (
            f"FAIL: W2_clustered ({_w2_clust:.1f}) should be > W2_fibonacci ({_w2_fib:.1f})"
        )
        print("  PASS: W2_clustered > W2_fibonacci — global imbalance correctly detected.")
    elif len(sys.argv) == 1:
        # Bare `python homogeneous.py`: plot the homogeneously distributed
        # W2 reference points (per-area CV + self W2).
        run_reference()
    else:
        main()
