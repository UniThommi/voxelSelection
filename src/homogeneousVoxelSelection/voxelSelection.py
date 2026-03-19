"""
voxelSelection.py — CLI tool for voxel generation and homogeneous PMT selection
on the cylindrical SSD detector.

Usage:
    # Generate all voxels + 2D/3D plots + JSON
    python voxelSelection.py --mode generate [--geometry currentDist] [--output-dir ./output]

    # Select N voxels homogeneously across all areas
    python voxelSelection.py --mode select -N 300 [--geometry currentDist] [--output-dir ./output]

    # Select N voxels on specific areas only
    python voxelSelection.py --mode select -N 100 --areas pit wall [--output-dir ./output]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Import geometry constants from sibling pmtopt package
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pmtopt.geometry import (  # noqa: E402
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
# Homogeneity statistics
# ---------------------------------------------------------------------------

def compute_nn_stats(voxels: list) -> tuple:
    """Euclidean nearest-neighbour distance stats."""
    if len(voxels) < 2:
        return np.array([]), {}
    centers = np.array([v["center"] for v in voxels])
    tree = KDTree(centers)
    dists, _ = tree.query(centers, k=2)
    nn = dists[:, 1]
    mean = float(np.mean(nn))
    return nn, {
        "mean": mean,
        "std":  float(np.std(nn)),
        "min":  float(np.min(nn)),
        "max":  float(np.max(nn)),
        "cv":   float(np.std(nn) / mean) if mean > 0 else 0.0,
    }


def compute_nn_stats_wall(voxels: list) -> tuple:
    """Geodesic NN stats on cylinder wall: arc length in φ, Euclidean in z."""
    if len(voxels) < 2:
        return np.array([]), {}
    centers = np.array([v["center"] for v in voxels])
    phi = np.arctan2(centers[:, 1], centers[:, 0])
    z = centers[:, 2]
    n = len(centers)
    nn = np.full(n, np.inf)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dphi = abs(phi[i] - phi[j])
            dphi = min(dphi, 2 * np.pi - dphi)
            dist = np.sqrt((R_ZYLINDER * dphi)**2 + (z[i] - z[j])**2)
            if dist < nn[i]:
                nn[i] = dist
    mean = float(np.mean(nn))
    return nn, {
        "mean": mean,
        "std":  float(np.std(nn)),
        "min":  float(np.min(nn)),
        "max":  float(np.max(nn)),
        "cv":   float(np.std(nn) / mean) if mean > 0 else 0.0,
    }


def compute_nn_stats_for_area(area: str, voxels: list) -> tuple:
    if area == "wall":
        return compute_nn_stats_wall(voxels)
    return compute_nn_stats(voxels)


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


def plot_homogeneity(selected_by_area: dict, nn_data: dict, output_path: str) -> None:
    """2D homogeneity plots per area: XY for pit/bot/top, φ-z for wall, with CV."""
    area_order = [a for a in VALID_AREAS if a in selected_by_area and selected_by_area[a]]
    if not area_order:
        return
    ncols = 2
    nrows = math.ceil(len(area_order) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6 * nrows), squeeze=False)

    for idx, area in enumerate(area_order):
        ax = axes[idx // ncols][idx % ncols]
        voxels = selected_by_area[area]
        _, stats = nn_data.get(area, (np.array([]), {}))
        cv_str = f"CV={stats['cv']:.3f}" if stats else "n/a"

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

        ax.set_title(f"{area.capitalize()} (N={len(voxels)}, {cv_str})")
        ax.grid(True, alpha=0.3)

    for idx in range(len(area_order), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.suptitle(
        "PMT positions per area — homogeneity check\n"
        "CV = std/mean of nearest-neighbour distances (lower = more uniform)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


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


def run_select(geometry: str, N: int, areas: list, output_dir: str) -> None:
    print(f"Mode: select | geometry: {geometry} | N={N} | areas={areas}")
    os.makedirs(output_dir, exist_ok=True)
    stem = build_output_stem("select", geometry, N, areas)

    # Generate voxels only for requested areas
    voxels_by_area: dict = {}
    for area in areas:
        print(f"  Generating {area} voxels...", end="", flush=True)
        voxels_by_area[area] = generate_voxels_for_area(area)
        print(f" {len(voxels_by_area[area])} voxels")

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

    # Homogeneity stats
    print("\n" + "=" * 72)
    print("HOMOGENEITY CHECK: Nearest-Neighbour Distance Analysis")
    print("=" * 72)
    print(f"  {'Area':<6} {'N':>4} {'Mean NN':>10} {'Std':>8} "
          f"{'Min':>8} {'Max':>8} {'CV':>8}")
    print(f"  {'-' * 56}")
    nn_data: dict = {}
    for area in areas:
        voxels = selected_by_area.get(area, [])
        nn_dists, stats = compute_nn_stats_for_area(area, voxels)
        nn_data[area] = (nn_dists, stats)
        if stats:
            print(f"  {area:<6} {len(voxels):>4} {stats['mean']:>10.1f} "
                  f"{stats['std']:>8.1f} {stats['min']:>8.1f} "
                  f"{stats['max']:>8.1f} {stats['cv']:>8.3f}")
        else:
            print(f"  {area:<6} {len(voxels):>4}   (too few points for stats)")
    print(f"\n  CV = std/mean  |  lower = more uniform")
    print(f"  Fibonacci ideal: CV ~0.05-0.15  |  after voxel snapping: CV <= 0.25 is good")

    # Plots
    print("\n  Generating plots...")
    plot_selected_3d(
        selected_by_area, voxels_by_area,
        os.path.join(output_dir, f"{stem}_3d.png")
    )
    plot_homogeneity(
        selected_by_area, nn_data,
        os.path.join(output_dir, f"{stem}_homogeneity.png")
    )

    # JSON — selected voxels in VALID_AREAS order
    all_selected = [v for area in areas for v in selected_by_area[area]]
    json_path = os.path.join(output_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(all_selected, f, indent=2)
    print(f"  Saved: {json_path}")
    print("Done.")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Voxel generation and homogeneous PMT selection for the SSD detector.\n"
            "Geometry constants from src/pmtopt/geometry.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all voxels + 2D/3D plots + JSON
  python voxelSelection.py --mode generate --output-dir ./output

  # Select 300 PMTs across all areas
  python voxelSelection.py --mode select -N 300 --output-dir ./output

  # Select 100 PMTs on pit and wall only
  python voxelSelection.py --mode select -N 100 --areas pit wall --output-dir ./output
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

    args = parser.parse_args()

    if args.mode == "select":
        if args.num_voxels < 1:
            parser.error("-N must be a positive integer.")
        areas = sorted(set(args.areas), key=VALID_AREAS.index)
        run_select(args.geometry, args.num_voxels, areas, args.output_dir)
    else:
        run_generate(args.geometry, args.output_dir)


if __name__ == "__main__":
    main()
