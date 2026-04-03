#!/usr/bin/env python3
"""
W2-span sampling
================
Generate PMT configurations spanning the 2-Wasserstein homogeneity range
using geometry-driven clustering algorithms — no greedy selection.

For each of n_configs configurations the algorithm:
  1. Randomly picks which areas to include (excluded area N is redistributed).
  2. For every included area, independently draws a clustering algorithm and
     its geometric starting parameters (phi_center, z_center, k, …).
  3. Probes W2 at alpha=0 (most uniform) and alpha=1 (most concentrated).
  4. Draws alpha uniformly from [0, 1] and generates the actual configuration.
  5. Snaps geometric points to nearest valid voxels (KD-tree), enforces minimum
     spacing, and redirects overflow round-robin when an area is exhausted.
  6. Computes NC detection efficiency from the B matrix and saves JSON + PNG.

Available algorithms per area
------------------------------
  fibonacci       Fibonacci-spiral reference distribution (no concentration)
  z_band          Concentrate wall points into a horizontal z-band (wall only)
  radial          Power-law radial warping toward inner or outer edge (disk only)
  multi_cluster   k clusters at random (phi, r/z) positions
  superposition   Fibonacci blend + ring/band concentrated at one r or z value
  multi_ring      n_rings rings at distinct r (disk) or z (wall) values, phi uniform

All algorithms generate point-symmetric configurations: n//2 geometric points
are produced internally and then mirrored at phi → phi+pi (C2 symmetry around
the z-axis). Small deviations from perfect symmetry arise only from voxel snapping.

Usage
-----
    python -m pmtopt.sample_w2_range \\
        --hdf5 data.hdf5 -N 300 --n-configs 50 --output-dir w2_setups/

    python src/pmtopt/main.py sample-w2-range \\
        --hdf5 data.hdf5 -N 300 --n-configs 50 --output-dir w2_setups/

Author: Thomas Buerger (University of Tübingen)
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import h5py
import numpy as np
from scipy.spatial import KDTree

from pmtopt.data_loading import load_raw_sparse, binarize_from_raw
from pmtopt.geometry import (
    PMT_RADIUS, R_PIT, R_ZYL_BOT, R_ZYL_TOP, R_ZYLINDER,
    Z_BASE_GLOBAL, H_ZYLINDER, Z_CUT_TOP,
    DEFAULT_AREA_RATIOS, compute_per_area_N,
)
from pmtopt.homogeneous import compute_wasserstein_homogeneity, get_w2_ref
from pmtopt.plotting import plot_selected_voxels
from pmtopt.sensitivity import run_sensitivity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_GOLDEN = (1.0 + 5.0 ** 0.5) / 2.0
_AREA_ORDER: list[str] = ["pit", "bot", "top", "wall"]

_Z_WALL_MIN = float(Z_BASE_GLOBAL)
_Z_WALL_MAX = float(Z_CUT_TOP)

_ALG_DISK: list[str] = ["fibonacci", "radial", "multi_cluster", "superposition", "multi_ring", "phi_sectors", "clustered_surface"]
_ALG_WALL: list[str] = ["fibonacci", "z_band", "multi_cluster", "superposition", "multi_ring", "phi_sectors", "phi_band", "pole", "clustered_surface"]



# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _area_r_bounds(area: str) -> tuple[float, float]:
    if area == "pit":
        return 0.0, float(R_PIT)
    elif area == "bot":
        return float(R_ZYL_BOT), float(R_ZYLINDER)
    elif area == "top":
        return float(R_ZYL_TOP), float(R_ZYLINDER)
    raise ValueError(f"No r-bounds for area '{area}'")


def _area_z(area: str) -> float:
    if area in ("pit", "bot"):
        return float(Z_BASE_GLOBAL)
    elif area == "top":
        return float(Z_CUT_TOP)
    raise ValueError(f"No fixed z for area '{area}'")


def _fib_angles(n: int) -> np.ndarray:
    """Fibonacci-spiral azimuthal angles in [0, 2π) for n points."""
    return (2.0 * np.pi * np.arange(n) / _GOLDEN) % (2.0 * np.pi)


def _fib_annulus_r(n: int, r_min: float, r_max: float) -> np.ndarray:
    """Fibonacci radii for n points on an annular disk [r_min, r_max]."""
    t = (np.arange(n) + 0.5) / n
    return np.sqrt(r_min ** 2 + t * (r_max ** 2 - r_min ** 2))


# ---------------------------------------------------------------------------
# Per-surface geometric point generators
# ---------------------------------------------------------------------------

def _gen_disk_points(
    n: int,
    r_min: float,
    r_max: float,
    z: float,
    alg: str,
    params: dict,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (n, 3) geometric points on an annular disk at height z.

    All distributions are point-symmetric through the cylinder axis: each of the
    n//2 generated points at (r, phi) is paired with a mirror at (r, phi+pi).
    The odd remainder (if any) gets one extra point at the radial midpoint.
    """
    n_half = n // 2
    phi_fib = _fib_angles(n_half)
    r_fib = _fib_annulus_r(n_half, r_min, r_max)

    if alg == "fibonacci":
        r, phi = r_fib, phi_fib

    elif alg == "radial":
        beta_max = params["beta_max"]
        direction = params["radial_direction"]
        t = (np.arange(n_half) + 0.5) / n_half
        if direction == "inward":        # concentrate toward r_min
            beta = 1.0 + alpha * (beta_max - 1.0)
        else:                            # concentrate toward r_max
            beta = 1.0 - alpha * (1.0 - 1.0 / beta_max)
            beta = max(beta, 0.05)
        r = np.sqrt(r_min ** 2 + (t ** beta) * (r_max ** 2 - r_min ** 2))
        phi = phi_fib

    elif alg == "multi_cluster":
        k = params["k"]
        c_phi = params["cluster_phi"]
        c_r = params["cluster_r"]
        sigma_min = float(PMT_RADIUS) * 1.5
        sigma_max = max(
            sigma_min * 2.0,
            min(r_max - r_min, np.pi * (r_min + r_max) / max(k, 1)) * 0.7,
        )
        sigma = sigma_min + (1.0 - alpha) * (sigma_max - sigma_min)
        pts_per = [n_half // k + (1 if i < n_half % k else 0) for i in range(k)]
        r_parts, phi_parts = [], []
        for ci, nc in enumerate(pts_per):
            if nc == 0:
                continue
            r_off = rng.uniform(0.0, sigma, nc)
            a_off = rng.uniform(0.0, 2.0 * np.pi, nc)
            rc = np.clip(c_r[ci] + r_off * np.cos(a_off), r_min, r_max)
            ref_r = max(float(c_r[ci]), r_min + 1e-6)
            pc = (c_phi[ci] + r_off * np.sin(a_off) / ref_r) % (2.0 * np.pi)
            r_parts.append(rc)
            phi_parts.append(pc)
        r = np.concatenate(r_parts)
        phi = np.concatenate(phi_parts)

    elif alg == "superposition":
        # Concentrated part is a ring at conc_r with uniform phi — no preferred direction.
        n_conc = int(round(alpha * n_half))
        n_uni = n_half - n_conc
        c_r = np.clip(params["conc_r"], r_min, r_max)
        blob = max(float(PMT_RADIUS) * 1.5,
                   (r_max - r_min) / max(float(np.sqrt(n_half)), 1.0))
        r_parts, phi_parts = [], []
        if n_uni > 0:
            r_parts.append(_fib_annulus_r(n_uni, r_min, r_max))
            phi_parts.append(_fib_angles(n_uni))
        if n_conc > 0:
            r_off = rng.uniform(-blob, blob, n_conc)
            rc = np.clip(c_r + r_off, r_min, r_max)
            pc = rng.uniform(0.0, 2.0 * np.pi, n_conc)
            r_parts.append(rc)
            phi_parts.append(pc)
        r = np.concatenate(r_parts) if r_parts else r_fib
        phi = np.concatenate(phi_parts) if phi_parts else phi_fib

    elif alg == "multi_ring":
        # Multiple annular rings at fixed radii, each with uniform phi.
        # Few rings (high alpha) → concentrated; many rings → spread.
        n_rings = params["n_rings"]
        ring_radii = params["ring_radii"]
        pts_per_ring = [n_half // n_rings + (1 if i < n_half % n_rings else 0)
                        for i in range(n_rings)]
        r_parts, phi_parts = [], []
        for ri, np_r in enumerate(pts_per_ring):
            if np_r == 0:
                continue
            rr = float(np.clip(ring_radii[ri], r_min, r_max))
            r_parts.append(np.full(np_r, rr))
            phi_parts.append(rng.uniform(0.0, 2.0 * np.pi, np_r))
        r = np.concatenate(r_parts) if r_parts else r_fib
        phi = np.concatenate(phi_parts) if phi_parts else phi_fib

    elif alg == "phi_sectors":
        # Concentrate points into n_sectors azimuthal wedges, r from fibonacci.
        # alpha=1 → narrow sectors; alpha=0 → sectors widen to fill full circle.
        n_sectors = params["n_sectors"]
        centers = params["sector_centers"]
        max_w = np.pi / max(n_sectors, 1)
        min_w = max_w * 0.1
        half_width = min_w + (1.0 - alpha) * (max_w - min_w)
        pts_per = [n_half // n_sectors + (1 if i < n_half % n_sectors else 0)
                   for i in range(n_sectors)]
        r_parts, phi_parts = [], []
        r_idx = 0
        for ci, nc in enumerate(pts_per):
            if nc == 0:
                continue
            pc = (centers[ci] + rng.uniform(-half_width, half_width, nc)) % (2.0 * np.pi)
            r_parts.append(r_fib[r_idx:r_idx + nc])
            phi_parts.append(pc)
            r_idx += nc
        r = np.concatenate(r_parts) if r_parts else r_fib
        phi = np.concatenate(phi_parts) if phi_parts else phi_fib

    elif alg == "clustered_surface":
        # Many small clusters whose centers are Fibonacci-distributed over the annulus.
        # The Fibonacci placement guarantees even global coverage regardless of n_clusters.
        # W2 diversity comes from two independent knobs:
        #   - n_clusters (random, drawn in _draw_surface_params): few → large gaps → high W2;
        #     many → surface covered with clusters → lower W2.
        #   - alpha: controls intra-cluster spread; alpha=1 → tight (highest W2 contribution),
        #     alpha=0 → widest-before-merge (lowest W2 contribution).
        # Hard separation: sigma is capped at half the minimum Fibonacci center distance so
        # that even at alpha=0 clusters remain geometrically distinct.

        n_clusters = min(params["n_clusters"], n_half)  # guard against tiny n_half

        # Fibonacci cluster centers on the annulus
        c_phi = _fib_angles(n_clusters)
        c_r = _fib_annulus_r(n_clusters, r_min, r_max)

        # Minimum pairwise Euclidean distance between cluster centers (flat disk)
        if n_clusters >= 2:
            cx = c_r * np.cos(c_phi)
            cy = c_r * np.sin(c_phi)
            diff_x = cx[:, None] - cx[None, :]   # (k, k)
            diff_y = cy[:, None] - cy[None, :]
            center_dists = np.sqrt(diff_x ** 2 + diff_y ** 2)
            np.fill_diagonal(center_dists, np.inf)
            min_center_dist = float(center_dists.min())
        else:
            min_center_dist = r_max - r_min  # single cluster: full radial extent

        # sigma range: tight (1.5 × PMT footprint) to hard-separation limit (half of min gap)
        sigma_min = float(PMT_RADIUS) * 1.5
        sigma_max = max(sigma_min * 1.01, min_center_dist * 0.5)
        sigma = sigma_min + (1.0 - alpha) * (sigma_max - sigma_min)

        # Distribute n_half points evenly across clusters
        pts_per = [n_half // n_clusters + (1 if i < n_half % n_clusters else 0)
                   for i in range(n_clusters)]
        r_parts, phi_parts = [], []
        for ci, nc in enumerate(pts_per):
            if nc == 0:
                continue
            # Uniform draw in a disk of radius sigma around the cluster center.
            # r_off ∈ [0, sigma] with uniform linear sampling (slightly edge-weighted),
            # matching the style of multi_cluster.
            r_off = rng.uniform(0.0, sigma, nc)
            a_off = rng.uniform(0.0, 2.0 * np.pi, nc)
            rc = np.clip(c_r[ci] + r_off * np.cos(a_off), r_min, r_max)
            ref_r = max(float(c_r[ci]), r_min + 1e-6)
            pc = (c_phi[ci] + r_off * np.sin(a_off) / ref_r) % (2.0 * np.pi)
            r_parts.append(rc)
            phi_parts.append(pc)
        r = np.concatenate(r_parts)
        phi = np.concatenate(phi_parts)

    else:
        raise ValueError(f"Unknown disk algorithm: '{alg}'")

    # Apply point symmetry: mirror each of the n_half points by phi + pi
    r_full = np.concatenate([r, r])
    phi_full = np.concatenate([phi, (phi + np.pi) % (2.0 * np.pi)])

    # Odd n: one extra point at radial midpoint (small deviation allowed)
    if n % 2:
        r_full = np.append(r_full, (r_min + r_max) / 2.0)
        phi_full = np.append(phi_full, float(rng.uniform(0.0, 2.0 * np.pi)))

    x = r_full * np.cos(phi_full)
    y = r_full * np.sin(phi_full)
    return np.column_stack([x, y, np.full(n, z)])


def _gen_wall_points(
    n: int,
    alg: str,
    params: dict,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (n, 3) geometric points on the cylindrical wall.

    All distributions are point-symmetric through the cylinder axis: each of the
    n//2 generated points at (phi, z) is paired with a mirror at (phi+pi, z).
    The odd remainder (if any) gets one extra point at the z midpoint.
    """
    z_min, z_max = _Z_WALL_MIN, _Z_WALL_MAX
    n_half = n // 2
    phi_fib = (2.0 * np.pi * np.arange(n_half) / _GOLDEN) % (2.0 * np.pi)
    z_fib = z_min + (np.arange(n_half) + 0.5) / n_half * (z_max - z_min)

    if alg == "fibonacci":
        phi, z = phi_fib, z_fib

    elif alg == "z_band":
        z_center = params["z_center"]
        h_min = float(PMT_RADIUS)
        h_max = (z_max - z_min) / 2.0
        h_band = h_min + (1.0 - alpha) * (h_max - h_min)
        z_norm = (z_fib - z_min) / (z_max - z_min)
        z = np.clip(z_center + (z_norm - 0.5) * 2.0 * h_band, z_min, z_max)
        phi = phi_fib

    elif alg == "multi_cluster":
        k = params["k"]
        c_phi = params["cluster_phi"]
        c_z = params["cluster_z"]
        sig_phi_min = 2.0 * np.pi / max(n_half, 1) * 3.0
        sig_phi_max = max(sig_phi_min * 2.0, np.pi / max(k, 1) * 0.7)
        sig_z_min = float(PMT_RADIUS)
        sig_z_max = max(sig_z_min * 2.0, (z_max - z_min) / (2.0 * max(k, 1)) * 0.7)
        sig_phi = sig_phi_min + (1.0 - alpha) * (sig_phi_max - sig_phi_min)
        sig_z = sig_z_min + (1.0 - alpha) * (sig_z_max - sig_z_min)
        pts_per = [n_half // k + (1 if i < n_half % k else 0) for i in range(k)]
        phi_parts, z_parts = [], []
        for ci, nc in enumerate(pts_per):
            if nc == 0:
                continue
            pc = (c_phi[ci] + rng.normal(0.0, sig_phi, nc)) % (2.0 * np.pi)
            zc = np.clip(c_z[ci] + rng.normal(0.0, sig_z, nc), z_min, z_max)
            phi_parts.append(pc)
            z_parts.append(zc)
        phi = np.concatenate(phi_parts) if phi_parts else phi_fib
        z = np.concatenate(z_parts) if z_parts else z_fib

    elif alg == "superposition":
        # Concentrated part is a horizontal band at conc_z with uniform phi — no preferred direction.
        n_conc = int(round(alpha * n_half))
        n_uni = n_half - n_conc
        c_z = np.clip(params["conc_z"], z_min, z_max)
        blob_z = max(float(PMT_RADIUS) * 1.5,
                     (z_max - z_min) / max(float(np.sqrt(n_half)), 1.0))
        phi_parts, z_parts = [], []
        if n_uni > 0:
            phi_parts.append(phi_fib[:n_uni])
            z_parts.append(z_fib[:n_uni])
        if n_conc > 0:
            pc = rng.uniform(0.0, 2.0 * np.pi, n_conc)
            zc = np.clip(c_z + rng.normal(0.0, blob_z, n_conc), z_min, z_max)
            phi_parts.append(pc)
            z_parts.append(zc)
        phi = np.concatenate(phi_parts) if phi_parts else phi_fib
        z = np.concatenate(z_parts) if z_parts else z_fib

    elif alg == "multi_ring":
        # Multiple horizontal rings at fixed z levels, each with uniform phi.
        # Few rings (high alpha) → concentrated; many rings → spread.
        n_rings = params["n_rings"]
        ring_z = params["ring_z"]
        pts_per_ring = [n_half // n_rings + (1 if i < n_half % n_rings else 0)
                        for i in range(n_rings)]
        phi_parts, z_parts = [], []
        for ri, np_r in enumerate(pts_per_ring):
            if np_r == 0:
                continue
            zz = float(np.clip(ring_z[ri], z_min, z_max))
            phi_parts.append(rng.uniform(0.0, 2.0 * np.pi, np_r))
            z_parts.append(np.full(np_r, zz))
        phi = np.concatenate(phi_parts) if phi_parts else phi_fib
        z = np.concatenate(z_parts) if z_parts else z_fib

    elif alg == "phi_sectors":
        # Same logic as disk phi_sectors but z from fibonacci instead of r.
        n_sectors = params["n_sectors"]
        centers = params["sector_centers"]
        max_w = np.pi / max(n_sectors, 1)
        min_w = max_w * 0.1
        half_width = min_w + (1.0 - alpha) * (max_w - min_w)
        pts_per = [n_half // n_sectors + (1 if i < n_half % n_sectors else 0)
                   for i in range(n_sectors)]
        phi_parts, z_parts = [], []
        z_idx = 0
        for ci, nc in enumerate(pts_per):
            if nc == 0:
                continue
            pc = (centers[ci] + rng.uniform(-half_width, half_width, nc)) % (2.0 * np.pi)
            phi_parts.append(pc)
            z_parts.append(z_fib[z_idx:z_idx + nc])
            z_idx += nc
        phi = np.concatenate(phi_parts) if phi_parts else phi_fib
        z = np.concatenate(z_parts) if z_parts else z_fib

    elif alg == "phi_band":
        # Concentrate all points in a narrow azimuthal stripe; z stays uniform.
        # After point-symmetry mirroring, two stripes appear (phi_center and phi_center+π).
        # alpha=1 → very narrow band; alpha=0 → band widens to half the circle.
        phi_center = params["phi_center"]
        min_w = float(PMT_RADIUS) / float(R_ZYLINDER)   # ~one PMT angular size
        max_w = np.pi / 2.0                               # quarter-circle half-width
        half_width = min_w + (1.0 - alpha) * (max_w - min_w)
        phi = (phi_center + rng.uniform(-half_width, half_width, n_half)) % (2.0 * np.pi)
        z = z_fib

    elif alg == "pole":
        # Concentrate points near the top or bottom edge of the wall.
        # alpha=1 → tightly packed at the edge; alpha=0 → spread over half the wall height.
        pole_pos = params["pole_pos"]
        h_full = z_max - z_min
        h_min = float(PMT_RADIUS) * 2.0
        h_pole = h_min + (1.0 - alpha) * (h_full / 2.0 - h_min)
        t = (np.arange(n_half) + 0.5) / n_half   # uniform [0, 1) spacing
        if pole_pos == "bottom":
            z = z_min + t * h_pole
        else:
            z = z_max - t * h_pole
        phi = phi_fib

    elif alg == "clustered_surface":
        # Analogue of the disk clustered_surface algorithm for the cylindrical wall.
        # Cluster centers: Fibonacci phi + uniformly-spaced z — covers the cylinder evenly.
        # Intra-cluster offsets are drawn in the (arc-length, z) plane so that sigma
        # has consistent physical units (mm) on both axes.

        n_clusters = min(params["n_clusters"], n_half)

        # Cluster centers: Fibonacci azimuth + uniform z
        c_phi = (2.0 * np.pi * np.arange(n_clusters) / _GOLDEN) % (2.0 * np.pi)
        c_z = z_min + (np.arange(n_clusters) + 0.5) / max(n_clusters, 1) * (z_max - z_min)

        # Minimum pairwise distance between cluster centers in 3D Cartesian space.
        # Using chord distance (≤ arc-length) is conservative: sigma_max is slightly
        # smaller than it would be under arc-length, so the hard-separation guarantee
        # holds with margin.
        if n_clusters >= 2:
            cx = float(R_ZYLINDER) * np.cos(c_phi)
            cy = float(R_ZYLINDER) * np.sin(c_phi)
            diff_x = cx[:, None] - cx[None, :]
            diff_y = cy[:, None] - cy[None, :]
            diff_z = c_z[:, None] - c_z[None, :]
            center_dists = np.sqrt(diff_x ** 2 + diff_y ** 2 + diff_z ** 2)
            np.fill_diagonal(center_dists, np.inf)
            min_center_dist = float(center_dists.min())
        else:
            min_center_dist = z_max - z_min

        sigma_min = float(PMT_RADIUS) * 1.5
        sigma_max = max(sigma_min * 1.01, min_center_dist * 0.5)
        sigma = sigma_min + (1.0 - alpha) * (sigma_max - sigma_min)

        pts_per = [n_half // n_clusters + (1 if i < n_half % n_clusters else 0)
                   for i in range(n_clusters)]
        phi_parts, z_parts = [], []
        for ci, nc in enumerate(pts_per):
            if nc == 0:
                continue
            # Uniform disk in (arc-length, z) space: r_off is the spatial radius,
            # converted to Δphi via Δphi = r_off * cos(a) / R.
            r_off = rng.uniform(0.0, sigma, nc)
            a_off = rng.uniform(0.0, 2.0 * np.pi, nc)
            pc = (c_phi[ci] + r_off * np.cos(a_off) / float(R_ZYLINDER)) % (2.0 * np.pi)
            zc = np.clip(c_z[ci] + r_off * np.sin(a_off), z_min, z_max)
            phi_parts.append(pc)
            z_parts.append(zc)
        phi = np.concatenate(phi_parts) if phi_parts else phi_fib
        z = np.concatenate(z_parts) if z_parts else z_fib

    else:
        raise ValueError(f"Unknown wall algorithm: '{alg}'")

    # Apply point symmetry: mirror each of the n_half points by phi + pi
    phi_full = np.concatenate([phi, (phi + np.pi) % (2.0 * np.pi)])
    z_full = np.concatenate([z, z])

    # Odd n: one extra point at z midpoint (small deviation allowed)
    if n % 2:
        phi_full = np.append(phi_full, float(rng.uniform(0.0, 2.0 * np.pi)))
        z_full = np.append(z_full, (z_min + z_max) / 2.0)

    x = R_ZYLINDER * np.cos(phi_full)
    y = R_ZYLINDER * np.sin(phi_full)
    return np.column_stack([x, y, z_full])


def _gen_area_points(
    area: str,
    n: int,
    surface_params: dict,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    alg = surface_params["algorithm"]
    if area == "wall":
        return _gen_wall_points(n, alg, surface_params, alpha, rng)
    r_min, r_max = _area_r_bounds(area)
    z = _area_z(area)
    return _gen_disk_points(n, r_min, r_max, z, alg, surface_params, alpha, rng)


# ---------------------------------------------------------------------------
# Surface-parameter drawing
# ---------------------------------------------------------------------------

def _draw_surface_params(
    area: str,
    n_area: int,
    rng: np.random.Generator,
    allowed_algorithms: list[str] | None = None,
) -> dict:
    """Draw random algorithm + geometric starting parameters for one area.

    All algorithms are designed to produce point-symmetric distributions:
    the generator uses n//2 points and mirrors them at phi+pi.  Parameters
    therefore need no special phi symmetry constraints — the mirroring in
    _gen_disk_points/_gen_wall_points handles that automatically.

    Parameters
    ----------
    allowed_algorithms : list of str, optional
        If given, restrict selection to algorithms that appear in both this list
        and the area's natural pool.  If the intersection is empty the full pool
        is used as fallback so generation never fails silently.
    """
    pool = _ALG_WALL if area == "wall" else _ALG_DISK
    if allowed_algorithms is not None:
        filtered = [a for a in pool if a in allowed_algorithms]
        if filtered:
            pool = filtered
        # else: silently fall back to the full pool
    alg = str(rng.choice(pool))
    p: dict = {"algorithm": alg}

    if alg == "z_band":                  # wall only
        z_range = _Z_WALL_MAX - _Z_WALL_MIN
        margin = z_range * 0.25
        p["z_center"] = float(rng.uniform(_Z_WALL_MIN + margin, _Z_WALL_MAX - margin))

    elif alg == "radial":
        p["radial_direction"] = str(rng.choice(["inward", "outward"]))
        p["beta_max"] = float(rng.uniform(2.0, 6.0))

    elif alg == "multi_cluster":
        # k clusters drawn with random phi and r/z — the generator uses only
        # n//2 points and mirrors them, so any phi placement becomes symmetric.
        n_half = max(1, n_area // 2)
        max_k = max(1, min(6, n_half // 5))
        k = int(rng.integers(1, max_k + 1))
        p["k"] = k
        p["cluster_phi"] = rng.uniform(0.0, 2.0 * np.pi, k).tolist()
        if area == "wall":
            p["cluster_z"] = rng.uniform(
                _Z_WALL_MIN + float(PMT_RADIUS),
                _Z_WALL_MAX - float(PMT_RADIUS),
                k,
            ).tolist()
        else:
            r_min, r_max = _area_r_bounds(area)
            p["cluster_r"] = rng.uniform(r_min, r_max, k).tolist()

    elif alg == "superposition":
        # No conc_phi: concentrated part uses uniform phi (ring/band), making
        # the half-distribution already azimuthally unbiased before mirroring.
        if area == "wall":
            p["conc_z"] = float(rng.uniform(
                _Z_WALL_MIN + float(PMT_RADIUS),
                _Z_WALL_MAX - float(PMT_RADIUS),
            ))
        else:
            r_min, r_max = _area_r_bounds(area)
            p["conc_r"] = float(rng.uniform(r_min, r_max))

    elif alg == "multi_ring":
        # n_rings rings at independent r (disk) or z (wall) positions.
        # Fewer rings → more radially/vertically concentrated (higher W2).
        n_half = max(1, n_area // 2)
        max_rings = max(1, min(8, n_half // 4))
        n_rings = int(rng.integers(1, max_rings + 1))
        p["n_rings"] = n_rings
        if area == "wall":
            p["ring_z"] = rng.uniform(
                _Z_WALL_MIN + float(PMT_RADIUS),
                _Z_WALL_MAX - float(PMT_RADIUS),
                n_rings,
            ).tolist()
        else:
            r_min, r_max = _area_r_bounds(area)
            p["ring_radii"] = rng.uniform(r_min, r_max, n_rings).tolist()

    elif alg == "phi_sectors":
        n_half = max(1, n_area // 2)
        max_k = max(1, min(3, n_half // 3))
        n_sectors = int(rng.integers(1, max_k + 1))
        p["n_sectors"] = n_sectors
        p["sector_centers"] = rng.uniform(0.0, 2.0 * np.pi, n_sectors).tolist()

    elif alg == "phi_band":   # wall only
        p["phi_center"] = float(rng.uniform(0.0, 2.0 * np.pi))

    elif alg == "pole":       # wall only
        p["pole_pos"] = str(rng.choice(["top", "bottom"]))

    elif alg == "clustered_surface":
        # n_clusters: number of Fibonacci-distributed cluster centers (for one half; mirrored
        # to 2*n_clusters after C2 symmetry).  Range 2..max ensures at least 2 points per
        # cluster on average and keeps max clusters physically sensible.
        n_half = max(1, n_area // 2)
        max_clusters = max(2, min(n_half // 2, 40))
        n_clusters = int(rng.integers(2, max_clusters + 1))
        p["n_clusters"] = n_clusters

    # Ensure all list-valued params are plain Python lists for JSON serialisation
    for key in ("cluster_phi", "cluster_z", "cluster_r", "ring_z", "ring_radii",
                "sector_centers"):
        if key in p and not isinstance(p[key], list):
            p[key] = list(p[key])

    return p


# ---------------------------------------------------------------------------
# Voxel snapping with spacing enforcement and round-robin overflow
# ---------------------------------------------------------------------------

def _snap_to_voxels(
    geo_pts_by_area: dict[str, np.ndarray],
    all_centers: np.ndarray,
    all_layers: np.ndarray,
    N_by_area: dict[str, int],
    min_spacing: float,
) -> tuple[list[int], bool]:
    """
    Snap geometric points per area to nearest valid voxels with spacing.

    Returns
    -------
    selected_cols : list[int]
    used_overflow : bool   — True if the round-robin workaround was triggered.
    """
    min_sq = min_spacing ** 2

    # Build per-area index arrays and KD-trees
    area_col_indices: dict[str, np.ndarray] = {}
    area_trees: dict[str, KDTree | None] = {}
    for a in _AREA_ORDER:
        idx = np.where(all_layers == a)[0]
        area_col_indices[a] = idx
        area_trees[a] = KDTree(all_centers[idx]) if len(idx) > 0 else None

    # Tracking selected state
    selected: list[int] = []
    selected_set: set[int] = set()
    # Per-layer list of selected centers for spacing checks (fast path)
    sel_by_layer: dict[str, list[np.ndarray]] = {a: [] for a in _AREA_ORDER}

    def _spacing_ok(col: int) -> bool:
        c = all_centers[col]
        layer = str(all_layers[col])
        for sc in sel_by_layer[layer]:
            diff = c - sc
            if float(diff @ diff) < min_sq:
                return False
        return True

    def _try_place(geo_pt: np.ndarray, area: str, k_query: int = 300) -> int | None:
        tree = area_trees.get(area)
        if tree is None:
            return None
        idxs = area_col_indices[area]
        k = min(k_query, len(idxs))
        _, nn_local = tree.query(geo_pt, k=k)
        candidates = [nn_local] if k == 1 else nn_local
        for li in candidates:
            col = int(idxs[li])
            if col in selected_set:
                continue
            if _spacing_ok(col):
                return col
        return None

    def _accept(col: int) -> None:
        selected.append(col)
        selected_set.add(col)
        layer = str(all_layers[col])
        sel_by_layer[layer].append(all_centers[col].copy())

    # Primary placement: iterate areas in order
    used_overflow = False
    deficits: list[tuple[str, int]] = []   # (area, n_missing) pairs

    for area in _AREA_ORDER:
        n_target = N_by_area.get(area, 0)
        if n_target == 0:
            continue
        geo_pts = geo_pts_by_area.get(area, np.empty((0, 3)))
        placed = 0
        for pt in geo_pts:
            if placed >= n_target:
                break
            col = _try_place(pt, area)
            if col is not None:
                _accept(col)
                placed += 1
        if placed < n_target:
            deficit = n_target - placed
            deficits.append((area, deficit))
            warnings.warn(
                f"[sample_w2_range] Area '{area}' placed {placed}/{n_target} voxels. "
                f"Redirecting {deficit} to other areas (round-robin).",
                stacklevel=4,
            )
            used_overflow = True

    # Round-robin overflow: fill deficits from any available area
    if deficits:
        total_deficit = sum(d for _, d in deficits)
        # Cycle through all areas (any area can absorb overflow)
        area_cycle = [a for a in _AREA_ORDER if area_col_indices[a].size > 0]
        cycle_idx = 0
        for _ in range(total_deficit):
            placed = False
            for _attempt in range(len(area_cycle)):
                overflow_area = area_cycle[cycle_idx % len(area_cycle)]
                cycle_idx += 1
                # Find any available, spaced voxel in this area
                for col in area_col_indices[overflow_area]:
                    col = int(col)
                    if col in selected_set:
                        continue
                    if _spacing_ok(col):
                        _accept(col)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                raise RuntimeError(
                    "[sample_w2_range] Ran out of valid, spaced voxels across ALL "
                    "areas. Cannot place all N PMTs. Try reducing N or min_spacing."
                )

    return selected, used_overflow


# ---------------------------------------------------------------------------
# Coverage computation
# ---------------------------------------------------------------------------

def _compute_efficiency(
    selected_cols: list[int],
    B,  # scipy.sparse.csc_matrix
    M: int,
) -> float:
    from scipy import sparse
    coverage = np.zeros(B.shape[0], dtype=np.int16)
    for col in selected_cols:
        s, e = B.indptr[col], B.indptr[col + 1]
        coverage[B.indices[s:e]] += 1
    return float(np.sum(coverage >= M)) / B.shape[0]


# ---------------------------------------------------------------------------
# JSON output (standard selected_voxels format + W2 metadata)
# ---------------------------------------------------------------------------

def _write_config_json(
    hdf5_path: str,
    voxel_ids: np.ndarray,
    selected_cols: list[int],
    efficiency: float,
    num_ncs: int,
    num_primaries: int,
    N: int,
    M: int,
    m: int,
    area_ratios: dict,
    min_spacing: float,
    included_areas: list[str],
    surface_params: dict,
    alpha_by_area: dict[str, float],
    w2: float | None,
    w2_lo: float | None,
    w2_hi: float | None,
    output_path: Path,
) -> None:
    selected_voxels_json = []
    with h5py.File(hdf5_path, "r") as f:
        for col_idx in selected_cols:
            vid = voxel_ids[col_idx]
            center = f[f"voxels/{vid}/center"][:].tolist()
            corners_x = f[f"voxels/{vid}/corners/x"][:].tolist()
            corners_y = f[f"voxels/{vid}/corners/y"][:].tolist()
            corners_z = f[f"voxels/{vid}/corners/z"][:].tolist()
            corners = [[x, y, z]
                       for x, y, z in zip(corners_x, corners_y, corners_z)]
            layer_raw = f[f"voxels/{vid}/layer"][()]
            layer = (layer_raw.decode() if isinstance(layer_raw, bytes)
                     else str(layer_raw))
            selected_voxels_json.append({
                "index": vid,
                "center": center,
                "corners": corners,
                "layer": layer,
            })

    json_data = {
        "config": {
            "optimize": "nc",
            "N": N, "M": M, "m": m, "W": None,
            "area_ratios": area_ratios,
            "min_spacing": min_spacing,
            "included_areas": included_areas,
            "algorithms": {a: surface_params[a]["algorithm"] for a in _AREA_ORDER},
            "alpha_by_area": alpha_by_area,
            "w2": w2,
            "w2_probe_lo": w2_lo,
            "w2_probe_hi": w2_hi,
            "generator": "sample_w2_range",
        },
        "efficiency": efficiency,
        "num_ncs": num_ncs,
        "num_primaries": num_primaries,
        "selected_voxels": selected_voxels_json,
    }
    with open(output_path, "w") as jf:
        json.dump(json_data, jf, indent=2)


# ---------------------------------------------------------------------------
# Main sampling logic
# ---------------------------------------------------------------------------

def sample_w2_range(
    hdf5_path: str,
    N: int,
    M: int,
    m: int,
    n_configs: int,
    seed: int,
    area_ratios: dict,
    min_spacing: float,
    output_dir: Path,
    sensitivity: bool,
    deltas: list[float] | None,
    verbose: bool,
    exclude_areas: list[str] | None = None,
    allowed_algorithms: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    main_rng = np.random.default_rng(seed)
    w2_ref = get_w2_ref()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 65)
        print("Loading simulation data")
        print("=" * 65)

    (raw_rows, raw_cols, raw_vals,
     voxel_ids, all_centers, all_layers,
     num_ncs, num_primaries) = load_raw_sparse(hdf5_path, verbose=verbose)
    # Keep raw sparse data for optional sensitivity analysis
    _raw_rows, _raw_cols, _raw_vals = raw_rows, raw_cols, raw_vals

    B = binarize_from_raw(
        raw_rows, raw_cols, raw_vals,
        num_ncs=num_ncs,
        num_voxels=len(voxel_ids),
        layers=all_layers,
        area_ratios=area_ratios,
        m=m,
        seed=seed,
    )
    if verbose:
        print(f"B matrix: {num_ncs} × {len(voxel_ids)}, nnz={B.nnz:,}")

    # ------------------------------------------------------------------
    # Generate configs
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n{'=' * 65}")
        print(f"Generating {n_configs} W2-spanning configurations")
        print(f"{'=' * 65}")

    results: list[dict] = []

    for cfg_idx in range(n_configs):
        t0 = time.time()

        # Separate seed stream for each config so probe runs do not pollute
        # the main rng or each other.
        cfg_seed = int(main_rng.integers(0, 2 ** 31))

        params_rng = np.random.default_rng([cfg_seed, 0])
        probe_rng_lo = np.random.default_rng([cfg_seed, 1])
        probe_rng_hi = np.random.default_rng([cfg_seed, 2])
        actual_rng = np.random.default_rng([cfg_seed, 3])

        # ---- Draw included areas (at least one; each included with p=0.65) ----
        allowed_areas = [a for a in _AREA_ORDER if a not in (exclude_areas or [])]
        included = {a: bool(params_rng.random() < 0.65) for a in allowed_areas}
        if not any(included.values()):
            included[str(params_rng.choice(allowed_areas))] = True
        included_areas = [a for a in allowed_areas if included[a]]

        # ---- N allocation across included areas ----
        N_by_area = compute_per_area_N(N, areas=included_areas, verbose=False)

        # ---- Draw surface-specific algorithm + geometric parameters ----
        surface_params: dict[str, dict] = {}
        for a in _AREA_ORDER:
            n_a = N_by_area.get(a, 0)
            if n_a > 0:
                surface_params[a] = _draw_surface_params(
                    a, n_a, params_rng, allowed_algorithms=allowed_algorithms
                )
            else:
                surface_params[a] = {"algorithm": "fibonacci"}

        # ---- Draw per-area alpha independently ----
        alpha_by_area = {a: float(main_rng.uniform(0.0, 1.0)) for a in included_areas}

        # ---- Inner helper: run one alpha-per-area dict ----
        def _run(alpha_map: dict[str, float], rng: np.random.Generator) -> tuple[list[int], float | None, bool]:
            geo_pts_by_area: dict[str, np.ndarray] = {}
            for a in _AREA_ORDER:
                n_a = N_by_area.get(a, 0)
                if n_a > 0:
                    geo_pts_by_area[a] = _gen_area_points(
                        a, n_a, surface_params[a], alpha_map[a], rng
                    )
            sel, ovf = _snap_to_voxels(
                geo_pts_by_area, all_centers, all_layers, N_by_area, min_spacing,
            )
            if len(sel) < 2:
                return sel, None, ovf
            w2 = compute_wasserstein_homogeneity(
                all_centers[np.array(sel)], reference=w2_ref
            )["w2"]
            return sel, w2, ovf

        # ---- Probe runs for W2 bounds (all areas at 0 / all at 1) ----
        _, w2_lo, _ = _run({a: 0.0 for a in included_areas}, probe_rng_lo)
        _, w2_hi, _ = _run({a: 1.0 for a in included_areas}, probe_rng_hi)

        if w2_lo is None or w2_hi is None:
            if verbose:
                print(f"  config_{cfg_idx:03d}: skipped (too few valid voxels)")
            continue

        # ---- Actual run ----
        sel_cols, w2_val, used_ovf = _run(alpha_by_area, actual_rng)

        if used_ovf:
            warnings.warn(
                f"[config_{cfg_idx:03d}] Bot-area overflow workaround was triggered.",
                stacklevel=1,
            )

        eff = _compute_efficiency(sel_cols, B, M)

        # ---- Save JSON ----
        json_path = output_dir / f"config_{cfg_idx:03d}.json"
        _write_config_json(
            hdf5_path=hdf5_path,
            voxel_ids=voxel_ids,
            selected_cols=sel_cols,
            efficiency=eff,
            num_ncs=num_ncs,
            num_primaries=num_primaries,
            N=N, M=M, m=m,
            area_ratios=area_ratios,
            min_spacing=min_spacing,
            included_areas=included_areas,
            surface_params=surface_params,
            alpha_by_area=alpha_by_area,
            w2=w2_val,
            w2_lo=w2_lo,
            w2_hi=w2_hi,
            output_path=json_path,
        )

        # ---- Save PNG ----
        sel_arr = np.array(sel_cols)
        w2_s = f"{w2_val:.1f}" if w2_val is not None else "N/A"
        alpha_avg = sum(alpha_by_area.values()) / len(alpha_by_area)
        alg_tag = "|".join(surface_params[a]["algorithm"][:3] for a in included_areas)
        plot_selected_voxels(
            all_centers[sel_arr],
            all_layers[sel_arr],
            [str(voxel_ids[c]) for c in sel_cols],
            output_path=output_dir / f"config_{cfg_idx:03d}.png",
            title_extra=(
                f"W2={w2_s} mm  eff={eff:.4%}  α_avg={alpha_avg:.2f}"
                f"  [{alg_tag}]"
            ),
        )

        elapsed = time.time() - t0
        results.append({
            "name": f"config_{cfg_idx:03d}",
            "w2": w2_val,
            "w2_probe_lo": w2_lo,
            "w2_probe_hi": w2_hi,
            "alpha_by_area": alpha_by_area,
            "efficiency": eff,
            "included_areas": included_areas,
            "algorithms": {a: surface_params[a]["algorithm"] for a in _AREA_ORDER},
            "used_overflow": used_ovf,
            "selected": sel_cols,
        })

        if verbose:
            alpha_str = " ".join(f"{a}:{v:.2f}" for a, v in alpha_by_area.items())
            print(
                f"  config_{cfg_idx:03d}  W2={w2_s:>8} mm"
                f"  [probe {w2_lo:.1f}–{w2_hi:.1f}]"
                f"  eff={eff:.4%}  α=[{alpha_str}]"
                f"  areas={included_areas}"
                f"  ({elapsed:.1f}s)"
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary_txt = output_dir / "config_summary.txt"
    with open(summary_txt, "w") as sf:
        sf.write(f"# W2-range sampling\n")
        sf.write(f"# N={N}, M={M}, m={m}, seed={seed}, n_configs={n_configs}\n\n")
        hdr = (f"{'Config':<14} {'W2 mm':>9} {'W2_lo':>7} {'W2_hi':>7}"
               f" {'α_avg':>6} {'Efficiency':>12}  Areas / Algorithms\n")
        sf.write(hdr)
        sf.write("-" * 90 + "\n")
        for r in results:
            w2_s = f"{r['w2']:.1f}" if r["w2"] is not None else "N/A"
            alpha_avg = sum(r["alpha_by_area"].values()) / len(r["alpha_by_area"])
            alg_str = "  ".join(
                f"{a}:{r['algorithms'][a][:3]}" for a in r["included_areas"]
            )
            sf.write(
                f"{r['name']:<14} {w2_s:>9} {r['w2_probe_lo']:>7.1f} {r['w2_probe_hi']:>7.1f}"
                f" {alpha_avg:>6.3f} {r['efficiency']:>12.4%}  {alg_str}\n"
            )

    summary_json = output_dir / "config_summary.json"
    with open(summary_json, "w") as jf:
        json.dump(results, jf, indent=2)

    if verbose:
        print(f"\nSummary written to {summary_txt}")
        print(f"Summary JSON   → {summary_json}")
        w2_vals = [r["w2"] for r in results if r["w2"] is not None]
        if w2_vals:
            print(f"W2 range achieved: {min(w2_vals):.1f} – {max(w2_vals):.1f} mm"
                  f"  (n={len(w2_vals)} configs)")

    # ------------------------------------------------------------------
    # Optional: sensitivity analysis for every generated config
    # ------------------------------------------------------------------
    if sensitivity:
        if verbose:
            print("\n" + "=" * 65)
            print("Running sensitivity analysis for all configs")
            print("=" * 65)
        for r in results:
            if verbose:
                print(f"\n  -- {r['name']} (eff={r['efficiency']:.4%}) --")
            run_sensitivity(
                filepath=hdf5_path,
                N=N,
                m=m,
                area_ratios=area_ratios,
                optimize="nc",
                M=M,
                min_spacing=min_spacing,
                seed=seed,
                deltas=deltas,
                output_dir=str(output_dir / "sensitivity" / r["name"]),
                baseline_selected=r["selected"],
                baseline_eff=r["efficiency"],
                raw_rows=_raw_rows,
                raw_cols=_raw_cols,
                raw_vals=_raw_vals,
                voxel_ids=voxel_ids,
                centers=all_centers,
                layers=all_layers,
                num_ncs=num_ncs,
                num_primaries=num_primaries,
                verbose=verbose,
            )

    if verbose:
        print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate n_configs PMT selections spanning the W2 homogeneity range "
            "using geometry-driven clustering algorithms (no greedy selection)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hdf5", type=str, required=True,
                        help="Path to raw SSD HDF5 data file.")
    parser.add_argument("-N", type=int, required=True,
                        help="Number of PMTs per configuration.")
    parser.add_argument("-M", type=int, default=1,
                        help="Multiplicity threshold for NC efficiency.")
    parser.add_argument("-m", type=int, default=1,
                        help="Per-voxel hit threshold for binarisation.")
    parser.add_argument("--n-configs", type=int, default=50,
                        help="Total number of configurations to generate.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master random seed.")
    parser.add_argument("--min-spacing", type=float, default=2 * PMT_RADIUS,
                        help="Minimum inter-voxel spacing on the same layer (mm).")
    parser.add_argument("--output-dir", type=str, default="w2_setups",
                        help="Output directory.")
    parser.add_argument("--pit",  type=float, default=None)
    parser.add_argument("--bot",  type=float, default=None)
    parser.add_argument("--top",  type=float, default=None)
    parser.add_argument("--wall", type=float, default=None)
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run sensitivity analysis on every generated config.")
    parser.add_argument("--deltas", type=str, default=None,
                        help="Comma-separated area-ratio perturbation values for "
                             "sensitivity analysis (e.g. '-0.10,0.10'). "
                             "Default: -0.20,-0.10,-0.05,+0.05,+0.10,+0.20.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--exclude-areas", nargs="+", default=[],
        choices=["pit", "bot", "top", "wall"], metavar="AREA",
        help="Areas to exclude from voxel selection (e.g. --exclude-areas pit bot).",
    )
    all_algs = sorted(set(_ALG_DISK) | set(_ALG_WALL))
    parser.add_argument(
        "--algorithms", nargs="+", default=None,
        metavar="ALG",
        choices=all_algs,
        help=(
            "Restrict algorithm selection to this subset, applied globally to all areas "
            "(e.g. --algorithms clustered_surface fibonacci).  Algorithms not valid for a "
            "given area type are silently ignored; if none match, the full pool is used. "
            f"Available: {', '.join(all_algs)}."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    area_ratios = dict(DEFAULT_AREA_RATIOS)
    if args.pit  is not None: area_ratios["pit"]  = args.pit
    if args.bot  is not None: area_ratios["bot"]  = args.bot
    if args.top  is not None: area_ratios["top"]  = args.top
    if args.wall is not None: area_ratios["wall"] = args.wall

    exclude_areas = args.exclude_areas or []
    if set(exclude_areas) >= set(_AREA_ORDER):
        raise ValueError("--exclude-areas cannot exclude all areas.")

    deltas = None
    if args.deltas is not None:
        deltas = [float(d) for d in args.deltas.split(",")]

    sample_w2_range(
        hdf5_path=args.hdf5,
        N=args.N,
        M=args.M,
        m=args.m,
        n_configs=args.n_configs,
        seed=args.seed,
        area_ratios=area_ratios,
        min_spacing=args.min_spacing,
        output_dir=Path(args.output_dir),
        sensitivity=args.sensitivity,
        deltas=deltas,
        verbose=not args.quiet,
        exclude_areas=exclude_areas or None,
        allowed_algorithms=args.algorithms or None,
    )


if __name__ == "__main__":
    main()
