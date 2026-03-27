#!/usr/bin/env python3
"""
Sanity tests for the Wasserstein homogeneity metric.

No HDF5 data or voxel snapping — all configurations are built analytically
from Fibonacci spirals and geometric primitives.  This isolates the metric
itself from discretisation artefacts.

Run:
    cd tests && python -m unittest test_wasserstein -v

Test groups
-----------
TestSampleReferenceDistribution
    Shape, per-area geometric bounds, areas-filter, area-proportional count.

TestWassersteinHomogeneity
    - ValueError for < 2 points
    - Return-dict keys present and non-negative
    - Result is reusable when a pre-computed reference is passed

TestWassersteinSanityChecks
    Three cases that MUST hold for the metric to be useful:
    1. Global region imbalance: all-on-wall has higher W2 than Fibonacci.
    2. Large-scale clustering: two tight clusters have higher W2 than Fibonacci.
    3. Rotation invariance: rotating Fibonacci by 60° leaves W2 within 5%.

TestWassersteinWithBaseline
    compute_wasserstein_homogeneity_with_baseline:
    - Fibonacci vs itself → w2_normalized == 1.0 exactly.
    - All-on-wall vs Fibonacci baseline → w2_normalized > 1.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pmtopt.homogeneous import (
    DZ_PIT,
    H_CYLINDER,
    T_ZYLINDER,
    Z_BASE,
    allocate_N_per_area,
    compute_wasserstein_homogeneity,
    compute_wasserstein_homogeneity_with_baseline,
    fibonacci_cylinder_wall,
    fibonacci_disk,
    sample_reference_distribution,
)
from pmtopt.geometry import (
    PMT_RADIUS,
    R_PIT,
    R_ZYL_BOT,
    R_ZYL_TOP,
    R_ZYLINDER,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

N      = 300
M_REF  = 3000
SEED   = 42

_Z_PIT      = float(Z_BASE + DZ_PIT / 2)
_Z_BOT      = float(Z_BASE + T_ZYLINDER / 2)
_Z_TOP      = float(Z_BASE + H_CYLINDER + T_ZYLINDER / 2)
_Z_WALL_MIN = float(Z_BASE)
_Z_WALL_MAX = float(Z_BASE + H_CYLINDER)

# Surface areas (mm²) — must mirror sample_reference_distribution
_SURFACE = {
    "pit":  np.pi * R_PIT**2,
    "bot":  np.pi * (R_ZYLINDER**2 - R_ZYL_BOT**2),
    "top":  np.pi * (R_ZYLINDER**2 - R_ZYL_TOP**2),
    "wall": 2.0 * np.pi * R_ZYLINDER * (_Z_WALL_MAX - _Z_WALL_MIN),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fibonacci_config(n: int, areas: list) -> np.ndarray:
    """N Fibonacci-spiral points distributed across the given areas (no snapping)."""
    alloc = allocate_N_per_area(n, areas)
    parts = []
    if "pit" in areas and alloc["pit"] > 0:
        pts = fibonacci_disk(alloc["pit"], 0.0, R_PIT - PMT_RADIUS)
        parts.append(np.column_stack([pts, np.full(alloc["pit"], _Z_PIT)]))
    if "bot" in areas and alloc["bot"] > 0:
        pts = fibonacci_disk(alloc["bot"], R_ZYL_BOT + PMT_RADIUS, R_ZYLINDER - PMT_RADIUS)
        parts.append(np.column_stack([pts, np.full(alloc["bot"], _Z_BOT)]))
    if "top" in areas and alloc["top"] > 0:
        pts = fibonacci_disk(alloc["top"], R_ZYL_TOP + PMT_RADIUS, R_ZYLINDER - PMT_RADIUS)
        parts.append(np.column_stack([pts, np.full(alloc["top"], _Z_TOP)]))
    if "wall" in areas and alloc["wall"] > 0:
        z_min = float(Z_BASE + PMT_RADIUS)
        z_max = float(Z_BASE + H_CYLINDER - PMT_RADIUS)
        parts.append(fibonacci_cylinder_wall(alloc["wall"], float(R_ZYLINDER), z_min, z_max))
    return np.vstack(parts)


def _rotate_z(centers: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate (x, y) columns by angle_rad; z unchanged."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    rot = centers.copy()
    rot[:, 0] = centers[:, 0] * c - centers[:, 1] * s
    rot[:, 1] = centers[:, 0] * s + centers[:, 1] * c
    return rot


def _expected_allocation(m: int, areas: list) -> dict:
    """Largest-remainder allocation matching sample_reference_distribution."""
    total = sum(_SURFACE[a] for a in areas)
    raw    = {a: m * _SURFACE[a] / total for a in areas}
    floors = {a: int(np.floor(raw[a])) for a in raw}
    rems   = {a: raw[a] - floors[a] for a in raw}
    left   = m - sum(floors.values())
    for a in sorted(rems, key=lambda x: rems[x], reverse=True)[:left]:
        floors[a] += 1
    return floors


def _classify_points(pts: np.ndarray) -> dict:
    """Classify rows of pts into detector areas by geometry.

    Classification rules (no overlap given detector geometry):
      wall : r ≈ R_ZYLINDER (within 1 mm) and z in [Z_WALL_MIN, Z_WALL_MAX]
      top  : z ≈ _Z_TOP (within 1 mm) and R_ZYL_TOP ≤ r ≤ R_ZYLINDER
      pit  : z ≈ _Z_PIT (within 1 mm) and r ≤ R_PIT
      bot  : z ≈ _Z_BOT (within 1 mm) and R_ZYL_BOT ≤ r ≤ R_ZYLINDER
    (pit and bot share the same z-level but differ in radial range.)
    """
    r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    z = pts[:, 2]

    mask_wall = np.abs(r - R_ZYLINDER) < 1.0
    mask_top  = (np.abs(z - _Z_TOP) < 1.0) & ~mask_wall
    mask_pit  = (np.abs(z - _Z_PIT) < 1.0) & (r <= R_PIT + 1.0) & ~mask_wall
    mask_bot  = (np.abs(z - _Z_BOT) < 1.0) & (r >= R_ZYL_BOT - 1.0) & ~mask_wall

    return {
        "wall": int(mask_wall.sum()),
        "top":  int(mask_top.sum()),
        "pit":  int(mask_pit.sum()),
        "bot":  int(mask_bot.sum()),
    }


# ===========================================================================
# TestSampleReferenceDistribution
# ===========================================================================

class TestSampleReferenceDistribution(unittest.TestCase):
    """Tests for sample_reference_distribution."""

    def test_shape_full(self):
        """Combined reference has exactly (M, 3) shape."""
        ref = sample_reference_distribution(M=M_REF, seed=SEED)
        self.assertEqual(ref.shape, (M_REF, 3))

    # --- geometric bounds per area ---

    def test_pit_bounds(self):
        ref = sample_reference_distribution(M=M_REF, seed=SEED, areas=["pit"])
        self.assertEqual(ref.shape, (M_REF, 3))
        r = np.sqrt(ref[:, 0]**2 + ref[:, 1]**2)
        z = ref[:, 2]
        self.assertTrue(np.all(r <= R_PIT + 1e-6),
                        f"pit: r_max={r.max():.2f} > R_PIT={R_PIT}")
        self.assertTrue(np.allclose(z, _Z_PIT, atol=1e-6),
                        f"pit: z not at _Z_PIT={_Z_PIT} (range [{z.min():.1f}, {z.max():.1f}])")

    def test_bot_bounds(self):
        ref = sample_reference_distribution(M=M_REF, seed=SEED, areas=["bot"])
        r = np.sqrt(ref[:, 0]**2 + ref[:, 1]**2)
        z = ref[:, 2]
        self.assertTrue(np.all(r >= R_ZYL_BOT - 1e-6),
                        f"bot: r_min={r.min():.2f} < R_ZYL_BOT={R_ZYL_BOT}")
        self.assertTrue(np.all(r <= R_ZYLINDER + 1e-6),
                        f"bot: r_max={r.max():.2f} > R_ZYLINDER={R_ZYLINDER}")
        self.assertTrue(np.allclose(z, _Z_BOT, atol=1e-6),
                        "bot: z not at correct level")

    def test_top_bounds(self):
        ref = sample_reference_distribution(M=M_REF, seed=SEED, areas=["top"])
        r = np.sqrt(ref[:, 0]**2 + ref[:, 1]**2)
        z = ref[:, 2]
        self.assertTrue(np.all(r >= R_ZYL_TOP - 1e-6),
                        f"top: r_min={r.min():.2f} < R_ZYL_TOP={R_ZYL_TOP}")
        self.assertTrue(np.all(r <= R_ZYLINDER + 1e-6),
                        f"top: r_max={r.max():.2f} > R_ZYLINDER={R_ZYLINDER}")
        self.assertTrue(np.allclose(z, _Z_TOP, atol=1e-6),
                        "top: z not at correct level")

    def test_wall_bounds(self):
        ref = sample_reference_distribution(M=M_REF, seed=SEED, areas=["wall"])
        r = np.sqrt(ref[:, 0]**2 + ref[:, 1]**2)
        z = ref[:, 2]
        self.assertTrue(np.allclose(r, R_ZYLINDER, atol=1e-6),
                        f"wall: r not equal to R_ZYLINDER={R_ZYLINDER} "
                        f"(max dev {np.abs(r - R_ZYLINDER).max():.2e})")
        self.assertTrue(np.all(z >= _Z_WALL_MIN - 1e-6),
                        f"wall: z_min={z.min():.2f} < _Z_WALL_MIN={_Z_WALL_MIN}")
        self.assertTrue(np.all(z <= _Z_WALL_MAX + 1e-6),
                        f"wall: z_max={z.max():.2f} > _Z_WALL_MAX={_Z_WALL_MAX}")

    # --- areas filter ---

    def test_areas_filter_returns_M_points(self):
        """Requesting a single area always returns exactly M points."""
        for area in ["pit", "bot", "top", "wall"]:
            ref = sample_reference_distribution(M=500, seed=SEED, areas=[area])
            self.assertEqual(ref.shape[0], 500,
                             f"area={area}: expected 500 points, got {ref.shape[0]}")

    def test_areas_filter_geometry_matches(self):
        """Points returned for a single area lie in the correct geometric region."""
        for area in ["pit", "bot", "top", "wall"]:
            ref = sample_reference_distribution(M=M_REF, seed=SEED, areas=[area])
            counts = _classify_points(ref)
            self.assertEqual(counts[area], M_REF,
                             f"area={area}: only {counts[area]}/{M_REF} points "
                             f"classified as '{area}'")

    # --- area-proportional allocation ---

    def test_allocation_total(self):
        """Full reference has exactly M points."""
        ref = sample_reference_distribution(M=M_REF, seed=SEED)
        self.assertEqual(len(ref), M_REF)

    def test_allocation_per_area_count(self):
        """Count of points per area matches the largest-remainder allocation exactly."""
        ref  = sample_reference_distribution(M=M_REF, seed=SEED)
        expected = _expected_allocation(M_REF, ["pit", "bot", "top", "wall"])
        observed = _classify_points(ref)
        for area in ["pit", "bot", "top", "wall"]:
            self.assertEqual(observed[area], expected[area],
                             f"{area}: expected {expected[area]}, got {observed[area]}")


# ===========================================================================
# TestWassersteinHomogeneity
# ===========================================================================

class TestWassersteinHomogeneity(unittest.TestCase):
    """Unit tests for compute_wasserstein_homogeneity."""

    def setUp(self):
        self.ref = sample_reference_distribution(M=M_REF, seed=SEED)
        self.config = _fibonacci_config(N, ["pit", "bot", "top", "wall"])

    def test_raises_for_fewer_than_2_points(self):
        with self.assertRaises(ValueError):
            compute_wasserstein_homogeneity(self.config[:1], reference=self.ref)

    def test_return_dict_keys(self):
        result = compute_wasserstein_homogeneity(self.config, reference=self.ref)
        for key in ("w2", "ot_cost", "n_config", "m_reference"):
            self.assertIn(key, result, f"missing key '{key}'")

    def test_w2_non_negative(self):
        result = compute_wasserstein_homogeneity(self.config, reference=self.ref)
        self.assertGreater(result["w2"], 0.0)

    def test_ot_cost_equals_w2_squared(self):
        result = compute_wasserstein_homogeneity(self.config, reference=self.ref)
        self.assertAlmostEqual(result["ot_cost"], result["w2"]**2, places=3)

    def test_counts_in_result(self):
        result = compute_wasserstein_homogeneity(self.config, reference=self.ref)
        self.assertEqual(result["n_config"], N)
        self.assertEqual(result["m_reference"], M_REF)

    def test_precomputed_reference_reused(self):
        """Passing pre-computed reference gives same W2 as computing from scratch."""
        r1 = compute_wasserstein_homogeneity(self.config, reference=self.ref)
        r2 = compute_wasserstein_homogeneity(self.config, M=M_REF, seed=SEED)
        self.assertAlmostEqual(r1["w2"], r2["w2"], places=1,
                               msg="W2 differs when reference is pre-computed vs recomputed")


# ===========================================================================
# TestWassersteinSanityChecks
# ===========================================================================

class TestWassersteinSanityChecks(unittest.TestCase):
    """The three fundamental properties the metric must satisfy."""

    @classmethod
    def setUpClass(cls):
        # Single shared reference for all W2 computations in this class
        cls.ref = sample_reference_distribution(M=M_REF, seed=SEED)

        # ── Config A: ideal Fibonacci spread across all 4 areas ──────────
        cls.fib_centers = _fibonacci_config(N, ["pit", "bot", "top", "wall"])
        cls.w2_fib = compute_wasserstein_homogeneity(
            cls.fib_centers, reference=cls.ref
        )["w2"]

        # ── Config B: all N points on the wall (global region imbalance) ─
        z_min = float(Z_BASE + PMT_RADIUS)
        z_max = float(Z_BASE + H_CYLINDER - PMT_RADIUS)
        wall_pts = fibonacci_cylinder_wall(N, float(R_ZYLINDER), z_min, z_max)
        cls.w2_wall_only = compute_wasserstein_homogeneity(
            wall_pts, reference=cls.ref
        )["w2"]

        # ── Config C: two tight clusters ──────────────────────────────────
        # Cluster 1 — small Fibonacci disk at pit centre (r ≤ 300 mm)
        cluster_r = 300.0   # mm, << R_PIT = 3800 mm
        c1_xy  = fibonacci_disk(N // 2, 0.0, cluster_r)
        c1     = np.column_stack([c1_xy, np.full(N // 2, _Z_PIT)])
        # Cluster 2 — tight ring on the wall at φ = 0, narrow z band (300 mm)
        half   = N - N // 2
        phi2   = np.linspace(-0.05, 0.05, half)   # ≈ ±3° azimuthal spread
        z_mid  = (_Z_WALL_MIN + _Z_WALL_MAX) / 2
        z2     = np.linspace(z_mid - 150.0, z_mid + 150.0, half)
        c2     = np.column_stack([
            float(R_ZYLINDER) * np.cos(phi2),
            float(R_ZYLINDER) * np.sin(phi2),
            z2,
        ])
        cls.two_cluster_centers = np.vstack([c1, c2])
        cls.w2_two_clusters = compute_wasserstein_homogeneity(
            cls.two_cluster_centers, reference=cls.ref
        )["w2"]

    # --- test 1: global region imbalance ---

    def test_wall_only_worse_than_fibonacci(self):
        """All-on-wall config must have higher W2 than the Fibonacci ideal.

        This is the primary sanity check: W2 detects global region imbalance
        (CV near 0 for all-wall because it only measures local uniformity).
        """
        self.assertGreater(
            self.w2_wall_only, self.w2_fib,
            f"FAIL: W2_wall_only ({self.w2_wall_only:.1f}) should be "
            f"> W2_fibonacci ({self.w2_fib:.1f})",
        )

    # --- test 2: large-scale clustering ---

    def test_two_clusters_worse_than_fibonacci(self):
        """Two tight clusters must have higher W2 than the Fibonacci ideal.

        This checks that large-scale clustering is detected even when each
        cluster is internally uniform (CV ≈ 0 within each cluster).
        """
        self.assertGreater(
            self.w2_two_clusters, self.w2_fib,
            f"FAIL: W2_two_clusters ({self.w2_two_clusters:.1f}) should be "
            f"> W2_fibonacci ({self.w2_fib:.1f})",
        )

    # --- test 3: rotation invariance ---

    def test_rotation_invariance(self):
        """Rotating the Fibonacci config by 60° must leave W2 within 5%.

        The reference distribution is azimuthally symmetric in expectation;
        with M=3000 the Monte Carlo noise is O(M^{-1/2}) ≈ 1.8%, so 5%
        gives a comfortable margin.
        """
        rotated = _rotate_z(self.fib_centers, math.radians(60.0))
        w2_rotated = compute_wasserstein_homogeneity(
            rotated, reference=self.ref
        )["w2"]
        rel_change = abs(w2_rotated - self.w2_fib) / self.w2_fib
        self.assertLess(
            rel_change, 0.05,
            f"Rotation changed W2 by {rel_change:.1%} "
            f"(before={self.w2_fib:.1f}, after={w2_rotated:.1f}); "
            f"expected < 5%",
        )

    # --- informational print ---

    @classmethod
    def tearDownClass(cls):
        print(
            f"\n  [W2 values]  Fibonacci={cls.w2_fib:.1f} mm  |  "
            f"wall-only={cls.w2_wall_only:.1f} mm  |  "
            f"two-clusters={cls.w2_two_clusters:.1f} mm"
        )


# ===========================================================================
# TestWassersteinWithBaseline
# ===========================================================================

class TestWassersteinWithBaseline(unittest.TestCase):
    """Tests for compute_wasserstein_homogeneity_with_baseline."""

    @classmethod
    def setUpClass(cls):
        cls.fib_centers = _fibonacci_config(N, ["pit", "bot", "top", "wall"])
        z_min = float(Z_BASE + PMT_RADIUS)
        z_max = float(Z_BASE + H_CYLINDER - PMT_RADIUS)
        cls.wall_centers = fibonacci_cylinder_wall(
            N, float(R_ZYLINDER), z_min, z_max
        )

    def test_return_dict_keys(self):
        result = compute_wasserstein_homogeneity_with_baseline(
            self.fib_centers, self.fib_centers, M=M_REF, seed=SEED
        )
        for key in ("w2", "w2_fibonacci_baseline", "w2_normalized",
                    "ot_cost", "n_config", "m_reference"):
            self.assertIn(key, result, f"missing key '{key}'")

    def test_fibonacci_vs_itself_normalized_is_one(self):
        """Fibonacci config vs itself as baseline → w2_normalized == 1.0 exactly."""
        result = compute_wasserstein_homogeneity_with_baseline(
            self.fib_centers, self.fib_centers, M=M_REF, seed=SEED
        )
        self.assertAlmostEqual(
            result["w2_normalized"], 1.0, places=10,
            msg=f"Expected w2_normalized=1.0, got {result['w2_normalized']}"
        )

    def test_wall_only_normalized_greater_than_one(self):
        """All-on-wall config is less homogeneous than Fibonacci → w2_normalized > 1."""
        result = compute_wasserstein_homogeneity_with_baseline(
            self.wall_centers, self.fib_centers, M=M_REF, seed=SEED
        )
        self.assertGreater(
            result["w2_normalized"], 1.0,
            f"Expected w2_normalized > 1 for all-wall config, "
            f"got {result['w2_normalized']:.4f}"
        )


# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
