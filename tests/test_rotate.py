#!/usr/bin/env python3
"""
Unit tests for pmtopt.rotate — no file I/O, no HDF5.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pmtopt.rotate as rv


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_voxel(index, x, y, z, layer):
    return {"index": index, "center": [x, y, z], "corners": [], "layer": layer}


# ── TestRotatePoint ───────────────────────────────────────────────────────────

class TestRotatePoint(unittest.TestCase):
    def test_zero_rotation(self):
        x, y, z = rv.rotate_point(3.0, 4.0, 5.0, 0.0)
        self.assertAlmostEqual(x, 3.0)
        self.assertAlmostEqual(y, 4.0)
        self.assertAlmostEqual(z, 5.0)

    def test_90_deg(self):
        x, y, z = rv.rotate_point(1.0, 0.0, 7.0, math.pi / 2)
        self.assertAlmostEqual(x,  0.0, places=10)
        self.assertAlmostEqual(y,  1.0, places=10)
        self.assertAlmostEqual(z,  7.0)

    def test_180_deg(self):
        x, y, z = rv.rotate_point(1.0, 0.0, 0.0, math.pi)
        self.assertAlmostEqual(x, -1.0, places=10)
        self.assertAlmostEqual(y,  0.0, places=10)

    def test_z_unchanged(self):
        _, _, z = rv.rotate_point(100.0, 200.0, -999.5, 1.234)
        self.assertAlmostEqual(z, -999.5)


# ── TestBuildCandidatePool ────────────────────────────────────────────────────

class TestBuildCandidatePool(unittest.TestCase):
    def setUp(self):
        self.all_voxels = [
            _make_voxel("pit_a",   100,    0, -4000, "pit"),
            _make_voxel("pit_b",  -100,    0, -4000, "pit"),
            _make_voxel("wall_a", 4300,    0, -2000, "wall"),
            _make_voxel("wall_b",    0, 4300, -2000, "wall"),
            _make_voxel("wall_c", 4300,    0, -1000, "wall"),   # different z
        ]

    def test_filters_by_layer(self):
        voxel = _make_voxel("sel", 50, 0, -4000, "pit")
        pool = rv.build_candidate_pool(voxel, self.all_voxels)
        self.assertEqual(len(pool), 2)
        for v in pool:
            self.assertEqual(v["layer"], "pit")

    def test_wall_same_z_included(self):
        voxel = _make_voxel("sel", 4300, 0, -2000, "wall")
        pool = rv.build_candidate_pool(voxel, self.all_voxels)
        self.assertEqual(len(pool), 2)

    def test_wall_z_filter_excludes_different_z(self):
        voxel = _make_voxel("sel", 4300, 0, -2000, "wall")
        pool = rv.build_candidate_pool(voxel, self.all_voxels)
        for v in pool:
            self.assertAlmostEqual(v["center"][2], -2000, places=0)

    def test_empty_pool_raises(self):
        voxel = _make_voxel("sel", 0, 0, 99999, "top")
        with self.assertRaises(ValueError):
            rv.build_candidate_pool(voxel, self.all_voxels)


# ── TestFindNearestCandidate ──────────────────────────────────────────────────

class TestFindNearestCandidate(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            _make_voxel("a",  10, 0, 0, "pit"),
            _make_voxel("b", 100, 0, 0, "pit"),
            _make_voxel("c",  50, 0, 0, "pit"),
        ]

    def test_nearest_is_closest(self):
        result = rv.find_nearest_candidate((12, 0, 0), self.candidates)
        self.assertEqual(result["index"], "a")

    def test_single_candidate_returned(self):
        result = rv.find_nearest_candidate((99999, 0, 0), [self.candidates[1]])
        self.assertEqual(result["index"], "b")

    def test_exact_match_returned(self):
        result = rv.find_nearest_candidate((50, 0, 0), self.candidates)
        self.assertEqual(result["index"], "c")


# ── TestComputeVoxelMapping ───────────────────────────────────────────────────

class TestComputeVoxelMapping(unittest.TestCase):
    def _ring_voxels(self, r=3000, z=-4000, layer="pit"):
        """4 voxels at cardinal positions on a ring."""
        return [
            _make_voxel("v0",  r,  0, z, layer),
            _make_voxel("v1",  0,  r, z, layer),
            _make_voxel("v2", -r,  0, z, layer),
            _make_voxel("v3",  0, -r, z, layer),
        ]

    def test_90_deg_cyclic_shift(self):
        voxels = self._ring_voxels()
        mapping = rv.compute_voxel_mapping(voxels, voxels, phi_frac=0.25)
        # v0 (r,0) → rotated to (0,r) → nearest is v1
        self.assertEqual(mapping[0][1]["index"], "v1")
        self.assertEqual(mapping[1][1]["index"], "v2")
        self.assertEqual(mapping[2][1]["index"], "v3")
        self.assertEqual(mapping[3][1]["index"], "v0")

    def test_returns_pairs(self):
        voxels = self._ring_voxels()
        mapping = rv.compute_voxel_mapping(voxels, voxels, phi_frac=0.25)
        self.assertEqual(len(mapping), 4)
        for orig, tgt in mapping:
            self.assertIn("index", orig)
            self.assertIn("index", tgt)

    def test_no_per_voxel_self_check(self):
        """compute_voxel_mapping must NOT raise for phi=0; check_mapping does."""
        voxels = self._ring_voxels()
        mapping = rv.compute_voxel_mapping(voxels, voxels, phi_frac=0.0)
        self.assertEqual(len(mapping), 4)


# ── TestCheckMapping ──────────────────────────────────────────────────────────

class TestCheckMapping(unittest.TestCase):
    def _mapping(self, pairs):
        """Build a mapping from (orig_id, tgt_id) string pairs."""
        return [
            (_make_voxel(o, 0, 0, 0, "pit"), _make_voxel(t, 0, 0, 0, "pit"))
            for o, t in pairs
        ]

    def test_valid_mapping_passes(self):
        m = self._mapping([("a", "d"), ("b", "e"), ("c", "f")])
        rv.check_mapping(m)   # must not raise

    def test_collision_raises(self):
        m = self._mapping([("a", "x"), ("b", "x"), ("c", "z")])
        with self.assertRaises(RuntimeError) as ctx:
            rv.check_mapping(m)
        self.assertIn("Collision", str(ctx.exception))

    def test_self_overlap_direct(self):
        """Target 'a' is the same index as original 'a'."""
        m = self._mapping([("a", "a"), ("b", "e"), ("c", "f")])
        with self.assertRaises(RuntimeError) as ctx:
            rv.check_mapping(m)
        self.assertIn("Self-overlap", str(ctx.exception))

    def test_self_overlap_indirect(self):
        """Original set = {a,b,c}; target 'b' is in that set."""
        m = self._mapping([("a", "d"), ("b", "b"), ("c", "f")])
        with self.assertRaises(RuntimeError) as ctx:
            rv.check_mapping(m)
        self.assertIn("Self-overlap", str(ctx.exception))


# ── TestAssembleOutputVoxels ──────────────────────────────────────────────────

class TestAssembleOutputVoxels(unittest.TestCase):
    def test_output_preserves_order(self):
        mapping = [
            (_make_voxel("o1", 0, 0, 0, "pit"), _make_voxel("t1", 1, 0, 0, "pit")),
            (_make_voxel("o2", 0, 0, 0, "pit"), _make_voxel("t2", 2, 0, 0, "pit")),
        ]
        result = rv.assemble_output_voxels(mapping)
        self.assertEqual(result[0]["index"], "t1")
        self.assertEqual(result[1]["index"], "t2")

    def test_output_uses_target_geometry(self):
        orig = _make_voxel("orig", 10, 20, 30, "pit")
        tgt  = _make_voxel("tgt",  99, 88, 77, "pit")
        result = rv.assemble_output_voxels([(orig, tgt)])
        self.assertEqual(result[0]["center"], [99, 88, 77])


# ── TestComputeSpacingStats ───────────────────────────────────────────────────

class TestComputeSpacingStats(unittest.TestCase):
    def test_single_voxel_returns_none(self):
        c = np.array([[0.0, 0.0, 0.0]])
        self.assertIsNone(rv.compute_spacing_stats(c))

    def test_two_voxels(self):
        c = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        s = rv.compute_spacing_stats(c)
        self.assertIsNotNone(s)
        self.assertAlmostEqual(s["pw_min"],  100.0)
        self.assertAlmostEqual(s["pw_max"],  100.0)
        self.assertAlmostEqual(s["pw_mean"], 100.0)
        self.assertAlmostEqual(s["nn_min"],  100.0)
        self.assertAlmostEqual(s["nn_mean"], 100.0)

    def test_equilateral_triangle(self):
        # All pairwise distances = 100 mm
        h = 100 * math.sqrt(3) / 2
        c = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [50.0, h, 0.0]])
        s = rv.compute_spacing_stats(c)
        self.assertAlmostEqual(s["pw_min"],  100.0, places=5)
        self.assertAlmostEqual(s["pw_max"],  100.0, places=5)
        self.assertAlmostEqual(s["pw_std"],    0.0, places=5)
        self.assertAlmostEqual(s["nn_min"],  100.0, places=5)

    def test_required_keys_present(self):
        c = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        s = rv.compute_spacing_stats(c)
        for k in ("pw_min", "pw_mean", "pw_max", "pw_std",
                  "nn_min", "nn_mean", "nn_max", "nn_std"):
            self.assertIn(k, s)


# ── TestSpacingTestPassFail ───────────────────────────────────────────────────

class TestSpacingTestPassFail(unittest.TestCase):
    def _uniform_ring(self, n, r, layer):
        angles = [2 * math.pi * i / n for i in range(n)]
        return [
            _make_voxel(f"v{i}", r * math.cos(a), r * math.sin(a), 0.0, layer)
            for i, a in enumerate(angles)
        ]

    def test_identical_selection_passes(self):
        voxels = self._uniform_ring(10, 3000, "pit")
        res = rv.run_spacing_test(voxels, voxels)
        self.assertTrue(res["pit"]["passed"])

    def test_clustered_after_fails(self):
        before = self._uniform_ring(8, 3000, "pit")
        # All rotated voxels clustered at nearly the same point
        after = [
            _make_voxel(f"t{i}", 3000.0 + i * 0.1, 0.0, 0.0, "pit")
            for i in range(8)
        ]
        res = rv.run_spacing_test(before, after)
        self.assertFalse(res["pit"]["passed"])

    def test_single_voxel_layer_skipped(self):
        before = [_make_voxel("a", 100, 0, 0, "bot")]
        after  = [_make_voxel("b", 200, 0, 0, "bot")]
        res = rv.run_spacing_test(before, after)
        self.assertIsNone(res["bot"]["passed"])


# ── TestWallZFilter ───────────────────────────────────────────────────────────

class TestWallZFilter(unittest.TestCase):
    def test_rotation_stays_in_z_ring(self):
        """Wall voxels at z=-4000 must never map to voxels at z=-3000."""
        r = 4300
        all_voxels = []
        for z, grp in [(-4000.0, "lo"), (-3000.0, "hi")]:
            for i, (cx, cy) in enumerate([(r, 0), (0, r), (-r, 0), (0, -r)]):
                all_voxels.append(_make_voxel(f"{grp}_{i}", cx, cy, z, "wall"))

        selected = [v for v in all_voxels if v["index"].startswith("lo_")]
        mapping = rv.compute_voxel_mapping(selected, all_voxels, phi_frac=0.25)

        for _, tgt in mapping:
            self.assertAlmostEqual(tgt["center"][2], -4000.0, places=0)


# ── TestDeriveOutputPath ──────────────────────────────────────────────────────

class TestDeriveOutputPath(unittest.TestCase):
    def test_angle_in_filename(self):
        p = rv.derive_output_path("results/selection.json", 0.5)
        self.assertIn("0.5000x2pi", p.name)

    def test_stem_from_input(self):
        p = rv.derive_output_path("results/greedy_N300.json", 0.25)
        self.assertTrue(p.name.startswith("greedy_N300"))

    def test_parent_preserved(self):
        p = rv.derive_output_path("results/sel.json", 0.1)
        self.assertEqual(str(p.parent), "results")


if __name__ == "__main__":
    unittest.main(verbosity=2)
