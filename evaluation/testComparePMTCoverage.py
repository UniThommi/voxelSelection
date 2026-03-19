#!/usr/bin/env python3
"""
Unit tests for comparePMTCoverage.py
Uses mock data to verify analysis logic without h5py dependency.
"""

import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

# ── Mock h5py before importing the module ────────────────────────────
sys.modules["h5py"] = type(sys)("h5py")

import comparePMTCoverage as cpc


class TestDetectNCs(unittest.TestCase):
    """Test NC detection logic."""

    def _make_nc_df(self, records: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(records)

    def _make_optical_df(self, records: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(records)

    def test_basic_detection(self):
        """NC with enough PMTs should be detected."""
        nc_df = self._make_nc_df(
            [{"muon_id": 1, "nc_id": 10, "nc_time_ns": 5000.0}]
        )
        # 7 distinct PMTs, all within 200 ns
        optical_records = [
            {
                "muon_track_id": 1,
                "nC_track_id": 10,
                "det_uid": uid,
                "time_in_ns": 5100.0,  # 100 ns after NC
            }
            for uid in range(7)
        ]
        optical_df = self._make_optical_df(optical_records)

        detected_set, mult_df = cpc.detect_ncs(optical_df, nc_df, M_threshold=6, m_threshold=1)

        self.assertIn((1, 10), detected_set)
        row = mult_df[(mult_df["muon_track_id"] == 1) & (mult_df["nC_track_id"] == 10)]
        self.assertEqual(row["n_firing_pmts"].values[0], 7)

    def test_below_threshold(self):
        """NC with too few PMTs should not be detected."""
        nc_df = self._make_nc_df(
            [{"muon_id": 1, "nc_id": 10, "nc_time_ns": 5000.0}]
        )
        optical_records = [
            {
                "muon_track_id": 1,
                "nC_track_id": 10,
                "det_uid": uid,
                "time_in_ns": 5100.0,
            }
            for uid in range(3)  # only 3 PMTs
        ]
        optical_df = self._make_optical_df(optical_records)

        detected_set, _ = cpc.detect_ncs(optical_df, nc_df, M_threshold=6, m_threshold=1)
        self.assertNotIn((1, 10), detected_set)

    def test_time_cut(self):
        """Photons arriving >200ns after NC should not count."""
        nc_df = self._make_nc_df(
            [{"muon_id": 1, "nc_id": 10, "nc_time_ns": 5000.0}]
        )
        optical_records = [
            {
                "muon_track_id": 1,
                "nC_track_id": 10,
                "det_uid": uid,
                "time_in_ns": 5300.0,  # 300 ns after NC → outside cut
            }
            for uid in range(10)
        ]
        optical_df = self._make_optical_df(optical_records)

        detected_set, _ = cpc.detect_ncs(optical_df, nc_df, M_threshold=6, m_threshold=1)
        self.assertEqual(len(detected_set), 0)

    def test_m_threshold(self):
        """PMT with fewer than m hits should not count as firing."""
        nc_df = self._make_nc_df(
            [{"muon_id": 1, "nc_id": 10, "nc_time_ns": 5000.0}]
        )
        # 6 PMTs but each with only 1 hit, m=2 → none fire
        optical_records = [
            {
                "muon_track_id": 1,
                "nC_track_id": 10,
                "det_uid": uid,
                "time_in_ns": 5050.0,
            }
            for uid in range(6)
        ]
        optical_df = self._make_optical_df(optical_records)

        detected_set, _ = cpc.detect_ncs(optical_df, nc_df, M_threshold=6, m_threshold=2)
        self.assertEqual(len(detected_set), 0)

    def test_m_threshold_passes(self):
        """PMTs with enough hits should fire."""
        nc_df = self._make_nc_df(
            [{"muon_id": 1, "nc_id": 10, "nc_time_ns": 5000.0}]
        )
        # 6 PMTs, each with 2 hits
        optical_records = []
        for uid in range(6):
            for _ in range(2):
                optical_records.append(
                    {
                        "muon_track_id": 1,
                        "nC_track_id": 10,
                        "det_uid": uid,
                        "time_in_ns": 5050.0,
                    }
                )
        optical_df = self._make_optical_df(optical_records)

        detected_set, _ = cpc.detect_ncs(optical_df, nc_df, M_threshold=6, m_threshold=2)
        self.assertIn((1, 10), detected_set)

    def test_empty_optical(self):
        """Empty optical data should return nothing."""
        nc_df = self._make_nc_df(
            [{"muon_id": 1, "nc_id": 10, "nc_time_ns": 5000.0}]
        )
        optical_df = pd.DataFrame(
            columns=["muon_track_id", "nC_track_id", "det_uid", "time_in_ns"]
        )
        detected_set, _ = cpc.detect_ncs(optical_df, nc_df, M_threshold=6, m_threshold=1)
        self.assertEqual(len(detected_set), 0)

    def test_float_tolerance(self):
        """Photon arriving barely before NC (float rounding) should count."""
        nc_df = self._make_nc_df(
            [{"muon_id": 1, "nc_id": 10, "nc_time_ns": 5000.0}]
        )
        optical_records = [
            {
                "muon_track_id": 1,
                "nC_track_id": 10,
                "det_uid": uid,
                "time_in_ns": 4999.5,  # -0.5 ns before NC, within tolerance
            }
            for uid in range(6)
        ]
        optical_df = self._make_optical_df(optical_records)

        detected_set, _ = cpc.detect_ncs(optical_df, nc_df, M_threshold=6, m_threshold=1)
        self.assertIn((1, 10), detected_set)


class TestDetectability(unittest.TestCase):
    """Test NC detectability categorization."""

    def test_categories(self):
        nc_df = pd.DataFrame(
            {
                "muon_id": [1, 1, 1],
                "nc_id": [10, 20, 30],
                "nc_time_ns": [5000.0, 6000.0, 7000.0],
            }
        )
        all_keys = {(1, 10), (1, 20), (1, 30)}

        optical_df = pd.DataFrame(
            {
                "muon_track_id": [1, 1, 1],
                "nC_track_id": [10, 20, 30],
                "det_uid": [0, 0, 0],
                "time_in_ns": [
                    5100.0,  # NC 10: within 200 ns
                    6500.0,  # NC 20: 500 ns after → outside
                    8000.0,  # NC 30: 1000 ns after → outside
                ],
            }
        )

        any_p, within, only_out = cpc.compute_detectability(
            optical_df, nc_df, all_keys
        )

        self.assertEqual(any_p, {(1, 10), (1, 20), (1, 30)})
        self.assertEqual(within, {(1, 10)})
        self.assertEqual(only_out, {(1, 20), (1, 30)})


class TestMuonClassification(unittest.TestCase):
    """Test full analysis pipeline with mock RunData."""

    def test_ge77_classification(self):
        """End-to-end: one Ge77 muon correctly classified, one non-Ge77 not."""
        # Muon 1: Ge77 muon with 7 NCs, all in [1µs, 200µs]
        # Muon 2: non-Ge77 with 2 NCs
        nc_records = []
        for i in range(7):
            nc_records.append(
                {
                    "muon_id": 1,
                    "nc_id": 100 + i,
                    "nc_time_ns": 10_000.0 + i * 1000,
                    "flag_ge77": 1 if i == 0 else 0,
                    "nc_x": 0.0,
                    "nc_y": 0.0,
                    "nc_z": 0.0,
                }
            )
        for i in range(2):
            nc_records.append(
                {
                    "muon_id": 2,
                    "nc_id": 200 + i,
                    "nc_time_ns": 50_000.0 + i * 1000,
                    "flag_ge77": 0,
                    "nc_x": 0.0,
                    "nc_y": 0.0,
                    "nc_z": 0.0,
                }
            )
        nc_df = pd.DataFrame(nc_records)

        # Optical: all 7 NCs of muon 1 detected (6+ PMTs), muon 2 NCs not
        optical_records = []
        for i in range(7):
            nc_time = 10_000.0 + i * 1000
            for uid in range(8):  # 8 PMTs per NC
                optical_records.append(
                    {
                        "muon_track_id": 1,
                        "nC_track_id": 100 + i,
                        "det_uid": uid,
                        "time_in_ns": nc_time + 50.0,
                    }
                )
        # Muon 2: only 2 PMTs per NC
        for i in range(2):
            nc_time = 50_000.0 + i * 1000
            for uid in range(2):
                optical_records.append(
                    {
                        "muon_track_id": 2,
                        "nC_track_id": 200 + i,
                        "det_uid": uid,
                        "time_in_ns": nc_time + 50.0,
                    }
                )
        optical_df = pd.DataFrame(optical_records)

        # Create mock RunData
        rd = cpc.RunData(
            run_id=1,
            nc_df=nc_df,
            optical_hom=optical_df,
            optical_opt=optical_df,
            n_vertices_hom=9,
            n_vertices_opt=9,
        )

        result = cpc.analyze_setup([rd], "hom", M_threshold=6, m_threshold=1, W_threshold=6)

        # Muon 1: Ge77, 7 detected NCs in window → classified → TP
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fn, 0)
        # Muon 2: non-Ge77, 0 detected NCs → not classified → TN
        self.assertEqual(result.tn, 1)
        self.assertEqual(result.fp, 0)
        # NC totals
        self.assertEqual(result.nc_total, 9)
        self.assertEqual(result.nc_detected, 7)  # muon 1's NCs
        self.assertEqual(result.ge77_nc_total, 7)  # all NCs of Ge77 muon

    def test_false_negative(self):
        """Ge77 muon with too few detected NCs → FN."""
        nc_records = []
        for i in range(7):
            nc_records.append(
                {
                    "muon_id": 1,
                    "nc_id": 100 + i,
                    "nc_time_ns": 10_000.0 + i * 1000,
                    "flag_ge77": 1 if i == 0 else 0,
                    "nc_x": 0.0,
                    "nc_y": 0.0,
                    "nc_z": 0.0,
                }
            )
        nc_df = pd.DataFrame(nc_records)

        # Only 3 NCs have enough PMTs to be detected
        optical_records = []
        for i in range(3):
            nc_time = 10_000.0 + i * 1000
            for uid in range(8):
                optical_records.append(
                    {
                        "muon_track_id": 1,
                        "nC_track_id": 100 + i,
                        "det_uid": uid,
                        "time_in_ns": nc_time + 50.0,
                    }
                )
        # Remaining 4 NCs: not enough PMTs
        for i in range(3, 7):
            nc_time = 10_000.0 + i * 1000
            for uid in range(2):
                optical_records.append(
                    {
                        "muon_track_id": 1,
                        "nC_track_id": 100 + i,
                        "det_uid": uid,
                        "time_in_ns": nc_time + 50.0,
                    }
                )
        optical_df = pd.DataFrame(optical_records)

        rd = cpc.RunData(
            run_id=1,
            nc_df=nc_df,
            optical_hom=optical_df,
            optical_opt=optical_df,
            n_vertices_hom=7,
            n_vertices_opt=7,
        )

        result = cpc.analyze_setup([rd], "hom", M_threshold=6, m_threshold=1, W_threshold=6)

        # Only 3 detected NCs < W=6 → FN
        self.assertEqual(result.fn, 1)
        self.assertEqual(result.tp, 0)

    def test_nc_outside_muon_window(self):
        """NCs outside [1µs, 200µs] should not count for classification."""
        nc_records = []
        # 10 NCs, but all before 1µs
        for i in range(10):
            nc_records.append(
                {
                    "muon_id": 1,
                    "nc_id": 100 + i,
                    "nc_time_ns": 500.0 + i * 10,  # all < 1µs
                    "flag_ge77": 1 if i == 0 else 0,
                    "nc_x": 0.0,
                    "nc_y": 0.0,
                    "nc_z": 0.0,
                }
            )
        nc_df = pd.DataFrame(nc_records)

        # All NCs detected (enough PMTs)
        optical_records = []
        for i in range(10):
            nc_time = 500.0 + i * 10
            for uid in range(8):
                optical_records.append(
                    {
                        "muon_track_id": 1,
                        "nC_track_id": 100 + i,
                        "det_uid": uid,
                        "time_in_ns": nc_time + 50.0,
                    }
                )
        optical_df = pd.DataFrame(optical_records)

        rd = cpc.RunData(
            run_id=1,
            nc_df=nc_df,
            optical_hom=optical_df,
            optical_opt=optical_df,
            n_vertices_hom=10,
            n_vertices_opt=10,
        )

        result = cpc.analyze_setup([rd], "hom", M_threshold=6, m_threshold=1, W_threshold=6)

        # All NCs detected, but none in [1µs, 200µs] → FN
        self.assertEqual(result.nc_detected, 10)
        self.assertEqual(result.fn, 1)  # Ge77 muon not classified
        self.assertEqual(result.tp, 0)


class TestValidation(unittest.TestCase):
    """Test validation checks."""

    def test_vertex_mismatch(self):
        """Mismatched vertex counts should raise RuntimeError."""
        rd = cpc.RunData(
            run_id=1,
            nc_df=pd.DataFrame(
                {"muon_id": [1], "nc_id": [1], "nc_time_ns": [100.0],
                 "flag_ge77": [0], "nc_x": [0], "nc_y": [0], "nc_z": [0]}
            ),
            optical_hom=pd.DataFrame(
                columns=["muon_track_id", "nC_track_id", "det_uid", "time_in_ns"]
            ),
            optical_opt=pd.DataFrame(
                columns=["muon_track_id", "nC_track_id", "det_uid", "time_in_ns"]
            ),
            n_vertices_hom=100,
            n_vertices_opt=99,
        )
        with self.assertRaises(RuntimeError):
            cpc.validate_runs([rd])

    def test_nc_time_too_negative(self):
        """NC time below tolerance should raise RuntimeError."""
        rd = cpc.RunData(
            run_id=1,
            nc_df=pd.DataFrame(
                {"muon_id": [1], "nc_id": [1], "nc_time_ns": [-5.0],
                 "flag_ge77": [0], "nc_x": [0], "nc_y": [0], "nc_z": [0]}
            ),
            optical_hom=pd.DataFrame(
                columns=["muon_track_id", "nC_track_id", "det_uid", "time_in_ns"]
            ),
            optical_opt=pd.DataFrame(
                columns=["muon_track_id", "nC_track_id", "det_uid", "time_in_ns"]
            ),
            n_vertices_hom=1,
            n_vertices_opt=1,
        )
        with self.assertRaises(RuntimeError):
            cpc.validate_runs([rd])


if __name__ == "__main__":
    unittest.main(verbosity=2)