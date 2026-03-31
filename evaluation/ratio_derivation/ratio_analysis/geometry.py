"""Detector geometry parameters — thin wrapper around pmtopt.geometry.

All constants are sourced from the single source of truth in
src/pmtopt/geometry.py. This module re-exports them as a GeometryConfig
dataclass for backwards compatibility with the ratio_analysis codebase.
"""

import numpy as np
from dataclasses import dataclass, field

from pmtopt.geometry import (
    R_PIT, R_ZYL_BOT, R_ZYL_TOP, R_ZYLINDER,
    Z_ORIGIN, Z_OFFSET, H_ZYLINDER,
    Z_CUT_BOT, Z_CUT_TOP, WALL_HEIGHT,
    AREA_SURFACES,
)

__all__ = [
    "GeometryConfig",
    # Re-export constants so callers can do:
    #   from ratio_analysis.geometry import Z_CUT_BOT
    "R_PIT", "R_ZYL_BOT", "R_ZYL_TOP", "R_ZYLINDER",
    "Z_ORIGIN", "Z_OFFSET", "H_ZYLINDER",
    "Z_CUT_BOT", "Z_CUT_TOP", "WALL_HEIGHT", "AREA_SURFACES",
]


@dataclass
class GeometryConfig:
    """Detector geometry parameters.

    Default values are sourced from pmtopt.geometry (single source of truth).
    geometry_name is a free-form metadata tag stored in output JSON files.
    """
    geometry_name: str = "currentDist"

    # Radial boundaries (mm)
    r_pit:      float = float(R_PIT)
    r_zyl_bot:  float = float(R_ZYL_BOT)
    r_zyl_top:  float = float(R_ZYL_TOP)
    r_zylinder: float = float(R_ZYLINDER)

    # Vertical origin / offset (mm)
    z_origin: float = float(Z_ORIGIN)
    z_offset: float = float(Z_OFFSET)

    # Full cylinder height (mm) — h = H_ZYLINDER + 1
    h: float = float(H_ZYLINDER + 1)

    @property
    def h_zylinder(self) -> float:
        """Wall-zone height: full cylinder minus 1 mm top cap (= H_ZYLINDER = 8899)."""
        return float(H_ZYLINDER)

    @property
    def z_cut_bot(self) -> float:
        """Lower wall-zone boundary (mm)."""
        return float(Z_CUT_BOT)

    @property
    def z_cut_top(self) -> float:
        """Upper wall-zone boundary (mm)."""
        return float(Z_CUT_TOP)

    @property
    def wall_height(self) -> float:
        return float(WALL_HEIGHT)

    @property
    def area_pit(self) -> float:
        return AREA_SURFACES["pit"]

    @property
    def area_bot(self) -> float:
        return AREA_SURFACES["bot"]

    @property
    def area_top(self) -> float:
        return AREA_SURFACES["top"]

    @property
    def area_wall(self) -> float:
        return AREA_SURFACES["wall"]
