"""Detector geometry parameters."""

import numpy as np
from dataclasses import dataclass


@dataclass
class GeometryConfig:
    """Detector geometry parameters."""
    geometry_name: str = "currentDist"
    h: float = 8900
    r_zylinder: float = 4300
    z_origin: float = 20
    r_pit: float = 3800
    r_zyl_bot: float = 3950
    r_zyl_top: float = 1200
    z_offset: float = -5000

    @property
    def h_zylinder(self) -> float:
        return self.h - 1  # 8899

    @property
    def z_cut_bot(self) -> float:
        return -4979

    @property
    def z_cut_top(self) -> float:
        return self.z_cut_bot + self.h_zylinder - 2  # 3918

    @property
    def wall_height(self) -> float:
        return self.z_cut_top - self.z_cut_bot  # 8897

    @property
    def area_pit(self) -> float:
        return np.pi * self.r_pit**2

    @property
    def area_bot(self) -> float:
        return np.pi * (self.r_zylinder**2 - self.r_zyl_bot**2)

    @property
    def area_top(self) -> float:
        return np.pi * (self.r_zylinder**2 - self.r_zyl_top**2)

    @property
    def area_wall(self) -> float:
        return 2 * np.pi * self.r_zylinder * self.wall_height
