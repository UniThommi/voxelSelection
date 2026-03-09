"""
Detector geometry constants and PMT placement validation.

All dimensions in mm. Coordinate system origin at (0, 0, Z_ORIGIN + Z_OFFSET).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Detector geometry constants (mm)
# ---------------------------------------------------------------------------
PMT_RADIUS: int = 131          # PMT physical radius
R_PIT: int = 3800              # Pit radius
R_ZYL_BOT: int = 3950          # Inner radius of bottom ring
R_ZYL_TOP: int = 1200          # Inner radius of top ring
R_ZYLINDER: int = 4300         # Apothem (outer radius of rings / wall)
Z_ORIGIN: int = 20
Z_OFFSET: int = -5000
H_ZYLINDER: int = 8900 - 1    # h - 1
Z_BASE_GLOBAL: int = Z_ORIGIN + Z_OFFSET

# Muon-Ge77 time window (ns)
MUON_TIME_WINDOW_MIN_NS: float = 1_000.0    # 1 µs
MUON_TIME_WINDOW_MAX_NS: float = 200_000.0  # 200 µs

# ---------------------------------------------------------------------------
# Default area-dependent hit scaling ratios (SSD / PMT).
# ---------------------------------------------------------------------------
DEFAULT_AREA_RATIOS: dict[str, float] = {
    "pit":  2.0731,
    "bot":  2.3843,
    "top":  2.2004,
    "wall": 1.8776,
}

# ---------------------------------------------------------------------------
# Detector surface areas (mm²), derived from geometry constants.
# ---------------------------------------------------------------------------
Z_CUT_BOT: int = Z_BASE_GLOBAL
Z_CUT_TOP: int = Z_CUT_BOT + H_ZYLINDER - 2
WALL_HEIGHT: int = Z_CUT_TOP - Z_CUT_BOT

AREA_SURFACES: dict[str, float] = {
    "pit":  np.pi * R_PIT**2,
    "bot":  np.pi * (R_ZYLINDER**2 - R_ZYL_BOT**2),
    "top":  np.pi * (R_ZYLINDER**2 - R_ZYL_TOP**2),
    "wall": 2 * np.pi * R_ZYLINDER * WALL_HEIGHT,
}


def compute_per_area_N(
    N: int,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Distribute N PMTs across areas proportional to surface area.

    Uses the largest remainder method to ensure sum(N_area) == N exactly.

    Parameters
    ----------
    N : int
        Total number of PMTs to distribute.
    verbose : bool
        Print allocation table.

    Returns
    -------
    allocation : dict[str, int]
        Number of PMTs per area.
    """
    total_area = sum(AREA_SURFACES.values())
    fractions = {area: N * a / total_area for area, a in AREA_SURFACES.items()}
    floors = {area: int(np.floor(f)) for area, f in fractions.items()}
    remainders = {area: fractions[area] - floors[area] for area in fractions}
    leftover = N - sum(floors.values())
    sorted_areas = sorted(remainders, key=lambda a: remainders[a], reverse=True)
    allocation = dict(floors)
    for i in range(leftover):
        allocation[sorted_areas[i]] += 1

    if verbose:
        print(f"\nPer-area PMT allocation (N={N}):")
        print(f"  {'Area':<6} {'N_PMTs':>7} {'Fläche (M mm²)':>16} "
              f"{'Dichte':>14} {'Abw. von Ziel':>14}")
        print(f"  {'-' * 58}")
        target_density = N / total_area
        for area in ["pit", "bot", "top", "wall"]:
            n_a = allocation[area]
            a_mm2 = AREA_SURFACES[area]
            density = n_a / a_mm2 if a_mm2 > 0 else 0.0
            dev = (density - target_density) / target_density * 100
            print(f"  {area:<6} {n_a:>7} {a_mm2/1e6:>16.2f} "
                  f"{density:>14.6e} {dev:>+13.1f}%")

    return allocation


def is_valid_pmt_position(
    center: np.ndarray,
    layer: str,
    pmt_r: float = PMT_RADIUS,
) -> bool:
    """
    Check whether a PMT of given radius can be physically placed at
    the voxel center without protruding beyond the detector boundaries.

    Parameters
    ----------
    center : np.ndarray, shape (3,)
        Voxel center coordinates (x, y, z) in mm.
    layer : str
        Detector layer: "pit", "bot", "top", or "wall".
    pmt_r : float
        PMT radius in mm.

    Returns
    -------
    bool
        True if the PMT fits within the layer boundaries.
    """
    x, y, z = center
    r_center = np.sqrt(x**2 + y**2)

    if layer == "pit":
        return r_center + pmt_r <= R_PIT
    elif layer == "bot":
        return (r_center - pmt_r >= R_ZYL_BOT) and (r_center + pmt_r <= R_ZYLINDER)
    elif layer == "top":
        return (r_center - pmt_r >= R_ZYL_TOP) and (r_center + pmt_r <= R_ZYLINDER)
    elif layer == "wall":
        z_min_allowed = Z_BASE_GLOBAL + pmt_r
        z_max_allowed = Z_BASE_GLOBAL + H_ZYLINDER - pmt_r
        return z_min_allowed <= z <= z_max_allowed

    return False