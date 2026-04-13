"""
Detector geometry constants and PMT placement validation.

All dimensions in mm. Coordinate system origin at (0, 0, Z_ORIGIN + Z_OFFSET).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Detector geometry constants (mm)
# ---------------------------------------------------------------------------
PMT_RADIUS: int = 131          # PMT physical radius (used for placement validation)
PMT_CATHODE_RADIUS: int = 110  # PMT cathode (sensitive surface) radius
#                                (used for MC overlap fraction sampling)
R_PIT: int = 3800              # Pit radius
R_ZYL_BOT: int = 3950          # Inner radius of bottom ring
R_ZYL_TOP: int = 1200          # Inner radius of top ring
R_ZYLINDER: int = 4300         # Apothem (outer radius of rings / wall)
Z_ORIGIN: int = 20
Z_OFFSET: int = -5000
H_ZYLINDER: int = 8900 - 1    # h - 1; wall-zone height excl. 1 mm top cap
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
Z_CUT_BOT: int = Z_BASE_GLOBAL + 1   # +1 mm extra clearance offset at cylinder bottom
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
    areas: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Distribute N PMTs across areas proportional to surface area.

    Uses the largest remainder method to ensure sum(N_area) == N exactly.

    Parameters
    ----------
    N : int
        Total number of PMTs to distribute.
    areas : list of str or None
        Subset of areas to distribute across (e.g. ["pit", "wall"]).
        If None, all four areas are used.
    verbose : bool
        Print allocation table.

    Returns
    -------
    allocation : dict[str, int]
        Number of PMTs per area. Areas not in ``areas`` receive 0.
    """
    _all_areas = ["pit", "bot", "top", "wall"]
    areas_to_use = (
        [a for a in _all_areas if a in areas]
        if areas is not None
        else _all_areas
    )

    selected_surfaces = {a: AREA_SURFACES[a] for a in areas_to_use}
    total_area = sum(selected_surfaces.values())
    fractions = {a: N * s / total_area for a, s in selected_surfaces.items()}
    floors = {a: int(np.floor(f)) for a, f in fractions.items()}
    remainders = {a: fractions[a] - floors[a] for a in fractions}
    leftover = N - sum(floors.values())
    sorted_areas = sorted(remainders, key=lambda a: remainders[a], reverse=True)
    allocation = dict(floors)
    for i in range(leftover):
        allocation[sorted_areas[i]] += 1

    # Fill zeros for areas that were excluded
    for a in _all_areas:
        if a not in allocation:
            allocation[a] = 0

    if verbose:
        print(f"\nPer-area PMT allocation (N={N}, areas={areas_to_use}):")
        print(f"  {'Area':<6} {'N_PMTs':>7} {'Fläche (M mm²)':>16} "
              f"{'Dichte':>14} {'Abw. von Ziel':>14}")
        print(f"  {'-' * 58}")
        target_density = N / total_area
        for area in areas_to_use:
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

#FoM parameters and fuctions
N_a = 6.022e23 # 1 / mol
m_76 = 75.9214036 *1e-3 # kg/mol
epsilon_tot = 0.67 # detector efficiency * analysis efficiency
exposure = 10000 # kg yr
T_half = 1e28 # yr
r_prod = 0.167 # atoms/(kg·yr)
a_rel = 0.50 # fraction of Ge-77m produced
c_ge77m = 15.1 * 1e-5 # 1 / keV
c_ge77 = 0.74 * 1e-5 # 1 / keV
BI_other = 8e-6 # counts/(kg·yr·keV)
delta_E = 5 # keV

def calc_deadtime(nCaptures, threshold, runtime):
    """ Requires number of nCaptures, threshold for vetoing, and runtime in hours"""
    above_threshold = nCaptures >= threshold
    above_threshold_sum = np.sum(above_threshold)
    veto_time = above_threshold_sum * 5 * 54 / np.log(2) / (60 * 60) # in hours, single veto is ~ 5 minutes
    deadtime = veto_time / runtime # runtime has to be in hours
    return deadtime # find new way to calculate uncertainty

def calc_germanium_efficiency(nCaptures, threshold, ge77_counts):
    total_events = np.sum(ge77_counts)
    cut_events = np.sum((nCaptures >= threshold) * ge77_counts) # This actually works. Weight each event by its Ge-77 count
    efficiency = cut_events / total_events
    return efficiency # Find new way to calculate uncertainty

def figure_of_merit(ge_survival_eff, signal_survival_eff):
    s = (np.log(2) * N_a / m_76) * epsilon_tot * exposure * (1 / T_half) * signal_survival_eff
    b = exposure * delta_E * signal_survival_eff * (r_prod * (a_rel * ge_survival_eff * c_ge77m + (1 - a_rel * (1 - 0.19)) * c_ge77) + BI_other)

    if b <= 0 or s <= 0:  
        return np.nan
    
    fom = np.sqrt(2 * ((s + b) * np.log(1 + s/b) - s))
    return fom

def calc_figure_of_merit(nCaptures, threshold, ge77_counts, runtime):
    ge_survival_eff = 1 - calc_germanium_efficiency(nCaptures, threshold=threshold, ge77_counts=ge77_counts)
    signal_survival_eff = 1 - calc_deadtime(nCaptures, threshold=threshold, runtime=runtime)
    return figure_of_merit(ge_survival_eff, signal_survival_eff)