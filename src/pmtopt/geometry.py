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

# ---------------------------------------------------------------------------
# Figure of Merit — physical parameters
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Figure of Merit — Musun simulation constants
# ---------------------------------------------------------------------------
MUSUN_RATE: float = 504.0
"""Musun simulation rate: muons generated per hour of livetime."""

MUONS_PER_RUN_DIR: int = 10_000_000
"""Number of primary muons simulated per run directory.
Used as a fallback when /primaries in the HDF5 is 0 (placeholder).
"""

VETO_DURATION_H: float = 5.0 * 54.0 / np.log(2) / 3600.0
"""Dead-time per classified-positive muon (hours).

Derived from vetoing for 5 half-lives of the Ge-77 activation product
(T_1/2 ≈ 54 min → mean lifetime τ = T_1/2 / ln 2 ≈ 77.9 min).
"""


# ---------------------------------------------------------------------------
# Figure of Merit — general (array-based) functions
# ---------------------------------------------------------------------------

def calc_deadtime(nCaptures, threshold, runtime):
    """Dead-time fraction from per-muon capture arrays.

    Parameters
    ----------
    nCaptures : array-like
        Per-muon NC capture counts.
    threshold : int
        Veto threshold W: muons with nCaptures >= threshold are vetoed.
    runtime : float
        Simulated runtime in **hours**.

    Returns
    -------
    float
        Dead-time fraction (veto time / runtime).
    """
    above_threshold = nCaptures >= threshold
    veto_time = np.sum(above_threshold) * VETO_DURATION_H  # hours
    return veto_time / runtime


def calc_germanium_efficiency(nCaptures, threshold, ge77_muon_counts):
    """Ge-77 muon recall from per-muon arrays.

    Returns Recall = detected Ge77 muons / total Ge77 muons, where
    "detected" means the muon's NC capture count meets the threshold W.

    Parameters
    ----------
    nCaptures : array-like
        Per-muon NC capture counts (length = total muons in simulation).
    threshold : int
        Veto threshold W: muon is classified positive if nCaptures >= threshold.
    ge77_muon_counts : array-like
        Binary per-muon flag: 1 for Ge77 muons, 0 for non-Ge77.

    Returns
    -------
    float
        Recall = TP / (TP + FN).
    """
    total_ge77 = np.sum(ge77_muon_counts)
    detected   = np.sum((nCaptures >= threshold) * ge77_muon_counts)
    return detected / total_ge77 if total_ge77 > 0 else 0.0


def figure_of_merit(ge_survival_eff: float, signal_survival_eff: float) -> float:
    """Core Figure of Merit formula.

    Parameters
    ----------
    ge_survival_eff : float
        Fraction of Ge-77 muons that are *not* detected (1 − Recall).
    signal_survival_eff : float
        Fraction of livetime that is *not* vetoed (1 − deadtime).

    Returns
    -------
    float
        FoM value, or np.nan when the formula is undefined (s ≤ 0 or b ≤ 0).
    """
    s = (np.log(2) * N_a / m_76) * epsilon_tot * exposure * (1.0 / T_half) * signal_survival_eff
    b = (exposure * delta_E * signal_survival_eff
         * (r_prod * (a_rel * ge_survival_eff * c_ge77m
                      + (1.0 - a_rel * (1.0 - 0.19)) * c_ge77)
            + BI_other))
    if b <= 0 or s <= 0:
        return np.nan
    return float(np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s)))


def calc_figure_of_merit(nCaptures, threshold, ge77_muon_counts, runtime):
    """Figure of Merit from per-muon arrays (combines helpers above)."""
    ge_survival_eff     = 1.0 - calc_germanium_efficiency(nCaptures, threshold, ge77_muon_counts)
    signal_survival_eff = 1.0 - calc_deadtime(nCaptures, threshold, runtime)
    return figure_of_merit(ge_survival_eff, signal_survival_eff)


# ---------------------------------------------------------------------------
# Figure of Merit — confusion-matrix helpers
# ---------------------------------------------------------------------------

def calc_deadtime_confusion(
    n_positives: int,
    total_muons: int,
    musun_rate: float = MUSUN_RATE,
) -> float:
    """Dead-time fraction from confusion-matrix counts.

    Parameters
    ----------
    n_positives : int
        TP + FP — muons classified as Ge77 and therefore vetoed.
    total_muons : int
        TP + FP + TN + FN — total simulated muons.
    musun_rate : float
        Musun rate in muons / hour (default: 504).

    Returns
    -------
    float
        Dead-time fraction; 0.0 if total_muons == 0.
    """
    if total_muons <= 0:
        return 0.0
    runtime_h = total_muons / musun_rate
    veto_h    = n_positives * VETO_DURATION_H
    return veto_h / runtime_h


def calc_ge_survival_confusion(TP: int, FN: int) -> float:
    """Ge77-muon survival fraction from confusion-matrix counts.

    Returns ``1 − Recall = FN / (TP + FN)``: the fraction of true Ge77
    muons that are *not* detected and therefore survive the veto.
    Returns 1.0 when TP + FN == 0 (no Ge77 muons — all survive by default).
    """
    denom = TP + FN
    return FN / denom if denom > 0 else 1.0


def calc_fom_confusion(
    TP: int,
    FP: int,
    FN: int,
    total_muons: int,
    musun_rate: float = MUSUN_RATE,
) -> float:
    """Figure of Merit computed from a confusion matrix.

    Convenience wrapper that calls:
      - ``calc_ge_survival_confusion``  →  ge_survival_eff  =  1 − Recall
      - ``calc_deadtime_confusion``     →  deadtime
      - ``figure_of_merit``             →  FoM

    Parameters
    ----------
    TP, FP, FN : int
        Confusion-matrix counts (TN is not required).
    total_muons : int
        TP + FP + TN + FN — total simulated muons.
    musun_rate : float
        Musun rate in muons / hour (default: 504).

    Returns
    -------
    float
        FoM value, or np.nan when undefined.
    """
    ge_surv  = calc_ge_survival_confusion(TP, FN)
    deadtime = calc_deadtime_confusion(TP + FP, total_muons, musun_rate)
    return figure_of_merit(ge_surv, 1.0 - deadtime)