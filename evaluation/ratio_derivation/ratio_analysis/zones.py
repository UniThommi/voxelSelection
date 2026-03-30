"""Zone data structures, boundary optimization, and compare-mode helpers."""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .pmt_data import PMTInfo
from .photon_filters import PMT_RADIUS, MC_SAMPLES
from .geometry import GeometryConfig


# =============================================================================
# ZONE DATA STRUCTURE
# =============================================================================

@dataclass
class Zone:
    """A single zone within a detector area."""
    zone_id: int
    area_name: str
    boundary_low: float    # r_min (radial zones) or z_min (wall zones)
    boundary_high: float   # r_max or z_max
    area_mm2: float
    pmt_fractions: Optional[Dict[str, float]] = None  # pmt_index → fraction
    effective_n_pmts: float = 0.0

    @property
    def pmt_density(self) -> float:
        return self.effective_n_pmts / self.area_mm2 if self.area_mm2 > 0 else 0.0


# =============================================================================
# ZONE ASSIGNMENT HELPERS
# =============================================================================

def assign_radial_zone(r: np.ndarray, boundaries: List[float]) -> np.ndarray:
    """Assign radial positions to zone IDs via searchsorted."""
    ids = np.searchsorted(boundaries, r, side='right') - 1
    n_zones = len(boundaries) - 1
    ids = np.clip(ids, 0, n_zones - 1)
    outside = (r < boundaries[0]) | (r > boundaries[-1])
    ids[outside] = -1
    return ids


def assign_z_zone(z: np.ndarray, boundaries: List[float]) -> np.ndarray:
    """Assign z positions to zone IDs via searchsorted."""
    ids = np.searchsorted(boundaries, z, side='right') - 1
    n_zones = len(boundaries) - 1
    ids = np.clip(ids, 0, n_zones - 1)
    outside = (z < boundaries[0]) | (z > boundaries[-1])
    ids[outside] = -1
    return ids


# =============================================================================
# FRACTIONAL PMT OVERLAP (Monte Carlo)
# =============================================================================

def compute_radial_fractions(
    pmt: PMTInfo,
    boundaries: List[float],
    n_samples: int = MC_SAMPLES,
) -> Dict[int, float]:
    """Fraction of PMT cathode area in each radial zone via MC sampling."""
    rng = np.random.default_rng(seed=hash(pmt.index) % (2**31))
    angles = rng.uniform(0, 2 * np.pi, n_samples)
    radii  = PMT_RADIUS * np.sqrt(rng.uniform(0, 1, n_samples))

    sample_x = pmt.center[0] + radii * np.cos(angles)
    sample_y = pmt.center[1] + radii * np.sin(angles)
    sample_r = np.sqrt(sample_x**2 + sample_y**2)

    fractions: Dict[int, float] = {}
    n_zones = len(boundaries) - 1
    for zone_id in range(n_zones):
        r_low  = boundaries[zone_id]
        r_high = boundaries[zone_id + 1]
        if zone_id == n_zones - 1:
            mask = (sample_r >= r_low) & (sample_r <= r_high)
        else:
            mask = (sample_r >= r_low) & (sample_r < r_high)
        frac = float(np.sum(mask)) / n_samples
        if frac > 0:
            fractions[zone_id] = frac
    return fractions


def compute_z_fractions(
    pmt: PMTInfo,
    boundaries: List[float],
    n_samples: int = MC_SAMPLES,
) -> Dict[int, float]:
    """Fraction of PMT cathode area in each z-zone via MC sampling (wall PMTs)."""
    rng = np.random.default_rng(seed=hash(pmt.index) % (2**31))
    angles = rng.uniform(0, 2 * np.pi, n_samples)
    radii  = PMT_RADIUS * np.sqrt(rng.uniform(0, 1, n_samples))
    sample_z = pmt.z + radii * np.cos(angles)

    fractions: Dict[int, float] = {}
    n_zones = len(boundaries) - 1
    for zone_id in range(n_zones):
        z_low  = boundaries[zone_id]
        z_high = boundaries[zone_id + 1]
        if zone_id == n_zones - 1:
            mask = (sample_z >= z_low) & (sample_z <= z_high)
        else:
            mask = (sample_z >= z_low) & (sample_z < z_high)
        frac = float(np.sum(mask)) / n_samples
        if frac > 0:
            fractions[zone_id] = frac
    return fractions


# =============================================================================
# ZONE BUILDERS (iterative density-equalization optimization)
# =============================================================================

def build_radial_zones(
    pmts: List[PMTInfo],
    n_zones: int,
    r_min: float,
    r_max: float,
    area_name: str,
    max_iter: int = 200,
    tol: float = 0.001,
) -> Tuple[List[Zone], List[float]]:
    """Build radial zones with equal-density PMT boundaries."""
    if n_zones == 1:
        total_area = np.pi * (r_max**2 - r_min**2)
        fracs_all: Dict[str, float] = {}
        eff = 0.0
        bounds = [r_min, r_max]
        for pmt in pmts:
            f = compute_radial_fractions(pmt, bounds)
            if 0 in f:
                fracs_all[pmt.index] = f[0]
                eff += f[0]
        return [Zone(
            zone_id=0, area_name=area_name,
            boundary_low=r_min, boundary_high=r_max,
            area_mm2=total_area, pmt_fractions=fracs_all,
            effective_n_pmts=eff,
        )], bounds

    n_total = len(pmts)
    total_area = np.pi * (r_max**2 - r_min**2)
    target_density = n_total / total_area

    inner_bounds = np.array(
        [np.sqrt(r_min**2 + i / n_zones * (r_max**2 - r_min**2))
         for i in range(1, n_zones)],
        dtype=np.float64,
    )

    def evaluate(inner_b):
        bounds = [r_min] + list(inner_b) + [r_max]
        zones_data = []
        for i in range(n_zones):
            area = np.pi * (bounds[i+1]**2 - bounds[i]**2)
            zones_data.append({'area': area, 'eff': 0.0, 'fracs': {}})
        for pmt in pmts:
            fracs = compute_radial_fractions(pmt, bounds)
            for zid, frac in fracs.items():
                if 0 <= zid < n_zones:
                    zones_data[zid]['eff'] += frac
                    zones_data[zid]['fracs'][pmt.index] = frac
        densities = [z['eff'] / z['area'] if z['area'] > 0 else 0.0 for z in zones_data]
        return zones_data, densities, bounds

    best_bounds = inner_bounds.copy()
    best_max_dev = float('inf')

    for iteration in range(max_iter):
        zones_data, densities, bounds = evaluate(inner_bounds)
        mean_d = np.mean(densities)
        if mean_d == 0:
            break
        devs = [abs(d - mean_d) / mean_d for d in densities]
        max_dev = max(devs)

        if max_dev < best_max_dev:
            best_max_dev = max_dev
            best_bounds = inner_bounds.copy()

        if max_dev < tol:
            break

        new_bounds = inner_bounds.copy()
        for b_idx in range(len(inner_bounds)):
            d_left  = densities[b_idx]
            d_right = densities[b_idx + 1]
            if mean_d == 0:
                continue
            imbalance = (d_left - d_right) / mean_d
            step = 0.3 * (1.0 - iteration / max_iter)
            r_current = inner_bounds[b_idx]
            r2_range  = r_max**2 - r_min**2
            delta_r2  = imbalance * step * r2_range / n_zones
            new_r2 = r_current**2 + delta_r2
            r_low  = r_min if b_idx == 0 else inner_bounds[b_idx - 1]
            r_high = r_max if b_idx == len(inner_bounds) - 1 else inner_bounds[b_idx + 1]
            new_r2 = np.clip(new_r2, (r_low + 50)**2, (r_high - 50)**2)
            new_bounds[b_idx] = np.sqrt(new_r2)
        inner_bounds = new_bounds

    zones_data, densities, bounds = evaluate(best_bounds)
    mean_d = np.mean(densities)

    zones = [
        Zone(
            zone_id=i, area_name=area_name,
            boundary_low=bounds[i], boundary_high=bounds[i+1],
            area_mm2=zones_data[i]['area'],
            pmt_fractions=zones_data[i]['fracs'],
            effective_n_pmts=zones_data[i]['eff'],
        )
        for i in range(n_zones)
    ]

    max_dev_pct = max(abs(d - mean_d) / mean_d * 100 for d in densities) if mean_d > 0 else 0
    print(f"  {area_name}: {n_zones} optimized zones, "
          f"target density = {target_density:.6e} PMTs/mm²")
    for z in zones:
        dev = abs(z.pmt_density - mean_d) / mean_d * 100 if mean_d > 0 else 0
        print(f"    Zone {z.zone_id}: r=[{z.boundary_low:.1f}, {z.boundary_high:.1f}]mm, "
              f"eff_PMTs={z.effective_n_pmts:.2f}, area={z.area_mm2:.0f}mm², "
              f"density={z.pmt_density:.6e} (dev={dev:.1f}%)")
    print(f"    Max density deviation: {max_dev_pct:.2f}% "
          f"(after {min(iteration+1, max_iter)} iterations)")

    return zones, bounds


def build_z_zones(
    pmts: List[PMTInfo],
    n_zones: int,
    z_min: float,
    z_max: float,
    area_name: str,
    r_zylinder: float,
    max_iter: int = 200,
    tol: float = 0.001,
) -> Tuple[List[Zone], List[float]]:
    """Build wall z-zones with equal-density PMT boundaries."""
    circumference = 2 * np.pi * r_zylinder

    if n_zones == 1:
        total_area = circumference * (z_max - z_min)
        fracs_all: Dict[str, float] = {}
        eff = 0.0
        bounds = [z_min, z_max]
        for pmt in pmts:
            f = compute_z_fractions(pmt, bounds)
            if 0 in f:
                fracs_all[pmt.index] = f[0]
                eff += f[0]
        return [Zone(
            zone_id=0, area_name=area_name,
            boundary_low=z_min, boundary_high=z_max,
            area_mm2=total_area, pmt_fractions=fracs_all,
            effective_n_pmts=eff,
        )], bounds

    n_total = len(pmts)
    total_area = circumference * (z_max - z_min)
    target_density = n_total / total_area

    dz = (z_max - z_min) / n_zones
    inner_bounds = np.array(
        [z_min + i * dz for i in range(1, n_zones)], dtype=np.float64
    )

    def evaluate(inner_b):
        bounds = [z_min] + list(inner_b) + [z_max]
        zones_data = []
        for i in range(n_zones):
            height = bounds[i+1] - bounds[i]
            area = circumference * height
            zones_data.append({'area': area, 'eff': 0.0, 'fracs': {}})
        for pmt in pmts:
            fracs = compute_z_fractions(pmt, bounds)
            for zid, frac in fracs.items():
                if 0 <= zid < n_zones:
                    zones_data[zid]['eff'] += frac
                    zones_data[zid]['fracs'][pmt.index] = frac
        densities = [z['eff'] / z['area'] if z['area'] > 0 else 0.0 for z in zones_data]
        return zones_data, densities, bounds

    best_bounds = inner_bounds.copy()
    best_max_dev = float('inf')

    for iteration in range(max_iter):
        zones_data, densities, bounds = evaluate(inner_bounds)
        mean_d = np.mean(densities)
        if mean_d == 0:
            break
        devs = [abs(d - mean_d) / mean_d for d in densities]
        max_dev = max(devs)

        if max_dev < best_max_dev:
            best_max_dev = max_dev
            best_bounds = inner_bounds.copy()

        if max_dev < tol:
            break

        new_bounds = inner_bounds.copy()
        for b_idx in range(len(inner_bounds)):
            d_left  = densities[b_idx]
            d_right = densities[b_idx + 1]
            if mean_d == 0:
                continue
            imbalance = (d_left - d_right) / mean_d
            step = 0.3 * (1.0 - iteration / max_iter)
            z_range   = z_max - z_min
            delta_z   = imbalance * step * z_range / n_zones
            new_z = inner_bounds[b_idx] + delta_z
            z_low  = z_min if b_idx == 0 else inner_bounds[b_idx - 1]
            z_high = z_max if b_idx == len(inner_bounds) - 1 else inner_bounds[b_idx + 1]
            new_z = np.clip(new_z, z_low + 100, z_high - 100)
            new_bounds[b_idx] = new_z
        inner_bounds = new_bounds

    zones_data, densities, bounds = evaluate(best_bounds)
    mean_d = np.mean(densities)

    zones = [
        Zone(
            zone_id=i, area_name=area_name,
            boundary_low=bounds[i], boundary_high=bounds[i+1],
            area_mm2=zones_data[i]['area'],
            pmt_fractions=zones_data[i]['fracs'],
            effective_n_pmts=zones_data[i]['eff'],
        )
        for i in range(n_zones)
    ]

    max_dev_pct = max(abs(d - mean_d) / mean_d * 100 for d in densities) if mean_d > 0 else 0
    print(f"  {area_name}: {n_zones} optimized zones, "
          f"target density = {target_density:.6e} PMTs/mm²")
    for z in zones:
        dev = abs(z.pmt_density - mean_d) / mean_d * 100 if mean_d > 0 else 0
        print(f"    Zone {z.zone_id}: z=[{z.boundary_low:.1f}, {z.boundary_high:.1f}]mm, "
              f"eff_PMTs={z.effective_n_pmts:.2f}, area={z.area_mm2:.0f}mm², "
              f"density={z.pmt_density:.6e} (dev={dev:.1f}%)")
    print(f"    Max density deviation: {max_dev_pct:.2f}% "
          f"(after {min(iteration+1, max_iter)} iterations)")

    return zones, bounds


# =============================================================================
# COMPARE-MODE HELPERS (load reference zone boundaries)
# =============================================================================

def load_reference_json(json_path: Path) -> Dict:
    """Load reference zone results from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def _build_fixed_radial_zones(
    pmts: List[PMTInfo],
    boundaries: List[float],
    area_name: str,
) -> List[Zone]:
    """Build radial zones with fixed boundaries, computing PMT fractions."""
    n_zones = len(boundaries) - 1
    zones = [
        Zone(
            zone_id=i, area_name=area_name,
            boundary_low=boundaries[i], boundary_high=boundaries[i+1],
            area_mm2=np.pi * (boundaries[i+1]**2 - boundaries[i]**2),
            pmt_fractions={}, effective_n_pmts=0.0,
        )
        for i in range(n_zones)
    ]
    for pmt in pmts:
        fracs = compute_radial_fractions(pmt, boundaries)
        for zid, frac in fracs.items():
            if 0 <= zid < n_zones:
                zones[zid].pmt_fractions[pmt.index] = frac
                zones[zid].effective_n_pmts += frac
    for z in zones:
        print(f"    {area_name} zone {z.zone_id}: "
              f"r=[{z.boundary_low:.1f}, {z.boundary_high:.1f}]mm, "
              f"eff_PMTs={z.effective_n_pmts:.2f}")
    return zones


def _build_fixed_z_zones(
    pmts: List[PMTInfo],
    boundaries: List[float],
    area_name: str,
    r_zylinder: float,
) -> List[Zone]:
    """Build z-zones with fixed boundaries, computing PMT fractions."""
    n_zones = len(boundaries) - 1
    circumference = 2 * np.pi * r_zylinder
    zones = [
        Zone(
            zone_id=i, area_name=area_name,
            boundary_low=boundaries[i], boundary_high=boundaries[i+1],
            area_mm2=circumference * (boundaries[i+1] - boundaries[i]),
            pmt_fractions={}, effective_n_pmts=0.0,
        )
        for i in range(n_zones)
    ]
    for pmt in pmts:
        fracs = compute_z_fractions(pmt, boundaries)
        for zid, frac in fracs.items():
            if 0 <= zid < n_zones:
                zones[zid].pmt_fractions[pmt.index] = frac
                zones[zid].effective_n_pmts += frac
    for z in zones:
        print(f"    {area_name} zone {z.zone_id}: "
              f"z=[{z.boundary_low:.1f}, {z.boundary_high:.1f}]mm, "
              f"eff_PMTs={z.effective_n_pmts:.2f}")
    return zones


def build_zones_from_reference(
    ref_data: Dict,
    pmts: List[PMTInfo],
    pmt_by_layer: Dict[str, List[PMTInfo]],
    geometry: GeometryConfig,
) -> Tuple[
    List[Zone], List[Zone], List[Zone], List[Zone],
    List[float], List[float], List[float], List[float],
]:
    """Rebuild zones using boundaries from reference JSON.

    Returns:
        pit_zones, top_zones, wall_zones, bot_zones,
        pit_bounds, top_bounds, wall_bounds, bot_bounds
    """
    bounds   = ref_data['boundaries']
    pit_bounds  = bounds['pit']
    top_bounds  = bounds['top']
    wall_bounds = bounds['wall']
    bot_bounds  = bounds['bot']

    print("\n  Rebuilding zones from reference boundaries...")

    pit_zones  = _build_fixed_radial_zones(pmt_by_layer['pit'],  pit_bounds,  "pit")
    top_zones  = _build_fixed_radial_zones(pmt_by_layer['top'],  top_bounds,  "top")
    wall_zones = _build_fixed_z_zones(pmt_by_layer['wall'], wall_bounds, "wall",
                                      geometry.r_zylinder)

    bot_zones = [Zone(
        zone_id=0, area_name="bot",
        boundary_low=bot_bounds[0], boundary_high=bot_bounds[1],
        area_mm2=geometry.area_bot,
        pmt_fractions={pmt.index: 1.0 for pmt in pmt_by_layer['bot']},
        effective_n_pmts=float(len(pmt_by_layer['bot'])),
    )]
    print(f"    bot: 1 zone, {len(pmt_by_layer['bot'])} PMTs")

    return (pit_zones, top_zones, wall_zones, bot_zones,
            pit_bounds, top_bounds, wall_bounds, bot_bounds)
