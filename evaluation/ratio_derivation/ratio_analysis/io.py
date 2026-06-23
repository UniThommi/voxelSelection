"""Zone scan optimizer and result serialization."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from .zones import Zone
from .photon_filters import PMT_CATHODE_RADIUS, MC_SAMPLES


# ─── Zone scan ────────────────────────────────────────────────────────────────

def scan_optimal_zones(
    ssd_counts_by_n: Dict[int, Dict[str, int]],
    pmt_counts_by_n: Dict[int, Dict[str, float]],
    ssd_nc: int,
    pmt_nc: int,
    area_name: str,
    zone_configs: Dict[int, List[Zone]],
    snr_threshold: float = 3.0,
    min_pmts_per_zone: int = 8,
) -> Tuple[int, Dict]:
    """Find optimal zone count for one area via SNR scan.

    For each N_zones computes corrected ratios, σ_zone (signal), ⟨σ_R⟩ (noise),
    and SNR = σ_zone / ⟨σ_R⟩. Returns the largest N with SNR ≥ threshold.

    Returns:
        (optimal_n, scan_data_dict)
    """
    print(f"\n  Zone scan for {area_name}:")
    print(f"  {'N_ZONES':>8} {'σ_zone':>10} {'⟨σ_R⟩':>10} {'SNR':>8} {'STATUS':>10}")
    print(f"  " + "-" * 50)

    best_n = 2
    scan_data = {'n_zones': [], 'sigma_zone': [], 'mean_sigma_r': [], 'snr': []}
    consecutive_fails = 0

    for n_zones in sorted(zone_configs.keys()):
        zones  = zone_configs[n_zones]
        ssd_c  = ssd_counts_by_n[n_zones]
        pmt_c  = pmt_counts_by_n[n_zones]

        min_eff = min(z.effective_n_pmts for z in zones)
        if min_eff < min_pmts_per_zone:
            print(f"  {n_zones:>8} {'--':>10} {'--':>10} {'--':>8} "
                  f"{'<{0}PMTs'.format(min_pmts_per_zone):>10}")
            consecutive_fails += 1
            if consecutive_fails >= 2:
                print(f"  ⛔ Aborting: 2 consecutive failures at N={n_zones}")
                break
            continue

        total_eff  = sum(z.effective_n_pmts for z in zones)
        total_area = sum(z.area_mm2 for z in zones)
        mean_density = total_eff / total_area if total_area > 0 else 0.0

        corr_ratios:       List[float] = []
        stat_uncertainties: List[float] = []

        for zone in zones:
            key   = f"{area_name}_{zone.zone_id}"
            n_ssd = ssd_c.get(key, 0)
            n_pmt = pmt_c.get(key, 0.0)

            if n_ssd <= 0 or n_pmt <= 0:
                continue

            ssd_per_nc = n_ssd / ssd_nc
            pmt_per_nc = n_pmt / pmt_nc
            raw_ratio  = ssd_per_nc / pmt_per_nc

            if mean_density > 0 and zone.pmt_density > 0:
                corr_ratio = raw_ratio * (zone.pmt_density / mean_density)
            else:
                continue

            rel_unc = np.sqrt(1.0 / n_ssd + 1.0 / n_pmt)
            abs_unc = corr_ratio * rel_unc

            corr_ratios.append(corr_ratio)
            stat_uncertainties.append(abs_unc)

        if len(corr_ratios) < 2:
            print(f"  {n_zones:>8} {'--':>10} {'--':>10} {'--':>8} {'skip':>10}")
            consecutive_fails += 1
            if consecutive_fails >= 2:
                print(f"  ⛔ Aborting: 2 consecutive failures at N={n_zones}")
                break
            continue

        sigma_zone   = np.std(corr_ratios, ddof=1)
        mean_sigma_r = np.mean(stat_uncertainties)
        snr = sigma_zone / mean_sigma_r if mean_sigma_r > 0 else 0.0

        passed = snr >= snr_threshold
        status = "✅" if passed else "❌"
        print(f"  {n_zones:>8} {sigma_zone:>10.4f} {mean_sigma_r:>10.4f} {snr:>8.2f} {status:>10}")

        scan_data['n_zones'].append(n_zones)
        scan_data['sigma_zone'].append(sigma_zone)
        scan_data['mean_sigma_r'].append(mean_sigma_r)
        scan_data['snr'].append(snr)

        if passed:
            best_n = n_zones
            consecutive_fails = 0
        else:
            consecutive_fails += 1
            if consecutive_fails >= 2:
                print(f"  ⛔ Aborting: 2 consecutive failures at N={n_zones}")
                break

    print(f"  → Optimal zones for {area_name}: {best_n} (SNR ≥ {snr_threshold})")
    return best_n, scan_data


# ─── Result serialization ─────────────────────────────────────────────────────

def save_results_txt(
    output_file: Path,
    results: List[Dict],
    pit_zones: List[Zone],
    top_zones: List[Zone],
    wall_zones: List[Zone],
    ssd_nc: int,
    pmt_nc: int,
    ssd_files: int,
    pmt_files: int,
    zone_scan_areas: List[str],
    snr_threshold: float,
    min_pmts_per_zone: int,
) -> None:
    """Write human-readable results table to text file."""
    header_ext = (f"{'AREA':<8} {'ZONE':<6} {'BOUNDARY':<30} {'SSD_PH':>12} "
                  f"{'PMT_PH':>12} {'PH/NC_SSD':>12} {'PH/NC_PMT':>12} "
                  f"{'RAW_RATIO':>10} {'CORR_RATIO':>10} {'EFF_PMTs':>10} "
                  f"{'AREA_mm2':>12} {'DENSITY':>14} {'DENS_DEV':>9}")
    sep = "-" * len(header_ext)

    with open(output_file, 'w') as fout:
        fout.write("=" * 130 + "\n")
        fout.write("Zone-based PMT vs. SSD Photon Detection Efficiency Ratio "
                   "(RAW_RATIO, CORR_RATIO = PMT/SSD)\n")
        fout.write("=" * 130 + "\n\n")
        fout.write(f"Zone config: Pit={len(pit_zones)}, Wall={len(wall_zones)}, "
                   f"Top={len(top_zones)}, Bot=1\n")
        if zone_scan_areas:
            fout.write(f"Zone scan areas: {zone_scan_areas}\n")
            fout.write(f"SNR threshold: {snr_threshold}, "
                       f"Min PMTs/zone: {min_pmts_per_zone}\n")
        fout.write(f"PMT cathode radius: {PMT_CATHODE_RADIUS} mm\n")
        fout.write(f"MC samples: {MC_SAMPLES}\n")
        fout.write(f"Files: SSD={ssd_files}, PMT={pmt_files}\n")
        fout.write(f"NC events: SSD={ssd_nc:,}, PMT={pmt_nc:,}\n\n")
        fout.write(header_ext + "\n")
        fout.write(sep + "\n")
        for r in results:
            z = r['zone']
            if z.area_name in ('pit', 'bot', 'top'):
                bstr = f"r=[{z.boundary_low:.0f}, {z.boundary_high:.0f}]mm"
            else:
                bstr = f"z=[{z.boundary_low:.0f}, {z.boundary_high:.0f}]mm"
            fout.write(f"{z.area_name:<8} {z.zone_id:<6} {bstr:<30} "
                       f"{r['ssd_photons']:>12d} {r['pmt_photons']:>12.1f} "
                       f"{r['ssd_per_nc']:>12.6f} {r['pmt_per_nc']:>12.6f} "
                       f"{r['ratio']:>10.4f} {r['corr_ratio']:>10.4f} "
                       f"{z.effective_n_pmts:>10.2f} {z.area_mm2:>12.0f} "
                       f"{z.pmt_density:>14.6e} {r['dens_dev']:>+8.1f}%\n")
        fout.write(sep + "\n")
        fout.write(f"\nFiles: SSD={ssd_files}, PMT={pmt_files}\n")
        fout.write(f"NC events: SSD={ssd_nc:,}, PMT={pmt_nc:,}\n")
        fout.write("=" * 130 + "\n")

    print(f"\nResults written to: {output_file}")


def save_results_json(
    output_file: Path,
    results: List[Dict],
    pit_zones: List[Zone],
    top_zones: List[Zone],
    wall_zones: List[Zone],
    pit_bounds: List[float],
    top_bounds: List[float],
    wall_bounds: List[float],
    bot_bounds: List[float],
    ssd_nc: int,
    pmt_nc: int,
    ssd_files: int,
    pmt_files: int,
    geometry_name: str,
) -> None:
    """Write machine-readable JSON results (used as reference for compare mode)."""
    json_results = {
        'metadata': {
            'zone_config': {
                'pit': len(pit_zones), 'top': len(top_zones),
                'wall': len(wall_zones), 'bot': 1,
            },
            'ssd_nc': ssd_nc, 'pmt_nc': pmt_nc,
            'ssd_files': ssd_files, 'pmt_files': pmt_files,
            'pmt_cathode_radius': PMT_CATHODE_RADIUS,
            'mc_samples': MC_SAMPLES,
            'geometry': geometry_name,
        },
        'boundaries': {
            'pit':  [float(b) for b in pit_bounds],
            'top':  [float(b) for b in top_bounds],
            'wall': [float(b) for b in wall_bounds],
            'bot':  [float(b) for b in bot_bounds],
        },
        'zones': [],
    }
    for r in results:
        z = r['zone']
        json_results['zones'].append({
            'area':             z.area_name,
            'zone_id':          z.zone_id,
            'boundary_low':     float(z.boundary_low),
            'boundary_high':    float(z.boundary_high),
            'area_mm2':         float(z.area_mm2),
            'effective_n_pmts': float(z.effective_n_pmts),
            'pmt_density':      float(z.pmt_density),
            'ssd_photons':      int(r['ssd_photons']),
            'pmt_photons':      float(r['pmt_photons']),
            'ssd_per_nc':       float(r['ssd_per_nc']),
            'pmt_per_nc':       float(r['pmt_per_nc']),
            'raw_ratio_pmt_over_ssd':  float(r['ratio'])      if not np.isnan(r['ratio'])      else None,
            'corr_ratio_pmt_over_ssd': float(r['corr_ratio']) if not np.isnan(r['corr_ratio']) else None,
            'dens_dev_pct': float(r['dens_dev']),
        })

    with open(output_file, 'w') as f_json:
        json.dump(json_results, f_json, indent=2)
    print(f"Machine-readable results written to: {output_file}")
