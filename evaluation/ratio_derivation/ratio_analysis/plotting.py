"""Visualization utilities for zone ratio analysis."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .pmt_data import PMTInfo
from .zones import Zone
from .photon_filters import PMT_CATHODE_RADIUS


def _draw_radial_zones(
    ax: plt.Axes,
    zones: List[Zone],
    pmts: List[PMTInfo],
    ratios: List[float],
    area_name: str,
    r_max: float,
    cmap,
    norm: plt.Normalize,
    label_fontsize: int = 8,
) -> None:
    """Draw radial zones (pit, top, bot) as an x-y ratio heatmap onto ``ax``.

    Shared by :func:`plot_radial_zones` (standalone) and
    :func:`plot_zone_ratio_summary` (combined panel). Does not add a colorbar.
    """
    for zone, ratio in zip(zones, ratios):
        width = zone.boundary_high - zone.boundary_low
        color = cmap(norm(ratio)) if not np.isnan(ratio) else 'grey'
        ring = mpatches.Annulus((0, 0), zone.boundary_high, width,
                                color=color, alpha=0.6)
        ax.add_patch(ring)

        r_mid = (zone.boundary_low + zone.boundary_high) / 2
        label = (f"Z{zone.zone_id}\n"
                 f"r=[{zone.boundary_low:.0f},{zone.boundary_high:.0f}]\n"
                 f"eff_PMTs={zone.effective_n_pmts:.1f}\n")
        label += f"ratio={ratio:.3f}" if not np.isnan(ratio) else "ratio=NaN"
        ax.text(0, r_mid, label, ha='center', va='center', fontsize=label_fontsize,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    for pmt in pmts:
        circle = plt.Circle((pmt.center[0], pmt.center[1]), PMT_CATHODE_RADIUS,
                             fill=False, edgecolor='blue', linewidth=0.5, alpha=0.7)
        ax.add_patch(circle)
        ax.plot(pmt.center[0], pmt.center[1], 'b.', markersize=1)

    for zone in zones:
        for r_val in [zone.boundary_low, zone.boundary_high]:
            if r_val > 0:
                c = plt.Circle((0, 0), r_val, fill=False,
                                edgecolor='black', linewidth=1.5, linestyle='--')
                ax.add_patch(c)

    ax.set_xlim(-r_max * 1.15, r_max * 1.15)
    ax.set_ylim(-r_max * 1.15, r_max * 1.15)
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    valid = [r for r in ratios if not np.isnan(r)]
    mean_str = f"{np.mean(valid):.3f}" if valid else "NaN"
    ax.set_title(f'{area_name.upper()} - PMT/SSD Ratio per Zone '
                 f'(mean={mean_str})')
    ax.grid(True, alpha=0.3)


def plot_radial_zones(
    zones: List[Zone],
    pmts: List[PMTInfo],
    ratios: List[float],
    area_name: str,
    r_min: float,
    r_max: float,
    output_path: Path,
    global_norm: Optional[plt.Normalize] = None,
) -> None:
    """Plot radial zones (pit, top, bot) as x-y view with PMT circles and ratio heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    cmap = plt.cm.RdYlGn_r
    norm = global_norm if global_norm is not None else plt.Normalize(0, 1)

    _draw_radial_zones(ax, zones, pmts, ratios, area_name, r_max, cmap, norm)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7, label='PMT/SSD Ratio')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")


def _draw_wall_zones(
    ax: plt.Axes,
    zones: List[Zone],
    pmts: List[PMTInfo],
    ratios: List[float],
    cmap,
    norm: plt.Normalize,
    label_fontsize: int = 8,
) -> None:
    """Draw wall zones as a phi-z ratio heatmap onto ``ax``.

    Shared by :func:`plot_wall_zones` (standalone) and
    :func:`plot_zone_ratio_summary` (combined panel). Does not add a colorbar.
    """
    for zone, ratio in zip(zones, ratios):
        color = cmap(norm(ratio)) if not np.isnan(ratio) else 'grey'
        rect = mpatches.Rectangle(
            (-np.pi, zone.boundary_low),
            2 * np.pi, zone.boundary_high - zone.boundary_low,
            color=color, alpha=0.6,
        )
        ax.add_patch(rect)

        z_mid = (zone.boundary_low + zone.boundary_high) / 2
        label = (f"Z{zone.zone_id}\n"
                 f"z=[{zone.boundary_low:.0f},{zone.boundary_high:.0f}]\n"
                 f"eff_PMTs={zone.effective_n_pmts:.1f}\n")
        label += f"ratio={ratio:.3f}" if not np.isnan(ratio) else "ratio=NaN"
        ax.text(0, z_mid, label, ha='center', va='center', fontsize=label_fontsize,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    for pmt in pmts:
        phi_center = np.arctan2(pmt.center[1], pmt.center[0])
        # The phi-z axes carry different units on very different scales, so a
        # data-coordinate circle cannot render round here. Draw the cathode as a
        # round marker with a fixed display size (cosmetic, not to geometric scale).
        ax.plot(phi_center, pmt.z, marker='o', markersize=6,
                markerfacecolor='none', markeredgecolor='blue',
                markeredgewidth=0.5, alpha=0.7, linestyle='none')
        ax.plot(phi_center, pmt.z, 'b.', markersize=1)

    for zone in zones:
        for z_val in [zone.boundary_low, zone.boundary_high]:
            ax.axhline(z_val, color='black', linewidth=1.5, linestyle='--')

    ax.set_xlim(-np.pi, np.pi)
    all_z = [z.boundary_low for z in zones] + [z.boundary_high for z in zones]
    margin = (max(all_z) - min(all_z)) * 0.05
    ax.set_ylim(min(all_z) - margin, max(all_z) + margin)
    ax.set_xlabel('φ [rad]')
    ax.set_ylabel('z [mm]')
    valid = [r for r in ratios if not np.isnan(r)]
    mean_str = f"{np.mean(valid):.3f}" if valid else "NaN"
    ax.set_title(f'WALL - PMT/SSD Ratio per Zone (φ-z view, mean={mean_str})')
    ax.grid(True, alpha=0.3)


def plot_wall_zones(
    zones: List[Zone],
    pmts: List[PMTInfo],
    ratios: List[float],
    r_zylinder: float,
    output_path: Path,
    global_norm: Optional[plt.Normalize] = None,
) -> None:
    """Plot wall zones as phi-z view with PMT circles and ratio heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    cmap = plt.cm.RdYlGn_r
    norm = global_norm if global_norm is not None else plt.Normalize(0, 1)

    _draw_wall_zones(ax, zones, pmts, ratios, cmap, norm)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7, label='PMT/SSD Ratio')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")


def plot_zone_ratio_summary(
    pit_zones: List[Zone],
    top_zones: List[Zone],
    bot_zones: List[Zone],
    wall_zones: List[Zone],
    pit_pmts: List[PMTInfo],
    top_pmts: List[PMTInfo],
    bot_pmts: List[PMTInfo],
    wall_pmts: List[PMTInfo],
    pit_ratios: List[float],
    top_ratios: List[float],
    bot_ratios: List[float],
    wall_ratios: List[float],
    r_pit: float,
    r_zylinder: float,
    output_path: Path,
    global_norm: Optional[plt.Normalize] = None,
) -> None:
    """Summary figure: all 4 detector-area ratio maps in a single 2×2 panel.

    Combines the three radial views (pit, top, bot) and the wall φ-z view,
    sharing one colormap and colorbar so the corrected PMT/SSD ratios are
    directly comparable across areas.
    """
    cmap = plt.cm.RdYlGn_r
    norm = global_norm if global_norm is not None else plt.Normalize(0, 1)

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))

    _draw_radial_zones(axes[0, 0], pit_zones, pit_pmts, pit_ratios,
                       'pit', r_pit, cmap, norm, label_fontsize=7)
    _draw_radial_zones(axes[0, 1], top_zones, top_pmts, top_ratios,
                       'top', r_zylinder, cmap, norm, label_fontsize=7)
    _draw_radial_zones(axes[1, 0], bot_zones, bot_pmts, bot_ratios,
                       'bot', r_zylinder, cmap, norm, label_fontsize=7)
    _draw_wall_zones(axes[1, 1], wall_zones, wall_pmts, wall_ratios,
                     cmap, norm, label_fontsize=7)

    fig.suptitle('PMT/SSD Corrected Ratio per Zone — All Detector Areas',
                 fontsize=16, fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.6, label='PMT/SSD Ratio')

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def plot_snr_scan(
    scan_results: Dict[str, Dict],
    snr_threshold: float,
    optimal_zones: Dict[str, int],
    output_path: Path,
) -> None:
    """Plot SNR vs N_zones for each area (signal vs noise + SNR bar chart)."""
    areas = sorted(scan_results.keys())
    fig, axes = plt.subplots(len(areas), 2, figsize=(14, 4 * len(areas)),
                              squeeze=False)

    for row, area_name in enumerate(areas):
        data = scan_results[area_name]
        if not data['n_zones']:
            continue

        n_arr        = np.array(data['n_zones'])
        sigma_zone   = np.array(data['sigma_zone'])
        mean_sigma_r = np.array(data['mean_sigma_r'])
        snr          = np.array(data['snr'])
        opt_n        = optimal_zones.get(area_name, 2)

        ax1 = axes[row, 0]
        ax1.plot(n_arr, sigma_zone,   'bo-', label='σ_zone (signal)', linewidth=2)
        ax1.plot(n_arr, mean_sigma_r, 'r^-', label='⟨σ_R⟩ (noise)',   linewidth=2)
        ax1.axvline(opt_n, color='green', linestyle='--', linewidth=1.5,
                    label=f'optimal N={opt_n}')
        ax1.set_xlabel('N_zones')
        ax1.set_ylabel('Ratio units')
        ax1.set_title(f'{area_name.upper()} — Signal vs Noise')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(n_arr)

        ax2 = axes[row, 1]
        colors = ['green' if s >= snr_threshold else 'red' for s in snr]
        ax2.bar(n_arr, snr, color=colors, alpha=0.7, width=0.6)
        ax2.axhline(snr_threshold, color='black', linestyle='--', linewidth=1.5,
                    label=f'threshold = {snr_threshold}')
        ax2.axvline(opt_n, color='green', linestyle='--', linewidth=1.5,
                    label=f'optimal N={opt_n}')
        ax2.set_xlabel('N_zones')
        ax2.set_ylabel('SNR')
        ax2.set_title(f'{area_name.upper()} — SNR = σ_zone / ⟨σ_R⟩')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(n_arr)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")


def plot_comparison(
    results_new: List[Dict],
    ref_data: Dict,
    output_dir: Path,
) -> None:
    """Bar chart comparison between new and reference corrected ratios (compare mode)."""
    ref_lookup: Dict[Tuple[str, int], float] = {}
    for z in ref_data['zones']:
        cr = z['corr_ratio_pmt_over_ssd']
        ref_lookup[(z['area'], z['zone_id'])] = cr if cr is not None else float('nan')

    areas_order = ['pit', 'bot', 'top', 'wall']
    results_by_area: Dict[str, List[Dict]] = {a: [] for a in areas_order}
    for r in results_new:
        area = r['zone'].area_name
        if area in results_by_area:
            results_by_area[area].append(r)

    for area_name in areas_order:
        area_results = sorted(results_by_area[area_name],
                              key=lambda r: r['zone'].zone_id)
        if not area_results:
            continue

        zone_ids   = [r['zone'].zone_id for r in area_results]
        new_ratios = [r['corr_ratio'] for r in area_results]
        ref_ratios = [ref_lookup.get((area_name, zid), float('nan'))
                      for zid in zone_ids]
        diffs = [n - ref if not (np.isnan(n) or np.isnan(ref)) else float('nan')
                 for n, ref in zip(new_ratios, ref_ratios)]

        labels = []
        for r in area_results:
            z = r['zone']
            if area_name in ('pit', 'bot', 'top'):
                labels.append(f"Z{z.zone_id}\nr=[{z.boundary_low:.0f},\n{z.boundary_high:.0f}]")
            else:
                labels.append(f"Z{z.zone_id}\nz=[{z.boundary_low:.0f},\n{z.boundary_high:.0f}]")

        x     = np.arange(len(zone_ids))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(zone_ids) * 1.5), 6))

        ax1.bar(x - width/2, ref_ratios, width, label='Reference',
                color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, new_ratios, width, label='Musun NCs',
                color='coral',     alpha=0.8)
        ax1.set_xlabel('Zone')
        ax1.set_ylabel('Corrected PMT/SSD Ratio')
        ax1.set_title(f'{area_name.upper()} — Corrected Ratio: Reference vs Musun NCs')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=7)
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)

        colors_diff = ['green' if not np.isnan(d) and d >= 0
                       else 'red' if not np.isnan(d) else 'grey'
                       for d in diffs]
        diffs_plot = [d if not np.isnan(d) else 0 for d in diffs]
        ax2.bar(x, diffs_plot, width=0.5, color=colors_diff, alpha=0.8)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_xlabel('Zone')
        ax2.set_ylabel('Δ Corrected Ratio (new − ref)')
        ax2.set_title(f'{area_name.upper()} — Absolute Difference per Zone')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=7)
        ax2.grid(True, axis='y', alpha=0.3)

        for i, (d, label) in enumerate(zip(diffs, labels)):
            if not np.isnan(d):
                ax2.text(x[i], d, f'{d:+.4f}', ha='center',
                         va='bottom' if d >= 0 else 'top', fontsize=7)

        plt.tight_layout()
        plt.savefig(output_dir / f"comparison_{area_name}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Comparison plot saved: comparison_{area_name}.png")

    # Summary plot: all zones in one figure
    all_new, all_ref, all_labels = [], [], []
    for area_name in areas_order:
        area_results = sorted(results_by_area[area_name],
                              key=lambda r: r['zone'].zone_id)
        for r in area_results:
            z = r['zone']
            all_new.append(r['corr_ratio'])
            all_ref.append(ref_lookup.get((area_name, z.zone_id), float('nan')))
            all_labels.append(f"{area_name[0].upper()}{z.zone_id}")

    all_diffs = [n - ref if not (np.isnan(n) or np.isnan(ref)) else float('nan')
                 for n, ref in zip(all_new, all_ref)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, len(all_labels) * 0.6), 10))
    x = np.arange(len(all_labels))

    ax1.bar(x - 0.2, all_ref, 0.4, label='Reference', color='steelblue', alpha=0.8)
    ax1.bar(x + 0.2, all_new, 0.4, label='Musun NCs', color='coral',     alpha=0.8)
    ax1.set_ylabel('Corrected PMT/SSD Ratio')
    ax1.set_title('All Zones — Corrected Ratio: Reference vs Musun NCs')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_labels, fontsize=7, rotation=45)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    colors_diff = ['green' if not np.isnan(d) and d >= 0
                   else 'red' if not np.isnan(d)
                   else 'grey' for d in all_diffs]
    diffs_plot = [d if not np.isnan(d) else 0 for d in all_diffs]
    ax2.bar(x, diffs_plot, 0.5, color=colors_diff, alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel('Δ Corrected Ratio (new − ref)')
    ax2.set_title('All Zones — Absolute Difference')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_labels, fontsize=7, rotation=45)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_all_zones.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Summary comparison plot saved: comparison_all_zones.png")


def plot_area_flux(
    area_results: List[Dict],
    area_name: str,
    output_path: Path,
) -> None:
    """Diagnostic: area-normalized photon flux per zone (SSD vs PMT)."""
    if not area_results:
        return

    mids = [(r['zone'].boundary_low + r['zone'].boundary_high) / 2
            for r in area_results]
    xlabel = 'z [mm]' if area_name == 'wall' else 'r [mm]'

    ssd_flux = [r['ssd_per_nc'] / r['zone'].area_mm2 for r in area_results]
    pmt_flux = [r['pmt_per_nc'] / r['zone'].pmt_density
                if r['zone'].pmt_density > 0 else 0.0
                for r in area_results]

    color_ssd = 'tab:blue'
    color_pmt = 'tab:red'

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(mids, ssd_flux, 'o-', color=color_ssd,
             linewidth=2, markersize=6, label='SSD: photons/NC/mm²')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('SSD photons / NC / mm²', color=color_ssd)
    ax1.tick_params(axis='y', labelcolor=color_ssd)

    ax2 = ax1.twinx()
    ax2.plot(mids, pmt_flux, 's-', color=color_pmt,
             linewidth=2, markersize=6, label='PMT: photons/NC/(PMT/mm²)')
    ax2.set_ylabel('PMT photons / NC / (PMT/mm²)', color=color_pmt)
    ax2.tick_params(axis='y', labelcolor=color_pmt)

    ax1.set_title(f'{area_name.upper()}: Area-normalized photon flux (SSD vs PMT)')
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")
