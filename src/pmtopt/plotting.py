"""
Plotting functions for greedy voxel selection results.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from .geometry import (
    PMT_RADIUS, R_PIT, R_ZYL_BOT, R_ZYL_TOP, R_ZYLINDER,
    Z_BASE_GLOBAL, H_ZYLINDER, is_valid_pmt_position,
)


def _draw_voxels_on_ax(
    ax,
    selected_centers: np.ndarray,
    selected_layers: np.ndarray,
    *,
    title: "str | None" = None,
    legend_boundaries: bool = True,
    legend_fontsize: int = 10,
    label_fontsize: int = 13,
    title_fontsize: int = 13,
) -> tuple[list[int], list[tuple[int, int, float]]]:
    """Render the 3D detector scatter of selected voxels onto an existing axis.

    PMTs are colored/marked per area; the per-area count appears only in the
    legend label. Returns ``(boundary_failures, distance_violations)`` so the
    caller can report or fold the counts into a title.
    """
    Z_BASE = Z_BASE_GLOBAL
    Z_TOP = Z_BASE + H_ZYLINDER

    theta = np.linspace(0, 2 * np.pi, 200)
    n_vert = 24
    theta_lines = np.linspace(0, 2 * np.pi, n_vert, endpoint=False)

    for z in [Z_BASE, Z_TOP]:
        ax.plot(R_ZYLINDER * np.cos(theta), R_ZYLINDER * np.sin(theta), z,
                color="gray", alpha=0.3, linewidth=0.5)
    for t in theta_lines:
        ax.plot([R_ZYLINDER * np.cos(t)] * 2, [R_ZYLINDER * np.sin(t)] * 2,
                [Z_BASE, Z_TOP], color="gray", alpha=0.3, linewidth=0.5)

    pit_kw = dict(color="blue", alpha=0.7, linewidth=1.2)
    bot_kw = dict(color="green", alpha=0.7, linewidth=1.2)
    if legend_boundaries:
        pit_kw["label"] = f"Pit boundary (r = {R_PIT} mm)"
        bot_kw["label"] = f"Bot inner boundary (r = {R_ZYL_BOT} mm)"
    ax.plot(R_PIT * np.cos(theta), R_PIT * np.sin(theta), Z_BASE, **pit_kw)
    ax.plot(R_ZYL_BOT * np.cos(theta), R_ZYL_BOT * np.sin(theta), Z_BASE, **bot_kw)

    layer_markers = {"pit": "o", "bot": "s", "top": "^", "wall": "D"}
    layer_colors = {"pit": "#1f77b4", "bot": "#2ca02c", "top": "#ff7f0e", "wall": "#d62728"}
    layer_edge_colors = {"pit": "#0a3d6b", "bot": "#0a4d0a", "top": "#7f3f00", "wall": "#7f0000"}

    for layer in ["pit", "bot", "top", "wall"]:
        mask = selected_layers == layer
        if not np.any(mask):
            continue
        pts = selected_centers[mask]
        color = layer_colors.get(layer, "red")
        edge_color = layer_edge_colors.get(layer, "darkred")
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=color, marker=layer_markers.get(layer, "o"),
            s=40, alpha=0.85, edgecolors=edge_color, linewidths=0.5,
            label=f"{layer.capitalize()}  (N = {int(mask.sum())})",
        )

    boundary_failures = []
    for i, (c, lay) in enumerate(zip(selected_centers, selected_layers)):
        if not is_valid_pmt_position(c, lay):
            boundary_failures.append(i)

    distance_violations = []
    min_dist = 2 * PMT_RADIUS
    n = len(selected_centers)
    for i in range(n):
        diffs = selected_centers[i + 1:] - selected_centers[i]
        dists = np.linalg.norm(diffs, axis=1)
        too_close = np.where(dists < min_dist)[0]
        for idx in too_close:
            j = i + 1 + idx
            distance_violations.append((i, j, dists[idx]))

    if boundary_failures:
        fail_pts = selected_centers[boundary_failures]
        ax.scatter(
            fail_pts[:, 0], fail_pts[:, 1], fail_pts[:, 2],
            c="red", marker="x", s=120, linewidths=2,
            label=f"Boundary violations ({len(boundary_failures)})",
        )

    if distance_violations:
        for i, j, d in distance_violations:
            p1, p2 = selected_centers[i], selected_centers[j]
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color="red", linewidth=2, alpha=0.8,
            )

    ax.set_xlabel("x (mm)", fontsize=label_fontsize, labelpad=8)
    ax.set_ylabel("y (mm)", fontsize=label_fontsize, labelpad=8)
    ax.set_zlabel("z (mm)", fontsize=label_fontsize, labelpad=8)
    ax.tick_params(axis="both", labelsize=max(8, label_fontsize - 3))

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, pad=12)

    ax.legend(loc="upper left", fontsize=legend_fontsize, framealpha=0.85)

    max_range = max(R_ZYLINDER, (Z_TOP - Z_BASE) / 2)
    mid_z = (Z_BASE + Z_TOP) / 2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    return boundary_failures, distance_violations


def _print_selection_stats(
    selected_centers: np.ndarray,
    selected_layers: np.ndarray,
    boundary_failures: list[int],
    distance_violations: list[tuple[int, int, float]],
) -> None:
    """Print boundary/distance checks and per-area extent for a selection."""
    print(f"  Boundary check: {'PASS' if not boundary_failures else 'FAIL'}")
    print(f"  Distance check: {'PASS' if not distance_violations else 'FAIL'}")
    for layer in ["pit", "bot", "top", "wall"]:
        count = np.sum(selected_layers == layer)
        if count > 0:
            pts = selected_centers[selected_layers == layer]
            r_vals = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            print(f"  {layer:>4}: {count:>4} voxels "
                  f"(r: {r_vals.min():.0f}-{r_vals.max():.0f} mm, "
                  f"z: {pts[:, 2].min():.0f}-{pts[:, 2].max():.0f} mm)")


def plot_selected_voxels(
    selected_centers: np.ndarray,
    selected_layers: np.ndarray,
    selected_ids: list[str],
    output_path: Path,
    title_extra: str = "",
    w2: "float | None" = None,
) -> None:
    """3D scatter plot of selected voxel positions."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    boundary_failures, distance_violations = _draw_voxels_on_ax(
        ax, selected_centers, selected_layers, title=None,
    )

    title_parts = [f"Selected PMT Positions  (N = {len(selected_centers)})"]
    if w2 is not None:
        title_parts.append(f"W₂ = {w2:.1f} mm")
    if boundary_failures:
        title_parts.append(f"boundary fails = {len(boundary_failures)}")
    if distance_violations:
        title_parts.append(f"dist violations = {len(distance_violations)}")
    title_line1 = "   |   ".join(title_parts)
    title_str = title_line1 + (f"\n{title_extra}" if title_extra else "")
    ax.set_title(title_str, fontsize=13, pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nPlot saved to {output_path}")
    _print_selection_stats(selected_centers, selected_layers,
                           boundary_failures, distance_violations)


# Allowed panels-per-figure and their grid shapes / figure sizes.
_GRID_LAYOUT = {4: (2, 2), 2: (1, 2), 1: (1, 1)}
_GRID_FIGSIZE = {4: (20, 16), 2: (22, 9), 1: (14, 10)}


def _pack_group_sizes(n: int) -> list[int]:
    """Split *n* setups into figures of 4, 2, or 1 panels (never 3).

    Greedy-by-4, then the remainder: 3 → [2, 1]; 2 → [2]; 1 → [1].
    """
    if n <= 0:
        return []
    groups = [4] * (n // 4)
    rem = n % 4
    if rem == 3:
        groups += [2, 1]
    elif rem in (1, 2):
        groups.append(rem)
    return groups


def plot_selected_voxels_grid(
    setups: list[dict],
    output_dir: Path,
    name: str = "combined_grid",
) -> list[Path]:
    """Tile multiple selections into grid figures (4 / 2 / 1 panels each).

    Each entry of *setups* is a dict with keys ``centers`` (N×3 array),
    ``layers`` (N array), ``label`` (str) and optional ``w2`` (float). Each
    panel reuses the standard 3D detector scatter with a per-panel legend
    (per-area counts) and a per-panel title (label, N, W₂).

    Returns the list of written PNG paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = _pack_group_sizes(len(setups))
    n_figs = len(sizes)
    written: list[Path] = []

    cursor = 0
    for fig_idx, size in enumerate(sizes):
        nrows, ncols = _GRID_LAYOUT[size]
        fig = plt.figure(figsize=_GRID_FIGSIZE[size])

        for panel in range(size):
            s = setups[cursor + panel]
            centers = s["centers"]
            layers = s["layers"]
            ax = fig.add_subplot(nrows, ncols, panel + 1, projection="3d")

            title_parts = [str(s["label"]), f"N = {len(centers)}"]
            if s.get("w2") is not None:
                title_parts.append(f"W₂ = {s['w2']:.1f} mm")
            title = "   |   ".join(title_parts)

            _draw_voxels_on_ax(
                ax, centers, layers, title=title,
                legend_boundaries=False,
                legend_fontsize=8, label_fontsize=10, title_fontsize=12,
            )

        cursor += size

        out_name = f"{name}.png" if n_figs == 1 else f"{name}_{fig_idx + 1:02d}.png"
        out_path = output_dir / out_name
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
        print(f"Grid figure saved to {out_path}  ({size} setup(s))")

    return written


def plot_muon_nc_histogram(
    muon_det_counts: np.ndarray,
    output_path: Path,
    title_extra: str = "",
) -> None:
    """Histogram of detected NCs per muon."""
    fig, ax = plt.subplots(figsize=(10, 6))

    max_count = int(muon_det_counts.max()) if len(muon_det_counts) > 0 else 0
    bins = np.arange(0, max_count + 2) - 0.5

    ax.hist(muon_det_counts, bins=bins, edgecolor="black", linewidth=0.5,
            color="#1976d2", alpha=0.85)
    ax.set_xlabel("Detected NCs per muon", fontsize=12)
    ax.set_ylabel("Number of muons", fontsize=12)
    ax.set_title(f"Muon NC Detection Distribution"
                 + (f"\n{title_extra}" if title_extra else ""),
                 fontsize=13)
    ax.set_xlim(-0.5, min(max_count + 1.5, 50.5))
    ax.grid(True, axis="y", alpha=0.3)

    n_total = len(muon_det_counts)
    n_zero = int(np.sum(muon_det_counts == 0))
    mean_d = float(np.mean(muon_det_counts)) if n_total > 0 else 0.0
    median_d = float(np.median(muon_det_counts)) if n_total > 0 else 0.0
    textstr = (f"Total muons: {n_total:,}\n"
               f"Undetected (d=0): {n_zero:,}\n"
               f"Mean d: {mean_d:.1f}\n"
               f"Median d: {median_d:.0f}")
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Muon NC histogram saved to {output_path}")


def plot_jaccard_curves(
    results: dict,
    k_values: list[int],
    output_path: Path,
    optimize_mode: str,
) -> None:
    """Plot J_k(δ) curves: one line per δ, x-axis = k."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.RdYlGn_r
    deltas_sorted = sorted(results.keys(), key=lambda d: abs(d))

    for delta in deltas_sorted:
        color = cmap(abs(delta) / 0.25)
        label = f"δ = {delta:+.0%}"
        jvals = results[delta]["jaccard_curve"]
        ax.plot(k_values, jvals, marker="o", markersize=3,
                label=label, color=color, linewidth=1.5)

    ax.set_xlabel("k (top-k voxels)", fontsize=12)
    ax.set_ylabel("Jaccard Similarity $J_k(\\delta)$", fontsize=12)
    ax.set_title(f"Sensitivity Analysis: Jaccard vs. Top-k "
                 f"(mode: {optimize_mode})", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(k_values[0] - 5, k_values[-1] + 5)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5,
               label="J = 0.8 threshold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Jaccard curve plot saved to {output_path}")


def plot_coverage_change(
    results: dict,
    output_path: Path,
    optimize_mode: str,
) -> None:
    """Bar chart of relative coverage change ΔC(δ)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    deltas_sorted = sorted(results.keys())
    delta_c = [results[d]["delta_coverage"] for d in deltas_sorted]
    labels = [f"{d:+.0%}" for d in deltas_sorted]
    colors = ["#d32f2f" if dc < -0.01
              else "#388e3c" if dc > 0.01
              else "#757575" for dc in delta_c]

    bars = ax.bar(labels, [dc * 100 for dc in delta_c], color=colors,
                  edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Ratio perturbation δ", fontsize=12)
    ax.set_ylabel("ΔC(δ) [%]", fontsize=12)
    ax.set_title(f"Relative Coverage Change "
                 f"(mode: {optimize_mode})", fontsize=13)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, dc in zip(bars, delta_c):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{dc:+.3%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Coverage change plot saved to {output_path}")


def plot_per_area_jaccard(
    results: dict,
    output_path: Path,
    optimize_mode: str,
) -> None:
    """Grouped bar chart: per-area Jaccard for each δ."""
    areas = ["pit", "bot", "top", "wall"]
    deltas_sorted = sorted(results.keys())
    n_deltas = len(deltas_sorted)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_deltas)
    width = 0.18

    area_colors = {"pit": "#1976d2", "bot": "#388e3c",
                   "top": "#f57c00", "wall": "#7b1fa2"}

    for i, area in enumerate(areas):
        vals = [results[d]["per_area_jaccard"][area] for d in deltas_sorted]
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width, label=area.upper(),
               color=area_colors[area], edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Ratio perturbation δ", fontsize=12)
    ax.set_ylabel("Jaccard Similarity", fontsize=12)
    ax.set_title(f"Per-Area Jaccard Similarity "
                 f"(mode: {optimize_mode})", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:+.0%}" for d in deltas_sorted])
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Per-area Jaccard plot saved to {output_path}")


def _fmt_primaries(num_primaries: int) -> str:
    """Format a large integer as a clean exponential string."""
    import math
    exp = int(round(math.log10(num_primaries))) if num_primaries > 0 else 0
    if 10 ** exp == num_primaries:
        return f"$10^{{{exp}}}$"
    return f"{num_primaries:,}"


def _draw_area_patches(
    ax,
    centers: np.ndarray,
    hits: np.ndarray,
    area: str,
    half: float,
    phi_half: float,
    cmap,
    norm,
) -> None:
    """Draw colored 195×195 mm² voxel rectangles onto ax for one area."""
    if area == "wall":
        for (cx, cy, cz), h in zip(centers, hits):
            phi = np.arctan2(cy, cx)
            ax.add_patch(mpatches.Rectangle(
                (phi - phi_half, cz - half), 2 * phi_half, 2 * half,
                facecolor=cmap(norm(h)), edgecolor="none",
            ))
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(Z_BASE_GLOBAL, Z_BASE_GLOBAL + H_ZYLINDER)
        ax.set_xlabel(r"$\varphi$ [rad]", fontsize=11)
        ax.set_ylabel("z [mm]", fontsize=11)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels(
            [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
        )
    else:
        for (cx, cy, _), h in zip(centers, hits):
            ax.add_patch(mpatches.Rectangle(
                (cx - half, cy - half), 2 * half, 2 * half,
                facecolor=cmap(norm(h)), edgecolor="none",
            ))
        if area == "pit":
            ax.add_patch(mpatches.Circle(
                (0, 0), R_PIT, fill=False, edgecolor="black", linewidth=1.2))
            lim = R_PIT * 1.1
        elif area == "bot":
            for r in (R_ZYL_BOT, R_ZYLINDER):
                ax.add_patch(mpatches.Circle(
                    (0, 0), r, fill=False, edgecolor="black", linewidth=1.2))
            lim = R_ZYLINDER * 1.1
        elif area == "top":
            for r in (R_ZYL_TOP, R_ZYLINDER):
                ax.add_patch(mpatches.Circle(
                    (0, 0), r, fill=False, edgecolor="black", linewidth=1.2))
            lim = R_ZYLINDER * 1.1
        else:
            lim = R_ZYLINDER * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]", fontsize=10)
        ax.set_ylabel("y [mm]", fontsize=10)


def plot_hit_heatmap(
    areas_data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    voxel_size_mm: float = 195.0,
    num_primaries: int | None = None,
    num_ncs: int | None = None,
) -> None:
    """Heatmap of total photon hits per voxel.

    Layout: wall (large, top row spanning full width);
    pit / bot / top as three equal sub-panels in the bottom row.
    All panels share the same color scale.
    """
    half = voxel_size_mm / 2.0
    phi_half = half / R_ZYLINDER  # arc → radians for the unrolled wall

    all_hits = np.concatenate([h for _, h in areas_data.values() if len(h) > 0])
    vmin, vmax = float(all_hits.min()), float(all_hits.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mcm.viridis

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.5, 1], hspace=0.45, wspace=0.38)
    ax_wall = fig.add_subplot(gs[0, :])
    ax_pit  = fig.add_subplot(gs[1, 0])
    ax_bot  = fig.add_subplot(gs[1, 1])
    ax_top  = fig.add_subplot(gs[1, 2])
    area_axes = {"wall": ax_wall, "pit": ax_pit, "bot": ax_bot, "top": ax_top}

    for area, ax in area_axes.items():
        if area not in areas_data or len(areas_data[area][0]) == 0:
            ax.set_visible(False)
            continue
        centers_a, hits_a = areas_data[area]
        _draw_area_patches(ax, centers_a, hits_a, area, half, phi_half, cmap, norm)
        ax.set_title(f"{area.capitalize()}  ({len(centers_a):,} voxels)", fontsize=11)
        ax.grid(True, alpha=0.15)

    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=[ax_wall, ax_pit, ax_bot, ax_top], shrink=0.85, pad=0.03,
    )
    cbar.set_label("Total photon hits (summed over all NC events)", fontsize=11)

    title = "SSD Hit Distribution — Total Photon Hits per Voxel"
    if num_primaries is not None and num_ncs is not None:
        subtitle = (f"{_fmt_primaries(num_primaries)} simulated muons"
                    f"  ·  {num_ncs:,} neutron captures")
        fig.suptitle(f"{title}\n{subtitle}", fontsize=14, y=0.995)
    else:
        fig.suptitle(title, fontsize=14)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Hit heatmap saved to {output_path}")


def plot_marginal_gain_heatmap(
    gains_raw: np.ndarray,
    available: np.ndarray,
    selected_cols: list[int],
    centers: np.ndarray,
    layers: np.ndarray,
    output_path: Path,
    voxel_size_mm: float = 195.0,
    step: int = 10,
    invalid_centers: np.ndarray | None = None,
    invalid_layers: np.ndarray | None = None,
) -> None:
    """Heatmap of marginal gain per voxel after `step` greedy selections.

    Color coding:
      • available voxels  — viridis colormap scaled to [0, max_gain]
      • already selected  — bright red (#FF4444), clearly distinct
      • spacing-excluded  — medium gray (#AAAAAA)
    """
    half = voxel_size_mm / 2.0
    phi_half = half / R_ZYLINDER

    selected_set = set(selected_cols)
    avail_gains = gains_raw[available]
    vmax_g = float(avail_gains.max()) if len(avail_gains) > 0 else 1.0
    norm_g = mcolors.Normalize(vmin=0, vmax=max(vmax_g, 1))
    cmap_g = mcm.viridis

    SELECTED_COLOR = "#FF4444"
    EXCLUDED_COLOR = "#AAAAAA"

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.5, 1], hspace=0.45, wspace=0.38)
    ax_wall = fig.add_subplot(gs[0, :])
    ax_pit  = fig.add_subplot(gs[1, 0])
    ax_bot  = fig.add_subplot(gs[1, 1])
    ax_top  = fig.add_subplot(gs[1, 2])
    area_axes = {"wall": ax_wall, "pit": ax_pit, "bot": ax_bot, "top": ax_top}

    area_lims: dict[str, float] = {}
    layer_to_ax = {a: ax for a, ax in area_axes.items()}

    # Draw invalid voxels first (gray, underneath the main layer)
    n_invalid = 0
    if invalid_centers is not None and len(invalid_centers) > 0:
        n_invalid = len(invalid_centers)
        for (cx, cy, cz), layer in zip(invalid_centers, invalid_layers):
            area = str(layer)
            ax = layer_to_ax.get(area)
            if ax is None:
                continue
            if area == "wall":
                phi = np.arctan2(cy, cx)
                ax.add_patch(mpatches.Rectangle(
                    (phi - phi_half, cz - half), 2 * phi_half, 2 * half,
                    facecolor=EXCLUDED_COLOR, edgecolor="none",
                ))
            else:
                ax.add_patch(mpatches.Rectangle(
                    (cx - half, cy - half), 2 * half, 2 * half,
                    facecolor=EXCLUDED_COLOR, edgecolor="none",
                ))

    for i, (cx, cy, cz) in enumerate(centers):
        area = str(layers[i])
        ax = layer_to_ax.get(area)
        if ax is None:
            continue

        if i in selected_set:
            fc = SELECTED_COLOR
        elif not available[i]:
            fc = EXCLUDED_COLOR
        else:
            fc = cmap_g(norm_g(gains_raw[i]))

        if area == "wall":
            phi = np.arctan2(cy, cx)
            ax.add_patch(mpatches.Rectangle(
                (phi - phi_half, cz - half), 2 * phi_half, 2 * half,
                facecolor=fc, edgecolor="none",
            ))
        else:
            ax.add_patch(mpatches.Rectangle(
                (cx - half, cy - half), 2 * half, 2 * half,
                facecolor=fc, edgecolor="none",
            ))
            # Track max radius for limits
            r = np.hypot(cx, cy)
            area_lims[area] = max(area_lims.get(area, 0.0), r)

    # Finish wall axes
    ax_wall.set_xlim(-np.pi, np.pi)
    ax_wall.set_ylim(Z_BASE_GLOBAL, Z_BASE_GLOBAL + H_ZYLINDER)
    ax_wall.set_xlabel(r"$\varphi$ [rad]", fontsize=11)
    ax_wall.set_ylabel("z [mm]", fontsize=11)
    ax_wall.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax_wall.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax_wall.set_title(f"Wall", fontsize=11)
    ax_wall.grid(True, alpha=0.15)

    # Finish flat-area axes
    for area, ax in [("pit", ax_pit), ("bot", ax_bot), ("top", ax_top)]:
        if area not in layers:
            ax.set_visible(False)
            continue
        if area == "pit":
            ax.add_patch(mpatches.Circle(
                (0, 0), R_PIT, fill=False, edgecolor="black", linewidth=1.2))
            lim = R_PIT * 1.1
        elif area == "bot":
            for r in (R_ZYL_BOT, R_ZYLINDER):
                ax.add_patch(mpatches.Circle(
                    (0, 0), r, fill=False, edgecolor="black", linewidth=1.2))
            lim = R_ZYLINDER * 1.1
        elif area == "top":
            for r in (R_ZYL_TOP, R_ZYLINDER):
                ax.add_patch(mpatches.Circle(
                    (0, 0), r, fill=False, edgecolor="black", linewidth=1.2))
            lim = R_ZYLINDER * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]", fontsize=10)
        ax.set_ylabel("y [mm]", fontsize=10)
        ax.set_title(f"{area.capitalize()}", fontsize=11)
        ax.grid(True, alpha=0.15)

    # Colorbar for available voxels
    sm = mcm.ScalarMappable(norm=norm_g, cmap=cmap_g)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=[ax_wall, ax_pit, ax_bot, ax_top], shrink=0.85, pad=0.03,
    )
    cbar.set_label("Marginal gain (additional NCs covered)", fontsize=11)

    # Legend patches
    n_avail = int(available.sum())
    n_excl  = int((~available).sum()) - len(selected_set)
    legend_patches = [
        mpatches.Patch(facecolor=SELECTED_COLOR, label=f"Selected ({len(selected_set)})"),
        mpatches.Patch(facecolor=EXCLUDED_COLOR,
                       label=f"Spacing-excluded ({n_excl:,}) / invalid ({n_invalid:,})"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=10, framealpha=0.85, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        f"Marginal Gain after {step} Greedy Selections\n"
        f"Available: {n_avail:,}  ·  Selected: {len(selected_set)}  ·  "
        f"Spacing-excluded: {n_excl:,}  ·  Invalid: {n_invalid:,}",
        fontsize=14, y=0.995,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Marginal gain heatmap saved to {output_path}")


def _setup_flat_area_axis(ax, area: str) -> None:
    """Draw boundary circles and set limits/labels for a flat (x-y) sub-surface."""
    if area == "pit":
        ax.add_patch(mpatches.Circle(
            (0, 0), R_PIT, fill=False, edgecolor="black", linewidth=1.2))
        lim = R_PIT * 1.1
    elif area == "bot":
        for r in (R_ZYL_BOT, R_ZYLINDER):
            ax.add_patch(mpatches.Circle(
                (0, 0), r, fill=False, edgecolor="black", linewidth=1.2))
        lim = R_ZYLINDER * 1.1
    elif area == "top":
        for r in (R_ZYL_TOP, R_ZYLINDER):
            ax.add_patch(mpatches.Circle(
                (0, 0), r, fill=False, edgecolor="black", linewidth=1.2))
        lim = R_ZYLINDER * 1.1
    else:
        lim = R_ZYLINDER * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]", fontsize=10)
    ax.set_ylabel("y [mm]", fontsize=10)


def _draw_area_voxels(
    ax,
    centers: np.ndarray,
    area: str,
    color: str,
    half: float,
    phi_half: float,
    edgecolor: str = "black",
    alpha: float = 0.65,
) -> None:
    """Draw uniformly-colored voxel rectangles for one sub-surface onto ax."""
    if area == "wall":
        for cx, cy, cz in centers:
            phi = np.arctan2(cy, cx)
            ax.add_patch(mpatches.Rectangle(
                (phi - phi_half, cz - half), 2 * phi_half, 2 * half,
                facecolor=color, edgecolor=edgecolor, linewidth=0.2, alpha=alpha,
            ))
    else:
        for cx, cy, _ in centers:
            ax.add_patch(mpatches.Rectangle(
                (cx - half, cy - half), 2 * half, 2 * half,
                facecolor=color, edgecolor=edgecolor, linewidth=0.2, alpha=alpha,
            ))


def plot_ssd_voxels_2d(
    centers: np.ndarray,
    layers: np.ndarray,
    output_path: Path,
    voxel_size_mm: float = 195.0,
    invalid_centers: np.ndarray | None = None,
    invalid_layers: np.ndarray | None = None,
) -> None:
    """2D per-sub-surface view of all SSD voxels (pit / bot / top / wall).

    Layout matches the hit heatmaps: wall (unrolled) spans the top row;
    pit / bot / top occupy the bottom row. Valid (PMT-placeable) voxels are
    drawn cyan, voxels that are invalid from the start of the greedy selection
    (``invalid_centers`` / ``invalid_layers``) grey. Each panel title reports
    the per-sub-surface invalid count and total (valid + invalid).
    """
    half = voxel_size_mm / 2.0
    phi_half = half / R_ZYLINDER  # arc → radians for the unrolled wall

    if invalid_centers is None:
        invalid_centers = np.empty((0, 3), dtype=np.float64)
        invalid_layers = np.empty(0, dtype=object)

    VALID_COLOR = "cyan"
    INVALID_COLOR = "#AAAAAA"

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.5, 1], hspace=0.45, wspace=0.38)
    ax_wall = fig.add_subplot(gs[0, :])
    ax_pit  = fig.add_subplot(gs[1, 0])
    ax_bot  = fig.add_subplot(gs[1, 1])
    ax_top  = fig.add_subplot(gs[1, 2])
    area_axes = {"wall": ax_wall, "pit": ax_pit, "bot": ax_bot, "top": ax_top}

    for area, ax in area_axes.items():
        v_mask = layers == area
        i_mask = invalid_layers == area
        v_centers = centers[v_mask]
        i_centers = invalid_centers[i_mask]
        n_valid = len(v_centers)
        n_invalid = len(i_centers)
        n_total = n_valid + n_invalid

        # Invalid first (grey), then valid (cyan) on top
        _draw_area_voxels(ax, i_centers, area, INVALID_COLOR, half, phi_half,
                          edgecolor="#666666", alpha=0.5)
        _draw_area_voxels(ax, v_centers, area, VALID_COLOR, half, phi_half,
                          edgecolor="black", alpha=0.65)

        if area == "wall":
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(Z_BASE_GLOBAL, Z_BASE_GLOBAL + H_ZYLINDER)
            ax.set_xlabel(r"$\varphi$ [rad]", fontsize=11)
            ax.set_ylabel("z [mm]", fontsize=11)
            ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            ax.set_xticklabels(
                [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        else:
            _setup_flat_area_axis(ax, area)

        ax.set_title(
            f"{area.capitalize()} — invalid: {n_invalid:,} / total: {n_total:,}",
            fontsize=11,
        )
        ax.grid(True, alpha=0.15)

    legend_handles = [
        mpatches.Patch(facecolor=VALID_COLOR, edgecolor="black", label="Valid"),
        mpatches.Patch(facecolor=INVALID_COLOR, edgecolor="#666666",
                       label="Invalid"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=11, framealpha=0.85, bbox_to_anchor=(0.5, 0.01))

    n_valid_tot = len(centers)
    n_invalid_tot = len(invalid_centers)
    fig.suptitle(
        f"SSD Voxel Geometry (2D) — valid: {n_valid_tot:,}  ·  "
        f"invalid: {n_invalid_tot:,}  ·  total: {n_valid_tot + n_invalid_tot:,}",
        fontsize=14, y=0.995,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  2D voxel geometry plot saved to {output_path}")


def _voxel_polys(
    centers: np.ndarray,
    layers: np.ndarray,
    half: float,
) -> list[list]:
    """Build the 3D quad polygons for a set of voxels.

    Wall voxels: vertical patch in the tangential-z plane (normal = radial).
    Pit / bot / top voxels: horizontal patch in the x-y plane.
    """
    polys: list[list] = []
    for (cx, cy, cz), layer in zip(centers, layers):
        if layer == "wall":
            r = np.hypot(cx, cy)
            if r < 1e-6:
                continue
            tx, ty = -cy / r, cx / r  # tangent direction
            polys.append([
                [cx + half * tx, cy + half * ty, cz - half],
                [cx + half * tx, cy + half * ty, cz + half],
                [cx - half * tx, cy - half * ty, cz + half],
                [cx - half * tx, cy - half * ty, cz - half],
            ])
        else:  # pit, bot, top — horizontal
            polys.append([
                [cx - half, cy - half, cz],
                [cx + half, cy - half, cz],
                [cx + half, cy + half, cz],
                [cx - half, cy + half, cz],
            ])
    return polys


def plot_ssd_voxels_3d(
    centers: np.ndarray,
    layers: np.ndarray,
    output_path: Path,
    voxel_size_mm: float = 195.0,
    invalid_centers: np.ndarray | None = None,
    invalid_layers: np.ndarray | None = None,
) -> None:
    """3D plot of all SSD voxels as 195×195 mm² patches.

    Wall voxels: vertical patch in the tangential-z plane (normal = radial).
    Pit / bot / top voxels: horizontal patch in the x-y plane.

    Valid (PMT-placeable) voxels are drawn cyan. Voxels that are invalid from
    the start of the greedy selection (``invalid_centers`` / ``invalid_layers``,
    as returned by :func:`load_invalid_voxel_data`) are drawn grey.
    """
    half = voxel_size_mm / 2.0

    n_invalid = 0 if invalid_centers is None else len(invalid_centers)

    polys = _voxel_polys(centers, layers, half)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Invalid voxels first (grey, underneath the valid layer)
    if n_invalid > 0:
        inv_polys = _voxel_polys(invalid_centers, invalid_layers, half)
        if inv_polys:
            inv_coll = Poly3DCollection(inv_polys, alpha=0.4, linewidths=0.3)
            inv_coll.set_facecolor("#AAAAAA")
            inv_coll.set_edgecolor("#666666")
            ax.add_collection3d(inv_coll)

    coll = Poly3DCollection(polys, alpha=0.65, linewidths=0.3)
    coll.set_facecolor("cyan")
    coll.set_edgecolor("black")
    ax.add_collection3d(coll)

    # Cylinder outline for orientation
    theta = np.linspace(0, 2 * np.pi, 200)
    Z_top = Z_BASE_GLOBAL + H_ZYLINDER
    for z_lvl in [Z_BASE_GLOBAL, Z_top]:
        ax.plot(
            R_ZYLINDER * np.cos(theta), R_ZYLINDER * np.sin(theta), z_lvl,
            color="gray", alpha=0.3, linewidth=0.5,
        )

    ax.set_xlabel("x [mm]", fontsize=10)
    ax.set_ylabel("y [mm]", fontsize=10)
    ax.set_zlabel("z [mm]", fontsize=10)

    legend_handles = [
        mpatches.Patch(facecolor="cyan", edgecolor="black",
                       label=f"Valid ({len(centers):,})"),
    ]
    if n_invalid > 0:
        legend_handles.append(
            mpatches.Patch(facecolor="#AAAAAA", edgecolor="#666666",
                           label=f"Invalid ({n_invalid:,})")
        )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10,
              framealpha=0.85)

    layer_counts = {a: int(np.sum(layers == a))
                    for a in ["pit", "bot", "top", "wall"]}
    count_lines = "  ".join(
        f"{a.upper()}: {n}" for a, n in layer_counts.items() if n > 0
    )
    title = f"SSD Voxel Geometry  (valid N={len(centers):,}"
    if n_invalid > 0:
        title += f"  ·  invalid N={n_invalid:,}"
    title += f")\n{count_lines}"
    ax.set_title(title, fontsize=12)

    R_lim = R_ZYLINDER * 1.1
    ax.set_xlim(-R_lim, R_lim)
    ax.set_ylim(-R_lim, R_lim)
    ax.set_zlim(Z_BASE_GLOBAL - 200, Z_top + 200)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  3D voxel geometry plot saved to {output_path}")