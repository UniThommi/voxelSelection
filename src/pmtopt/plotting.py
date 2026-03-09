"""
Plotting functions for greedy voxel selection results.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from .geometry import (
    PMT_RADIUS, R_PIT, R_ZYL_BOT, R_ZYLINDER,
    Z_BASE_GLOBAL, H_ZYLINDER, is_valid_pmt_position,
)


def plot_selected_voxels(
    selected_centers: np.ndarray,
    selected_layers: np.ndarray,
    selected_ids: list[str],
    output_path: Path,
    title_extra: str = "",
) -> None:
    """3D scatter plot of selected voxel positions."""
    Z_BASE = Z_BASE_GLOBAL
    Z_TOP = Z_BASE + H_ZYLINDER

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    theta = np.linspace(0, 2 * np.pi, 200)
    n_vert = 24
    theta_lines = np.linspace(0, 2 * np.pi, n_vert, endpoint=False)

    for z in [Z_BASE, Z_TOP]:
        ax.plot(R_ZYLINDER * np.cos(theta), R_ZYLINDER * np.sin(theta), z,
                color="gray", alpha=0.3, linewidth=0.5)
    for t in theta_lines:
        ax.plot([R_ZYLINDER * np.cos(t)] * 2, [R_ZYLINDER * np.sin(t)] * 2,
                [Z_BASE, Z_TOP], color="gray", alpha=0.3, linewidth=0.5)

    ax.plot(R_PIT * np.cos(theta), R_PIT * np.sin(theta), Z_BASE,
            color="blue", alpha=0.7, linewidth=1.2, label=f"Pit (r={R_PIT})")
    ax.plot(R_ZYL_BOT * np.cos(theta), R_ZYL_BOT * np.sin(theta), Z_BASE,
            color="green", alpha=0.7, linewidth=1.2,
            label=f"Bot ring inner (r={R_ZYL_BOT})")

    layer_markers = {"pit": "o", "bot": "s", "top": "^", "wall": "D"}

    for layer in ["pit", "bot", "top", "wall"]:
        mask = selected_layers == layer
        if not np.any(mask):
            continue
        pts = selected_centers[mask]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c="red", marker=layer_markers.get(layer, "o"),
            s=30, alpha=0.8, edgecolors="darkred", linewidths=0.5,
            label=f"Selected ({layer}: {mask.sum()})",
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
            c="yellow", marker="x", s=100, linewidths=2,
            label=f"Boundary violations ({len(boundary_failures)})",
        )

    if distance_violations:
        for i, j, d in distance_violations:
            p1, p2 = selected_centers[i], selected_centers[j]
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color="yellow", linewidth=2, alpha=0.8,
            )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(
        f"Selected Voxels (N={len(selected_centers)}, "
        f"boundary fails={len(boundary_failures)}, "
        f"dist violations={len(distance_violations)})"
        + (f"\n{title_extra}" if title_extra else "")
    )
    ax.legend(loc="upper left", fontsize=8)

    area_counts = {area: int(np.sum(selected_layers == area))
                   for area in ["pit", "bot", "top", "wall"]}
    count_str = "\n".join(f"{a.upper():>4}: {cnt:>3} PMTs"
                          for a, cnt in area_counts.items())
    count_str += f"\n{'Total':>4}: {len(selected_centers):>3} PMTs"
    ax.text2D(0.98, 0.50, count_str, transform=ax.transAxes, fontsize=9,
              verticalalignment="center", horizontalalignment="right",
              fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                        edgecolor="gray", alpha=0.85))

    max_range = max(R_ZYLINDER, (Z_TOP - Z_BASE) / 2)
    mid_z = (Z_BASE + Z_TOP) / 2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nPlot saved to {output_path}")
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