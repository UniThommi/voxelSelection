"""
Shared W2-correlation plotting helper and Wasserstein explanation visualizations.

Used by both:
  src/pmtopt/evaluate_coverages.py  (SSD / pmtopt pipeline)
  evaluation/ratio_derivation/compare_coverages.py  (LGDO pipeline)

Usage:
python src/pmtopt/main.py plot-w2 setup1.json setup2.json \
      --output-dir ./output \      
      --labels "Greedy" "Homogeneous".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist as _cdist

import ot as _ot

from pmtopt.geometry import (
    PMT_RADIUS, R_PIT, R_ZYL_BOT, R_ZYL_TOP, R_ZYLINDER,
    Z_BASE_GLOBAL, H_ZYLINDER, compute_per_area_N,
)
from pmtopt.homogeneous import (
    get_w2_ref, compute_wasserstein_homogeneity, sample_reference_distribution,
    fibonacci_disk, fibonacci_cylinder_wall,
    Z_BASE, H_CYLINDER, T_ZYLINDER, DZ_PIT,
)

# ── Per-area visual styles ────────────────────────────────────────────────────
_AREA_COLORS  = {"pit": "#e41a1c", "bot": "#377eb8", "top": "#4daf4a", "wall": "#ff7f00"}
_AREA_MARKERS = {"pit": "o", "bot": "s", "top": "^", "wall": "D"}

# ── Flat-surface z-coordinates (match sample_reference_distribution) ──────────
_Z_PIT      = float(Z_BASE + DZ_PIT / 2)
_Z_BOT      = float(Z_BASE + T_ZYLINDER / 2)
_Z_TOP_SURF = float(Z_BASE + H_CYLINDER + T_ZYLINDER / 2)
_Z_WMIN     = float(Z_BASE)
_Z_WMAX     = float(Z_BASE + H_CYLINDER)


# ─────────────────────────────────────────────────────────────────────────────
# W2 decomposition helpers (pure numpy — no external deps)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_w2_z(centers: np.ndarray) -> float:
    """1-D W2 between z-marginal of config and Uniform([_Z_WMIN, _Z_WMAX]).

    Reference: 10001 quantile levels of Uniform([Z_BASE, Z_BASE + H_CYLINDER]).
    Quantile function: Q_ref(q) = _Z_WMIN + (_Z_WMAX - _Z_WMIN) * q.
    """
    q = np.linspace(0.0, 1.0, 10001)
    q_cfg = np.quantile(centers[:, 2], q)
    q_ref = _Z_WMIN + (_Z_WMAX - _Z_WMIN) * q
    return float(np.sqrt(np.mean((q_cfg - q_ref) ** 2)))


def _compute_w2_phi(centers: np.ndarray) -> float:
    """Circular W2 vs Uniform([0,2π)) via optimal-rotation search over N cyclic shifts."""
    phi   = (np.arctan2(centers[:, 1], centers[:, 0]) + 2 * np.pi) % (2 * np.pi)
    N     = len(phi)
    phi_s = np.sort(phi)
    phi_ext = np.concatenate([phi_s, phi_s + 2 * np.pi])
    q_uni = 2 * np.pi * (np.arange(N) + 0.5) / N
    costs = np.array([np.mean((phi_ext[k:k + N] - q_uni) ** 2) for k in range(N)])
    return float(np.sqrt(costs.min()))


# ─────────────────────────────────────────────────────────────────────────────
# Existing helper — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

def regression_overlay(
    ax_scatter: plt.Axes,
    ax_resid: plt.Axes,
    w2_arr: np.ndarray,
    y_arr: np.ndarray,
    color_pts: list,
    labels: list[str],
    y_label: str,
    x_label: str = "Global W2 (mm)",
) -> None:
    """Shared scatter + OLS regression + residual panel for correlation plots."""
    n = len(w2_arr)

    for w2v, yv, c, lbl in zip(w2_arr, y_arr, color_pts, labels):
        ax_scatter.scatter([w2v], [yv], color=c, s=55, zorder=3)
        ax_scatter.annotate(
            lbl, xy=(w2v, yv), xytext=(4, 3),
            textcoords="offset points", fontsize=6, color=c,
        )

    if n >= 3 and np.std(w2_arr) > 0 and np.std(y_arr) > 0:
        try:
            r_val, p_r   = scipy_stats.pearsonr(w2_arr,  y_arr)
            rho,   p_rho = scipy_stats.spearmanr(w2_arr, y_arr)
            ann = (
                f"Pearson  r = {r_val:+.3f}  (p={p_r:.3g})\n"
                f"Spearman ρ = {rho:+.3f}  (p={p_rho:.3g})"
            )
        except ValueError:
            ann = "constant data — no stats"
    elif n < 3:
        ann = "n < 3 — no stats"
    else:
        ann = "constant data — no stats"

    ax_scatter.text(
        0.03, 0.97, ann,
        transform=ax_scatter.transAxes,
        ha="left", va="top", fontsize=7, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.85),
    )

    if n >= 3 and np.std(w2_arr) > 0 and np.std(y_arr) > 0:
        slope, intercept, *_ = scipy_stats.linregress(w2_arr, y_arr)
        x_fit  = np.linspace(w2_arr.min(), w2_arr.max(), 200)
        y_fit  = slope * x_fit + intercept
        ax_scatter.plot(x_fit, y_fit, color="black", linewidth=1.2,
                        linestyle="--", zorder=2)
        y_pred    = slope * w2_arr + intercept
        residuals = y_arr - y_pred
        se        = np.sqrt(np.sum(residuals ** 2) / max(n - 2, 1))
        x_mean    = w2_arr.mean()
        t_crit    = scipy_stats.t.ppf(0.975, df=max(n - 2, 1))
        ci = t_crit * se * np.sqrt(
            1 / n + (x_fit - x_mean) ** 2 / np.sum((w2_arr - x_mean) ** 2)
        )
        ax_scatter.fill_between(x_fit, y_fit - ci, y_fit + ci,
                                color="black", alpha=0.08)
        y_pred_pts = slope * w2_arr + intercept
        resids     = y_arr - y_pred_pts
        for w2v, rv, c in zip(w2_arr, resids, color_pts):
            ax_resid.scatter([w2v], [rv], color=c, s=40, zorder=3)
            ax_resid.plot([w2v, w2v], [0, rv], color=c, linewidth=0.8, alpha=0.6)
        ax_resid.axhline(0, color="black", linewidth=0.8, linestyle="--")
    else:
        ax_resid.text(0.5, 0.5, "n < 3", transform=ax_resid.transAxes,
                      ha="center", va="center", color="gray")

    ax_scatter.set_xlabel(x_label, fontsize=8)
    ax_scatter.set_ylabel(y_label, fontsize=8)
    ax_scatter.grid(alpha=0.3)
    ax_resid.set_xlabel(x_label, fontsize=8)
    ax_resid.set_ylabel("Residual", fontsize=8)
    ax_resid.grid(alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_pmt_json(json_path: str) -> tuple[np.ndarray, np.ndarray, str]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    voxels = data if isinstance(data, list) else data.get("selected_voxels", [])
    voxels = [v for v in voxels if isinstance(v, dict) and "center" in v]
    centers = np.array([v["center"] for v in voxels], dtype=float)
    layers  = np.array([v.get("layer", "wall") for v in voxels])
    return centers, layers, Path(json_path).stem


def _classify_ref_by_area(ref: np.ndarray) -> np.ndarray:
    """Classify 3000-point reference array into pit/bot/top/wall by geometry."""
    r = np.sqrt(ref[:, 0] ** 2 + ref[:, 1] ** 2)
    z = ref[:, 2]
    r_mid    = (R_PIT + R_ZYL_BOT) / 2          # ≈ 3875 mm
    mask_low = np.abs(z - _Z_PIT) < 50          # bottom flat surface
    mask_pit = mask_low & (r < r_mid)
    mask_bot = mask_low & ~mask_pit
    mask_top = np.abs(z - _Z_TOP_SURF) < 50     # top flat surface
    areas = np.full(len(ref), "wall", dtype=object)
    areas[mask_pit] = "pit"
    areas[mask_bot] = "bot"
    areas[mask_top] = "top"
    return areas


def _compute_ot_plan(
    centers: np.ndarray, ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (T, cost_sq) — full N×M optimal transport plan + squared-Euclidean cost."""
    N, M = len(centers), len(ref)
    a    = np.ones(N, dtype=np.float64) / N
    b    = np.ones(M, dtype=np.float64) / M
    cost = _cdist(centers.astype(np.float64), ref.astype(np.float64), metric="sqeuclidean")
    T    = _ot.emd(a, b, cost)
    return T, cost


def _draw_3d_wireframe(ax, z_base: float, z_top: float) -> None:
    theta = np.linspace(0, 2 * np.pi, 200)
    for z in [z_base, z_top]:
        ax.plot(R_ZYLINDER * np.cos(theta), R_ZYLINDER * np.sin(theta), z,
                color="gray", alpha=0.25, linewidth=0.5)
    for t in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        ax.plot([R_ZYLINDER * np.cos(t)] * 2, [R_ZYLINDER * np.sin(t)] * 2,
                [z_base, z_top], color="gray", alpha=0.2, linewidth=0.5)
    ax.plot(R_PIT * np.cos(theta), R_PIT * np.sin(theta), z_base,
            color="#1976d2", alpha=0.5, linewidth=1.0)


def _fibonacci_centers_for_N(N: int) -> np.ndarray:
    alloc = compute_per_area_N(N, verbose=False)
    parts: list[np.ndarray] = []
    if alloc.get("pit", 0):
        xy = fibonacci_disk(alloc["pit"], 0.0, float(R_PIT))
        parts.append(np.column_stack([xy, np.full(alloc["pit"], _Z_PIT)]))
    if alloc.get("bot", 0):
        xy = fibonacci_disk(alloc["bot"], float(R_ZYL_BOT), float(R_ZYLINDER))
        parts.append(np.column_stack([xy, np.full(alloc["bot"], _Z_BOT)]))
    if alloc.get("top", 0):
        xy = fibonacci_disk(alloc["top"], float(R_ZYL_TOP), float(R_ZYLINDER))
        parts.append(np.column_stack([xy, np.full(alloc["top"], _Z_TOP_SURF)]))
    if alloc.get("wall", 0):
        parts.append(fibonacci_cylinder_wall(alloc["wall"], float(R_ZYLINDER),
                                             _Z_WMIN, _Z_WMAX))
    return np.vstack(parts) if parts else np.empty((0, 3))


# ─────────────────────────────────────────────────────────────────────────────
# Plot A — 3D scatter: PMT positions vs reference distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_w2_3d_scatter(
    centers: np.ndarray,
    layers: np.ndarray,
    ref: np.ndarray,
    output_path: str,
    label: str = "",
    w2_global: Optional[float] = None,
) -> None:
    """Plot A — 3D scatter of PMT positions overlaid on the reference distribution."""
    z_base = float(Z_BASE_GLOBAL)
    z_top  = z_base + float(H_ZYLINDER)

    fig = plt.figure(figsize=(13, 9))
    ax  = fig.add_subplot(111, projection="3d")

    _draw_3d_wireframe(ax, z_base, z_top)

    # Reference points — small, gray, semi-transparent
    ref_areas = _classify_ref_by_area(ref)
    ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2],
               c="lightgray", s=4, alpha=0.25, linewidths=0, zorder=1,
               label=f"Reference (M={len(ref):,})")

    # PMT centers — colored by area
    for area in ["pit", "bot", "top", "wall"]:
        mask = layers == area
        if not mask.any():
            continue
        pts = centers[mask]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=_AREA_COLORS[area], marker=_AREA_MARKERS[area],
                   s=40, alpha=0.9, edgecolors="black", linewidths=0.3,
                   zorder=3, label=f"PMT {area} (N={mask.sum()})")

    w2_str = f"  W2_global = {w2_global:.1f} mm" if w2_global is not None else ""
    ax.set_title(
        f"PMT Distribution vs Uniform Reference\n{label}{w2_str}",
        fontsize=13,
    )
    ax.set_xlabel("x (mm)", fontsize=9)
    ax.set_ylabel("y (mm)", fontsize=9)
    ax.set_zlabel("z (mm)", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)

    half = max(float(R_ZYLINDER), float(H_ZYLINDER) / 2)
    mid_z = (z_base + z_top) / 2
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_zlim(mid_z - half, mid_z + half)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {Path(output_path).name}")


def plot_w2_3d_comparison(
    centers_best: np.ndarray, layers_best: np.ndarray,
    centers_worst: np.ndarray, layers_worst: np.ndarray,
    ref: np.ndarray,
    output_path: str,
    label_best: str = "Low W2", label_worst: str = "High W2",
    w2_best: Optional[float] = None, w2_worst: Optional[float] = None,
) -> None:
    """Plot A (comparison) — side-by-side 3D scatter: best vs worst W2 config."""
    z_base = float(Z_BASE_GLOBAL)
    z_top  = z_base + float(H_ZYLINDER)
    half   = max(float(R_ZYLINDER), float(H_ZYLINDER) / 2)
    mid_z  = (z_base + z_top) / 2

    fig = plt.figure(figsize=(20, 9))
    for col, (centers, layers, lbl, w2val) in enumerate([
        (centers_best,  layers_best,  label_best,  w2_best),
        (centers_worst, layers_worst, label_worst, w2_worst),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        _draw_3d_wireframe(ax, z_base, z_top)
        ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2],
                   c="lightgray", s=3, alpha=0.2, linewidths=0, zorder=1)
        for area in ["pit", "bot", "top", "wall"]:
            mask = layers == area
            if not mask.any():
                continue
            pts = centers[mask]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=_AREA_COLORS[area], marker=_AREA_MARKERS[area],
                       s=40, alpha=0.9, edgecolors="black", linewidths=0.3, zorder=3)
        w2_str = f"W2_global = {w2val:.1f} mm" if w2val is not None else ""
        ax.set_title(f"{lbl}\n{w2_str}", fontsize=12)
        ax.set_xlabel("x (mm)", fontsize=8); ax.set_ylabel("y (mm)", fontsize=8)
        ax.set_zlabel("z (mm)", fontsize=8)
        ax.set_xlim(-half, half); ax.set_ylim(-half, half)
        ax.set_zlim(mid_z - half, mid_z + half)

    fig.suptitle("PMT Spatial Homogeneity: Low vs High W2", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {Path(output_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot B — CDF comparison: z-marginal and phi-marginal
# ─────────────────────────────────────────────────────────────────────────────

def plot_w2_cdf_comparison(
    centers: np.ndarray,
    ref: np.ndarray,
    output_path: str,
    label: str = "",
    w2_z: Optional[float] = None,
    w2_phi: Optional[float] = None,
) -> None:
    """Plot B — Quantile-function comparison: z-marginal and phi-marginal."""
    q = np.linspace(0.0, 1.0, 10001)

    # ── z panel ───────────────────────────────────────────────────────────────
    z_pmt = centers[:, 2]
    z_ref = ref[:, 2]
    q_z_pmt = np.quantile(z_pmt, q)
    q_z_ref = np.quantile(z_ref, q)

    # ── phi panel (circular W2 with optimal rotation) ─────────────────────────
    phi_pmt = (np.arctan2(centers[:, 1], centers[:, 0]) + 2 * np.pi) % (2 * np.pi)
    N_pmt   = len(phi_pmt)
    phi_s   = np.sort(phi_pmt)
    phi_ext = np.concatenate([phi_s, phi_s + 2 * np.pi])
    q_uni_N = 2 * np.pi * (np.arange(N_pmt) + 0.5) / N_pmt
    costs   = np.array([np.mean((phi_ext[k:k + N_pmt] - q_uni_N) ** 2)
                        for k in range(N_pmt)])
    k_star  = int(np.argmin(costs))
    # Rotation angle that puts phi_s[k_star] at position q_uni_N[0]
    theta_opt = (q_uni_N[0] - phi_s[k_star] + 2 * np.pi) % (2 * np.pi)
    phi_rotated = (phi_pmt + theta_opt) % (2 * np.pi)

    q_phi_pmt = np.quantile(phi_rotated, q)   # quantile function of rotated PMT phi
    q_phi_uni = 2 * np.pi * q                  # exact quantile of Uniform([0,2π])

    fig, (ax_z, ax_phi) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Marginal Quantile Functions vs Reference  —  {label}\n"
        "(gap area ∝ W2 cost)",
        fontsize=13,
    )

    # z panel
    w2_z_str = f"  W2_z = {w2_z:.1f} mm" if w2_z is not None else ""
    ax_z.plot(q, q_z_pmt, color="#1f77b4", linewidth=1.8, label="PMT config")
    ax_z.plot(q, q_z_ref, color="#d62728", linewidth=1.8, linestyle="--",
              label="Reference distribution")
    ax_z.fill_between(q, q_z_pmt, q_z_ref, alpha=0.25, color="#1f77b4")
    ax_z.set_xlabel("Quantile level  t", fontsize=12)
    ax_z.set_ylabel("z  (mm)", fontsize=12)
    ax_z.set_title(f"z-coordinate marginal{w2_z_str}", fontsize=12)
    ax_z.legend(fontsize=10)
    ax_z.grid(alpha=0.3)
    ax_z.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v/1000:.1f} m"))

    # phi panel
    w2_phi_str = f"  W2_φ = {w2_phi:.4f} rad" if w2_phi is not None else ""
    ax_phi.plot(q, q_phi_pmt, color="#1f77b4", linewidth=1.8, label="PMT config (optimal rotation)")
    ax_phi.plot(q, q_phi_uni, color="#d62728", linewidth=1.8, linestyle="--",
                label="Uniform reference")
    ax_phi.fill_between(q, q_phi_pmt, q_phi_uni, alpha=0.25, color="#ff7f0e")
    ax_phi.set_xlabel("Quantile level  t", fontsize=12)
    ax_phi.set_ylabel("φ  (rad)", fontsize=12)
    ax_phi.set_title(f"Azimuthal (φ) marginal  [after optimal rotation]{w2_phi_str}",
                     fontsize=12)
    ax_phi.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax_phi.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax_phi.legend(fontsize=10)
    ax_phi.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {Path(output_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot C — Optimal transport arrows (4-panel unrolled detector)
# ─────────────────────────────────────────────────────────────────────────────

def plot_w2_transport_arrows(
    centers: np.ndarray,
    layers: np.ndarray,
    ref: np.ndarray,
    output_path: str,
    label: str = "",
    T: Optional[np.ndarray] = None,
    cost_sq: Optional[np.ndarray] = None,
    max_arrows: int = 25,
) -> None:
    """Plot C — OT transport arrows on 4-panel unrolled detector surface."""
    if T is None or cost_sq is None:
        T, cost_sq = _compute_ot_plan(centers, ref)

    # Per-PMT dominant target and transport distance
    j_star   = np.argmax(T, axis=1)                                # (N,)
    dist_arr = np.sqrt(cost_sq[np.arange(len(centers)), j_star])   # (N,) in mm
    d_norm   = mcolors.Normalize(vmin=0, vmax=np.percentile(dist_arr, 95))
    cmap_arr = mcm.plasma

    ref_areas = _classify_ref_by_area(ref)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    ax_wall, ax_pit = axes[0]
    ax_bot,  ax_top = axes[1]

    panel_data = [
        ("wall", ax_wall, "Wall  (unrolled: φ vs z)"),
        ("pit",  ax_pit,  "Pit  (top view: x vs y)"),
        ("bot",  ax_bot,  "Bot ring  (top view: x vs y)"),
        ("top",  ax_top,  "Top ring  (top view: x vs y)"),
    ]

    def _pmt_proj(area, c):
        if area == "wall":
            return np.arctan2(c[1], c[0]), c[2]
        return c[0], c[1]

    def _ref_proj(area, r):
        if area == "wall":
            return np.arctan2(r[1], r[0]), r[2]
        return r[0], r[1]

    for area, ax, title in panel_data:
        pmt_mask = layers == area
        ref_mask = ref_areas == area

        if not pmt_mask.any():
            ax.set_visible(False)
            continue

        # Reference points in this area
        ref_sub = ref[ref_mask]
        if ref_sub.size:
            rpx = [_ref_proj(area, r)[0] for r in ref_sub]
            rpy = [_ref_proj(area, r)[1] for r in ref_sub]
            ax.scatter(rpx, rpy, c="lightgray", s=5, alpha=0.5,
                       linewidths=0, zorder=1, label="Reference")

        # PMTs and arrows
        pmt_idx = np.where(pmt_mask)[0]
        dist_sub = dist_arr[pmt_idx]
        # Select arrows: top max_arrows by transport distance
        if len(pmt_idx) > max_arrows:
            arrow_idx = pmt_idx[np.argsort(dist_sub)[-max_arrows:]]
        else:
            arrow_idx = pmt_idx

        for i in pmt_idx:
            px, py = _pmt_proj(area, centers[i])
            c = cmap_arr(d_norm(dist_arr[i]))
            ax.scatter([px], [py], c=[c], s=40, zorder=3,
                       marker=_AREA_MARKERS.get(area, "o"),
                       edgecolors="black", linewidths=0.3)
            if i in arrow_idx:
                rx, ry = _ref_proj(area, ref[j_star[i]])
                ax.annotate(
                    "", xy=(rx, ry), xytext=(px, py),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=cmap_arr(d_norm(dist_arr[i])),
                        lw=1.2, mutation_scale=10,
                    ),
                    zorder=2,
                )

        # Area-specific boundary overlays
        if area == "wall":
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(_Z_WMIN, _Z_WMAX)
            ax.set_xlabel("φ (rad)", fontsize=10)
            ax.set_ylabel("z (mm)", fontsize=10)
            ax.axvline(-np.pi, color="gray", lw=0.7, ls="--")
            ax.axvline( np.pi, color="gray", lw=0.7, ls="--")
        else:
            for r, ls in _get_area_circles(area):
                circ = plt.Circle((0, 0), r, fill=False, edgecolor="gray",
                                   linewidth=1.2, linestyle=ls)
                ax.add_patch(circ)
            lim = float(R_ZYLINDER) * 1.1
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xlabel("x (mm)", fontsize=10)
            ax.set_ylabel("y (mm)", fontsize=10)

        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.2)

    sm = mcm.ScalarMappable(norm=d_norm, cmap=cmap_arr)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.55, pad=0.04)
    cbar.set_label("Transport distance  ||x_i − y_j*||  (mm)", fontsize=11)

    fig.suptitle(
        f"Optimal Transport Plan — {label}\n"
        f"(arrows: top-{max_arrows} transports per area by distance; "
        "dots: all PMTs colored by cost)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {Path(output_path).name}")


def _get_area_circles(area: str) -> list[tuple[float, str]]:
    if area == "pit":
        return [(float(R_PIT), "-")]
    if area == "bot":
        return [(float(R_ZYL_BOT), "--"), (float(R_ZYLINDER), "-")]
    if area == "top":
        return [(float(R_ZYL_TOP), "--"), (float(R_ZYLINDER), "-")]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Plot D — Per-PMT cost contribution heatmap (3D)
# ─────────────────────────────────────────────────────────────────────────────

def plot_w2_cost_heatmap(
    centers: np.ndarray,
    layers: np.ndarray,
    ref: np.ndarray,
    output_path: str,
    label: str = "",
    T: Optional[np.ndarray] = None,
    cost_sq: Optional[np.ndarray] = None,
) -> None:
    """Plot D — 3D scatter where each PMT is colored by its OT cost contribution."""
    if T is None or cost_sq is None:
        T, cost_sq = _compute_ot_plan(centers, ref)

    # Per-PMT cost: sqrt(sum_j T[i,j] * cost[i,j]) — units of mm
    per_pmt_cost_mm = np.sqrt(np.sum(T * cost_sq, axis=1))

    z_base = float(Z_BASE_GLOBAL)
    z_top  = z_base + float(H_ZYLINDER)
    half   = max(float(R_ZYLINDER), float(H_ZYLINDER) / 2)
    mid_z  = (z_base + z_top) / 2

    vmax = np.percentile(per_pmt_cost_mm, 95)
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap = mcm.RdYlGn_r  # red = expensive, green = cheap

    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection="3d")
    _draw_3d_wireframe(ax, z_base, z_top)

    for area in ["pit", "bot", "top", "wall"]:
        mask = layers == area
        if not mask.any():
            continue
        pts   = centers[mask]
        costs = per_pmt_cost_mm[mask]
        colors = cmap(norm(costs))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=colors, marker=_AREA_MARKERS[area],
                   s=50, alpha=0.95, edgecolors="black", linewidths=0.3, zorder=3)

    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08,
                 label="Per-PMT OT cost  √(Σ_j T[i,j]·||x_i−y_j||²)  (mm)")

    ax.set_title(
        f"Per-PMT Transport Cost Heatmap — {label}\n"
        "(red = high cost = local underdensity; green = well-covered)",
        fontsize=12,
    )
    ax.set_xlabel("x (mm)", fontsize=9); ax.set_ylabel("y (mm)", fontsize=9)
    ax.set_zlabel("z (mm)", fontsize=9)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_zlim(mid_z - half, mid_z + half)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {Path(output_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot E — Density difference map (2D unrolled per area)
# ─────────────────────────────────────────────────────────────────────────────

def plot_w2_density_difference(
    centers: np.ndarray,
    layers: np.ndarray,
    ref: np.ndarray,
    output_path: str,
    label: str = "",
    n_grid: int = 60,
) -> None:
    """Plot E — 2D KDE density difference (PMT minus reference) per area."""
    from scipy.stats import gaussian_kde

    ref_areas = _classify_ref_by_area(ref)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    ax_wall, ax_pit = axes[0]
    ax_bot,  ax_top = axes[1]

    panel_cfg = [
        ("wall", ax_wall, "Wall  (φ vs z)"),
        ("pit",  ax_pit,  "Pit  (x vs y)"),
        ("bot",  ax_bot,  "Bot ring  (x vs y)"),
        ("top",  ax_top,  "Top ring  (x vs y)"),
    ]

    for area, ax, title in panel_cfg:
        pmt_mask = layers == area
        ref_mask = ref_areas == area

        if not pmt_mask.any() or not ref_mask.any():
            ax.set_visible(False)
            continue

        pmt_pts = centers[pmt_mask]
        ref_pts = ref[ref_mask]

        if area == "wall":
            px = np.arctan2(pmt_pts[:, 1], pmt_pts[:, 0])
            py = pmt_pts[:, 2]
            rx = np.arctan2(ref_pts[:, 1], ref_pts[:, 0])
            ry = ref_pts[:, 2]
            xlo, xhi = -np.pi, np.pi
            ylo, yhi = _Z_WMIN, _Z_WMAX
            xlabel, ylabel = "φ (rad)", "z (mm)"
        else:
            px, py = pmt_pts[:, 0], pmt_pts[:, 1]
            rx, ry = ref_pts[:, 0], ref_pts[:, 1]
            lim = float(R_ZYLINDER) * 1.05
            xlo, xhi, ylo, yhi = -lim, lim, -lim, lim
            xlabel, ylabel = "x (mm)", "y (mm)"

        gx = np.linspace(xlo, xhi, n_grid)
        gy = np.linspace(ylo, yhi, n_grid)
        GX, GY = np.meshgrid(gx, gy)
        grid_pts = np.vstack([GX.ravel(), GY.ravel()])

        try:
            kde_pmt = gaussian_kde(np.vstack([px, py]))
            kde_ref = gaussian_kde(np.vstack([rx, ry]))
            diff = (kde_pmt(grid_pts) - kde_ref(grid_pts)).reshape(n_grid, n_grid)
        except np.linalg.LinAlgError:
            ax.set_visible(False)
            continue

        abs_max = np.percentile(np.abs(diff), 99)
        norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        im = ax.pcolormesh(GX, GY, diff, cmap="RdBu_r", norm=norm,
                           shading="auto", rasterized=True)
        ax.scatter(px, py, c="black", s=15, alpha=0.7, zorder=3,
                   marker=_AREA_MARKERS.get(area, "o"), label="PMTs")

        for r, ls in _get_area_circles(area):
            circ = plt.Circle((0, 0), r, fill=False, edgecolor="gray",
                               linewidth=1.2, linestyle=ls)
            ax.add_patch(circ)

        if area != "wall":
            ax.set_aspect("equal")
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xlabel(xlabel, fontsize=10); ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.2)
        fig.colorbar(im, ax=ax, shrink=0.8, label="KDE(PMT) − KDE(ref)")

    fig.suptitle(
        f"PMT Density − Reference Density  —  {label}\n"
        "(red = overconcentration, blue = underdensity)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {Path(output_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot F — W2 vs N scaling curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_w2_n_scaling(
    output_path: str,
    ref: Optional[np.ndarray] = None,
    n_values: Optional[list[int]] = None,
    n_repeats: int = 3,
    seed: int = 42,
    actual_configs: Optional[list[tuple[str, np.ndarray, float]]] = None,
) -> None:
    """Plot F — W2 vs N scaling curve (Fibonacci-ideal vs random).

    Parameters
    ----------
    actual_configs : list of (label, centers, w2_val) tuples to overlay as markers.
    """
    if ref is None:
        ref = get_w2_ref()
    if n_values is None:
        n_values = [50, 75, 100, 150, 200, 250, 300, 400, 500]

    print("  Computing W2 vs N scaling (this may take ~30–60 s) ...")
    rng = np.random.default_rng(seed)

    w2_fib  = []
    w2_rand_mean = []
    w2_rand_err  = []

    for N in n_values:
        # Fibonacci-ideal
        centers_fib = _fibonacci_centers_for_N(N)
        if len(centers_fib) >= 2:
            w2_f = compute_wasserstein_homogeneity(centers_fib, reference=ref)["w2"]
        else:
            w2_f = np.nan
        w2_fib.append(w2_f)

        # Random (multiple seeds → mean ± std)
        rand_vals = []
        for s in range(n_repeats):
            centers_rnd = sample_reference_distribution(M=N, seed=int(rng.integers(1, 10**6)))
            if len(centers_rnd) >= 2:
                rand_vals.append(
                    compute_wasserstein_homogeneity(centers_rnd, reference=ref)["w2"]
                )
        w2_rand_mean.append(np.mean(rand_vals) if rand_vals else np.nan)
        w2_rand_err.append(np.std(rand_vals)  if rand_vals else np.nan)
        print(f"    N={N:4d}  Fibonacci={w2_f:.1f}  Random={w2_rand_mean[-1]:.1f} ± {w2_rand_err[-1]:.1f}")

    x_arr   = np.array(n_values, dtype=float)
    w2_fib  = np.array(w2_fib)
    w2_rand_mean = np.array(w2_rand_mean)
    w2_rand_err  = np.array(w2_rand_err)

    # N^{-1/2} reference line anchored at Fibonacci N=300 (or closest available)
    anchor_idx = np.argmin(np.abs(x_arr - 300))
    if np.isfinite(w2_fib[anchor_idx]):
        c_ref = w2_fib[anchor_idx] * x_arr[anchor_idx] ** 0.5
        w2_theory = c_ref * x_arr ** (-0.5)
    else:
        w2_theory = None

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_arr, w2_fib, "o-", color="#2ca02c", linewidth=2, markersize=6,
            label="Fibonacci (ideal uniform)")
    ax.plot(x_arr, w2_rand_mean, "s-", color="#d62728", linewidth=2, markersize=6,
            label=f"Random (mean ± 1σ, {n_repeats} draws)")
    if w2_rand_err.any():
        ax.fill_between(x_arr,
                        w2_rand_mean - w2_rand_err,
                        w2_rand_mean + w2_rand_err,
                        color="#d62728", alpha=0.15)

    if w2_theory is not None:
        ax.plot(x_arr, w2_theory, "k--", linewidth=1.2, alpha=0.6,
                label=r"$W_2 \propto N^{-1/2}$  (2-D surface scaling)")

    # Overlay actual config markers
    if actual_configs:
        for lbl, _, w2_val in actual_configs:
            if w2_val is not None:
                ax.axhline(w2_val, color="gray", linewidth=0.8, linestyle=":")
                ax.text(x_arr[-1] * 1.01, w2_val, lbl, fontsize=8,
                        va="center", color="gray")

    ax.set_xlabel("Number of PMTs  N", fontsize=13)
    ax.set_ylabel("W2 distance  (mm)", fontsize=13)
    ax.set_title(
        "W2 Homogeneity vs Number of PMTs\n"
        "(Fibonacci-ideal vs random; reference: 3000-point uniform surface sample)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {Path(output_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_w2_explanation(
    json_paths: list[str],
    output_dir: str,
    labels: Optional[list[str]] = None,
) -> None:
    """Generate all W2 explanation plots (A–F) from a list of voxel JSON files.

    Automatically selects the config with lowest W2 as the primary example,
    and highest W2 as the comparison config (if ≥2 configs are provided).
    All plots are written to *output_dir* with prefix ``w2_``.

    Called from compare_coverages.py main() with one line:
        plot_all_w2_explanation(args.configs, args.output_dir, labels=args.labels)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if not json_paths:
        print("  [SKIP] plot_all_w2_explanation: no JSON paths provided.")
        return

    print("\n── W2 explanation visualizations ──────────────────────────────────")

    # ── Load all configs ──────────────────────────────────────────────────────
    loaded: list[tuple[np.ndarray, np.ndarray, str, float]] = []
    for i, jp in enumerate(json_paths):
        try:
            centers, lyrs, stem = _load_pmt_json(jp)
            if len(centers) < 2:
                continue
            lbl = (labels[i] if labels and i < len(labels) else stem)
            w2v = compute_wasserstein_homogeneity(centers, reference=get_w2_ref())["w2"]
            loaded.append((centers, lyrs, lbl, w2v))
        except Exception as exc:
            print(f"  [WARN] Could not load {jp}: {exc}")

    if not loaded:
        print("  [SKIP] No valid configs loaded.")
        return

    loaded.sort(key=lambda x: x[3])  # sort by W2 ascending
    ref = get_w2_ref()

    best_centers, best_layers, best_label, best_w2 = loaded[0]

    # Compute decomposed metrics for the best config
    try:
        w2_z   = _compute_w2_z(best_centers)   # uniform z reference
        w2_phi = _compute_w2_phi(best_centers)
    except Exception:
        w2_z = w2_phi = None

    # ── Compute OT plan once (shared by plots C and D) ────────────────────────
    print(f"  Computing OT plan for '{best_label}' (N={len(best_centers)}, M={len(ref)}) ...")
    T, cost_sq = _compute_ot_plan(best_centers, ref)

    def _p(name: str) -> str:
        return os.path.join(output_dir, name)

    # Plot A — 3D scatter (best config)
    plot_w2_3d_scatter(
        best_centers, best_layers, ref,
        _p(f"w2_A_3d_scatter_{best_label}.png"),
        label=best_label, w2_global=best_w2,
    )

    # Plot A — side-by-side comparison (if ≥2 configs)
    if len(loaded) >= 2:
        worst_centers, worst_layers, worst_label, worst_w2 = loaded[-1]
        plot_w2_3d_comparison(
            best_centers, best_layers,
            worst_centers, worst_layers,
            ref,
            _p("w2_A_comparison.png"),
            label_best=best_label, label_worst=worst_label,
            w2_best=best_w2, w2_worst=worst_w2,
        )

    # Plot B — CDF comparison
    plot_w2_cdf_comparison(
        best_centers, ref,
        _p(f"w2_B_cdf_{best_label}.png"),
        label=best_label, w2_z=w2_z, w2_phi=w2_phi,
    )

    # Plot C — transport arrows (reuse T)
    plot_w2_transport_arrows(
        best_centers, best_layers, ref,
        _p(f"w2_C_transport_{best_label}.png"),
        label=best_label, T=T, cost_sq=cost_sq,
    )

    # Plot D — cost heatmap (reuse T)
    plot_w2_cost_heatmap(
        best_centers, best_layers, ref,
        _p(f"w2_D_cost_heatmap_{best_label}.png"),
        label=best_label, T=T, cost_sq=cost_sq,
    )

    # Plot E — density difference
    plot_w2_density_difference(
        best_centers, best_layers, ref,
        _p(f"w2_E_density_diff_{best_label}.png"),
        label=best_label,
    )

    # Plot F — W2 vs N scaling (synthetic, no JSON needed)
    actual = [(lbl, c, w2v) for c, _, lbl, w2v in loaded]
    plot_w2_n_scaling(
        _p("w2_F_n_scaling.png"),
        ref=ref,
        actual_configs=actual,
    )

    print("── W2 explanation visualizations done ─────────────────────────────\n")
