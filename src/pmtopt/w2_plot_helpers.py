"""
Shared W2-correlation plotting helper.

Used by both:
  src/pmtopt/evaluate_coverages.py  (SSD / pmtopt pipeline)
  evaluation/ratio_derivation/compare_coverages.py  (LGDO pipeline)

Keeping this in one place guarantees identical visual style and statistical
definitions across both analysis pipelines so their outputs can be compared
directly.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as scipy_stats


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
    """Shared scatter + OLS regression + residual panel for correlation plots.

    Parameters
    ----------
    ax_scatter : top axes — receives scatter points, regression line, CI, stats box.
    ax_resid   : bottom axes (shared x with ax_scatter) — receives residual stems.
    w2_arr     : 1-D array of x-axis values (typically W2), one per config.
    y_arr      : 1-D array of the metric being plotted, same length as w2_arr.
    color_pts  : list of matplotlib colours, one per config.
    labels     : config label strings for point annotations.
    y_label    : y-axis label text placed on ax_scatter.
    x_label    : x-axis label text (default: "Global W2 (mm)").
    """
    n = len(w2_arr)

    # ── scatter points with config labels ────────────────────────────
    for w2v, yv, c, lbl in zip(w2_arr, y_arr, color_pts, labels):
        ax_scatter.scatter([w2v], [yv], color=c, s=55, zorder=3)
        ax_scatter.annotate(
            lbl, xy=(w2v, yv), xytext=(4, 3),
            textcoords="offset points", fontsize=6, color=c,
        )

    # ── correlation statistics ────────────────────────────────────────
    # Guard against constant arrays: pearsonr raises ValueError in scipy≥1.9
    # when either input has zero variance; spearmanr returns NaN with a warning.
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

    # ── OLS regression line + 95 % CI ────────────────────────────────
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
            1 / n + (x_fit - x_mean) ** 2
            / np.sum((w2_arr - x_mean) ** 2)
        )
        ax_scatter.fill_between(x_fit, y_fit - ci, y_fit + ci,
                                color="black", alpha=0.08)

        # ── residual panel ────────────────────────────────────────────
        y_pred_pts = slope * w2_arr + intercept
        resids     = y_arr - y_pred_pts
        for w2v, rv, c in zip(w2_arr, resids, color_pts):
            ax_resid.scatter([w2v], [rv], color=c, s=40, zorder=3)
            ax_resid.plot([w2v, w2v], [0, rv], color=c,
                          linewidth=0.8, alpha=0.6)
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
