"""matplotlib display of tvbounds objects (port of ``R/plot.R``).

The style follows the paper's figure scripts (deep blue #1F4E79, plain
``theme_bw`` with no grid) and reproduces the ggplot2 layers of the R
package one by one: the dashed gray reference line at ``tau_star``, the
light outer confidence ribbons delimited by dashed lines, the shaded
identified region, the two bound paths, the dotted breakdown line with its
annotation, and the baseline point at ``delta = 0``.
"""
from __future__ import annotations

import math

import numpy as np

from ._rcolors import r_color
from ._object import is_tvbounds, is_na, or_null, warn, _r_signif, _format_r
from .summary import _tvb_signed_path, _tvb_cross0, _num_scalar

_GRAY50 = "#7F7F7F"   # R's "gray50"
_LW = 1.0             # ggplot2 linewidth -> matplotlib points


def _col(b, name) -> np.ndarray:
    return b[name].to_numpy(dtype=np.float64)


def tvbounds_plot(x, bands=True, baseline=True, breakdown=True, tau_star=0,
                  color="#1F4E79", xlab=None, ylab=None, title=None,
                  log_x=False, **kwargs):
    """See the tvbounds manual."""
    if not is_tvbounds(x):
        raise ValueError("`x` must be a `tvbounds` object.")
    for nm, v in (("bands", bands), ("baseline", baseline),
                  ("breakdown", breakdown), ("log_x", log_x)):
        if not isinstance(v, (bool, np.bool_)):
            raise ValueError(f"`{nm}` must be True or False.")
    bands, baseline, breakdown, log_x = (bool(bands), bool(baseline),
                                         bool(breakdown), bool(log_x))
    tau_v = _num_scalar(tau_star)
    if tau_v is None or not math.isfinite(tau_v):
        raise ValueError("`tau_star` must be a finite numeric scalar.")
    tau_star = tau_v
    if not isinstance(color, str):
        raise ValueError("`color` must be a single colour string.")
    color = r_color(color)                 # R colour names (e.g. "gray50") -> hex

    import matplotlib.pyplot as plt

    b = x.bounds
    b = b[~np.isnan(_col(b, "delta"))]
    if log_x:
        npos = _col(b, "delta") <= 0
        if bool(npos.any()):
            warn("`log_x = True`: dropped %d row(s) with `delta` <= 0, which cannot "
                 "be placed on a logarithmic axis." % int(npos.sum()))
            b = b[~npos]
    if len(b) < 2:
        raise ValueError("`x$bounds` must contain at least two budget values to plot.")
    cols = list(b.columns)
    has_cil = "ci_lower" in cols and bool(np.any(~np.isnan(_col(b, "ci_lower"))))
    has_ciu = "ci_upper" in cols and bool(np.any(~np.isnan(_col(b, "ci_upper"))))

    fig, ax = plt.subplots()
    ax.axhline(tau_star, linestyle="--", linewidth=0.7 * _LW, color=_GRAY50)

    # Following the paper's figures, only the OUTER confidence limit of each
    # bound is drawn -- the lower limit for the lower bound and the upper
    # limit for the upper bound -- so the dashed pair traces the outward
    # envelope of the identified set; the band between each estimate and its
    # outer limit is shaded more lightly than the identified region.
    if bands and has_cil:
        cl = b[~np.isnan(_col(b, "ci_lower")) & ~np.isnan(_col(b, "lower"))]
        if len(cl) > 1:
            ax.fill_between(_col(cl, "delta"), _col(cl, "ci_lower"), _col(cl, "lower"),
                            facecolor=color, alpha=0.12, edgecolor="none", linewidth=0)
            ax.plot(_col(cl, "delta"), _col(cl, "ci_lower"), color=color,
                    linestyle="--", linewidth=0.8 * _LW)
    if bands and has_ciu:
        cu = b[~np.isnan(_col(b, "ci_upper")) & ~np.isnan(_col(b, "upper"))]
        if len(cu) > 1:
            ax.fill_between(_col(cu, "delta"), _col(cu, "upper"), _col(cu, "ci_upper"),
                            facecolor=color, alpha=0.12, edgecolor="none", linewidth=0)
            ax.plot(_col(cu, "delta"), _col(cu, "ci_upper"), color=color,
                    linestyle="--", linewidth=0.8 * _LW)

    core = b[~np.isnan(_col(b, "lower")) & ~np.isnan(_col(b, "upper"))]
    if len(core) > 1:
        ax.fill_between(_col(core, "delta"), _col(core, "lower"), _col(core, "upper"),
                        facecolor=color, alpha=0.25, edgecolor="none", linewidth=0)
    lo = b[~np.isnan(_col(b, "lower"))]
    if len(lo) > 1:
        ax.plot(_col(lo, "delta"), _col(lo, "lower"), color=color, linewidth=1.2 * _LW)
    up = b[~np.isnan(_col(b, "upper"))]
    if len(up) > 1:
        ax.plot(_col(up, "delta"), _col(up, "upper"), color=color, linewidth=1.2 * _LW)

    # Plug-in breakdown budget, drawn only when interior to the grid. The
    # direction follows the summary method: the lower path when the baseline
    # point estimate exceeds tau_star, the upper path otherwise.
    point = x.point
    has_point = point is not None and _num_scalar(point) is not None \
        and not is_na(_num_scalar(point))
    if breakdown and has_point:
        pt = _num_scalar(point)
        dirn = "lower" if pt > tau_star else "upper"
        sp = _tvb_signed_path(x, dirn, tau_star)
        if sp["d"].size >= 2:
            db = _tvb_cross0(sp["d"], sp["path"])
            if not math.isnan(db) and db > sp["d"][0] and db < sp["right"]:
                hj = 1.15 if db > (sp["d"][0] + sp["right"]) / 2 else -0.15
                ax.axvline(db, linestyle=":", linewidth=0.9 * _LW, color=color)
                label = r"$\hat{\delta}_b = " + _format_r(_r_signif(db, 3), 15) + "$"
                ax.annotate(label, xy=(db, 1.0),
                            xycoords=("data", "axes fraction"),
                            xytext=(-4 if hj > 0 else 4, -9),
                            textcoords="offset points",
                            ha="right" if hj > 0 else "left", va="top",
                            color=color, fontsize=10)

    if baseline and not log_x and has_point:
        ax.plot([0.0], [_num_scalar(point)], marker="o", markersize=6,
                color=color, linestyle="none")

    if log_x:
        ax.set_xscale("log")
    if xlab is None:
        xlab = r"$\delta$"
    if ylab is None:
        ylab = or_null(x.estimand_label, "estimand")

    # theme_bw(base_size = 12) with no grid and no legend
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    if title is not None:
        ax.set_title(title, fontweight="bold", loc="center", fontsize=14.4)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)
    ax.grid(False)
    ax.tick_params(labelsize=9.6)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    return fig
