"""Summary measures for tvbounds objects (port of ``R/summary.R``).

Summary measures for tvbounds objects: breakdown budgets, shadow prices,
robustness standard errors, and the certification frontier.  The
computational helpers below port, essentially verbatim, the audited
helpers of the paper's summary-measure script
(SA9_summary_measures_run.R): cross0(), eta_fun(), lin(), nstar(), and the
body of summarize(), with the paper's defaults (two-sided critical value
qnorm(0.975) at level 0.95, cost_per_unit = 50, jump = 0.05, tau_star = 0,
integer frontier floor(n_j) + 1, and censored breakdown budgets reported at
the right endpoint of the budget grid).

Every floating-point operation follows the R code in the same order, and
the R-compatible primitives of ``_rcompat`` (``rapprox`` for
``stats::approx(rule = 2)``, ``rqnorm`` for ``qnorm``) are used so that the
measures agree with the R package bit for bit.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from ._object import (TVBoundsSummary, is_tvbounds, is_na, or_null,
                      format_call, warn)
from ._rcompat import rapprox, rqnorm

_DIRECTIONS = ("auto", "lower", "upper")


# ---------------------------------------------------------------------------
# small R-semantics helpers
# ---------------------------------------------------------------------------
def _num_scalar(x) -> Optional[float]:
    """The value of ``x`` as a float when R would see a length-one numeric.

    Returns ``None`` when ``x`` is not a numeric scalar (R's
    ``!is.numeric(x) || length(x) != 1L``): logicals are not numeric in R,
    so Python/numpy booleans are rejected, and only 0-d or size-one numeric
    arrays are accepted as length-one vectors.
    """
    if isinstance(x, (bool, np.bool_)):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, np.ndarray):
        if x.size != 1 or not np.issubdtype(x.dtype, np.number) \
                or np.issubdtype(x.dtype, np.bool_):
            return None
        return float(x.reshape(-1)[0])
    return None


def _match_arg(arg, choices, what: str = "arg") -> str:
    """R's ``match.arg(arg, choices)`` (exact or unique partial match)."""
    if arg is None:
        return choices[0]
    if not isinstance(arg, str):
        raise ValueError(f"'{what}' must be None or a character vector")
    if arg in choices:
        return arg
    partial = [c for c in choices if c.startswith(arg)] if arg else []
    if len(partial) == 1:
        return partial[0]
    quoted = ", ".join("“" + c + "”" for c in choices)
    raise ValueError(f"'{what}' should be one of {quoted}")


def _rdiv(a, b) -> float:
    """IEEE division with R semantics (``x / 0`` gives ``Inf``/``NaN``)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(a) / np.float64(b))


def _rsqrt(x: float) -> float:
    """R's ``sqrt()`` of a scalar (``NaN`` for negative arguments)."""
    if math.isnan(x) or x < 0:
        return math.nan
    return math.sqrt(x)


class _Sym(str):
    """A bare symbol inside :func:`format_call` (printed without quotes)."""

    def __repr__(self):  # pragma: no cover - cosmetic
        return str(self)


# ---------------------------------------------------------------------------
# ported kernels
# ---------------------------------------------------------------------------
def _tvb_cross0(d, f) -> float:
    """First crossing of a signed bound path with zero.

    First budget delta at which ``f`` reaches zero from above, linearly
    interpolated between grid points.  Returns the smallest grid value when
    ``f`` is already nonpositive there, and ``NaN`` when ``f`` stays
    strictly positive on the whole grid, that is, when the breakdown budget
    is censored at the right endpoint of the budget grid.  Rows where ``d``
    or ``f`` is ``NaN`` are ignored.
    """
    d = np.asarray(d, dtype=np.float64).ravel()
    f = np.asarray(f, dtype=np.float64).ravel()
    ok = ~np.isnan(d) & ~np.isnan(f)
    d = d[ok]
    f = f[ok]
    if d.size == 0:
        return math.nan
    neg = np.where(f <= 0)[0]
    if neg.size == 0:
        return math.nan
    i = int(neg[0])
    if i == 0:
        return float(d[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(d[i - 1] + (d[i] - d[i - 1]) * f[i - 1] / (f[i - 1] - f[i]))


def _tvb_eta_fun(d, tl) -> np.ndarray:
    """Shadow price of robustness along a bound path.

    The shadow price eta(delta) = -tau_lower'(delta), computed by central
    differences on the interior of the grid and one-sided differences at
    the endpoints.  The grid need not be uniform.  With only two grid
    points the single slope is repeated; with fewer, ``NaN`` s are returned.
    """
    d = np.asarray(d, dtype=np.float64).ravel()
    tl = np.asarray(tl, dtype=np.float64).ravel()
    n = d.size
    if n < 2:
        return np.full(n, math.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        if n == 2:
            s = -(tl[1] - tl[0]) / (d[1] - d[0])
            return np.array([s, s], dtype=np.float64)
        i = np.arange(1, n - 1)
        return np.concatenate((
            [-(tl[1] - tl[0]) / (d[1] - d[0])],
            -(tl[i + 1] - tl[i - 1]) / (d[i + 1] - d[i - 1]),
            [-(tl[n - 1] - tl[n - 2]) / (d[n - 1] - d[n - 2])],
        )).astype(np.float64)


def _tvb_lin(d, y, at) -> float:
    """Linear interpolation with flat extrapolation.

    Interpolates ``y`` on the grid ``d`` at the point ``at``, extrapolating
    flat beyond the grid (``rule = 2``), as in the paper's summary-measure
    script.  Returns ``NaN`` when ``at`` is ``NaN`` or fewer than two
    complete pairs exist.
    """
    if is_na(at):
        return math.nan
    d = np.asarray(d, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ok = ~np.isnan(d) & ~np.isnan(y)
    if int(ok.sum()) < 2:
        return math.nan
    return float(rapprox(d[ok], y[ok], float(at)))


def _tvb_nstar(d, tl, se, n, at, zc) -> float:
    """Certification frontier.

    The frontier n*(delta; alpha) = z^2_{1-alpha/2} sigma^2(delta) /
    (tau_lower(delta) - tau_star)^2, evaluated with the estimated standard
    deviation sigma_hat_n(delta) = sqrt(n) se(delta) and the signed path in
    place of tau_lower(delta) - tau_star: the sample size at which the
    normal-approximation confidence limit of the bound path would just
    touch the reference value at budget ``at``.  ``NaN`` once the path has
    crossed the reference value: past the breakdown budget no sample size
    certifies the conclusion.
    """
    t0 = _tvb_lin(d, tl, at)
    s0 = _tvb_lin(d, se, at)
    if is_na(t0) or t0 <= 0 or is_na(s0):
        return math.nan
    # R evaluates x^2 as x * x
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return float(np.float64(zc * zc) * np.float64(s0 * s0 * n) / np.float64(t0 * t0))


def _tvb_resolve_direction(direction: str, point, tau_star: float) -> str:
    """Resolve the direction of the robustness exercise.

    ``"auto"`` selects the lower bound path tau_lower(delta) when the
    baseline point estimate exceeds ``tau_star`` and the upper bound path
    tau_upper(delta) otherwise, mirroring the paper's symmetric treatment
    of the two directions through the integrand -g.
    """
    if direction != "auto":
        return direction
    if point is None or is_na(point):
        raise ValueError("`direction = \"auto\"` requires a non-missing baseline point "
                         "estimate in the object; set `direction` explicitly.")
    return "lower" if float(point) > tau_star else "upper"


def _tvb_signed_path(obj, direction: str, tau_star: float) -> dict:
    """Extract the signed bound path adjacent to the reference value.

    For ``direction = "lower"`` the signed path is tau_lower(delta) -
    tau_star; for ``direction = "upper"`` it is tau_star - tau_upper(delta),
    so that in both cases the path starts positive when the conclusion
    holds at the baseline and the breakdown budget is its first crossing
    with zero.  Standard errors are invariant to this sign-and-shift
    transformation; the outer confidence limit maps to ``ci_lower -
    tau_star`` and ``tau_star - ci_upper`` respectively.  Rows with a
    missing budget or a missing path value are dropped; ``se`` and ``ci``
    are ``None`` when the object does not carry the corresponding columns.
    """
    b = obj.bounds
    cols = list(b.columns)

    def col(name):
        return b[name].to_numpy(dtype=np.float64)

    delta = col("delta")
    if direction == "lower":
        path = col("lower") - tau_star
        se = col("lower_se") if "lower_se" in cols else None
        ci = col("ci_lower") - tau_star if "ci_lower" in cols else None
    else:
        path = tau_star - col("upper")
        se = col("upper_se") if "upper_se" in cols else None
        ci = tau_star - col("ci_upper") if "ci_upper" in cols else None
    ok = ~np.isnan(delta) & ~np.isnan(path)
    return {
        "d": delta[ok],
        "path": path[ok],
        "se": se[ok] if se is not None else None,
        "ci": ci[ok] if ci is not None else None,
        "right": float(np.max(delta[ok])) if bool(ok.any()) else math.nan,
        "direction": direction,
    }


# ---------------------------------------------------------------------------
# the user-facing function
# ---------------------------------------------------------------------------
def tvbounds_summary(object, delta=None, tau_star=0, level=0.95,
                     cost_per_unit=50, jump=0.05, direction="auto"):
    """See the tvbounds manual."""
    if not is_tvbounds(object):
        raise ValueError("`object` must be a `tvbounds` object.")
    direction = _match_arg(direction, _DIRECTIONS)
    if delta is not None:
        delta_v = _num_scalar(delta)
        if delta_v is None or not math.isfinite(delta_v) or delta_v < 0:
            raise ValueError("`delta` must be None or a single nonnegative finite number.")
        delta = delta_v
    tau_star_v = _num_scalar(tau_star)
    if tau_star_v is None or not math.isfinite(tau_star_v):
        raise ValueError("`tau_star` must be a finite numeric scalar.")
    tau_star = tau_star_v
    level_v = _num_scalar(level)
    if level_v is None or math.isnan(level_v) or level_v <= 0 or level_v >= 1:
        raise ValueError("`level` must be a number strictly between 0 and 1.")
    level = level_v
    cpu_v = _num_scalar(cost_per_unit)
    if cpu_v is None or not math.isfinite(cpu_v) or cpu_v < 0:
        raise ValueError("`cost_per_unit` must be a single nonnegative number.")
    cost_per_unit = cpu_v
    jump_v = _num_scalar(jump)
    if jump_v is None or not math.isfinite(jump_v) or jump_v <= 0:
        raise ValueError("`jump` must be a single positive number.")
    jump = jump_v

    notes = []
    dirn = _tvb_resolve_direction(direction, object.point, tau_star)
    sp = _tvb_signed_path(object, dirn, tau_star)
    if sp["d"].size < 2:
        raise ValueError("`object$bounds` must contain at least two non-missing budget "
                         f"values on the {dirn} bound path.")

    # Two-sided critical value; the empirical applications report the
    # certified budget off the outer band, so z_{1-alpha/2} applies.
    zc = rqnorm(1 - (1 - level) / 2)
    obj_level = object.level
    if not is_na(obj_level) and abs(float(obj_level) - level) > 1e-12:
        notes.append(
            f"`level` ({level:.3f}) differs from the level of the band stored in "
            f"the object ({float(obj_level):.3f}); the certified breakdown is read off the "
            "stored band.")

    has_se = sp["se"] is not None and bool(np.any(~np.isnan(sp["se"])))
    has_ci = sp["ci"] is not None and bool(np.any(~np.isnan(sp["ci"])))

    eta_path = _tvb_eta_fun(sp["d"], sp["path"])
    db_raw = _tvb_cross0(sp["d"], sp["path"])
    censored = math.isnan(db_raw)
    # A censored breakdown budget is reported at the right endpoint of the
    # budget grid (1 for the total-variation and contamination
    # neighborhoods), the value the empty-set convention assigns it, rather
    # than as "> 1": no larger budget is on the grid.
    db = sp["right"] if censored else db_raw

    # The certified budget is read off the SAME band that the figures plot,
    # so that tables, figures, and text agree on one number. The normal
    # floor is retained only as a diagnostic: the two estimate the same
    # population object and differ when the bootstrap distribution of the
    # bound is asymmetric.
    db_ci_raw = _tvb_cross0(sp["d"], sp["ci"]) if has_ci else math.nan
    censored_ci = has_ci and math.isnan(db_ci_raw)
    db_ci = math.nan if not has_ci else (sp["right"] if censored_ci else db_ci_raw)
    if has_se:
        db_ci_norm = _tvb_cross0(sp["d"], sp["path"] - zc * sp["se"])
    else:
        db_ci_norm = math.nan

    # Evaluation budget. Default: the plug-in breakdown; when that is
    # censored, the shadow price and the robustness standard error are
    # anchored at the certified budget instead, as in the paper. The
    # frontier is always priced at the certified budget unless the user
    # supplies an explicit `delta`.
    if delta is not None:
        anchor = delta
        frontier_at = delta
        if delta > sp["right"]:
            warn("`delta` exceeds the largest budget on the grid; the "
                 "measures use flat extrapolation beyond the grid.")
    else:
        anchor = db_raw if not censored else db_ci_raw
        frontier_at = db_ci_raw
        if censored and has_ci and not censored_ci:
            notes.append(
                "the plug-in breakdown is censored at the right endpoint of the "
                "budget grid; the shadow price and the robustness standard error "
                "are anchored "
                "at the certified breakdown instead.")
        if math.isnan(anchor):
            notes.append(
                "no interior breakdown budget is available to anchor the shadow "
                "price; pass an explicit `delta` to evaluate the measures at a "
                "chosen budget.")

    # nn <- suppressWarnings(as.numeric(object$n)); NA unless length one
    nn = _num_scalar(object.n)
    if nn is None:
        try:
            nn = float(object.n) if object.n is not None and not isinstance(
                object.n, (bool, np.bool_)) else math.nan
        except (TypeError, ValueError):
            nn = math.nan

    eta_b = _tvb_lin(sp["d"], eta_path, anchor)
    se_b = _tvb_lin(sp["d"], sp["se"], anchor) if has_se else math.nan
    varsig = _rdiv(se_b * _rsqrt(nn), eta_b)
    varsig_sc = _rdiv(se_b, eta_b)

    # Certification frontier. n_cur is the frontier at the pricing budget;
    # n_star the smallest integer sample size strictly above the frontier at
    # the pricing budget raised by `jump` (certification requires an integer
    # n strictly above the real-valued frontier: floor(n) + 1). delta_n is
    # the additional sample the study would actually have to collect: the
    # frontier at the raised budget against the realized n.
    if has_se and not math.isnan(nn):
        n_cur = _tvb_nstar(sp["d"], sp["path"], sp["se"], nn, frontier_at, zc)
    else:
        n_cur = math.nan
    if has_se and not math.isnan(nn) and not math.isnan(frontier_at):
        n_jmp = _tvb_nstar(sp["d"], sp["path"], sp["se"], nn, frontier_at + jump, zc)
    else:
        n_jmp = math.nan
    n_star = math.nan if math.isnan(n_jmp) else float(math.floor(n_jmp) + 1) \
        if math.isfinite(n_jmp) else n_jmp
    delta_n = n_star - nn
    cost_pp = _rdiv(delta_n * cost_per_unit, 100 * jump)

    if not has_ci:
        by_design = (" (this application carries no inference by design)"
                     if object.application in ("riv", "counterfactual") else "")
        notes.append(
            f"no confidence band is attached to the {dirn} bound path, so the "
            "certified breakdown and the certification frontier are "
            f"reported as NA{by_design}.")
    if not has_se:
        notes.append(
            f"no bootstrap standard errors are attached to the {dirn} bound "
            "path, so the normal-floor diagnostic and the robustness "
            "standard error "
            "are reported as NA.")

    point = object.point
    point_v = math.nan if point is None else float(point)
    measures = pd.DataFrame({
        "label": [or_null(object.estimand_label, "estimand")],
        "direction": [dirn],
        "n": [nn],
        "point": [point_v],
        "tau_star": [tau_star],
        "delta_b": [db],
        "censored": [bool(censored)],
        "delta_b_ci": [db_ci],
        "censored_ci": [bool(censored_ci)],
        "delta_b_ci_norm": [db_ci_norm],
        "delta_eval": [float(anchor)],
        "eta": [eta_b],
        "se": [se_b],
        "varsigma": [varsig],
        "varsigma_sc": [varsig_sc],
        "frontier_at": [float(frontier_at)],
        "n_cur": [n_cur],
        "n_star": [n_star],
        "delta_n": [delta_n],
        "cost_per_pp": [cost_pp],
    })

    call = format_call("tvbounds_summary", {
        "object": _Sym("<tvbounds>"), "delta": delta, "tau_star": tau_star,
        "level": level, "cost_per_unit": cost_per_unit, "jump": jump,
        "direction": direction})

    return TVBoundsSummary(
        measures=measures,
        application=object.application,
        estimand_label=object.estimand_label,
        neighborhood=object.neighborhood,
        divergence=object.divergence,
        direction=dirn,
        tau_star=tau_star,
        delta_eval=float(anchor),
        level=level,
        band_level=object.level,
        zc=zc,
        cost_per_unit=cost_per_unit,
        jump=jump,
        has_se=has_se,
        has_ci=has_ci,
        notes=notes,
        call=call,
    )
