"""Computational kernels for :func:`tvbounds_riv` (port of ``R/riv-kernels.R``).

Ported from the formula-instrument sensitivity exercise of Section SA7.3 of
Palomba (2026) (Propositions "formula instruments bounds" and "formula
instruments contamination").  Nothing here reads or writes a file.  All
functions are internal.

Notation (manuscript Section SA7.3).  The exercise is conducted CONDITIONALLY
on the realized sample, so everything below is a deterministic function of
the data and of the budget delta::

    i = 1..n    units, y_i and x_i already residualized on the controls
    v           realized shock vector; v^(1),...,v^(S) the counterfactual draws
    f_i(v')     the formula, i.e. unit i's instrument at shock configuration v'
    z_i         = f_i(v), the un-recentered instrument
    mu_i(P)     = E_P[f_i], the expected instrument under assignment law P
    beta(P)     = sum_i (z_i - mu_i(P)) y_i / sum_i (z_i - mu_i(P)) x_i
                = G_y(P) / G_x(P)
    g_y(v')     = sum_i y_i f_i(v'),      g_x(v') = sum_i x_i f_i(v')
    G_y(P)      = g_y(v) - E_P[g_y],      G_x(P)  = g_x(v) - E_P[g_x]
    g_b         = g_y - b * g_x

Every quantity the exercise needs is therefore a function of the S+1 numbers
(g_y(v), g_y(v^(1)),...,g_y(v^(S))) and their g_x counterparts: the problem is
two-dimensional however large the shock space is.  The design matrix is
consumed once, in :func:`fi_criteria`, and never again.

Two conventions used throughout, both harmless and both stated in SA7.3:

(a) SHIFT INVARIANCE.  Replacing f_i(.) by f_i(.) - c_i and z_i by z_i - c_i
    for any unit-specific constant c_i leaves G_y(P) and G_x(P) unchanged,
    because E_P[.] is an average over a probability distribution.  We exploit
    this to center the design matrix, which makes E_Pbase[g_y] = E_Pbase[g_x]
    = 0 exactly rather than up to rounding.

(b) SIGN NORMALIZATION.  Replacing (g_y, g_x) by (-g_y, -g_x) leaves beta(P)
    unchanged and flips the sign of the recentered first stage, so we may and
    do take G_x(Pbase) > 0, as the propositions assume.

Numerical fidelity: every base-R primitive of the R kernels (``sum``,
``cumsum``, ``order``, ``which.max``, ``uniroot``) is replaced by its
R-compatible counterpart from :mod:`tvbounds._rcompat`, in the same order
of floating-point operations, so that the Python and R results agree to
machine precision.  The two BLAS products of :func:`fi_criteria` are issued
on column-major (Fortran-ordered) operands, which makes numpy call the same
``dgemv`` routines R calls for ``Fmat %*% p`` and ``crossprod(Fc, y)``; on a
machine where R and numpy link the same BLAS they agree bitwise as well.
The one step of the application without a bitwise-reproducible counterpart
is the residualization on the controls performed upstream in
:mod:`tvbounds.riv` (LINPACK Householder in R, LAPACK Householder in numpy),
whose relative differences of order 1e-16 propagate to every quantity below.

Index conventions: the ``arg_lower``/``arg_upper`` entries returned by
:func:`fi_bounds_cont` are 0-based Python indices into the S draws (the R
kernels return the corresponding 1-based ``which.min``/``which.max``).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._object import BudgetValue
from ._rcompat import rcumsum, rorder, rsum, runiroot, rwhich_max, rwhich_min

_DBL_EPSILON = float(np.finfo(np.float64).eps)


def _as_f64(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


def _rsign(v) -> float:
    """R's ``sign()`` of a scalar (``NaN`` propagates, as in R)."""
    v = float(v)
    if math.isnan(v):
        return math.nan
    return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)


# R's three-valued logic on scalars, used where the R kernels compare bounds
# that may be NaN (a vanishing first stage at the baseline): a comparison
# with NaN is NA, `&&` short-circuits on FALSE and propagates NA otherwise,
# and `if (NA)` is an error.  NA is represented by ``None``.
def _r_le(a, b):
    """R's ``a <= b`` on two scalars: ``None`` (NA) when either is NaN."""
    a = float(a)
    b = float(b)
    if math.isnan(a) or math.isnan(b):
        return None
    return a <= b


def _r_and(a, b):
    """R's ``a && b`` with ``b`` a thunk evaluated only when needed."""
    if a is False:
        return False
    b = b()
    if a is True:
        return b
    return False if b is False else None          # NA && FALSE is FALSE


def _r_if(cond):
    """R's ``if (cond)``: the condition, or R's error when it is NA."""
    if cond is None:
        raise RuntimeError("missing value where TRUE/FALSE needed")
    return cond


# -----------------------------------------------------------------------------
# 1.  From the design to the two criterion functions
# -----------------------------------------------------------------------------
def fi_criteria(y, x, z, Fmat, p=None) -> dict:
    """Reduce a formula-instrument design to the criterion functions of SA7.3.

    Parameters
    ----------
    y : n-vector, outcome, already residualized on the controls
    x : n-vector, endogenous regressor, already residualized on the controls
    z : n-vector, the realized (un-recentered) instrument, z_i = f_i(v)
    Fmat : n x S matrix, ``Fmat[i, s] = f_i(v^(s))``, the formula at draw s
    p : S-vector of baseline probabilities; defaults to uniform, which is
        the postulated assignment distribution of Borusyak and Hull (2023)

    Returns
    -------
    dict with the realized criteria ``gy_v``, ``gx_v``, the S-vectors
    ``gy_s``, ``gx_s`` of criterion values at the counterfactual draws, the
    baseline weights ``p``, the baseline aggregates ``Egy``, ``Egx``, the
    recentered numerator/denominator ``Gy0``, ``Gx0`` at the baseline, the
    reported estimate ``beta_hat``, the sign ``flip`` applied by the
    normalization, and the sizes ``S`` and ``n``.
    """
    y = _as_f64(y).ravel()
    x = _as_f64(x).ravel()
    z = _as_f64(z).ravel()
    Fmat = _as_f64(Fmat)
    if Fmat.ndim != 2:
        raise ValueError("`Fmat` must be a 2-D array (n x S).")
    # stopifnot(length(y) == length(x), length(y) == length(z),
    #           nrow(Fmat) == length(y))
    if y.size != x.size:
        raise ValueError("length(y) == length(x) is not TRUE")
    if y.size != z.size:
        raise ValueError("length(y) == length(z) is not TRUE")
    if Fmat.shape[0] != y.size:
        raise ValueError("nrow(Fmat) == length(y) is not TRUE")
    S = int(Fmat.shape[1])
    if p is None:
        p = np.full(S, 1.0 / S, dtype=np.float64)   # rep(1 / S, S)
    else:
        p = _as_f64(p).ravel()
    # stopifnot(length(p) == S, all(p >= 0), abs(sum(p) - 1) < 1e-10)
    if p.size != S:
        raise ValueError("length(p) == S is not TRUE")
    if not bool(np.all(p >= 0)):
        raise ValueError("all(p >= 0) is not TRUE")
    if not abs(rsum(p) - 1) < 1e-10:
        raise ValueError("abs(sum(p) - 1) < 1e-10 is not TRUE")

    # R stores matrices column-major and evaluates `Fmat %*% p` and
    # `crossprod(Fc, y)` with BLAS dgemv (no-transpose and transpose,
    # respectively) on that layout.  numpy issues the very same two dgemv
    # calls when the operands are Fortran-ordered, so the products are
    # reproduced bitwise wherever R and numpy share the BLAS library; with
    # C-ordered operands numpy would call the transposed kernels instead,
    # whose summation order differs.
    Fmat = np.asfortranarray(Fmat)

    # Shift invariance (a): center the formula at its baseline mean unit by
    # unit, which is the recentering itself.  z must be shifted by the same
    # constant.
    mu = Fmat @ p                          # as.numeric(Fmat %*% p)
    Fc = np.asfortranarray(Fmat - mu[:, None])
    zc = z - mu

    gy_s = Fc.T @ y                        # as.numeric(crossprod(Fc, y)): S-vector
    gx_s = Fc.T @ x                        # g_y, g_x at each draw
    gy_v = rsum(y * zc)                    # scalar, g_y at the realized shocks
    gx_v = rsum(x * zc)

    # Exactly zero by construction after the centering above; kept explicit so
    # the formulas below read as they do in the manuscript.
    Egy = rsum(p * gy_s)
    Egx = rsum(p * gx_s)

    Gy0 = gy_v - Egy                       # G_y(Pbase) = recentered numerator
    Gx0 = gx_v - Egx                       # G_x(Pbase) = recentered first stage

    # Sign normalization (b).
    flip = -1.0 if Gx0 < 0 else 1.0
    gy_s = flip * gy_s
    gx_s = flip * gx_s
    gy_v = flip * gy_v
    gx_v = flip * gx_v
    Egy = flip * Egy
    Egx = flip * Egx
    Gy0 = flip * Gy0
    Gx0 = flip * Gx0

    # IEEE division as in R: a vanishing recentered first stage (Gx0 == 0,
    # e.g. a regressor that is identically zero after residualization) gives
    # +/-Inf, or NaN when the numerator vanishes too, never an exception.
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_hat = float(np.float64(Gy0) / np.float64(Gx0))

    return {"gy_s": gy_s, "gx_s": gx_s, "gy_v": float(gy_v), "gx_v": float(gx_v),
            "p": p, "Egy": float(Egy), "Egx": float(Egx),
            "Gy0": float(Gy0), "Gx0": float(Gx0),
            "beta_hat": beta_hat, "flip": flip, "S": S, "n": int(y.size)}


# -----------------------------------------------------------------------------
# 2.  Criterion bounds over the total variation ball  (Proposition SA7.3(i))
# -----------------------------------------------------------------------------
# For a criterion h with baseline distribution Pbase,
#
#   sup_{TV(P,Pbase) <= delta} E_P[h] = delta * sup h + int_delta^1 q_h(u) du,
#   inf_{TV(P,Pbase) <= delta} E_P[h] = delta * inf h + int_0^{1-delta} q_h(u) du,
#
# with q_h the quantile function of h under Pbase: the baseline mean of h once
# the lower (upper) tail of mass delta has been trimmed away and relocated to
# the most (least) favorable shock configuration.
#
# The shock space of the application is the finite set of counterfactual
# draws, so sup h = max_s h_s and the integral is a weighted sum over the
# atoms with the boundary atom split at the exact fraction of its mass that
# survives the trimming.  Nothing is rounded to whole atoms.
def _upper_tail_integral(h, p, a) -> float:
    """Integral of the quantile function of a discrete criterion over the top mass.

    Returns int_{a}^{1} q_h(u) du for h taking value ``h[s]`` with probability
    ``p[s]``, i.e. the baseline-weighted sum of the largest values of h
    carrying a total mass of 1 - a.  The atom straddling the cut is included
    at the fraction of its mass that lies above a.
    """
    h = _as_f64(h).ravel()
    p = _as_f64(p).ravel()
    if a >= 1:
        return 0.0
    if a <= 0:
        return rsum(p * h)
    o = rorder(h, decreasing=True)         # largest first
    hs = h[o]
    ps = p[o]
    cum = rcumsum(ps)
    m = 1 - a                              # mass to keep, from the top
    hit = np.flatnonzero(cum >= m - 1e-15)  # atom that straddles the cut
    if hit.size == 0:
        return rsum(ps * hs)               # numerical guard: keep everything
    k = int(hit[0])                        # 0-based; R's k - 1
    head_mass = float(cum[k - 1]) if k > 0 else 0.0
    return rsum(ps[:k] * hs[:k]) + (m - head_mass) * float(hs[k])


def tv_bound_upper(h, p, delta, h_sup=None) -> float:
    """Sensitivity bound on ``E_P[h]`` over a total variation ball of radius ``delta``.

    Parameters
    ----------
    h : S-vector of criterion values at the points of the shock space
    p : S-vector of baseline probabilities
    delta : scalar budget in ``[0, 1]``
    h_sup : the supremum of the criterion over the WHOLE shock space.
        Defaults to ``max(h)``, which is correct when the shock space is the
        finite set carrying ``p`` -- the reading of the exercise used in the
        application (see "Restricting the configuration space" in the
        manuscript remark on unbounded formulas).  Passing a wider value
        covers a shock space larger than the support of the baseline.
    """
    h = _as_f64(h).ravel()
    if h_sup is None:
        h_sup = float(np.max(h))
    return delta * h_sup + _upper_tail_integral(h, p, delta)


def tv_bound_lower(h, p, delta, h_inf=None) -> float:
    """Lower counterpart of :func:`tv_bound_upper`, by the reflection h -> -h."""
    h = _as_f64(h).ravel()
    if h_inf is None:
        h_inf = float(np.min(h))
    return -tv_bound_upper(-h, p, delta, h_sup=-h_inf)


# -----------------------------------------------------------------------------
# 3.  First-stage breakdown budgets  (Propositions SA7.3(ii) and SA7.4(ii))
# -----------------------------------------------------------------------------
# Below delta_FS the recentered first stage is bounded away from zero over the
# whole robustness set, so the estimator is well defined and the bounds on
# beta are finite; at and above it the neighborhood contains assignment
# distributions that make the first stage vanish and the bounds are vacuous.
def _fs_margin_tv(cr: dict, delta) -> float:
    """Is the recentered first stage bounded away from zero over the TV ball?

    Proposition SA7.3(ii): every P in the ball has
    ``G_x(P) >= g_x(v) - sup_P E_P[g_x]``, so a strictly positive right-hand
    side certifies that the estimator is well defined throughout.  This is
    the primitive condition; testing it directly, rather than comparing delta
    with the breakdown budget, is what keeps the boundary case right when the
    budget is censored at 1 by the empty-set convention (see
    :func:`fi_delta_fs_tv`).
    """
    return cr["gx_v"] - tv_bound_upper(cr["gx_s"], cr["p"], delta)


def _fs_margin_cont(cr: dict, delta) -> float:
    """Is the recentered first stage bounded away from zero over the
    contamination neighborhood?  Proposition SA7.4(ii): the relevant quantity
    is ``inf_{v'} G_x(v';delta) = G_x(Pbase) - delta * sup_{v'} (g_x(v') - E_Pbase[g_x])``.
    """
    return cr["Gx0"] - delta * (float(np.max(_as_f64(cr["gx_s"]))) - cr["Egx"])


def fi_delta_fs_tv(cr: dict, tol: float = 1e-12) -> BudgetValue:
    """First-stage breakdown budget over the total variation ball.

    ``delta_FS^TV := inf{ delta in [0,1] : g_x(v) <= sup_{P} E_P[g_x] }``,
    with the convention that the infimum over an empty set equals 1.  The map
    delta -> tv_bound_upper(g_x, .) is nondecreasing and continuous, so the
    infimum is located by bisection.

    The returned value carries the flag ``censored``, True when the set in
    the definition is empty, that is when the first stage never breaks down
    on ``[0,1]`` and the value 1 is the convention rather than an actual
    breakdown.  Callers that need to know whether the estimator is well
    defined AT a budget should use :func:`_fs_margin_tv` instead of comparing
    with this number.
    """
    def f(d):                              # nondecreasing in d
        return -_fs_margin_tv(cr, d)

    if f(1) < 0:
        return BudgetValue(1.0, censored=True)
    if f(0) >= 0:
        return BudgetValue(0.0, censored=False)
    lo = 0.0
    hi = 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return BudgetValue(hi, censored=False)


def fi_delta_fs_cont(cr: dict) -> BudgetValue:
    """First-stage breakdown budget over the contamination neighborhood.

    ``delta_FS^cont := min{1, G_x(Pbase) / sup_{v'} (g_x(v') - E_Pbase[g_x])}``,
    the ratio being +Inf when its denominator vanishes.  Carries the same
    ``censored`` flag as :func:`fi_delta_fs_tv`.
    """
    den = float(np.max(_as_f64(cr["gx_s"]))) - cr["Egx"]
    if den <= 0:
        return BudgetValue(1.0, censored=True)
    r = cr["Gx0"] / den
    return BudgetValue(min(1.0, r), censored=bool(r > 1))


# -----------------------------------------------------------------------------
# 4.  Bounds on the estimate over the total variation ball  (Prop. SA7.3(iii))
# -----------------------------------------------------------------------------
# beta(P) = b holds exactly when E_P[g_b] = g_b(v) with g_b = g_y - b g_x.
# The largest attainable estimate is therefore the b at which the realized
# number g_b(v) meets the LOWER criterion bound, and the smallest is the b at
# which it meets the UPPER one:
#
#   psi_delta(b) := g_b(v) - inf_P E_P[g_b]   is strictly DEcreasing, zero at
#                                             the upper bound on beta
#   phi_delta(b) := sup_P E_P[g_b] - g_b(v)   is strictly INcreasing, zero at
#                                             the lower bound on beta
#
# Both are convex, so each has a unique zero and a bracketed root finder is
# guaranteed to locate it.
def _psi(b, cr: dict, delta) -> float:
    """psi_delta of Proposition SA7.3(iii).

    The shock space is the finite set of counterfactual draws, so the
    extremes of g_b = g_y - b g_x over it are simply the extremes of the S
    evaluated values.
    """
    h = cr["gy_s"] - b * cr["gx_s"]
    return (cr["gy_v"] - b * cr["gx_v"]) - tv_bound_lower(h, cr["p"], delta)


def _phi(b, cr: dict, delta) -> float:
    """phi_delta of Proposition SA7.3(iii)."""
    h = cr["gy_s"] - b * cr["gx_s"]
    return tv_bound_upper(h, cr["p"], delta) - (cr["gy_v"] - b * cr["gx_v"])


def _monotone_root(f, start: float = 1, max_double: int = 60) -> float:
    """Solve for the unique zero of a strictly monotone function by bracket
    expansion followed by ``uniroot``.

    Grows a symmetric bracket around zero until the sign changes.  Below the
    first-stage breakdown budget ``f`` is finite, continuous and strictly
    monotone on the whole line and has a unique zero by Proposition
    SA7.3(iii), so this terminates; failing to bracket after ``max_double``
    doublings means one of those hypotheses is violated and is an error
    rather than an infinite bound.
    """
    lo = -float(start)
    hi = float(start)
    k = 0
    while True:
        s_lo = _rsign(f(lo))
        s_hi = _rsign(f(hi))
        if math.isnan(s_lo) or math.isnan(s_hi):
            raise RuntimeError("missing value where TRUE/FALSE needed")
        if s_lo != s_hi:
            break
        lo = 2 * lo
        hi = 2 * hi
        k += 1
        if k > max_double:
            raise RuntimeError("failed to bracket the root of a monotone function; "
                               "check that the budget is below the first-stage breakdown")
    return runiroot(f, lo, hi, tol=_DBL_EPSILON ** 0.75)


def fi_bounds_tv(cr: dict, delta) -> dict:
    """Bounds on the recentered IV estimate over the total variation ball.

    Parameters
    ----------
    cr : output of :func:`fi_criteria`
    delta : scalar budget.  Once the recentered first stage can vanish the
        bounds are (-Inf, +Inf), which is what Proposition SA7.3(iv) reports.

    Returns
    -------
    dict with keys ``lower`` and ``upper``.
    """
    if delta <= 0:
        return {"lower": cr["beta_hat"], "upper": cr["beta_hat"]}
    # Well-definedness is decided by the primitive positivity condition, not
    # by a comparison with the (possibly censored) breakdown budget.
    if _fs_margin_tv(cr, delta) <= 0:
        return {"lower": -math.inf, "upper": math.inf}
    up = _monotone_root(lambda b: _psi(b, cr, delta))
    lo = _monotone_root(lambda b: _phi(b, cr, delta))
    return {"lower": lo, "upper": up}


# -----------------------------------------------------------------------------
# 5.  Bounds on the estimate over the contamination neighborhood (Prop. SA7.4)
# -----------------------------------------------------------------------------
# Every P in the neighborhood is P = (1-delta) Pbase + delta R, so
#
#   G_y(P) = int G_y(v';delta) dR(v'),
#   G_y(v';delta) := G_y(Pbase) - delta (g_y(v') - E_Pbase[g_y])
#
# and likewise for G_x.  Below the first-stage breakdown budget the bounds
# are therefore the extremes of the ratio G_y(v';delta)/G_x(v';delta) over the
# shock space, each approached by a contaminating distribution degenerate at
# a single configuration.  No optimization is involved: one evaluates S
# ratios per budget.
def fi_bounds_cont(cr: dict, delta) -> dict:
    """Bounds on the recentered IV estimate over the contamination neighborhood.

    Returns
    -------
    dict with the two bounds ``lower``, ``upper`` and the index of the least
    favorable shock configuration attaining each of them, ``arg_lower`` and
    ``arg_upper``.  The indices are 0-based Python indices into the S draws
    (the R kernel returns the 1-based ``which.min``/``which.max``); they are
    ``nan`` at ``delta <= 0`` and at vacuous budgets, as in R.
    """
    if delta <= 0:
        return {"lower": cr["beta_hat"], "upper": cr["beta_hat"],
                "arg_lower": math.nan, "arg_upper": math.nan}
    if _fs_margin_cont(cr, delta) <= 0:
        return {"lower": -math.inf, "upper": math.inf,
                "arg_lower": math.nan, "arg_upper": math.nan}
    num = cr["Gy0"] - delta * (cr["gy_s"] - cr["Egy"])
    den = cr["Gx0"] - delta * (cr["gx_s"] - cr["Egx"])
    r = num / den
    return {"lower": float(np.min(r)), "upper": float(np.max(r)),
            "arg_lower": rwhich_min(r), "arg_upper": rwhich_max(r)}


# -----------------------------------------------------------------------------
# 6.  Paths and summary measures
# -----------------------------------------------------------------------------
def fi_paths(cr: dict, deltas) -> pd.DataFrame:
    """Trace both sets of bounds over a grid of budgets.

    Returns a data frame with one row per budget and columns ``delta``,
    ``tv_lower``, ``tv_upper``, ``cont_lower``, ``cont_upper``.
    """
    deltas = _as_f64(deltas).ravel()
    tv = [fi_bounds_tv(cr, float(d)) for d in deltas]
    ct = [fi_bounds_cont(cr, float(d)) for d in deltas]
    return pd.DataFrame({
        "delta": deltas,
        "tv_lower": np.array([r["lower"] for r in tv], dtype=np.float64),
        "tv_upper": np.array([r["upper"] for r in tv], dtype=np.float64),
        "cont_lower": np.array([r["lower"] for r in ct], dtype=np.float64),
        "cont_upper": np.array([r["upper"] for r in ct], dtype=np.float64),
    })


def fi_breakdown(lower, upper, tau_star: float = 0, hi: float = 1,
                 tol: float = 1e-10) -> float:
    """Breakdown budget for the reference value ``tau_star``.

    delta_b := inf{ delta : lower(delta) <= tau_star <= upper(delta) }, with
    the convention that the infimum over an empty set equals 1 (Section
    SA5.2).  ``lower`` and ``upper`` must be monotone functions of delta,
    which they are by Propositions SA7.3(iii) and SA7.4.

    Parameters
    ----------
    lower, upper : functions of a scalar budget

    Returns
    -------
    the budget, or ``nan`` when the reference value is never covered on
    ``[0, hi]`` -- reported as censored by the caller

    Notes
    -----
    The result is a discontinuous function of ``tau_star`` at the baseline
    estimate: when ``tau_star`` equals ``lower(0) == upper(0)`` exactly the
    budget is 0, whereas one ulp away the bisection runs down to its
    tolerance and returns about ``2 ** -34``.  A bound equal to NaN (a first
    stage vanishing at the baseline together with a vanishing numerator)
    makes the coverage test NA and is an error, as in R.
    """
    def covered(d):                        # lower(d) <= tau_star && tau_star <= upper(d)
        return _r_and(_r_le(lower(d), tau_star),
                      lambda: _r_le(tau_star, upper(d)))

    if _r_if(covered(0)):
        return 0.0
    if not _r_if(covered(hi)):
        return math.nan
    lo = 0.0
    up = float(hi)
    while up - lo > tol:
        mid = 0.5 * (lo + up)
        if _r_if(covered(mid)):
            up = mid
        else:
            lo = mid
    return up


# -----------------------------------------------------------------------------
# 7.  Least favorable assignment distributions
# -----------------------------------------------------------------------------
def fi_lf_law_tv(cr: dict, delta, b, direction: str = "upper") -> np.ndarray:
    """The least favorable distribution behind a total variation bound.

    At the bound ``b`` the binding program is the criterion program for
    g_b = g_y - b g_x, whose solution is described by Proposition SA7.3(i):
    the baseline with its lower (respectively upper) tail of mass ``delta``
    trimmed away and that mass relocated to the configuration at which g_b
    is largest (smallest).  The distribution is returned so that the
    exercise can report how concentrated the least favorable reweighting is.

    Parameters
    ----------
    direction : ``"upper"`` for the distribution solving ``sup E_P[g_b]``,
        which delivers the LOWER bound on the estimate, and ``"lower"`` for
        its mirror image

    Returns
    -------
    an S-vector of probabilities
    """
    if direction not in ("upper", "lower"):
        raise ValueError("`direction` should be one of \"upper\", \"lower\".")
    h = cr["gy_s"] - b * cr["gx_s"]
    p = _as_f64(cr["p"]).ravel()
    if delta <= 0:
        return p.copy()
    # Trim the tail of h that hurts, then relocate its mass to the extreme.
    o = rorder(h, decreasing=(direction == "lower"))  # trimmed first
    q = p.copy()
    left = delta
    for s in o:                            # remove a total mass of delta
        take = min(q[s], left)
        q[s] = q[s] - take
        left = left - take
        if left <= 1e-15:
            break
    j = rwhich_max(h) if direction == "upper" else rwhich_min(h)
    q[j] = q[j] + delta
    return q / rsum(q)
