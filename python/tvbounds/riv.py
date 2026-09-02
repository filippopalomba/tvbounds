"""User-facing interface for the recentered instrumental variables (formula
instruments) application of Palomba (2026), Section SA7.3 (port of
``R/riv.R``).  The computational kernels live in :mod:`tvbounds._riv_kernels`.
"""
from __future__ import annotations

import math
import numbers

import numpy as np
import pandas as pd

from ._object import NA, format_call, is_na, message, new_tvbounds
from ._rcompat import rsum
from ._riv_kernels import (fi_bounds_cont, fi_bounds_tv, fi_breakdown,
                           fi_criteria, fi_delta_fs_cont, fi_delta_fs_tv)

_NEIGHBORHOODS = ("tv", "contamination")


def tvbounds_riv(y, x, z, Fmat, p=None, controls=None,
                 delta=np.linspace(0, 1, 501), neighborhood="tv",
                 tau_star=0, verbose=False):
    """See the tvbounds manual."""
    cl = format_call("tvbounds_riv", {
        "y": y, "x": x, "z": z, "Fmat": Fmat, "p": p, "controls": controls,
        "delta": delta, "neighborhood": neighborhood, "tau_star": tau_star,
        "verbose": verbose})
    neighborhood = _tvb_match_arg(neighborhood, _NEIGHBORHOODS, "neighborhood")

    # --- input validation (fail early, name the argument) ----------------------
    y = _tvb_check_numeric_vector(y, "y")
    x = _tvb_check_numeric_vector(x, "x")
    z = _tvb_check_numeric_vector(z, "z")
    n = int(y.size)
    if x.size != n or z.size != n:
        raise ValueError("`y`, `x`, and `z` must have the same length.")
    if n < 2:
        raise ValueError("`y` must contain at least 2 observations.")
    if isinstance(Fmat, pd.DataFrame):
        Fmat = Fmat.to_numpy()
    Fmat = _tvb_as_numeric(Fmat)
    if Fmat is None or Fmat.ndim != 2:
        raise ValueError("`Fmat` must be a numeric matrix (n x S) of "
                         "counterfactual instrument draws.")
    if Fmat.shape[0] != n:
        raise ValueError("`Fmat` must have one row per observation "
                         "(nrow(Fmat) == length(y)).")
    S = int(Fmat.shape[1])
    if S < 2:
        raise ValueError("`Fmat` must contain at least 2 counterfactual draws (columns).")
    if not bool(np.all(np.isfinite(Fmat))):
        raise ValueError("`Fmat` must not contain missing or non-finite values.")
    if p is not None:
        p = _tvb_as_numeric(p)
        if p is None or p.ravel().size != S:
            raise ValueError("`p` must be a numeric vector with one probability "
                             "per column of `Fmat`.")
        p = p.ravel()
        if bool(np.any(np.isnan(p))) or bool(np.any(p < 0)):
            raise ValueError("`p` must be nonnegative with no missing values.")
        if abs(rsum(p) - 1) > 1e-8:
            raise ValueError("`p` must sum to 1.")
        p = p / rsum(p)                    # exact renormalization
    delta = _tvb_as_numeric(delta)
    if (delta is None or delta.size == 0
            or not bool(np.all(np.isfinite(delta)))):
        raise ValueError("`delta` must be a non-empty numeric vector with no "
                         "missing values.")
    delta = delta.ravel()
    if bool(np.any(delta < 0)) or bool(np.any(delta > 1)):
        raise ValueError("`delta` must lie in [0, 1] for the \"tv\" and "
                         "\"contamination\" neighborhoods.")
    if np.unique(delta).size != delta.size:
        raise ValueError("`delta` must not contain duplicated budget values.")
    tau_star = _tvb_check_scalar(tau_star)
    if tau_star is None:
        raise ValueError("`tau_star` must be a finite numeric scalar.")
    if not isinstance(verbose, (bool, np.bool_)):
        raise ValueError("`verbose` must be True or False.")
    verbose = bool(verbose)

    # --- Frisch-Waugh-Lovell residualization on the controls -------------------
    # In a just-identified two-stage least squares whose controls appear in
    # both stages, the coefficient on x is unchanged by partialling the
    # controls out.  The projection is idempotent, so applying it to z and F as
    # well (a column at a time) leaves every criterion value unchanged while
    # keeping all four inputs on the same residualized scale.
    W = _tvb_controls_matrix(controls, n)
    n_controls = int(W.shape[1] - 1)       # W always carries the constant
    if np.linalg.matrix_rank(W) >= n:
        raise ValueError("`controls` has as many columns as observations; "
                         "residualization is degenerate.")
    Q = _tvb_qr_basis(W)
    yr = y - Q @ (Q.T @ y)
    xr = x - Q @ (Q.T @ x)
    zr = z - Q @ (Q.T @ z)
    Fr = Fmat - Q @ (Q.T @ Fmat)

    # --- reduce the design to the two criterion functions ----------------------
    cr = fi_criteria(y=yr, x=xr, z=zr, Fmat=Fr, p=p)
    if verbose:
        message("Recentered IV design: n = %d observations, S = %d "
                "counterfactual draws." % (cr["n"], cr["S"]))
        message("Baseline recentered IV estimate: %.6g." % cr["beta_hat"])

    # --- first-stage breakdown budgets -----------------------------------------
    dfs_tv = fi_delta_fs_tv(cr)
    dfs_cont = fi_delta_fs_cont(cr)
    dfs = dfs_tv if neighborhood == "tv" else dfs_cont
    if verbose:
        message("First-stage breakdown budget (%s): %.4f%s." % (
            neighborhood, float(dfs),
            " (censored: never breaks down)" if dfs.censored else ""))

    # --- bounds along the budget grid ------------------------------------------
    if neighborhood == "tv":
        bm = [fi_bounds_tv(cr, float(d)) for d in delta]
    else:
        bm = [fi_bounds_cont(cr, float(d)) for d in delta]
    lower = np.array([r["lower"] for r in bm], dtype=np.float64)
    upper = np.array([r["upper"] for r in bm], dtype=np.float64)
    # Budgets at which the neighborhood contains a distribution collapsing the
    # first stage have a vacuous identified set (the whole real line); encode
    # those rows as NA in the common bounds schema.
    vacuous = ~np.isfinite(lower) | ~np.isfinite(upper)
    lower[vacuous] = NA
    upper[vacuous] = NA
    bounds = pd.DataFrame({"delta": delta, "lower": lower, "upper": upper})

    # --- breakdown budget for the reference value ------------------------------
    if neighborhood == "tv":
        def lo_fun(d):
            return fi_bounds_tv(cr, d)["lower"]

        def up_fun(d):
            return fi_bounds_tv(cr, d)["upper"]
    else:
        def lo_fun(d):
            return fi_bounds_cont(cr, d)["lower"]

        def up_fun(d):
            return fi_bounds_cont(cr, d)["upper"]
    db = fi_breakdown(lo_fun, up_fun, tau_star=tau_star)
    db_censored = bool(is_na(db))
    if db_censored:
        db = 1.0
    if verbose:
        message("Breakdown budget for tau_star = %.6g: %.4f%s." % (
            tau_star, db, " (censored: never covered)" if db_censored else ""))

    return new_tvbounds(
        application="riv",
        bounds=bounds,
        point=cr["beta_hat"],
        n=cr["n"],
        neighborhood=neighborhood,
        level=NA,
        B=NA,
        estimand_label="IV coefficient",
        call=cl,
        details={
            "criteria": cr,
            "delta_fs": float(dfs),
            "delta_fs_censored": bool(dfs.censored),
            "delta_fs_tv": dfs_tv,
            "delta_fs_cont": dfs_cont,
            "delta_breakdown": float(db),
            "delta_breakdown_censored": db_censored,
            "tau_star": tau_star,
            "n_controls": n_controls,
        },
    )


# -----------------------------------------------------------------------------
# Small validation helpers
# -----------------------------------------------------------------------------
def _tvb_match_arg(value, choices, name):
    """R's ``match.arg()`` (exact or unique partial match) with the argument name."""
    if value is None:
        return choices[0]                  # match.arg(NULL) returns the first choice
    if isinstance(value, (list, tuple)) and tuple(value) == tuple(choices):
        return choices[0]                  # the unevaluated default vector
    if isinstance(value, str):
        if value in choices:
            return value
        hits = [c for c in choices if value and c.startswith(value)]
        if len(hits) == 1:
            return hits[0]
    raise ValueError("`%s` should be one of %s." % (
        name, ", ".join('"%s"' % c for c in choices)))


def _tvb_as_numeric(v):
    """Float64 copy of ``v`` when it is numeric in R's sense (``is.numeric``),
    ``None`` otherwise.  Logical/boolean and character inputs are not numeric;
    ``None``/``NA`` entries of an object array become ``nan`` (R's ``NA``)."""
    if isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, (pd.Series, pd.Index)):
        v = v.to_numpy()
    a = np.asarray(v)
    if a.dtype.kind in "iuf":
        return a.astype(np.float64)
    if a.dtype.kind == "O":
        flat = a.ravel()
        out = np.empty(flat.size, dtype=np.float64)
        for k, el in enumerate(flat):
            if el is None or el is pd.NA:
                out[k] = math.nan
            elif isinstance(el, (bool, np.bool_)) or not isinstance(el, numbers.Real):
                return None
            else:
                out[k] = float(el)
        return out.reshape(a.shape)
    return None


def _tvb_check_numeric_vector(v, name):
    """Check that an argument is a finite numeric vector with no missing values."""
    a = _tvb_as_numeric(v)
    if a is None or a.ndim != 1:
        raise ValueError("`%s` must be a numeric vector." % name)
    if not bool(np.all(np.isfinite(a))):
        raise ValueError("`%s` must not contain missing or non-finite values." % name)
    return a


def _tvb_check_scalar(v):
    """A finite numeric scalar as a Python float, or ``None`` when ``v`` is not one."""
    a = _tvb_as_numeric(v)
    if a is None or a.size != 1:
        return None
    val = float(a.ravel()[0])
    if not math.isfinite(val):
        return None
    return val


def _tvb_controls_matrix(controls, n):
    """Assemble the control design matrix (constant always included)."""
    if controls is None:
        return np.ones((n, 1), dtype=np.float64)
    if isinstance(controls, pd.DataFrame):
        for col in controls.columns:
            dt = controls[col].dtype
            if not pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt):
                raise ValueError("`controls` must contain numeric columns only.")
        controls = controls.to_numpy()
    C = _tvb_as_numeric(controls)
    if C is None or C.ndim != 2:
        raise ValueError("`controls` must be a numeric matrix or data frame.")
    if C.shape[0] != n:
        raise ValueError("`controls` must have one row per observation.")
    if not bool(np.all(np.isfinite(C))):
        raise ValueError("`controls` must not contain missing or non-finite values.")
    return np.column_stack([np.ones(n, dtype=np.float64), C])


def _tvb_qr_basis(W, tol=1e-7):
    """Orthonormal basis of the column space of ``W`` for the residualization.

    R's ``qr()`` (LINPACK ``dqrdc2`` with ``tol = 1e-07``) cycles a column to
    the end of the pivot order when the norm of its part orthogonal to the
    columns accepted before it falls below ``tol`` times its original norm,
    and ``qr.resid()`` then projects on the accepted columns only.  We
    reproduce that column selection and, in the full-rank case (every
    column accepted, the common situation), return ``Q`` from the Householder
    factorization of ``W`` itself, as PY-DESIGN.md prescribes.
    """
    W = np.asarray(W, dtype=np.float64)
    accepted = []
    Q_acc = np.zeros((W.shape[0], 0), dtype=np.float64)
    for j in range(W.shape[1]):
        w = W[:, j]
        nrm = float(np.linalg.norm(w))
        r = w - Q_acc @ (Q_acc.T @ w) if Q_acc.shape[1] else w
        rn = float(np.linalg.norm(r))
        if nrm > 0 and rn >= tol * nrm:
            accepted.append(j)
            Q_acc = np.column_stack([Q_acc, r / rn])
    if len(accepted) == W.shape[1]:
        Q, _ = np.linalg.qr(W, mode="reduced")
        return Q
    Q, _ = np.linalg.qr(W[:, accepted], mode="reduced")
    return Q
