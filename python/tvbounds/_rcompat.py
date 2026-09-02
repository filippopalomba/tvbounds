"""R-compatible numerical primitives.

The R package computes every quantity with base R primitives whose
floating-point behaviour is well defined: sequential (left-to-right)
accumulation in ``sum()``/``cumsum()``, the two-pass mean of
``mean.default()``, the two-pass variance of ``var()``/``sd()``, the type-7
quantile of ``quantile()``, the bisection interpolation of ``approx()``,
Wichura's AS241 algorithm behind ``qnorm()``, and Brent's zeroin behind
``uniroot()``.  NumPy's defaults differ in the last bits (pairwise
summation in ``np.sum``, a different interpolation formula in
``np.quantile``, ...), so the helpers below reproduce the R algorithms
operation by operation.  On machines where R's ``long double`` accumulator
is a plain double (Apple Silicon and every platform without an 80-bit
extended type) the results are identical bit for bit; elsewhere they agree
to a few units in the last place.

All helpers accept anything ``np.asarray`` understands and return Python
floats or ``float64`` arrays.
"""
from __future__ import annotations

import math

import numpy as np

_DBL_EPSILON = float(np.finfo(np.float64).eps)

# R on Apple Silicon (and on every platform where the C compiler contracts
# a*b + c into a fused multiply-add) evaluates a handful of its compiled
# kernels with FMA instructions: the variance accumulation of var()/sd()
# (stats/src/cov.c), the linear interpolation of approx() (stats/src/approx.c),
# the inverse-quadratic step of uniroot() (src/library/stats/src/zeroin.c)
# and the Horner polynomials of qnorm() (src/nmath/qnorm.c).  Reproducing
# those results bit for bit requires the same fused rounding, so the helpers
# below use math.fma (Python >= 3.13) with an exact rational fallback.
try:
    from math import fma as _math_fma
except ImportError:  # Python < 3.13
    _math_fma = None
from fractions import Fraction as _Fraction


def fma(a: float, b: float, c: float) -> float:
    """Fused multiply-add ``a * b + c`` rounded once (exact fallback)."""
    if _math_fma is not None:
        return _math_fma(a, b, c)
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        return a * b + c
    return float(_Fraction(a) * _Fraction(b) + _Fraction(c))


# ---------------------------------------------------------------------------
# sum / cumsum / mean / var / sd
# ---------------------------------------------------------------------------
def rsum(x) -> float:
    """R's ``sum()`` for doubles: sequential left-to-right accumulation."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    return float(np.cumsum(x)[-1])


def rcumsum(x) -> np.ndarray:
    """R's ``cumsum()`` for doubles (sequential accumulation)."""
    return np.cumsum(np.asarray(x, dtype=np.float64).ravel())


def rmean(x) -> float:
    """R's ``mean.default()`` for doubles: ``s = sum(x)/n; s + sum(x - s)/n``.

    Integer-valued inputs that R would store as integers (counts, 0/1
    indicators) are handled by :func:`rmean_int`.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return math.nan
    s = rsum(x) / n
    if math.isfinite(s):
        s = s + rsum(x - s) / n
    return float(s)


def rmean_int(x) -> float:
    """R's ``mean()`` of an integer/logical vector: ``sum(x) / n`` (exact)."""
    x = np.asarray(x).ravel()
    n = x.size
    if n == 0:
        return math.nan
    return float(int(np.sum(x.astype(np.int64))) / n)


def rvar(x, na_rm: bool = False) -> float:
    """R's ``var(x, na.rm = na_rm)`` of a double vector.

    Two-pass mean, then the sequential sum of squared deviations divided by
    ``n - 1``.  R evaluates the two settings of ``na.rm`` in different C code
    paths of ``cov.c``; on this package's reference build (R 4.5, Apple
    Silicon) the ``na.rm = TRUE`` path accumulates with fused multiply-adds
    while the default path does not, and both are reproduced (verified bit
    for bit against R).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if na_rm:
        x = x[~np.isnan(x)]
    elif np.isnan(x).any():
        return math.nan
    n = x.size
    if n < 2:
        return math.nan
    m = rmean(x)
    d = x - m
    if not na_rm:
        return rsum(d * d) / (n - 1)
    s = 0.0
    for dk in d.tolist():                 # sum += (x[k] - m) * (x[k] - m), fused
        s = fma(dk, dk, s)
    return s / (n - 1)


def rsd(x, na_rm: bool = False) -> float:
    """R's ``sd(x, na.rm = na_rm)``."""
    v = rvar(x, na_rm=na_rm)
    return math.sqrt(v) if not math.isnan(v) else math.nan


# ---------------------------------------------------------------------------
# quantile (type 7) and approx
# ---------------------------------------------------------------------------
def rquantile7(x, probs, na_rm: bool = False):
    """R's ``quantile(x, probs, type = 7, na.rm = na_rm)`` (default type).

    Returns a float for a scalar ``probs`` and a float64 array otherwise.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if na_rm:
        x = x[~np.isnan(x)]
    elif np.isnan(x).any():
        raise ValueError("missing values and NaN's not allowed if 'na.rm' is FALSE")
    scalar = np.ndim(probs) == 0
    pr = np.atleast_1d(np.asarray(probs, dtype=np.float64))
    eps = 100 * _DBL_EPSILON
    if np.any((pr < -eps) | (pr > 1 + eps)):
        raise ValueError("'probs' outside [0,1]")
    pr = np.maximum(0.0, np.minimum(1.0, pr))
    n = x.size
    out = np.empty(pr.size, dtype=np.float64)
    if n == 0:
        out[:] = math.nan
        return float(out[0]) if scalar else out
    xs = np.sort(x)
    for k, p in enumerate(pr):
        index = 1 + max(n - 1, 0) * p
        lo = int(math.floor(index))
        hi = int(math.ceil(index))
        qs = xs[lo - 1]
        if index > lo and xs[hi - 1] != qs:
            h = index - lo
            qs = (1 - h) * qs + h * xs[hi - 1]
        out[k] = qs
    return float(out[0]) if scalar else out


def _approx1(v: float, x: np.ndarray, y: np.ndarray) -> float:
    """R's ``approx1()`` (linear, ``rule = 2``): bisection on sorted ``x``."""
    n = x.size
    if n == 0:
        return math.nan
    i = 0
    j = n - 1
    if v < x[i]:
        return float(y[0])
    if v > x[j]:
        return float(y[j])
    while i < j - 1:
        ij = (i + j) // 2
        if v < x[ij]:
            j = ij
        else:
            i = ij
    if v == x[j]:
        return float(y[j])
    if v == x[i]:
        return float(y[i])
    # y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i])), fused as in R's build
    return fma(float(y[j] - y[i]), float((v - x[i]) / (x[j] - x[i])), float(y[i]))


def rapprox(x, y, xout, ties: str = "mean"):
    """R's ``approx(x, y, xout, method = "linear", rule = 2, ties = ties)``.

    With ``ties = "mean"`` (R's default) tied ``x`` values are collapsed to
    the mean of their ``y`` values and the knots sorted; with
    ``ties = "ordered"`` the knots are used as supplied (they must be
    nondecreasing).  Rows with a missing ``x`` or ``y`` are dropped, as in R.
    Returns a float for scalar ``xout`` and a float64 array otherwise.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ok = ~(np.isnan(x) | np.isnan(y))
    x = x[ok]
    y = y[ok]
    if ties != "ordered":
        o = np.argsort(x, kind="stable")
        x = x[o]
        y = y[o]
        if x.size and np.unique(x).size < x.size:
            ux, inv = np.unique(x, return_inverse=True)
            uy = np.array([rmean(y[inv == k]) for k in range(ux.size)])
            x, y = ux, uy
    scalar = np.ndim(xout) == 0
    xo = np.atleast_1d(np.asarray(xout, dtype=np.float64))
    if x.size < 2:
        raise ValueError("need at least two non-NA values to interpolate")
    out = np.array([_approx1(float(v), x, y) if not math.isnan(v) else math.nan
                    for v in xo], dtype=np.float64)
    return float(out[0]) if scalar else out


# ---------------------------------------------------------------------------
# qnorm (Wichura, AS241, as in R's qnorm.c)
# ---------------------------------------------------------------------------
def _horner(r: float, coefs) -> float:
    """``(((c0 * r + c1) * r + c2) ...)`` with each step fused (R's build)."""
    v = fma(r, coefs[0], coefs[1])
    for c in coefs[2:]:
        v = fma(v, r, c)
    return v


def rqnorm(p: float) -> float:
    """R's ``qnorm(p)`` (lower tail, natural scale): the AS241 algorithm."""
    p = float(p)
    if math.isnan(p):
        return math.nan
    if p < 0 or p > 1:
        return math.nan
    if p == 0:
        return -math.inf
    if p == 1:
        return math.inf
    q = p - 0.5
    if abs(q) <= 0.425:
        r = fma(-q, q, 0.180625)
        num = _horner(r, (2509.0809287301226727, 33430.575583588128105,
                          67265.770927008700853, 45921.953931549871457,
                          13731.693765509461125, 1971.5909503065514427,
                          133.14166789178437745, 3.387132872796366608))
        den = _horner(r, (5226.495278852545925, 28729.085735721942674,
                          39307.89580009271061, 21213.794301586595867,
                          5394.1960214247511077, 687.1870074920579083,
                          42.313330701600911252, 1.0))
        return q * num / den
    r = 1.0 - p if q > 0 else p
    r = math.sqrt(-math.log(r))
    if r <= 5.0:
        r += -1.6
        num = _horner(r, (7.7454501427834140764e-4, 0.0227238449892691845833,
                          0.24178072517745061177, 1.27045825245236838258,
                          3.64784832476320460504, 5.7694972214606914055,
                          4.6303378461565452959, 1.42343711074968357734))
        den = _horner(r, (1.05075007164441684324e-9, 5.475938084995344946e-4,
                          0.0151986665636164571966, 0.14810397642748007459,
                          0.68976733498510000455, 1.6763848301838038494,
                          2.05319162663775882187, 1.0))
        val = num / den
    else:
        r += -5.0
        num = _horner(r, (2.01033439929228813265e-7, 2.71155556874348757815e-5,
                          0.0012426609473880784386, 0.026532189526576123093,
                          0.29656057182850489123, 1.7848265399172913358,
                          5.4637849111641143699, 6.6579046435011037772))
        den = _horner(r, (2.04426310338993978564e-15, 1.4215117583164458887e-7,
                          1.8463183175100546818e-5, 7.868691311456132591e-4,
                          0.0148753612908506148525, 0.13692988092273580531,
                          0.59983220655588793769, 1.0))
        val = num / den
    if q < 0.0:
        val = -val
    return val


# ---------------------------------------------------------------------------
# uniroot (R_zeroin2, Brent's method as in R's zeroin.c)
# ---------------------------------------------------------------------------
def r_zeroin2(f, ax: float, bx: float, fa: float, fb: float,
              tol: float, maxit: int):
    """R's ``R_zeroin2()``: Brent's root finder on ``[ax, bx]``.

    ``fa``/``fb`` are ``f(ax)``/``f(bx)``.  Returns ``(root, iterations,
    estimated_precision)``; ``iterations`` is ``-1`` when ``maxit`` was
    exhausted, as in R.
    """
    a = ax
    b = bx
    c = a
    fc = fa
    maxit_total = maxit + 1
    if fa == 0.0:
        return a, 0, 0.0
    if fb == 0.0:
        return b, 0, 0.0
    remaining = maxit_total
    while remaining:
        remaining -= 1
        prev_step = b - a
        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa
        tol_act = 2 * _DBL_EPSILON * abs(b) + tol / 2
        new_step = (c - b) / 2
        if abs(new_step) <= tol_act or fb == 0.0:
            return b, maxit_total - remaining - 1, abs(c - b)
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:
                q = fa / fc
                t1 = fb / fc
                t2 = fb / fa
                # t2 * (cb*q*(q-t1) - (b-a)*(t1-1)), first product fused (R's build)
                p = t2 * fma(cb * q, (q - t1), -((b - a) * (t1 - 1.0)))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            if (p < (0.75 * cb * q - abs(tol_act * q) / 2)
                    and p < abs(prev_step * q / 2)):
                new_step = p / q
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0.0 else -tol_act
        a = b
        fa = fb
        b += new_step
        fb = f(b)
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c = a
            fc = fa
    return b, -1, -1.0


def runiroot(f, lower: float, upper: float, tol: float = _DBL_EPSILON ** 0.25,
             maxiter: int = 1000) -> float:
    """R's ``uniroot(f, c(lower, upper), tol = tol, maxiter = maxiter)$root``."""
    if not (lower < upper):
        raise ValueError("lower < upper  is not fulfilled")
    f_lower = f(lower)
    f_upper = f(upper)
    if math.isnan(f_lower):
        raise ValueError("f.lower = f(lower) is NA")
    if math.isnan(f_upper):
        raise ValueError("f.upper = f(upper) is NA")
    if f_lower * f_upper > 0:
        raise ValueError("f() values at end points not of opposite sign")
    root, it, _ = r_zeroin2(f, lower, upper, f_lower, f_upper, tol, maxiter)
    return float(root)


# ---------------------------------------------------------------------------
# order / sort helpers (R's default radix order is stable)
# ---------------------------------------------------------------------------
def rorder(x, decreasing: bool = False) -> np.ndarray:
    """R's ``order(x, decreasing = decreasing)`` (0-based, ties stable)."""
    x = np.asarray(x)
    if decreasing:
        # stable descending order: sort ascending on the negated ranks
        idx = np.argsort(-x if np.issubdtype(x.dtype, np.number) else x,
                         kind="stable")
        if not np.issubdtype(x.dtype, np.number):
            idx = idx[::-1]
        return idx
    return np.argsort(x, kind="stable")


def rwhich_max(x) -> int:
    """R's ``which.max(x)`` as a 0-based index (first maximum)."""
    x = np.asarray(x, dtype=np.float64)
    return int(np.argmax(x))


def rwhich_min(x) -> int:
    """R's ``which.min(x)`` as a 0-based index (first minimum)."""
    x = np.asarray(x, dtype=np.float64)
    return int(np.argmin(x))


def rsignif(x: float, digits: int) -> float:
    """R's ``signif(x, digits)`` for a scalar (nmath's fprec algorithm)."""
    from ._object import _r_signif
    return float(_r_signif(float(x), int(digits)))
