"""The common result object of the estimators, plus shared small helpers.

Mirrors ``R/tvbounds-object.R`` and the formatting helpers of
``R/methods.R`` of the R package.  Owned by the coordinator: the estimator
modules build their return values through :func:`new_tvbounds` (or the
:class:`TVBounds` constructor, which performs the same validation) and must
not modify this file.
"""
from __future__ import annotations

import math
import sys
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

NA = math.nan  # R's NA for numeric fields

_APPLICATIONS = ("attrition", "counterfactual", "riv")
_NEIGHBORHOODS = ("tv", "contamination")


# ---------------------------------------------------------------------------
# messages, warnings, errors (R's message()/warning()/stop() equivalents)
# ---------------------------------------------------------------------------
def message(text: str) -> None:
    """R's ``message()``: a diagnostic line on standard error."""
    print(text, file=sys.stderr, flush=True)


def warn(text: str) -> None:
    """R's ``warning(..., call. = FALSE)``."""
    warnings.warn(text, UserWarning, stacklevel=3)


def is_na(x) -> bool:
    """True for ``None`` and for float ``NaN`` (R's ``is.na`` on a scalar)."""
    if x is None:
        return True
    try:
        return bool(np.isnan(x))
    except TypeError:
        return False


def or_null(x, y):
    """R's ``x %||% y``: ``y`` when ``x`` is ``None``."""
    return y if x is None else x


def format_call(fname: str, args: dict) -> str:
    """A compact textual record of the user-facing call (R's ``match.call()``)."""
    parts = []
    for k, v in args.items():
        parts.append(f"{k}={_short_repr(v)}")
    return f"{fname}({', '.join(parts)})"


def _short_repr(v) -> str:
    if isinstance(v, pd.DataFrame):
        return f"<DataFrame {v.shape[0]}x{v.shape[1]}>"
    if isinstance(v, np.ndarray):
        return f"<ndarray {v.shape}>"
    if isinstance(v, pd.Series):
        return f"<Series ({v.shape[0]},)>"
    if isinstance(v, (list, tuple)) and len(v) > 6:
        return f"<{type(v).__name__} of length {len(v)}>"
    if callable(v):
        return getattr(v, "__name__", "<function>")
    r = repr(v)
    return r if len(r) <= 60 else r[:57] + "..."


# ---------------------------------------------------------------------------
# a float carrying R's "censored" attribute
# ---------------------------------------------------------------------------
class BudgetValue(float):
    """A budget value carrying the ``censored`` attribute of the R package.

    R attaches ``attr(x, "censored")`` to the first-stage breakdown budgets
    of ``tvbounds_riv()``; this float subclass carries the same flag as
    ``x.censored`` while behaving as an ordinary number.
    """

    def __new__(cls, value, censored: bool = False):
        obj = float.__new__(cls, value)
        obj.censored = bool(censored)
        return obj

    def __repr__(self):
        return f"{float(self)!r} (censored={self.censored})"

    def __reduce__(self):
        return (BudgetValue, (float(self), self.censored))


# ---------------------------------------------------------------------------
# formatting helpers shared by the print methods (R/methods.R)
# ---------------------------------------------------------------------------
def _fmt(v, digits: int = 3) -> str:
    """R's ``.tvb_fmt()``: ``format(signif(v, digits))``, ``"---"`` when missing."""
    if v is None:
        return "---"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "---"
    if math.isnan(v):
        return "---"
    if math.isinf(v):
        return "Inf" if v > 0 else "-Inf"
    return _format_r(_r_signif(v, digits), 7)


# R's signif() (src/nmath/fprec.c) and format() (formatReal() in
# src/main/format.c) reproduced operation by operation, so that the print
# methods render exactly the text of the R package.  On this platform R's
# long double is a plain double, and the scaled-rounding arithmetic below is
# the one R performs.
_MAX_DIGITS = 22          # fprec.c
_KP_MAX = 22              # format.c (double table)
_TBL = [1e-1] + [10.0 ** k for k in range(0, _KP_MAX + 1)]   # _TBL[k + 1] == 10^k
_INT_MAX = 2147483647


def _r_pow_di(x: float, n: int) -> float:
    """R's ``R_pow_di(x, n)``: ``x^n`` by repeated squaring."""
    xn = 1.0
    if math.isnan(x):
        return x
    if n != 0:
        if not math.isfinite(x):
            return x ** n
        is_neg = n < 0
        if is_neg:
            n = -n
        while True:
            if n & 1:
                xn *= x
            n >>= 1
            if n:
                x *= x
            else:
                break
        if is_neg:
            xn = 1.0 / xn
    return xn


def _r_nearbyint(x: float) -> float:
    """C ``nearbyint()`` in the default rounding mode (half to even)."""
    if not math.isfinite(x):
        return x
    return float(round(x))


def _r_signif(x, digits) -> float:
    """R's ``signif(x, digits)`` for a scalar (``fprec()`` in nmath)."""
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return x
    if x == 0:
        return x
    dig = int(math.floor(digits + 0.5)) if digits >= 0 else int(math.ceil(digits - 0.5))
    if dig > _MAX_DIGITS:
        return x
    if dig < 1:
        dig = 1
    sgn = 1.0
    if x < 0.0:
        sgn = -sgn
        x = -x
    max10e = 308  # DBL_MAX_10_EXP
    l10 = math.log10(x)
    e10 = int(dig - 1 - math.floor(l10))
    if abs(l10) < max10e - 2:
        p10 = 1.0
        if e10 > max10e:
            p10 = _r_pow_di(10.0, e10 - max10e)
            e10 = max10e
        if e10 > 0:
            pow10 = _r_pow_di(10.0, e10)
            return sgn * (_r_nearbyint((x * pow10) * p10) / pow10) / p10
        pow10 = _r_pow_di(10.0, -e10)
        return sgn * (_r_nearbyint(x / pow10) * pow10)
    do_round = max10e - l10 >= _r_pow_di(10.0, -dig)
    e2 = dig + (_MAX_DIGITS if e10 > 0 else -_MAX_DIGITS)
    p10 = _r_pow_di(10.0, e2)
    x *= p10
    P10 = _r_pow_di(10.0, e10 - e2)
    x *= P10
    if do_round:
        x += 0.5
    x = math.floor(x) / p10
    return sgn * x / P10


def _r_scientific(x: float, digits: int):
    """R's ``scientific()`` (format.c): ``(neg, kpower, nsig, roundingwidens)``."""
    if x == 0.0:
        return 0, 0, 1, False
    if x < 0.0:
        neg, r = 1, -x
    else:
        neg, r = 0, x
    if digits >= 16:  # DBL_DIG + 1: format_via_sprintf()
        mant, ex = f"{r:.{digits - 1}e}".split("e")
        digs = mant.replace(".", "").rstrip("0")
        return neg, int(ex), max(len(digs), 1), False
    kp = int(math.floor(math.log10(r))) - digits + 1
    r_prec = r
    if abs(kp) < 23:
        if kp > 0:
            r_prec /= _TBL[kp + 1]
        elif kp < 0:
            r_prec *= _TBL[-kp + 1]
    elif kp <= -307:  # R_dec_min_exponent
        r_prec = (r * 1e+303) / (10.0 ** (kp + 303))
    else:
        r_prec /= 10.0 ** kp
    if r_prec < _TBL[digits]:
        r_prec *= 10.0
        kp -= 1
    alpha = _r_nearbyint(r_prec)
    nsig = digits
    for _ in range(1, digits + 1):
        alpha /= 10.0
        if alpha == math.floor(alpha):
            nsig -= 1
        else:
            break
    if nsig == 0 and digits > 0:
        nsig = 1
        kp += 1
    kpower = kp + digits - 1
    rgt = digits - kpower
    rgt = 0 if rgt < 0 else (_KP_MAX if rgt > _KP_MAX else rgt)
    fuzz = 0.5 / _TBL[1 + rgt]
    roundingwidens = kpower > 0 and kpower <= _KP_MAX and r < _TBL[kpower + 1] - fuzz
    return neg, kpower, nsig, roundingwidens


def _format_r(x, digits: int = 7) -> str:
    """R's ``format(x)`` of one double at ``getOption("digits")`` significant digits.

    Reproduces the fixed-versus-scientific choice of ``formatReal()`` (fixed
    notation whenever it is not wider than scientific, ``scipen = 0``) and
    the C formatting of ``encodeReal0()``.
    """
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == 0.0:
        x = 0.0  # IEEE signed zero prints as "0"
    neg, kpower, nsig, rw = _r_scientific(x, digits)
    left = kpower + 1
    if rw:
        left -= 1
    sleft = neg + (1 if left <= 0 else left)
    right = nsig - left
    rgt = right if right > 0 else 0
    mxsl = sleft if left >= 0 else 1 + neg
    wF = mxsl + rgt + (1 if rgt != 0 else 0)
    e = 2 if (left > 100 or left <= -99) else 1
    d = nsig - 1
    w = neg + (1 if d > 0 else 0) + d + 4 + e
    if wF <= w:  # scipen = 0
        return f"{x:.{rgt}f}"
    return f"{x:.{d}e}"


def _format_signif(v: float, digits: int) -> str:
    """R's ``format(signif(v, digits))`` for a scalar."""
    return _format_r(_r_signif(v, digits), 7)


def _fmt_int(v, signed: bool = False) -> str:
    """R's ``.tvb_fmt_int()``: integer with thousands separators."""
    if v is None:
        return "---"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "---"
    if math.isnan(v):
        return "---"
    if math.isinf(v):
        return "Inf" if v > 0 else "-Inf"
    r = float(round(abs(v)))  # R's round(): half to even
    out = "NA" if r > _INT_MAX else f"{int(r):,}"
    sgn = "-" if v < 0 else ("+" if signed else "")
    return sgn + out


def _neigh_label(neighborhood, divergence) -> str:
    """R's ``.tvb_neigh_label()``."""
    if neighborhood is not None:
        return {"tv": "total-variation neighborhood",
                "contamination": "contamination neighborhood"}.get(
                    neighborhood, str(neighborhood))
    if divergence is not None:
        return f"{divergence} divergence neighborhood"
    return "unspecified neighborhood"


# ---------------------------------------------------------------------------
# the result object
# ---------------------------------------------------------------------------
class TVBounds:
    """Result object shared by the three estimators (R's class ``"tvbounds"``).

    Attributes mirror the components of the R list: ``application``,
    ``bounds`` (a pandas DataFrame with columns ``delta``, ``lower``,
    ``upper`` and, with inference, ``lower_se``, ``upper_se``, ``ci_lower``,
    ``ci_upper``), ``point``, ``n``, ``neighborhood``, ``divergence``,
    ``level``, ``B``, ``estimand_label``, ``call``, and ``details`` (a dict).
    Components can also be read with ``fit["bounds"]``.  ``print(fit)``
    displays the compact summary of the R print method; ``fit.plot()`` and
    ``fit.summary()`` dispatch to :func:`tvbounds_plot` and
    :func:`tvbounds_summary`.
    """

    def __init__(self, application: str, bounds: pd.DataFrame,
                 point: float = NA, n: float = NA,
                 neighborhood: Optional[str] = None,
                 divergence: Optional[str] = None,
                 level: float = NA, B: float = NA,
                 estimand_label: str = "estimand",
                 call: Optional[str] = None,
                 details: Optional[dict] = None):
        if not isinstance(application, str) or application not in _APPLICATIONS:
            raise ValueError("`application` must be one of "
                             "\"attrition\", \"counterfactual\", \"riv\".")
        if not isinstance(bounds, pd.DataFrame):
            raise ValueError("`bounds` must be a data frame.")
        for col in ("delta", "lower", "upper"):
            if col not in bounds.columns:
                raise ValueError("`bounds` must have columns `delta`, `lower`, `upper`.")
        details = {} if details is None else details
        if not isinstance(details, dict):
            raise ValueError("`details` must be a dict.")
        if neighborhood is not None and neighborhood not in _NEIGHBORHOODS:
            raise ValueError("`neighborhood` must be \"tv\" or \"contamination\".")
        d = bounds["delta"].to_numpy(dtype=np.float64)
        # anyDuplicated(bounds$delta): two NA budgets count as duplicates too
        if pd.Series(d).duplicated().any():
            raise ValueError("`bounds$delta` must not contain duplicated budget values.")
        if np.any(d[~np.isnan(d)] < 0):
            raise ValueError("`bounds$delta` must be nonnegative.")
        gap = bounds["upper"].to_numpy(dtype=np.float64) - \
            bounds["lower"].to_numpy(dtype=np.float64)
        bad = np.where(gap < -1e-8)[0]
        if bad.size:
            vals = ", ".join(_format_r(_r_signif(float(v), 4), 15) for v in d[bad])
            raise ValueError(f"Lower bound exceeds upper bound at delta = {vals}.")
        bounds = bounds.sort_values("delta", kind="stable").reset_index(drop=True)
        self.application = application
        self.bounds = bounds
        self.point = point
        self.n = n
        self.neighborhood = neighborhood
        self.divergence = divergence
        self.level = level
        self.B = B
        self.estimand_label = estimand_label
        self.call = call
        self.details = details

    # R's list-style access: fit["bounds"]
    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def keys(self):
        return ["application", "bounds", "point", "n", "neighborhood",
                "divergence", "level", "B", "estimand_label", "call", "details"]

    # S3 dispatch equivalents
    def plot(self, *args, **kwargs):
        from .plot import tvbounds_plot
        return tvbounds_plot(self, *args, **kwargs)

    def summary(self, *args, **kwargs):
        from .summary import tvbounds_summary
        return tvbounds_summary(self, *args, **kwargs)

    # print.tvbounds
    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return self.format()

    def format(self, digits: int = 3) -> str:
        """The text of R's ``print.tvbounds(x, digits)``."""
        d = self.bounds["delta"].to_numpy(dtype=np.float64)
        lines = [f"<tvbounds> {self.application} bounds under a "
                 f"{_neigh_label(self.neighborhood, self.divergence)}",
                 f"  estimand: {or_null(self.estimand_label, 'estimand')}; "
                 f"baseline point estimate (delta = 0): {_fmt(self.point, digits)}; "
                 f"n = {_fmt_int(self.n)}",
                 f"  budget grid: {d.size} values of delta in "
                 f"[{_fmt(np.nanmin(d) if d.size else NA, digits)}, "
                 f"{_fmt(np.nanmax(d) if d.size else NA, digits)}]"]
        if not is_na(self.level):
            lines.append(f"  inference: {_format_r(100 * float(self.level), 7)}% "
                         f"bootstrap confidence bands (B = {_fmt_int(self.B)})")
        else:
            lines.append("  inference: none attached")
        lines.append("  Use summary() for breakdown and price-of-robustness "
                     "measures; plot() to display.")
        return "\n".join(lines)

    def print(self, digits: int = 3, **kwargs):
        """R's ``print.tvbounds()``: print the summary and return ``self``."""
        print(self.format(digits))
        return self


def new_tvbounds(application: str, bounds: pd.DataFrame, point: float = NA,
                 n: float = NA, neighborhood: Optional[str] = None,
                 divergence: Optional[str] = None, level: float = NA,
                 B: float = NA, estimand_label: str = "estimand",
                 call: Optional[str] = None,
                 details: Optional[dict] = None) -> TVBounds:
    """Internal constructor shared by the estimators (R's ``new_tvbounds()``)."""
    return TVBounds(application, bounds, point=point, n=n,
                    neighborhood=neighborhood, divergence=divergence,
                    level=level, B=B, estimand_label=estimand_label,
                    call=call, details=details)


def is_tvbounds(x: Any) -> bool:
    """Test whether an object is a :class:`TVBounds` result."""
    return isinstance(x, TVBounds)


class TVBoundsSummary:
    """The object returned by :func:`tvbounds_summary` (R's ``"tvbounds_summary"``).

    Attributes: ``measures`` (a one-row pandas DataFrame) and the metadata
    fields ``application``, ``estimand_label``, ``neighborhood``,
    ``divergence``, ``direction``, ``tau_star``, ``delta_eval``, ``level``,
    ``band_level``, ``zc``, ``cost_per_unit``, ``jump``, ``has_se``,
    ``has_ci``, ``notes``, ``call``.  ``print(s)`` renders the compact
    display of the R print method.
    """

    _FIELDS = ("measures", "application", "estimand_label", "neighborhood",
               "divergence", "direction", "tau_star", "delta_eval", "level",
               "band_level", "zc", "cost_per_unit", "jump", "has_se",
               "has_ci", "notes", "call")

    def __init__(self, **fields):
        missing = [f for f in self._FIELDS if f not in fields]
        if missing:
            raise ValueError("TVBoundsSummary is missing fields: " + ", ".join(missing))
        for k, v in fields.items():
            setattr(self, k, v)

    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def keys(self):
        return list(self._FIELDS)

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return self.format()

    def format(self, digits: int = 3) -> str:
        """The text of R's ``print.tvbounds_summary(x, digits)``."""
        m = self.measures.iloc[0]
        lines = [f"<tvbounds summary> {or_null(self.estimand_label, 'estimand')} "
                 f"({self.application} application, "
                 f"{_neigh_label(self.neighborhood, self.divergence)})",
                 f"  direction: {m['direction']} bound path relative to tau_star = "
                 f"{_fmt(m['tau_star'], digits)} (baseline point = "
                 f"{_fmt(m['point'], digits)}, n = {_fmt_int(m['n'])})"]
        bd_txt = _fmt(m["delta_b"], digits)
        if bool(m["censored"]):
            bd_txt += " (censored)"
        ci_txt = _fmt(m["delta_b_ci"], digits)
        if bool(m["censored_ci"]):
            ci_txt += " (censored)"
        lines.append(f"  breakdown budget: plug-in = {bd_txt}; certified = {ci_txt}; "
                     f"normal floor = {_fmt(m['delta_b_ci_norm'], digits)}")
        lines.append(f"  at delta = {_fmt(m['delta_eval'], digits)}: shadow price eta = "
                     f"{_fmt(m['eta'], digits)}; robustness SE varsigma = "
                     f"{_fmt(m['varsigma'], digits)} (scale-free "
                     f"{_fmt(m['varsigma_sc'], digits)})")
        lines.append(f"  certification frontier: n* = {_fmt_int(m['n_star'])} at budget "
                     f"{_fmt(m['frontier_at'], digits)} + jump {_fmt(self.jump, digits)} "
                     f"(Delta n = {_fmt_int(m['delta_n'], signed=True)}, cost per pp = "
                     f"{_fmt(m['cost_per_pp'], max(digits, 4))})")
        for nt in self.notes:
            lines.append(f"  note: {nt} ")
        return "\n".join(lines)

    def print(self, digits: int = 3, **kwargs):
        print(self.format(digits))
        return self
