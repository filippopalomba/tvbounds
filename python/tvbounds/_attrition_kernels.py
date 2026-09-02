"""Computational kernels for :func:`tvbounds_attrition` (port of R/attrition-kernels.R).

The functions ``Eq_q()``, ``one_group_grid()``, ``pooled_bounds_alloc()``,
``tv_bounds()``, ``lee_bounds()``, ``boot_tv_bounds()``, and
``boot_summary()`` are ported from the audited project scripts accompanying
Palomba (2026) ("Sensitivity Analysis in Population Shares"); their names are
kept unchanged to ease auditing, but none of them is public. The cluster
resampling in ``boot_tv_bounds()`` replicates the participant-level bootstrap
of the Christensen-Nino-Osman exercise; ``breakdown_delta()`` is ported from
the clinical-trials attrition exercise. The contamination kernel
``_tvb_cont_grid()`` implements the closed-form bounds under a contamination
neighborhood derived from the paper (Section SA5.3, Lemma "contamination as
constrained TV", and the attrition robustness set of the RCT-with-attrition
section); see the documentation of ``tvbounds_attrition()`` for the
statement.

Every floating-point operation mirrors the R code in the same order, using
the R-compatible primitives of ``tvbounds._rcompat`` (sequential sums, the
two-pass mean, type-7 quantiles, R's ``approx()``), so that the numbers agree
with the R package bit for bit.  The internal "slim" data frame of the R
package (columns ``Y``, ``S``, ``D`` and the stratum factor
``.tvb_stratum``) is represented by :class:`_Slim`, a bundle of NumPy arrays
whose stratum factor is stored as 0-based integer codes plus the fixed list
of R-style level labels (see :func:`_r_interaction`).
"""
from __future__ import annotations

import math
import os
import unicodedata
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._object import message
from ._rcompat import rapprox, rcumsum, rmean, rmean_int, rorder, rquantile7, rsd, rsum
from ._rrng import RRNG

# (the fused-multiply-add variants of sd() and approx() that R's compiled
# code uses on this platform live in tvbounds._rcompat: rsd(na_rm=True) and
# rapprox(ties="ordered"))
# ---------------------------------------------------------------------------
# R factor emulation: interaction(data[covariates], drop = TRUE)
# ---------------------------------------------------------------------------
# ICU root-collation order of the ASCII punctuation and symbols (R sorts
# character vectors with the locale's collation; on this platform R uses ICU
# in an en_US locale, where whitespace < punctuation/symbols < digits <
# letters, letters compare case-insensitively at the primary level and
# lowercase precedes uppercase at the tertiary level).
_ICU_PUNCT = "_-,;:!?.'\"()[]{}@*/\\&#%`^+<=>|~$"


def _icu_key(s: str):
    """Sort key approximating ICU root collation (exact for ASCII labels).

    Returns (primary, secondary, tertiary) weight tuples: primary weights
    order character classes (whitespace, punctuation, digits, letters) and
    compare letters case-insensitively (accents stripped); secondary weights
    carry the accents; tertiary weights put lowercase before uppercase.
    """
    prim = []
    sec = []
    ter = []
    for ch in s:
        if ch.isspace():
            prim.append((0, ord(ch)))
            sec.append(0)
            ter.append(0)
        elif ch.isdigit():
            prim.append((2, unicodedata.digit(ch, ord(ch))))
            sec.append(0)
            ter.append(0)
        elif ch.isalpha():
            dec = unicodedata.normalize("NFD", ch)
            base = dec[0].lower()
            prim.append((3, ord(base)))
            sec.append(tuple(ord(c) for c in dec[1:]))
            ter.append(1 if ch.isupper() else 0)
        else:
            k = _ICU_PUNCT.find(ch)
            prim.append((1, k if k >= 0 else 100 + ord(ch)))
            sec.append(0)
            ter.append(0)
    return (tuple(prim), tuple(sec), tuple(ter))


def _r_as_character_double(x: float) -> str:
    """R's ``as.character()`` of a double: 15 significant digits, R's rule
    for choosing fixed versus scientific notation, trailing zeros dropped."""
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == 0.0:
        return "0"
    neg = 1 if x < 0 else 0
    mant, expo = f"{abs(x):.14e}".split("e")
    kpower = int(expo)
    mant = mant.rstrip("0").rstrip(".")
    nsig = len(mant.replace(".", ""))
    # R's formatReal(): fixed width versus scientific width (scipen = 0)
    mxsl = neg + (kpower + 1 if kpower >= 0 else 1)
    rgt = max(0, nsig - kpower - 1)
    w_fixed = mxsl + rgt + (1 if rgt > 0 else 0)
    e_w = 2 if (kpower >= 100 or kpower <= -99) else 1
    w_sci = neg + (1 if nsig > 1 else 0) + (nsig - 1) + 4 + e_w
    if w_fixed <= w_sci:
        s = f"{x:.{rgt}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    return ("-" if neg else "") + mant + "e" + ("+" if kpower >= 0 else "-") + \
        f"{abs(kpower):02d}"


def _r_label_scalar(v) -> str:
    """R's ``as.character()`` of one element of a covariate column."""
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return _r_as_character_double(float(v))
    return str(v)


def _r_factor(col: pd.Series) -> Tuple[np.ndarray, List[str]]:
    """R's ``as.factor()`` of one covariate column: 0-based codes and the
    level labels (sorted unique values, labelled as R's ``as.character``)."""
    dt = col.dtype
    if isinstance(dt, pd.CategoricalDtype):
        levels = [_r_label_scalar(c) for c in col.cat.categories]
        codes = np.asarray(col.cat.codes, dtype=np.int64)
        return codes, levels
    if pd.api.types.is_bool_dtype(dt):
        vals = col.to_numpy().astype(bool)
        return vals.astype(np.int64), ["FALSE", "TRUE"]
    if pd.api.types.is_integer_dtype(dt):
        vals = col.to_numpy().astype(np.int64)
        u = np.unique(vals)
        return np.searchsorted(u, vals).astype(np.int64), [str(int(v)) for v in u]
    if pd.api.types.is_float_dtype(dt):
        vals = col.to_numpy(dtype=np.float64)
        u = np.unique(vals)
        # R's factor() takes the levels as unique(as.character(sorted values)):
        # distinct doubles sharing the same 15-significant-digit label (e.g.
        # 0.1 + 0.2 and 0.3) fall into ONE level.
        raw_labels = [_r_as_character_double(float(v)) for v in u]
        levels = list(dict.fromkeys(raw_labels))
        index = {s: i for i, s in enumerate(levels)}
        lab_code = np.array([index[lab] for lab in raw_labels], dtype=np.int64)
        return lab_code[np.searchsorted(u, vals)], levels
    vals = [_r_label_scalar(v) for v in col.tolist()]
    levels = sorted(set(vals), key=_icu_key)
    index = {s: i for i, s in enumerate(levels)}
    return np.array([index[v] for v in vals], dtype=np.int64), levels


def _r_interaction(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """R's ``interaction(df, drop = TRUE)``: 0-based codes and level labels.

    Labels are ``level1.level2....``; the first column varies fastest in
    the level order; combinations absent from the data are dropped.
    """
    facs = [_r_factor(df.iloc[:, j]) for j in range(df.shape[1])]
    n = len(df)
    comb = np.zeros(n, dtype=np.int64)
    mult = 1
    for codes, levels in facs:
        comb = comb + mult * codes
        mult *= max(len(levels), 1)
    present = np.unique(comb)
    new_codes = np.searchsorted(present, comb).astype(np.int64)
    labels = []
    for i in present:
        parts = []
        rem = int(i)
        for codes, levels in facs:
            k = max(len(levels), 1)
            parts.append(levels[rem % k])
            rem //= k
        labels.append(".".join(parts))
    return new_codes, labels


# ---------------------------------------------------------------------------
# the internal slim data frame
# ---------------------------------------------------------------------------
class _Slim:
    """The R package's slim frame ``data.frame(Y, S, D[, .tvb_stratum])``.

    ``Y`` is float64 (``NaN`` = missing outcome), ``S`` and ``D`` are int64
    0/1 vectors, ``st`` holds the 0-based stratum codes (or ``None``) and
    ``levels`` the fixed list of stratum labels in R's level order.
    """

    __slots__ = ("Y", "S", "D", "st", "levels")

    def __init__(self, Y, S, D, st=None, levels=None):
        self.Y = np.asarray(Y, dtype=np.float64)
        self.S = np.asarray(S, dtype=np.int64)
        self.D = np.asarray(D, dtype=np.int64)
        self.st = None if st is None else np.asarray(st, dtype=np.int64)
        self.levels = None if levels is None else list(levels)

    @property
    def n(self) -> int:
        return int(self.Y.size)

    def take(self, rows) -> "_Slim":
        """R's ``slim[rows, , drop = FALSE]`` (levels of the factor kept)."""
        return _Slim(self.Y[rows], self.S[rows], self.D[rows],
                     None if self.st is None else self.st[rows], self.levels)


def _as_slim(data, covs=None) -> _Slim:
    """Coerce a slim pandas frame (columns Y, S, D[, .tvb_stratum]) to :class:`_Slim`."""
    if isinstance(data, _Slim):
        return data
    Y = np.asarray(pd.Series(data["Y"]).astype("float64"), dtype=np.float64)
    S = np.asarray(data["S"]).astype(np.int64)
    D = np.asarray(data["D"]).astype(np.int64)
    st = levels = None
    if covs is not None:
        if ".tvb_stratum" in data.columns:
            st, levels = _r_factor(data[".tvb_stratum"])
        else:
            st, levels = _r_interaction(data[list(covs)])
    return _Slim(Y, S, D, st, levels)


def _rdiv(a: float, b: float) -> float:
    """R's ``a / b`` for doubles (``NaN``/``Inf`` instead of an exception)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(a) / np.float64(b))


def _rmax0(x: float) -> float:
    """R's ``max(x, 0)``: ``NaN`` propagates, otherwise the larger value."""
    if math.isnan(x):
        return math.nan
    return x if x >= 0 else 0.0


# ---------------------------------------------------------------------------
# group kernels
# ---------------------------------------------------------------------------
def Eq_q(Y_sort, q: float) -> float:
    """Atom-safe fractional lower partial mean (scalar audited reference).

    Computes ``E_Q[Y * 1{Y in bottom-q mass}]`` for the empirical
    distribution of the sorted vector ``Y_sort``, splitting the boundary
    observation proportionally so that point masses (atoms) are handled
    correctly. Kept as the scalar audited reference for
    :func:`one_group_grid`.
    """
    Y_sort = np.asarray(Y_sort, dtype=np.float64)
    n = Y_sort.size
    mu = rmean(Y_sort)
    if q <= 0:
        return 0.0
    if q >= 1:
        return mu
    k = math.floor(q * n)
    alpha = q * n - k
    s = rsum(Y_sort[:k]) if k > 0 else 0.0
    return (s + alpha * Y_sort[min(k + 1, n) - 1]) / n


def _Eqv_factory(Ys: np.ndarray, n: int, mu: float, cs: np.ndarray):
    """The vectorized partial-mean ``Eqv(qs)`` shared by the two group kernels."""
    def Eqv(qs):
        qs = np.asarray(qs, dtype=np.float64)
        k = np.floor(qs * n)
        alpha = qs * n - k
        ik = np.minimum(k, n).astype(np.int64)            # cs[pmin(k, n) + 1L]
        iy = np.minimum(k + 1, n).astype(np.int64) - 1    # Ys[pmin(k + 1L, n)]
        val = (cs[ik] + alpha * Ys[iy]) / n
        val[qs <= 0] = 0.0
        val[qs >= 1] = mu
        return val
    return Eqv


def one_group_grid(Ys, p_star: float, eps) -> dict:
    """Total variation bounds for one group over a whole budget grid.

    Vectorized version of the closed-form total variation bounds on the
    always-observed mean of one treated group (Palomba 2026, Theorem "TV
    sensitivity bounds for the average treatment effect on the
    always-observed"), given the SORTED observed outcomes. The sort and the
    cumulative sum are hoisted out of the grid loop: every partial sum
    ``sum(Ys[seq_len(k)])`` is read off ``cs = c(0, cumsum(Ys))`` in O(1),
    and the whole budget grid is evaluated in a few vectorized operations.
    ``cumsum`` and ``sum`` share R's sequential accumulator, so ``cs[k+1]``
    equals ``sum(Ys[seq_len(k)])`` bitwise and the results are identical to
    the scalar path (audited against :func:`Eq_q` in the source scripts).

    ``p_star = 0`` is admissible and is NOT a degenerate case: the group then
    has no compliers, every treated respondent is always-observed, and the
    closed form correctly collapses to the point-identified
    ``lo = up = mean(Ys)``.

    Returns a dict with ``lo`` and ``up``, the lower and upper bounds on the
    always-observed mean at each budget value.
    """
    Ys = np.asarray(Ys, dtype=np.float64)
    eps = np.atleast_1d(np.asarray(eps, dtype=np.float64))
    n = int(Ys.size)
    mu = rmean(Ys)
    cs = np.concatenate(([0.0], rcumsum(Ys)))
    Eqv = _Eqv_factory(Ys, n, mu, cs)
    Ea = Eqv((1 - p_star) * eps)
    Eb = Eqv(1 - p_star * eps)
    Ec = Eqv(p_star * eps)
    Ed = Eqv(1 - (1 - p_star) * eps)
    return {"lo": (1 / (1 - p_star)) * Ea + (Eb - Ea),
            "up": (Ed - Ec) + (1 / (1 - p_star)) * (mu - Ed)}


def _tvb_cont_grid(Ys, p_star: float, eps) -> dict:
    """Contamination bounds for one group over a whole budget grid.

    Closed-form bounds on the always-observed mean when the complier outcome
    distribution is restricted to the contamination neighborhood of the
    always-observed outcome distribution: ``P_C = (1 - delta) * P_AO +
    delta * R`` with ``R`` an arbitrary distribution. Combining this mixture
    restriction with the sample-splitting mixture ``P_T = p_star * P_C +
    (1 - p_star) * P_AO`` yields the density constraints (with respect to
    ``P_T``) ``(1 - delta) / (1 - delta * p_star) <= r <= 1 / p_star``, whose
    extremal solutions are trimmed means of ``P_T`` with trimming mass
    ``delta * p_star``:

      ``lower:  E_PT[ Y * 1{bottom (1 - delta * p_star) mass} ] / (1 - delta * p_star)``
      ``upper:  E_PT[ Y * 1{top    (1 - delta * p_star) mass} ] / (1 - delta * p_star)``

    i.e. Lee (2009)-type trimming with effective trimming share
    ``delta * p_star``. At ``delta = 0`` both collapse to ``mean(Ys)``; at
    ``delta = 1`` they equal the Lee bounds, matching the total variation
    endpoints. Uses the same atom-safe fractional partial mean as
    :func:`one_group_grid`, so atoms are handled exactly.
    """
    Ys = np.asarray(Ys, dtype=np.float64)
    eps = np.atleast_1d(np.asarray(eps, dtype=np.float64))
    n = int(Ys.size)
    mu = rmean(Ys)
    cs = np.concatenate(([0.0], rcumsum(Ys)))
    Eqv = _Eqv_factory(Ys, n, mu, cs)
    q = p_star * eps                       # trimmed mass; q < 1 since p_star < 1
    return {"lo": Eqv(1 - q) / (1 - q),
            "up": (mu - Eqv(q)) / (1 - q)}


# ---------------------------------------------------------------------------
# pooled (joint) covariate bounds
# ---------------------------------------------------------------------------
def pooled_bounds_alloc(Y_strata: Sequence, px, wts, eps_grid) -> dict:
    """Pooled (joint) covariate bounds via exact greedy budget allocation.

    One total variation budget allocated across the retained covariate cells
    rather than the same budget imposed inside every cell (Palomba 2026,
    eq. "rct covariate robustness set" / "rct covariate budget allocation").
    In the normalized cell budget ``a`` in ``[0, 1]`` (``a = 1`` exhausts the
    cell's complier mass, so the cell bound is its Lee bound), the pooled
    program reads

      ``upper:  max sum_s w_s * up_s(a_s)  s.t.  sum_s w_s * a_s <= delta``
      ``lower:  min sum_s w_s * lo_s(a_s)  s.t.  sum_s w_s * a_s <= delta``

    with ``w_s`` the control-respondent shares (the covariate distribution of
    the always-observed population). Each cell value function is piecewise
    linear in ``a`` with kinks where the trimming quantiles cross the cell's
    order-statistic masses ``i/n_s``, and it is concave (upper) / convex
    (lower), so the allocation solves exactly by a greedy fill: list every
    linear segment of every cell, sort by marginal value per unit of pooled
    budget (the segment slope), and consume the budget in that order. One
    pass yields the whole curve ``delta -> pooled bound``; evaluating the
    cumulative fill at the grid is exact because the fill curve is itself
    piecewise linear in the budget.

    Returns a dict with ``up`` and ``lo``, the pooled bounds on the
    aggregated always-observed treated mean at each budget value.
    """
    eps_grid = np.atleast_1d(np.asarray(eps_grid, dtype=np.float64))
    S = len(Y_strata)
    base_up = 0.0
    base_lo = 0.0
    seg_up = []
    seg_lo = []

    def keep01(x):
        return x[np.isfinite(x) & (x > 0) & (x < 1)]

    # Merge kinks closer than 1e-9: two nearly coincident breakpoints span a
    # segment of negligible budget and gain, but their slope Delta_v / Delta_a
    # is numerical noise that would be sorted into the wrong fill position.
    # The gap filter keeps the first of a close pair, so re-pin the endpoint.
    def dedupe(bp):
        keep = np.concatenate(([True], np.diff(bp) > 1e-9))
        bp = bp[keep]
        bp[-1] = 1.0
        return bp

    for j in range(S):
        Ys = np.sort(np.asarray(Y_strata[j], dtype=np.float64))
        n = int(Ys.size)
        p = float(px[j])
        w = float(wts[j])
        ci = np.arange(1, n, dtype=np.float64) / n
        # Kinks of up_s: its arguments p*a and 1-(1-p)*a cross the masses i/n.
        # Kinks of lo_s: its arguments (1-p)*a and 1-p*a cross the masses i/n.
        # p = 0 is admissible (no compliers): both value functions are flat and
        # the divisions by p are filtered out with the out-of-range kinks.
        with np.errstate(divide="ignore", invalid="ignore"):
            bp_up = dedupe(np.unique(np.concatenate((
                [0.0, 1.0], keep01(ci / p), keep01((1 - ci) / (1 - p))))))
            bp_lo = dedupe(np.unique(np.concatenate((
                [0.0, 1.0], keep01(ci / (1 - p)), keep01((1 - ci) / p)))))
        vu = one_group_grid(Ys, p, bp_up)["up"]
        vl = one_group_grid(Ys, p, bp_lo)["lo"]
        base_up = base_up + w * vu[0]
        base_lo = base_lo + w * vl[0]
        seg_up.append((np.diff(vu) / np.diff(bp_up),
                       w * np.diff(bp_up), w * np.diff(vu)))
        seg_lo.append((-np.diff(vl) / np.diff(bp_lo),
                       w * np.diff(bp_lo), -w * np.diff(vl)))

    # Greedy fill: cumulative (budget, value-gain) knots in decreasing-slope
    # order; the pooled curve is their linear interpolant. Filling every
    # segment (total cost sum_s w_s = 1) reaches the covariate Lee endpoint
    # exactly.
    def fill(segs):
        slope = np.concatenate([s[0] for s in segs])
        cost = np.concatenate([s[1] for s in segs])
        gain = np.concatenate([s[2] for s in segs])
        o = rorder(slope, decreasing=True)
        kb = np.concatenate(([0.0], rcumsum(cost[o])))
        kv = np.concatenate(([0.0], rcumsum(gain[o])))
        kb_max = float(np.max(kb))

        def evaluate(d):
            return rapprox(kb, kv, np.minimum(d, kb_max), ties="ordered")
        return evaluate

    gain_up = fill(seg_up)
    gain_lo = fill(seg_lo)
    return {"up": base_up + gain_up(eps_grid),
            "lo": base_lo - gain_lo(eps_grid)}


# ---------------------------------------------------------------------------
# bounds over a budget grid
# ---------------------------------------------------------------------------
def _na_frame(eps_grid: np.ndarray) -> dict:
    nan = np.full(eps_grid.size, np.nan)
    return {"delta": eps_grid.copy(), "tau_lower": nan, "tau_upper": nan.copy()}


def _tv_bounds_core(data: _Slim, delta, covs=None, neighborhood: str = "tv",
                    min_obs: int = 5, verbose: bool = False):
    """The computation behind :func:`tv_bounds`, returning ``(columns, info)``
    where ``columns`` is a dict of equal-length arrays (``delta``,
    ``tau_lower``, ``tau_upper``[, ``tau_lower_pw``, ``tau_upper_pw``]) and
    ``info`` the detail dict (``None`` on the all-NA path)."""
    eps_grid = np.atleast_1d(np.asarray(delta, dtype=np.float64))
    grid_fun = _tvb_cont_grid if neighborhood == "contamination" else one_group_grid

    if covs is None:
        # ---- Case 1 (no covariates) ----------------------------------------
        Y = data.Y
        S = data.S
        D = data.D
        okY = ~np.isnan(Y)
        Yt = Y[(D == 1) & (S == 1) & okY]
        Yc = Y[(D == 0) & (S == 1) & okY]
        mu0 = rmean(Yc)
        ps = _rdiv(rmean_int(S[D == 1]) - rmean_int(S[D == 0]),
                   rmean_int(S[D == 1]))
        # Monotonicity implies p1 >= p0, so a negative estimate is sampling
        # noise: project onto the admissible region (as in the covariate
        # branch) rather than silently NA-ing the whole replicate.
        ps = _rmax0(ps)
        # Too few observed outcomes or an inadmissible complier share yield the
        # all-NA path; a NaN complier share (empty control arm in a resample)
        # errors here (R's `if (NA)`), so failed replicates are still caught
        # by the error handler in boot_tv_bounds.
        if Yt.size < 2:
            return _na_frame(eps_grid), None
        if math.isnan(ps):
            raise RuntimeError("missing value where TRUE/FALSE needed")
        if ps < 0 or ps >= 1:
            return _na_frame(eps_grid), None
        g = grid_fun(np.sort(Yt), ps, eps_grid)
        out = {"delta": eps_grid.copy(), "tau_lower": g["lo"] - mu0,
               "tau_upper": g["up"] - mu0}
        info = {"p_star": ps, "mu0": mu0,
                "n_treated_obs": int(Yt.size), "n_control_obs": int(Yc.size)}
        return out, info

    # ---- Case 3 (stratum-by-stratum) ----------------------------------------
    # The bootstrap driver precomputes the stratum factor once on the original
    # data and resamples it with the rows; direct calls on raw data build it
    # here (see _as_slim).
    st = data.st
    levels = data.levels
    # Bookkeeping on atomic vectors: the masks below select the same elements
    # in the same order as the R code, so every count and mean is unchanged.
    yv = data.Y
    sv = data.S
    dv = data.D
    okY = ~np.isnan(yv)

    sc = np.bincount(st[(dv == 0) & (sv == 1)], minlength=len(levels))
    sl = [i for i in range(len(levels)) if sc[i] > 0]
    usable = []
    px = []
    nt_kept = []
    nc_kept = []
    dropped_small = []
    dropped_pstar = []
    for li in sl:
        s = levels[li]
        idx = st == li
        nt = int(np.sum(idx & (dv == 1) & (sv == 1) & okY))
        nc = int(np.sum(idx & (dv == 0) & (sv == 1) & okY))
        if nt < min_obs or nc < min_obs:
            dropped_small.append(s)
            continue
        p1 = rmean_int(sv[idx & (dv == 1)])
        p0 = rmean_int(sv[idx & (dv == 0)])
        ps = _rdiv(p1 - p0, p1)
        if math.isnan(ps) or ps >= 1:
            dropped_pstar.append(s)
            continue
        # Monotonicity implies p1 >= p0, so a negative estimate is sampling
        # noise: project onto the admissible region rather than discarding
        # the stratum.
        ps = _rmax0(ps)
        usable.append(li)
        px.append(ps)
        nt_kept.append(nt)
        nc_kept.append(nc)
    # A resample can in principle retain no cell at all; return NA rather than
    # silently aggregating over an empty set (which would yield tau = 0).
    if len(usable) == 0:
        return _na_frame(eps_grid), None

    usable_labels = [levels[li] for li in usable]
    uc = sc[usable].astype(np.float64)
    wts = uc / rsum(uc)
    px = np.asarray(px, dtype=np.float64)

    # mu0_AT must be aggregated over the SAME strata and with the SAME weights
    # as the bounds below: both terms of tau estimate the always-observed in
    # the retained cells. Averaging Y over all control respondents instead
    # would mix two different populations whenever some strata are dropped.
    mu0_x = np.array([rmean(yv[(st == li) & (dv == 0) & (sv == 1) & okY])
                      for li in usable], dtype=np.float64)
    mu0_AT = rsum(wts * mu0_x)

    # Coverage of the retained cells, as a share of the control-respondent mass.
    attr_cov = rsum(uc) / rsum(sc.astype(np.float64))
    if verbose:
        message("strata kept: %d/%d | coverage: %.1f%% | mu0_AT: %.4f"
                % (len(usable), len(sl), 100 * attr_cov, mu0_AT))

    Y_strata = [yv[(st == li) & (dv == 1) & (sv == 1) & okY] for li in usable]

    # One vectorized sweep per stratum: each stratum's outcomes are sorted once
    # and the whole bounds path comes from the group kernel. The weighted
    # aggregation accumulates over strata elementwise in the grid.
    lo = np.zeros(eps_grid.size)
    up = np.zeros(eps_grid.size)
    for j in range(len(usable)):
        Ys = Y_strata[j]
        # Every retained stratum must return a bound: skipping one here without
        # renormalizing wts would silently drop weight and bias the aggregate.
        if Ys.size < 2 or px[j] < 0 or px[j] >= 1:
            raise RuntimeError("group kernel returned NA for retained stratum "
                               + usable_labels[j])
        g = grid_fun(np.sort(Ys), float(px[j]), eps_grid)
        w_j = float(wts[j])
        lo = lo + w_j * g["lo"]
        up = up + w_j * g["up"]

    info = {
        "strata": pd.DataFrame({
            "stratum": usable_labels,
            "weight": wts,
            "p_star": px,
            "n_treated_obs": np.asarray(nt_kept, dtype=np.int64),
            "n_control_obs": np.asarray(nc_kept, dtype=np.int64),
            "mu0": mu0_x}),
        "coverage": attr_cov, "mu0_AT": mu0_AT,
        "n_strata_total": len(sl),
        "dropped_small": dropped_small, "dropped_pstar": dropped_pstar}

    if neighborhood == "contamination":
        # Within-stratum common-budget aggregation of the cell-level
        # contamination bounds; the pooled allocation below is total variation
        # only (its budget-transfer arithmetic is specific to total variation).
        out = {"delta": eps_grid.copy(), "tau_lower": lo - mu0_AT,
               "tau_upper": up - mu0_AT}
        return out, info

    # ---- Pooled (joint) bounds: the default covariate bounds ----------------
    pooled = pooled_bounds_alloc(Y_strata, px, wts, eps_grid)

    # The pooled program nests the common-budget (within-stratum) allocation
    # and coincides with it at delta = 0 (the MCAR point) and delta = 1 (the
    # covariate Lee bounds); a violation beyond floating-point noise means the
    # greedy fill went wrong, so fail the call (a bootstrap replicate then
    # counts as failed and the n_fail warning surfaces it) rather than
    # returning a bad curve. The endpoint identities are checked at whichever
    # grid points equal 0 or 1.
    tol = 1e-7 * (1 + float(np.max(np.abs(np.concatenate((lo, up))))))
    i0 = eps_grid == 0
    i1 = eps_grid == 1
    if (np.any(pooled["up"] < up - tol) or np.any(pooled["lo"] > lo + tol) or
            np.any(np.abs(pooled["up"][i0] - up[i0]) > tol) or
            np.any(np.abs(pooled["lo"][i0] - lo[i0]) > tol) or
            np.any(np.abs(pooled["up"][i1] - up[i1]) > tol) or
            np.any(np.abs(pooled["lo"][i1] - lo[i1]) > tol)):
        raise RuntimeError("pooled allocation violates its envelope/endpoint identities")

    out = {"delta": eps_grid.copy(),
           "tau_lower": pooled["lo"] - mu0_AT,
           "tau_upper": pooled["up"] - mu0_AT,
           "tau_lower_pw": lo - mu0_AT,
           "tau_upper_pw": up - mu0_AT}
    return out, info


def tv_bounds(data, delta, covs=None, neighborhood: str = "tv",
              min_obs: int = 5, verbose: bool = False):
    """Sensitivity bounds on the treatment effect over a budget grid.

    Computes the sensitivity bounds on the average treatment effect for the
    always-observed subpopulation over the budget grid ``delta``, without
    covariates (closed form) or with covariates (stratum-by-stratum). With
    covariates and ``neighborhood = "tv"``, the default columns
    ``tau_lower`` / ``tau_upper`` are the pooled (joint) bounds of
    :func:`pooled_bounds_alloc`, and the within-stratum common-budget bounds
    ride along as ``tau_lower_pw`` / ``tau_upper_pw``. With
    ``neighborhood = "contamination"``, the bounds are the within-stratum
    common-budget aggregation of the cell-level contamination bounds of
    :func:`_tvb_cont_grid`.

    ``data`` is a :class:`_Slim` (or a pandas frame with columns ``Y``,
    ``S``, ``D`` and optionally ``.tvb_stratum``; without the latter the
    covariate branch builds ``interaction(data[covs])``). Degenerate inputs
    (too few observed outcomes, inadmissible complier share, no retained
    cell) return an all-``NaN`` data frame rather than raising, so that
    failed bootstrap replicates are counted rather than fatal; a ``NaN``
    complier share (empty control arm in a resample) raises and is caught by
    :func:`boot_tv_bounds`.

    Returns ``(frame, info)``: a pandas DataFrame with columns ``delta``,
    ``tau_lower``, ``tau_upper`` (and ``tau_lower_pw``, ``tau_upper_pw`` in
    the covariate total variation case) and the ``info`` dict carrying
    baseline and stratum detail (R's ``attr(out, "info")``; ``None`` on the
    all-``NaN`` path).
    """
    slim = _as_slim(data, covs)
    cols, info = _tv_bounds_core(slim, delta, covs, neighborhood, min_obs, verbose)
    return pd.DataFrame(cols), info


def lee_bounds(data, covs=None, min_obs: int = 5) -> dict:
    """Lee (2009) bounds as the budget-one endpoint.

    Lee bounds equal the total variation bounds at budget one; computed with
    the same atom-safe formula for consistency. Returns a dict with
    ``lower`` and ``upper``.
    """
    b, _ = tv_bounds(data, delta=[1.0], covs=covs, neighborhood="tv",
                     min_obs=min_obs, verbose=False)
    return {"lower": float(b["tau_lower"].iloc[0]),
            "upper": float(b["tau_upper"].iloc[0])}


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def boot_tv_bounds(data, delta, covs=None, neighborhood: str = "tv",
                   B: int = 1000, min_obs: int = 5, cluster=None,
                   rng: Optional[RRNG] = None) -> dict:
    """Nonparametric bootstrap of the whole bounds curve.

    The bound carries no structural parameter and no estimated moment, so the
    Fang-Santos bootstrap of the paper reduces to the standard nonparametric
    bootstrap of the empirical process. Crucially the FULL sample
    ``(Y, S, D, X)`` is resampled jointly rather than only the treated
    respondents that form the baseline: every replicate then redraws the
    arm-specific response rates, hence the complier share ``p_star`` (and
    each cell-level share), so the bootstrap automatically carries the
    estimation uncertainty in ``p_star`` that a resample of the baseline
    alone would hold fixed. When ``cluster`` is supplied, whole clusters are
    resampled with replacement and all their rows are kept, replicating
    cluster-level randomization designs.

    Draws come from the R-compatible stream ``rng`` (an :class:`RRNG`); the
    user-facing :func:`tvbounds_attrition` creates it from ``seed``
    immediately before calling this function, as R's
    ``withr::local_seed(seed)`` does.

    Returns a dict with ``delta``, ``lo`` and ``up`` (``B x len(delta)``
    arrays of bound draws), ``n_fail``, and ``B``.
    """
    slim = _as_slim(data, covs)
    n = slim.n
    delta = np.atleast_1d(np.asarray(delta, dtype=np.float64))
    if rng is None:
        rng = RRNG(int.from_bytes(os.urandom(4), "little"))
    # Each replicate copies the resampled frame, so carry only the columns
    # tv_bounds reads. The stratum factor is a row-wise function of the
    # covariates, so it is built once and resampled with the rows; levels
    # absent from a resample are dropped by the sc > 0 filter either way.
    if covs is None:
        slim = _Slim(slim.Y, slim.S, slim.D)
    if cluster is not None:
        codes, ids = pd.factorize(np.asarray(cluster))   # unique(cluster), match()
        codes = np.asarray(codes, dtype=np.int64)
        order = np.argsort(codes, kind="stable")
        counts = np.bincount(codes, minlength=len(ids))
        cidx = np.split(order, np.cumsum(counts)[:-1])
        n_ids = len(ids)
    lo_mat = np.full((B, delta.size), np.nan)
    up_mat = np.full((B, delta.size), np.nan)
    n_fail = 0
    for b in range(B):
        if cluster is None:
            rows = rng.sample_int_replace(n, n) - 1
        else:
            draw = rng.sample_int_replace(n_ids, n_ids)
            rows = np.concatenate([cidx[d - 1] for d in draw])
        db = slim.take(rows)
        try:
            bb, _ = _tv_bounds_core(db, delta, covs, neighborhood, min_obs,
                                    verbose=False)
        except Exception:
            n_fail += 1
            continue
        lo_mat[b, :] = bb["tau_lower"]
        up_mat[b, :] = bb["tau_upper"]
    return {"delta": delta, "lo": lo_mat, "up": up_mat, "n_fail": n_fail, "B": B}


def boot_summary(bt: dict, level: float = 0.95) -> pd.DataFrame:
    """Bootstrap summaries: standard errors and percentile band at each budget.

    The width is bootstrapped directly rather than combining the two bounds'
    standard errors: the lower and upper bounds are computed from the same
    resample and are strongly correlated, so ``sd(up)`` and ``sd(lo)`` do
    not determine ``sd(up - lo)``.

    Returns a DataFrame with per-budget standard errors, percentile interval
    endpoints for each bound, the width standard error, and the number of
    successful replicates.
    """
    alpha = (1 - level) / 2
    lo = np.asarray(bt["lo"], dtype=np.float64)
    up = np.asarray(bt["up"], dtype=np.float64)

    def q(M, p):
        return np.array([rquantile7(M[:, j], p, na_rm=True)
                         for j in range(M.shape[1])], dtype=np.float64)

    def sd_cols(M):
        return np.array([rsd(M[:, j], na_rm=True) for j in range(M.shape[1])],
                        dtype=np.float64)

    wid = up - lo
    return pd.DataFrame({
        "delta": np.asarray(bt["delta"], dtype=np.float64),
        "lo_est_se": sd_cols(lo),
        "up_est_se": sd_cols(up),
        "w_est_se": sd_cols(wid),
        "lo_ci_low": q(lo, alpha), "lo_ci_high": q(lo, 1 - alpha),
        "up_ci_low": q(up, alpha), "up_ci_high": q(up, 1 - alpha),
        "n_eff": np.array([int(np.sum(~np.isnan(lo[:, j])))
                           for j in range(lo.shape[1])], dtype=np.int64)})


# ---------------------------------------------------------------------------
# breakdown budget
# ---------------------------------------------------------------------------
def breakdown_delta(dg, lo, up) -> float:
    """Breakdown budget: first budget at which zero enters the bounds.

    Smallest grid budget at which ``0`` lies inside ``[lo, up]``, linearly
    interpolated between grid points; ``0`` if the interval already covers
    zero at the first grid point, ``NaN`` if it never does on the grid
    (censored breakdown).
    """
    dg = np.atleast_1d(np.asarray(dg, dtype=np.float64))
    lo = np.atleast_1d(np.asarray(lo, dtype=np.float64))
    up = np.atleast_1d(np.asarray(up, dtype=np.float64))
    if np.all(np.isnan(lo)) or np.all(np.isnan(up)):
        return math.nan
    inside = (lo <= 0) & (up >= 0)
    if bool(inside[0]):
        return 0.0
    k = np.flatnonzero(inside)
    if k.size == 0:
        return math.nan
    i = int(k[0])
    d0 = float(dg[i - 1])
    d1 = float(dg[i])
    lo_prev = float(lo[i - 1])
    up_prev = float(up[i - 1])
    if math.isnan(lo_prev):
        raise RuntimeError("missing value where TRUE/FALSE needed")
    if lo_prev > 0:
        return d0 + (lo_prev / (lo_prev - float(lo[i]))) * (d1 - d0)
    if math.isnan(up_prev):
        raise RuntimeError("missing value where TRUE/FALSE needed")
    if up_prev < 0:
        return d0 + ((-up_prev) / (float(up[i]) - up_prev)) * (d1 - d0)
    return d1
