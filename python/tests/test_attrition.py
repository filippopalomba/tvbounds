"""Tests for tvbounds_attrition(), mirroring tests/testthat/test-attrition.R.

All data are simulated in-memory; no files are written.
"""
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    import tvbounds  # noqa: F401
except Exception:
    # Another module of the package may be mid-edit while the package is being
    # assembled; register the namespace so that the attrition modules import.
    import types
    for _k in [k for k in sys.modules if k == "tvbounds" or k.startswith("tvbounds.")]:
        del sys.modules[_k]
    _pkg = types.ModuleType("tvbounds")
    _pkg.__path__ = [os.path.join(ROOT, "tvbounds")]
    sys.modules["tvbounds"] = _pkg

from tvbounds._attrition_kernels import (Eq_q, _r_interaction, breakdown_delta,  # noqa: E402
                                         one_group_grid)
from tvbounds._object import TVBounds  # noqa: E402
from tvbounds.attrition import tvbounds_attrition  # noqa: E402


# Hand-checkable design: 20 treated units, all respond, outcomes 1..20;
# 20 control units, 10 respond, outcomes all zero. Then p1 = 1, p0 = 0.5,
# p_star = 0.5, mu0 = 0, and (atom-safe fractional trimming):
#   Lee lower  = mean(1:10) = 5.5,  Lee upper  = mean(11:20) = 15.5
#   TV  at 0.5 = [6.75, 14.25],  contamination at 0.5 = [8, 13].
def make_hand_data():
    return pd.DataFrame({
        "y": [float(v) for v in range(1, 21)] + [0.0] * 10 + [math.nan] * 10,
        "d": [1] * 20 + [0] * 20,
        "s": [1] * 20 + [1] * 10 + [0] * 10})


def make_sim_data(n=300, seed=42, effect=0.4):
    rng = np.random.default_rng(seed)
    d = rng.binomial(1, 0.5, n)
    s = rng.binomial(1, np.where(d == 1, 0.9, 0.7))
    y = np.where(s == 1, rng.normal(loc=effect * d), math.nan)
    x = rng.binomial(1, 0.5, n)
    g = rng.integers(1, 51, n)
    return pd.DataFrame({"y": y, "d": d, "s": s, "x": x, "g": g})


def expect_equal(a, b, tolerance=1.5e-8):
    """testthat's expect_equal (all.equal's mean relative difference)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    assert a.shape == b.shape
    xy = float(np.mean(np.abs(a - b)))
    xn = float(np.mean(np.abs(b)))
    if math.isfinite(xn) and xn > tolerance:
        xy = xy / xn
    assert xy < tolerance, f"mean relative difference {xy} >= {tolerance}"


def test_delta0_collapses_to_baseline_difference_in_means():
    dat = make_sim_data()
    fit = tvbounds_attrition(dat, "y", "d", "s", delta=[0, 0.5, 1], bootstrap=False)
    dim_val = np.nanmean(dat.y[(dat.d == 1) & (dat.s == 1)]) - \
        np.nanmean(dat.y[(dat.d == 0) & (dat.s == 1)])
    expect_equal(fit.bounds["lower"].iloc[0], dim_val, tolerance=1e-12)
    expect_equal(fit.bounds["upper"].iloc[0], dim_val, tolerance=1e-12)
    expect_equal(fit.point, dim_val, tolerance=1e-12)


def test_delta1_reproduces_hand_computed_lee_bounds():
    dat = make_hand_data()
    for nb in ("tv", "contamination"):
        fit = tvbounds_attrition(dat, "y", "d", "s", delta=[0, 1],
                                 neighborhood=nb, bootstrap=False)
        expect_equal(fit.details["p_star"], 0.5, tolerance=1e-12)
        i1 = np.flatnonzero(fit.bounds["delta"].to_numpy() == 1)
        expect_equal(fit.bounds["lower"].to_numpy()[i1], 5.5, tolerance=1e-10)
        expect_equal(fit.bounds["upper"].to_numpy()[i1], 15.5, tolerance=1e-10)
        expect_equal(fit.details["lee"]["lower"], 5.5, tolerance=1e-10)
        expect_equal(fit.details["lee"]["upper"], 15.5, tolerance=1e-10)


def test_interior_budgets_match_hand_computed_closed_forms():
    dat = make_hand_data()
    fit_tv = tvbounds_attrition(dat, "y", "d", "s", delta=0.5, bootstrap=False)
    expect_equal(fit_tv.bounds["lower"], 6.75, tolerance=1e-10)
    expect_equal(fit_tv.bounds["upper"], 14.25, tolerance=1e-10)
    fit_c = tvbounds_attrition(dat, "y", "d", "s", delta=0.5,
                               neighborhood="contamination", bootstrap=False)
    expect_equal(fit_c.bounds["lower"], 8, tolerance=1e-10)
    expect_equal(fit_c.bounds["upper"], 13, tolerance=1e-10)


def test_bounds_are_monotone_in_the_budget():
    dat = make_sim_data(seed=7)
    grid = np.linspace(0, 1, 21)
    for nb in ("tv", "contamination"):
        fit = tvbounds_attrition(dat, "y", "d", "s", delta=grid,
                                 neighborhood=nb, bootstrap=False)
        assert np.all(np.diff(fit.bounds["lower"]) <= 1e-10)
        assert np.all(np.diff(fit.bounds["upper"]) >= -1e-10)
        fitx = tvbounds_attrition(dat, "y", "d", "s", covariates="x",
                                  delta=grid, neighborhood=nb, bootstrap=False)
        assert np.all(np.diff(fitx.bounds["lower"]) <= 1e-10)
        assert np.all(np.diff(fitx.bounds["upper"]) >= -1e-10)


def test_contamination_bounds_lie_weakly_inside_total_variation_bounds():
    grid = np.linspace(0, 1, 11)
    for sd_ in (1, 11):
        dat = make_sim_data(seed=sd_)
        tv = tvbounds_attrition(dat, "y", "d", "s", delta=grid, bootstrap=False)
        cc = tvbounds_attrition(dat, "y", "d", "s", delta=grid,
                                neighborhood="contamination", bootstrap=False)
        assert np.all(cc.bounds["lower"].to_numpy() >= tv.bounds["lower"].to_numpy() - 1e-10)
        assert np.all(cc.bounds["upper"].to_numpy() <= tv.bounds["upper"].to_numpy() + 1e-10)
        # with covariates: cell-level contamination inside the pooled TV default
        tvx = tvbounds_attrition(dat, "y", "d", "s", covariates="x",
                                 delta=grid, bootstrap=False)
        ccx = tvbounds_attrition(dat, "y", "d", "s", covariates="x",
                                 delta=grid, neighborhood="contamination",
                                 bootstrap=False)
        assert np.all(ccx.bounds["lower"].to_numpy() >= tvx.bounds["lower"].to_numpy() - 1e-10)
        assert np.all(ccx.bounds["upper"].to_numpy() <= tvx.bounds["upper"].to_numpy() + 1e-10)


def test_no_differential_attrition_collapses_bounds_to_the_baseline():
    rng = np.random.default_rng(3)
    n = 200
    d = rng.binomial(1, 0.5, n)
    s = np.ones(n, dtype=np.int64)          # everyone responds: p_star = 0
    y = rng.normal(loc=0.2 * d)
    dat = pd.DataFrame({"y": y, "d": d, "s": s})
    fit = tvbounds_attrition(dat, "y", "d", "s", delta=[0, 0.5, 1], bootstrap=False)
    assert fit.details["p_star"] == 0
    assert np.all(np.abs(fit.bounds["lower"].to_numpy() - fit.point) < 1e-12)
    assert np.all(np.abs(fit.bounds["upper"].to_numpy() - fit.point) < 1e-12)


def test_covariate_pooled_bounds_nest_the_within_stratum_reference():
    dat = make_sim_data(seed=5, n=500)
    grid = np.linspace(0, 1, 11)
    fit = tvbounds_attrition(dat, "y", "d", "s", covariates="x",
                             delta=grid, bootstrap=False)
    pw = fit.details["pooled"]["pw"]
    assert isinstance(pw, pd.DataFrame)
    assert np.all(fit.bounds["upper"].to_numpy() >= pw["upper"].to_numpy() - 1e-8)
    assert np.all(fit.bounds["lower"].to_numpy() <= pw["lower"].to_numpy() + 1e-8)
    # endpoints coincide (delta = 0 baseline and delta = 1 covariate Lee)
    expect_equal(fit.bounds["upper"].iloc[0], pw["upper"].iloc[0], tolerance=1e-8)
    i1 = grid == 1
    expect_equal(fit.bounds["lower"].to_numpy()[i1], pw["lower"].to_numpy()[i1],
                 tolerance=1e-8)
    # per-stratum detail present with weights summing to one
    st = fit.details["pooled"]["strata"]
    assert isinstance(st, pd.DataFrame)
    expect_equal(st["weight"].sum(), 1, tolerance=1e-12)


def test_small_covariate_cells_are_dropped_with_a_warning():
    dat = make_sim_data(seed=9, n=200)
    dat["rare"] = 0
    # a cell with control respondents but < min_obs treated observed outcomes
    idx = np.flatnonzero((dat.d == 0) & (dat.s == 1))[:3]
    dat.loc[dat.index[idx], "rare"] = 1
    with pytest.warns(UserWarning, match="dropped"):
        tvbounds_attrition(dat, "y", "d", "s", covariates="rare",
                           delta=[0, 0.5, 1], bootstrap=False)


def test_bootstrap_is_reproducible_under_seed_and_restores_the_rng():
    dat = make_sim_data(seed=21, n=200)
    grid = [0, 0.25, 0.5, 0.75, 1]
    f1 = tvbounds_attrition(dat, "y", "d", "s", delta=grid, B=30, seed=99)
    f2 = tvbounds_attrition(dat, "y", "d", "s", delta=grid, B=30, seed=99)
    pd.testing.assert_frame_equal(f1.bounds, f2.bounds, check_exact=True)
    f3 = tvbounds_attrition(dat, "y", "d", "s", delta=grid, B=30, seed=100)
    assert not np.array_equal(f1.bounds["lower_se"].to_numpy(),
                              f3.bounds["lower_se"].to_numpy())
    # global RNG states are untouched: draws after the call match draws without it
    np.random.seed(123)
    random.seed(123)
    before = (np.random.rand(3), [random.random() for _ in range(3)])
    np.random.seed(123)
    random.seed(123)
    tvbounds_attrition(dat, "y", "d", "s", delta=grid, B=10, seed=7)
    after = (np.random.rand(3), [random.random() for _ in range(3)])
    assert np.array_equal(before[0], after[0]) and before[1] == after[1]


def test_bootstrap_output_has_the_documented_schema():
    dat = make_sim_data(seed=13, n=200)
    grid = [0, 0.5, 1]
    fit = tvbounds_attrition(dat, "y", "d", "s", delta=grid, B=30, seed=1, level=0.90)
    assert all(c in fit.bounds.columns for c in
               ("delta", "lower", "upper", "lower_se", "upper_se", "ci_lower", "ci_upper"))
    assert fit.level == 0.90
    assert fit.B == 30
    # the outer band contains the point bounds
    assert np.all(fit.bounds["ci_lower"].to_numpy() <= fit.bounds["lower"].to_numpy() + 1e-10)
    assert np.all(fit.bounds["ci_upper"].to_numpy() >= fit.bounds["upper"].to_numpy() - 1e-10)
    assert isinstance(fit.details["boot"]["draws"]["lower"], np.ndarray)
    assert fit.details["boot"]["draws"]["lower"].shape == (30, len(grid))


def test_cluster_bootstrap_runs_and_is_reproducible():
    dat = make_sim_data(seed=17, n=200)
    grid = [0, 0.5, 1]
    f1 = tvbounds_attrition(dat, "y", "d", "s", delta=grid, B=20, cluster="g", seed=5)
    f2 = tvbounds_attrition(dat, "y", "d", "s", delta=grid, B=20, cluster="g", seed=5)
    pd.testing.assert_frame_equal(f1.bounds, f2.bounds, check_exact=True)
    assert f1.details["n_clusters"] == len(np.unique(dat.g))
    # same seed, unclustered resampling gives a different bootstrap
    f3 = tvbounds_attrition(dat, "y", "d", "s", delta=grid, B=20, seed=5)
    assert not np.array_equal(f1.bounds["lower_se"].to_numpy(),
                              f3.bounds["lower_se"].to_numpy())


def test_endpoints_are_computed_even_when_absent_from_the_budget_grid():
    dat = make_sim_data(seed=23)
    fit = tvbounds_attrition(dat, "y", "d", "s", delta=[0.3, 0.6], bootstrap=False)
    assert fit.bounds.shape[0] == 2
    assert math.isfinite(fit.point)
    assert math.isfinite(fit.details["lee"]["lower"])
    assert math.isfinite(fit.details["lee"]["upper"])


def test_returned_object_satisfies_the_shared_contract():
    dat = make_sim_data(seed=29)
    fit = tvbounds_attrition(dat, "y", "d", "s", delta=[0, 0.5, 1], bootstrap=False)
    assert isinstance(fit, TVBounds)
    assert fit.application == "attrition"
    assert fit.neighborhood == "tv"
    assert fit.estimand_label == "treatment effect"
    assert fit.n == len(dat)
    assert math.isnan(fit.level) and math.isnan(fit.B)
    assert np.all(fit.bounds["upper"].to_numpy() - fit.bounds["lower"].to_numpy() >= -1e-10)


def test_input_validation_fails_early_with_informative_errors():
    dat = make_sim_data(seed=31)
    with pytest.raises(ValueError, match="outcome"):
        tvbounds_attrition(dat, "nope", "d", "s")
    with pytest.raises(ValueError, match="cluster"):
        tvbounds_attrition(dat, "y", "d", "s", cluster="nope")
    with pytest.raises(ValueError, match="covariates"):
        tvbounds_attrition(dat, "y", "d", "s", covariates="nope")
    bad = dat.copy()
    bad.loc[0, "d"] = 2
    with pytest.raises(ValueError, match="treatment"):
        tvbounds_attrition(bad, "y", "d", "s")
    with pytest.raises(ValueError, match="delta"):
        tvbounds_attrition(dat, "y", "d", "s", delta=[-0.1, 0.5])
    with pytest.raises(ValueError, match="delta"):
        tvbounds_attrition(dat, "y", "d", "s", delta=1.5)
    with pytest.raises(ValueError, match="level"):
        tvbounds_attrition(dat, "y", "d", "s", level=1.2)
    with pytest.raises(ValueError, match="B"):
        tvbounds_attrition(dat, "y", "d", "s", B=1)
    with pytest.raises(ValueError, match="min_obs"):
        tvbounds_attrition(dat, "y", "d", "s", min_obs=1)
    badx = dat.copy()
    badx["x"] = badx["x"].astype("float64")
    badx.loc[4, "x"] = math.nan
    with pytest.raises(ValueError, match="missing"):
        tvbounds_attrition(badx, "y", "d", "s", covariates="x")
    with pytest.raises(ValueError, match="data frame"):
        tvbounds_attrition(dat.to_dict("list"), "y", "d", "s")


def test_item_non_response_among_respondents_is_tolerated():
    dat = make_sim_data(seed=37)
    resp = np.flatnonzero(dat.s == 1)[:4]
    dat.loc[dat.index[resp], "y"] = math.nan   # respondents with missing outcome
    fit = tvbounds_attrition(dat, "y", "d", "s", delta=[0, 1], bootstrap=False)
    assert np.all(np.isfinite(fit.bounds["lower"]))
    assert np.all(np.isfinite(fit.bounds["upper"]))


# ---- kernel-level checks specific to the port ------------------------------
def test_stratum_labels_follow_r_interaction():
    codes, levels = _r_interaction(pd.DataFrame({"a": ["x", "y", "x", "y"],
                                                 "b": [1, 1, 2, 2]}))
    assert levels == ["x.1", "y.1", "x.2", "y.2"]       # first factor varies fastest
    assert codes.tolist() == [0, 1, 2, 3]
    codes, levels = _r_interaction(pd.DataFrame({"a": [True, False, True],
                                                 "b": [1.5, 2.0, 1.5],
                                                 "c": ["p", "q", "p"]}))
    assert levels == ["TRUE.1.5.p", "FALSE.2.q"]         # unused combinations dropped
    assert codes.tolist() == [0, 1, 0]
    codes, levels = _r_interaction(pd.DataFrame({"z": ["beta", "Alpha", "gamma", "Alpha"]}))
    assert levels == ["Alpha", "beta", "gamma"]           # locale (ICU) collation


def test_vectorized_group_kernel_matches_scalar_reference_with_atoms():
    Ys = np.sort(np.array([0.0, 0.0, 1.0, 1.0, 1.0, 2.5, 2.5, 3.0]))
    p = 0.3
    eps = np.linspace(0, 1, 11)
    g = one_group_grid(Ys, p, eps)
    mu = float(np.mean(Ys))
    for i, e in enumerate(eps):
        Ea = Eq_q(Ys, (1 - p) * e)
        Eb = Eq_q(Ys, 1 - p * e)
        Ec = Eq_q(Ys, p * e)
        Ed = Eq_q(Ys, 1 - (1 - p) * e)
        assert g["lo"][i] == (1 / (1 - p)) * Ea + (Eb - Ea)
        assert g["up"][i] == (Ed - Ec) + (1 / (1 - p)) * (mu - Ed)


def test_breakdown_delta_hand_cases():
    dg = np.array([0.0, 0.5, 1.0])
    assert breakdown_delta(dg, [-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]) == 0.0
    assert math.isnan(breakdown_delta(dg, [1.0, 0.5, 0.25], [2.0, 2.0, 2.0]))
    # lower path crosses zero halfway between the first two grid points
    assert breakdown_delta(dg, [1.0, -1.0, -2.0], [2.0, 2.0, 2.0]) == 0.25
    # upper path crosses zero
    assert breakdown_delta(dg, [-2.0, -2.0, -2.0], [-1.0, 3.0, 3.0]) == 0.125
    assert math.isnan(breakdown_delta(dg, [math.nan] * 3, [1.0, 1.0, 1.0]))
