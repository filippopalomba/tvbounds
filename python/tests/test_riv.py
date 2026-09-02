"""Tests for tvbounds_riv() and the formula-instrument kernels.

Translation of ``tests/testthat/test-riv.R``.  Every independent check below
recomputes a quantity by a route that shares no code with the kernels
(hand-computed closed forms, brute-force simplex grids, random-search
containment, direct two-stage least squares).
"""
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from tvbounds import TVBounds, tvbounds_riv
from tvbounds._riv_kernels import (_fs_margin_cont, _fs_margin_tv,
                                   _upper_tail_integral, fi_bounds_cont,
                                   fi_bounds_tv, fi_breakdown, fi_criteria,
                                   fi_delta_fs_cont, fi_delta_fs_tv,
                                   fi_lf_law_tv, tv_bound_lower,
                                   tv_bound_upper)

# testthat's expect_equal() default tolerance
TOL = 1.5e-8


# ---------------------------------------------------------------------------
# Local helpers (test-only, no package code)
# ---------------------------------------------------------------------------
def fake_cr(gy, gx, gy_v, gx_v):
    """A reduced criterion object with prescribed criterion values, mirroring
    the output contract of fi_criteria() on an already-centered design."""
    gy = np.asarray(gy, dtype=float)
    gx = np.asarray(gx, dtype=float)
    gy = gy - gy.mean()
    gx = gx - gx.mean()
    S = gy.size
    return {"gy_s": gy, "gx_s": gx, "gy_v": float(gy_v), "gx_v": float(gx_v),
            "p": np.full(S, 1 / S), "Egy": 0.0, "Egx": 0.0,
            "Gy0": float(gy_v), "Gx0": float(gx_v),
            "beta_hat": gy_v / gx_v, "flip": 1.0, "S": S, "n": 1}


def sim_design(n=60, S=12, seed=1, strength=1):
    """Simulated formula-instrument design with a strong or weak first stage."""
    rng = np.random.default_rng(seed)
    Fmat = rng.standard_normal((n, S))
    z = Fmat.mean(axis=1) + rng.standard_normal(n)
    x = strength * z + rng.standard_normal(n)
    y = 0.4 * x + rng.standard_normal(n)
    return {"y": y, "x": x, "z": z, "Fmat": Fmat, "n": n, "S": S}


def beta_at(cr, q):
    """beta(P) computed directly from a criterion object and a candidate
    distribution q over the draws."""
    return (cr["gy_v"] - np.sum(q * cr["gy_s"])) / (cr["gx_v"] - np.sum(q * cr["gx_s"]))


def assert_equal(a, b, tol=TOL):
    """testthat's expect_equal() for scalars (relative tolerance, +/-Inf equal)."""
    a = float(a)
    b = float(b)
    if math.isinf(a) or math.isinf(b):
        assert a == b
        return
    assert abs(a - b) <= tol * max(1.0, abs(b)), f"{a} != {b}"


# ---------------------------------------------------------------------------
# 1. Criterion bounds: closed form vs hand computation and a simplex grid
# ---------------------------------------------------------------------------
def test_upper_tail_integral_and_tv_criterion_bounds_match_hand_computed_values():
    h = np.array([-1.5, -0.5, 0.5, 1.5])
    p = np.full(4, 0.25)
    # keep the top 0.7 mass: 0.25*1.5 + 0.25*0.5 + 0.20*(-0.5) = 0.40
    assert_equal(_upper_tail_integral(h, p, 0.3), 0.40)
    assert_equal(_upper_tail_integral(h, p, 0), np.sum(p * h))
    assert_equal(_upper_tail_integral(h, p, 1), 0)
    # delta * max(h) + integral: 0.3 * 1.5 + 0.40 = 0.85
    assert_equal(tv_bound_upper(h, p, 0.3), 0.85)
    # small-budget branch: only the bottom atom is trimmed
    assert_equal(tv_bound_upper(h, p, 0.2), 0.6)
    # reflection identity for the lower bound
    assert_equal(tv_bound_lower(h, p, 0.3), -0.85)
    assert_equal(tv_bound_lower(h, p, 0.2), -0.6)
    # degenerate budgets
    assert_equal(tv_bound_upper(h, p, 0), np.sum(p * h))
    assert_equal(tv_bound_upper(h, p, 1), h.max())
    assert_equal(tv_bound_lower(h, p, 1), h.min())


def test_tv_criterion_bounds_agree_with_a_brute_force_simplex_grid():
    # Enumerate distributions on a fine grid over the 2-simplex, keep those in
    # the TV ball, and maximize/minimize the linear criterion directly.
    def brute(h, q, d, step=1 / 200):
        g = np.arange(0, 201) * step
        p1, p2 = np.meshgrid(g, g, indexing="ij")
        p1 = p1.ravel()
        p2 = p2.ravel()
        keep = p1 + p2 <= 1 + 1e-12
        p1 = p1[keep]
        p2 = p2[keep]
        p3 = 1 - p1 - p2
        tv = 0.5 * (np.abs(p1 - q[0]) + np.abs(p2 - q[1]) + np.abs(p3 - q[2]))
        keep = tv <= d + 1e-12
        val = p1[keep] * h[0] + p2[keep] * h[1] + p3[keep] * h[2]
        return {"lo": val.min(), "up": val.max()}

    rng = np.random.default_rng(42)
    for rep in range(1, 6):
        h = rng.normal(0, 2, size=3)
        if rep % 2:
            q = np.full(3, 1 / 3)
        else:
            w = rng.uniform(size=3)
            q = w / w.sum()
        d = rng.uniform(0.05, 0.95)
        b = brute(h, q, d)
        tol = 4 * (1 / 200) * (h.max() - h.min())   # grid discretization error
        # the closed form dominates every feasible grid point ...
        assert tv_bound_upper(h, q, d) >= b["up"] - 1e-10
        assert tv_bound_lower(h, q, d) <= b["lo"] + 1e-10
        # ... and the grid gets within discretization tolerance of it
        assert tv_bound_upper(h, q, d) <= b["up"] + tol
        assert tv_bound_lower(h, q, d) >= b["lo"] - tol


# ---------------------------------------------------------------------------
# 2. First-stage breakdown budgets: hand-computed piecewise-linear example
# ---------------------------------------------------------------------------
def test_first_stage_breakdown_budgets_match_a_hand_solved_example():
    # Centered gx values (-1.5, -0.5, 0.5, 1.5), uniform baseline, gx_v = 1.
    # TV margin: 1 - tv_bound_upper = 1 - 3d on [0, .25], 0.75 - 2d on
    # (.25, .5]; root at d = 0.375.  Contamination: 1 - 1.5 d, root at 2/3.
    cr = fake_cr(gy=[0, 0, 0, 0], gx=[-1.5, -0.5, 0.5, 1.5], gy_v=0.3, gx_v=1)
    d_tv = fi_delta_fs_tv(cr)
    d_ct = fi_delta_fs_cont(cr)
    assert abs(float(d_tv) - 0.375) <= 1e-9
    assert d_tv.censored is False
    assert abs(float(d_ct) - 2 / 3) <= 1e-12
    assert d_ct.censored is False
    # TV breaks first: its neighborhood is the larger one
    assert float(d_tv) <= float(d_ct) + 1e-12
    # margins change sign exactly at the breakdown budgets
    assert _fs_margin_tv(cr, 0.374) > 0
    assert _fs_margin_tv(cr, 0.376) < 0
    assert _fs_margin_cont(cr, 0.66) > 0
    assert _fs_margin_cont(cr, 0.67) < 0


def test_a_first_stage_dominating_every_draw_is_censored_at_1():
    cr = fake_cr(gy=[-1, 0, 1], gx=[-1, 0, 1], gy_v=2, gx_v=10)
    d_tv = fi_delta_fs_tv(cr)
    d_ct = fi_delta_fs_cont(cr)
    assert float(d_tv) == 1
    assert d_tv.censored is True
    assert float(d_ct) == 1
    assert d_ct.censored is True
    # both bound routes stay finite on the whole budget range
    bt = fi_bounds_tv(cr, 1)
    assert math.isfinite(bt["lower"]) and math.isfinite(bt["upper"])
    bc = fi_bounds_cont(cr, 1)
    assert math.isfinite(bc["lower"]) and math.isfinite(bc["upper"])


# ---------------------------------------------------------------------------
# 3. Bounds on the estimate: containment, attainment, nesting, cross-check
# ---------------------------------------------------------------------------
def test_random_feasible_distributions_never_escape_the_tv_bounds_and_the_lf_law_attains_them():
    rng = np.random.default_rng(7)
    for _rep in range(10):
        S = int(rng.integers(4, 10))
        gx = rng.standard_normal(S)
        cr = fake_cr(rng.standard_normal(S), gx, rng.standard_normal(),
                     abs(rng.standard_normal()) + 3 * np.max(np.abs(gx)))
        d = rng.uniform(0.05, 0.9)
        if _fs_margin_tv(cr, d) <= 1e-6:
            continue
        bb = fi_bounds_tv(cr, d)
        assert bb["lower"] <= cr["beta_hat"] + 1e-10
        assert bb["upper"] >= cr["beta_hat"] - 1e-10
        # containment: mixtures (1-a) p + a w have TV distance at most a <= d
        for _k in range(200):
            w = rng.exponential(size=S)
            w = w / w.sum()
            a = rng.uniform(0, d)
            q = (1 - a) * cr["p"] + a * w
            val = beta_at(cr, q)
            assert val >= bb["lower"] - 1e-9
            assert val <= bb["upper"] + 1e-9
        # attainment: the least favorable distribution is a probability
        # distribution, lies within budget, and attains its bound
        for tag in ("lower", "upper"):
            q = fi_lf_law_tv(cr, d, bb[tag],
                             direction="upper" if tag == "lower" else "lower")
            assert abs(np.sum(q) - 1) <= 1e-12
            assert np.all(q >= -1e-15)
            assert 0.5 * np.sum(np.abs(q - cr["p"])) <= d + 1e-9
            assert_equal(beta_at(cr, q), bb[tag], tol=1e-7)


def test_random_contaminating_distributions_never_escape_the_contamination_bounds():
    rng = np.random.default_rng(11)
    for _rep in range(10):
        S = int(rng.integers(4, 10))
        gx = rng.standard_normal(S)
        cr = fake_cr(rng.standard_normal(S), gx, rng.standard_normal(),
                     abs(rng.standard_normal()) + 3 * np.max(np.abs(gx)))
        d = rng.uniform(0.05, 0.9)
        if _fs_margin_cont(cr, d) <= 1e-6:
            continue
        bb = fi_bounds_cont(cr, d)
        for _k in range(200):
            w = rng.exponential(size=S)
            w = w / w.sum()
            q = (1 - d) * cr["p"] + d * w
            val = beta_at(cr, q)
            assert val >= bb["lower"] - 1e-9
            assert val <= bb["upper"] + 1e-9
        # each bound is attained by contamination degenerate at its arg index
        # (0-based in the Python kernels)
        for tag in ("lower", "upper"):
            s0 = bb["arg_" + tag]
            w = np.zeros(S)
            w[s0] = 1
            q = (1 - d) * cr["p"] + d * w
            assert_equal(beta_at(cr, q), bb[tag], tol=1e-10)


def test_contamination_bounds_are_nested_in_tv_bounds_and_the_two_routes_agree_at_delta_1():
    rng = np.random.default_rng(3)
    S = 8
    gx = rng.standard_normal(S)
    cr = fake_cr(rng.standard_normal(S), gx, rng.standard_normal(),
                 abs(rng.standard_normal()) + 3 * np.max(np.abs(gx)))
    for d in (0.1, 0.3, 0.6, 0.9):
        if _fs_margin_tv(cr, d) <= 0:
            continue
        tv = fi_bounds_tv(cr, d)
        ct = fi_bounds_cont(cr, d)
        assert ct["lower"] >= tv["lower"] - 1e-8
        assert ct["upper"] <= tv["upper"] + 1e-8
    # at delta = 1 both neighborhoods are the whole simplex over the draws, so
    # the trimming/root-finding route and the ratio-enumeration route (which
    # share no code) must coincide
    assert fi_delta_fs_tv(cr).censored   # the design is built to be censored
    b_tv = fi_bounds_tv(cr, 1)
    b_ct = fi_bounds_cont(cr, 1)
    assert_equal(b_tv["lower"], b_ct["lower"], tol=1e-8)
    assert_equal(b_tv["upper"], b_ct["upper"], tol=1e-8)


def test_shift_invariance_and_sign_normalization_hold_on_a_simulated_design():
    des = sim_design(n=40, S=10, seed=5)
    y = des["y"] - des["y"].mean()
    x = des["x"] - des["x"].mean()
    cr0 = fi_criteria(y, x, des["z"], des["Fmat"])
    # unit-specific shifts of the formula and the realized instrument together
    cc = np.random.default_rng(6).normal(0, 10, size=des["n"])
    cr1 = fi_criteria(y, x, des["z"] - cc, des["Fmat"] - cc[:, None])
    for d in (0.1, 0.4, 0.8, 1):
        b0 = fi_bounds_tv(cr0, d)
        b1 = fi_bounds_tv(cr1, d)
        assert_equal(b0["lower"], b1["lower"], tol=1e-10)
        assert_equal(b0["upper"], b1["upper"], tol=1e-10)
        c0 = fi_bounds_cont(cr0, d)
        c1 = fi_bounds_cont(cr1, d)
        assert_equal(c0["lower"], c1["lower"], tol=1e-10)
        assert_equal(c0["upper"], c1["upper"], tol=1e-10)
    # replacing (y, x) by (-y, -x) flips the sign normalization, not the bounds
    cr2 = fi_criteria(-y, -x, des["z"], des["Fmat"])
    assert cr2["flip"] == -cr0["flip"]
    assert_equal(cr0["beta_hat"], cr2["beta_hat"], tol=1e-12)
    for d in (0.2, 0.6):
        b0 = fi_bounds_tv(cr0, d)
        b2 = fi_bounds_tv(cr2, d)
        assert_equal(b0["lower"], b2["lower"], tol=1e-9)
        assert_equal(b0["upper"], b2["upper"], tol=1e-9)


# ---------------------------------------------------------------------------
# 4. Breakdown budget for a reference value
# ---------------------------------------------------------------------------
def test_fi_breakdown_locates_the_first_covering_budget_and_handles_censoring():
    # synthetic monotone bounds with a known crossing: lower(d) = 1 - 4d covers
    # tau_star = 0 first, at d = 0.25
    def lo(d):
        return 1 - 4 * d

    def up(d):
        return 1 + d

    assert abs(fi_breakdown(lo, up, tau_star=0) - 0.25) <= 1e-8
    # covered already at the baseline
    assert fi_breakdown(lo, up, tau_star=1) == 0
    # never covered on [0, 1]
    assert math.isnan(fi_breakdown(lo, up, tau_star=5))


# ---------------------------------------------------------------------------
# 5. The user-facing function: schema, collapse at 0, NA encoding, controls
# ---------------------------------------------------------------------------
def test_tvbounds_riv_returns_the_common_object_with_the_required_schema():
    des = sim_design(seed=2)
    grid = np.linspace(0, 1, 11)
    fit = tvbounds_riv(des["y"], des["x"], des["z"], des["Fmat"], delta=grid)
    assert isinstance(fit, TVBounds)
    assert fit.application == "riv"
    assert fit.neighborhood == "tv"
    assert list(fit.bounds.columns) == ["delta", "lower", "upper"]
    assert fit.bounds.shape[0] == 11
    assert fit.n == des["y"].size
    assert math.isnan(fit.level)
    assert math.isnan(fit.B)
    assert fit.estimand_label == "IV coefficient"
    # delta = 0 row collapses to the baseline estimate
    assert_equal(fit.bounds["lower"].iloc[0], fit.point)
    assert_equal(fit.bounds["upper"].iloc[0], fit.point)
    # monotone in the budget on the finite range
    fin = fit.bounds.notna().all(axis=1).to_numpy()
    assert np.all(np.diff(fit.bounds["lower"].to_numpy()[fin]) <= 1e-8)
    assert np.all(np.diff(fit.bounds["upper"].to_numpy()[fin]) >= -1e-8)
    # deterministic: the same call reproduces the same object
    fit2 = tvbounds_riv(des["y"], des["x"], des["z"], des["Fmat"], delta=grid)
    pd.testing.assert_frame_equal(fit.bounds, fit2.bounds)
    for k, v in fit.details["criteria"].items():
        np.testing.assert_array_equal(np.asarray(v), np.asarray(fit2.details["criteria"][k]))


def test_bounds_rows_are_nan_exactly_where_the_first_stage_can_vanish():
    # weak first stage so that the breakdown budget is interior
    des = sim_design(n=50, S=10, seed=8, strength=0.05)
    grid = np.linspace(0, 1, 51)
    fit = tvbounds_riv(des["y"], des["x"], des["z"], des["Fmat"], delta=grid)
    dfs = fit.details["delta_fs"]
    assert fit.details["delta_fs_censored"] is False
    assert dfs > 0
    assert dfs < 1
    na_rows = fit.bounds["lower"].isna().to_numpy()
    np.testing.assert_array_equal(na_rows, fit.bounds["upper"].isna().to_numpy())
    # finite strictly below the breakdown budget, NA strictly above it
    assert np.all(~na_rows[grid < dfs - 0.02])
    assert np.all(na_rows[grid > dfs + 0.02])
    # the reported breakdown budget for the sign is covered by the bounds
    db = fit.details["delta_breakdown"]
    if not fit.details["delta_breakdown_censored"] and db < dfs:
        lo = fi_bounds_tv(fit.details["criteria"], min(db + 1e-6, 1))
        assert lo["lower"] <= fit.details["tau_star"] + 1e-6
        assert lo["upper"] >= fit.details["tau_star"] - 1e-6


def test_contamination_bounds_are_nested_in_tv_bounds_through_the_user_facing_interface():
    des = sim_design(seed=4)
    grid = np.linspace(0, 1, 11)
    fit_tv = tvbounds_riv(des["y"], des["x"], des["z"], des["Fmat"], delta=grid)
    fit_ct = tvbounds_riv(des["y"], des["x"], des["z"], des["Fmat"], delta=grid,
                          neighborhood="contamination")
    assert fit_ct.neighborhood == "contamination"
    assert_equal(fit_tv.point, fit_ct.point)
    fin = (fit_tv.bounds.notna().all(axis=1) & fit_ct.bounds.notna().all(axis=1)).to_numpy()
    assert np.any(fin)
    assert np.all(fit_ct.bounds["lower"].to_numpy()[fin]
                  >= fit_tv.bounds["lower"].to_numpy()[fin] - 1e-8)
    assert np.all(fit_ct.bounds["upper"].to_numpy()[fin]
                  <= fit_tv.bounds["upper"].to_numpy()[fin] + 1e-8)
    # the TV first stage breaks (weakly) before the contamination one
    assert fit_tv.details["delta_fs"] <= fit_ct.details["delta_fs"] + 1e-12


def test_fwl_residualization_replicates_a_direct_two_stage_least_squares_fit():
    des = sim_design(n=70, S=8, seed=9)
    rng = np.random.default_rng(909)
    W = np.column_stack([rng.standard_normal(des["n"]), rng.uniform(size=des["n"])])
    fit = tvbounds_riv(des["y"], des["x"], des["z"], des["Fmat"], controls=W,
                       delta=np.array([0, 0.1]))
    # direct just-identified 2SLS of y on (x, 1, W) with instruments
    # (z - F p, 1, W), no residualization anywhere
    zr = des["z"] - des["Fmat"].mean(axis=1)
    Zmat = np.column_stack([zr, np.ones(des["n"]), W])
    Xmat = np.column_stack([des["x"], np.ones(des["n"]), W])
    b_direct = np.linalg.solve(Zmat.T @ Xmat, Zmat.T @ des["y"])[0]
    assert_equal(fit.point, b_direct, tol=1e-9)
    assert fit.details["n_controls"] == 2
    # a data-frame controls argument gives the same result
    fit_df = tvbounds_riv(des["y"], des["x"], des["z"], des["Fmat"],
                          controls=pd.DataFrame(W, columns=["w1", "w2"]),
                          delta=np.array([0, 0.1]))
    assert_equal(fit_df.point, fit.point)
    pd.testing.assert_frame_equal(fit_df.bounds, fit.bounds)


def test_censoring_flags_propagate_to_details_for_a_strong_first_stage():
    des = sim_design(seed=10, strength=2)
    fit = tvbounds_riv(des["y"], des["x"], des["z"], des["Fmat"],
                       delta=np.linspace(0, 1, 5))
    assert fit.details["delta_fs_censored"] is True
    assert fit.details["delta_fs"] == 1
    assert fit.details["delta_fs_tv"].censored is True
    assert fit.details["delta_fs_cont"].censored is True
    # no NA rows: the bounds stay finite over the whole budget range
    assert fit.bounds.notna().all(axis=None)
    # criteria stored in details reproduce the bounds without the design
    cr = fit.details["criteria"]
    redo = [fi_bounds_tv(cr, d) for d in fit.bounds["delta"]]
    for k, r in enumerate(redo):
        assert_equal(fit.bounds["lower"].iloc[k], r["lower"])
        assert_equal(fit.bounds["upper"].iloc[k], r["upper"])


def test_a_supplied_baseline_distribution_p_is_honored():
    des = sim_design(n=40, S=6, seed=12)
    p = np.array([0.4, 0.2, 0.1, 0.1, 0.1, 0.1])
    fit = tvbounds_riv(des["y"], des["x"], des["z"], des["Fmat"], p=p,
                       delta=np.array([0, 0.2, 0.5]))
    np.testing.assert_allclose(fit.details["criteria"]["p"], p, rtol=TOL)
    # the point estimate matches the direct recentering under p
    mu = des["Fmat"] @ p
    yr = des["y"] - des["y"].mean()
    xr = des["x"] - des["x"].mean()
    b_direct = np.sum(yr * (des["z"] - mu)) / np.sum(xr * (des["z"] - mu))
    assert_equal(fit.point, b_direct, tol=1e-10)


# ---------------------------------------------------------------------------
# 6. Input validation
# ---------------------------------------------------------------------------
def test_tvbounds_riv_validates_its_inputs_with_informative_errors():
    des = sim_design(n=20, S=5, seed=13)
    y, x, z, Fmat = des["y"], des["x"], des["z"], des["Fmat"]

    def ok(**kw):
        return tvbounds_riv(y, x, z, Fmat, **kw)

    with pytest.raises(ValueError, match="same length"):
        tvbounds_riv(y[1:], x, z, Fmat)
    with pytest.raises(ValueError, match="`y`"):
        tvbounds_riv(np.concatenate([y[1:], [np.nan]]), x, z, Fmat)
    with pytest.raises(ValueError, match="one row per observation"):
        tvbounds_riv(y, x, z, Fmat[1:, :])
    with pytest.raises(ValueError, match="`Fmat`"):
        tvbounds_riv(y, x, z, "not a matrix")
    with pytest.raises(ValueError, match="at least 2"):
        tvbounds_riv(y, x, z, Fmat[:, :1])
    with pytest.raises(ValueError, match="sum to 1"):
        ok(p=np.ones(5))
    with pytest.raises(ValueError, match="nonnegative"):
        ok(p=np.array([-0.2, 0.3, 0.3, 0.3, 0.3]))
    with pytest.raises(ValueError, match="one probability per column"):
        ok(p=np.full(4, 0.25))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ok(delta=np.array([-0.1, 0.5]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ok(delta=np.array([0.2, 1.5]))
    with pytest.raises(ValueError, match="duplicated"):
        ok(delta=np.array([0.2, 0.2]))
    with pytest.raises(ValueError, match="non-empty"):
        ok(delta=np.array([]))
    with pytest.raises(ValueError, match="missing"):
        ok(delta=np.array([0.1, np.nan]))
    with pytest.raises(ValueError, match="`tau_star`"):
        ok(tau_star=np.array([0, 1]))
    with pytest.raises(ValueError, match="`tau_star`"):
        ok(tau_star=math.inf)
    with pytest.raises(ValueError, match="`verbose`"):
        ok(verbose=None)
    rng = np.random.default_rng(13)
    with pytest.raises(ValueError, match="one row per observation"):
        ok(controls=rng.standard_normal((5, 2)))
    W_bad = pd.DataFrame({"w1": rng.standard_normal(20),
                          "w2": list("abcdefghijklmnopqrst")})
    with pytest.raises(ValueError, match="numeric columns"):
        ok(controls=W_bad)
    with pytest.raises(ValueError, match="missing or non-finite"):
        ok(controls=np.concatenate([rng.standard_normal(19), [np.nan]]).reshape(20, 1))
    with pytest.raises(ValueError, match="`neighborhood`"):
        ok(neighborhood="kl")


def test_verbose_true_emits_progress_messages_and_verbose_false_is_silent(capsys):
    des = sim_design(n=30, S=5, seed=14)
    tvbounds_riv(des["y"], des["x"], des["z"], des["Fmat"], delta=np.array([0, 0.5]),
                 verbose=True)
    msgs = capsys.readouterr().err
    assert "Recentered IV design" in msgs
    assert "First-stage breakdown budget" in msgs
    assert "Breakdown budget for tau_star" in msgs
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tvbounds_riv(des["y"], des["x"], des["z"], des["Fmat"], delta=np.array([0, 0.5]))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert len(w) == 0
