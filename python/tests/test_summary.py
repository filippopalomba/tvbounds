"""Tests for tvbounds_summary() and its print method (port of test-summary.R),
on fixtures with analytic breakdown budgets, shadow prices, and frontier
values."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tvbounds import tvbounds_summary, TVBoundsSummary
from tvbounds._rcompat import rqnorm
from helpers import (tvb_fixture_linear, tvb_fixture_linear_upper,
                     tvb_fixture_censored, tvb_fixture_noinf,
                     tvb_fixture_counterfactual, tvb_fixture_quadratic)


def _m(s):
    return s.measures.iloc[0]


def test_linear_fixture_reproduces_all_analytic_summary_measures():
    obj = tvb_fixture_linear()          # point 0.5, slope 1, se 0.1, gap 0.15
    s = tvbounds_summary(obj)
    m = _m(s)
    zc = rqnorm(0.975)

    assert isinstance(s, TVBoundsSummary)
    assert s.measures.shape[0] == 1
    assert m["direction"] == "lower"    # point 0.5 > tau* = 0

    # Breakdown budgets: plug-in at 0.5, certified at 0.35, normal floor at
    # 0.5 - zc * 0.1.
    assert m["delta_b"] == pytest.approx(0.5, rel=1e-6)
    assert not m["censored"]
    assert m["delta_b_ci"] == pytest.approx(0.35, rel=1e-6)
    assert not m["censored_ci"]
    assert m["delta_b_ci_norm"] == pytest.approx(0.5 - zc * 0.1, rel=1e-6)

    # Shadow price and robustness standard error at the plug-in breakdown:
    # eta = slope = 1, se = 0.1, n = 100 -> varsigma = 0.1 * 10 / 1 = 1.
    assert m["delta_eval"] == pytest.approx(0.5, rel=1e-6)
    assert m["eta"] == pytest.approx(1, rel=1e-6)
    assert m["se"] == pytest.approx(0.1, rel=1e-6)
    assert m["varsigma"] == pytest.approx(1, rel=1e-6)
    assert m["varsigma_sc"] == pytest.approx(0.1, rel=1e-6)

    # Certification frontier priced at the certified breakdown 0.35 and at
    # 0.35 + jump = 0.40: path values 0.15 and 0.10, sigma^2 = se^2 * n = 1.
    assert m["frontier_at"] == pytest.approx(0.35, rel=1e-6)
    assert m["n_cur"] == pytest.approx(zc * zc / 0.15 ** 2, rel=1e-6)
    assert float(m["n_star"]) == math.floor(zc * zc / 0.10 ** 2) + 1   # 385
    assert m["delta_n"] == m["n_star"] - 100
    assert m["cost_per_pp"] == pytest.approx(m["delta_n"] * 50 / (100 * 0.05), rel=1e-12)


def test_direction_auto_uses_the_upper_path_when_point_below_tau_star():
    lo = tvbounds_summary(tvb_fixture_linear())
    up = tvbounds_summary(tvb_fixture_linear_upper())
    assert _m(up)["direction"] == "upper"
    # The mirrored object has the same signed path, hence identical measures.
    same = ["delta_b", "delta_b_ci", "delta_b_ci_norm", "eta", "se",
            "varsigma", "varsigma_sc", "n_cur", "n_star", "delta_n",
            "cost_per_pp"]
    for col in same:
        assert _m(up)[col] == pytest.approx(_m(lo)[col], rel=1e-6), col


def test_forced_directions_are_respected():
    obj = tvb_fixture_linear()
    s_lo = tvbounds_summary(obj, direction="lower")
    assert _m(s_lo)["direction"] == "lower"
    # Forcing the upper path on an object whose point exceeds tau_star gives
    # a path that is already nonpositive at the smallest budget: the
    # breakdown collapses to the left endpoint of the grid.
    s_up = tvbounds_summary(obj, direction="upper")
    assert _m(s_up)["direction"] == "upper"
    assert _m(s_up)["delta_b"] == pytest.approx(0, abs=1e-12)


def test_censored_plugin_breakdown_reported_at_right_endpoint_and_anchored_at_certified():
    obj = tvb_fixture_censored()        # lower = 0.5 - 0.3 delta, never crosses
    s = tvbounds_summary(obj)
    m = _m(s)

    assert m["censored"]
    assert m["delta_b"] == 1            # right endpoint, not NA / "> 1"
    assert m["delta_b_ci"] == pytest.approx(0.25 / 0.3, rel=1e-6)
    assert not m["censored_ci"]
    # Anchored at the certified budget: eta = 0.3 there, varsigma = 0.1 * 10 / 0.3.
    assert m["delta_eval"] == pytest.approx(0.25 / 0.3, rel=1e-6)
    assert m["eta"] == pytest.approx(0.3, rel=1e-6)
    assert m["varsigma"] == pytest.approx(1 / 0.3, rel=1e-5)
    assert any("censored" in nt for nt in s.notes)


def test_objects_without_inference_degrade_gracefully():
    obj = tvb_fixture_noinf()
    s = tvbounds_summary(obj)
    m = _m(s)

    assert m["delta_b"] == pytest.approx(0.3, rel=1e-6)
    assert not s.has_se
    assert not s.has_ci
    assert math.isnan(m["delta_b_ci"])
    assert math.isnan(m["delta_b_ci_norm"])
    assert math.isnan(m["se"])
    assert math.isnan(m["varsigma"])
    assert math.isnan(m["n_cur"])
    assert math.isnan(m["n_star"])
    # eta is a plug-in quantity and survives: slope of the lower path is 1.
    assert m["eta"] == pytest.approx(1, rel=1e-6)
    assert len(s.notes) >= 1
    assert any(("no inference by design" in nt) or ("no confidence band" in nt)
               for nt in s.notes)


def test_counterfactual_grids_exceeding_1_are_handled_including_censoring():
    s = tvbounds_summary(tvb_fixture_counterfactual(slope=0.4))
    # 1 - 0.4 delta crosses zero at 2.5, interior to the uneven grid.
    assert _m(s)["delta_b"] == pytest.approx(2.5, rel=1e-6)
    assert not _m(s)["censored"]

    s2 = tvbounds_summary(tvb_fixture_counterfactual(slope=0.1))
    # 1 - 0.1 delta stays positive up to delta = 5: censored at the right
    # endpoint of THIS grid (5), not at 1.
    assert _m(s2)["censored"]
    assert _m(s2)["delta_b"] == 5


def test_user_supplied_evaluation_budget_overrides_the_anchor():
    obj = tvb_fixture_quadratic()       # lower = 0.5 - delta^2, eta = 2 delta
    zc = rqnorm(0.975)
    s = tvbounds_summary(obj, delta=0.3)
    m = _m(s)

    assert m["delta_eval"] == 0.3
    assert m["eta"] == pytest.approx(0.6, rel=1e-8)
    assert m["varsigma"] == pytest.approx(0.1 * 10 / 0.6, rel=1e-6)
    # Frontier priced at the evaluation budget and at budget + jump.
    assert m["frontier_at"] == 0.3
    assert m["n_cur"] == pytest.approx(zc * zc / (0.5 - 0.09) ** 2, rel=1e-6)
    assert float(m["n_star"]) == math.floor(zc * zc / (0.5 - 0.35 ** 2) ** 2) + 1


def test_nonzero_tau_star_shifts_the_breakdown_budget():
    obj = tvb_fixture_linear()          # lower = 0.5 - delta
    s = tvbounds_summary(obj, tau_star=0.2)
    assert _m(s)["delta_b"] == pytest.approx(0.3, rel=1e-6)


def test_input_validation_fails_early_with_informative_errors():
    obj = tvb_fixture_linear()
    with pytest.raises(ValueError, match="tvbounds"):
        tvbounds_summary(42)
    with pytest.raises(ValueError, match="delta"):
        tvbounds_summary(obj, delta=[0.1, 0.2])
    with pytest.raises(ValueError, match="delta"):
        tvbounds_summary(obj, delta=np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="delta"):
        tvbounds_summary(obj, delta=-0.1)
    with pytest.raises(ValueError, match="level"):
        tvbounds_summary(obj, level=1.2)
    with pytest.raises(ValueError, match="jump"):
        tvbounds_summary(obj, jump=0)
    with pytest.raises(ValueError, match="cost_per_unit"):
        tvbounds_summary(obj, cost_per_unit=-1)
    with pytest.raises(ValueError, match="tau_star"):
        tvbounds_summary(obj, tau_star=[0, 1])
    with pytest.raises(ValueError):
        tvbounds_summary(obj, direction="sideways")

    # direction = "auto" needs a baseline point estimate.
    nopt = tvb_fixture_linear()
    nopt.point = math.nan
    with pytest.raises(ValueError, match="direction"):
        tvbounds_summary(nopt)
    assert isinstance(tvbounds_summary(nopt, direction="lower"), TVBoundsSummary)


def test_validation_messages_match_the_r_package():
    obj = tvb_fixture_linear()
    with pytest.raises(ValueError, match=r"^`object` must be a `tvbounds` object\.$"):
        tvbounds_summary(42)
    with pytest.raises(ValueError,
                       match=r"^`delta` must be None or a single nonnegative finite number\.$"):
        tvbounds_summary(obj, delta=math.inf)
    with pytest.raises(ValueError, match=r"^`tau_star` must be a finite numeric scalar\.$"):
        tvbounds_summary(obj, tau_star=math.nan)
    with pytest.raises(ValueError,
                       match=r"^`level` must be a number strictly between 0 and 1\.$"):
        tvbounds_summary(obj, level=0)
    with pytest.raises(ValueError,
                       match=r"^`cost_per_unit` must be a single nonnegative number\.$"):
        tvbounds_summary(obj, cost_per_unit="50")
    with pytest.raises(ValueError, match=r"^`jump` must be a single positive number\.$"):
        tvbounds_summary(obj, jump=True)
    with pytest.raises(ValueError, match="should be one of"):
        tvbounds_summary(obj, direction="sideways")
    # R's match.arg accepts a unique partial match.
    assert tvbounds_summary(obj, direction="low").direction == "lower"
    # Fewer than two usable budgets on the path.
    short = tvb_fixture_linear(delta=[0, 1])
    short.bounds.loc[1, "lower"] = math.nan
    with pytest.raises(ValueError, match="at least two non-missing budget values on the lower"):
        tvbounds_summary(short)


def test_evaluation_budgets_beyond_the_grid_warn_and_extrapolate_flat():
    obj = tvb_fixture_linear()
    with pytest.warns(UserWarning, match="extrapolation"):
        s = tvbounds_summary(obj, delta=1.5)
    # Flat extrapolation: eta at 1.5 equals eta at the right endpoint.
    assert _m(s)["eta"] == pytest.approx(1, rel=1e-6)


def test_summary_method_dispatches_to_tvbounds_summary():
    obj = tvb_fixture_linear()
    s1 = obj.summary()
    s2 = tvbounds_summary(obj)
    assert isinstance(s1, TVBoundsSummary)
    pd.testing.assert_frame_equal(s1.measures, s2.measures, rtol=1e-12, atol=0)


def test_measures_columns_follow_the_r_order_and_types():
    s = tvbounds_summary(tvb_fixture_linear())
    assert list(s.measures.columns) == [
        "label", "direction", "n", "point", "tau_star", "delta_b", "censored",
        "delta_b_ci", "censored_ci", "delta_b_ci_norm", "delta_eval", "eta",
        "se", "varsigma", "varsigma_sc", "frontier_at", "n_cur", "n_star",
        "delta_n", "cost_per_pp"]
    assert s.measures["censored"].dtype == bool
    assert s.measures["censored_ci"].dtype == bool
    assert s.measures["n_star"].dtype == np.float64
    assert s.measures["label"].iloc[0] == "treatment effect"
    assert list(s.keys()) == [
        "measures", "application", "estimand_label", "neighborhood",
        "divergence", "direction", "tau_star", "delta_eval", "level",
        "band_level", "zc", "cost_per_unit", "jump", "has_se", "has_ci",
        "notes", "call"]
    assert s["zc"] == rqnorm(0.975)
    assert s.band_level == 0.95
    assert s.call.startswith("tvbounds_summary(")


def test_level_note_when_the_band_level_differs():
    s = tvbounds_summary(tvb_fixture_linear(), level=0.9)
    assert s.notes == ["`level` (0.900) differs from the level of the band stored in the "
                       "object (0.950); the certified breakdown is read off the stored band."]
    # No such note when the object carries no band level (NaN).
    s2 = tvbounds_summary(tvb_fixture_noinf(), level=0.9)
    assert not any("differs" in nt for nt in s2.notes)


def test_print_methods_render_compactly_and_return_invisibly(capsys):
    obj = tvb_fixture_linear()
    print(obj)
    out = capsys.readouterr().out
    assert "tvbounds" in out
    assert "total-variation neighborhood" in out
    assert "bootstrap" in out

    s = tvbounds_summary(obj)
    print(s)
    out = capsys.readouterr().out
    assert "breakdown budget" in out
    assert "shadow price" in out
    assert "certification frontier" in out
    # print methods return their argument (R: invisibly).
    vis = s.print()
    capsys.readouterr()
    assert vis is s
    assert obj.print() is obj
    capsys.readouterr()

    # Censored and no-inference objects print without error.
    assert "censored" in str(tvbounds_summary(tvb_fixture_censored()))
    assert "note" in str(tvbounds_summary(tvb_fixture_noinf()))
    assert "none attached" in str(tvb_fixture_noinf())
