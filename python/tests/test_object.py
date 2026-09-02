"""Tests of the common result object: constructor validation, BudgetValue,
and the print methods (the text of R's print.tvbounds and
print.tvbounds_summary, verified against the R package)."""
from __future__ import annotations

import math
import pickle

import numpy as np
import pandas as pd
import pytest

from tvbounds import (TVBounds, TVBoundsSummary, BudgetValue, new_tvbounds,
                      is_tvbounds, tvbounds_summary)
from helpers import (tvb_fixture_linear, tvb_fixture_linear_upper,
                     tvb_fixture_censored, tvb_fixture_noinf,
                     tvb_fixture_counterfactual, tvb_fixture_quadratic)


def _bounds(delta, lower=0.0, upper=1.0):
    delta = np.asarray(delta, dtype=float)
    return pd.DataFrame({"delta": delta,
                         "lower": np.broadcast_to(np.asarray(lower, float), delta.shape),
                         "upper": np.broadcast_to(np.asarray(upper, float), delta.shape)})


# ---------------------------------------------------------------------------
# constructor
# ---------------------------------------------------------------------------
def test_constructor_rejects_duplicated_budgets():
    with pytest.raises(ValueError,
                       match=r"^`bounds\$delta` must not contain duplicated budget values\.$"):
        new_tvbounds("attrition", _bounds([0, 0.5, 0.5]))


def test_constructor_rejects_negative_budgets():
    with pytest.raises(ValueError, match=r"^`bounds\$delta` must be nonnegative\.$"):
        new_tvbounds("attrition", _bounds([0, -0.5, 1]))


def test_constructor_rejects_lower_above_upper_with_r_message():
    with pytest.raises(ValueError,
                       match=r"^Lower bound exceeds upper bound at delta = 0\.5\.$"):
        new_tvbounds("attrition", _bounds([0, 0.5, 1], lower=[0, 1, 0], upper=[0, 0.5, 0]))
    # several offending budgets, formatted with signif(., 4) as R does
    with pytest.raises(ValueError,
                       match=r"^Lower bound exceeds upper bound at delta = 0\.1235, 1\.$"):
        new_tvbounds("attrition", _bounds([0.123456, 0.5, 1], lower=[1, 0, 1],
                                          upper=[0, 1, 0]))
    # the tolerance of 1e-8 is respected
    obj = new_tvbounds("attrition", _bounds([0, 1], lower=[0, 0.5 + 5e-9],
                                            upper=[0, 0.5]))
    assert isinstance(obj, TVBounds)


def test_constructor_validates_application_neighborhood_and_columns():
    with pytest.raises(ValueError, match="application"):
        new_tvbounds("other", _bounds([0, 1]))
    with pytest.raises(ValueError,
                       match=r'^`neighborhood` must be "tv" or "contamination"\.$'):
        new_tvbounds("attrition", _bounds([0, 1]), neighborhood="kl")
    with pytest.raises(ValueError, match="bounds"):
        new_tvbounds("attrition", pd.DataFrame({"delta": [0, 1], "lower": [0, 0]}))
    with pytest.raises(ValueError, match="bounds"):
        new_tvbounds("attrition", {"delta": [0, 1]})
    with pytest.raises(ValueError, match="details"):
        new_tvbounds("attrition", _bounds([0, 1]), details=[1, 2])


def test_constructor_sorts_by_delta_and_resets_the_index():
    b = _bounds([1, 0, 0.5], lower=[-1, 0, -0.5], upper=[1, 0, 0.5])
    b["lower_se"] = [0.3, 0.1, 0.2]
    obj = new_tvbounds("attrition", b, point=0, n=10, neighborhood="tv")
    assert list(obj.bounds["delta"]) == [0, 0.5, 1]
    assert list(obj.bounds["lower_se"]) == [0.1, 0.2, 0.3]
    assert isinstance(obj.bounds.index, pd.RangeIndex)
    assert list(obj.bounds.columns) == ["delta", "lower", "upper", "lower_se"]
    # NaN budgets are kept and sorted last, as R's order() does
    b2 = _bounds([1, math.nan, 0])
    obj2 = new_tvbounds("riv", b2)
    d = obj2.bounds["delta"].tolist()
    assert d[:2] == [0, 1] and math.isnan(d[2])


def test_object_fields_item_access_and_defaults():
    obj = new_tvbounds("riv", _bounds([0, 1]))
    assert is_tvbounds(obj) and not is_tvbounds(42)
    assert math.isnan(obj.point) and math.isnan(obj.n)
    assert obj.neighborhood is None and obj.divergence is None
    assert math.isnan(obj.level) and math.isnan(obj.B)
    assert obj.estimand_label == "estimand"
    assert obj.call is None and obj.details == {}
    assert obj["application"] == "riv"
    assert obj["bounds"] is obj.bounds
    with pytest.raises(KeyError):
        obj["nonexistent"]
    assert list(obj.keys()) == ["application", "bounds", "point", "n", "neighborhood",
                                "divergence", "level", "B", "estimand_label",
                                "call", "details"]


# ---------------------------------------------------------------------------
# BudgetValue
# ---------------------------------------------------------------------------
def test_budget_value_is_a_float_carrying_the_censored_flag():
    b = BudgetValue(0.3, censored=True)
    assert isinstance(b, float)
    assert float(b) == 0.3
    assert b.censored is True
    assert BudgetValue(1.0).censored is False
    assert b + 1 == pytest.approx(1.3)
    assert type(b + 1) is float
    assert "censored=True" in repr(b)
    c = pickle.loads(pickle.dumps(b))
    assert isinstance(c, BudgetValue) and c.censored is True and float(c) == 0.3
    assert math.isnan(BudgetValue(math.nan, censored=True))


# ---------------------------------------------------------------------------
# print methods (text verified against the R package)
# ---------------------------------------------------------------------------
def test_print_tvbounds_matches_the_r_text(capsys):
    obj = tvb_fixture_linear()
    print(obj)
    out = capsys.readouterr().out
    assert out == (
        "<tvbounds> attrition bounds under a total-variation neighborhood\n"
        "  estimand: treatment effect; baseline point estimate (delta = 0): 0.5; n = 100\n"
        "  budget grid: 101 values of delta in [0, 1]\n"
        "  inference: 95% bootstrap confidence bands (B = 500)\n"
        "  Use summary() for breakdown and price-of-robustness measures; plot() to display.\n")
    assert repr(obj) == str(obj) == obj.format()

    assert str(tvb_fixture_noinf()) == (
        "<tvbounds> riv bounds under a total-variation neighborhood\n"
        "  estimand: IV coefficient; baseline point estimate (delta = 0): 0.3; n = 250\n"
        "  budget grid: 51 values of delta in [0, 1]\n"
        "  inference: none attached\n"
        "  Use summary() for breakdown and price-of-robustness measures; plot() to display.")

    assert str(tvb_fixture_counterfactual()) == (
        "<tvbounds> counterfactual bounds under a KL divergence neighborhood\n"
        "  estimand: counterfactual mean; baseline point estimate (delta = 0): 1; n = 1,000\n"
        "  budget grid: 6 values of delta in [0, 5]\n"
        "  inference: none attached\n"
        "  Use summary() for breakdown and price-of-robustness measures; plot() to display.")

    nopt = tvb_fixture_linear()
    nopt.point = math.nan
    assert "baseline point estimate (delta = 0): ---; n = 100" in str(nopt)
    assert "baseline point estimate (delta = 0): -0.5" in str(tvb_fixture_linear_upper())


def test_print_tvbounds_summary_matches_the_r_text():
    s = tvbounds_summary(tvb_fixture_linear())
    assert isinstance(s, TVBoundsSummary)
    assert str(s) == (
        "<tvbounds summary> treatment effect (attrition application, total-variation neighborhood)\n"
        "  direction: lower bound path relative to tau_star = 0 (baseline point = 0.5, n = 100)\n"
        "  breakdown budget: plug-in = 0.5; certified = 0.35; normal floor = 0.304\n"
        "  at delta = 0.5: shadow price eta = 1; robustness SE varsigma = 1 (scale-free 0.1)\n"
        "  certification frontier: n* = 385 at budget 0.35 + jump 0.05 (Delta n = +285, cost per pp = 2850)")

    assert str(tvbounds_summary(tvb_fixture_censored())) == (
        "<tvbounds summary> treatment effect (attrition application, total-variation neighborhood)\n"
        "  direction: lower bound path relative to tau_star = 0 (baseline point = 0.5, n = 100)\n"
        "  breakdown budget: plug-in = 1 (censored); certified = 0.833; normal floor = ---\n"
        "  at delta = 0.833: shadow price eta = 0.3; robustness SE varsigma = 3.33 (scale-free 0.333)\n"
        "  certification frontier: n* = 70 at budget 0.833 + jump 0.05 (Delta n = -30, cost per pp = -300)\n"
        "  note: the plug-in breakdown is censored at the right endpoint of the budget grid; "
        "the shadow price and the robustness standard error are anchored at the certified "
        "breakdown instead. ")

    assert str(tvbounds_summary(tvb_fixture_noinf())) == (
        "<tvbounds summary> IV coefficient (riv application, total-variation neighborhood)\n"
        "  direction: lower bound path relative to tau_star = 0 (baseline point = 0.3, n = 250)\n"
        "  breakdown budget: plug-in = 0.3; certified = ---; normal floor = ---\n"
        "  at delta = 0.3: shadow price eta = 1; robustness SE varsigma = --- (scale-free ---)\n"
        "  certification frontier: n* = --- at budget --- + jump 0.05 (Delta n = ---, cost per pp = ---)\n"
        "  note: no confidence band is attached to the lower bound path, so the certified "
        "breakdown and the certification frontier are reported as NA (this application "
        "carries no inference by design). \n"
        "  note: no bootstrap standard errors are attached to the lower bound path, so the "
        "normal-floor diagnostic and the robustness standard error are reported as NA. ")

    assert str(tvbounds_summary(tvb_fixture_counterfactual(0.1))) == (
        "<tvbounds summary> counterfactual mean (counterfactual application, KL divergence neighborhood)\n"
        "  direction: lower bound path relative to tau_star = 0 (baseline point = 1, n = 1,000)\n"
        "  breakdown budget: plug-in = 5 (censored); certified = ---; normal floor = ---\n"
        "  at delta = ---: shadow price eta = ---; robustness SE varsigma = --- (scale-free ---)\n"
        "  certification frontier: n* = --- at budget --- + jump 0.05 (Delta n = ---, cost per pp = ---)\n"
        "  note: no interior breakdown budget is available to anchor the shadow price; pass an "
        "explicit `delta` to evaluate the measures at a chosen budget. \n"
        "  note: no confidence band is attached to the lower bound path, so the certified "
        "breakdown and the certification frontier are reported as NA (this application "
        "carries no inference by design). \n"
        "  note: no bootstrap standard errors are attached to the lower bound path, so the "
        "normal-floor diagnostic and the robustness standard error are reported as NA. ")

    assert str(tvbounds_summary(tvb_fixture_quadratic(), delta=0.3)) == (
        "<tvbounds summary> treatment effect (attrition application, total-variation neighborhood)\n"
        "  direction: lower bound path relative to tau_star = 0 (baseline point = 0.5, n = 100)\n"
        "  breakdown budget: plug-in = 0.707; certified = 0.592; normal floor = 0.551\n"
        "  at delta = 0.3: shadow price eta = 0.6; robustness SE varsigma = 1.67 (scale-free 0.167)\n"
        "  certification frontier: n* = 27 at budget 0.3 + jump 0.05 (Delta n = -73, cost per pp = -730)")

    s9 = tvbounds_summary(tvb_fixture_linear(), delta=0.3, level=0.9)
    assert str(s9).endswith(
        "  certification frontier: n* = 121 at budget 0.3 + jump 0.05 (Delta n = +21, cost per pp = 210)\n"
        "  note: `level` (0.900) differs from the level of the band stored in the object "
        "(0.950); the certified breakdown is read off the stored band. ")
    assert s9.format(digits=5).startswith(
        "<tvbounds summary> treatment effect (attrition application, total-variation neighborhood)\n"
        "  direction: lower bound path relative to tau_star = 0 (baseline point = 0.5, n = 100)\n"
        "  breakdown budget: plug-in = 0.5; certified = 0.35; normal floor = 0.33551\n")
    assert s9["measures"] is s9.measures
    with pytest.raises(KeyError):
        s9["nonexistent"]


def test_r_number_formatting_helpers():
    from tvbounds._object import _fmt, _fmt_int, _format_r, _r_signif
    # format(signif(v, 3)) as R renders it, including the fixed/scientific choice
    assert _fmt(100000, 3) == "1e+05"
    assert _fmt(123456, 3) == "123000"
    assert _fmt(1234567890, 3) == "1.23e+09"
    assert _fmt(1.234e-5, 3) == "1.23e-05"
    assert _fmt(0.0001234, 3) == "0.000123"
    assert _fmt(1 / 3, 3) == "0.333"
    assert _fmt(1 / 3, 4) == "0.3333"
    assert _fmt(-0.000123456, 3) == "-0.000123"
    assert _fmt(0, 3) == "0"
    assert _fmt(math.nan, 3) == "---"
    assert _fmt(None, 3) == "---"
    assert _fmt(math.inf, 3) == "Inf"
    assert _fmt(-math.inf, 3) == "-Inf"
    # R's signif() rounds the scaled double half to even (fprec.c)
    assert _r_signif(0.1235, 3) == 0.124
    assert _r_signif(0.15, 1) == 0.2
    assert _r_signif(0.125, 2) == 0.12
    assert _format_r(95.0) == "95"
    assert _format_r(99.9) == "99.9"
    assert _format_r(-0.0) == "0"
    # formatC(round(abs(v)), big.mark = ",", format = "d")
    assert _fmt_int(2.5) == "2"
    assert _fmt_int(1234567.5) == "1,234,568"
    assert _fmt_int(-285, signed=True) == "-285"
    assert _fmt_int(285, signed=True) == "+285"
    assert _fmt_int(285) == "285"
    assert _fmt_int(math.nan) == "---"
    assert _fmt_int(math.inf) == "Inf"
