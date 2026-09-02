"""Tests for tvbounds_plot() and the plot() method (port of test-plot.R):
structure checks only, on objects with and without confidence bands.
The ggplot layer counts of the R tests become counts of matplotlib artists
(lines, collections, patches, texts) on the single axes of the figure."""
from __future__ import annotations

import math
import warnings

import matplotlib
import pytest
from matplotlib.figure import Figure

from tvbounds import tvbounds_plot
from helpers import (tvb_fixture_linear, tvb_fixture_censored,
                     tvb_fixture_noinf, tvb_fixture_counterfactual)


def n_artists(fig) -> int:
    """Number of drawn artists, the analogue of the built ggplot layers."""
    ax = fig.axes[0]
    return len(ax.lines) + len(ax.collections) + len(ax.patches) + len(ax.texts)


def _draw(fig):
    """The analogue of ggplot_build(): render on the Agg canvas."""
    fig.canvas.draw()
    return fig


def test_plot_returns_a_drawable_figure_for_objects_with_a_confidence_band():
    obj = tvb_fixture_linear()
    p = tvbounds_plot(obj)
    assert isinstance(p, Figure)
    _draw(p)
    assert len(p.axes) == 1


def test_bands_add_layers_only_when_ci_columns_exist():
    with_ci = tvb_fixture_linear()
    no_ci = tvb_fixture_noinf()

    p_bands = tvbounds_plot(with_ci, bands=True)
    p_nobands = tvbounds_plot(with_ci, bands=False)
    assert n_artists(p_bands) > n_artists(p_nobands)
    # two ribbons + two dashed outer limits
    assert n_artists(p_bands) - n_artists(p_nobands) == 4

    # bands = True on an object without CI columns is a no-op, not an error.
    p1 = tvbounds_plot(no_ci, bands=True)
    p2 = tvbounds_plot(no_ci, bands=False)
    assert n_artists(p1) == n_artists(p2)


def test_breakdown_and_baseline_marks_are_optional_and_skipped_when_not_drawable():
    obj = tvb_fixture_linear()
    p_all = tvbounds_plot(obj, breakdown=True, baseline=True)
    p_min = tvbounds_plot(obj, breakdown=False, baseline=False)
    assert n_artists(p_all) > n_artists(p_min)
    # vline + annotation + baseline point
    assert n_artists(p_all) - n_artists(p_min) == 3
    ax = p_all.axes[0]
    assert any(r"\hat{\delta}_b = 0.5" in t.get_text() for t in ax.texts)

    # Censored breakdown (no interior crossing): no vline layer, still draws.
    cens = tvb_fixture_censored()
    _draw(tvbounds_plot(cens))
    assert n_artists(tvbounds_plot(cens, baseline=False)) == \
        n_artists(tvbounds_plot(cens, breakdown=False, baseline=False))

    # A missing baseline point disables both marks without error.
    nopt = tvb_fixture_linear()
    nopt.point = math.nan
    _draw(tvbounds_plot(nopt))
    assert n_artists(tvbounds_plot(nopt)) == n_artists(tvbounds_plot(nopt, breakdown=False,
                                                                       baseline=False))


def test_log_x_handles_counterfactual_grids_exceeding_1():
    cf = tvb_fixture_counterfactual()
    p_lin = tvbounds_plot(cf)
    assert isinstance(p_lin, Figure)
    assert p_lin.axes[0].get_xscale() == "linear"

    # The grid contains delta = 0, which a log axis cannot show: dropped with
    # a warning, and the plot still draws.
    with pytest.warns(UserWarning, match="log_x"):
        p_log = tvbounds_plot(cf, log_x=True)
    assert isinstance(p_log, Figure)
    _draw(p_log)
    assert p_log.axes[0].get_xscale() == "log"

    # Without nonpositive budgets there is nothing to drop and no warning.
    cf_pos = tvb_fixture_counterfactual()
    cf_pos.bounds = cf_pos.bounds[cf_pos.bounds["delta"] > 0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tvbounds_plot(cf_pos, log_x=True)


def test_labels_default_to_the_budget_and_the_estimand_label():
    obj = tvb_fixture_noinf()
    p = tvbounds_plot(obj)
    _draw(p)
    ax = p.axes[0]
    assert ax.get_xlabel() == r"$\delta$"
    assert ax.get_ylabel() == "IV coefficient"
    assert ax.get_title() == ""
    p2 = tvbounds_plot(obj, xlab="budget", ylab="beta", title="sensitivity")
    _draw(p2)
    ax2 = p2.axes[0]
    assert ax2.get_xlabel() == "budget"
    assert ax2.get_ylabel() == "beta"
    assert ax2.get_title() == "sensitivity"
    assert ax2.title.get_fontweight() == "bold"
    assert ax2.get_legend() is None


def test_tau_star_moves_the_reference_line_without_error():
    obj = tvb_fixture_linear()
    p = _draw(tvbounds_plot(obj, tau_star=0.2))
    ax = p.axes[0]
    ys = [ln.get_ydata()[0] for ln in ax.lines if ln.get_linestyle() == "--"
          and len(set(map(float, ln.get_ydata()))) == 1]
    assert 0.2 in ys


def test_plot_method_dispatches_to_tvbounds_plot():
    obj = tvb_fixture_linear()
    p = obj.plot(bands=False)
    assert isinstance(p, Figure)


def test_plot_input_validation_fails_early():
    obj = tvb_fixture_linear()
    with pytest.raises(ValueError, match="tvbounds"):
        tvbounds_plot(42)
    with pytest.raises(ValueError, match="bands"):
        tvbounds_plot(obj, bands=None)
    with pytest.raises(ValueError, match="log_x"):
        tvbounds_plot(obj, log_x="yes")
    with pytest.raises(ValueError, match="tau_star"):
        tvbounds_plot(obj, tau_star=[0, 1])
    with pytest.raises(ValueError, match="color"):
        tvbounds_plot(obj, color=3)
    short = tvb_fixture_linear(delta=[0, 1])
    short.bounds.loc[1, "delta"] = math.nan
    with pytest.raises(ValueError, match="at least two budget values"):
        tvbounds_plot(short)
    # extra keyword arguments are accepted and ignored, as R's `...`
    assert isinstance(tvbounds_plot(obj, foo=1), Figure)


def test_backend_is_non_interactive_in_tests():
    assert matplotlib.get_backend().lower() == "agg"


def test_r_colour_names_are_accepted():
    """R colour names such as "gray50" (valid in R's graphics) draw as in R."""
    import matplotlib.pyplot as plt
    obj = tvb_fixture_linear()
    for colour in ("gray50", "steelblue3", "grey30", "transparent", "#1F4E79"):
        fig = tvbounds_plot(obj, color=colour)
        assert fig is not None
        plt.close(fig)
