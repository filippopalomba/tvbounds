"""pytest configuration shared by the tvbounds test files.

Makes the package importable from source regardless of the working
directory, forces the non-interactive matplotlib backend, and closes the
figures created by each test so that they do not accumulate.
"""
from __future__ import annotations

import os
import sys

import matplotlib
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# The shared objects of helper-objects.R, also exposed as pytest fixtures.
@pytest.fixture
def fixture_linear():
    from helpers import tvb_fixture_linear
    return tvb_fixture_linear()


@pytest.fixture
def fixture_linear_upper():
    from helpers import tvb_fixture_linear_upper
    return tvb_fixture_linear_upper()


@pytest.fixture
def fixture_censored():
    from helpers import tvb_fixture_censored
    return tvb_fixture_censored()


@pytest.fixture
def fixture_noinf():
    from helpers import tvb_fixture_noinf
    return tvb_fixture_noinf()


@pytest.fixture
def fixture_counterfactual():
    from helpers import tvb_fixture_counterfactual
    return tvb_fixture_counterfactual()


@pytest.fixture
def fixture_quadratic():
    from helpers import tvb_fixture_quadratic
    return tvb_fixture_quadratic()
