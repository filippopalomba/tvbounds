"""tvbounds: Sensitivity Analysis and Bounds under Total Variation Neighborhoods.

Python companion package to Palomba (2026), "Sensitivity Analysis in
Population Shares": :func:`tvbounds_attrition`,
:func:`tvbounds_counterfactual`, :func:`tvbounds_riv`,
:func:`tvbounds_plot`, :func:`tvbounds_summary`, and
:func:`tvbounds_control`.
"""
import os as _os

from ._object import (TVBounds, TVBoundsSummary, BudgetValue, new_tvbounds,
                      is_tvbounds)
from .control import TVBoundsControl, tvbounds_control
from .attrition import tvbounds_attrition
from .riv import tvbounds_riv
from .counterfactual import tvbounds_counterfactual
from .summary import tvbounds_summary
from .plot import tvbounds_plot
from ._julia_setup import tvb_julia_available
from ._docs import DOCS as _DOCS

__version__ = "0.1.2"


def julia_file(*parts):
    """Path of a file shipped in the package's ``julia/`` directory.

    The Python counterpart of R's ``system.file("julia", ..., package = "tvbounds")``:
    ``julia_file("examples", "toy.jl")`` is the toy moments example and
    ``julia_file("opt", "outer_fast.opt")`` a shipped KNITRO option file.
    """
    return _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "julia", *parts)


__all__ = [
    "tvbounds_attrition", "tvbounds_counterfactual", "tvbounds_riv",
    "tvbounds_plot", "tvbounds_summary", "tvbounds_control",
    "TVBounds", "TVBoundsSummary", "TVBoundsControl", "BudgetValue",
    "new_tvbounds", "is_tvbounds", "tvb_julia_available", "julia_file",
    "__version__",
]

# Attach the generated manual pages as the public docstrings.
for _name, _obj in (("tvbounds_attrition", tvbounds_attrition),
                    ("tvbounds_counterfactual", tvbounds_counterfactual),
                    ("tvbounds_riv", tvbounds_riv),
                    ("tvbounds_plot", tvbounds_plot),
                    ("tvbounds_summary", tvbounds_summary),
                    ("tvbounds_control", tvbounds_control),
                    ("print.tvbounds", TVBounds.print),
                    ("print.tvbounds_summary", TVBoundsSummary.print)):
    _text = _DOCS.get(_name)
    if _text:
        try:
            _obj.__doc__ = _text
        except (AttributeError, TypeError):
            pass
if _DOCS.get("tvbounds-package"):
    __doc__ = _DOCS["tvbounds-package"]
del _name, _obj, _text
