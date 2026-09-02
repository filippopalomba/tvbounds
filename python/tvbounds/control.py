"""Tuning-options constructor for the counterfactual solver, plus the helper
that merges user overrides of individual KNITRO options into the shipped
option files (port of ``R/control.R``)."""
from __future__ import annotations

import math
import os
import re
import tempfile

import numpy as np

from ._julia_setup import _julia_file


class TVBoundsControl(dict):
    """R's ``"tvbounds_control"`` list: a dict whose entries are also attributes."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

    def copy(self):
        return TVBoundsControl(self)

    def __repr__(self):
        return "TVBoundsControl(" + dict.__repr__(self) + ")"


def tvbounds_control(maxsolves=10,
                     startptrange=0.01,
                     use_optim=False,
                     time_limit=60,
                     iterations=100,
                     outer_iterations=3,
                     inner_opt=None,
                     outer_opt=None,
                     knitro_options={},  # noqa: B006 - R's `list()`; copied below
                     eta_min=1e-120,
                     lower_limit=-10,
                     psi_tv_eps=1e-4,
                     tvac_tau=1e-3,
                     tvmix_tau=1e-3,
                     purekl_acap=500,
                     tvmix_kappa=None):
    """See the tvbounds manual."""
    if knitro_options is None:
        knitro_options = {}
    _tvb_check_count(maxsolves, "maxsolves")
    _tvb_check_positive_scalar(startptrange, "startptrange")
    _tvb_check_flag(use_optim, "use_optim")
    _tvb_check_positive_scalar(time_limit, "time_limit")
    _tvb_check_count(iterations, "iterations")
    _tvb_check_count(outer_iterations, "outer_iterations")
    _tvb_check_opt_path(inner_opt, "inner_opt")
    _tvb_check_opt_path(outer_opt, "outer_opt")
    if not isinstance(knitro_options, dict):
        raise ValueError("`knitro_options` must be a named list.")
    if len(knitro_options) > 0:
        nms = list(knitro_options.keys())
        if any(not isinstance(nm, str) or nm == "" for nm in nms):
            raise ValueError("every entry of `knitro_options` must be named.")
        ok = all(_is_length_one_value(v) for v in knitro_options.values())
        if not ok:
            raise ValueError("entries of `knitro_options` must be length-one "
                             "character, numeric, or logical values.")
    _tvb_check_positive_scalar(eta_min, "eta_min")
    ll = _num_scalar(lower_limit)
    if ll is None or not math.isfinite(ll):
        raise ValueError("`lower_limit` must be a finite numeric scalar.")
    _tvb_check_positive_scalar(psi_tv_eps, "psi_tv_eps")
    _tvb_check_positive_scalar(tvac_tau, "tvac_tau")
    _tvb_check_positive_scalar(tvmix_tau, "tvmix_tau")
    _tvb_check_positive_scalar(purekl_acap, "purekl_acap")
    if tvmix_kappa is not None:
        kap = _num_scalar(tvmix_kappa)
        if kap is None or not math.isfinite(kap) or kap < 0 or kap > 1:
            raise ValueError("`tvmix_kappa` must be None or a scalar in [0, 1].")
    return TVBoundsControl(
        maxsolves=int(_num_scalar(maxsolves)),
        startptrange=float(_num_scalar(startptrange)),
        use_optim=bool(use_optim),
        time_limit=float(_num_scalar(time_limit)),
        iterations=int(_num_scalar(iterations)),
        outer_iterations=int(_num_scalar(outer_iterations)),
        inner_opt=inner_opt,
        outer_opt=outer_opt,
        knitro_options=dict(knitro_options),
        eta_min=float(_num_scalar(eta_min)),
        lower_limit=float(ll),
        psi_tv_eps=float(_num_scalar(psi_tv_eps)),
        tvac_tau=float(_num_scalar(tvac_tau)),
        tvmix_tau=float(_num_scalar(tvmix_tau)),
        purekl_acap=float(_num_scalar(purekl_acap)),
        tvmix_kappa=None if tvmix_kappa is None else float(_num_scalar(tvmix_kappa)),
    )


# --- small validators -------------------------------------------------

def _num_scalar(x):
    """``float(x)`` when ``x`` is what R calls a length-one numeric, else None.

    Booleans are not numeric (as in R), and a length-one list/array is
    accepted as R accepts a length-one vector.
    """
    if isinstance(x, (bool, np.bool_)):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return _num_scalar(x[0])            # R's c(0.6) is a length-one vector
    if isinstance(x, np.ndarray) and x.ndim <= 1 and x.size == 1 \
            and x.dtype.kind in "iuf":
        return float(x.reshape(-1)[0])
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return _num_scalar(x[0])
    return None


def _is_length_one_value(v):
    """A length-one character, numeric, or logical value (R's test)."""
    if isinstance(v, (str, bool, int, float, np.bool_, np.integer, np.floating)):
        return True
    if isinstance(v, np.ndarray) and v.size == 1 and v.dtype.kind in "biufU":
        return True
    return False


def _tvb_check_count(x, name):
    """Validate a positive integer-like scalar."""
    v = _num_scalar(x)
    if v is None or not math.isfinite(v) or v < 1 or v != int(v):
        raise ValueError("`%s` must be a positive integer." % name)
    return True


def _tvb_check_positive_scalar(x, name):
    """Validate a strictly positive finite scalar."""
    v = _num_scalar(x)
    if v is None or not math.isfinite(v) or v <= 0:
        raise ValueError("`%s` must be a positive finite scalar." % name)
    return True


def _tvb_check_flag(x, name):
    """Validate a length-one logical."""
    if not isinstance(x, (bool, np.bool_)):
        raise ValueError("`%s` must be True or False." % name)
    return True


def _tvb_check_opt_path(x, name):
    """Validate an optional KNITRO option-file path."""
    if x is None:
        return True
    if isinstance(x, os.PathLike):
        x = os.fspath(x)
    if not isinstance(x, str) or x == "":
        raise ValueError("`%s` must be None or a single file path." % name)
    if not os.path.exists(x):
        raise ValueError("`%s` points to a file that does not exist: %s"
                         % (name, x))
    return True


# --- R's as.character() for the option values ---------------------------

def _r_format_double(x):
    """R's ``as.character()`` of a double: up to 15 significant digits, fixed
    notation unless the scientific representation is narrower (R's
    ``formatReal`` with ``scipen = 0``)."""
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == 0:
        return "0"
    neg = x < 0
    ax = abs(x)
    mant, exp = ("%.14e" % ax).split("e")
    e10 = int(exp)
    digits = mant.replace(".", "").rstrip("0")
    nsig = max(len(digits), 1)
    sci_w = (1 if neg else 0) + nsig + (1 if nsig > 1 else 0) + \
        (4 if abs(e10) < 100 else 5)
    if e10 >= 0:
        rgt = max(nsig - e10 - 1, 0)
        fix_w = (1 if neg else 0) + (e10 + 1) + (rgt + 1 if rgt > 0 else 0)
    else:
        rgt = nsig - e10 - 1
        fix_w = (1 if neg else 0) + 2 + rgt
    if fix_w <= sci_w:
        return "%.*f" % (rgt, x)
    m = digits[0] + ("." + digits[1:] if nsig > 1 else "")
    return ("-" if neg else "") + m + "e" + ("-" if e10 < 0 else "+") + \
        "%02d" % abs(e10)


def _r_as_character(val):
    """R's ``as.character()`` of a length-one character/numeric/logical value."""
    if isinstance(val, (bool, np.bool_)):
        return "TRUE" if val else "FALSE"
    if isinstance(val, str):
        return val
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, np.ndarray):
        return _r_as_character(val.reshape(-1)[0].item())
    return _r_format_double(float(val))


def _tvb_merge_opt_file(base, overrides):
    """Merge user overrides into a KNITRO option file.

    Reads ``base``, replaces the value of every option named in ``overrides``
    (appending options not already present), and writes the merged file to
    ``tempfile.gettempdir()``. The base file is never modified.

    :param base: Path to an existing KNITRO ``.opt`` file.
    :param overrides: Dict of length-one character/numeric/logical values
        (logicals are written as 0/1).
    :return: The path of the merged temporary file.
    """
    if isinstance(base, os.PathLike):
        base = os.fspath(base)
    if not (isinstance(base, str) and os.path.exists(base)
            and isinstance(overrides, dict) and len(overrides) > 0):
        raise ValueError("_tvb_merge_opt_file(): `base` must be an existing "
                         "file and `overrides` a non-empty dict.")
    with open(base, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    for nm, val in overrides.items():
        if isinstance(val, (bool, np.bool_)):
            val = int(val)
        valstr = _r_as_character(val)
        pat = re.compile(r"^\s*" + nm + r"(\s|$)")
        repl = "%s %s" % (nm, valstr)
        hit = [bool(pat.search(ln)) for ln in lines]
        if any(hit):
            lines = [repl if h else ln for ln, h in zip(lines, hit)]
        else:
            lines.append(repl)
    fd, out = tempfile.mkstemp(prefix="tvbounds_", suffix=".opt",
                               dir=tempfile.gettempdir())
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return out


def _tvb_resolve_opt(control):
    """Resolve the inner/outer KNITRO option files for one call.

    Applies the shipped defaults when the control list does not name its own
    files, and merges ``knitro_options`` overrides into temporary copies when
    present.

    :param control: A :class:`TVBoundsControl`.
    :return: A dict with elements ``inner`` and ``outer`` (file paths).
    """
    inner = control["inner_opt"]
    if inner is None:
        inner = _julia_file("opt", "inner.opt")
    outer = control["outer_opt"]
    if outer is None:
        outer = _julia_file("opt", "outer.opt")
    if not inner or not os.path.exists(inner):
        raise RuntimeError("could not locate the inner KNITRO option file; "
                           "supply `inner_opt` in tvbounds_control().")
    if not outer or not os.path.exists(outer):
        raise RuntimeError("could not locate the outer KNITRO option file; "
                           "supply `outer_opt` in tvbounds_control().")
    if len(control["knitro_options"]) > 0:
        inner = _tvb_merge_opt_file(inner, control["knitro_options"])
        outer = _tvb_merge_opt_file(outer, control["knitro_options"])
    return {"inner": inner, "outer": outer}
