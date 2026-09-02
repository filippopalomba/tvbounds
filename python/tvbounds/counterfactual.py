"""Counterfactual predictions in structural models: sensitivity bounds
through the Julia/KNITRO backend (TVBoundsJulia). Port of
``R/counterfactual.R``."""
from __future__ import annotations

import inspect
import math
import os
import re

import numpy as np
import pandas as pd

from ._object import NA, format_call, new_tvbounds, warn
from ._julia_setup import _julia_main, _tvb_julia_setup
from .control import (TVBoundsControl, _tvb_check_flag, _tvb_resolve_opt,
                      tvbounds_control)

_DIVERGENCES = ("KL_chi2", "KL", "chi2", "TV", "TVmix", "TVmixC", "TVac")
_SIDES = ("both", "lower", "upper")


def _match_arg(value, choices, name):
    """R's ``match.arg()`` for a single string (exact or unique prefix)."""
    if value is None:
        return choices[0]                  # match.arg(NULL) returns the first choice
    if isinstance(value, (list, tuple)) and tuple(value) == tuple(choices):
        return choices[0]
    if isinstance(value, str):
        if value in choices:
            return value
        hits = [c for c in choices if c.startswith(value)] if value else []
        if len(hits) == 1:
            return hits[0]
    raise ValueError("'%s' should be one of %s" %
                     (name, ", ".join('"%s"' % c for c in choices)))


def _is_numeric_vector(x):
    """R's ``is.numeric()`` for the vector arguments (no booleans/strings)."""
    if isinstance(x, (bool, np.bool_, str, bytes)):
        return False
    if isinstance(x, (int, float, np.integer, np.floating)):
        return True
    try:
        a = np.asarray(x)
    except Exception:
        return False
    if a.dtype == object:
        return a.size > 0 and all(
            isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, (bool, np.bool_)) for v in a.reshape(-1))
    return a.dtype.kind in "iuf"


def _as_vector(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _is_int_scalar(x):
    """A length-one numeric that equals its integer part (R's ``x == as.integer(x)``)."""
    if isinstance(x, (bool, np.bool_)):
        return False
    if isinstance(x, (int, np.integer)):
        return True
    if isinstance(x, (float, np.floating)):
        return math.isfinite(x) and float(x) == int(x)
    if isinstance(x, np.ndarray) and x.size == 1 and x.dtype.kind in "iuf":
        return _is_int_scalar(x.reshape(-1)[0].item())
    return False


def tvbounds_counterfactual(moments, d, theta_lb, theta_ub, delta,
                            divergence="KL_chi2", side="both",
                            U=None, M=50000, u_dim=None, gamma=None,
                            gradient=None, theta_init=None, control=None,
                            seed=None, verbose=True):
    """See the tvbounds manual."""
    cl = format_call("tvbounds_counterfactual", {
        "moments": moments, "d": d, "theta_lb": theta_lb,
        "theta_ub": theta_ub, "delta": delta, "divergence": divergence,
        "side": side, "U": U, "M": M, "u_dim": u_dim, "gamma": gamma,
        "gradient": gradient, "theta_init": theta_init, "control": control,
        "seed": seed, "verbose": verbose})
    divergence = _match_arg(divergence, _DIVERGENCES, "divergence")
    side = _match_arg(side, _SIDES, "side")
    _tvb_check_flag(verbose, "verbose")
    if control is None:
        control = tvbounds_control()

    # ---- moments specification (pure Python, testable without Julia) ----
    spec = _tvb_parse_moments(moments)

    # ---- dimensions and boxes ------------------------------------------
    if not _is_numeric_vector(d) or np.size(d) != 1 or \
            not math.isfinite(float(np.reshape(d, -1)[0])) or \
            float(np.reshape(d, -1)[0]) < 1 or not _is_int_scalar(np.reshape(d, -1)[0].item()):
        raise ValueError("`d` must be a positive integer (the number of moment "
                         "conditions).")
    d = int(np.reshape(d, -1)[0])
    if not _is_numeric_vector(theta_lb) or not _is_numeric_vector(theta_ub) or \
            np.size(theta_lb) == 0 or np.size(theta_lb) != np.size(theta_ub):
        raise ValueError("`theta_lb` and `theta_ub` must be numeric vectors of "
                         "equal, positive length.")
    theta_lb = _as_vector(theta_lb)
    theta_ub = _as_vector(theta_ub)
    if np.any(~np.isfinite(theta_lb)) or np.any(~np.isfinite(theta_ub)):
        raise ValueError("`theta_lb` and `theta_ub` must be finite.")
    if np.any(theta_lb > theta_ub):
        raise ValueError("`theta_lb` must not exceed `theta_ub` in any "
                         "coordinate.")
    l = theta_lb.size
    theta_init_supplied = theta_init is not None
    if theta_init is None:
        theta_init = (theta_lb + theta_ub) / 2
    if not _is_numeric_vector(theta_init) or np.size(theta_init) != l or \
            np.any(~np.isfinite(_as_vector(theta_init))):
        raise ValueError("`theta_init` must be a finite numeric vector of "
                         "length %d." % l)
    theta_init = _as_vector(theta_init)
    if np.any(theta_init < theta_lb) or np.any(theta_init > theta_ub):
        raise ValueError("`theta_init` must lie inside the [theta_lb, theta_ub] "
                         "box.")

    # ---- budgets -------------------------------------------------------
    if not _is_numeric_vector(delta) or np.size(delta) == 0 or \
            np.any(~np.isfinite(_as_vector(delta))):
        raise ValueError("`delta` must be a non-empty numeric vector of finite "
                         "budgets.")
    delta = _as_vector(delta)
    if np.any(delta <= 0):
        raise ValueError("all budgets in `delta` must be strictly positive; the "
                         "budget-zero baseline is reported separately as the "
                         "`point` field (the plug-in counterfactual at "
                         "`theta_init`).")
    if np.unique(delta).size != delta.size:
        raise ValueError("`delta` must not contain duplicated budget values.")
    tv_family = divergence in ("TV", "TVmix", "TVmixC", "TVac")
    if tv_family and np.any(delta > 1):
        raise ValueError('for divergence "%s" all budgets must lie in (0, 1].'
                         % divergence)
    delta = np.sort(delta.astype(np.float64))

    # ---- latent draws --------------------------------------------------
    if U is not None:
        Ua = None
        if isinstance(U, pd.DataFrame):
            Ua = U.to_numpy()
        elif isinstance(U, np.ndarray):
            Ua = U
        if Ua is None or Ua.ndim != 2 or Ua.dtype.kind not in "iuf" or \
                not np.all(np.isfinite(Ua)):
            raise ValueError("`U` must be a finite numeric matrix (M x u_dim), "
                             "or None.")
        if Ua.shape[0] < 2:
            raise ValueError("`U` must have at least 2 rows.")
        if u_dim is not None and (np.size(u_dim) != 1 or
                                  float(np.reshape(u_dim, -1)[0]) != Ua.shape[1]):
            raise ValueError("`u_dim` disagrees with ncol(U); when `U` is "
                             "supplied, its dimensions are used and `M`/`u_dim` "
                             "need not be given.")
        U = np.ascontiguousarray(Ua, dtype=np.float64)
        M = U.shape[0]
        u_dim = U.shape[1]
    else:
        if u_dim is None:
            raise ValueError("`u_dim` is required when `U` is None (the "
                             "dimension of the latent draw for the "
                             "scrambled-Halton generator).")
        if not _is_numeric_vector(u_dim) or np.size(u_dim) != 1 or \
                not math.isfinite(float(np.reshape(u_dim, -1)[0])) or \
                float(np.reshape(u_dim, -1)[0]) < 1 or \
                not _is_int_scalar(np.reshape(u_dim, -1)[0].item()):
            raise ValueError("`u_dim` must be a positive integer.")
        if float(np.reshape(u_dim, -1)[0]) > 15:
            raise ValueError("the scrambled-Halton generator supports u_dim <= 15 "
                             "(one prime base per dimension); supply `U` "
                             "directly for higher-dimensional draws.")
        if not _is_numeric_vector(M) or np.size(M) != 1 or \
                not math.isfinite(float(np.reshape(M, -1)[0])) or \
                float(np.reshape(M, -1)[0]) < 2 or \
                not _is_int_scalar(np.reshape(M, -1)[0].item()):
            raise ValueError("`M` must be an integer >= 2.")
    M = int(np.reshape(M, -1)[0])
    u_dim = int(np.reshape(u_dim, -1)[0])

    # ---- payload, gradient, control, seed ------------------------------
    if gamma is not None and not isinstance(gamma, dict):
        raise ValueError("`gamma` must be None or a dict.")
    grad_spec = _tvb_parse_gradient(gradient, spec)
    if not isinstance(control, TVBoundsControl):
        raise ValueError("`control` must be created by tvbounds_control().")
    if control["tvmix_kappa"] is not None:
        if divergence not in ("TVmix", "TVmixC"):
            warn('`control$tvmix_kappa` is ignored for divergence "%s".'
                 % divergence)
        elif divergence == "TVmix" and \
                float(control["tvmix_kappa"]) < 1 - float(np.min(delta)):
            raise ValueError(
                'for divergence "TVmix" the mixing weight `tvmix_kappa` must '
                "satisfy kappa >= 1 - delta at every budget (the reduced "
                "program requires the total-variation constraint to be "
                'redundant); use divergence "TVmixC" for kappa < 1 - delta.')
    if seed is not None:
        if not _is_numeric_vector(seed) or np.size(seed) != 1 or \
                not math.isfinite(float(np.reshape(seed, -1)[0])) or \
                not _is_int_scalar(np.reshape(seed, -1)[0].item()):
            raise ValueError("`seed` must be None or a single integer.")
        seed = int(np.reshape(seed, -1)[0])

    # ---- KNITRO option files -------------------------------------------
    opts = _tvb_resolve_opt(control)

    # ---- local RNG ------------------------------------------------------
    # R scopes its own RNG with withr::local_seed()/local_preserve_seed();
    # the scrambled-Halton draws and the multi-start perturbations are
    # generated by Julia's RNG from `seed`, so nothing touches Python's or
    # NumPy's random state here.

    # ---- Julia session (lazy; once per Python session) ------------------
    _tvb_julia_setup(verbose=verbose)
    jl = _julia_main()

    # ---- put the moments (and gradient) in place in Julia --------------
    moments_name = _tvb_stage_moments(spec, gamma)
    jac_name = _tvb_stage_gradient(grad_spec, gamma)

    # ---- solve ---------------------------------------------------------
    args = {
        "moments_name": moments_name,
        "delta": np.asarray(delta, dtype=np.float64),
        "d": int(d),
        "theta_lb": np.asarray(theta_lb, dtype=np.float64),
        "theta_ub": np.asarray(theta_ub, dtype=np.float64),
        "theta_init": np.asarray(theta_init, dtype=np.float64),
        "inner_opt": opts["inner"],
        "outer_opt": opts["outer"],
        "U": U,
        "M": int(M),
        "u_dim": int(u_dim),
        "seed": seed,
        # the payload is curried on the Python side for Python moments
        "gamma": None if spec["type"] == "pyfun" else gamma,
        "divergence": divergence,
        "side": side,
        "jac_mode": grad_spec["mode"],
        "jac_name": jac_name,
        "maxsolves": int(control["maxsolves"]),
        "startptrange": float(control["startptrange"]),
        "use_optim": bool(control["use_optim"]),
        "time_limit": float(control["time_limit"]),
        "iterations": int(control["iterations"]),
        "outer_iterations": int(control["outer_iterations"]),
        "lower_limit": float(control["lower_limit"]),
        "eta_min": float(control["eta_min"]),
        "psi_tv_eps": float(control["psi_tv_eps"]),
        "tvac_tau": float(control["tvac_tau"]),
        "tvmix_tau": float(control["tvmix_tau"]),
        "purekl_acap": float(control["purekl_acap"]),
        "tvmix_kappa": (math.nan if control["tvmix_kappa"] is None
                        else float(control["tvmix_kappa"])),
        "verbose": bool(verbose),
    }
    try:
        res = jl.TVBoundsPyBridge.tvb_solve_py(args)
    except Exception as e:  # juliacall.JuliaError
        raise RuntimeError("the Julia solver failed: %s" % e) from e

    # ---- assemble the common return object -----------------------------
    def clean(x):
        x = np.array(x, dtype=np.float64).reshape(-1)
        x[~np.isfinite(x) | (np.abs(x) >= 1e9)] = NA
        return x

    def ints(x):
        return np.array(x).reshape(-1).astype(np.int64)

    delta_out = np.array(res["delta"], dtype=np.float64).reshape(-1)
    bounds = pd.DataFrame({
        "delta": delta_out,
        "lower": clean(res["lower"]),
        "upper": clean(res["upper"]),
    })
    solver = pd.DataFrame({
        "delta": delta_out,
        "status_outer_lower": ints(res["status_outer_lower"]),
        "status_inner_lower": ints(res["status_inner_lower"]),
        "time_lower": clean(res["time_lower"]),
        "status_outer_upper": ints(res["status_outer_upper"]),
        "status_inner_upper": ints(res["status_inner_upper"]),
        "time_upper": clean(res["time_upper"]),
    })
    control_resolved = control.copy()
    control_resolved["inner_opt"] = opts["inner"]
    control_resolved["outer_opt"] = opts["outer"]

    nd = delta_out.size
    details = {
        "side": side,
        "M": int(res["M"]),
        "u_dim": int(res["u_dim"]),
        "theta_lb": theta_lb,
        "theta_ub": theta_ub,
        "theta_init": theta_init,
        "theta_init_supplied": theta_init_supplied,
        "fixed_theta": bool(res["fixed_theta"]),
        "solver": solver,
        "theta_lower": np.array(res["theta_lower"], dtype=np.float64).reshape(l, nd, order="F"),
        "theta_upper": np.array(res["theta_upper"], dtype=np.float64).reshape(l, nd, order="F"),
        "control": control_resolved,
        "moments": spec["label"],
        "gradient": grad_spec["mode"],
        "seed": seed,
    }

    point = float(np.reshape(np.asarray(res["point"], dtype=np.float64), -1)[0])
    return new_tvbounds(
        application="counterfactual",
        bounds=bounds,
        point=(point if theta_init_supplied and math.isfinite(point) else NA),
        n=NA,
        neighborhood=None,
        divergence=divergence,
        level=NA,
        B=NA,
        estimand_label="counterfactual prediction",
        call=cl,
        details=details,
    )


# ---------------------------------------------------------------------
# Moments / gradient specification parsing (pure Python; no Julia needed).
# ---------------------------------------------------------------------

_JULIA_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_!]*$")


def _tvb_is_julia_name(x):
    """Is a string a plausible Julia identifier?

    Restricting Julia names to this pattern also keeps the interpolated
    ``isdefined`` checks free of code injection.
    """
    return isinstance(x, str) and _JULIA_NAME.match(x) is not None


def _n_params(fn):
    """Number of positional parameters of a callable (R's ``length(formals())``);
    a ``*args`` parameter counts as unlimited."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return 3
    n = 0
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            return 3
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD):
            n += 1
    return n


def _tvb_parse_moments(moments):
    """Parse the ``moments`` argument of tvbounds_counterfactual().

    :param moments: A ``(file, fname)`` pair, a single string naming a Julia
        function, or a Python callable.
    :return: A dict with elements ``type`` (``"file"``, ``"name"``, or
        ``"pyfun"``), ``file``, ``name``, ``fun``, and a human-readable
        ``label``.
    """
    if callable(moments) and not isinstance(moments, (str, bytes)):
        if _n_params(moments) < 2:
            raise ValueError("a Python `moments` function must accept (theta, "
                             "U, gamma) (at least theta and U).")
        return {"type": "pyfun", "file": None, "name": None, "fun": moments,
                "label": "Python function"}
    if isinstance(moments, (tuple, list)) and len(moments) == 2 and \
            all(isinstance(m, (str, os.PathLike)) for m in moments):
        file = os.fspath(moments[0])
        name = moments[1]
        if not os.path.exists(file):
            raise ValueError("the Julia moments file does not exist: %s" % file)
        if not _tvb_is_julia_name(name):
            raise ValueError("`moments[1]` must be a valid Julia function name.")
        return {"type": "file", "file": file, "name": name, "fun": None,
                "label": "Julia function %s (from %s)"
                         % (name, os.path.basename(file))}
    if isinstance(moments, str):
        if not _tvb_is_julia_name(moments):
            raise ValueError("`moments` must be a valid Julia function name.")
        return {"type": "name", "file": None, "name": moments, "fun": None,
                "label": "Julia function %s" % moments}
    raise ValueError("`moments` must be a Python function, a single Julia "
                     "function name, or a length-2 tuple (file, fname).")


def _tvb_parse_gradient(gradient, spec):
    """Parse the ``gradient`` argument of tvbounds_counterfactual().

    :param gradient: ``None``, the string ``"fd"``, the name of a Julia
        function, or a Python callable.
    :param spec: The parsed moments specification.
    :return: A dict with elements ``mode`` (``"forwarddiff"``, ``"fd"``, or
        ``"user"``), ``julia_name`` (or ``None``), ``fun`` (or ``None``).
    """
    if gradient is None:
        if spec["type"] == "pyfun":
            # ForwardDiff cannot differentiate through Python code.
            return {"mode": "fd", "julia_name": None, "fun": None}
        return {"mode": "forwarddiff", "julia_name": None, "fun": None}
    if callable(gradient) and not isinstance(gradient, (str, bytes)):
        return {"mode": "user", "julia_name": None, "fun": gradient}
    if isinstance(gradient, str):
        if gradient == "fd":
            return {"mode": "fd", "julia_name": None, "fun": None}
        if spec["type"] == "pyfun":
            raise ValueError("with a Python `moments` function, `gradient` "
                             "must be None, \"fd\", or a Python function.")
        if not _tvb_is_julia_name(gradient):
            raise ValueError("`gradient` must be None, \"fd\", the name of a "
                             "Julia function, or a Python function.")
        return {"mode": "user", "julia_name": gradient, "fun": None}
    raise ValueError("`gradient` must be None, \"fd\", the name of a Julia "
                     "function, or a Python function.")


# ---------------------------------------------------------------------
# Julia staging helpers (require an initialized Julia session).
# ---------------------------------------------------------------------

def _wrap_kg_result(res):
    """Normalize a Python moments result into a dict of float64 arrays
    ``K`` (shape (M,)) and ``G`` (shape (M, d)) for the Julia bridge.

    A ``G`` given as a flat vector of length ``M * d`` is read column by
    column (as R's ``copyto!`` of a vector does); a 2-d ``G`` must already
    have the ``(M, d)`` orientation.
    """
    K = G = None
    if isinstance(res, dict):
        K = res.get("K")
        G = res.get("G")
    elif isinstance(res, (tuple, list)) and len(res) == 2:
        K, G = res
    if K is None or G is None:
        raise ValueError("the moments function must return a dict with "
                         "components `K` (length M) and `G` (M x d); got %s"
                         % type(res).__name__)
    K = np.asarray(K, dtype=np.float64).reshape(-1)
    G = np.asarray(G, dtype=np.float64)
    if G.ndim == 2 and G.shape[0] != K.size and G.size % K.size == 0:
        raise ValueError("moments component `G` has shape %s; expected (M, d) "
                         "with M = %d rows." % (G.shape, K.size))
    if G.ndim != 2:
        if K.size == 0 or G.size % K.size != 0:
            raise ValueError("moments component `G` has %d elements; expected "
                             "M x d." % G.size)
        G = G.reshape((K.size, G.size // K.size), order="F")
    return {"K": K, "G": np.ascontiguousarray(G)}


def _wrap_jac_result(raw):
    """Normalize a Python gradient result for the Julia bridge: a dict / pair
    with ``K`` and ``G`` becomes a dict of float64 arrays, anything else
    the float64 stacked Jacobian."""
    if isinstance(raw, dict):
        if raw.get("K") is None or raw.get("G") is None:
            raise ValueError("user gradient must return either the stacked "
                             "(M*(d+1)) x l Jacobian or a collection with "
                             "components `K` (M x l) and `G` (M x d x l); got "
                             "a dict with keys %s" % sorted(map(str, raw.keys())))
        return {"K": np.asarray(raw.get("K"), dtype=np.float64),
                "G": np.asarray(raw.get("G"), dtype=np.float64)}
    if isinstance(raw, (tuple, list)) and len(raw) == 2 and \
            not isinstance(raw[0], (int, float)):
        return {"K": np.asarray(raw[0], dtype=np.float64),
                "G": np.asarray(raw[1], dtype=np.float64)}
    return np.asarray(raw, dtype=np.float64)


def _curry(fn, gamma):
    """R's ``function(theta, U) fn(theta, U, gamma)``: the payload is fixed
    on the Python side so it never crosses the bridge; ``theta`` arrives as
    a fresh float64 vector and ``U`` as a read-only (M x u_dim) view of
    the Bundle's draws."""
    with_gamma = _n_params(fn) >= 3

    def prepare(theta, U):
        th = np.array(theta, dtype=np.float64).reshape(-1)
        Um = np.asarray(U)
        try:
            Um.flags.writeable = False
        except (ValueError, AttributeError):
            pass
        return th, Um

    if with_gamma:
        def wrapped(theta, U):
            th, Um = prepare(theta, U)
            return fn(th, Um, gamma)
    else:
        def wrapped(theta, U):
            th, Um = prepare(theta, U)
            return fn(th, Um)
    return wrapped


def _tvb_stage_moments(spec, gamma):
    """Stage the moments function in the Julia session.

    Includes the user's file and/or checks that the named function is
    defined; for Python moments, assigns the (gamma-currying) closure to
    Julia and wraps it into a Bundle-compatible in-place function.

    :return: The name (in ``Main``) of the Julia-side moments function.
    """
    jl = _julia_main()
    if spec["type"] == "file":
        jl._tvb_moments_file = os.path.normpath(os.path.abspath(spec["file"]))
        jl.seval("Base.include(Main, Main._tvb_moments_file);")
    if spec["type"] in ("file", "name"):
        ok = bool(jl.seval('isdefined(Main, Symbol("%s"))' % spec["name"]))
        if not ok:
            raise ValueError(
                "the Julia function `%s` is not defined in Main%s." % (
                    spec["name"],
                    (" after including %s (it must be defined at the top level)"
                     % os.path.basename(spec["file"]))
                    if spec["type"] == "file" else ""))
        return spec["name"]
    # Python function: curry gamma on the Python side so the payload never
    # crosses the bridge, then wrap into the in-place Bundle signature in
    # Julia.
    fn = spec["fun"]
    curried = _curry(fn, gamma)

    def wrapped(theta, U):
        return _wrap_kg_result(curried(theta, U))

    jl._tvb_py_moments_fun = wrapped
    jl.seval("_tvb_moments_pybridge = Main.TVBoundsPyBridge."
             "make_py_moments_wrapper(Main._tvb_py_moments_fun);")
    return "_tvb_moments_pybridge"


def _tvb_stage_gradient(grad_spec, gamma):
    """Stage the user gradient (if any) in the Julia session.

    :return: The name (in ``Main``) of the Julia-side Jacobian function, or
        ``""`` when the gradient mode needs none.
    """
    if grad_spec["mode"] != "user":
        return ""
    jl = _julia_main()
    if grad_spec["julia_name"] is not None:
        ok = bool(jl.seval('isdefined(Main, Symbol("%s"))'
                           % grad_spec["julia_name"]))
        if not ok:
            raise ValueError("the Julia gradient function `%s` is not defined "
                             "in Main." % grad_spec["julia_name"])
        return grad_spec["julia_name"]
    fn = grad_spec["fun"]
    curried = _curry(fn, gamma)

    def wrapped(theta, U):
        return _wrap_jac_result(curried(theta, U))

    jl._tvb_py_gradient_fun = wrapped
    jl.seval("_tvb_jac_pybridge = Main.TVBoundsPyBridge."
             "make_py_jac_wrapper(Main._tvb_py_gradient_fun);")
    return "_tvb_jac_pybridge"
