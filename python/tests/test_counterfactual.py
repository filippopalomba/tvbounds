"""Tests for tvbounds_counterfactual(), tvbounds_control(), and the Julia
bridge (port of tests/testthat/test-counterfactual.R). Pure-Python argument
handling is tested unconditionally; everything that touches Julia/KNITRO is
skipped when tvb_julia_available() is False (which returns False, without
error, when juliacall, Julia, or a licensed KNITRO installation is missing).
"""
import math
import os
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

from tvbounds import (TVBounds, TVBoundsControl, tvb_julia_available,
                      tvbounds_control, tvbounds_counterfactual)
from tvbounds._julia_setup import _julia_file
from tvbounds._rrng import RRNG
from tvbounds.control import _tvb_merge_opt_file, _tvb_resolve_opt
from tvbounds.counterfactual import _tvb_parse_gradient, _tvb_parse_moments


def _skip_unless_julia():
    if not tvb_julia_available():
        pytest.skip("Julia/KNITRO backend not available")


def _rng_state():
    st = np.random.get_state()
    return (st[0], st[1].copy(), st[2], st[3], st[4])


def _same_state(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


# ---------------------------------------------------------------------
# tvbounds_control(): defaults and validation (pure Python)
# ---------------------------------------------------------------------

def test_control_returns_the_documented_defaults():
    ctrl = tvbounds_control()
    assert isinstance(ctrl, TVBoundsControl)
    assert ctrl["maxsolves"] == 10 and isinstance(ctrl["maxsolves"], int)
    assert ctrl.maxsolves == 10
    assert ctrl["startptrange"] == 0.01
    assert ctrl["use_optim"] is False
    assert ctrl["time_limit"] == 60
    assert ctrl["iterations"] == 100 and isinstance(ctrl["iterations"], int)
    assert ctrl["outer_iterations"] == 3 and isinstance(ctrl["outer_iterations"], int)
    assert ctrl["inner_opt"] is None
    assert ctrl["outer_opt"] is None
    assert ctrl["knitro_options"] == {}
    assert ctrl["eta_min"] == 1e-120
    assert ctrl["lower_limit"] == -10
    assert ctrl["psi_tv_eps"] == 1e-4
    assert ctrl["tvac_tau"] == 1e-3
    assert ctrl["tvmix_tau"] == 1e-3
    assert ctrl["purekl_acap"] == 500
    assert ctrl["tvmix_kappa"] is None


def test_control_validates_its_arguments():
    with pytest.raises(ValueError, match="maxsolves"):
        tvbounds_control(maxsolves=0)
    with pytest.raises(ValueError, match="maxsolves"):
        tvbounds_control(maxsolves=2.5)
    with pytest.raises(ValueError, match="startptrange"):
        tvbounds_control(startptrange=-1)
    with pytest.raises(ValueError, match="use_optim"):
        tvbounds_control(use_optim=None)
    with pytest.raises(ValueError, match="time_limit"):
        tvbounds_control(time_limit=0)
    with pytest.raises(ValueError, match="iterations"):
        tvbounds_control(iterations=-3)
    with pytest.raises(ValueError, match="does not exist"):
        tvbounds_control(inner_opt="no/such/file.opt")
    with pytest.raises(ValueError, match="named"):
        tvbounds_control(knitro_options=[1, 2])
    with pytest.raises(ValueError, match="named"):
        tvbounds_control(knitro_options={"": 1})
    with pytest.raises(ValueError, match="length-one"):
        tvbounds_control(knitro_options={"maxit": [1, 2]})
    with pytest.raises(ValueError, match="named list"):
        tvbounds_control(knitro_options="maxit 5")
    with pytest.raises(ValueError, match="eta_min"):
        tvbounds_control(eta_min=0)
    with pytest.raises(ValueError, match="lower_limit"):
        tvbounds_control(lower_limit=math.inf)
    with pytest.raises(ValueError, match="psi_tv_eps"):
        tvbounds_control(psi_tv_eps=0)
    with pytest.raises(ValueError, match="tvmix_kappa"):
        tvbounds_control(tvmix_kappa=1.5)
    with pytest.raises(ValueError, match="tvmix_kappa"):
        tvbounds_control(tvmix_kappa=-0.1)


def test_merge_opt_file_overrides_and_appends_options(tmp_path, monkeypatch):
    # the merged copies go to tempfile.gettempdir(): point it at tmp_path
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    base = tmp_path / "base.opt"
    base.write_text("# comment\nmaxit  3000\noutlev 0\nfeastol 1e-08\n")
    merged = _tvb_merge_opt_file(
        str(base), {"maxit": 500, "outlev": True, "newopt": "yes"})
    assert os.path.exists(merged)
    assert os.path.basename(merged).startswith("tvbounds_")
    assert merged.endswith(".opt")
    assert os.path.dirname(merged) == str(tmp_path)
    lines = open(merged).read().splitlines()
    assert "maxit 500" in lines
    assert "outlev 1" in lines          # logical written as 0/1
    assert "newopt yes" in lines        # appended
    assert "feastol 1e-08" in lines     # untouched
    assert "# comment" in lines         # comments preserved
    # the base file is never modified
    assert base.read_text().splitlines()[1] == "maxit  3000"
    # names that prefix other names are not clobbered
    merged2 = _tvb_merge_opt_file(str(base), {"feastol_abs": "1e-3"})
    lines2 = open(merged2).read().splitlines()
    assert "feastol 1e-08" in lines2
    assert "feastol_abs 1e-3" in lines2
    # numeric values are written as R's as.character() writes them
    merged3 = _tvb_merge_opt_file(
        str(base), {"opttol": 1e-4, "ftol": 0.001, "maxtime_real": 300.0,
                    "big": 100000, "xtol": 1e-12})
    lines3 = open(merged3).read().splitlines()
    assert "opttol 1e-04" in lines3
    assert "ftol 0.001" in lines3
    assert "maxtime_real 300" in lines3
    assert "big 100000" in lines3
    assert "xtol 1e-12" in lines3


def test_resolve_opt_uses_the_shipped_defaults_and_merges(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    opts = _tvb_resolve_opt(tvbounds_control())
    assert os.path.basename(opts["inner"]) == "inner.opt"
    assert os.path.basename(opts["outer"]) == "outer.opt"
    assert os.path.exists(opts["inner"]) and os.path.exists(opts["outer"])
    opts2 = _tvb_resolve_opt(tvbounds_control(knitro_options={"maxit": 7}))
    assert os.path.basename(opts2["inner"]).startswith("tvbounds_")
    assert "maxit 7" in open(opts2["outer"]).read().splitlines()


# ---------------------------------------------------------------------
# Moments / gradient specification parsing (pure Python)
# ---------------------------------------------------------------------

def test_parse_moments_accepts_the_three_documented_forms(tmp_path):
    f = tmp_path / "mom.jl"
    f.write_text("f() = 1\n")

    s1 = _tvb_parse_moments((str(f), "my_moments!"))
    assert s1["type"] == "file"
    assert s1["name"] == "my_moments!"
    assert s1["label"] == "Julia function my_moments! (from mom.jl)"

    s2 = _tvb_parse_moments("my_moments!")
    assert s2["type"] == "name"
    assert s2["label"] == "Julia function my_moments!"

    s3 = _tvb_parse_moments(lambda theta, U, gamma: None)
    assert s3["type"] == "pyfun"
    assert s3["label"] == "Python function"

    with pytest.raises(ValueError, match="does not exist"):
        _tvb_parse_moments(("no/such/file.jl", "g"))
    with pytest.raises(ValueError, match="valid Julia"):
        _tvb_parse_moments((str(f), "not a name"))
    with pytest.raises(ValueError, match="valid Julia"):
        _tvb_parse_moments("bad name")
    with pytest.raises(ValueError, match="length-2"):
        _tvb_parse_moments(("a", "b", "c"))
    with pytest.raises(ValueError, match="moments"):
        _tvb_parse_moments(42)
    with pytest.raises(ValueError, match="theta, U"):
        _tvb_parse_moments(lambda theta: None)


def test_parse_gradient_resolves_the_differentiation_mode():
    pyfun_spec = {"type": "pyfun"}
    julia_spec = {"type": "name"}

    assert _tvb_parse_gradient(None, julia_spec)["mode"] == "forwarddiff"
    # Python moments cannot be differentiated by ForwardDiff -> FD fallback
    assert _tvb_parse_gradient(None, pyfun_spec)["mode"] == "fd"
    assert _tvb_parse_gradient("fd", julia_spec)["mode"] == "fd"

    g = _tvb_parse_gradient("my_jac", julia_spec)
    assert g["mode"] == "user"
    assert g["julia_name"] == "my_jac"

    g2 = _tvb_parse_gradient(lambda theta, U, gamma: None, julia_spec)
    assert g2["mode"] == "user"
    assert g2["julia_name"] is None

    with pytest.raises(ValueError, match="Python function"):
        _tvb_parse_gradient("my_jac", pyfun_spec)
    with pytest.raises(ValueError, match="Julia"):
        _tvb_parse_gradient("bad name", julia_spec)
    with pytest.raises(ValueError, match="gradient"):
        _tvb_parse_gradient(1, julia_spec)


# ---------------------------------------------------------------------
# tvbounds_counterfactual(): argument validation (pure Python; all of
# these errors trigger before any Julia code runs)
# ---------------------------------------------------------------------

def test_counterfactual_validates_arguments_before_julia():
    def mom(theta, U, gamma):     # never called
        return None

    with pytest.raises(ValueError, match="moments"):
        tvbounds_counterfactual(42, d=1, theta_lb=0, theta_ub=1,
                                delta=0.5, u_dim=1)
    with pytest.raises(ValueError, match="`d`"):
        tvbounds_counterfactual(mom, d=0, theta_lb=0, theta_ub=1,
                                delta=0.5, u_dim=1)
    with pytest.raises(ValueError, match="equal"):
        tvbounds_counterfactual(mom, d=1, theta_lb=[0, 0], theta_ub=1,
                                delta=0.5, u_dim=1)
    with pytest.raises(ValueError, match="theta_lb"):
        tvbounds_counterfactual(mom, d=1, theta_lb=1, theta_ub=0,
                                delta=0.5, u_dim=1)
    with pytest.raises(ValueError, match="theta_init"):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=0.5, u_dim=1, theta_init=2)
    with pytest.raises(ValueError, match="delta"):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=[], u_dim=1)
    with pytest.raises(ValueError, match="strictly positive"):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=[0, 0.5], u_dim=1)
    with pytest.raises(ValueError, match="duplicated"):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=[0.5, 0.5], u_dim=1)
    # TV-family budgets are capped at 1; KL-family budgets are not
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=[0.5, 2], divergence="TV", u_dim=1)
    with pytest.raises(ValueError):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=0.5, divergence="nope", u_dim=1)
    with pytest.raises(ValueError):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=0.5, side="nope", u_dim=1)
    with pytest.raises(ValueError, match="u_dim"):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=0.5)
    with pytest.raises(ValueError, match="u_dim <= 15"):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=0.5, u_dim=16)
    with pytest.raises(ValueError, match="M"):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=0.5, u_dim=1, M=1)
    with pytest.raises(ValueError, match="matrix"):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=0.5, U="not a matrix")
    with pytest.raises(ValueError, match="ncol"):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=0.5,
                                U=(np.arange(1, 7) / 7).reshape((3, 2), order="F"),
                                u_dim=3)
    with pytest.raises(ValueError, match="gamma"):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=0.5, u_dim=1, gamma="x")
    with pytest.raises(ValueError, match="Python function"):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=0.5, u_dim=1, gradient="my_jac")
    with pytest.raises(ValueError, match="tvbounds_control"):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=0.5, u_dim=1, control={})
    with pytest.raises(ValueError, match="seed"):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=0.5, u_dim=1, seed=1.5)
    with pytest.raises(ValueError, match="verbose"):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=0.5, u_dim=1, verbose="yes")
    # the reduced TVmix program needs kappa >= 1 - delta
    with pytest.raises(ValueError, match="TVmixC"):
        tvbounds_counterfactual(mom, d=1, theta_lb=0, theta_ub=1,
                                delta=[0.2, 0.9], divergence="TVmix",
                                u_dim=1,
                                control=tvbounds_control(tvmix_kappa=0.3))


def test_validation_failures_leave_the_callers_rng_stream_untouched():
    np.random.seed(42)
    before = _rng_state()
    with pytest.raises(ValueError, match="strictly positive"):
        tvbounds_counterfactual(lambda theta, U, gamma: None, d=1,
                                theta_lb=0, theta_ub=1, delta=-1,
                                u_dim=1, seed=99)
    assert _same_state(_rng_state(), before)


def test_local_random_streams_round_trip_the_rng_state():
    # R scopes its RNG through withr::local_seed() /
    # withr::local_preserve_seed(); the Python module never touches the
    # global NumPy stream: seeded draws come from local generators (the
    # R-compatible RRNG of the bootstrap, Julia's own RNG for the Halton
    # draws), which leave the caller's stream untouched.
    np.random.seed(7)
    before = _rng_state()
    RRNG(123456).sample_int_replace(10, 10)
    np.random.default_rng(123456).uniform(size=10)
    assert _same_state(_rng_state(), before)


# ---------------------------------------------------------------------
# Guarded integration tests (need Julia + juliacall + licensed KNITRO)
# ---------------------------------------------------------------------

def test_toy_julia_model_tvmix_bounds_equal_the_theta_box_endpoints():
    _skip_unless_julia()
    toy = _julia_file("examples", "toy.jl")
    if not os.path.exists(toy):
        pytest.skip("toy.jl not found")

    # Model: U ~ Uniform(0,1), moment E[U] = theta, counterfactual K = U.
    # Under "TVmix" (mixture weight 1 - delta) the inner value equals
    # theta whenever the moment is satisfiable, so the outer bounds are
    # the box endpoints (up to O(tau log M) smoothing).
    t0 = time.time()
    fit = tvbounds_counterfactual(
        moments=(toy, "tvb_toy_moments!"),
        d=1,
        theta_lb=0.4, theta_ub=0.6,
        delta=[0.5, 1],
        divergence="TVmix",
        M=500, u_dim=1,
        theta_init=0.5,
        control=tvbounds_control(maxsolves=2),
        seed=1234, verbose=False)
    print("toy TVmix run: %.1f s" % (time.time() - t0))

    assert isinstance(fit, TVBounds)
    assert fit.application == "counterfactual"
    assert fit.divergence == "TVmix"
    assert fit.neighborhood is None
    assert np.array_equal(fit.bounds["delta"].to_numpy(), [0.5, 1.0])
    assert fit.bounds["lower"].to_numpy() == pytest.approx([0.4, 0.4], rel=0.02)
    assert fit.bounds["upper"].to_numpy() == pytest.approx([0.6, 0.6], rel=0.02)
    # baseline point = mean of the Halton draws ~ 1/2
    assert fit.point == pytest.approx(0.5, rel=0.02)
    # monotone in the budget by construction
    assert np.all(np.diff(fit.bounds["lower"].to_numpy()) <= 1e-8)
    assert np.all(np.diff(fit.bounds["upper"].to_numpy()) >= -1e-8)
    # diagnostics present
    assert isinstance(fit.details["solver"], pd.DataFrame)
    assert list(fit.details["solver"].columns) == [
        "delta", "status_outer_lower", "status_inner_lower", "time_lower",
        "status_outer_upper", "status_inner_upper", "time_upper"]
    assert fit.details["theta_lower"].shape == (1, 2)
    assert fit.details["theta_upper"].shape == (1, 2)
    assert fit.details["fixed_theta"] is False
    assert fit.details["moments"] == "Julia function tvb_toy_moments! (from toy.jl)"
    assert fit.details["gradient"] == "forwarddiff"
    assert fit.details["seed"] == 1234
    assert fit.details["M"] == 500 and fit.details["u_dim"] == 1
    assert isinstance(fit.details["control"], TVBoundsControl)
    assert os.path.exists(fit.details["control"]["inner_opt"])
    assert math.isnan(fit.n) and math.isnan(fit.level) and math.isnan(fit.B)
    assert "counterfactual bounds under a TVmix divergence neighborhood" in str(fit)


def test_toy_julia_model_fixed_theta_tv_bounds_at_delta_1():
    _skip_unless_julia()
    toy = _julia_file("examples", "toy.jl")
    # Degenerate box: inner-only fixed-theta path. At delta = 1 the TV
    # ball contains every distribution on the draws, so the bound solves
    # the Manski/Lee linear program with the moment E[U] = 0.5 pinned:
    # both bounds equal theta = 0.5.
    t0 = time.time()
    fit = tvbounds_counterfactual(
        moments=(toy, "tvb_toy_moments!"),
        d=1,
        theta_lb=0.5, theta_ub=0.5,
        delta=1,
        divergence="TV",
        M=500, u_dim=1,
        theta_init=0.5,
        seed=1234, verbose=False)
    print("toy fixed-theta TV run: %.1f s" % (time.time() - t0))

    assert fit.details["fixed_theta"] is True
    assert fit.bounds["lower"].to_numpy() == pytest.approx([0.5], rel=0.02)
    assert fit.bounds["upper"].to_numpy() == pytest.approx([0.5], rel=0.02)
    assert fit.details["theta_lower"].shape == (1, 1)
    # outer flags do not apply in fixed-theta mode
    assert int(fit.details["solver"]["status_outer_lower"].iloc[0]) == -999


def test_python_moments_bridge_reproduces_the_toy_fixed_theta_bounds():
    _skip_unless_julia()

    def py_mom(theta, U, gamma):
        return {"K": U[:, 0], "G": (U[:, 0] - theta[0]).reshape(-1, 1)}

    t0 = time.time()
    fit = tvbounds_counterfactual(
        moments=py_mom,
        d=1,
        theta_lb=0.5, theta_ub=0.5,
        delta=1,
        divergence="TVmix",
        M=300, u_dim=1,
        theta_init=0.5,
        seed=1234, verbose=False)
    print("Python bridge run: %.1f s" % (time.time() - t0))

    assert fit.bounds["lower"].to_numpy() == pytest.approx([0.5], rel=0.02)
    assert fit.bounds["upper"].to_numpy() == pytest.approx([0.5], rel=0.02)
    assert fit.details["gradient"] == "fd"
    assert fit.details["moments"] == "Python function"


def test_seeded_counterfactual_runs_are_reproducible():
    _skip_unless_julia()
    toy = _julia_file("examples", "toy.jl")

    def run():
        return tvbounds_counterfactual(
            moments=(toy, "tvb_toy_moments!"), d=1,
            theta_lb=0.5, theta_ub=0.5, delta=1,
            divergence="TVmix", M=300, u_dim=1, theta_init=0.5,
            seed=2024, verbose=False)
    f1 = run()
    f2 = run()
    assert np.array_equal(f1.bounds["lower"].to_numpy(), f2.bounds["lower"].to_numpy())
    assert np.array_equal(f1.bounds["upper"].to_numpy(), f2.bounds["upper"].to_numpy())
    assert f1.point == f2.point


# ---------------------------------------------------------------------
# Additional bridge checks (Python-specific: array orientation and the
# payload conversion)
# ---------------------------------------------------------------------

_TOY2_JL = """
# two moments E[U] = theta[1], E[U^2] = theta[2]; counterfactual K = U
function tvb_toy2_moments!(K, G, theta, U, obj)
    @inbounds for m in 1:size(U, 1)
        K[m]    = U[m, 1]
        G[m, 1] = U[m, 1] - theta[1]
        G[m, 2] = U[m, 1] * U[m, 1] - theta[2]
    end
    return nothing
end
"""

_GAMMA_JL = """
# reads the payload as a Julia moments file written for the R package does
function tvb_gamma_moments!(K, G, theta, U, obj)
    g = obj.gamma
    shift = g[:shift]                     # Float64
    w     = g[:w]                         # Vector{Float64}
    n     = g[:n]                         # Int
    flag  = g[:flag]                      # Bool
    A     = g[:A]                         # 2 x 3 matrix
    a12   = g[:sub][:a]                   # nested dict
    name  = g[:name]                      # String
    c = shift + w[2] + (flag ? 0.0 : 1.0) + A[1, 2] + a12 + n + length(name)
    @inbounds for m in 1:size(U, 1)
        K[m]    = U[m, 1] + c
        G[m, 1] = w[1] * U[m, 1] - theta[1]
    end
    return nothing
end
"""


def test_python_moments_keep_the_m_by_d_orientation_of_g(tmp_path):
    _skip_unless_julia()
    f = tmp_path / "toy2.jl"
    f.write_text(_TOY2_JL)

    def py_mom(theta, U):
        u = U[:, 0]
        return {"K": u, "G": np.column_stack([u - theta[0], u ** 2 - theta[1]])}

    common = dict(d=2, theta_lb=[0.5, 1 / 3], theta_ub=[0.5, 1 / 3],
                  delta=[0.5, 1], divergence="TVmix", M=300, u_dim=1,
                  theta_init=[0.5, 1 / 3], seed=77, verbose=False)
    fj = tvbounds_counterfactual(moments=(str(f), "tvb_toy2_moments!"), **common)
    fp = tvbounds_counterfactual(moments=py_mom, **common)
    # identical draws and identical (bitwise) K, G columns -> identical solves
    assert np.allclose(fj.bounds["lower"].to_numpy(), fp.bounds["lower"].to_numpy(),
                       rtol=1e-8, atol=1e-10)
    assert np.allclose(fj.bounds["upper"].to_numpy(), fp.bounds["upper"].to_numpy(),
                       rtol=1e-8, atol=1e-10)
    assert fj.point == fp.point
    # a transposed G is rejected on the Python side
    def bad_mom(theta, U):
        u = U[:, 0]
        return {"K": u, "G": np.vstack([u - theta[0], u ** 2 - theta[1]])}
    with pytest.raises(RuntimeError, match="expected \\(M, d\\)"):
        tvbounds_counterfactual(moments=bad_mom, **common)


def test_py_to_gamma_gives_julia_moments_files_the_r_view_of_the_payload(tmp_path):
    _skip_unless_julia()
    from tvbounds._julia_setup import _julia_main
    jl = _julia_main()
    f = tmp_path / "gam.jl"
    f.write_text(_GAMMA_JL)
    A = np.arange(6, dtype=float).reshape(2, 3)     # A[0, 1] = 1.0
    # (the constant c added to K keeps |bound| below control$lower_limit = 10,
    # beyond which the inner solver reports the lower problem as unbounded)
    gamma = {"shift": 0.25, "w": [2.0, 0.5], "n": 1, "flag": True,
             "A": A, "sub": {"a": 0.125}, "name": "ab"}
    # direct conversion checks
    g = jl.TVBoundsPyBridge.py_to_gamma(gamma)
    probe = jl.seval("g -> (typeof(g[:shift]), typeof(g[:w]), typeof(g[:n]), "
                     "typeof(g[:flag]), typeof(g[:A]), size(g[:A]), g[:A][1, 2], "
                     "typeof(g[:sub]), typeof(g[:name]), g[:missing_key_test])")
    with pytest.raises(Exception):
        probe(g)   # missing key -> Julia KeyError, as for an R payload
    probe = jl.seval("g -> (string(typeof(g[:shift])), string(typeof(g[:w])), "
                     "string(typeof(g[:n])), string(typeof(g[:flag])), "
                     "string(typeof(g[:A])), size(g[:A]), g[:A][1, 2], "
                     "string(typeof(g[:sub])), string(typeof(g[:name])))")
    out = tuple(probe(g))
    assert out[0] == "Float64" and out[1] == "Vector{Float64}"
    assert out[2] == "Int64" and out[3] == "Bool"
    assert out[4] == "Matrix{Float64}" and tuple(out[5]) == (2, 3) and out[6] == 1.0
    assert out[7] == "Dict{Symbol, Any}" and out[8] == "String"
    # the payload reaches a Julia moments file through obj.gamma and a
    # Python callable through its third argument: same numbers
    c = 0.25 + 0.5 + 0.0 + 1.0 + 0.125 + 1 + 2

    def py_mom(theta, U, gamma):
        u = U[:, 0]
        cc = (gamma["shift"] + gamma["w"][1] + (0.0 if gamma["flag"] else 1.0)
              + gamma["A"][0, 1] + gamma["sub"]["a"] + gamma["n"]
              + len(gamma["name"]))
        return {"K": u + cc, "G": (gamma["w"][0] * u - theta[0]).reshape(-1, 1)}

    common = dict(d=1, theta_lb=1.0, theta_ub=1.0, delta=1, divergence="TVmix",
                  M=200, u_dim=1, theta_init=1.0, seed=5, verbose=False,
                  gamma=gamma)
    fj = tvbounds_counterfactual(moments=(str(f), "tvb_gamma_moments!"), **common)
    fp = tvbounds_counterfactual(moments=py_mom, **common)
    assert fj.point == pytest.approx(0.5 + c, rel=0.02)
    assert fj.point == fp.point
    assert np.all(np.isfinite(fj.bounds[["lower", "upper"]].to_numpy()))
    assert fj.bounds["lower"].to_numpy() == pytest.approx([0.5 + c], rel=0.02)
    assert np.allclose(fj.bounds[["lower", "upper"]].to_numpy(),
                       fp.bounds[["lower", "upper"]].to_numpy(),
                       rtol=1e-8, atol=1e-10, equal_nan=True)


def test_user_gradient_bridge_matches_automatic_differentiation():
    """A Python-callable gradient is only consulted when the outer solver
    asks for an analytic gradient (Optim.jl, or KNITRO with gradopt 1); with
    the toy model its bounds must agree with ForwardDiff on the Julia file."""
    _skip_unless_julia()
    toy = _julia_file("examples", "toy.jl")
    common = dict(d=1, theta_lb=0.4, theta_ub=0.6, delta=[0.5, 1],
                  divergence="TVmix", M=300, u_dim=1, theta_init=0.5,
                  seed=1234, verbose=False,
                  control=tvbounds_control(maxsolves=2, use_optim=True))

    def py_mom(theta, U, gamma):
        return {"K": U[:, 0], "G": (U[:, 0] - theta[0]).reshape(-1, 1)}

    def py_jac(theta, U, gamma):
        M = U.shape[0]
        return {"K": np.zeros((M, 1)), "G": -np.ones((M, 1, 1))}

    ad = tvbounds_counterfactual(moments=(toy, "tvb_toy_moments!"), **common)
    user = tvbounds_counterfactual(moments=py_mom, gradient=py_jac, **common)
    assert user.details["gradient"] == "user"
    assert np.allclose(user.bounds["lower"], ad.bounds["lower"], atol=1e-6)
    assert np.allclose(user.bounds["upper"], ad.bounds["upper"], atol=1e-6)


def test_malformed_gradient_collection_is_rejected_with_r_message():
    from tvbounds.counterfactual import _wrap_jac_result
    with pytest.raises(ValueError, match="stacked"):
        _wrap_jac_result({"K": np.zeros((3, 1))})
