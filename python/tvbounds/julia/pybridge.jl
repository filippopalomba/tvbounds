# ===================================================================
# TVBoundsPyBridge -- Python-side glue for the tvbounds Python module.
#
# The solver itself is TVBoundsJulia (tvbounds.jl, shared byte for byte
# with the R package and never edited here).  This module adds what the
# Python front end needs, mirroring the R bridge helpers of tvbounds.jl:
#
#   make_py_moments_wrapper(pyfun)  analogue of make_r_moments_wrapper:
#       wraps a Python callable f(theta, U) returning a dict with `K`
#       (length M) and `G` (M x d) into the in-place Bundle signature
#       moments!(K, G, theta, U, obj);
#   make_py_jac_wrapper(pyfun)      analogue of make_r_jac_wrapper for a
#       Python gradient f(theta, U) returning the stacked Jacobian or a
#       dict with `K` (M x l) and `G` (M x d x l);
#   py_to_gamma(obj)                converts a Python payload recursively
#       into the Julia view an R payload gets through JuliaCall (dicts
#       keyed by symbols, so `obj.gamma[:name]` works unchanged);
#   tvb_solve_py(args)              converts a Python dict of arguments
#       into the Julia types tvb_solve expects and returns its Dict.
#
# The Python callables are curried with the user's gamma payload on the
# Python side (as the R package curries on the R side), so the payload
# never crosses the bridge for Python moments.
# ===================================================================

module TVBoundsPyBridge

# PythonCall is loaded into Main by juliacall before this file is
# included; binding it through Main keeps the bridge working also when
# the solver stack lives in a separately activated project (the
# activate/instantiate fallback of the Python package).
using Main.PythonCall

const TVB = Main.TVBoundsJulia

export make_py_moments_wrapper, make_py_jac_wrapper, py_to_gamma, tvb_solve_py

# ------------------------------------------------------------------
# numpy handles (imported lazily; needed only to recognise numpy
# arrays and scalars).
# ------------------------------------------------------------------
const _NUMPY = Ref{Py}()
function _numpy()
    isassigned(_NUMPY) || (_NUMPY[] = pyimport("numpy"))
    return _NUMPY[]
end
_is_np_array(x::Py)  = pyisinstance(x, _numpy().ndarray)
_is_np_scalar(x::Py) = pyisinstance(x, _numpy().generic)
_is_mapping(x::Py)   = pyisinstance(x, pyimport("collections.abc").Mapping)
_is_sequence(x::Py)  = pyisinstance(x, pybuiltins.list) || pyisinstance(x, pybuiltins.tuple)

# Any Python-derived object (a raw `Py` or one of PythonCall's wrappers
# such as PyDict/PyList/PyArray, which is what juliacall hands over when a
# Python dict/list/array is passed to a Julia function) as a raw `Py`.
_as_py(x) = ispy(x) ? Py(x) : x

# ------------------------------------------------------------------
# Python payload -> Julia (the view an R list gets through JuliaCall).
# ------------------------------------------------------------------
"""
    py_to_gamma(obj)

Convert a Python payload recursively into Julia values: `None` -> `nothing`,
`bool` -> `Bool`, `int` -> `Int`, `float` -> `Float64`, `str` -> `String`,
numpy arrays -> `Array{Float64/Int64/Bool,N}` with the same logical
indexing, numpy scalars -> the matching Julia scalar, dicts ->
`Dict{Symbol,Any}` (keys as symbols, so a Julia moments file written for
the R package reads `obj.gamma[:name]` unchanged), lists/tuples ->
`Vector{Float64}`/`Vector{Int}`/`Vector{Bool}`/`Vector{String}` when
homogeneous and `Vector{Any}` otherwise.  Julia values pass through.
"""
function py_to_gamma(obj)
    ispy(obj) || return obj
    x = _as_py(obj)
    if pyis(x, pybuiltins.None)
        return nothing
    elseif pyisinstance(x, pybuiltins.bool)
        return pyconvert(Bool, x)
    elseif pyisinstance(x, pybuiltins.int)
        return pyconvert(Int, x)
    elseif pyisinstance(x, pybuiltins.float)
        return pyconvert(Float64, x)
    elseif pyisinstance(x, pybuiltins.str)
        return pyconvert(String, x)
    elseif _is_np_array(x)
        return _np_array_to_julia(x)
    elseif _is_np_scalar(x)
        return py_to_gamma(x.item())
    elseif _is_mapping(x)
        out = Dict{Symbol,Any}()
        for k in x
            out[Symbol(pyconvert(String, pystr(k)))] = py_to_gamma(x[k])
        end
        return out
    elseif _is_sequence(x)
        return _tighten(Any[py_to_gamma(v) for v in x])
    else
        return pyconvert(Any, x)
    end
end

function _np_array_to_julia(x::Py)
    kind = pyconvert(String, x.dtype.kind)
    if kind == "b"
        return pyconvert(Array{Bool}, x)
    elseif kind == "i" || kind == "u"
        return pyconvert(Array{Int64}, x)
    elseif kind == "f"
        return pyconvert(Array{Float64}, x)
    elseif kind == "U" || kind == "S"
        return pyconvert(Array{String}, x)
    else
        # object (or other) arrays: convert element by element
        return py_to_gamma(x.tolist())
    end
end

# Homogeneous Python lists become typed vectors (the Julia view of an R
# atomic vector); mixed int/float lists promote to Float64 as Julia's own
# vector literals do.
function _tighten(vals::Vector{Any})
    isempty(vals) && return vals
    if all(v -> v isa Bool, vals)
        return Vector{Bool}(vals)
    elseif all(v -> v isa Int, vals)
        return Vector{Int}(vals)
    elseif all(v -> (v isa Float64) || (v isa Int), vals)
        return Vector{Float64}(vals)
    elseif all(v -> v isa String, vals)
        return Vector{String}(vals)
    else
        return vals
    end
end

# ------------------------------------------------------------------
# Typed array conversions used by the wrappers.
# ------------------------------------------------------------------
# A Python array-like as a Julia Array{Float64,N} with the same logical
# indexing (numpy (M, d) -> Julia M x d); lists become vectors.
function _to_f64array(x)
    x = _as_py(x)
    if _is_np_array(x)
        return pyconvert(Array{Float64}, x)
    elseif _is_np_scalar(x) || pyisinstance(x, pybuiltins.float) ||
           pyisinstance(x, pybuiltins.int)
        return [pyconvert(Float64, x)]
    else
        return pyconvert(Vector{Float64}, x)
    end
end
_to_f64vec(x)    = vec(_to_f64array(x))
function _to_f64matrix(x)
    a = _to_f64array(x)
    a isa AbstractMatrix || error("`U` must be a 2-d array (M x u_dim)")
    return Matrix{Float64}(a)
end

# Split a Python result into its K and G components (a mapping with
# keys "K"/"G", or a 2-sequence (K, G)); returns Py objects.
function _py_split_kg(res::Py, what::AbstractString)
    K = nothing
    G = nothing
    if _is_mapping(res)
        pycontains(res, "K") && (K = res["K"])
        pycontains(res, "G") && (G = res["G"])
    elseif _is_sequence(res) && pylen(res) == 2
        K = res[0]
        G = res[1]
    end
    (K === nothing || G === nothing) &&
        error("the $(what) must return a dict with components `K` and " *
              "`G` (or a (K, G) pair); got a Python " *
              pyconvert(String, pytype(res).__name__))
    return K, G
end

# ------------------------------------------------------------------
# Python moments / gradient wrappers.
# ------------------------------------------------------------------
"""
    make_py_moments_wrapper(pyfun) -> Function

Wrap a Python callable `pyfun(theta, U)` returning a dict with `K`
(length M) and `G` (M x d) into a Bundle-compatible in-place
`moments!(K, G, theta, U, obj)`.  `theta` reaches Python as a fresh
Julia vector and `U` as the Bundle's M x u_dim matrix (both wrapped
without copying by juliacall; `numpy.asarray` gives the (M, u_dim)
view).  The result components are converted to Float64 arrays with the
same logical shape and copied into the Bundle's buffers; the length
checks and messages are those of the R bridge (`_split_kg_result`).
"""
function make_py_moments_wrapper(pyfun)
    pf = _as_py(pyfun)
    return function (K, G, theta, U, obj)
        res = pf(collect(Float64, theta), U)
        Kp, Gp = _py_split_kg(res, "moments function")
        Kv = _to_f64array(Kp)
        Gv = _to_f64array(Gp)
        Kj, Gj = TVB._split_kg_result(Dict("K" => Kv, "G" => Gv),
                                      length(K), length(G))
        copyto!(K, Kj)
        copyto!(G, Gj)
        return nothing
    end
end

"""
    make_py_jac_wrapper(pyfun) -> Function

Wrap a Python gradient `pyfun(theta, U)` into a Bundle-compatible
`moments_jac(theta, U, obj)`.  The Python value (the stacked
`(M*(d+1)) x l` Jacobian, or a dict / pair with `K` (M x l) and `G`
(M x d x l)) is converted to Julia arrays with the same logical shape;
the downstream `_normalize_jac` of TVBoundsJulia validates the sizes.
"""
function make_py_jac_wrapper(pyfun)
    pf = _as_py(pyfun)
    return function (theta, U, obj)
        raw = pf(collect(Float64, theta), U)
        if _is_mapping(raw) || (_is_sequence(raw) && pylen(raw) == 2 &&
                                !_is_np_array(raw))
            Kp, Gp = _py_split_kg(raw, "gradient function")
            return Dict{String,Any}("K" => _to_f64array(Kp),
                                    "G" => _to_f64array(Gp))
        end
        return _to_f64array(raw)
    end
end

# ------------------------------------------------------------------
# Entry point: a Python dict of arguments -> tvb_solve(; kwargs...).
# ------------------------------------------------------------------
const _STR_ARGS  = (:moments_name, :inner_opt, :outer_opt, :divergence,
                    :side, :jac_mode, :jac_name)
const _VEC_ARGS  = (:delta, :theta_lb, :theta_ub, :theta_init)
const _INT_ARGS  = (:d, :M, :u_dim, :maxsolves, :iterations,
                    :outer_iterations, :seed)
const _FLT_ARGS  = (:startptrange, :time_limit, :lower_limit, :eta_min,
                    :psi_tv_eps, :tvac_tau, :tvmix_tau, :purekl_acap,
                    :tvmix_kappa, :fd_step)
const _BOOL_ARGS = (:use_optim, :verbose)

function _convert_arg(name::Symbol, v)
    v = _as_py(v)
    v isa Py || return v
    pyis(v, pybuiltins.None) && return nothing
    name in _STR_ARGS  && return pyconvert(String, v)
    name in _VEC_ARGS  && return _to_f64vec(v)
    name in _INT_ARGS  && return pyconvert(Int, v)
    name in _FLT_ARGS  && return pyconvert(Float64, v)
    name in _BOOL_ARGS && return pyconvert(Bool, v)
    name === :U        && return _to_f64matrix(v)
    name === :gamma    && return py_to_gamma(v)
    return pyconvert(Any, v)
end

"""
    tvb_solve_py(args) -> Dict{String,Any}

Call `TVBoundsJulia.tvb_solve` with the arguments of the Python dict
`args` (the same names as the R package's argument list), converted to
the Julia types tvb_solve expects: Float64 vectors for the budget grid
and the theta box, `Matrix{Float64}` for `U`, `nothing` for `None`,
`Int`, `Bool`, `String`, and `py_to_gamma` for the payload.
"""
function tvb_solve_py(args)
    pyargs = _as_py(args)
    kw = Dict{Symbol,Any}()
    for k in pyargs
        name = Symbol(pyconvert(String, pystr(k)))
        kw[name] = _convert_arg(name, pyargs[k])
    end
    return TVB.tvb_solve(; kw...)
end

end # module TVBoundsPyBridge
