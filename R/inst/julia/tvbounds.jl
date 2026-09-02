# ===================================================================
# TVBoundsJulia -- Julia solver backend for the tvbounds R package.
#
# Computes sharp bounds on a counterfactual functional E_P[K(U; theta)]
# when the distribution P of the latent draws ranges over a divergence
# neighborhood of the simulated baseline and theta ranges over the set
# of parameters compatible with the moment conditions E_P[G(U; theta)] = 0
# (Palomba, 2026, "Sensitivity Analysis in Population Shares";
# Christensen & Connault, 2023, ECMA).
#
# Self-contained adaptation of the audited replication library
# (CounterfactualStructural/replication_mine/code/julia/aux.jl):
#   (1) divergence conjugates Psi and derivatives -- seven divergences:
#       :KL_chi2 (CC23 hybrid), :KL, :chi2, :TV, :TVmix, :TVmixC, :TVac
#   (2) scrambled Halton quasi-MC draws (rhalton)
#   (3) KNITRO inner loop (profiled dual over (eta, zeta, lambda)),
#       raw KN_* C API
#   (4) KNITRO / Optim outer loop with multi-start over theta
#   (5) thin entry points for R (JuliaCall): tvb_solve, tvb_check_knitro
#
# The model-specific CSW machinery of the replication (csw_moments!,
# Phi_row!, bootstrap draws, cohort handling) is deliberately NOT here:
# the user supplies a generic `moments!(K, G, theta, U, obj)` function
# and an arbitrary payload `gamma` carried by the Bundle.
#
# KNITRO is a commercial solver by Artelys and requires a valid
# license.  The module loads without it (so that tvb_check_knitro can
# report a useful status), but every solve requires it.
# ===================================================================

module TVBoundsJulia

using ForwardDiff, LinearAlgebra, Statistics, Random, Printf
using Optim, NLSolversBase, Parameters

export Bundle, tvb_solve, tvb_check_knitro, tvb_generate_halton,
       rhalton, inner_loop, outer_loop, outer_loop_optim, outer_loop_multi

# ------------------------------------------------------------------
# Guarded KNITRO import.  If KNITRO.jl cannot be loaded (library or
# license machinery missing) the module still loads; solves then fail
# with a clear error, and tvb_check_knitro() reports the cause.
# ------------------------------------------------------------------
const _KNITRO_OK  = Ref(false)
const _KNITRO_ERR = Ref("")
try
    @eval using KNITRO
    _KNITRO_OK[] = true
catch err
    _KNITRO_ERR[] = sprint(showerror, err)
end

"""
    tvb_check_knitro() -> String

Check that KNITRO.jl is loadable AND that a KNITRO solver context can
be created and freed (which exercises the Artelys license check).
Returns a status string starting with `"ok"` on success and with
`"error:"` otherwise.  Called once per R session by the tvbounds
package.
"""
function tvb_check_knitro()
    _KNITRO_OK[] ||
        return "error: KNITRO.jl could not be loaded: " * _KNITRO_ERR[]
    kc = nothing
    try
        kc = KNITRO.KN_new()
        rel = try
            string(KNITRO.get_release())
        catch
            ""
        end
        return isempty(rel) ? "ok" : "ok (KNITRO " * rel * ")"
    catch err
        return "error: a KNITRO context could not be created " *
               "(missing or invalid Artelys license?): " *
               sprint(showerror, err)
    finally
        if kc !== nothing
            try
                KNITRO.KN_free(kc)
            catch
            end
        end
    end
end

# ------------------------------------------------------------------
# Lightweight timing instrumentation (opt-in).  When TIMING_ON[] is
# true, every inner-problem solve (one inner_loop / inner_loop_internal
# call, incl. its warm-start refinements) pushes its wall time onto
# INNER_TIMES.  Guarded so production runs are unaffected.
# ------------------------------------------------------------------
const INNER_TIMES = Float64[]
const TIMING_ON   = Ref(false)
@inline function _record_inner!(t)
    TIMING_ON[] && push!(INNER_TIMES, t)
    return nothing
end

# ------------------------------------------------------------------
# Tuning constants.  These are Refs (not consts) so that the R-side
# control list tvbounds_control() can override them per call through
# tvb_solve; the defaults below are the paper's.  Single-threaded by
# construction (as in the replication library).
# ------------------------------------------------------------------
const PSI_TV_EPS  = Ref(1e-4)    # Huber smoothing scale for max(a, -1/2)
const TVAC_TAU    = Ref(1e-3)    # soft-max temperature, :TVac Manski term
const TVMIX_TAU   = Ref(1e-3)    # soft-max temperature, :TVmix Manski term
const PUREKL_ACAP = Ref(500.0)   # overflow clamp for the pure-KL conjugate

# Effective contamination weight kappa for :TVmix / :TVmixC.  By
# default kappa is tied to the budget (kappa = 1 - delta, as in the
# replication scripts); setting TVMIX_KAPPA_OVERRIDE to a number in
# [0, 1] decouples them (general two-parameter set T_delta \cap C_kappa,
# Lemma SA-40 of the paper).  NaN means "no override".
const TVMIX_KAPPA_OVERRIDE = Ref(NaN)
const TVMIX_KAPPA          = Ref(0.0)   # consumed by Psi_TVmix*/dPsi_TVmix_red!

@inline function _effective_kappa(delta::Float64)
    o = TVMIX_KAPPA_OVERRIDE[]
    return isnan(o) ? 1.0 - delta : o
end

# ------------------------------------------------------------------
# Hybrid KL / chi^2 divergence: conjugate Psi and derivatives
# phi(x) = x log x - x + 1        if x <= e
# phi(x) = (x-e)^2/(2e) + (x-e)+1 if x > e
# phi*(y) = exp(y) - 1            if y <= 1
# phi*(y) = (e/2)(y^2 + 1) - 1    if y > 1
# ------------------------------------------------------------------
function Psi_KL!(out, a)
    @inbounds for i in eachindex(a)
        out[i] = a[i] <= 1.0 ? exp(a[i]) : 0.5 * exp(1) * (a[i]^2 + 1.0)
    end
    out .-= 1.0
end

function dPsi_KL!(out, a)
    @inbounds for i in eachindex(a)
        out[i] = a[i] <= 1.0 ? exp(a[i]) : exp(1) * a[i]
    end
end

function ddPsi_KL!(out, a)
    @inbounds for i in eachindex(a)
        out[i] = a[i] <= 1.0 ? exp(a[i]) : exp(1)
    end
end

# ------------------------------------------------------------------
# Total Variation divergence (Lemma 35 in Palomba JMP, with Remark 8
# specialised to phi_TV(t) = |t-1|/2):
#   phi*(a) = max(a, -1/2)        on a <= 1/2
#   phi*(a) = +infty              on a > 1/2
# Recession phi^infty(1) = 1/2; the dual feasibility constraint is
#   ell_s + ell_eps/2 >= sup_z { g(z) - ell^T m(z) }
# which we enforce via M linear inequalities on the simulated draws.
# We use a smooth Huber-type approximation of max(a, -1/2) so the
# objective is C^infty, and rely on KNITRO's linear constraints to
# enforce the upper-feasibility condition exactly.
# ------------------------------------------------------------------
@inline function _tv_root(a)
    s = a + 0.5
    e = PSI_TV_EPS[]
    return sqrt(s * s + e * e)
end

function Psi_TV!(out, a)
    @inbounds for i in eachindex(a)
        out[i] = 0.5 * (a[i] - 0.5 + _tv_root(a[i]))
    end
end

function dPsi_TV!(out, a)
    @inbounds for i in eachindex(a)
        out[i] = 0.5 * (1.0 + (a[i] + 0.5) / _tv_root(a[i]))
    end
end

function ddPsi_TV!(out, a)
    e = PSI_TV_EPS[]
    @inbounds for i in eachindex(a)
        r = _tv_root(a[i])
        out[i] = 0.5 * e * e / (r * r * r)
    end
end

# ------------------------------------------------------------------
# Total Variation restricted to distributions absolutely continuous
# w.r.t. the baseline ("TV^<<", Remark SA-27 in Palomba JMP): the
# pointwise dual-feasibility constraint is enforced F0-a.e. by the
# domain of the perspective of phi_TV^* and can be dropped from the
# dual.  Concretely, substituting the cap u := zeta + eta/2 and the
# floor v := zeta - eta/2 in the TV inner dual, the partial minimum
# over u is u* = max(max_m y_m, v); for 0 < delta <= 1 the region
# v > max_m y_m never contains the minimizer (there F increases at
# rate 1 - delta >= 0, with the delta = 1 infimum approached as
# v grows), so the inner dual reduces to the UNCONSTRAINED convex
# program over (v, lambda):
#
#   F(v, lambda) = delta * max_m y_m(lambda) + (1 - delta) * v
#                  + (1/M) sum_m max(y_m(lambda) - v, 0),
#   y_m = s*K_m - lambda^T G_m,  s = (-1)^find_smallest.
#
# No M linear feasibility constraints.  REQUIRES 0 < delta <= 1 (the
# full TV range): for delta > 1 the reduced program is unbounded below
# in v (slope 1 - delta < 0), unlike the constrained :TV program which
# stays well-posed; run_inner_KNITRO errors on such requests.
# The max over draws is smoothed by a shifted log-sum-exp with
# temperature TVAC_TAU; max(., 0) is smoothed by the same Huber-type
# scheme as Psi_TV (scale PSI_TV_EPS).  Both smoothings sit ABOVE the
# exact piecewise-linear functions, so the computed bounds remain
# valid (outward-conservative, O(1e-3)).
#
# Implementation detail: the Bundle keeps the (2+d)-dim x layout with
# x[1] a dummy fixed at 1.0 (the eta slot), x[2] = v (the zeta slot),
# x[3:end] = lambda, so warm starts, multiplier extraction and the
# envelope-gradient plumbing are shared with the other divergences.
# ------------------------------------------------------------------
function Psi_TVac!(out, a)     # smoothed max(a, 0)
    e = PSI_TV_EPS[]
    @inbounds for i in eachindex(a)
        out[i] = 0.5 * (a[i] + sqrt(a[i] * a[i] + e * e))
    end
end

function dPsi_TVac!(out, a)
    e = PSI_TV_EPS[]
    @inbounds for i in eachindex(a)
        out[i] = 0.5 * (1.0 + a[i] / sqrt(a[i] * a[i] + e * e))
    end
end

function ddPsi_TVac!(out, a)
    e = PSI_TV_EPS[]
    @inbounds for i in eachindex(a)
        r = sqrt(a[i] * a[i] + e * e)
        out[i] = 0.5 * e * e / (r * r * r)
    end
end

# ------------------------------------------------------------------
# Mixture-constrained Total Variation ("TV^mix", end of Section SA5 in
# Palomba JMP): the TV ball intersected with the Huber contamination
# neighborhood C_kappa(F0) = {kappa*F0 + (1-kappa)*R}, i.e. the
# baseline is a mixing component of the candidate with weight kappa
# (= 1 - delta by default).  By Lemma "contamination as constrained
# TV"(ii) the mixture restriction is the measure inequality
# P >= kappa*F0 (density floor u >= kappa), and by the lemma's closing
# paragraph the TV constraint is redundant once kappa >= 1 - delta, so
# the dual is Theorem "strong duality with mixture constraint"(ii)
# with the lower-truncated conjugate of Example "Total Variation II":
#   phi*(a) = max(a, kappa*a + (kappa-1)/2)   on a <= 1/2,
#   phi*(a) = +infty                          on a >  1/2.
# The recession phi^infty(1) = 1/2 is inherited from phi_TV, so the M
# linear dual-feasibility constraints of the :TV mode carry over
# UNCHANGED; only the conjugate below the kink at a = -1/2 changes,
# from the flat -1/2 to the slope-kappa line through (-1/2, -1/2)
# (nature can no longer remove more than the share 1 - kappa of
# baseline mass from any region).  At kappa = 0 this is exactly
# Psi_TV.  Smoothing: max(x, y) = (x+y)/2 + |x-y|/2 with
# |x-y| = (1-kappa)|a+1/2| replaced by
# (1-kappa)*sqrt((a+1/2)^2 + PSI_TV_EPS^2) -- the same Huber scheme
# as Psi_TV, sitting ABOVE the exact conjugate, so computed bounds
# stay outward-conservative (pointwise bias <= (1-kappa)*PSI_TV_EPS/2).
# kappa defaults to 1 - delta (refreshed at every objective evaluation
# and at every inner solve; single-threaded by construction); an
# explicit kappa can be set through TVMIX_KAPPA_OVERRIDE.
#
# These three functions serve the REFERENCE mode :TVmixC, which solves
# this dual literally, in the (eta, zeta, lambda) variables and with
# the M linear feasibility constraints.  That program is correct but
# slow and numerically nasty, because its optimum sits exactly ON the
# eta = 0 boundary (see the :TVmix block below); production runs use
# the reduced :TVmix instead, and :TVmixC is retained as the
# independent cross-check of that reduction -- and as the ONLY mode
# that supports a kappa decoupled from delta (kappa < 1 - delta).
# ------------------------------------------------------------------
function Psi_TVmix!(out, a)
    kap = TVMIX_KAPPA[]
    @inbounds for i in eachindex(a)
        out[i] = 0.5 * ((1.0 + kap) * a[i] + 0.5 * (kap - 1.0)) +
                 0.5 * (1.0 - kap) * _tv_root(a[i])
    end
end

function dPsi_TVmix!(out, a)
    kap = TVMIX_KAPPA[]
    @inbounds for i in eachindex(a)
        out[i] = 0.5 * ((1.0 + kap) + (1.0 - kap) * (a[i] + 0.5) / _tv_root(a[i]))
    end
end

function ddPsi_TVmix!(out, a)
    kap = TVMIX_KAPPA[]
    e = PSI_TV_EPS[]
    @inbounds for i in eachindex(a)
        r = _tv_root(a[i])
        out[i] = 0.5 * (1.0 - kap) * e * e / (r * r * r)
    end
end

# ------------------------------------------------------------------
# TV^mix, REDUCED form (production mode :TVmix).
#
# The :TVmixC dual admits an exact closed-form reduction in which
# both eta and zeta are eliminated.  Write Y := max_m y_m and
# e_m := Y - y_m >= 0, with y_m = s*K_m - lambda^T G_m.
#
#   (1) zeta.  d f / d zeta = 1 - (1/M) sum_m phi*'(y_m - zeta), and
#       phi*' takes values in {1, kappa} with kappa <= 1, so the
#       derivative is >= 0 and zeta is pushed DOWN onto the feasibility
#       constraint:  zeta = Y - eta/2.
#   (2) eta.  Substituting that, f is nondecreasing in eta and the
#       infimum is at eta -> 0+.  eta is the multiplier on the TV
#       constraint, so eta* = 0 is exactly the redundancy recorded in
#       the closing paragraph of Lemma "contamination as constrained
#       TV": when kappa >= 1 - delta the mixture restriction already
#       exhausts the budget and the TV ball is REDUNDANT.
#   (3) Value.  f(0+) = (1-kappa)*Y + kappa*mean(y).
#
# So the inner dual is the UNCONSTRAINED convex program in lambda
# alone
#
#   F(lambda) = (1-kappa) * max_m y_m(lambda) + kappa * (1/M) sum_m y_m(lambda),
#
# which is just the primal read off directly: every candidate is
# P = kappa*F0 + (1-kappa)*R, so E_P[g] = kappa*E_F0[g] + (1-kappa)*E_R[g]
# and nature's best R puts all of its mass on the arg-max draw.  With
# the default tie kappa = 1 - delta and delta = 1 it is the
# Manski/Lee LP, i.e. TV(1), matching :TV there.
#
# Versus :TVmixC this removes the M linear constraints AND the eta = 0
# boundary degeneracy that made the reference program slow.  The max
# is smoothed by a shifted log-sum-exp with temperature TVMIX_TAU,
# which sits ABOVE the exact max, so both bounds stay
# outward-conservative.  REQUIRES 0 < delta <= 1 AND kappa >= 1 - delta
# (so the TV constraint is redundant and the reduction exact; for
# kappa < 1 - delta use :TVmixC, where the TV constraints are kept).
#
# F is unbounded below exactly when the primal is infeasible -- at a
# theta outside the (delta-dependent) identified set.  KNITRO then
# hits the lower_limit guard and inner_loop returns its -1e10
# sentinel: a genuine infeasibility signal, not a solver failure.
#
# Implementation detail: the Bundle keeps the (2+d)-dim x layout with
# x[1] a dummy fixed at 1.0 (the eta slot) and x[2] a dummy fixed at
# 0.0 (the zeta slot), x[3:end] = lambda, so warm starts, multiplier
# extraction and the envelope-gradient plumbing are shared with the
# other divergences.  The outer envelope weights are
# w_m = kappa/M + (1-kappa)*softmax_m: the first term is delivered by
# dPsi_TVmix_red! (constant kappa) through the shared -mean(dpsi .* dy)
# term, the second by tv_mu (stored post-solve), exactly as :TVac
# splits its weights.
# ------------------------------------------------------------------
# Per-draw contribution of the kappa*mean(y) term and its derivative;
# the (1-kappa)*max term is handled by the log-sum-exp / tv_mu path.
function Psi_TVmix_red!(out, a)
    kap = TVMIX_KAPPA[]
    @inbounds for i in eachindex(a)
        out[i] = kap * a[i]
    end
end

function dPsi_TVmix_red!(out, a)
    kap = TVMIX_KAPPA[]
    @inbounds for i in eachindex(a)
        out[i] = kap
    end
end

function ddPsi_TVmix_red!(out, a)
    @inbounds for i in eachindex(a)
        out[i] = 0.0
    end
end

# ------------------------------------------------------------------
# PURE Kullback-Leibler divergence (no chi^2 hybridisation):
# phi(x)  = x log x - x + 1                (x >= 0)
# phi*(y) = exp(y) - 1                     (everywhere)
# The exponent is clamped at PUREKL_ACAP (default 500; exp(500) ~
# 1.4e217, finite) so a bad line-search iterate cannot overflow to
# Inf; the clamp is far outside any optimal region (the zeta-FOC
# forces E[phi*'(a)] = 1).
# ------------------------------------------------------------------
function Psi_pureKL!(out, a)
    cap = PUREKL_ACAP[]
    @inbounds for i in eachindex(a)
        out[i] = exp(min(a[i], cap))
    end
    out .-= 1.0
end

function dPsi_pureKL!(out, a)
    cap = PUREKL_ACAP[]
    @inbounds for i in eachindex(a)
        out[i] = exp(min(a[i], cap))
    end
end

function ddPsi_pureKL!(out, a)
    cap = PUREKL_ACAP[]
    @inbounds for i in eachindex(a)
        out[i] = exp(min(a[i], cap))
    end
end

# ------------------------------------------------------------------
# PURE Pearson chi^2 divergence:
# phi(x)  = (x - 1)^2                      (x >= 0)
# phi*(y) = y + y^2/4    if y >= -2   (interior maximiser x = 1 + y/2)
# phi*(y) = -1           if y <  -2   (corner x = 0; -phi(0) = -1)
# C^1 at y = -2 (value -1, slope 0); second derivative jumps 0 -> 1/2.
# ------------------------------------------------------------------
function Psi_chi2!(out, a)
    @inbounds for i in eachindex(a)
        out[i] = a[i] >= -2.0 ? a[i] + 0.25 * a[i]^2 : -1.0
    end
end

function dPsi_chi2!(out, a)
    @inbounds for i in eachindex(a)
        out[i] = a[i] >= -2.0 ? 1.0 + 0.5 * a[i] : 0.0
    end
end

function ddPsi_chi2!(out, a)
    @inbounds for i in eachindex(a)
        out[i] = a[i] >= -2.0 ? 0.5 : 0.0
    end
end

# Dispatch helpers: a Bundle carries
# `divergence in (:KL_chi2, :TV, :TVmix, :TVmixC, :TVac, :KL, :chi2)`
# (default = CC23 hybrid).  :TVmix is the reduced program and :TVmixC
# its literal-dual reference; both are short-circuited in inner_value!,
# so the Psi!/ddPsi! entries below matter only for :TVmixC (:TVmix
# reaches dPsi! from the outer envelope gradient).
@inline function Psi!(out, a, divergence::Symbol)
    if divergence === :TV
        Psi_TV!(out, a)
    elseif divergence === :TVmix
        Psi_TVmix_red!(out, a)
    elseif divergence === :TVmixC
        Psi_TVmix!(out, a)
    elseif divergence === :TVac
        Psi_TVac!(out, a)
    elseif divergence === :KL
        Psi_pureKL!(out, a)
    elseif divergence === :chi2
        Psi_chi2!(out, a)
    else
        Psi_KL!(out, a)
    end
end
@inline function dPsi!(out, a, divergence::Symbol)
    if divergence === :TV
        dPsi_TV!(out, a)
    elseif divergence === :TVmix
        dPsi_TVmix_red!(out, a)
    elseif divergence === :TVmixC
        dPsi_TVmix!(out, a)
    elseif divergence === :TVac
        dPsi_TVac!(out, a)
    elseif divergence === :KL
        dPsi_pureKL!(out, a)
    elseif divergence === :chi2
        dPsi_chi2!(out, a)
    else
        dPsi_KL!(out, a)
    end
end
@inline function ddPsi!(out, a, divergence::Symbol)
    if divergence === :TV
        ddPsi_TV!(out, a)
    elseif divergence === :TVmix
        ddPsi_TVmix_red!(out, a)
    elseif divergence === :TVmixC
        ddPsi_TVmix!(out, a)
    elseif divergence === :TVac
        ddPsi_TVac!(out, a)
    elseif divergence === :KL
        ddPsi_pureKL!(out, a)
    elseif divergence === :chi2
        ddPsi_chi2!(out, a)
    else
        ddPsi_KL!(out, a)
    end
end

# ------------------------------------------------------------------
# Scrambled Halton sequence (Owen, 2017), ported from CC23.
# Produces n points in d dimensions, with optional seeding.
# ------------------------------------------------------------------
const _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

function randradinv(ind::AbstractVector{Int}, b::Int)
    b2r = 1.0 / b
    x = zeros(Float64, length(ind))
    res = copy(ind)
    while 1 - b2r < 1
        dig  = res .% b
        perm = shuffle(1:b) .- 1
        pdig = perm[dig .+ 1]
        x   .+= pdig .* b2r
        b2r /= b
        res .-= dig
        res .÷= b
    end
    return x
end

function rhalton(n::Int, d::Int; singleseed::Union{Nothing,Int} = nothing)
    x = zeros(n, d)
    seedvec = isnothing(singleseed) ? nothing : (1:d) .+ singleseed .- 1
    for j in 1:d
        isnothing(seedvec) || Random.seed!(seedvec[j])
        x[:, j] .= randradinv(collect(0:n-1), _PRIMES[j])
    end
    return x
end

"""
    tvb_generate_halton(M, u_dim; seed = nothing) -> Matrix{Float64}

Generate `M` scrambled-Halton points in `(0, 1)^u_dim` (Owen, 2017).
`u_dim` is capped at $(length(_PRIMES)) (one prime base per
dimension); for higher-dimensional latent draws supply `U` directly.
The user's moments function is responsible for mapping these uniform
coordinates into draws from the baseline distribution (e.g. through
quantile transforms).
"""
function tvb_generate_halton(M::Integer, u_dim::Integer; seed = nothing)
    1 <= u_dim <= length(_PRIMES) ||
        error("u_dim must be between 1 and $(length(_PRIMES)) " *
              "(one prime base per dimension); supply U directly for " *
              "higher-dimensional draws")
    M >= 2 || error("M must be at least 2")
    return rhalton(Int(M), Int(u_dim);
                   singleseed = seed === nothing ? nothing : Int(seed))
end

# ------------------------------------------------------------------
# ObjectiveBundle: encapsulates the inner-loop dual problem
# ("PsiObjectiveBundleExplicit" in CC23's terminology).
#
# moments! has signature  moments!(K, G, theta, U, obj)  and must fill
# K (M-vector: counterfactual values at the draws) and G (M x d moment
# functions) in place; `obj` is the Bundle itself, whose `gamma` field
# carries the arbitrary user payload.
# ------------------------------------------------------------------
@with_kw mutable struct Bundle{T}
    delta::Float64
    find_smallest::Bool

    gamma::T
    moments!::Function
    d::Int
    l::Int
    U::Array{Float64,2}
    M::Int = size(U, 1)

    # :KL_chi2 (CC23 hybrid), :TV (Palomba JMP, Lemma 35 + Remark 8),
    # :TVmix (TV + mixture constraint, kappa = 1 - delta by default;
    #         Section SA5; reduced program), :TVmixC (same, literal
    #         dual: reference, and the only mode with kappa decoupled
    #         from delta), :TVac (TV restricted to a.c. distributions,
    #         Remark SA-27), :KL / :chi2 (pure divergences)
    divergence::Symbol = :KL_chi2

    inner_opt::String
    outer_opt::String
    lower_limit::Float64 = -1e1
    use_cached_x::Bool   = false
    eta_min::Float64     = 1e-120

    # How to differentiate the moments w.r.t. theta in the outer
    # envelope gradient:
    #   :forwarddiff -- ForwardDiff through moments! (requires a
    #                   Dual-generic Julia function)
    #   :fd          -- one-sided finite differences (step fd_step);
    #                   the only automatic option for R moments
    #   :user        -- moments_jac(theta, U, obj) supplies the
    #                   Jacobian (stacked (M*(d+1)) x l matrix, or a
    #                   collection with components K (M x l) and
    #                   G (M x d x l))
    jac_mode::Symbol = :forwarddiff
    moments_jac::Union{Nothing,Function} = nothing
    fd_step::Float64 = 1e-6

    # cache buffers
    H::Array{Float64,2}     = hcat(zeros(M), ones(M), zeros(M, d))
    H_copy::Array{Float64,2}= hcat(zeros(M), ones(M), zeros(M, d))
    arg0::Vector{Float64}   = zeros(M)
    arg1::Vector{Float64}   = zeros(M)
    arg2::Vector{Float64}   = zeros(M)
    jac_h::Array{Float64,3} = zeros(M, d + 2, l)
    x::Vector{Float64}      = fill(NaN, 2 + d)
    dx_dtheta::Array{Float64,2} = zeros(2 + d, l)
    ddf::Array{Float64,2}   = zeros(2 + d, 2 + d)
    ddfxtheta::Array{Float64,2} = zeros(2 + d, l)

    # Pre-allocated COO buffers for the M linear feasibility constraints
    # used in the TV inner loop:  eta/2 + zeta + lambda^T G_m  >=  s*K_m,
    # i.e. row m has nonzeros in columns (eta=0, zeta=1, lambda_j=2..d+1).
    # Stored as Int32 so KNITRO's C API receives the right types.
    tv_con_idx::Vector{Int32}  = zeros(Int32, M * (2 + d))
    tv_var_idx::Vector{Int32}  = zeros(Int32, M * (2 + d))
    tv_coefs::Vector{Float64}  = zeros(Float64, M * (2 + d))
    tv_rhs::Vector{Float64}    = zeros(Float64, M)

    # Most recent KKT multipliers for the M TV feasibility constraints,
    # populated after every inner KNITRO solve and consumed by the outer
    # envelope-theorem gradient.
    tv_mu::Vector{Float64}     = zeros(Float64, M)
end

# ------------------------------------------------------------------
# Jacobian of the stacked moments [K; vec(G)] w.r.t. theta, as an
# (M*(d+1)) x l matrix (K rows first, then G in column-major order),
# used by the outer envelope-theorem gradient.  Dispatches on
# Q.jac_mode; see the Bundle definition.
# ------------------------------------------------------------------
function _kg_eval(Q::Bundle, theta::AbstractVector{Float64})
    K = zeros(Q.M)
    G = zeros(Q.M, Q.d)
    Q.moments!(K, G, theta, Q.U, Q)
    return vcat(K, vec(G))
end

function _normalize_jac(raw, M::Int, d::Int, l::Int)
    if raw isa AbstractArray{<:Real}
        length(raw) == M * (d + 1) * l ||
            error("user gradient returned $(length(raw)) elements; " *
                  "expected the stacked (M*(d+1)) x l Jacobian with " *
                  "M*(d+1)*l = $(M * (d + 1) * l) elements")
        return Matrix{Float64}(reshape(collect(Float64, vec(raw)),
                                       M * (d + 1), l))
    end
    JK = nothing
    JG = nothing
    if raw isa AbstractDict
        for (k, v) in raw
            s = string(k)
            s == "K" && (JK = v)
            s == "G" && (JG = v)
        end
    elseif raw isa NamedTuple
        JK = haskey(raw, :K) ? raw[:K] : nothing
        JG = haskey(raw, :G) ? raw[:G] : nothing
    elseif raw isa Tuple && length(raw) == 2
        JK, JG = raw
    end
    (JK === nothing || JG === nothing) &&
        error("user gradient must return either the stacked " *
              "(M*(d+1)) x l Jacobian or a collection with components " *
              "`K` (M x l) and `G` (M x d x l); got $(typeof(raw))")
    length(JK) == M * l ||
        error("gradient component `K` has $(length(JK)) elements; " *
              "expected M*l = $(M * l)")
    length(JG) == M * d * l ||
        error("gradient component `G` has $(length(JG)) elements; " *
              "expected M*d*l = $(M * d * l)")
    Jk = reshape(collect(Float64, vec(JK)), M, l)
    Jg = reshape(collect(Float64, vec(JG)), M * d, l)
    return vcat(Jk, Jg)
end

function _moments_jacobian(Q::Bundle, theta::AbstractVector)
    th = collect(Float64, theta)
    if Q.jac_mode === :user
        Q.moments_jac === nothing &&
            error("jac_mode = :user requires a moments_jac function")
        return _normalize_jac(Q.moments_jac(th, Q.U, Q), Q.M, Q.d, Q.l)
    elseif Q.jac_mode === :fd
        # One-sided forward differences; uses scratch buffers, so the
        # cached H (moments at the current theta) is left untouched.
        h = Q.fd_step
        base = _kg_eval(Q, th)
        J = zeros(Q.M * (Q.d + 1), Q.l)
        for i in 1:Q.l
            th[i] += h
            J[:, i] .= (_kg_eval(Q, th) .- base) ./ h
            th[i] -= h
        end
        return J
    else
        function kg(t)
            Ktmp = zeros(eltype(t), Q.M)
            Gtmp = zeros(eltype(t), Q.M, Q.d)
            Q.moments!(Ktmp, Gtmp, t, Q.U, Q)
            return hcat(Ktmp, Gtmp)
        end
        return ForwardDiff.jacobian(kg, th)
    end
end

# ------------------------------------------------------------------
# Inner-loop objective/gradient/Hessian.
# f(eta, zeta, lambda) = eta * (E_F*[Psi(((-1)^fs * K - zeta - lambda'G)/eta)] + delta) + zeta
# ------------------------------------------------------------------
function inner_value!(Q::Bundle, x::AbstractVector;
                     grad::AbstractVector = Float64[],
                     hess::AbstractVector = Float64[])
    Q.divergence === :TVac && return inner_value_tvac!(Q, x; grad = grad, hess = hess)
    Q.divergence === :TVmix && return inner_value_tvmix!(Q, x; grad = grad, hess = hess)
    # :TVmixC ties the conjugate's truncation point to the budget
    # (or to the user's kappa override)
    Q.divergence === :TVmixC && (TVMIX_KAPPA[] = _effective_kappa(Q.delta))
    @unpack delta, H, arg0, arg1, M, d, find_smallest, lower_limit, divergence = Q
    eta = x[1]; zeta = x[2]
    lam = @view x[3:end]

    BLAS.gemv!('N', 1.0, @view(H[:, 1:2 + d]),
               vcat((-1.0)^find_smallest, -zeta, -lam) ./ eta, 0.0, arg0)
    Psi!(arg1, arg0, divergence)
    f = eta * (sum(arg1) / M + delta) + zeta

    if length(grad) > 0
        dPsi!(arg1, arg0, divergence)
        grad[1] = -dot(arg0, arg1) / M + (f - zeta) / eta
        grad[2] = 1.0 - sum(arg1) / M
        @views BLAS.gemv!('T', -1 / M, H[:, 3:2 + d], arg1, 0.0, grad[3:end])
    end

    if length(hess) > 0
        ddPsi!(arg2, arg0, divergence)
        Hc = Q.H_copy
        @views Hc[:, 2:2 + d] .= H[:, 2:2 + d]
        @views Hc[:, 1]       .= arg0
        @views Hc[:, 1:2 + d] .*= sqrt.(arg2)
        @views BLAS.gemm!('T', 'N', 1 / (eta * M),
                          Hc[:, 1:2 + d], Hc[:, 1:2 + d], 0.0, Q.ddf)
        k = 1
        for i in 1:size(Q.ddf, 2)
            for j in i:size(Q.ddf, 2)
                hess[k] = Q.ddf[i, j]; k += 1
            end
        end
    end

    return f <= lower_limit ? -KNITRO.KN_INFINITY : f
end

# ------------------------------------------------------------------
# TV^<< inner objective/gradient (see the TVac block above):
#   F(v, lambda) = delta * LSE_tau(y) + (1 - delta) * v
#                  + (1/M) sum_m Psi_TVac(y_m - v),
# with y_m = s*K_m - lambda^T G_m and LSE_tau the max-shifted
# log-sum-exp.  x = (dummy = 1, v, lambda); the dummy gradient is 0.
# Convex in (v, lambda): LSE and Psi_TVac are convex, y is affine.
# ------------------------------------------------------------------
function inner_value_tvac!(Q::Bundle, x::AbstractVector;
                           grad::AbstractVector = Float64[],
                           hess::AbstractVector = Float64[])
    @unpack delta, H, arg0, arg1, arg2, M, d, find_smallest, lower_limit = Q
    v   = x[2]
    lam = @view x[3:end]
    s   = (-1.0) ^ find_smallest
    tau = TVAC_TAU[]

    # arg0 <- a = y - v  (H columns: [K | 1 | G])
    BLAS.gemv!('N', 1.0, @view(H[:, 1:2 + d]), vcat(s, -v, -lam), 0.0, arg0)

    # shifted log-sum-exp of y = a + v; arg2 holds unnormalised weights
    abar = maximum(arg0)
    ssum = 0.0
    @inbounds for m in 1:M
        arg2[m] = exp((arg0[m] - abar) / tau)
        ssum   += arg2[m]
    end
    lse = (abar + v) + tau * log(ssum)

    Psi_TVac!(arg1, arg0)
    f = delta * lse + (1.0 - delta) * v + sum(arg1) / M

    if length(grad) > 0
        dPsi_TVac!(arg1, arg0)                        # p_m = relu_eps'(a_m)
        grad[1] = 0.0                                 # dummy (fixed) variable
        grad[2] = (1.0 - delta) - sum(arg1) / M
        # q_m = delta * w_m + p_m / M;  grad_lambda = -G' q
        @inbounds for m in 1:M
            arg1[m] = delta * arg2[m] / ssum + arg1[m] / M
        end
        @views BLAS.gemv!('T', -1.0, H[:, 3:2 + d], arg1, 0.0, grad[3:end])
    end

    # hess intentionally not provided (hessopt=2 BFGS, as for the others)

    return f <= lower_limit ? -KNITRO.KN_INFINITY : f
end

# ------------------------------------------------------------------
# TV^mix reduced inner objective/gradient (see the TVmix block above):
#   F(lambda) = (1-kappa) * LSE_tau(y) + kappa * mean(y),
# with y_m = s*K_m - lambda^T G_m and LSE_tau the max-shifted
# log-sum-exp; kappa = 1 - delta unless overridden.  x = (dummy = 1,
# dummy = 0, lambda); both dummy gradients are 0.  Convex in lambda:
# LSE and mean are convex, y affine.
# ------------------------------------------------------------------
function inner_value_tvmix!(Q::Bundle, x::AbstractVector;
                            grad::AbstractVector = Float64[],
                            hess::AbstractVector = Float64[])
    @unpack delta, H, arg0, arg1, arg2, M, d, find_smallest, lower_limit = Q
    lam = @view x[3:end]
    s   = (-1.0) ^ find_smallest
    kap = _effective_kappa(delta)
    TVMIX_KAPPA[] = kap                  # consumed by dPsi_TVmix_red!
    w   = 1.0 - kap                      # weight on the max/LSE term
    tau = TVMIX_TAU[]

    # arg0 <- y = s*K - lambda' G   (H columns: [K | 1 | G]; the zeta
    # slot gets coefficient 0, so x[2] cannot enter the objective)
    BLAS.gemv!('N', 1.0, @view(H[:, 1:2 + d]), vcat(s, 0.0, -lam), 0.0, arg0)

    # shifted log-sum-exp of y; arg2 holds unnormalised soft-max weights
    ybar = maximum(arg0)
    ssum = 0.0
    @inbounds for m in 1:M
        arg2[m] = exp((arg0[m] - ybar) / tau)
        ssum   += arg2[m]
    end
    lse = ybar + tau * log(ssum)
    f   = w * lse + kap * (sum(arg0) / M)

    if length(grad) > 0
        grad[1] = 0.0                                  # dummy (fixed) eta slot
        grad[2] = 0.0                                  # dummy (fixed) zeta slot
        # w_m = (1-kappa) * softmax_m + kappa/M;  grad_lambda = -G' w
        @inbounds for m in 1:M
            arg1[m] = w * arg2[m] / ssum + kap / M
        end
        @views BLAS.gemv!('T', -1.0, H[:, 3:2 + d], arg1, 0.0, grad[3:end])
    end

    # hess intentionally not provided (hessopt=2 BFGS, as for the others)

    return f <= lower_limit ? -KNITRO.KN_INFINITY : f
end

# Call inner optimizer once at a given theta; sign-flip if find_smallest
function inner_loop(Q::Bundle, theta::AbstractVector)
    Q.moments!(@view(Q.H[:, 1]), @view(Q.H[:, 3:2 + Q.d]), theta, Q.U, Q)
    Q.H[:, 2] .= 1.0

    status, val, x = run_inner_KNITRO(Q)

    # Up to 3 refinements with warm start
    iter = 1
    while iter <= 3 && status in (-100, -101, -102, -103) && val >= Q.lower_limit
        Q.x .= x
        cached = Q.use_cached_x
        Q.use_cached_x = true
        status, val, x = run_inner_KNITRO(Q)
        Q.use_cached_x = cached
        iter += 1
    end

    if status == 0 || (status in (-100, -101, -103) && val >= Q.lower_limit)
        Q.x .= x
    else
        Q.x .= NaN
        val  = -1e10
    end

    return (Q.find_smallest ? -val : val), x, status
end

# ------------------------------------------------------------------
# KNITRO callbacks
# ------------------------------------------------------------------
# Function-only callback (for use without eval_fcga=yes).
function inner_cb_f(_, _, req, res, user)
    res.obj[1] = inner_value!(user, req.x)
    return 0
end

# Separate gradient callback. KNITRO requests this via KN_set_cb_grad.
function inner_cb_g(_, _, req, res, user)
    inner_value!(user, req.x; grad = res.objGrad)
    return 0
end

# Build the COO representation of the M linear feasibility constraints
# used in TV mode.  Row m has the form
#     0.5 * eta + 1.0 * zeta + sum_j G[m,j] * lambda_j  >=  s * K_m,
# where s = (-1)^find_smallest.  KNITRO uses 0-based variable indices
# (eta=0, zeta=1, lambda_j=j+1).  The buffers are allocated once in the
# Bundle and refilled in place at every inner solve, since K and G change
# with theta.
function _fill_tv_constraints!(Q::Bundle)
    M, d = Q.M, Q.d
    nnz_per_row = 2 + d
    s = (-1.0) ^ Q.find_smallest

    @inbounds for m in 1:M
        base = (m - 1) * nnz_per_row
        # eta column
        Q.tv_con_idx[base + 1] = Int32(m - 1)
        Q.tv_var_idx[base + 1] = Int32(0)
        Q.tv_coefs[base + 1]   = 0.5
        # zeta column
        Q.tv_con_idx[base + 2] = Int32(m - 1)
        Q.tv_var_idx[base + 2] = Int32(1)
        Q.tv_coefs[base + 2]   = 1.0
        # lambda_j columns
        for j in 1:d
            Q.tv_con_idx[base + 2 + j] = Int32(m - 1)
            Q.tv_var_idx[base + 2 + j] = Int32(1 + j)
            Q.tv_coefs[base + 2 + j]   = Q.H[m, 2 + j]   # G[m, j]
        end
        Q.tv_rhs[m] = s * Q.H[m, 1]                      # s * K_m
    end
    return nothing
end

function run_inner_KNITRO(Q::Bundle)
    _KNITRO_OK[] ||
        error("KNITRO.jl is not available; solves require the Artelys " *
              "KNITRO solver and a valid license. Load error: " *
              _KNITRO_ERR[])
    _t0 = TIMING_ON[] ? time() : 0.0
    kc = KNITRO.KN_new()
    nvar = Cint(2 + Q.d)
    KNITRO.KN_add_vars(kc, nvar, C_NULL)

    # TVac's reduced dual requires a TV radius: unbounded below for delta > 1.
    if Q.divergence === :TVac && !(0.0 < Q.delta <= 1.0)
        error(":TVac requires 0 < delta <= 1 (TV radius); got delta = $(Q.delta)")
    end
    # TVmix/TVmixC involve the mixing weight kappa (= 1 - delta unless
    # overridden): need 0 < delta <= 1, and for the reduced :TVmix the
    # exactness of the reduction additionally needs kappa >= 1 - delta.
    if Q.divergence in (:TVmix, :TVmixC)
        (0.0 < Q.delta <= 1.0) ||
            error("$(Q.divergence) requires 0 < delta <= 1; got delta = $(Q.delta)")
        kap = _effective_kappa(Q.delta)
        (0.0 <= kap <= 1.0) ||
            error("$(Q.divergence) requires a mixing weight kappa in [0, 1]; got kappa = $kap")
        if Q.divergence === :TVmix && kap < 1.0 - Q.delta - 1e-12
            error(":TVmix's reduced program requires kappa >= 1 - delta " *
                  "(the TV constraint must be redundant, Lemma SA-40(ii) and its closing paragraph); " *
                  "got kappa = $kap < 1 - delta = $(1.0 - Q.delta). " *
                  "Use :TVmixC for kappa < 1 - delta.")
        end
        TVMIX_KAPPA[] = kap
    end
    # TVac has no eta variable: x[1] is a dummy fixed at 1.0.
    # TVmix has neither eta nor zeta: x[1] = 1.0 and x[2] = 0.0 are dummies.
    lb = vcat(Q.divergence in (:TVac, :TVmix) ? 1.0 : Q.eta_min,
              Q.divergence === :TVmix ? 0.0 : -KNITRO.KN_INFINITY,
              fill(-KNITRO.KN_INFINITY, Q.d))
    KNITRO.KN_set_var_lobnds_all(kc, lb)
    if Q.divergence === :TVac
        KNITRO.KN_set_var_upbnds_all(kc, vcat(1.0, fill(KNITRO.KN_INFINITY, 1 + Q.d)))
    elseif Q.divergence === :TVmix
        KNITRO.KN_set_var_upbnds_all(kc, vcat(1.0, 0.0, fill(KNITRO.KN_INFINITY, Q.d)))
    end

    # Feasible cold-start for TV: pick zeta large enough so that
    # eta/2 + zeta >= s*K_m for all m (with lambda = 0, eta tiny).
    # That makes the M linear constraints trivially feasible.
    # For TVac (unconstrained) start the floor v at the delta-quantile of
    # s*K, the lambda = 0 first-order condition of F in v.
    # TVmix (unconstrained, lambda-only) starts at lambda = 0, where
    # F is finite.
    init = if Q.use_cached_x && all(isfinite.(Q.x)) && norm(Q.x) < 1e6
        Q.x
    elseif Q.divergence === :TVmix
        vcat(1.0, 0.0, zeros(Q.d))
    elseif Q.divergence in (:TV, :TVmixC)
        s = (-1.0) ^ Q.find_smallest
        zeta_init = maximum(s .* @view(Q.H[:, 1])) + 1e-3
        vcat(1e-3, zeta_init, zeros(Q.d))
    elseif Q.divergence === :TVac
        s = (-1.0) ^ Q.find_smallest
        v_init = quantile(s .* @view(Q.H[:, 1]), clamp(Q.delta, 0.01, 0.99))
        vcat(1.0, v_init, zeros(Q.d))
    else
        vcat(Q.eta_min, zeros(1 + Q.d))
    end
    KNITRO.KN_set_var_primal_init_values_all(kc, init)

    # TV/TVmixC: add the M linear feasibility constraints
    # eta/2 + zeta + lambda^T G >= s*K (same recession 1/2 in both).
    # :TVmix eliminates them analytically (see the TVmix block).
    if Q.divergence in (:TV, :TVmixC)
        _fill_tv_constraints!(Q)
        cidx = zeros(Int32, Q.M)
        KNITRO.KN_add_cons(kc, Cint(Q.M), cidx)
        KNITRO.KN_set_con_lobnds_all(kc, copy(Q.tv_rhs))
        KNITRO.KN_set_con_upbnds_all(kc, fill(KNITRO.KN_INFINITY, Q.M))
        KNITRO.KN_add_con_linear_struct(kc,
                                        Cint(length(Q.tv_coefs)),
                                        Q.tv_con_idx, Q.tv_var_idx, Q.tv_coefs)
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, Cint[], inner_cb_f)
    KNITRO.KN_set_cb_user_params(kc, cb, Q)
    # Register analytic gradient separately so KNITRO doesn't fall back to FD.
    KNITRO.KN_set_cb_grad(kc, cb, inner_cb_g)

    KNITRO.KN_load_param_file(kc, Q.inner_opt)

    KNITRO.KN_solve(kc)
    status, objval, xsol, _ = KNITRO.KN_get_solution(kc)
    if Q.divergence in (:TV, :TVmixC)
        # Save Lagrange multipliers for the M TV feasibility constraints
        # so the outer loop can include them in the envelope-theorem
        # gradient (see outer_cb_g).  KNITRO uses the convention
        # L = f + lambda^T c, which gives negative multipliers for active
        # `c >= 0` constraints; we flip sign to match L = f - mu^T c with
        # mu >= 0 used in the gradient derivation.
        KNITRO.KN_get_con_dual_values_all(kc, Q.tv_mu)
        Q.tv_mu .*= -1.0
    elseif Q.divergence === :TVac
        # The Manski term delta * LSE_tau(y) plays the role the M
        # feasibility constraints play under :TV; its envelope weight is
        # tv_mu = delta * softmax(y / tau) evaluated at the solution.
        s = (-1.0) ^ Q.find_smallest
        tau = TVAC_TAU[]
        BLAS.gemv!('N', 1.0, @view(Q.H[:, 1:2 + Q.d]),
                   vcat(s, -xsol[2], -xsol[3:end]), 0.0, Q.arg0)
        abar = maximum(Q.arg0)
        ssum = 0.0
        @inbounds for m in 1:Q.M
            Q.tv_mu[m] = exp((Q.arg0[m] - abar) / tau)
            ssum      += Q.tv_mu[m]
        end
        Q.tv_mu .*= Q.delta / ssum
    elseif Q.divergence === :TVmix
        # Same split as :TVac: tv_mu carries the (1-kappa)*LSE_tau(y)
        # weights, while the kappa*mean(y) weights reach the outer
        # gradient through dPsi_TVmix_red! (constant kappa).  Here the
        # zeta slot is a dummy fixed at 0, so y = s*K - lambda'G directly.
        s = (-1.0) ^ Q.find_smallest
        kap = _effective_kappa(Q.delta)
        w   = 1.0 - kap
        tau = TVMIX_TAU[]
        BLAS.gemv!('N', 1.0, @view(Q.H[:, 1:2 + Q.d]),
                   vcat(s, 0.0, -xsol[3:end]), 0.0, Q.arg0)
        ybar = maximum(Q.arg0)
        ssum = 0.0
        @inbounds for m in 1:Q.M
            Q.tv_mu[m] = exp((Q.arg0[m] - ybar) / tau)
            ssum      += Q.tv_mu[m]
        end
        Q.tv_mu .*= w / ssum
        # Report the EXACT objective at the converged lambda instead of
        # the smoothed one.  KNITRO optimises the log-sum-exp surrogate
        # (smooth => fast, reliable), but LSE_tau overstates the max by
        # up to (1-kappa)*tau*log(#near-max draws), FIRST order in tau.
        # Since F_exact <= F_smooth pointwise and the dual value is
        # min_lambda F_exact <= F_exact(lambda*), evaluating the exact
        # objective at the smoothed arg-min stays a valid outward bound
        # (too wide, never too narrow, on both sides) while cutting the
        # error to the SECOND-order lambda*-suboptimality -- F is flat
        # at its minimum, so this is ~1e-5 in practice.
        # The INNER solve stays fully self-consistent (KNITRO sees the
        # smoothed objective and its matching gradient throughout); only
        # the OUTER loop then reads an exact inner value whose envelope
        # gradient is still the surrogate's, an O(tau) mismatch.  That is
        # harmless here because the outer search only PROPOSES theta:
        # tvb_solve re-derives every reported bound by a strict inner
        # re-solve at the selected theta, never taking the outer value.
        if status == 0 || status in (-100, -101, -103)
            objval = w * ybar + kap * (sum(Q.arg0) / Q.M)
        end
    end
    KNITRO.KN_free(kc)
    _record_inner!(TIMING_ON[] ? time() - _t0 : 0.0)
    return status, objval, xsol
end

# ------------------------------------------------------------------
# Outer loop: minimize/maximize K(theta) over theta, using KNITRO
# with a small multi-start.
# ------------------------------------------------------------------
# Outer loop callbacks with analytic gradient via envelope theorem.
# KNITRO minimizes  f(theta)  =  -objSol(theta)  (i.e. for
# find_smallest=true  --> min K_delta;   find_smallest=false --> max Kbar).
# The gradient we supply is
#    df/dtheta = -(d f_min / dtheta)  (envelope theorem; sign verified by FD).
# ------------------------------------------------------------------
function outer_cb_f(_, _, req, res, user::Bundle)
    theta = req.x
    objSol, xstar, _ = inner_loop_internal(user, theta)
    res.obj[1] = -objSol
    user.H_copy[1, 1] = objSol  # stash in a spare cell for gradient callback reuse
    return 0
end

function outer_cb_g(_, _, req, res, user::Bundle)
    theta = req.x
    # Re-run the inner solve at `theta` (cheap with warm start) so we have
    # the current (eta, zeta, lambda); using KNITRO warm-start mechanism.
    user.use_cached_x = true
    objSol, xstar, _ = inner_loop_internal(user, theta)
    user.use_cached_x = false

    eta  = xstar[1]
    zeta = xstar[2]
    lam  = @view xstar[3:end]
    sign_k = (-1.0) ^ user.find_smallest

    # Jacobian of [K; G] wrt theta (ForwardDiff, finite differences, or
    # user-supplied, per user.jac_mode).
    J = _moments_jacobian(user, theta)           # (M*(d+1)) x l

    arg = (sign_k .* user.H[:, 1] .- zeta .- user.H[:, 3:2 + user.d] * lam) ./ eta
    dpsi = similar(arg); dPsi!(dpsi, arg, user.divergence)

    Jk  = @view J[1:user.M, :]
    Jg  = @view J[user.M + 1:end, :]
    Jg3 = reshape(Jg, user.M, user.d, user.l)

    # For TV/TVmixC, augment the envelope-theorem gradient with the inner
    # Lagrange multipliers on the M feasibility constraints; for TVac and
    # TVmix, with the soft-max weights of the Manski term (stored in
    # tv_mu by the inner solve); otherwise (KL/chi^2) tv_mu is zero and
    # this term drops out.  Note for TVac the shared `arg` formula gives
    # a = y - v since eta = xstar[1] = 1 and zeta = xstar[2] = v, so
    # dpsi = relu_eps'(a); for TVmix eta = 1 and zeta = 0, so arg = y and
    # dpsi = kappa, the constant weight of the mean(y) term.
    mu = user.divergence in (:TV, :TVmixC, :TVmix, :TVac) ? user.tv_mu : zero(user.tv_mu)

    # KNITRO outer objective is  f = -objSol, so df/dtheta = -d(objSol)/dtheta.
    # objSol == f_min (the inner minimum),
    # envelope thm:  d f_min / dtheta_i = sum_m (Psi'(a_m)/M + mu_m) * d y_m/d theta_i.
    for i in 1:user.l
        contrib_k = sign_k .* @view(Jk[:, i])
        contrib_g = @view(Jg3[:, :, i]) * lam
        dy_dtheta = contrib_k .- contrib_g
        res.objGrad[i] = -mean(dpsi .* dy_dtheta) - dot(mu, dy_dtheta)
    end
    return 0
end

function inner_loop_internal(Q::Bundle, theta::AbstractVector)
    Q.moments!(@view(Q.H[:, 1]), @view(Q.H[:, 3:2 + Q.d]), theta, Q.U, Q)
    Q.H[:, 2] .= 1.0

    status, val, x = run_inner_KNITRO(Q)
    iter = 1
    while iter <= 3 && status in (-100, -101, -102, -103) && val >= Q.lower_limit
        Q.x .= x
        cached = Q.use_cached_x
        Q.use_cached_x = true
        status, val, x = run_inner_KNITRO(Q)
        Q.use_cached_x = cached
        iter += 1
    end

    if status == 0 || (status in (-100, -101, -103) && val >= Q.lower_limit)
        Q.x .= x
        return val, x, status
    else
        Q.x .= NaN
        return -1e10, x, status
    end
end

function outer_loop(Q::Bundle, theta_lb, theta_ub, theta_init)
    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, Q.outer_opt)

    KNITRO.KN_add_vars(kc, Cint(length(theta_init)), C_NULL)
    KNITRO.KN_set_var_lobnds_all(kc, theta_lb)
    KNITRO.KN_set_var_upbnds_all(kc, theta_ub)
    KNITRO.KN_set_var_primal_init_values_all(kc, theta_init)

    cb = KNITRO.KN_add_eval_callback(kc, true, Cint[], outer_cb_f)
    KNITRO.KN_set_cb_user_params(kc, cb, Q)
    KNITRO.KN_set_cb_grad(kc, cb, outer_cb_g)

    KNITRO.KN_solve(kc)
    status, obj_raw, theta_star, _ = KNITRO.KN_get_solution(kc)
    runtime_ref = Ref{Cdouble}(0.0)
    KNITRO.KN_get_solve_time_real(kc, runtime_ref)
    runtime = runtime_ref[]
    KNITRO.KN_free(kc)

    # KNITRO sees -objSol, so the "value" is -obj_raw (in find_smallest sign).
    # We return the counterfactual bound in its natural sign (K_delta or Kbar_delta).
    kappa = Q.find_smallest ? obj_raw : -obj_raw
    return kappa, theta_star, status, runtime
end

# ------------------------------------------------------------------
# Outer loop via Optim.jl with analytic gradient (envelope theorem).
# Avoids nested KNITRO contexts and the segfault observed with some
# KNITRO.jl versions, and is ~10x faster than Optim's own
# finite-difference gradient.
# ------------------------------------------------------------------
function outer_loop_optim(Q::Bundle, theta_lb, theta_ub, theta_init;
                          time_limit::Float64 = 60.0,
                          iterations::Int    = 100,
                          outer_iterations::Int = 3)
    sign_out = Q.find_smallest ? 1.0 : -1.0     # MINIMIZE sign_out * kappa

    # We need the analytic gradient of the outer objective w.r.t. theta.
    # By the envelope theorem,
    #   d/dtheta { Kbar(theta) } = (partial L / partial theta) at x*(theta),
    # L(x, theta) = eta * E[phi*(arg)] + eta*delta + zeta,
    # arg = (sign_k * k(U, theta) - zeta - lambda' g(U, theta)) / eta.
    # => partial L / partial theta
    #    = E[dphi*(arg) * (sign_k * grad_theta k - lambda' grad_theta g)].
    x_last = Ref{Vector{Float64}}(Float64[])
    function f_and_grad!(F, G, theta)
        objSol, xstar, _ = inner_loop_internal(Q, theta)
        x_last[] = copy(xstar)
        kappa = Q.find_smallest ? -objSol : objSol

        if G !== nothing
            eta = xstar[1]
            zeta = xstar[2]
            lam  = xstar[3:end]
            sign_k = (-1.0) ^ Q.find_smallest

            # Jacobian (M*(d+1)) x l of [k; g] w.r.t. theta.
            J = _moments_jacobian(Q, theta)

            # Recompute arg and dpsi at x* (already available in Q.arg1
            # after inner_value!, but safer to recompute).
            arg = (sign_k * Q.H[:, 1] .- zeta .- Q.H[:, 3:2+Q.d] * lam) ./ eta
            dpsi_vec = similar(arg)
            dPsi!(dpsi_vec, arg, Q.divergence)

            # For TV, the inner has M linear feasibility constraints whose
            # KKT multipliers contribute to the envelope-theorem gradient.
            # `tv_mu[m]` is the multiplier on
            #   c_m := eta/2 + zeta + lambda^T G_m - s*K_m >= 0,
            # and  d c_m / d theta_i = -d y_m / d theta_i.  So the
            # inner-Lagrangian gradient w.r.t. theta is
            #   sum_m (Psi'(a_m)/M + mu_m) * d y_m / d theta_i.
            # For TVac and TVmix the same formula applies with mu = the
            # soft-max weights of the Manski term (stored in tv_mu by
            # the inner solve); TVmixC shares the TV constraint
            # structure, so its multipliers enter identically.  For
            # other runs, Q.tv_mu stays zero, leaving the gradient
            # unchanged.
            mu_vec = Q.divergence in (:TV, :TVmixC, :TVmix, :TVac) ? Q.tv_mu : zero(Q.tv_mu)

            # Row order of J: K rows first (M), then G in column-major
            # order (M*d), matching hcat(K, G) flattened column-major.
            Jmat = J   # (M*(d+1)) x l
            Jk = @view Jmat[1:Q.M, :]                           # M x l for K
            Jg = @view Jmat[Q.M+1:end, :]                       # (M*d) x l for G
            Jg3 = reshape(Jg, Q.M, Q.d, Q.l)                    # M x d x l

            # By the envelope theorem,
            #   d f_min / d theta = mean( dpsi * (sign_k * dk/dtheta - lambda' dg/dtheta) ).
            # Our outer objective f obeys:
            #   find_smallest=true :  f = K_delta =  -f_min   =>  df/dtheta = -df_min/dtheta
            #   find_smallest=false:  f = -Kbar_d =  -f_min   =>  df/dtheta = -df_min/dtheta
            # so in both cases df/dtheta = -(df_min/dtheta), numerically verified by FD.
            for i in 1:Q.l
                contrib_k = sign_k .* @view(Jk[:, i])
                contrib_g = @view(Jg3[:, :, i]) * lam           # M vector
                G[i] = -mean(dpsi_vec .* (contrib_k .- contrib_g)) -
                       dot(mu_vec, contrib_k .- contrib_g)
            end
        end
        return sign_out * kappa
    end

    lower = theta_lb; upper = theta_ub
    result = Optim.optimize(NLSolversBase.only_fg!(f_and_grad!),
                            lower, upper, theta_init,
                            Optim.Fminbox(Optim.LBFGS()),
                            Optim.Options(iterations = iterations,
                                          outer_iterations = outer_iterations,
                                          time_limit = time_limit,
                                          show_trace = false))
    theta_star = Optim.minimizer(result)
    f_star     = Optim.minimum(result)
    kappa      = sign_out * f_star
    # Optim treats time-limit / iteration-limit termination as "not converged"
    # (returns false), even when f_star is a perfectly good local optimum.
    # Treat any finite outcome as flag = 0 so outer_loop_multi accepts it; we
    # still surface a non-zero flag for non-finite or NaN values.
    flag = isfinite(f_star) ? 0 : 1
    return kappa, theta_star, flag, Optim.time_run(result)
end

function outer_loop_multi(Q::Bundle, theta_lb, theta_ub, theta_init;
                          maxsolves::Int = 10, startptrange::Float64 = 0.01,
                          use_optim::Bool = false,
                          time_limit::Float64 = 60.0,
                          iterations::Int    = 100,
                          outer_iterations::Int = 3,
                          verbose::Bool = true,
                          rng_seed::Union{Nothing,Int} = 1234)
    Theta = zeros(Q.l, maxsolves)
    kappa = zeros(maxsolves)
    flag  = zeros(Int, maxsolves)
    rng_seed === nothing || Random.seed!(rng_seed)
    for i in 1:maxsolves
        Q.x .= NaN                              # reset inner warm start
        init = copy(theta_init)
        if i > 1
            init .+= (rand(Q.l) .* 2.0 .- 1.0) .* startptrange
            init .= max.(min.(init, theta_ub), theta_lb)
        end
        if use_optim
            (kappa[i], Theta[:, i], flag[i], runtime) = outer_loop_optim(
                Q, theta_lb, theta_ub, init;
                time_limit = time_limit,
                iterations = iterations,
                outer_iterations = outer_iterations)
        else
            (kappa[i], Theta[:, i], flag[i], runtime) = outer_loop(
                Q, theta_lb, theta_ub, init)
        end
        if verbose
            @printf("  restart %2d  flag=%5d  kappa=%+9.5f  time=%6.2fs\n",
                    i, flag[i], kappa[i], runtime)
            flush(stdout)
        end
    end

    # Selection hierarchy (matches CC23 outer_loop_functions.jl):
    #   1. locally optimal (flag == 0)
    #   2. feasible, near-optimal (flag in {-100, -101, -103})
    #   3. feasible but iteration-limited (flag in {-400, -401, -402})
    #   4. otherwise: unfeasible -- return a placeholder bound.
    function pick_best(allowed_flags::Tuple{Vararg{Int}})
        idx = findall(f -> f in allowed_flags, flag)
        isempty(idx) && return nothing
        κ_ok = kappa[idx]
        # find_smallest=true  => smallest κ wins
        # find_smallest=false => largest κ wins
        return idx[Q.find_smallest ? argmin(κ_ok) : argmax(κ_ok)]
    end
    best_ix = pick_best((0,))
    best_ix === nothing && (best_ix = pick_best((-100, -101, -103)))
    best_ix === nothing && (best_ix = pick_best((-400, -401, -402)))
    if best_ix === nothing
        return (Q.find_smallest ? +1e10 : -1e10, copy(theta_init), 999)
    end
    return (kappa[best_ix], Theta[:, best_ix], flag[best_ix])
end

# ===================================================================
# R bridge helpers.
#
# An R moments function (bridged by JuliaCall/RCall as a Julia
# callable) has signature  f(theta, U)  and must return a collection
# with components `K` (numeric, length M) and `G` (numeric, M x d).
# An R gradient function has signature  f(theta, U)  and must return
# either the stacked (M*(d+1)) x l Jacobian or a collection with
# components `K` (M x l) and `G` (M x d x l).
# (The R package curries the user-level gamma payload into these
# closures before assigning them to Julia.)
# ===================================================================
function _split_kg_result(res, nK::Int, nG::Int)
    K = nothing
    G = nothing
    if res isa AbstractDict
        for (k, v) in res
            s = string(k)
            s == "K" && (K = v)
            s == "G" && (G = v)
        end
    elseif res isa NamedTuple
        K = haskey(res, :K) ? res[:K] : nothing
        G = haskey(res, :G) ? res[:G] : nothing
    elseif res isa Tuple && length(res) == 2
        K, G = res
    end
    (K === nothing || G === nothing) &&
        error("the moments function must return a named collection with " *
              "components `K` (length M) and `G` (M x d); got $(typeof(res))")
    length(K) == nK ||
        error("moments component `K` has length $(length(K)); expected $nK")
    length(G) == nG ||
        error("moments component `G` has $(length(G)) elements; " *
              "expected $nG (M x d)")
    return K, G
end

"""
    make_r_moments_wrapper(rfun) -> Function

Wrap a bridged R function `rfun(theta, U)` returning `list(K=, G=)`
into a Bundle-compatible in-place `moments!(K, G, theta, U, obj)`.
"""
function make_r_moments_wrapper(rfun)
    return function (K, G, theta, U, obj)
        res = rfun(collect(Float64, theta), U)
        Kv, Gv = _split_kg_result(res, length(K), length(G))
        copyto!(K, Kv)
        copyto!(G, Gv)
        return nothing
    end
end

"""
    make_r_jac_wrapper(rfun) -> Function

Wrap a bridged R gradient function `rfun(theta, U)` into a
Bundle-compatible `moments_jac(theta, U, obj)`; the raw return value
is normalized downstream by `_normalize_jac`.
"""
function make_r_jac_wrapper(rfun)
    return (theta, U, obj) -> rfun(collect(Float64, theta), U)
end

# ===================================================================
# Thin entry point for R (JuliaCall-friendly: plain numbers, vectors,
# matrices, and strings in; a plain Dict out).
# ===================================================================
_as_f64_vec(x::AbstractVector) = collect(Float64, x)
_as_f64_vec(x::Real)           = [Float64(x)]

"""
    tvb_solve(; kwargs...) -> Dict{String, Any}

Solve the outer/inner bound problems over a grid of budgets `delta`
and return a plain Dict (JuliaCall-friendly).  Keywords:

- `moments_name::String`: name of a function defined in `Main` with
  the Bundle signature `moments!(K, G, theta, U, obj)`.
- `delta`: budget grid (sorted internally in increasing order).
- `d::Int`: number of moment conditions (columns of G).
- `theta_lb`, `theta_ub`, `theta_init`: outer parameter box and
  initial point (`l = length(theta_lb)`).  A degenerate box
  (`theta_lb == theta_ub`) skips the outer optimization and evaluates
  the inner (fixed-theta) bounds at `theta_init`.
- `U`: M x u_dim matrix of latent draws, or `nothing` to generate
  scrambled-Halton uniforms via `tvb_generate_halton(M, u_dim; seed)`.
- `gamma`: arbitrary payload stored in the Bundle (`obj.gamma`).
- `divergence`: one of "KL_chi2", "KL", "chi2", "TV", "TVmix",
  "TVmixC", "TVac".
- `side`: "both", "lower", or "upper".
- `jac_mode`: "forwarddiff", "fd", or "user"; `jac_name` names a
  `Main` function for "user"; `fd_step` is the finite-difference step.
- `inner_opt`, `outer_opt`: paths to KNITRO option files.
- `maxsolves`, `startptrange`, `use_optim`, `time_limit`,
  `iterations`, `outer_iterations`: outer multi-start controls.
- `lower_limit`, `eta_min`, `psi_tv_eps`, `tvac_tau`, `tvmix_tau`,
  `purekl_acap`, `tvmix_kappa` (NaN = tie kappa to 1 - delta):
  solver tuning (see the R help of `tvbounds_control()`).
- `verbose::Bool`.

For each budget and side the reported bound is the inner value re-
solved at the best theta candidate (the multi-start optimum or the
previous budget's optimum, whichever is wider), which makes the
reported curves monotone in the budget by construction and guards
against outer/inner bookkeeping drift.
"""
function tvb_solve(;
        moments_name::AbstractString,
        delta,
        d::Integer,
        theta_lb,
        theta_ub,
        theta_init,
        inner_opt::AbstractString,
        outer_opt::AbstractString,
        U = nothing,
        M::Integer = 50_000,
        u_dim::Integer = 0,
        seed = nothing,
        gamma = nothing,
        divergence::AbstractString = "KL_chi2",
        side::AbstractString = "both",
        jac_mode::AbstractString = "forwarddiff",
        jac_name::AbstractString = "",
        fd_step::Real = 1e-6,
        maxsolves::Integer = 10,
        startptrange::Real = 0.01,
        use_optim::Bool = false,
        time_limit::Real = 60.0,
        iterations::Integer = 100,
        outer_iterations::Integer = 3,
        lower_limit::Real = -10.0,
        eta_min::Real = 1e-120,
        psi_tv_eps::Real = 1e-4,
        tvac_tau::Real = 1e-3,
        tvmix_tau::Real = 1e-3,
        purekl_acap::Real = 500.0,
        tvmix_kappa::Real = NaN,
        verbose::Bool = false)

    # -- tuning constants (module-level Refs; single-threaded) --------
    PSI_TV_EPS[]  = Float64(psi_tv_eps)
    TVAC_TAU[]    = Float64(tvac_tau)
    TVMIX_TAU[]   = Float64(tvmix_tau)
    PUREKL_ACAP[] = Float64(purekl_acap)
    TVMIX_KAPPA_OVERRIDE[] = Float64(tvmix_kappa)

    # -- inputs -------------------------------------------------------
    div_sym = Symbol(divergence)
    div_sym in (:KL_chi2, :KL, :chi2, :TV, :TVmix, :TVmixC, :TVac) ||
        error("unknown divergence: $divergence")
    side in ("both", "lower", "upper") || error("unknown side: $side")
    jm = Symbol(jac_mode)
    jm in (:forwarddiff, :fd, :user) || error("unknown jac_mode: $jac_mode")

    isdefined(Main, Symbol(moments_name)) ||
        error("the moments function `$moments_name` is not defined in Main")
    mom = getproperty(Main, Symbol(moments_name))
    mom isa Function || error("`$moments_name` is not a function")

    jfun = nothing
    if jm === :user
        isempty(jac_name) && error("jac_mode = \"user\" requires jac_name")
        isdefined(Main, Symbol(jac_name)) ||
            error("the gradient function `$jac_name` is not defined in Main")
        jfun = getproperty(Main, Symbol(jac_name))
        jfun isa Function || error("`$jac_name` is not a function")
    end

    Umat = U === nothing ?
        tvb_generate_halton(Int(M), Int(u_dim); seed = seed) :
        Matrix{Float64}(U)
    Meff = size(Umat, 1)

    deltas = sort(unique(_as_f64_vec(delta)))
    isempty(deltas) && error("delta must be non-empty")
    all(>(0.0), deltas) || error("all budgets delta must be strictly positive")
    if div_sym in (:TV, :TVmix, :TVmixC, :TVac)
        all(<=(1.0), deltas) ||
            error("for the total-variation divergences all budgets " *
                  "delta must lie in (0, 1]")
    end

    lbv = _as_f64_vec(theta_lb)
    ubv = _as_f64_vec(theta_ub)
    thi = _as_f64_vec(theta_init)
    l   = length(lbv)
    length(ubv) == l || error("theta_lb and theta_ub must have equal length")
    length(thi) == l || error("theta_init must have length $(l)")
    all(lbv .<= ubv) || error("theta_lb must be <= theta_ub elementwise")
    all(lbv .<= thi .<= ubv) || error("theta_init must lie in the theta box")

    fixed = maximum(ubv .- lbv) <= 0.0

    make_bundle(fs::Bool) = Bundle(
        delta         = deltas[1],
        find_smallest = fs,
        gamma         = gamma,
        moments!      = mom,
        d             = Int(d),
        l             = l,
        U             = Umat,
        divergence    = div_sym,
        inner_opt     = String(inner_opt),
        outer_opt     = String(outer_opt),
        lower_limit   = Float64(lower_limit),
        eta_min       = Float64(eta_min),
        jac_mode      = jm,
        moments_jac   = jfun,
        fd_step       = Float64(fd_step))

    # -- baseline point: plug-in counterfactual at theta_init ---------
    # (a single moments evaluation; also validates the user function
    # before any expensive solve)
    probe = make_bundle(true)
    Kbuf  = zeros(Meff)
    Gbuf  = zeros(Meff, Int(d))
    mom(Kbuf, Gbuf, thi, Umat, probe)
    point = mean(Kbuf)

    nd = length(deltas)
    out = Dict{String,Any}(
        "delta"       => deltas,
        "lower"       => fill(NaN, nd),
        "upper"       => fill(NaN, nd),
        "point"       => point,
        "M"           => Meff,
        "u_dim"       => size(Umat, 2),
        "divergence"  => String(divergence),
        "fixed_theta" => fixed,
        "status_outer_lower" => fill(-999, nd),
        "status_outer_upper" => fill(-999, nd),
        "status_inner_lower" => fill(-999, nd),
        "status_inner_upper" => fill(-999, nd),
        "time_lower"  => fill(NaN, nd),
        "time_upper"  => fill(NaN, nd),
        "theta_lower" => fill(NaN, l, nd),
        "theta_upper" => fill(NaN, l, nd))

    for fs in (true, false)
        fs  && side == "upper" && continue
        !fs && side == "lower" && continue
        tag = fs ? "lower" : "upper"
        Q = fs ? probe : make_bundle(fs)
        vals, st_out, st_in, times, thetas = _solve_side!(
            Q, deltas, lbv, ubv, thi;
            fixed = fixed, maxsolves = Int(maxsolves),
            startptrange = Float64(startptrange), use_optim = use_optim,
            time_limit = Float64(time_limit), iterations = Int(iterations),
            outer_iterations = Int(outer_iterations), verbose = verbose)
        out[tag] = vals
        out["status_outer_" * tag] = st_out
        out["status_inner_" * tag] = st_in
        out["time_" * tag]  = times
        out["theta_" * tag] = thetas
    end

    return out
end

# Dict method so the R side can pass one named list instead of
# relying on keyword-argument bridging.
function tvb_solve(args::AbstractDict)
    kw = Dict{Symbol,Any}()
    for (k, v) in args
        kw[Symbol(string(k))] = v
    end
    return tvb_solve(; kw...)
end

# One side (lower or upper) over the ascending budget grid, with the
# warm-start chain of the replication drivers: the previous budget's
# optimal theta seeds the next budget's multi-start AND remains a
# candidate, so the reported curve is monotone in the budget by
# construction (the neighborhoods are nested across budgets for every
# supported divergence).  Every reported value is an inner value
# actually attained at the recorded theta (re-solved after the outer
# search), never the outer loop's bookkept objective.
function _solve_side!(Q::Bundle, deltas, theta_lb, theta_ub, theta_init;
                      fixed::Bool, maxsolves::Int, startptrange::Float64,
                      use_optim::Bool, time_limit::Float64,
                      iterations::Int, outer_iterations::Int,
                      verbose::Bool)
    nd     = length(deltas)
    vals   = fill(NaN, nd)
    st_out = fill(-999, nd)
    st_in  = fill(-999, nd)
    times  = fill(NaN, nd)
    thetas = fill(NaN, Q.l, nd)
    theta_prev = collect(Float64, theta_init)

    for (i, del) in enumerate(deltas)
        t0 = time()
        Q.delta = del
        Q.x .= NaN

        if fixed
            v, _, st = inner_loop(Q, theta_init)
            st_in[i] = st
            if isfinite(v) && abs(v) < 1e9
                vals[i] = v
                thetas[:, i] .= theta_init
            end
        else
            # Candidate A: previous budget's optimum at the current budget.
            vA, _, stA = inner_loop(Q, theta_prev)
            okA = isfinite(vA) && abs(vA) < 1e9

            # Candidate B: multi-start outer optimization.
            kapB, thB, flB = outer_loop_multi(
                Q, theta_lb, theta_ub, theta_prev;
                maxsolves = maxsolves, startptrange = startptrange,
                use_optim = use_optim, time_limit = time_limit,
                iterations = iterations,
                outer_iterations = outer_iterations, verbose = verbose)
            st_out[i] = flB

            vB  = NaN
            stB = -999
            okB = false
            if all(isfinite, thB) && isfinite(kapB) && abs(kapB) < 1e9
                Q.x .= NaN
                vB, _, stB = inner_loop(Q, thB)   # verify at the outer optimum
                okB = isfinite(vB) && abs(vB) < 1e9
            end

            better_B = okB && (!okA ||
                               (Q.find_smallest ? vB < vA : vB > vA))
            if better_B
                vals[i]  = vB
                st_in[i] = stB
                thetas[:, i] .= thB
                theta_prev = collect(Float64, thB)
            elseif okA
                vals[i]  = vA
                st_in[i] = stA
                thetas[:, i] .= theta_prev
            end
        end

        times[i] = time() - t0
        if verbose
            @printf("  [tvbounds] side=%-5s delta=%-8g value=%+10.6f  (outer flag %d, inner status %d, %.1fs)\n",
                    Q.find_smallest ? "lower" : "upper", del, vals[i],
                    st_out[i], st_in[i], times[i])
            flush(stdout)
        end
    end

    return vals, st_out, st_in, times, thetas
end

end # module TVBoundsJulia
