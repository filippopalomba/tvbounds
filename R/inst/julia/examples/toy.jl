# ===================================================================
# Toy moments example for tvbounds_counterfactual() -- used by the
# guarded integration tests and by the (future) vignette.
#
# Model.  The latent draw is a scalar U with baseline distribution
# Uniform(0, 1) (the scrambled-Halton coordinates are already uniform,
# so no quantile transform is needed).  The single parameter theta is
# the mean of U under the candidate distribution P:
#
#   moment:         G(U; theta) = U - theta        (d = 1, l = 1)
#   counterfactual: K(U; theta) = U
#
# Under the mixture-constrained total-variation neighborhood
# (divergence "TVmix", kappa = 1 - delta), every candidate is
# P = (1 - delta) * P_star + delta * R, hence
#
#   E_P[K] = E_P[U] = theta
#
# whenever the moment is satisfiable, i.e. whenever
# E_R[U] = (theta - (1 - delta)/2) / delta lies inside the range of
# the draws (approximately [0, 1]).  The sensitivity bounds over a theta box
# [t_lb, t_ub] inside that feasible range are therefore exactly
#
#   lower = t_lb,   upper = t_ub    (up to the O(tau log M) smoothing),
#
# and at delta = 1 the same identity holds for the plain "TV"
# neighborhood (the budget-1 total-variation ball contains every
# distribution on the draws).  These closed forms are what the
# integration test checks.
#
# The function is Dual-generic (theta enters linearly), so
# ForwardDiff-based outer gradients work out of the box.
# ===================================================================

function tvb_toy_moments!(K, G, theta, U, obj)
    M = size(U, 1)
    @inbounds for m in 1:M
        K[m]    = U[m, 1]
        G[m, 1] = U[m, 1] - theta[1]
    end
    return nothing
end
