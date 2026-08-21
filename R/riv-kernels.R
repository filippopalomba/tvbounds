# =============================================================================
# riv-kernels.R -- computational kernels for tvbounds_riv(), ported from the
# formula-instrument sensitivity exercise of Section SA7.3 of Palomba (2026)
# (Propositions "formula instruments bounds" and
#  "formula instruments contamination").
#
# Nothing here reads or writes a file.  All functions are internal.
#
# -----------------------------------------------------------------------------
# Notation (manuscript Section SA7.3).  The exercise is conducted CONDITIONALLY
# on the realized sample, so everything below is a deterministic function of the
# data and of the budget delta.
#
#   i = 1..n    units, y_i and x_i already residualized on the controls
#   v           realized shock vector; v^(1),...,v^(S) the counterfactual draws
#   f_i(v')     the formula, i.e. unit i's instrument at shock configuration v'
#   z_i         = f_i(v), the un-recentered instrument
#   mu_i(P)     = E_P[f_i], the expected instrument under assignment law P
#   beta(P)     = sum_i (z_i - mu_i(P)) y_i / sum_i (z_i - mu_i(P)) x_i
#               = G_y(P) / G_x(P)
#   g_y(v')     = sum_i y_i f_i(v'),      g_x(v') = sum_i x_i f_i(v')
#   G_y(P)      = g_y(v) - E_P[g_y],      G_x(P)  = g_x(v) - E_P[g_x]
#   g_b         = g_y - b * g_x
#
# Every quantity the exercise needs is therefore a function of the S+1 numbers
# (g_y(v), g_y(v^(1)),...,g_y(v^(S))) and their g_x counterparts: the problem is
# two-dimensional however large the shock space is.  The design matrix is
# consumed once, in fi_criteria(), and never again.
#
# Two conventions used throughout, both harmless and both stated in SA7.3:
#
#  (a) SHIFT INVARIANCE.  Replacing f_i(.) by f_i(.) - c_i and z_i by z_i - c_i
#      for any unit-specific constant c_i leaves G_y(P) and G_x(P) unchanged,
#      because E_P[.] is an average over a probability distribution.  We exploit
#      this to center the design matrix, which makes E_Pbase[g_y] = E_Pbase[g_x]
#      = 0 exactly rather than up to rounding.
#
#  (b) SIGN NORMALIZATION.  Replacing (g_y, g_x) by (-g_y, -g_x) leaves beta(P)
#      unchanged and flips the sign of the recentered first stage, so we may and
#      do take G_x(Pbase) > 0, as the propositions assume.
# =============================================================================


# -----------------------------------------------------------------------------
# 1.  From the design to the two criterion functions
# -----------------------------------------------------------------------------

#' Reduce a formula-instrument design to the criterion functions of SA7.3
#'
#' @param y  n-vector, outcome, already residualized on the controls
#' @param x  n-vector, endogenous regressor, already residualized on the controls
#' @param z  n-vector, the realized (un-recentered) instrument, z_i = f_i(v)
#' @param Fmat  n x S matrix, `Fmat[i,s] = f_i(v^(s))`, the formula at draw s
#' @param p  S-vector of baseline probabilities; defaults to uniform, which is
#'   the postulated assignment distribution of Borusyak and Hull (2023)
#'
#' @return a list with the realized criteria `gy_v`, `gx_v`, the S-vectors
#'   `gy_s`, `gx_s` of criterion values at the counterfactual draws, the
#'   baseline weights `p`, the baseline aggregates `Egy`, `Egx`, the recentered
#'   numerator/denominator `Gy0`, `Gx0` at the baseline, the reported estimate
#'   `beta_hat`, and the sign `flip` applied by the normalization.
#' @keywords internal
#' @noRd
fi_criteria <- function(y, x, z, Fmat, p = NULL) {
  stopifnot(length(y) == length(x), length(y) == length(z),
            nrow(Fmat) == length(y))
  S <- ncol(Fmat)
  if (is.null(p)) p <- rep(1 / S, S)
  stopifnot(length(p) == S, all(p >= 0), abs(sum(p) - 1) < 1e-10)

  # Shift invariance (a): center the formula at its baseline mean unit by unit,
  # which is the recentering itself.  z must be shifted by the same constant.
  mu <- as.numeric(Fmat %*% p)
  Fc <- Fmat - mu
  zc <- z - mu

  gy_s <- as.numeric(crossprod(Fc, y))   # S-vector, g_y at each draw
  gx_s <- as.numeric(crossprod(Fc, x))
  gy_v <- sum(y * zc)                    # scalar, g_y at the realized shocks
  gx_v <- sum(x * zc)

  # Exactly zero by construction after the centering above; kept explicit so the
  # formulas below read as they do in the manuscript.
  Egy <- sum(p * gy_s)
  Egx <- sum(p * gx_s)

  Gy0 <- gy_v - Egy                      # G_y(Pbase) = recentered numerator
  Gx0 <- gx_v - Egx                      # G_x(Pbase) = recentered first stage

  # Sign normalization (b).
  flip <- if (Gx0 < 0) -1 else 1
  gy_s <- flip * gy_s; gx_s <- flip * gx_s
  gy_v <- flip * gy_v; gx_v <- flip * gx_v
  Egy  <- flip * Egy;  Egx  <- flip * Egx
  Gy0  <- flip * Gy0;  Gx0  <- flip * Gx0

  list(gy_s = gy_s, gx_s = gx_s, gy_v = gy_v, gx_v = gx_v,
       p = p, Egy = Egy, Egx = Egx, Gy0 = Gy0, Gx0 = Gx0,
       beta_hat = Gy0 / Gx0, flip = flip, S = S, n = length(y))
}


# -----------------------------------------------------------------------------
# 2.  Criterion bounds over the total variation ball  (Proposition SA7.3(i))
# -----------------------------------------------------------------------------
# For a criterion h with baseline distribution Pbase,
#
#   sup_{TV(P,Pbase) <= delta} E_P[h] = delta * sup h + int_delta^1 q_h(u) du,
#   inf_{TV(P,Pbase) <= delta} E_P[h] = delta * inf h + int_0^{1-delta} q_h(u) du,
#
# with q_h the quantile function of h under Pbase: the baseline mean of h once
# the lower (upper) tail of mass delta has been trimmed away and relocated to
# the most (least) favorable shock configuration.
#
# The shock space of the application is the finite set of counterfactual draws,
# so sup h = max_s h_s and the integral is a weighted sum over the atoms with
# the boundary atom split at the exact fraction of its mass that survives the
# trimming.  Nothing is rounded to whole atoms.

#' Integral of the quantile function of a discrete criterion over the top mass
#'
#' Returns int_{a}^{1} q_h(u) du for h taking value `h[s]` with probability
#' `p[s]`, i.e. the baseline-weighted sum of the largest values of h carrying a
#' total mass of 1 - a.  The atom straddling the cut is included at the fraction
#' of its mass that lies above a.
#' @keywords internal
#' @noRd
.upper_tail_integral <- function(h, p, a) {
  if (a >= 1) return(0)
  if (a <= 0) return(sum(p * h))
  o  <- order(h, decreasing = TRUE)      # largest first
  hs <- h[o]; ps <- p[o]
  cum <- cumsum(ps)
  m   <- 1 - a                           # mass to keep, from the top
  k   <- which(cum >= m - 1e-15)[1]      # atom that straddles the cut
  if (is.na(k)) return(sum(ps * hs))     # numerical guard: keep everything
  head_mass <- if (k > 1) cum[k - 1] else 0
  sum(ps[seq_len(k - 1)] * hs[seq_len(k - 1)]) + (m - head_mass) * hs[k]
}

#' Sensitivity bounds on `E_P[h]` over a total variation ball of radius `delta`
#'
#' @param h        S-vector of criterion values at the points of the shock space
#' @param p        S-vector of baseline probabilities
#' @param delta    scalar budget in `[0,1]`
#' @param h_sup,h_inf  the supremum and infimum of the criterion over the WHOLE
#'   shock space.  They default to max(h) and min(h), which is correct when the
#'   shock space is the finite set carrying `p` -- the reading of the exercise
#'   used in the application (see "Restricting the configuration space" in the
#'   manuscript remark on unbounded formulas).  Passing wider values covers a
#'   shock space larger than the support of the baseline.
#' @keywords internal
#' @noRd
tv_bound_upper <- function(h, p, delta, h_sup = max(h)) {
  delta * h_sup + .upper_tail_integral(h, p, delta)
}

#' Lower counterpart of `tv_bound_upper()`, by the reflection h -> -h.
#' @keywords internal
#' @noRd
tv_bound_lower <- function(h, p, delta, h_inf = min(h)) {
  -tv_bound_upper(-h, p, delta, h_sup = -h_inf)
}


# -----------------------------------------------------------------------------
# 3.  First-stage breakdown budgets  (Propositions SA7.3(ii) and SA7.4(ii))
# -----------------------------------------------------------------------------
# Below delta_FS the recentered first stage is bounded away from zero over the
# whole robustness set, so the estimator is well defined and the bounds on beta
# are finite; at and above it the neighborhood contains assignment distributions
# that make the first stage vanish and the bounds are vacuous.

#' Is the recentered first stage bounded away from zero over the TV ball?
#'
#' Proposition SA7.3(ii): every P in the ball has
#' `G_x(P) >= g_x(v) - sup_P E_P[g_x]`, so a strictly positive right-hand side
#' certifies that the estimator is well defined throughout.  This is the
#' primitive condition; testing it directly, rather than comparing delta with
#' the breakdown budget, is what keeps the boundary case right when the budget
#' is censored at 1 by the empty-set convention (see `fi_delta_fs_tv`).
#' @keywords internal
#' @noRd
.fs_margin_tv <- function(cr, delta) {
  cr$gx_v - tv_bound_upper(cr$gx_s, cr$p, delta)
}

#' Is the recentered first stage bounded away from zero over the contamination
#' neighborhood?  Proposition SA7.4(ii): the relevant quantity is
#' `inf_{v'} G_x(v';delta) = G_x(Pbase) - delta * sup_{v'} (g_x(v') - E_Pbase[g_x])`.
#' @keywords internal
#' @noRd
.fs_margin_cont <- function(cr, delta) {
  cr$Gx0 - delta * (max(cr$gx_s) - cr$Egx)
}

#' First-stage breakdown budget over the total variation ball
#'
#' `delta_FS^TV := inf{ delta in [0,1] : g_x(v) <= sup_{P} E_P[g_x] }`, with the
#' convention that the infimum over an empty set equals 1.  The map
#' delta -> tv_bound_upper(g_x, .) is nondecreasing and continuous, so the
#' infimum is located by bisection.
#'
#' The returned value carries an attribute "censored", TRUE when the set in the
#' definition is empty, that is when the first stage never breaks down on `[0,1]`
#' and the value 1 is the convention rather than an actual breakdown.  Callers
#' that need to know whether the estimator is well defined AT a budget should
#' use `.fs_margin_tv()` instead of comparing with this number.
#' @keywords internal
#' @noRd
fi_delta_fs_tv <- function(cr, tol = 1e-12) {
  f <- function(d) -.fs_margin_tv(cr, d)      # nondecreasing in d
  if (f(1) < 0) return(structure(1, censored = TRUE))
  if (f(0) >= 0) return(structure(0, censored = FALSE))
  lo <- 0; hi <- 1
  while (hi - lo > tol) {
    mid <- 0.5 * (lo + hi)
    if (f(mid) < 0) lo <- mid else hi <- mid
  }
  structure(hi, censored = FALSE)
}

#' First-stage breakdown budget over the contamination neighborhood
#'
#' `delta_FS^cont := min{1, G_x(Pbase) / sup_{v'} (g_x(v') - E_Pbase[g_x])}`, the
#' ratio being +Inf when its denominator vanishes.  Carries the same "censored"
#' attribute as `fi_delta_fs_tv`.
#' @keywords internal
#' @noRd
fi_delta_fs_cont <- function(cr) {
  den <- max(cr$gx_s) - cr$Egx
  if (den <= 0) return(structure(1, censored = TRUE))
  r <- cr$Gx0 / den
  structure(min(1, r), censored = r > 1)
}


# -----------------------------------------------------------------------------
# 4.  Bounds on the estimate over the total variation ball  (Prop. SA7.3(iii))
# -----------------------------------------------------------------------------
# beta(P) = b holds exactly when E_P[g_b] = g_b(v) with g_b = g_y - b g_x.  The
# largest attainable estimate is therefore the b at which the realized number
# g_b(v) meets the LOWER criterion bound, and the smallest is the b at which it
# meets the UPPER one:
#
#   psi_delta(b) := g_b(v) - inf_P E_P[g_b]   is strictly DEcreasing, zero at the
#                                             upper bound on beta
#   phi_delta(b) := sup_P E_P[g_b] - g_b(v)   is strictly INcreasing, zero at the
#                                             lower bound on beta
#
# Both are convex, so each has a unique zero and a bracketed root finder is
# guaranteed to locate it.

#' psi_delta of Proposition SA7.3(iii)
#'
#' The shock space is the finite set of counterfactual draws, so the extremes of
#' g_b = g_y - b g_x over it are simply the extremes of the S evaluated values.
#' @keywords internal
#' @noRd
.psi <- function(b, cr, delta) {
  h <- cr$gy_s - b * cr$gx_s
  (cr$gy_v - b * cr$gx_v) - tv_bound_lower(h, cr$p, delta)
}

#' phi_delta of Proposition SA7.3(iii)
#' @keywords internal
#' @noRd
.phi <- function(b, cr, delta) {
  h <- cr$gy_s - b * cr$gx_s
  tv_bound_upper(h, cr$p, delta) - (cr$gy_v - b * cr$gx_v)
}

#' Solve for the unique zero of a strictly monotone function by bracket
#' expansion followed by `uniroot`.
#'
#' Grows a symmetric bracket around zero until the sign changes.  Below the
#' first-stage breakdown budget `f` is finite, continuous and strictly monotone
#' on the whole line and has a unique zero by Proposition SA7.3(iii), so this
#' terminates; failing to bracket after `max_double` doublings means one of
#' those hypotheses is violated and is an error rather than an infinite bound.
#' @keywords internal
#' @noRd
.monotone_root <- function(f, start = 1, max_double = 60) {
  lo <- -start; hi <- start; k <- 0
  while (sign(f(lo)) == sign(f(hi))) {
    lo <- 2 * lo; hi <- 2 * hi; k <- k + 1
    if (k > max_double)
      stop("failed to bracket the root of a monotone function; ",
           "check that the budget is below the first-stage breakdown")
  }
  stats::uniroot(f, c(lo, hi), tol = .Machine$double.eps^0.75)$root
}

#' Bounds on the recentered IV estimate over the total variation ball
#'
#' @param cr      output of `fi_criteria()`
#' @param delta   scalar budget.  Once the recentered first stage can vanish the
#'   bounds are (-Inf, +Inf), which is what Proposition SA7.3(iv) reports.
#' @keywords internal
#' @noRd
fi_bounds_tv <- function(cr, delta) {
  if (delta <= 0) return(c(lower = cr$beta_hat, upper = cr$beta_hat))
  # Well-definedness is decided by the primitive positivity condition, not by a
  # comparison with the (possibly censored) breakdown budget.
  if (.fs_margin_tv(cr, delta) <= 0) return(c(lower = -Inf, upper = Inf))
  up <- .monotone_root(function(b) .psi(b, cr, delta))
  lo <- .monotone_root(function(b) .phi(b, cr, delta))
  c(lower = lo, upper = up)
}


# -----------------------------------------------------------------------------
# 5.  Bounds on the estimate over the contamination neighborhood (Prop. SA7.4)
# -----------------------------------------------------------------------------
# Every P in the neighborhood is P = (1-delta) Pbase + delta R, so
#
#   G_y(P) = int G_y(v';delta) dR(v'),   G_y(v';delta) := G_y(Pbase) - delta (g_y(v') - E_Pbase[g_y])
#
# and likewise for G_x.  Below the first-stage breakdown budget the bounds are
# therefore the extremes of the ratio G_y(v';delta)/G_x(v';delta) over the shock
# space, each approached by a contaminating distribution degenerate at a single
# configuration.  No optimization is involved: one evaluates S ratios per budget.

#' Bounds on the recentered IV estimate over the contamination neighborhood
#'
#' @return a named vector with the two bounds and the index of the least
#'   favorable shock configuration attaining each of them
#' @keywords internal
#' @noRd
fi_bounds_cont <- function(cr, delta) {
  if (delta <= 0) return(c(lower = cr$beta_hat, upper = cr$beta_hat,
                           arg_lower = NA_real_, arg_upper = NA_real_))
  if (.fs_margin_cont(cr, delta) <= 0) return(c(lower = -Inf, upper = Inf,
                                  arg_lower = NA_real_, arg_upper = NA_real_))
  num <- cr$Gy0 - delta * (cr$gy_s - cr$Egy)
  den <- cr$Gx0 - delta * (cr$gx_s - cr$Egx)
  r <- num / den
  c(lower = min(r), upper = max(r),
    arg_lower = which.min(r), arg_upper = which.max(r))
}


# -----------------------------------------------------------------------------
# 6.  Paths and summary measures
# -----------------------------------------------------------------------------

#' Trace both sets of bounds over a grid of budgets
#'
#' @return a data frame with one row per budget and columns
#'   `delta`, `tv_lower`, `tv_upper`, `cont_lower`, `cont_upper`
#' @keywords internal
#' @noRd
fi_paths <- function(cr, deltas) {
  tv <- t(vapply(deltas, function(d) fi_bounds_tv(cr, d), numeric(2)))
  ct <- t(vapply(deltas, function(d) fi_bounds_cont(cr, d), numeric(4)))
  data.frame(delta = deltas,
             tv_lower = tv[, 1], tv_upper = tv[, 2],
             cont_lower = ct[, 1], cont_upper = ct[, 2])
}

#' Breakdown budget for the reference value `tau_star`
#'
#' delta_b := inf{ delta : lower(delta) <= tau_star <= upper(delta) }, with the
#' convention that the infimum over an empty set equals 1 (Section SA5.2).
#' `lower` and `upper` must be monotone functions of delta, which they are by
#' Propositions SA7.3(iii) and SA7.4.
#'
#' @param lower,upper functions of a scalar budget
#' @return the budget, or `NA` when the reference value is never covered on
#'   `[0, hi]` -- reported as censored by the caller
#' @keywords internal
#' @noRd
fi_breakdown <- function(lower, upper, tau_star = 0, hi = 1, tol = 1e-10) {
  covered <- function(d) lower(d) <= tau_star && tau_star <= upper(d)
  if (covered(0)) return(0)
  if (!covered(hi)) return(NA_real_)
  lo <- 0; up <- hi
  while (up - lo > tol) {
    mid <- 0.5 * (lo + up)
    if (covered(mid)) up <- mid else lo <- mid
  }
  up
}


# -----------------------------------------------------------------------------
# 7.  Least favorable assignment distributions
# -----------------------------------------------------------------------------

#' The least favorable distribution behind a total variation bound
#'
#' At the bound `b` the binding program is the criterion program for
#' g_b = g_y - b g_x, whose solution is described by Proposition SA7.3(i): the
#' baseline with its lower (respectively upper) tail of mass `delta` trimmed
#' away and that mass relocated to the configuration at which g_b is largest
#' (smallest).  The distribution is returned so that the exercise can report how
#' concentrated the least favorable reweighting is.
#'
#' @param direction "upper" for the distribution solving `sup E_P[g_b]`, which
#'   delivers the LOWER bound on the estimate, and "lower" for its mirror image
#' @return an S-vector of probabilities
#' @keywords internal
#' @noRd
fi_lf_law_tv <- function(cr, delta, b, direction = c("upper", "lower")) {
  direction <- match.arg(direction)
  h <- cr$gy_s - b * cr$gx_s
  p <- cr$p
  if (delta <= 0) return(p)
  # Trim the tail of h that hurts, then relocate its mass to the extreme.
  o <- order(h, decreasing = (direction == "lower"))  # trimmed first
  q <- p
  left <- delta
  for (s in o) {                       # remove a total mass of delta
    take <- min(q[s], left)
    q[s] <- q[s] - take
    left <- left - take
    if (left <= 1e-15) break
  }
  j <- if (direction == "upper") which.max(h) else which.min(h)
  q[j] <- q[j] + delta
  q / sum(q)
}
