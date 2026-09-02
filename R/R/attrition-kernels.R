# Computational kernels for tvbounds_attrition().
#
# The functions Eq_q(), one_group_grid(), pooled_bounds_alloc(), tv_bounds(),
# lee_bounds(), boot_tv_bounds(), and boot_summary() are ported from the
# audited project scripts accompanying Palomba (2026) ("Sensitivity Analysis
# in Population Shares"); their names are kept unchanged to ease auditing,
# but none of them is exported. The cluster resampling in boot_tv_bounds()
# replicates the participant-level bootstrap of the Christensen-Nino-Osman
# exercise; breakdown_delta() is ported from the clinical-trials attrition
# exercise. The contamination kernel .tvb_cont_grid() implements the
# closed-form bounds under a contamination neighborhood derived from the
# paper (Section SA5.3, Lemma "contamination as constrained TV", and the
# attrition robustness set of the RCT-with-attrition section); see the documentation of
# tvbounds_attrition() for the statement.

#' Atom-safe fractional lower partial mean (scalar audited reference)
#'
#' Computes `E_Q[Y * 1{Y in bottom-q mass}]` for the empirical distribution
#' of the sorted vector `Y_sort`, splitting the boundary observation
#' proportionally so that point masses (atoms) are handled correctly. Kept
#' as the scalar audited reference for `one_group_grid()`.
#'
#' @param Y_sort Sorted numeric vector of observed outcomes.
#' @param q Scalar mass level in `[0, 1]`.
#' @keywords internal
#' @noRd
Eq_q <- function(Y_sort, q) {
  n <- length(Y_sort); mu <- mean(Y_sort)
  if (q <= 0) return(0); if (q >= 1) return(mu)
  k <- floor(q * n); alpha <- q * n - k
  s <- if (k > 0) sum(Y_sort[seq_len(k)]) else 0
  (s + alpha * Y_sort[min(k + 1L, n)]) / n
}

#' Total variation bounds for one group over a whole budget grid
#'
#' Vectorized version of the closed-form total variation bounds on the
#' always-observed mean of one treated group (Palomba 2026, Theorem
#' "TV sensitivity bounds for the average treatment effect on the
#' always-observed"), given the SORTED observed outcomes. The sort and the
#' cumulative sum are hoisted out of the grid loop: every partial sum
#' `sum(Ys[seq_len(k)])` is read off `cs = c(0, cumsum(Ys))` in O(1), and the
#' whole budget grid is evaluated in a few vectorized operations. `cumsum`
#' and `sum` share R's long-double left-to-right accumulator, so `cs[k+1]`
#' equals `sum(Ys[seq_len(k)])` bitwise and the results are identical to the
#' scalar path (audited against `Eq_q()` in the source scripts).
#'
#' `p_star = 0` is admissible and is NOT a degenerate case: the group then
#' has no compliers, every treated respondent is always-observed, and the
#' closed form correctly collapses to the point-identified
#' `lo = up = mean(Ys)`.
#'
#' @param Ys Sorted numeric vector of observed treated outcomes.
#' @param p_star Complier share among treated respondents, in `[0, 1)`.
#' @param eps Numeric vector of budget values in `[0, 1]`.
#' @return List with components `lo` and `up`, the lower and upper bounds on
#'   the always-observed mean at each budget value.
#' @keywords internal
#' @noRd
one_group_grid <- function(Ys, p_star, eps) {
  n <- length(Ys); mu <- mean(Ys)
  cs <- c(0, cumsum(Ys))
  Eqv <- function(qs) {
    k <- floor(qs * n); alpha <- qs * n - k
    val <- (cs[pmin(k, n) + 1L] + alpha * Ys[pmin(k + 1L, n)]) / n
    val[qs <= 0] <- 0; val[qs >= 1] <- mu; val
  }
  Ea <- Eqv((1 - p_star) * eps); Eb <- Eqv(1 - p_star * eps)
  Ec <- Eqv(p_star * eps);       Ed <- Eqv(1 - (1 - p_star) * eps)
  list(lo = (1 / (1 - p_star)) * Ea + (Eb - Ea),
       up = (Ed - Ec) + (1 / (1 - p_star)) * (mu - Ed))
}

#' Contamination bounds for one group over a whole budget grid
#'
#' Closed-form bounds on the always-observed mean when the complier outcome
#' distribution is restricted to the contamination neighborhood of the
#' always-observed outcome distribution: `P_C = (1 - delta) * P_AO +
#' delta * R` with `R` an arbitrary distribution. Combining this mixture
#' restriction with the sample-splitting mixture `P_T = p_star * P_C +
#' (1 - p_star) * P_AO` yields the density constraints (with respect to
#' `P_T`) `(1 - delta) / (1 - delta * p_star) <= r <= 1 / p_star`, whose
#' extremal solutions are trimmed means of `P_T` with trimming mass
#' `delta * p_star`:
#'
#'   `lower:  E_PT[ Y * 1{bottom (1 - delta * p_star) mass} ] / (1 - delta * p_star)`
#'   `upper:  E_PT[ Y * 1{top    (1 - delta * p_star) mass} ] / (1 - delta * p_star)`
#'
#' i.e. Lee (2009)-type trimming with effective trimming share
#' `delta * p_star`. At `delta = 0` both collapse to `mean(Ys)`; at
#' `delta = 1` they equal the Lee bounds, matching the total variation
#' endpoints. Uses the same atom-safe fractional partial mean as
#' `one_group_grid()`, so atoms are handled exactly. See the documentation
#' of [tvbounds_attrition()] for the derivation and its source in the paper.
#'
#' @param Ys Sorted numeric vector of observed treated outcomes.
#' @param p_star Complier share among treated respondents, in `[0, 1)`.
#' @param eps Numeric vector of budget values in `[0, 1]`.
#' @return List with components `lo` and `up`, as in `one_group_grid()`.
#' @keywords internal
#' @noRd
.tvb_cont_grid <- function(Ys, p_star, eps) {
  n <- length(Ys); mu <- mean(Ys)
  cs <- c(0, cumsum(Ys))
  Eqv <- function(qs) {
    k <- floor(qs * n); alpha <- qs * n - k
    val <- (cs[pmin(k, n) + 1L] + alpha * Ys[pmin(k + 1L, n)]) / n
    val[qs <= 0] <- 0; val[qs >= 1] <- mu; val
  }
  q <- p_star * eps                       # trimmed mass; q < 1 since p_star < 1
  list(lo = Eqv(1 - q) / (1 - q),
       up = (mu - Eqv(q)) / (1 - q))
}

#' Pooled (joint) covariate bounds via exact greedy budget allocation
#'
#' One total variation budget allocated across the retained covariate cells
#' rather than the same budget imposed inside every cell (Palomba 2026,
#' eq. "rct covariate robustness set" / "rct covariate budget allocation").
#' In the normalized cell budget `a` in `[0, 1]` (`a = 1` exhausts the
#' cell's complier mass, so the cell bound is its Lee bound), the pooled
#' program reads
#'
#'   upper:  max sum_s w_s * up_s(a_s)  s.t.  sum_s w_s * a_s <= delta
#'   lower:  min sum_s w_s * lo_s(a_s)  s.t.  sum_s w_s * a_s <= delta
#'
#' with `w_s` the control-respondent shares (the covariate distribution of
#' the always-observed population). Each cell value function is piecewise
#' linear in `a` with kinks where the trimming quantiles cross the cell's
#' order-statistic masses `i/n_s`, and it is concave (upper) / convex
#' (lower), so the allocation solves exactly by a greedy fill: list every
#' linear segment of every cell, sort by marginal value per unit of pooled
#' budget (the segment slope), and consume the budget in that order. One
#' pass yields the whole curve `delta -> pooled bound`; evaluating the
#' cumulative fill at the grid is exact because the fill curve is itself
#' piecewise linear in the budget.
#'
#' @param Y_strata List of numeric vectors of observed treated outcomes, one
#'   per retained covariate cell (same order as `px` and `wts`).
#' @param px Numeric vector of cell-level complier shares.
#' @param wts Numeric vector of cell weights (control-respondent shares),
#'   summing to one.
#' @param eps_grid Numeric vector of budget values in `[0, 1]`.
#' @return List with components `up` and `lo`, the pooled bounds on the
#'   aggregated always-observed treated mean at each budget value.
#' @keywords internal
#' @noRd
pooled_bounds_alloc <- function(Y_strata, px, wts, eps_grid) {
  S <- length(Y_strata)
  base_up <- base_lo <- 0
  seg_up <- vector("list", S); seg_lo <- vector("list", S)
  keep01 <- function(x) x[is.finite(x) & x > 0 & x < 1]
  # Merge kinks closer than 1e-9: two nearly coincident breakpoints span a
  # segment of negligible budget and gain, but their slope Delta_v / Delta_a
  # is numerical noise that would be sorted into the wrong fill position. The
  # gap filter keeps the first of a close pair, so re-pin the endpoint.
  dedupe <- function(bp) { bp <- bp[c(TRUE, diff(bp) > 1e-9)]
                           bp[length(bp)] <- 1; bp }
  for (j in seq_len(S)) {
    Ys <- sort(Y_strata[[j]]); n <- length(Ys)
    p <- px[j]; w <- wts[j]
    ci <- seq_len(n - 1L) / n
    # Kinks of up_s: its arguments p*a and 1-(1-p)*a cross the masses i/n.
    # Kinks of lo_s: its arguments (1-p)*a and 1-p*a cross the masses i/n.
    # p = 0 is admissible (no compliers): both value functions are flat and
    # the divisions by p are filtered out with the out-of-range kinks.
    bp_up <- dedupe(sort(unique(c(0, 1, keep01(ci / p),
                                  keep01((1 - ci) / (1 - p))))))
    bp_lo <- dedupe(sort(unique(c(0, 1, keep01(ci / (1 - p)),
                                  keep01((1 - ci) / p)))))
    vu <- one_group_grid(Ys, p, bp_up)$up
    vl <- one_group_grid(Ys, p, bp_lo)$lo
    base_up <- base_up + w * vu[1]
    base_lo <- base_lo + w * vl[1]
    seg_up[[j]] <- data.frame(slope = diff(vu) / diff(bp_up),
                              cost = w * diff(bp_up), gain = w * diff(vu))
    seg_lo[[j]] <- data.frame(slope = -diff(vl) / diff(bp_lo),
                              cost = w * diff(bp_lo), gain = -w * diff(vl))
  }
  # Greedy fill: cumulative (budget, value-gain) knots in decreasing-slope
  # order; the pooled curve is their linear interpolant. Filling every segment
  # (total cost sum_s w_s = 1) reaches the covariate Lee endpoint exactly.
  fill <- function(segs) {
    segs <- do.call(rbind, segs)
    o <- order(segs$slope, decreasing = TRUE)
    kb <- c(0, cumsum(segs$cost[o])); kv <- c(0, cumsum(segs$gain[o]))
    function(d) stats::approx(kb, kv, xout = pmin(d, max(kb)), rule = 2,
                              ties = "ordered")$y
  }
  gain_up <- fill(seg_up); gain_lo <- fill(seg_lo)
  list(up = base_up + gain_up(eps_grid),
       lo = base_lo - gain_lo(eps_grid))
}

#' Sensitivity bounds on the treatment effect over a budget grid
#'
#' Computes the sensitivity bounds on the average treatment effect for the
#' always-observed subpopulation over the budget grid `delta`, without
#' covariates (closed form) or with covariates (stratum-by-stratum). With
#' covariates and `neighborhood = "tv"`, the default columns `tau_lower` /
#' `tau_upper` are the pooled (joint) bounds of `pooled_bounds_alloc()`,
#' and the within-stratum common-budget bounds ride along as
#' `tau_lower_pw` / `tau_upper_pw`. With `neighborhood = "contamination"`,
#' the bounds are the within-stratum common-budget aggregation of the
#' cell-level contamination bounds of `.tvb_cont_grid()`.
#'
#' `data` must carry columns `Y`, `S`, `D`; the covariate branch reads the
#' precomputed stratum factor `data$.tvb_stratum` when present (as built by
#' the bootstrap driver) and otherwise builds `interaction(data[covs])`.
#' Degenerate inputs (too few observed outcomes, inadmissible complier
#' share, no retained cell) return an all-`NA` data frame rather than
#' erroring, so that failed bootstrap replicates are counted rather than
#' fatal; a `NaN` complier share (empty treated arm in a resample) errors
#' and is caught by the `tryCatch` in `boot_tv_bounds()`.
#'
#' @param data Data frame with columns `Y`, `S`, `D` (and optionally
#'   `.tvb_stratum`).
#' @param delta Numeric vector of budget values in `[0, 1]`.
#' @param covs `NULL`, or a character vector of covariate column names.
#' @param neighborhood `"tv"` or `"contamination"`.
#' @param min_obs Minimum observed outcomes per arm for a cell to be kept.
#' @param verbose Emit a per-call stratum-retention `message()`.
#' @return Data frame with columns `delta`, `tau_lower`, `tau_upper` (and
#'   `tau_lower_pw`, `tau_upper_pw` in the covariate total variation case),
#'   with an attribute `"info"` carrying baseline and stratum detail.
#' @keywords internal
#' @noRd
tv_bounds <- function(data, delta, covs = NULL, neighborhood = "tv",
                      min_obs = 5, verbose = FALSE) {

  eps_grid <- delta
  grid_fun <- if (neighborhood == "contamination") .tvb_cont_grid
              else one_group_grid

  if (is.null(covs)) {
    # ---- Case 1 (no covariates) --------------------------------------------
    Yt <- data$Y[data$D == 1 & data$S == 1 & !is.na(data$Y)]
    Yc <- data$Y[data$D == 0 & data$S == 1 & !is.na(data$Y)]
    mu0 <- mean(Yc)
    ps  <- (mean(data$S[data$D == 1]) - mean(data$S[data$D == 0])) /
             mean(data$S[data$D == 1])
    # Monotonicity implies p1 >= p0, so a negative estimate is sampling noise:
    # project onto the admissible region (as in the covariate branch) rather
    # than silently NA-ing the whole replicate.
    ps  <- max(ps, 0)
    # Too few observed outcomes or an inadmissible complier share yield the
    # all-NA path; a NaN complier share (empty treated arm in a resample)
    # errors here, so failed replicates are still caught by the tryCatch in
    # boot_tv_bounds.
    if (length(Yt) < 2 || ps < 0 || ps >= 1)
      return(data.frame(delta = eps_grid, tau_lower = NA_real_,
                        tau_upper = NA_real_))
    g <- grid_fun(sort(Yt), ps, eps_grid)
    out <- data.frame(delta = eps_grid, tau_lower = g$lo - mu0,
                      tau_upper = g$up - mu0)
    attr(out, "info") <- list(
      p_star = ps, mu0 = mu0,
      n_treated_obs = length(Yt), n_control_obs = length(Yc))
    return(out)
  }

  # ---- Case 3 (stratum-by-stratum) ----------------------------------------
  # The bootstrap driver precomputes the stratum factor once on the original
  # data and resamples it with the rows; direct calls on raw data build it
  # here.
  st <- data[[".tvb_stratum"]]
  if (is.null(st)) st <- interaction(data[covs], drop = TRUE)
  # Bookkeeping on atomic vectors: subsetting the resampled data frame once
  # per stratum copies every column each time and dominates the bootstrap
  # runtime. The masks below select the same elements in the same order, so
  # every count and mean is unchanged.
  yv <- data$Y; sv <- data$S; dv <- data$D

  sc <- table(st[dv == 0 & sv == 1])
  sl <- names(sc[sc > 0]); usable <- c(); px <- c()
  nt_kept <- c(); nc_kept <- c()
  dropped_small <- character(0); dropped_pstar <- character(0)
  for (s in sl) {
    idx <- st == s
    nt <- sum(dv[idx] == 1 & sv[idx] == 1 & !is.na(yv[idx]))
    nc <- sum(dv[idx] == 0 & sv[idx] == 1 & !is.na(yv[idx]))
    if (nt < min_obs || nc < min_obs) {
      dropped_small <- c(dropped_small, s); next
    }
    p1 <- mean(sv[idx & dv == 1]); p0 <- mean(sv[idx & dv == 0])
    ps <- (p1 - p0) / p1
    if (is.na(ps) || ps >= 1) { dropped_pstar <- c(dropped_pstar, s); next }
    # Monotonicity implies p1 >= p0, so a negative estimate is sampling noise:
    # project onto the admissible region rather than discarding the stratum.
    ps <- max(ps, 0)
    usable <- c(usable, s); px[s] <- ps
    nt_kept[s] <- nt; nc_kept[s] <- nc
  }
  # A resample can in principle retain no cell at all; return NA rather than
  # silently aggregating over an empty set (which would yield tau = 0).
  if (length(usable) == 0)
    return(data.frame(delta = eps_grid, tau_lower = NA_real_,
                      tau_upper = NA_real_))

  uc <- as.numeric(sc[usable]); names(uc) <- usable
  wts <- uc / sum(uc)

  # mu0_AT must be aggregated over the SAME strata and with the SAME weights
  # as the bounds below: both terms of tau estimate the always-observed in
  # the retained cells. Averaging Y over all control respondents instead
  # would mix two different populations whenever some strata are dropped.
  mu0_x <- vapply(usable, function(s)
    mean(yv[st == s & dv == 0 & sv == 1], na.rm = TRUE),
    numeric(1))
  names(mu0_x) <- usable
  mu0_AT <- sum(wts * mu0_x)

  # Coverage of the retained cells, as a share of the control-respondent mass.
  attr_cov <- sum(uc) / sum(as.numeric(sc))
  if (verbose)
    message(sprintf("strata kept: %d/%d | coverage: %.1f%% | mu0_AT: %.4f",
                    length(usable), length(sl), 100 * attr_cov, mu0_AT))

  Y_strata <- lapply(usable, function(s)
    yv[st == s & dv == 1 & sv == 1 & !is.na(yv)])
  names(Y_strata) <- usable

  # One vectorized sweep per stratum: each stratum's outcomes are sorted once
  # and the whole bounds path comes from the group kernel. The weighted
  # aggregation accumulates over strata elementwise in the grid.
  lo <- up <- rep(0, length(eps_grid))
  for (j in seq_along(usable)) {
    s <- usable[j]
    Ys <- Y_strata[[s]]
    # Every retained stratum must return a bound: skipping one here without
    # renormalizing wts would silently drop weight and bias the aggregate.
    if (length(Ys) < 2 || px[s] < 0 || px[s] >= 1)
      stop("group kernel returned NA for retained stratum ", s)
    g <- grid_fun(sort(Ys), px[s], eps_grid)
    w_j <- wts[s]
    lo <- lo + w_j * g$lo
    up <- up + w_j * g$up
  }

  info <- list(
    strata = data.frame(
      stratum = usable,
      weight = unname(wts[usable]),
      p_star = unname(px[usable]),
      n_treated_obs = unname(nt_kept[usable]),
      n_control_obs = unname(nc_kept[usable]),
      mu0 = unname(mu0_x[usable]),
      stringsAsFactors = FALSE),
    coverage = attr_cov, mu0_AT = mu0_AT,
    n_strata_total = length(sl),
    dropped_small = dropped_small, dropped_pstar = dropped_pstar)

  if (neighborhood == "contamination") {
    # Within-stratum common-budget aggregation of the cell-level
    # contamination bounds; the pooled allocation below is total variation
    # only (its budget-transfer arithmetic is specific to total variation).
    out <- data.frame(delta = eps_grid,
                      tau_lower = unname(lo) - mu0_AT,
                      tau_upper = unname(up) - mu0_AT)
    attr(out, "info") <- info
    return(out)
  }

  # ---- Pooled (joint) bounds: the default covariate bounds ----------------
  pooled <- pooled_bounds_alloc(Y_strata, px, wts, eps_grid)

  # The pooled program nests the common-budget (within-stratum) allocation
  # and coincides with it at delta = 0 (the MCAR point) and delta = 1 (the
  # covariate Lee bounds); a violation beyond floating-point noise means the
  # greedy fill went wrong, so fail the call (a bootstrap replicate then
  # counts as failed and the n_fail warning surfaces it) rather than
  # returning a bad curve. The endpoint identities are checked at whichever
  # grid points equal 0 or 1.
  tol <- 1e-7 * (1 + max(abs(c(lo, up))))
  i0 <- which(eps_grid == 0); i1 <- which(eps_grid == 1)
  if (any(pooled$up < up - tol) || any(pooled$lo > lo + tol) ||
      any(abs(pooled$up[i0] - up[i0]) > tol) ||
      any(abs(pooled$lo[i0] - lo[i0]) > tol) ||
      any(abs(pooled$up[i1] - up[i1]) > tol) ||
      any(abs(pooled$lo[i1] - lo[i1]) > tol))
    stop("pooled allocation violates its envelope/endpoint identities")

  out <- data.frame(delta = eps_grid,
                    tau_lower = unname(pooled$lo) - mu0_AT,
                    tau_upper = unname(pooled$up) - mu0_AT,
                    tau_lower_pw = unname(lo) - mu0_AT,
                    tau_upper_pw = unname(up) - mu0_AT)
  attr(out, "info") <- info
  out
}

#' Lee (2009) bounds as the budget-one endpoint
#'
#' Lee bounds equal the total variation bounds at budget one; computed with
#' the same atom-safe formula for consistency.
#'
#' @param data Data frame with columns `Y`, `S`, `D`.
#' @param covs `NULL`, or a character vector of covariate column names.
#' @param min_obs Minimum observed outcomes per arm for a cell to be kept.
#' @return List with components `lower` and `upper`.
#' @keywords internal
#' @noRd
lee_bounds <- function(data, covs = NULL, min_obs = 5) {
  b <- tv_bounds(data, delta = 1, covs = covs, neighborhood = "tv",
                 min_obs = min_obs, verbose = FALSE)
  list(lower = b$tau_lower[1], upper = b$tau_upper[1])
}

#' Nonparametric bootstrap of the whole bounds curve
#'
#' The bound carries no structural parameter and no estimated moment, so the
#' Fang-Santos bootstrap of the paper reduces to the standard nonparametric
#' bootstrap of the empirical process. Crucially the FULL sample
#' `(Y, S, D, X)` is resampled jointly rather than only the treated
#' respondents that form the baseline: every replicate then redraws the
#' arm-specific response rates, hence the complier share `p_star` (and each
#' cell-level share), so the bootstrap automatically carries the estimation
#' uncertainty in `p_star` that a resample of the baseline alone would hold
#' fixed. When `cluster` is supplied, whole clusters are resampled with
#' replacement and all their rows are kept, replicating cluster-level
#' randomization designs.
#'
#' Uses the current RNG state; seeding (with save/restore) is handled by the
#' user-facing [tvbounds_attrition()].
#'
#' @param data Data frame with columns `Y`, `S`, `D` (and `.tvb_stratum`
#'   when `covs` is non-`NULL`).
#' @param delta Numeric vector of budget values in `[0, 1]`.
#' @param covs `NULL`, or a character vector of covariate column names.
#' @param neighborhood `"tv"` or `"contamination"`.
#' @param B Number of bootstrap replications.
#' @param min_obs Minimum observed outcomes per arm for a cell to be kept.
#' @param cluster `NULL`, or a vector of cluster identifiers (one per row).
#' @return List with components `delta`, `lo` and `up` (`B x length(delta)`
#'   matrices of bound draws), `n_fail`, and `B`.
#' @keywords internal
#' @noRd
boot_tv_bounds <- function(data, delta, covs = NULL, neighborhood = "tv",
                           B = 1000, min_obs = 5, cluster = NULL) {
  n <- nrow(data)
  # Each replicate copies the resampled data frame, so carry only the columns
  # tv_bounds reads. The stratum factor is a row-wise function of the
  # covariates, so it is built once and resampled with the rows; levels
  # absent from a resample are dropped by the sc > 0 filter either way.
  slim <- data.frame(Y = data$Y, S = data$S, D = data$D)
  if (!is.null(covs)) {
    slim$.tvb_stratum <- data[[".tvb_stratum"]]
    if (is.null(slim$.tvb_stratum))
      slim$.tvb_stratum <- interaction(data[covs], drop = TRUE)
  }
  if (!is.null(cluster)) {
    ids <- unique(cluster)
    cidx <- split(seq_len(n), match(cluster, ids))
  }
  lo_mat <- up_mat <- matrix(NA_real_, nrow = B, ncol = length(delta))
  n_fail <- 0L
  for (b in seq_len(B)) {
    rows <- if (is.null(cluster)) {
      sample.int(n, n, replace = TRUE)
    } else {
      draw <- sample.int(length(ids), length(ids), replace = TRUE)
      unlist(cidx[draw], use.names = FALSE)
    }
    db <- slim[rows, , drop = FALSE]
    bb <- tryCatch(tv_bounds(db, delta, covs, neighborhood, min_obs,
                             verbose = FALSE),
                   error = function(e) NULL)
    if (is.null(bb)) { n_fail <- n_fail + 1L; next }
    lo_mat[b, ] <- bb$tau_lower; up_mat[b, ] <- bb$tau_upper
  }
  list(delta = delta, lo = lo_mat, up = up_mat, n_fail = n_fail, B = B)
}

#' Bootstrap summaries: standard errors and percentile band at each budget
#'
#' The width is bootstrapped directly rather than combining the two bounds'
#' standard errors: the lower and upper bounds are computed from the same
#' resample and are strongly correlated, so `sd(up)` and `sd(lo)` do not
#' determine `sd(up - lo)`.
#'
#' @param bt Output of `boot_tv_bounds()`.
#' @param level Confidence level of the percentile band.
#' @return Data frame with per-budget standard errors, percentile interval
#'   endpoints for each bound, the width standard error, and the number of
#'   successful replicates.
#' @keywords internal
#' @noRd
boot_summary <- function(bt, level = 0.95) {
  alpha <- (1 - level) / 2
  q <- function(M, p) apply(M, 2, stats::quantile, probs = p, na.rm = TRUE)
  wid <- bt$up - bt$lo
  data.frame(
    delta       = bt$delta,
    lo_est_se   = apply(bt$lo, 2, stats::sd, na.rm = TRUE),
    up_est_se   = apply(bt$up, 2, stats::sd, na.rm = TRUE),
    w_est_se    = apply(wid,   2, stats::sd, na.rm = TRUE),
    lo_ci_low   = q(bt$lo, alpha),     lo_ci_high = q(bt$lo, 1 - alpha),
    up_ci_low   = q(bt$up, alpha),     up_ci_high = q(bt$up, 1 - alpha),
    n_eff       = apply(bt$lo, 2, function(x) sum(!is.na(x))))
}

#' Breakdown budget: first budget at which zero enters the bounds
#'
#' Smallest grid budget at which `0` lies inside `[lo, up]`,
#' linearly interpolated between grid points; `0` if the interval already
#' covers zero at the first grid point, `NA` if it never does on the grid
#' (censored breakdown).
#'
#' @param dg Numeric vector of budget grid values (sorted increasing).
#' @param lo,up Numeric vectors of lower/upper bounds on the same grid.
#' @return Scalar budget value, or `NA_real_`.
#' @keywords internal
#' @noRd
breakdown_delta <- function(dg, lo, up) {
  if (all(is.na(lo)) || all(is.na(up))) return(NA_real_)
  inside <- (lo <= 0) & (up >= 0)
  if (isTRUE(inside[1])) return(0)
  k <- which(inside); if (length(k) == 0) return(NA_real_)
  i <- k[1]; d0 <- dg[i - 1]; d1 <- dg[i]
  if (lo[i - 1] > 0)      return(d0 + (lo[i - 1] / (lo[i - 1] - lo[i])) * (d1 - d0))
  else if (up[i - 1] < 0) return(d0 + ((-up[i - 1]) / (up[i] - up[i - 1])) * (d1 - d0))
  d1
}
