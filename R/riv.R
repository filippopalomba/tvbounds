# =============================================================================
# riv.R -- user-facing interface for the recentered instrumental variables
# (formula instruments) application of Palomba (2026), Section SA7.3.
# The computational kernels live in R/riv-kernels.R.
# =============================================================================

#' Sensitivity bounds for recentered instrumental variables
#'
#' Computes sensitivity bounds on a recentered (formula) instrumental-variables
#' estimate when the postulated distribution of the shocks is allowed to vary
#' within a total variation or a contamination neighborhood of the baseline
#' assignment distribution \eqn{P_{*}}{P_*}, as in Palomba (2026).  The leading
#' use case is the recentered instruments of Borusyak and Hull (2023), whose
#' validity rests on a researcher-postulated distribution for the shock
#' process: the bounds quantify how far the estimate can move when up to a
#' fraction `delta` of that postulated probability mass is misspecified.
#'
#' @details
#' The exercise is conducted conditionally on the realized sample, so every
#' bound is a deterministic function of the data and of the budget `delta`;
#' accordingly, and by design (as in the paper), **no standard errors or
#' confidence bands are produced** for this application.
#'
#' **The model and the recentered estimate.**  For units
#' \eqn{i \in [n]}{i in [n]} the structural equation is
#' \deqn{y_i = \beta x_i + \varepsilon_i,}{y_i = beta * x_i + eps_i,}
#' with \eqn{\beta}{beta} the parameter of interest, \eqn{y_i}{y_i} the
#' outcome, \eqn{x_i}{x_i} the endogenous regressor and
#' \eqn{\varepsilon_i}{eps_i} the unobserved residual.  Let \eqn{v}{v} denote
#' the vector of exogenous shocks, taking values in a space
#' \eqn{\mathcal{V}}{V}, let \eqn{w}{w} collect predetermined covariates, and
#' let \eqn{f_i(\cdot\,;w) : \mathcal{V} \to \mathbb{R}}{f_i( . ; w) : V -> R}
#' be the known formula that maps a shock configuration into the instrument of
#' unit \eqn{i}{i}, so that \eqn{z_i := f_i(v;w)}{z_i := f_i(v; w)} is the
#' candidate instrument at the realized shocks.  For a distribution \eqn{P}{P}
#' on \eqn{\mathcal{V}}{V}, the expected instrument and the recentered
#' instrument are
#' \deqn{\mu_i(P) := \mathbb{E}_P[f_i(v;w) \mid w]
#'   = \int_{\mathcal{V}} f_i(v';w) \,\mathrm{d}P(v'), \qquad
#'   \widetilde{z}_i(P) := z_i - \mu_i(P).}{
#'   mu_i(P) := E_P[f_i(v; w) | w] = Int_V f_i(v'; w) dP(v'),
#'   ztilde_i(P) := z_i - mu_i(P).}
#' Borusyak and Hull (2023) postulate an assignment distribution
#' \eqn{P_{*}}{P_*} for the shocks and recenter at it.
#'
#' **The two criterion functions.**  The formula enters only through the two
#' sample aggregates
#' \deqn{g_y(\cdot) := \sum_{i=1}^{n} y_i f_i(\cdot\,;w), \qquad
#'   g_x(\cdot) := \sum_{i=1}^{n} x_i f_i(\cdot\,;w),}{
#'   g_y( . ) := sum_i y_i f_i( . ; w),   g_x( . ) := sum_i x_i f_i( . ; w),}
#' whose recentered values are the reduced form
#' \eqn{G_y(P) := g_y(v) - \mathbb{E}_P[g_y]}{G_y(P) := g_y(v) - E_P[g_y]} and
#' the first stage
#' \eqn{G_x(P) := g_x(v) - \mathbb{E}_P[g_x]}{G_x(P) := g_x(v) - E_P[g_x]}.
#' The estimate at a candidate assignment distribution is the ratio
#' \deqn{\widehat{\beta}(P) := \frac{G_y(P)}{G_x(P)}
#'   = \frac{\sum_{i=1}^{n} \widetilde{z}_i(P) \, y_i}{
#'           \sum_{i=1}^{n} \widetilde{z}_i(P) \, x_i},}{
#'   betahat(P) := G_y(P) / G_x(P)
#'     = sum_i ztilde_i(P) y_i / sum_i ztilde_i(P) x_i,}
#' and the reported estimate is
#' \eqn{\widehat{\beta}_{*} = \widehat{\beta}(P_{*})}{betahat_* = betahat(P_*)}.
#'
#' **How the arguments encode the shock space.**  The package represents
#' \eqn{P_{*}}{P_*} by `S` shock configurations
#' \eqn{v^{(1)}, \dots, v^{(S)}}{v^(1), ..., v^(S)}: entry `F[i, s]` holds
#' \eqn{f_i(v^{(s)};w)}{f_i(v^(s); w)}, the formula of unit `i` at the `s`-th
#' configuration, `p[s]` holds the probability that \eqn{P_{*}}{P_*} assigns to
#' that configuration, and `z[i]` holds \eqn{z_i = f_i(v;w)}{z_i = f_i(v; w)}.
#' The shock space is therefore taken to be the finite set
#' \eqn{\mathcal{V} = \{v^{(1)}, \dots, v^{(S)}\}}{V = {v^(1), ..., v^(S)}},
#' over which the candidate distributions \eqn{P}{P} range.
#'
#' **Neighborhoods.**  The bounds report the range of
#' \eqn{\widehat{\beta}(P)}{betahat(P)} as \eqn{P}{P} ranges over the chosen
#' neighborhood of \eqn{P_{*}}{P_*}, intersected with the set
#' \eqn{\mathcal{P}_{\neq 0} := \{P \in \Delta(\mathcal{V}) : G_x(P) \neq
#' 0\}}{P_nonzero := {P in Delta(V) : G_x(P) != 0}} of distributions at which
#' the ratio is defined:
#' * `neighborhood = "tv"` (the default): the total variation ball
#'   \deqn{\mathcal{P}^{\mathsf{FI}}_{\mathsf{TV}}(\delta) := \{P \in
#'     \Delta(\mathcal{V}) : \mathsf{TV}(P, P_{*}) \leq \delta\},}{
#'     P^FI_TV(delta) := {P in Delta(V) : TV(P, P_*) <= delta},}
#'   which delivers the bounds
#'   \eqn{\underline{\beta}_{\mathsf{TV}}(\delta)}{beta_TV_lower(delta)} and
#'   \eqn{\overline{\beta}_{\mathsf{TV}}(\delta)}{beta_TV_upper(delta)}.  They
#'   are computed from the closed form the paper gives for the criterion
#'   bounds
#'   \eqn{\underline{\mathsf{g}}_{\mathsf{TV}}(h;\delta)}{g_TV_lower(h; delta)}
#'   and
#'   \eqn{\overline{\mathsf{g}}_{\mathsf{TV}}(h;\delta)}{g_TV_upper(h; delta)},
#'   the smallest and the largest value of
#'   \eqn{\mathbb{E}_P[h]}{E_P[h]} over the ball: writing \eqn{q_h}{q_h} for
#'   the quantile function of a criterion \eqn{h}{h} under \eqn{P_{*}}{P_*},
#'   and \eqn{\overline{h}}{h_sup}, \eqn{\underline{h}}{h_inf} for its
#'   extremes over \eqn{\mathcal{V}}{V},
#'   \deqn{\overline{\mathsf{g}}_{\mathsf{TV}}(h;\delta)
#'       = \delta \overline{h} + \int_{\delta}^{1} q_h(u) \,\mathrm{d}u,
#'     \qquad
#'     \underline{\mathsf{g}}_{\mathsf{TV}}(h;\delta)
#'       = \delta \underline{h} + \int_{0}^{1-\delta} q_h(u) \,\mathrm{d}u,}{
#'     g_TV_upper(h; delta) = delta * h_sup + Int_delta^1 q_h(u) du,
#'     g_TV_lower(h; delta) = delta * h_inf + Int_0^(1 - delta) q_h(u) du,}
#'   that is, the baseline mean of \eqn{h}{h} once a tail of mass
#'   \eqn{\delta}{delta} has been trimmed and relocated to the most (least)
#'   favorable configuration.  Applied to the one-parameter family of criteria
#'   \eqn{g_b := g_y - b \, g_x}{g_b := g_y - b * g_x}, for which
#'   \eqn{\widehat{\beta}(P) = b}{betahat(P) = b} holds exactly when
#'   \eqn{\mathbb{E}_P[g_b] = g_b(v)}{E_P[g_b] = g_b(v)}, each bound is the
#'   unique root in \eqn{b}{b} of a strictly monotone function and is located
#'   by bracketed root finding;
#' * `neighborhood = "contamination"`: the contamination neighborhood
#'   \deqn{\mathcal{P}^{\mathsf{FI}}_{\mathsf{cont}}(\delta) := \{P \in
#'     \Delta(\mathcal{V}) : P = \delta R + (1-\delta) P_{*}, \;
#'     R \in \Delta(\mathcal{V})\},}{
#'     P^FI_cont(delta) := {P in Delta(V) : P = delta * R + (1 - delta) * P_*,
#'     R in Delta(V)},}
#'   in which the contamination share is the budget \eqn{\delta}{delta} itself,
#'   delivering the bounds
#'   \eqn{\underline{\beta}_{\mathsf{cont}}(\delta)}{beta_cont_lower(delta)}
#'   and \eqn{\overline{\beta}_{\mathsf{cont}}(\delta)}{beta_cont_upper(delta)}.
#'   These are attained by contaminating distributions \eqn{R}{R} degenerate at
#'   a single shock configuration, so they are obtained by enumerating the `S`
#'   configurations rather than by optimization.
#'
#' Every member of
#' \eqn{\mathcal{P}^{\mathsf{FI}}_{\mathsf{cont}}(\delta)}{P^FI_cont(delta)}
#' lies in
#' \eqn{\mathcal{P}^{\mathsf{FI}}_{\mathsf{TV}}(\delta)}{P^FI_TV(delta)}, so
#' the contamination bounds are weakly tighter at every budget.
#'
#' **First-stage breakdown.**  Once the budget is large enough that some
#' distribution in the neighborhood makes the recentered first stage
#' \eqn{G_x(P)}{G_x(P)} vanish, \eqn{\widehat{\beta}(P)}{betahat(P)} is no
#' longer well defined over the whole neighborhood and the identified set is
#' the entire real line.  The smallest such budget is the first-stage
#' breakdown budget,
#' \eqn{\delta^{\mathsf{TV}}_{\mathsf{FS}}}{delta^TV_FS} for the total
#' variation ball and
#' \eqn{\delta^{\mathsf{cont}}_{\mathsf{FS}}}{delta^cont_FS} for the
#' contamination neighborhood; since the contamination neighborhood is the
#' smaller of the two,
#' \eqn{\delta^{\mathsf{TV}}_{\mathsf{FS}} \leq
#' \delta^{\mathsf{cont}}_{\mathsf{FS}}}{delta^TV_FS <= delta^cont_FS}.  The
#' one for the neighborhood in use is reported in `details$delta_fs`.
#' Following the paper, the infimum over an empty set is set to 1, so a
#' reported value of 1 carrying `delta_fs_censored = TRUE` means that the
#' first stage never breaks down over the budget range, not that breakdown
#' occurs at 1.  Rows of `bounds` at budgets where the bounds are vacuous
#' carry `NA`.
#'
#' **Controls.**  When `controls` is supplied, `y`, `x`, `z`, and every column
#' of `F` are residualized on the controls and a constant (when
#' `controls = NULL`, on the constant alone) before the bounds are computed.
#' By the Frisch--Waugh--Lovell theorem this leaves the just-identified
#' two-stage least squares coefficient on `x` unchanged, because the
#' projection matrix is idempotent; the baseline estimate then replicates the
#' estimate from the full regression with controls.
#'
#' @param y Numeric n-vector, the outcome \eqn{y_i}{y_i}.
#' @param x Numeric n-vector, the endogenous regressor \eqn{x_i}{x_i}.
#' @param z Numeric n-vector, the realized (un-recentered) candidate
#'   instrument: `z[i]` is the formula of unit `i` evaluated at the realized
#'   shocks, \eqn{z_i = f_i(v;w)}{z_i = f_i(v; w)}.
#' @param F Numeric n x S matrix of counterfactual instrument draws:
#'   `F[i, s]` is the formula of unit `i` evaluated at the `s`-th
#'   counterfactual shock configuration, \eqn{f_i(v^{(s)};w)}{f_i(v^(s); w)}.
#'   These are the draws from the postulated assignment process used for
#'   recentering (in Borusyak and Hull (2023), permuted or re-simulated shock
#'   allocations).
#' @param p Optional numeric S-vector holding the probabilities that the
#'   postulated assignment distribution \eqn{P_{*}}{P_*} attaches to the
#'   columns of `F`, that is to the configurations
#'   \eqn{v^{(1)}, \dots, v^{(S)}}{v^(1), ..., v^(S)}.  Defaults to `NULL`,
#'   meaning the uniform distribution `1/S` on each draw, which is the
#'   postulated assignment distribution of Borusyak and Hull (2023).  Must be
#'   nonnegative and sum to one.
#' @param controls Optional numeric matrix or data frame of control
#'   variables (n rows) to be partialled out of `y`, `x`, `z`, and each
#'   column of `F`.  A constant is always included.  Default `NULL`.
#' @param delta Numeric vector of sensitivity budgets
#'   \eqn{\delta \in [0,1]}{delta in [0, 1]} at which the bounds are traced.
#'   Default `seq(0, 1, by = 0.002)`.
#' @param neighborhood Either `"tv"` (the total variation ball
#'   \eqn{\mathcal{P}^{\mathsf{FI}}_{\mathsf{TV}}(\delta)}{P^FI_TV(delta)},
#'   the default) or `"contamination"` (the contamination neighborhood
#'   \eqn{\mathcal{P}^{\mathsf{FI}}_{\mathsf{cont}}(\delta)}{P^FI_cont(delta)}).
#' @param tau_star Numeric scalar reference value \eqn{\tau_{\star}}{tau_star}
#'   for the breakdown budget \eqn{\delta_b(\tau_{\star})}{delta_b(tau_star)}:
#'   the smallest budget at which the bounds cover
#'   \eqn{\tau_{\star}}{tau_star}.  The default `0` gives the breakdown budget
#'   for the sign of \eqn{\beta}{beta}.
#' @param verbose Logical; if `TRUE`, print progress messages.  Default
#'   `FALSE`.
#'
#' @return An object of class `c("tvbounds_riv", "tvbounds")`, a list with
#'   entries
#' * `application`: `"riv"`.
#' * `bounds`: data frame with one row per budget and columns `delta`,
#'   `lower`, `upper`.  Rows at budgets where the neighborhood contains a
#'   distribution collapsing the recentered first stage carry `NA` (the
#'   bounds are vacuous there).  No standard error or confidence interval
#'   columns are attached: this application carries no inference by design.
#' * `point`: the reported recentered IV estimate
#'   \eqn{\widehat{\beta}_{*} = \widehat{\beta}(P_{*})}{betahat_* =
#'   betahat(P_*)}, which is the common value of the two bounds at
#'   `delta = 0`.
#' * `n`: number of observations.
#' * `neighborhood`: the neighborhood used.
#' * `level`, `B`: `NA` (no inference).
#' * `estimand_label`: `"IV coefficient"`.
#' * `call`: the matched call.
#' * `details`: a list with
#'     * `criteria`: the reduced criterion object consumed by the internal
#'       kernels, holding the realized values \eqn{g_y(v)}{g_y(v)} and
#'       \eqn{g_x(v)}{g_x(v)}, the two S-vectors
#'       \eqn{g_y(v^{(s)})}{g_y(v^(s))} and \eqn{g_x(v^{(s)})}{g_x(v^(s))},
#'       and the baseline aggregates
#'       \eqn{\mathbb{E}_{P_{*}}[g_y]}{E_{P_*}[g_y]} and
#'       \eqn{\mathbb{E}_{P_{*}}[g_x]}{E_{P_*}[g_x]}; bounds at additional
#'       budgets can be recomputed from it without touching the n x S design
#'       again.
#'     * `delta_fs`, `delta_fs_censored`: the first-stage breakdown budget
#'       for the neighborhood used,
#'       \eqn{\delta^{\mathsf{TV}}_{\mathsf{FS}}}{delta^TV_FS} or
#'       \eqn{\delta^{\mathsf{cont}}_{\mathsf{FS}}}{delta^cont_FS}, and
#'       whether it is censored at 1 (`TRUE` when the first stage never
#'       breaks down on \eqn{[0,1]}, so 1 is the empty-set convention rather
#'       than an actual breakdown).
#'     * `delta_fs_tv`, `delta_fs_cont`: both first-stage breakdown budgets,
#'       \eqn{\delta^{\mathsf{TV}}_{\mathsf{FS}}}{delta^TV_FS} and
#'       \eqn{\delta^{\mathsf{cont}}_{\mathsf{FS}}}{delta^cont_FS} (each
#'       carrying its `"censored"` attribute).
#'     * `delta_breakdown`, `delta_breakdown_censored`: the smallest budget
#'       \eqn{\delta_b(\tau_{\star})}{delta_b(tau_star)} at which the bounds
#'       for the chosen neighborhood cover \eqn{\tau_{\star}}{tau_star},
#'       reported as 1 with `delta_breakdown_censored = TRUE` when
#'       \eqn{\tau_{\star}}{tau_star} is never covered on \eqn{[0,1]}.
#'     * `tau_star`: the reference value \eqn{\tau_{\star}}{tau_star} used.
#'     * `n_controls`: number of control columns partialled out (0 when
#'       `controls = NULL`).
#'
#' @references
#' Palomba, F. (2026). "Sensitivity Analysis in Population Shares."
#' Working paper.
#'
#' Borusyak, K. and Hull, P. (2023). "Nonrandom Exposure to Exogenous
#' Shocks." *Econometrica*, 91(6), 2155--2185.
#'
#' @examples
#' # A small simulated formula-instrument design: S counterfactual shock
#' # configurations, a realized instrument, a first stage, and an outcome.
#' set.seed(123)
#' n <- 80
#' S <- 50
#' Fmat <- matrix(rnorm(n * S), n, S)     # Fmat[i, s] = f_i(v^(s); w)
#' z <- rowMeans(Fmat) + rnorm(n)         # realized instrument z_i = f_i(v; w)
#' x <- z + 0.5 * rnorm(n)                # endogenous regressor
#' y <- 0.4 * x + rnorm(n)                # outcome
#'
#' fit <- tvbounds_riv(y, x, z, Fmat, delta = seq(0, 1, by = 0.05))
#' fit$point                              # recentered IV estimate at P_*
#' head(fit$bounds)                       # bounds along the budget grid
#' fit$details$delta_fs                   # first-stage breakdown budget
#' fit$details$delta_breakdown            # breakdown budget for the sign
#'
#' # Contamination neighborhood, with controls partialled out.
#' W <- data.frame(w1 = rnorm(n), w2 = rnorm(n))
#' fit_cont <- tvbounds_riv(y, x, z, Fmat, controls = W,
#'                          delta = seq(0, 1, by = 0.05),
#'                          neighborhood = "contamination")
#' fit_cont$point
#'
#' @export
tvbounds_riv <- function(y, x, z, F, p = NULL, controls = NULL,
                         delta = seq(0, 1, by = 0.002),
                         neighborhood = c("tv", "contamination"),
                         tau_star = 0, verbose = FALSE) {
  cl <- match.call()
  neighborhood <- match.arg(neighborhood)
  Fmat <- F

  # --- input validation (fail early, name the argument) ----------------------
  .tvb_check_numeric_vector(y, "y")
  .tvb_check_numeric_vector(x, "x")
  .tvb_check_numeric_vector(z, "z")
  n <- length(y)
  if (length(x) != n || length(z) != n) {
    stop("`y`, `x`, and `z` must have the same length.", call. = FALSE)
  }
  if (n < 2L) stop("`y` must contain at least 2 observations.", call. = FALSE)
  if (is.data.frame(Fmat)) Fmat <- as.matrix(Fmat)
  if (!is.matrix(Fmat) || !is.numeric(Fmat)) {
    stop("`F` must be a numeric matrix (n x S) of counterfactual instrument draws.",
         call. = FALSE)
  }
  if (nrow(Fmat) != n) {
    stop("`F` must have one row per observation (nrow(F) == length(y)).",
         call. = FALSE)
  }
  S <- ncol(Fmat)
  if (S < 2L) {
    stop("`F` must contain at least 2 counterfactual draws (columns).",
         call. = FALSE)
  }
  if (anyNA(Fmat) || any(!is.finite(Fmat))) {
    stop("`F` must not contain missing or non-finite values.", call. = FALSE)
  }
  if (!is.null(p)) {
    if (!is.numeric(p) || length(p) != S) {
      stop("`p` must be a numeric vector with one probability per column of `F`.",
           call. = FALSE)
    }
    if (anyNA(p) || any(p < 0)) {
      stop("`p` must be nonnegative with no missing values.", call. = FALSE)
    }
    if (abs(sum(p) - 1) > 1e-8) {
      stop("`p` must sum to 1.", call. = FALSE)
    }
    p <- p / sum(p)                       # exact renormalization
  }
  if (!is.numeric(delta) || length(delta) == 0L || anyNA(delta) ||
      any(!is.finite(delta))) {
    stop("`delta` must be a non-empty numeric vector with no missing values.",
         call. = FALSE)
  }
  if (any(delta < 0) || any(delta > 1)) {
    stop("`delta` must lie in [0, 1] for the \"tv\" and \"contamination\" neighborhoods.",
         call. = FALSE)
  }
  if (anyDuplicated(delta)) {
    stop("`delta` must not contain duplicated budget values.", call. = FALSE)
  }
  if (!is.numeric(tau_star) || length(tau_star) != 1L || !is.finite(tau_star)) {
    stop("`tau_star` must be a finite numeric scalar.", call. = FALSE)
  }
  if (!is.logical(verbose) || length(verbose) != 1L || is.na(verbose)) {
    stop("`verbose` must be TRUE or FALSE.", call. = FALSE)
  }

  # --- Frisch-Waugh-Lovell residualization on the controls -------------------
  # In a just-identified two-stage least squares whose controls appear in both
  # stages, the coefficient on x is unchanged by partialling the controls out.
  # The projection is idempotent, so applying it to z and F as well (a column
  # at a time) leaves every criterion value unchanged while keeping all four
  # inputs on the same residualized scale.
  W <- .tvb_controls_matrix(controls, n)
  n_controls <- ncol(W) - 1L              # W always carries the constant
  qw <- qr(W)
  if (qw$rank >= n) {
    stop("`controls` has as many columns as observations; residualization is degenerate.",
         call. = FALSE)
  }
  yr <- qr.resid(qw, y)
  xr <- qr.resid(qw, x)
  zr <- qr.resid(qw, z)
  Fr <- qr.resid(qw, Fmat)

  # --- reduce the design to the two criterion functions ----------------------
  cr <- fi_criteria(y = yr, x = xr, z = zr, Fmat = Fr, p = p)
  if (verbose) {
    message(sprintf(
      "Recentered IV design: n = %d observations, S = %d counterfactual draws.",
      cr$n, cr$S))
    message(sprintf("Baseline recentered IV estimate: %.6g.", cr$beta_hat))
  }

  # --- first-stage breakdown budgets -----------------------------------------
  dfs_tv   <- fi_delta_fs_tv(cr)
  dfs_cont <- fi_delta_fs_cont(cr)
  dfs <- if (neighborhood == "tv") dfs_tv else dfs_cont
  if (verbose) {
    message(sprintf(
      "First-stage breakdown budget (%s): %.4f%s.", neighborhood,
      as.numeric(dfs),
      if (isTRUE(attr(dfs, "censored"))) " (censored: never breaks down)" else ""))
  }

  # --- bounds along the budget grid ------------------------------------------
  bm <- if (neighborhood == "tv") {
    t(vapply(delta, function(d) fi_bounds_tv(cr, d), numeric(2)))
  } else {
    t(vapply(delta, function(d) fi_bounds_cont(cr, d)[1:2], numeric(2)))
  }
  lower <- bm[, 1]
  upper <- bm[, 2]
  # Budgets at which the neighborhood contains a distribution collapsing the
  # first stage have a vacuous identified set (the whole real line); encode
  # those rows as NA in the common bounds schema.
  vacuous <- !is.finite(lower) | !is.finite(upper)
  lower[vacuous] <- NA_real_
  upper[vacuous] <- NA_real_
  bounds <- data.frame(delta = delta, lower = lower, upper = upper)

  # --- breakdown budget for the reference value ------------------------------
  lo_fun <- if (neighborhood == "tv") {
    function(d) fi_bounds_tv(cr, d)[["lower"]]
  } else {
    function(d) fi_bounds_cont(cr, d)[["lower"]]
  }
  up_fun <- if (neighborhood == "tv") {
    function(d) fi_bounds_tv(cr, d)[["upper"]]
  } else {
    function(d) fi_bounds_cont(cr, d)[["upper"]]
  }
  db <- fi_breakdown(lo_fun, up_fun, tau_star = tau_star)
  db_censored <- is.na(db)
  if (db_censored) db <- 1
  if (verbose) {
    message(sprintf(
      "Breakdown budget for tau_star = %.6g: %.4f%s.", tau_star, db,
      if (db_censored) " (censored: never covered)" else ""))
  }

  new_tvbounds(
    application    = "riv",
    bounds         = bounds,
    point          = cr$beta_hat,
    n              = cr$n,
    neighborhood   = neighborhood,
    level          = NA_real_,
    B              = NA_integer_,
    estimand_label = "IV coefficient",
    call           = cl,
    details        = list(
      criteria                 = cr,
      delta_fs                 = as.numeric(dfs),
      delta_fs_censored        = isTRUE(attr(dfs, "censored")),
      delta_fs_tv              = dfs_tv,
      delta_fs_cont            = dfs_cont,
      delta_breakdown          = db,
      delta_breakdown_censored = db_censored,
      tau_star                 = tau_star,
      n_controls               = n_controls
    )
  )
}


# -----------------------------------------------------------------------------
# Small validation helpers
# -----------------------------------------------------------------------------

#' Check that an argument is a finite numeric vector with no missing values
#' @keywords internal
#' @noRd
.tvb_check_numeric_vector <- function(v, name) {
  if (!is.numeric(v) || !is.null(dim(v))) {
    stop(sprintf("`%s` must be a numeric vector.", name), call. = FALSE)
  }
  if (anyNA(v) || any(!is.finite(v))) {
    stop(sprintf("`%s` must not contain missing or non-finite values.", name),
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Assemble the control design matrix (constant always included)
#' @keywords internal
#' @noRd
.tvb_controls_matrix <- function(controls, n) {
  if (is.null(controls)) {
    return(matrix(1, nrow = n, ncol = 1,
                  dimnames = list(NULL, "(Intercept)")))
  }
  if (is.data.frame(controls)) {
    if (!all(vapply(controls, is.numeric, logical(1)))) {
      stop("`controls` must contain numeric columns only.", call. = FALSE)
    }
    controls <- as.matrix(controls)
  }
  if (!is.matrix(controls) || !is.numeric(controls)) {
    stop("`controls` must be a numeric matrix or data frame.", call. = FALSE)
  }
  if (nrow(controls) != n) {
    stop("`controls` must have one row per observation.", call. = FALSE)
  }
  if (anyNA(controls) || any(!is.finite(controls))) {
    stop("`controls` must not contain missing or non-finite values.",
         call. = FALSE)
  }
  cbind("(Intercept)" = 1, controls)
}
