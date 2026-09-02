# Summary measures for tvbounds objects: breakdown budgets, shadow prices,
# robustness standard errors, and the certification frontier. The
# computational helpers below port, essentially verbatim, the audited
# helpers of the paper's summary-measure script
# (SA9_summary_measures_run.R): cross0(), eta_fun(),
# lin(), nstar(), and the body of summarize(), with the paper's defaults
# (two-sided critical value qnorm(0.975) at level 0.95, cost_per_unit = 50,
# jump = 0.05, tau_star = 0, integer frontier floor(n_j) + 1, and censored
# breakdown budgets reported at the right endpoint of the budget grid).

#' First crossing of a signed bound path with zero
#'
#' First budget \eqn{\delta}{delta} at which `f` reaches zero from above,
#' linearly interpolated between grid points. Returns the smallest grid
#' value when `f` is already nonpositive there, and `NA` when `f` stays
#' strictly positive on the whole grid, that is, when the breakdown budget
#' is censored at the right endpoint of the budget grid. Rows where `d` or
#' `f` is `NA` are ignored.
#'
#' @param d Increasing numeric grid of budgets \eqn{\delta}{delta}.
#' @param f Signed path evaluated on `d`.
#' @return A scalar budget, or `NA_real_` when censored.
#' @keywords internal
#' @noRd
.tvb_cross0 <- function(d, f) {
  ok <- !is.na(d) & !is.na(f)
  d <- d[ok]
  f <- f[ok]
  if (!length(d)) return(NA_real_)
  neg <- which(f <= 0)
  if (!length(neg)) return(NA_real_)
  i <- neg[1L]
  if (i == 1L) return(d[1L])
  d[i - 1L] + (d[i] - d[i - 1L]) * f[i - 1L] / (f[i - 1L] - f[i])
}

#' Shadow price of robustness along a bound path
#'
#' The shadow price \eqn{\underline{\eta}(\delta) = -\underline{\tau}'(\delta)}{eta(delta) = -tau_lower'(delta)},
#' computed by central differences on the interior of the grid and
#' one-sided differences at the endpoints. The grid need not be uniform.
#' With only two grid points the single slope is repeated; with fewer,
#' `NA`s are returned.
#'
#' @param d Increasing numeric grid of budgets \eqn{\delta}{delta}.
#' @param tl Signed bound path evaluated on `d`.
#' @return Numeric vector of the same length as `d`.
#' @keywords internal
#' @noRd
.tvb_eta_fun <- function(d, tl) {
  n <- length(d)
  if (n < 2L) return(rep(NA_real_, n))
  if (n == 2L) {
    s <- -(tl[2L] - tl[1L]) / (d[2L] - d[1L])
    return(c(s, s))
  }
  i <- 2L:(n - 1L)
  c(-(tl[2L] - tl[1L]) / (d[2L] - d[1L]),
    -(tl[i + 1L] - tl[i - 1L]) / (d[i + 1L] - d[i - 1L]),
    -(tl[n] - tl[n - 1L]) / (d[n] - d[n - 1L]))
}

#' Linear interpolation with flat extrapolation
#'
#' Interpolates `y` on the grid `d` at the point `at`, extrapolating flat
#' beyond the grid (`rule = 2`), as in the paper's summary-measure script.
#' Returns `NA` when `at` is `NA` or fewer than two complete pairs exist.
#'
#' @keywords internal
#' @noRd
.tvb_lin <- function(d, y, at) {
  if (is.na(at)) return(NA_real_)
  ok <- !is.na(d) & !is.na(y)
  if (sum(ok) < 2L) return(NA_real_)
  stats::approx(d[ok], y[ok], xout = at, rule = 2)$y
}

#' Certification frontier
#'
#' The frontier
#' \eqn{n^{\star}(\delta;\alpha) = z^{2}_{1-\alpha/2}\sigma^{2}(\delta)/(\underline{\tau}(\delta)-\tau_{\star})^{2}}{n*(delta; alpha) = z_{1-alpha/2}^2 sigma^2(delta) / (tau_lower(delta) - tau_star)^2},
#' evaluated with the estimated standard deviation
#' \eqn{\widehat{\sigma}_{n}(\delta) = \sqrt{n}\,\mathrm{se}(\delta)}{sigma_hat_n(delta) = sqrt(n) se(delta)}
#' and the signed path in place of
#' \eqn{\underline{\tau}(\delta)-\tau_{\star}}{tau_lower(delta) - tau_star}:
#' the sample size at which the normal-approximation confidence limit of
#' the bound path would just touch the reference value at budget `at`. `NA`
#' once the path has crossed the reference value: past the breakdown budget
#' no sample size certifies the conclusion.
#'
#' @keywords internal
#' @noRd
.tvb_nstar <- function(d, tl, se, n, at, zc) {
  t0 <- .tvb_lin(d, tl, at)
  s0 <- .tvb_lin(d, se, at)
  if (is.na(t0) || t0 <= 0 || is.na(s0)) return(NA_real_)
  zc^2 * (s0^2 * n) / t0^2
}

#' Resolve the direction of the robustness exercise
#'
#' `"auto"` selects the lower bound path
#' \eqn{\underline{\tau}(\delta)}{tau_lower(delta)} when the baseline point
#' estimate exceeds `tau_star` and the upper bound path
#' \eqn{\overline{\tau}(\delta)}{tau_upper(delta)} otherwise, mirroring the
#' paper's symmetric treatment of the two directions through the integrand
#' \eqn{-g}{-g}.
#'
#' @keywords internal
#' @noRd
.tvb_resolve_direction <- function(direction, point, tau_star) {
  if (direction != "auto") return(direction)
  if (is.null(point) || is.na(point)) {
    stop("`direction = \"auto\"` requires a non-missing baseline point ",
         "estimate in the object; set `direction` explicitly.",
         call. = FALSE)
  }
  if (point > tau_star) "lower" else "upper"
}

#' Extract the signed bound path adjacent to the reference value
#'
#' For `direction = "lower"` the signed path is
#' \eqn{\underline{\tau}(\delta) - \tau_{\star}}{tau_lower(delta) - tau_star};
#' for `direction = "upper"` it is
#' \eqn{\tau_{\star} - \overline{\tau}(\delta)}{tau_star - tau_upper(delta)},
#' so that in both cases the path starts positive when the conclusion holds
#' at the baseline and the breakdown budget is its first crossing with
#' zero. Standard errors are invariant to this sign-and-shift
#' transformation; the outer confidence limit maps to
#' `ci_lower - tau_star` and `tau_star - ci_upper` respectively. Rows with
#' a missing budget or a missing path value are dropped; `se` and `ci` are
#' `NULL` when the object does not carry the corresponding columns.
#'
#' @keywords internal
#' @noRd
.tvb_signed_path <- function(object, direction, tau_star) {
  b <- object$bounds
  if (direction == "lower") {
    path <- b$lower - tau_star
    se   <- if ("lower_se" %in% names(b)) b$lower_se else NULL
    ci   <- if ("ci_lower" %in% names(b)) b$ci_lower - tau_star else NULL
  } else {
    path <- tau_star - b$upper
    se   <- if ("upper_se" %in% names(b)) b$upper_se else NULL
    ci   <- if ("ci_upper" %in% names(b)) tau_star - b$ci_upper else NULL
  }
  ok <- !is.na(b$delta) & !is.na(path)
  list(d     = b$delta[ok],
       path  = path[ok],
       se    = if (!is.null(se)) se[ok] else NULL,
       ci    = if (!is.null(ci)) ci[ok] else NULL,
       right = if (any(ok)) max(b$delta[ok]) else NA_real_,
       direction = direction)
}

#' Summary measures for total-variation sensitivity bounds
#'
#' Computes the summary measures of Palomba (2026) for a `tvbounds` object:
#' the plug-in and certified breakdown budgets, the shadow price of
#' robustness, the robustness standard error, and the certification
#' frontier. The budget `delta` is the sensitivity parameter
#' \eqn{\delta}{delta} of the total-variation (or contamination, or
#' divergence) neighborhood over which the bounds were computed.
#'
#' @details
#' Write \eqn{\underline{\tau}(\delta)}{tau_lower(delta)} and
#' \eqn{\overline{\tau}(\delta)}{tau_upper(delta)} for the lower and upper
#' sensitivity bounds on the estimand at budget \eqn{\delta}{delta}. The
#' first is nonincreasing and the second nondecreasing in the budget, and
#' both collapse at \eqn{\delta = 0}{delta = 0} to the value of the
#' estimand under the baseline distribution \eqn{P_{*}}{P_*}. Robustness is
#' judged against a reference value \eqn{\tau_{\star}}{tau_star} of the
#' estimand, supplied through `tau_star`, and typically zero when it is the
#' sign of an effect rather than its magnitude that is of interest.
#'
#' All measures are computed on the signed bound path adjacent to
#' \eqn{\tau_{\star}}{tau_star}, namely
#' \eqn{\underline{\tau}(\delta) - \tau_{\star}}{tau_lower(delta) - tau_star}
#' when `direction` is `"lower"` and
#' \eqn{\tau_{\star} - \overline{\tau}(\delta)}{tau_star - tau_upper(delta)}
#' when it is `"upper"`, so that in both cases the path starts positive
#' when the conclusion holds at the baseline. The two directions are thus
#' treated symmetrically, as the paper treats them by replacing the
#' integrand \eqn{g}{g} by \eqn{-g}{-g}; the displays below are written for
#' the lower path. With `direction = "auto"` (the default) the lower path
#' is used when the baseline point estimate exceeds `tau_star` and the
#' upper path otherwise.
#'
#' The breakdown budget is the smallest budget at which the bounds cease to
#' exclude the reference value,
#' \deqn{\delta_{b}(\tau_{\star}) = \inf\{\delta \in [0,1] : \underline{\tau}(\delta) \le \tau_{\star} \le \overline{\tau}(\delta)\},}{delta_b(tau_star) = inf{delta in [0, 1] : tau_lower(delta) <= tau_star <= tau_upper(delta)},}
#' with the convention that the infimum over the empty set equals one. That
#' convention is exactly the package's censoring rule: when the estimated
#' path never reaches \eqn{\tau_{\star}}{tau_star} on the supplied budget
#' grid, the breakdown is reported at the right endpoint of the grid, which
#' is one for the total-variation and contamination neighborhoods, and is
#' flagged by `censored = TRUE` rather than recorded as an inequality.
#'
#' Sampling uncertainty is accounted for by the certified breakdown budget,
#' \deqn{\widehat{\delta}_{b}^{\,\mathsf{C}}(\alpha) = \inf\left\{\delta \in [0,1] : \widehat{\underline{\tau}}_{n}(\delta) - \frac{z_{1-\alpha/2}\,\widehat{\sigma}_{n}(\delta)}{\sqrt{n}} \le \tau_{\star}\right\},}{delta_b^C(alpha) = inf{delta in [0, 1] : tau_lower_hat(delta) - z_{1-alpha/2} sigma_hat(delta) / sqrt(n) <= tau_star},}
#' the largest budget at which the conclusion survives sampling
#' uncertainty. Here \eqn{\widehat{\underline{\tau}}_{n}(\delta)}{tau_lower_hat(delta)}
#' estimates the lower bound path, \eqn{\sigma(\delta)}{sigma(delta)} is
#' the asymptotic standard deviation of that estimator in units of the
#' estimand and \eqn{\widehat{\sigma}_{n}(\delta)}{sigma_hat(delta)} its
#' estimator, and \eqn{z_{1-\alpha/2}}{z_{1-alpha/2}} is the two-sided
#' normal critical value at the confidence level `level`.
#'
#' The shadow price of robustness is the marginal cost, in units of the
#' estimand, of one further unit of budget,
#' \eqn{\underline{\eta}(\delta) = -\underline{\tau}'(\delta)}{eta(delta) = -tau_lower'(delta)};
#' it is nonnegative and nonincreasing, and the package estimates it by
#' central differences on the budget grid. Dividing the sampling standard
#' deviation of the bound by the shadow price converts it from units of the
#' estimand into units of the budget, which yields the robustness standard
#' error
#' \deqn{\varsigma_{b} = \frac{\sigma(\delta_{b})}{\underline{\eta}(\delta_{b})}.}{varsigma_b = sigma(delta_b) / eta(delta_b).}
#'
#' The certification frontier is the sample size at which the population
#' counterpart of the certified-breakdown inequality just clears the
#' reference value at budget \eqn{\delta}{delta},
#' \deqn{n^{\star}(\delta;\alpha) = \frac{z^{2}_{1-\alpha/2}\,\sigma^{2}(\delta)}{(\underline{\tau}(\delta) - \tau_{\star})^{2}}.}{n*(delta; alpha) = z_{1-alpha/2}^2 sigma^2(delta) / (tau_lower(delta) - tau_star)^2.}
#' It is real-valued, so the smallest certifying integer sample size is
#' \eqn{\lfloor n^{\star}(\delta;\alpha)\rfloor + 1}{floor(n*(delta; alpha)) + 1}.
#' The frontier diverges as the budget approaches the breakdown budget and
#' is meaningful only below it: at and past the breakdown budget no sample
#' size certifies the conclusion, and the frontier is reported as `NA`. Its
#' semi-elasticity
#' \eqn{\mathrm{d}\log n^{\star}(\delta;\alpha)/\mathrm{d}\delta}{d log n*(delta; alpha) / d delta},
#' the certification elasticity, gives the rate at which the required
#' sample size grows with the budget; `cost_per_pp` prices a discrete
#' version of it.
#'
#' The one-row data frame `measures` reports the following, alongside the
#' symbol each one corresponds to in the paper:
#' * `delta_b` — the plug-in breakdown budget
#'   \eqn{\delta_{b}(\tau_{\star})}{delta_b(tau_star)}, obtained as the
#'   first crossing of the estimated signed path with zero, linearly
#'   interpolated between grid points; `censored` flags the empty-set
#'   convention described above.
#' * `delta_b_ci` — the certified breakdown budget
#'   \eqn{\widehat{\delta}_{b}^{\,\mathsf{C}}(\alpha)}{delta_b^C(alpha)},
#'   read off the outer confidence limit stored in the object (columns
#'   `ci_lower`/`ci_upper` of `object$bounds`), that is, off the same band
#'   the figures draw, so that tables and figures agree on one number. `NA`
#'   when the object carries no band, and `censored_ci` flags censoring as
#'   above.
#' * `delta_b_ci_norm` — the same certified breakdown budget in its
#'   normal-approximation form, the first crossing of
#'   \eqn{\widehat{\underline{\tau}}_{n}(\delta) - z_{1-\alpha/2}\widehat{\sigma}_{n}(\delta)/\sqrt{n}}{tau_lower_hat(delta) - z_{1-alpha/2} sigma_hat(delta) / sqrt(n)}
#'   with \eqn{\tau_{\star}}{tau_star}. It estimates the same population
#'   quantity as `delta_b_ci` and is retained as a diagnostic: the two
#'   differ only when the bootstrap distribution of the bound is
#'   asymmetric.
#' * `delta_eval` — the budget \eqn{\delta}{delta} at which the local
#'   measures `eta`, `se`, `varsigma`, and `varsigma_sc` are evaluated.
#' * `eta` — the shadow price of robustness
#'   \eqn{\underline{\eta}(\delta)}{eta(delta)} at `delta_eval`.
#' * `se` — the estimated standard error of the bound in units of the
#'   estimand at `delta_eval`, that is,
#'   \eqn{\widehat{\sigma}_{n}(\delta)/\sqrt{n}}{sigma_hat(delta) / sqrt(n)}.
#' * `varsigma` — the robustness standard error
#'   \eqn{\varsigma_{b} = \sigma(\delta_{b})/\underline{\eta}(\delta_{b})}{varsigma_b = sigma(delta_b) / eta(delta_b)},
#'   computed as `se * sqrt(n) / eta` and therefore expressed in units of
#'   the budget rather than in units of the estimand.
#' * `varsigma_sc` — the finite-sample analogue `se / eta`, equal to
#'   \eqn{\varsigma_{b}/\sqrt{n}}{varsigma_b / sqrt(n)}, which is the
#'   sampling standard deviation of the breakdown budget itself at the
#'   realized sample size.
#' * `frontier_at`, `n_cur`, `n_star`, `delta_n`, `cost_per_pp` — the
#'   certification frontier. `frontier_at` is the budget at which the
#'   frontier is priced and `n_cur` the real-valued frontier
#'   \eqn{n^{\star}(\delta;\alpha)}{n*(delta; alpha)} there, while `n_star`
#'   is the smallest certifying integer sample size
#'   \eqn{\lfloor n^{\star}(\delta;\alpha)\rfloor + 1}{floor(n*(delta; alpha)) + 1}
#'   at the budget raised by `jump`. `delta_n` is that sample size net of
#'   the realized `n`, and `cost_per_pp` the implied cost of raising the
#'   certified budget by one percentage point, at `cost_per_unit` per
#'   sampled unit.
#' * `label`, `direction`, `n`, `point`, `tau_star` — the estimand label,
#'   the resolved direction, the sample size \eqn{n}{n}, the baseline
#'   estimate of the estimand at \eqn{\delta = 0}{delta = 0}, and the
#'   reference value \eqn{\tau_{\star}}{tau_star}.
#'
#' The evaluation budget for `eta`, `se`, and `varsigma` is the `delta`
#' argument when supplied; when `delta = NULL` (the default) it is the
#' plug-in breakdown budget, replaced by the certified breakdown budget
#' when the plug-in breakdown is censored, as in the paper. The frontier is
#' priced at the certified breakdown (default) or at the supplied `delta`.
#'
#' Objects without inference (the recentered-IV and counterfactual
#' applications carry none by design) degrade gracefully: the plug-in
#' measures are reported, the certified and frontier measures are `NA`, and
#' the `notes` field explains why.
#'
#' @param object A `tvbounds` object returned by [tvbounds_attrition()],
#'   [tvbounds_counterfactual()], or [tvbounds_riv()].
#' @param delta Optional evaluation budget \eqn{\delta}{delta} (a single
#'   nonnegative number). `NULL` (default) evaluates the measures at the
#'   plug-in breakdown budget
#'   \eqn{\delta_{b}(\tau_{\star})}{delta_b(tau_star)}, with the censoring
#'   fallback described in Details.
#' @param tau_star Reference value \eqn{\tau_{\star}}{tau_star} of the
#'   estimand against which robustness is judged (default `0`).
#' @param level Confidence level \eqn{1 - \alpha}{1 - alpha} for the
#'   two-sided critical value \eqn{z_{1-\alpha/2}}{z_{1-alpha/2}} used by
#'   the normal-approximation breakdown budget and the certification
#'   frontier (default `0.95`, that is, `qnorm(0.975)`). The certified
#'   breakdown itself is read off the band stored in the object.
#' @param cost_per_unit Marginal cost of one additional sampled unit, used
#'   for the cost equivalent of the certification frontier (default `50`,
#'   the paper's USD benchmark from the 3ie closed-grant portfolio).
#' @param jump Increase in the certified budget that the frontier prices,
#'   so that `n_star` is evaluated at
#'   \eqn{\delta + \mathrm{jump}}{delta + jump} (default `0.05`).
#' @param direction One of `"auto"`, `"lower"`, `"upper"`; see Details.
#' @param ... For the `summary()` method: further arguments forwarded to
#'   `tvbounds_summary()`.
#' @return An object of class `"tvbounds_summary"`: a list with a one-row
#'   data frame `measures` (columns `label`, `direction`, `n`, `point`,
#'   `tau_star`, `delta_b`, `censored`, `delta_b_ci`, `censored_ci`,
#'   `delta_b_ci_norm`, `delta_eval`, `eta`, `se`, `varsigma`,
#'   `varsigma_sc`, `frontier_at`, `n_cur`, `n_star`, `delta_n`,
#'   `cost_per_pp`) and metadata fields (`application`, `estimand_label`,
#'   `neighborhood`, `divergence`, `direction`, `tau_star`, `delta_eval`,
#'   `level`, `band_level`, `zc`, `cost_per_unit`, `jump`, `has_se`,
#'   `has_ci`, `notes`, `call`). Details gives the symbol of the paper each
#'   column of `measures` corresponds to. Printed compactly by
#'   [print.tvbounds_summary()].
#' @references
#' Palomba, F. (2026). "Sensitivity Analysis in Population Shares."
#' Working paper.
#' @seealso [tvbounds_plot()] to display the bounds; `summary()` dispatches
#'   here for `tvbounds` objects.
#' @examples
#' set.seed(123)
#' n <- 400
#' d <- rbinom(n, 1, 0.5)
#' s <- rbinom(n, 1, plogis(0.5 + 0.5 * d))
#' y <- ifelse(s == 1, 0.3 * d + rnorm(n), NA)
#' dat <- data.frame(y = y, d = d, s = s)
#' fit <- tvbounds_attrition(dat, outcome = "y", treatment = "d",
#'                           response = "s", delta = seq(0, 1, by = 0.1),
#'                           B = 100, seed = 1)
#' tvbounds_summary(fit)
#'
#' # Evaluate the measures at a chosen budget instead of the breakdown:
#' tvbounds_summary(fit, delta = 0.2)
#' @export
tvbounds_summary <- function(object, delta = NULL, tau_star = 0,
                             level = 0.95, cost_per_unit = 50, jump = 0.05,
                             direction = c("auto", "lower", "upper")) {
  if (!is_tvbounds(object)) {
    stop("`object` must be a `tvbounds` object.", call. = FALSE)
  }
  direction <- match.arg(direction)
  if (!is.null(delta) &&
      (!is.numeric(delta) || length(delta) != 1L || !is.finite(delta) ||
       delta < 0)) {
    stop("`delta` must be NULL or a single nonnegative finite number.",
         call. = FALSE)
  }
  if (!is.numeric(tau_star) || length(tau_star) != 1L ||
      !is.finite(tau_star)) {
    stop("`tau_star` must be a finite numeric scalar.", call. = FALSE)
  }
  if (!is.numeric(level) || length(level) != 1L || is.na(level) ||
      level <= 0 || level >= 1) {
    stop("`level` must be a number strictly between 0 and 1.", call. = FALSE)
  }
  if (!is.numeric(cost_per_unit) || length(cost_per_unit) != 1L ||
      !is.finite(cost_per_unit) || cost_per_unit < 0) {
    stop("`cost_per_unit` must be a single nonnegative number.",
         call. = FALSE)
  }
  if (!is.numeric(jump) || length(jump) != 1L || !is.finite(jump) ||
      jump <= 0) {
    stop("`jump` must be a single positive number.", call. = FALSE)
  }

  notes <- character(0)
  dir <- .tvb_resolve_direction(direction, object$point, tau_star)
  sp <- .tvb_signed_path(object, dir, tau_star)
  if (length(sp$d) < 2L) {
    stop("`object$bounds` must contain at least two non-missing budget ",
         "values on the ", dir, " bound path.", call. = FALSE)
  }

  ## Two-sided critical value; the empirical applications report the
  ## certified budget off the outer band, so z_{1-alpha/2} applies.
  zc <- stats::qnorm(1 - (1 - level) / 2)
  if (!is.na(object$level) && abs(object$level - level) > 1e-12) {
    notes <- c(notes, sprintf(
      paste0("`level` (%.3f) differs from the level of the band stored in ",
             "the object (%.3f); the certified breakdown is read off the ",
             "stored band."), level, object$level))
  }

  has_se <- !is.null(sp$se) && any(!is.na(sp$se))
  has_ci <- !is.null(sp$ci) && any(!is.na(sp$ci))

  eta_path <- .tvb_eta_fun(sp$d, sp$path)
  db_raw   <- .tvb_cross0(sp$d, sp$path)
  censored <- is.na(db_raw)
  ## A censored breakdown budget is reported at the right endpoint of the
  ## budget grid (1 for the total-variation and contamination
  ## neighborhoods), the value the empty-set convention assigns it, rather
  ## than as "> 1": no larger budget is on the grid.
  db <- if (censored) sp$right else db_raw

  ## The certified budget is read off the SAME band that the figures plot,
  ## so that tables, figures, and text agree on one number. The normal
  ## floor is retained only as a diagnostic: the two estimate the same
  ## population object and differ when the bootstrap distribution of the
  ## bound is asymmetric.
  db_ci_raw   <- if (has_ci) .tvb_cross0(sp$d, sp$ci) else NA_real_
  censored_ci <- has_ci && is.na(db_ci_raw)
  db_ci <- if (!has_ci) NA_real_ else if (censored_ci) sp$right else db_ci_raw
  db_ci_norm <- if (has_se) {
    .tvb_cross0(sp$d, sp$path - zc * sp$se)
  } else {
    NA_real_
  }

  ## Evaluation budget. Default: the plug-in breakdown; when that is
  ## censored, the shadow price and the robustness standard error are
  ## anchored at the certified budget instead, as in the paper. The
  ## frontier is always priced at the certified budget unless the user
  ## supplies an explicit `delta`.
  if (!is.null(delta)) {
    anchor <- delta
    frontier_at <- delta
    if (delta > sp$right) {
      warning("`delta` exceeds the largest budget on the grid; the ",
              "measures use flat extrapolation beyond the grid.",
              call. = FALSE)
    }
  } else {
    anchor <- if (!censored) db_raw else db_ci_raw
    frontier_at <- db_ci_raw
    if (censored && has_ci && !censored_ci) {
      notes <- c(notes, paste0(
        "the plug-in breakdown is censored at the right endpoint of the ",
        "budget grid; the shadow price and the robustness standard error ",
        "are anchored ",
        "at the certified breakdown instead."))
    }
    if (is.na(anchor)) {
      notes <- c(notes, paste0(
        "no interior breakdown budget is available to anchor the shadow ",
        "price; pass an explicit `delta` to evaluate the measures at a ",
        "chosen budget."))
    }
  }

  nn <- suppressWarnings(as.numeric(object$n))
  if (length(nn) != 1L) nn <- NA_real_

  eta_b <- .tvb_lin(sp$d, eta_path, anchor)
  se_b  <- if (has_se) .tvb_lin(sp$d, sp$se, anchor) else NA_real_
  varsig    <- se_b * sqrt(nn) / eta_b
  varsig_sc <- se_b / eta_b

  ## Certification frontier. n_cur is the frontier at the pricing budget;
  ## n_star the smallest integer sample size strictly above the frontier at
  ## the pricing budget raised by `jump` (certification requires an integer
  ## n strictly above the real-valued frontier: floor(n) + 1). delta_n is
  ## the additional sample the study would actually have to collect: the
  ## frontier at the raised budget against the realized n.
  n_cur <- if (has_se && !is.na(nn)) {
    .tvb_nstar(sp$d, sp$path, sp$se, nn, frontier_at, zc)
  } else {
    NA_real_
  }
  n_jmp <- if (has_se && !is.na(nn) && !is.na(frontier_at)) {
    .tvb_nstar(sp$d, sp$path, sp$se, nn, frontier_at + jump, zc)
  } else {
    NA_real_
  }
  n_star  <- if (is.na(n_jmp)) NA_real_ else floor(n_jmp) + 1
  delta_n <- n_star - nn
  cost_pp <- delta_n * cost_per_unit / (100 * jump)

  if (!has_ci) {
    notes <- c(notes, sprintf(
      paste0("no confidence band is attached to the %s bound path, so the ",
             "certified breakdown and the certification frontier are ",
             "reported as NA%s."),
      dir,
      if (object$application %in% c("riv", "counterfactual")) {
        " (this application carries no inference by design)"
      } else ""))
  }
  if (!has_se) {
    notes <- c(notes, sprintf(
      paste0("no bootstrap standard errors are attached to the %s bound ",
             "path, so the normal-floor diagnostic and the robustness ",
             "standard error ",
             "are reported as NA."), dir))
  }

  measures <- data.frame(
    label           = object$estimand_label %||% "estimand",
    direction       = dir,
    n               = nn,
    point           = as.numeric(object$point),
    tau_star        = tau_star,
    delta_b         = db,
    censored        = censored,
    delta_b_ci      = db_ci,
    censored_ci     = censored_ci,
    delta_b_ci_norm = db_ci_norm,
    delta_eval      = as.numeric(anchor),
    eta             = eta_b,
    se              = se_b,
    varsigma        = varsig,
    varsigma_sc     = varsig_sc,
    frontier_at     = as.numeric(frontier_at),
    n_cur           = n_cur,
    n_star          = n_star,
    delta_n         = delta_n,
    cost_per_pp     = cost_pp,
    stringsAsFactors = FALSE
  )

  structure(
    list(
      measures       = measures,
      application    = object$application,
      estimand_label = object$estimand_label,
      neighborhood   = object$neighborhood,
      divergence     = object$divergence,
      direction      = dir,
      tau_star       = tau_star,
      delta_eval     = as.numeric(anchor),
      level          = level,
      band_level     = object$level,
      zc             = zc,
      cost_per_unit  = cost_per_unit,
      jump           = jump,
      has_se         = has_se,
      has_ci         = has_ci,
      notes          = notes,
      call           = match.call()
    ),
    class = "tvbounds_summary"
  )
}
