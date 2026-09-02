# S3 methods for tvbounds objects: plot(), summary(), and the two print
# methods, plus small internal formatting helpers shared by the printers.

#' Format a scalar number for printing ("---" when missing)
#' @keywords internal
#' @noRd
.tvb_fmt <- function(v, digits = 3) {
  if (is.null(v) || length(v) != 1L || is.na(v)) return("---")
  if (is.infinite(v)) return(if (v > 0) "Inf" else "-Inf")
  format(signif(v, digits))
}

#' Format a scalar as an integer with thousands separators
#' @keywords internal
#' @noRd
.tvb_fmt_int <- function(v, signed = FALSE) {
  if (is.null(v) || length(v) != 1L || is.na(v)) return("---")
  if (is.infinite(v)) return(if (v > 0) "Inf" else "-Inf")
  out <- formatC(round(abs(v)), big.mark = ",", format = "d")
  sgn <- if (v < 0) "-" else if (signed) "+" else ""
  paste0(sgn, out)
}

#' Human-readable label for the neighborhood of a tvbounds object
#' @keywords internal
#' @noRd
.tvb_neigh_label <- function(neighborhood, divergence) {
  if (!is.null(neighborhood)) {
    switch(neighborhood,
           tv = "total-variation neighborhood",
           contamination = "contamination neighborhood",
           neighborhood)
  } else if (!is.null(divergence)) {
    paste0(divergence, " divergence neighborhood")
  } else {
    "unspecified neighborhood"
  }
}

#' @rdname tvbounds_plot
#' @export
plot.tvbounds <- function(x, ...) {
  tvbounds_plot(x, ...)
}

#' @rdname tvbounds_summary
#' @export
summary.tvbounds <- function(object, ...) {
  tvbounds_summary(object, ...)
}

#' Print a tvbounds object
#'
#' Compact display of a `tvbounds` object: the application, the
#' neighborhood over which the sensitivity bounds were computed, the sample
#' size \eqn{n}{n}, the baseline point estimate — the value of the estimand
#' under the baseline distribution \eqn{P_{*}}{P_*}, reported at
#' \eqn{\delta = 0}{delta = 0} — the grid of budgets \eqn{\delta}{delta} on
#' which the bound paths \eqn{\underline{\tau}(\delta)}{tau_lower(delta)}
#' and \eqn{\overline{\tau}(\delta)}{tau_upper(delta)} were evaluated, and
#' whether inference is attached.
#'
#' @param x A `tvbounds` object.
#' @param digits Number of significant digits (default `3`).
#' @param ... Further arguments; ignored.
#' @return `x`, invisibly.
#' @references
#' Palomba, F. (2026). "Sensitivity Analysis in Population Shares."
#' Working paper.
#' @examples
#' set.seed(1)
#' n <- 200
#' d <- rbinom(n, 1, 0.5)
#' s <- rbinom(n, 1, plogis(0.4 + 0.4 * d))
#' y <- ifelse(s == 1, 0.3 * d + rnorm(n), NA)
#' fit <- tvbounds_attrition(data.frame(y = y, d = d, s = s),
#'                           outcome = "y", treatment = "d", response = "s",
#'                           delta = seq(0, 1, by = 0.1), bootstrap = FALSE)
#' print(fit)
#' @export
print.tvbounds <- function(x, digits = 3, ...) {
  d <- x$bounds$delta
  cat(sprintf("<tvbounds> %s bounds under a %s\n",
              x$application,
              .tvb_neigh_label(x$neighborhood, x$divergence)))
  cat(sprintf("  estimand: %s; baseline point estimate (delta = 0): %s; n = %s\n",
              x$estimand_label %||% "estimand",
              .tvb_fmt(x$point, digits), .tvb_fmt_int(x$n)))
  cat(sprintf("  budget grid: %d values of delta in [%s, %s]\n",
              length(d),
              .tvb_fmt(min(d, na.rm = TRUE), digits),
              .tvb_fmt(max(d, na.rm = TRUE), digits)))
  if (!is.na(x$level)) {
    cat(sprintf("  inference: %s%% bootstrap confidence bands (B = %s)\n",
                format(100 * x$level), .tvb_fmt_int(x$B)))
  } else {
    cat("  inference: none attached\n")
  }
  cat("  Use summary() for breakdown and price-of-robustness measures; plot() to display.\n")
  invisible(x)
}

#' Print a tvbounds summary
#'
#' Compact display of the summary measures computed by
#' [tvbounds_summary()]: the plug-in and certified breakdown budgets
#' \eqn{\delta_{b}(\tau_{\star})}{delta_b(tau_star)} and
#' \eqn{\widehat{\delta}_{b}^{\,\mathsf{C}}(\alpha)}{delta_b^C(alpha)}, the
#' shadow price of robustness
#' \eqn{\underline{\eta}(\delta) = -\underline{\tau}'(\delta)}{eta(delta) = -tau_lower'(delta)},
#' the robustness standard error
#' \eqn{\varsigma_{b} = \sigma(\delta_{b})/\underline{\eta}(\delta_{b})}{varsigma_b = sigma(delta_b) / eta(delta_b)},
#' and the certification frontier
#' \eqn{n^{\star}(\delta;\alpha)}{n*(delta; alpha)}, followed by any notes
#' on measures that could not be computed. The reference value of the
#' estimand, \eqn{\tau_{\star}}{tau_star}, is shown as `tau_star`,
#' matching the argument name.
#'
#' @param x A `"tvbounds_summary"` object.
#' @param digits Number of significant digits (default `3`).
#' @param ... Further arguments; ignored.
#' @return `x`, invisibly.
#' @references
#' Palomba, F. (2026). "Sensitivity Analysis in Population Shares."
#' Working paper.
#' @examples
#' set.seed(1)
#' n <- 200
#' d <- rbinom(n, 1, 0.5)
#' s <- rbinom(n, 1, plogis(0.4 + 0.4 * d))
#' y <- ifelse(s == 1, 0.3 * d + rnorm(n), NA)
#' fit <- tvbounds_attrition(data.frame(y = y, d = d, s = s),
#'                           outcome = "y", treatment = "d", response = "s",
#'                           delta = seq(0, 1, by = 0.1), bootstrap = FALSE)
#' # Without bootstrap draws the certified measures degrade to NA, with a
#' # note explaining why:
#' summary(fit)
#' @export
print.tvbounds_summary <- function(x, digits = 3, ...) {
  m <- x$measures
  cat(sprintf("<tvbounds summary> %s (%s application, %s)\n",
              x$estimand_label %||% "estimand", x$application,
              .tvb_neigh_label(x$neighborhood, x$divergence)))
  cat(sprintf("  direction: %s bound path relative to tau_star = %s (baseline point = %s, n = %s)\n",
              m$direction, .tvb_fmt(m$tau_star, digits),
              .tvb_fmt(m$point, digits), .tvb_fmt_int(m$n)))
  bd_txt <- .tvb_fmt(m$delta_b, digits)
  if (isTRUE(m$censored)) bd_txt <- paste0(bd_txt, " (censored)")
  ci_txt <- .tvb_fmt(m$delta_b_ci, digits)
  if (isTRUE(m$censored_ci)) ci_txt <- paste0(ci_txt, " (censored)")
  cat(sprintf("  breakdown budget: plug-in = %s; certified = %s; normal floor = %s\n",
              bd_txt, ci_txt, .tvb_fmt(m$delta_b_ci_norm, digits)))
  cat(sprintf("  at delta = %s: shadow price eta = %s; robustness SE varsigma = %s (scale-free %s)\n",
              .tvb_fmt(m$delta_eval, digits), .tvb_fmt(m$eta, digits),
              .tvb_fmt(m$varsigma, digits),
              .tvb_fmt(m$varsigma_sc, digits)))
  cat(sprintf("  certification frontier: n* = %s at budget %s + jump %s (Delta n = %s, cost per pp = %s)\n",
              .tvb_fmt_int(m$n_star), .tvb_fmt(m$frontier_at, digits),
              .tvb_fmt(x$jump, digits),
              .tvb_fmt_int(m$delta_n, signed = TRUE),
              .tvb_fmt(m$cost_per_pp, max(digits, 4))))
  if (length(x$notes)) {
    for (nt in x$notes) cat("  note:", nt, "\n")
  }
  invisible(x)
}
