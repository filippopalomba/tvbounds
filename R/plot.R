# ggplot2 display of tvbounds objects. The style follows the paper's
# figure scripts (deep blue #1F4E79, plain theme_bw with no grid), but is
# self-contained: pure ggplot2, no tikzDevice.

#' Plot sensitivity bounds against the budget
#'
#' Displays the bounds stored in a `tvbounds` object as functions of the
#' budget \eqn{\delta}{delta} of the total-variation (or contamination, or
#' divergence) neighborhood, supplied through the `delta` column of
#' `x$bounds` and drawn on the horizontal axis. The region between the
#' lower bound path \eqn{\underline{\tau}(\delta)}{tau_lower(delta)} and
#' the upper bound path \eqn{\overline{\tau}(\delta)}{tau_upper(delta)} is
#' shaded, the outer confidence band (when the object carries one) is drawn
#' as a lighter ribbon delimited by dashed lines, a dashed horizontal line
#' marks the reference value \eqn{\tau_{\star}}{tau_star}, a point marks
#' the baseline estimate at \eqn{\delta = 0}{delta = 0}, and a dotted
#' vertical line marks the plug-in breakdown budget
#' \eqn{\widehat{\delta}_{b}}{delta_b_hat} when it is interior to the
#' budget grid.
#'
#' @details
#' The breakdown budget drawn by `breakdown = TRUE` is the plug-in
#' breakdown budget of [tvbounds_summary()], the estimated counterpart of
#' \eqn{\delta_{b}(\tau_{\star}) = \inf\{\delta : \underline{\tau}(\delta) \le \tau_{\star} \le \overline{\tau}(\delta)\}}{delta_b(tau_star) = inf{delta : tau_lower(delta) <= tau_star <= tau_upper(delta)}}:
#' the first budget at which the bound path adjacent to
#' \eqn{\tau_{\star}}{tau_star} reaches it — the lower path when the
#' baseline point estimate exceeds `tau_star`, the upper path otherwise.
#' The line is annotated with the value of
#' \eqn{\widehat{\delta}_{b}}{delta_b_hat}. No line is drawn when the
#' breakdown budget is censored at the right endpoint of the grid, when it
#' sits at the left endpoint, or when the baseline point estimate is
#' missing.
#'
#' Rows of `x$bounds` with missing bound values (e.g. censored or
#' infeasible budgets) are omitted from the corresponding layer. For the
#' counterfactual application, whose divergence budgets may exceed one,
#' `log_x = TRUE` switches to a logarithmic budget axis.
#'
#' @param x A `tvbounds` object returned by [tvbounds_attrition()],
#'   [tvbounds_counterfactual()], or [tvbounds_riv()].
#' @param bands Logical; draw the outer confidence band when the object
#'   carries one (columns `ci_lower`/`ci_upper` of `x$bounds`). Sides
#'   without a band are drawn without one. Default `TRUE`.
#' @param baseline Logical; mark the baseline point estimate, that is, the
#'   value of the estimand under \eqn{P_{*}}{P_*}, at
#'   \eqn{\delta = 0}{delta = 0}. Ignored when `log_x = TRUE`, since
#'   \eqn{\delta = 0}{delta = 0} cannot be placed on a logarithmic axis.
#'   Default `TRUE`.
#' @param breakdown Logical; draw a dotted vertical line, with a label, at
#'   the plug-in breakdown budget
#'   \eqn{\widehat{\delta}_{b}}{delta_b_hat} when it is interior to the
#'   budget grid. Default `TRUE`.
#' @param tau_star Reference value \eqn{\tau_{\star}}{tau_star} of the
#'   estimand against which robustness is judged; drawn as a dashed
#'   horizontal line (default `0`).
#' @param color Colour of the bounds, ribbons, and breakdown mark (default
#'   `"#1F4E79"`, the paper's blue).
#' @param xlab,ylab,title Axis labels and plot title. `NULL` (the default)
#'   uses `expression(delta)` for the horizontal axis, the object's
#'   `estimand_label` for the vertical axis, and no title.
#' @param log_x Logical; use a logarithmic budget axis. Rows with
#'   `delta <= 0` are dropped with a warning. Default `FALSE`.
#' @param ... For `tvbounds_plot()`: currently unused, accepted for
#'   compatibility with the [plot()][base::plot] generic. For the `plot()`
#'   method: further arguments forwarded to `tvbounds_plot()`.
#' @return A ggplot object, which prints to the active graphics device and
#'   can be modified further with ggplot2 layers.
#' @references
#' Palomba, F. (2026). "Sensitivity Analysis in Population Shares."
#' Working paper.
#' @seealso [tvbounds_summary()] for the numerical summary measures;
#'   `plot()` dispatches here for `tvbounds` objects.
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
#' tvbounds_plot(fit)
#'
#' # Bounds only, no confidence band and no breakdown mark:
#' tvbounds_plot(fit, bands = FALSE, breakdown = FALSE)
#' @export
tvbounds_plot <- function(x, bands = TRUE, baseline = TRUE, breakdown = TRUE,
                          tau_star = 0, color = "#1F4E79", xlab = NULL,
                          ylab = NULL, title = NULL, log_x = FALSE, ...) {
  if (!is_tvbounds(x)) {
    stop("`x` must be a `tvbounds` object.", call. = FALSE)
  }
  for (nm in c("bands", "baseline", "breakdown", "log_x")) {
    v <- get(nm, inherits = FALSE)
    if (!is.logical(v) || length(v) != 1L || is.na(v)) {
      stop(sprintf("`%s` must be TRUE or FALSE.", nm), call. = FALSE)
    }
  }
  if (!is.numeric(tau_star) || length(tau_star) != 1L ||
      !is.finite(tau_star)) {
    stop("`tau_star` must be a finite numeric scalar.", call. = FALSE)
  }
  if (!is.character(color) || length(color) != 1L || is.na(color)) {
    stop("`color` must be a single colour string.", call. = FALSE)
  }

  b <- x$bounds
  b <- b[!is.na(b$delta), , drop = FALSE]
  if (log_x) {
    npos <- b$delta <= 0
    if (any(npos)) {
      warning(sprintf(
        "`log_x = TRUE`: dropped %d row(s) with `delta` <= 0, which cannot be placed on a logarithmic axis.",
        sum(npos)), call. = FALSE)
      b <- b[!npos, , drop = FALSE]
    }
  }
  if (nrow(b) < 2L) {
    stop("`x$bounds` must contain at least two budget values to plot.",
         call. = FALSE)
  }
  has_cil <- "ci_lower" %in% names(b) && any(!is.na(b$ci_lower))
  has_ciu <- "ci_upper" %in% names(b) && any(!is.na(b$ci_upper))

  p <- ggplot2::ggplot() +
    ggplot2::geom_hline(yintercept = tau_star, linetype = "dashed",
                        linewidth = 0.7, color = "gray50")

  ## Following the paper's figures, only the OUTER confidence limit of each
  ## bound is drawn -- the lower limit for the lower bound and the upper
  ## limit for the upper bound -- so the dashed pair traces the outward
  ## envelope of the identified set; the band between each estimate and its
  ## outer limit is shaded more lightly than the identified region.
  if (bands && has_cil) {
    cl <- b[!is.na(b$ci_lower) & !is.na(b$lower), , drop = FALSE]
    if (nrow(cl) > 1L) {
      p <- p +
        ggplot2::geom_ribbon(
          data = cl,
          ggplot2::aes(x = delta, ymin = ci_lower, ymax = lower),
          fill = color, alpha = 0.12) +
        ggplot2::geom_line(
          data = cl, ggplot2::aes(x = delta, y = ci_lower),
          color = color, linetype = "dashed", linewidth = 0.8)
    }
  }
  if (bands && has_ciu) {
    cu <- b[!is.na(b$ci_upper) & !is.na(b$upper), , drop = FALSE]
    if (nrow(cu) > 1L) {
      p <- p +
        ggplot2::geom_ribbon(
          data = cu,
          ggplot2::aes(x = delta, ymin = upper, ymax = ci_upper),
          fill = color, alpha = 0.12) +
        ggplot2::geom_line(
          data = cu, ggplot2::aes(x = delta, y = ci_upper),
          color = color, linetype = "dashed", linewidth = 0.8)
    }
  }

  core <- b[!is.na(b$lower) & !is.na(b$upper), , drop = FALSE]
  if (nrow(core) > 1L) {
    p <- p + ggplot2::geom_ribbon(
      data = core,
      ggplot2::aes(x = delta, ymin = lower, ymax = upper),
      fill = color, alpha = 0.25)
  }
  lo <- b[!is.na(b$lower), , drop = FALSE]
  if (nrow(lo) > 1L) {
    p <- p + ggplot2::geom_line(
      data = lo, ggplot2::aes(x = delta, y = lower),
      color = color, linewidth = 1.2)
  }
  up <- b[!is.na(b$upper), , drop = FALSE]
  if (nrow(up) > 1L) {
    p <- p + ggplot2::geom_line(
      data = up, ggplot2::aes(x = delta, y = upper),
      color = color, linewidth = 1.2)
  }

  ## Plug-in breakdown budget, drawn only when interior to the grid. The
  ## direction follows the summary method: the lower path when the baseline
  ## point estimate exceeds tau_star, the upper path otherwise.
  if (breakdown && !is.null(x$point) && length(x$point) == 1L &&
      !is.na(x$point)) {
    dirn <- if (x$point > tau_star) "lower" else "upper"
    sp <- .tvb_signed_path(x, dirn, tau_star)
    if (length(sp$d) >= 2L) {
      db <- .tvb_cross0(sp$d, sp$path)
      if (!is.na(db) && db > sp$d[1L] && db < sp$right) {
        hj <- if (db > (sp$d[1L] + sp$right) / 2) 1.15 else -0.15
        p <- p +
          ggplot2::geom_vline(xintercept = db, linetype = "dotted",
                              linewidth = 0.9, color = color) +
          ggplot2::annotate("text", x = db, y = Inf,
                            label = paste0("hat(delta)[b] == ",
                                           signif(db, 3)),
                            parse = TRUE, hjust = hj, vjust = 1.8,
                            color = color, size = 3.5)
      }
    }
  }

  if (baseline && !log_x && !is.null(x$point) && length(x$point) == 1L &&
      !is.na(x$point)) {
    p <- p + ggplot2::annotate("point", x = 0, y = x$point,
                               color = color, size = 2.4)
  }

  if (log_x) p <- p + ggplot2::scale_x_log10()
  if (is.null(xlab)) xlab <- expression(delta)
  if (is.null(ylab)) ylab <- x$estimand_label %||% "estimand"

  p +
    ggplot2::labs(x = xlab, y = ylab, title = title) +
    ggplot2::theme_bw(base_size = 12) +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      legend.position = "none",
      plot.title = ggplot2::element_text(face = "bold", hjust = 0.5))
}
