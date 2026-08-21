# Internal constructor and validator for the objects returned by the
# tvbounds estimators. Owned by the coordinator: implementers of the
# estimation functions build their return values through new_tvbounds()
# and must not modify this file.

`%||%` <- function(x, y) if (is.null(x)) y else x

#' Construct a tvbounds result object
#'
#' Internal constructor shared by [tvbounds_attrition()],
#' [tvbounds_counterfactual()], and [tvbounds_riv()]. Not exported.
#'
#' @param application One of `"attrition"`, `"counterfactual"`, `"riv"`.
#' @param bounds A data frame with required columns `delta`, `lower`,
#'   `upper` and optional columns `lower_se`, `upper_se`, `ci_lower`,
#'   `ci_upper` (outer confidence-band endpoints at `level`), one row per
#'   budget value.
#' @param point Scalar baseline (`delta = 0`) estimate of the estimand.
#' @param n Sample size used in estimation.
#' @param neighborhood `"tv"` or `"contamination"` for the attrition and
#'   recentered-IV applications; `NULL` for the counterfactual application.
#' @param divergence Divergence keyword (counterfactual application only);
#'   `NULL` otherwise.
#' @param level Confidence level of the reported bands (`NA` when the
#'   object carries no inference).
#' @param B Number of bootstrap replications (`NA` when the object carries
#'   no inference).
#' @param estimand_label Short label used by the plot and summary methods.
#' @param call The `match.call()` of the user-facing constructor.
#' @param details Application-specific list.
#' @param subclass Character vector of subclasses prepended to `"tvbounds"`.
#'
#' @return An object of class `c(subclass, "tvbounds")`.
#' @keywords internal
#' @noRd
new_tvbounds <- function(application,
                         bounds,
                         point = NA_real_,
                         n = NA_integer_,
                         neighborhood = NULL,
                         divergence = NULL,
                         level = NA_real_,
                         B = NA_integer_,
                         estimand_label = "estimand",
                         call = NULL,
                         details = list(),
                         subclass = paste0("tvbounds_", application)) {
  stopifnot(
    is.character(application), length(application) == 1L,
    application %in% c("attrition", "counterfactual", "riv"),
    is.data.frame(bounds),
    all(c("delta", "lower", "upper") %in% names(bounds)),
    is.list(details)
  )
  if (!is.null(neighborhood) &&
      !neighborhood %in% c("tv", "contamination")) {
    stop("`neighborhood` must be \"tv\" or \"contamination\".", call. = FALSE)
  }
  if (anyDuplicated(bounds$delta)) {
    stop("`bounds$delta` must not contain duplicated budget values.",
         call. = FALSE)
  }
  if (any(bounds$delta < 0, na.rm = TRUE)) {
    stop("`bounds$delta` must be nonnegative.", call. = FALSE)
  }
  bad <- which(bounds$upper - bounds$lower < -1e-8)
  if (length(bad) > 0L) {
    stop(sprintf(
      "Lower bound exceeds upper bound at delta = %s.",
      paste(signif(bounds$delta[bad], 4), collapse = ", ")
    ), call. = FALSE)
  }
  bounds <- bounds[order(bounds$delta), , drop = FALSE]
  rownames(bounds) <- NULL
  structure(
    list(
      application    = application,
      bounds         = bounds,
      point          = point,
      n              = n,
      neighborhood   = neighborhood,
      divergence     = divergence,
      level          = level,
      B              = B,
      estimand_label = estimand_label,
      call           = call,
      details        = details
    ),
    class = c(subclass, "tvbounds")
  )
}

#' Test whether an object is a tvbounds result
#' @param x Any object.
#' @return `TRUE` if `x` inherits from class `"tvbounds"`.
#' @keywords internal
#' @noRd
is_tvbounds <- function(x) inherits(x, "tvbounds")
