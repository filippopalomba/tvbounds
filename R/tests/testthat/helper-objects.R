# Shared fixtures for the summary and plot tests. All objects are built
# through new_tvbounds() so the constructor contract is exercised, and all
# bound paths are linear (or quadratic) in the budget so that breakdown
# budgets, shadow prices, and frontier values are analytic:
# linear interpolation and central differences are exact for them.

# Linear paths with full inference attached (attrition-like).
#   lower(delta) = point - slope * delta   -> plug-in breakdown point/slope
#   ci_lower     = lower - ci_gap          -> certified (point - ci_gap)/slope
#   eta(delta)   = slope, se constant      -> varsigma = se * sqrt(n)/slope
tvb_fixture_linear <- function(point = 0.5, slope = 1, se = 0.1,
                               ci_gap = 0.15, n = 100L,
                               delta = seq(0, 1, by = 0.01)) {
  lower <- point - slope * delta
  upper <- point + slope * delta
  bounds <- data.frame(delta = delta, lower = lower, upper = upper,
                       lower_se = se, upper_se = se,
                       ci_lower = lower - ci_gap, ci_upper = upper + ci_gap)
  new_tvbounds("attrition", bounds, point = point, n = n,
               neighborhood = "tv", level = 0.95, B = 500L,
               estimand_label = "treatment effect")
}

# Mirror image of the linear fixture: baseline point below tau_star = 0, so
# direction = "auto" must pick the UPPER path. The signed path tau* - upper
# coincides with the signed lower path of tvb_fixture_linear, hence all
# summary measures agree between the two fixtures.
tvb_fixture_linear_upper <- function(point = -0.5, slope = 1, se = 0.1,
                                     ci_gap = 0.15, n = 100L,
                                     delta = seq(0, 1, by = 0.01)) {
  lower <- point - slope * delta
  upper <- point + slope * delta
  bounds <- data.frame(delta = delta, lower = lower, upper = upper,
                       lower_se = se, upper_se = se,
                       ci_lower = lower - ci_gap, ci_upper = upper + ci_gap)
  new_tvbounds("attrition", bounds, point = point, n = n,
               neighborhood = "tv", level = 0.95, B = 500L,
               estimand_label = "treatment effect")
}

# Censored plug-in breakdown: lower = 0.5 - 0.3 * delta stays positive on
# [0, 1] (value 0.2 at delta = 1). The certified breakdown off
# ci_lower = lower - 0.25 crosses at 0.25 / 0.3 = 5/6, which anchors the
# shadow price (eta = 0.3) and the robustness standard error.
tvb_fixture_censored <- function(n = 100L) {
  delta <- seq(0, 1, by = 0.01)
  lower <- 0.5 - 0.3 * delta
  upper <- 0.5 + 0.3 * delta
  bounds <- data.frame(delta = delta, lower = lower, upper = upper,
                       lower_se = 0.1, upper_se = 0.1,
                       ci_lower = lower - 0.25, ci_upper = upper + 0.25)
  new_tvbounds("attrition", bounds, point = 0.5, n = n,
               neighborhood = "tv", level = 0.95, B = 500L,
               estimand_label = "treatment effect")
}

# No inference attached (riv-like): no SE and no CI columns, level/B = NA.
# Plug-in breakdown of the lower path at 0.3.
tvb_fixture_noinf <- function() {
  delta <- seq(0, 1, by = 0.02)
  bounds <- data.frame(delta = delta,
                       lower = 0.3 - delta,
                       upper = 0.3 + 0.5 * delta)
  new_tvbounds("riv", bounds, point = 0.3, n = 250L,
               neighborhood = "tv", level = NA_real_, B = NA_integer_,
               estimand_label = "IV coefficient")
}

# Counterfactual-like object on an uneven budget grid exceeding 1 (KL
# divergence budgets are not restricted to [0, 1]); no inference. With
# slope 0.4 the lower path 1 - 0.4 * delta crosses zero at 2.5; with slope
# 0.1 it never crosses on the grid, so the breakdown is censored at the
# right endpoint 5.
tvb_fixture_counterfactual <- function(slope = 0.4) {
  delta <- c(0, 0.1, 0.5, 1, 2, 5)
  bounds <- data.frame(delta = delta,
                       lower = 1 - slope * delta,
                       upper = 1 + slope * delta)
  new_tvbounds("counterfactual", bounds, point = 1, n = 1000L,
               neighborhood = NULL, divergence = "KL",
               level = NA_real_, B = NA_integer_,
               estimand_label = "counterfactual mean")
}

# Quadratic lower path 0.5 - delta^2, so the shadow price eta(delta) =
# 2 * delta varies along the grid: central differences are exact for
# quadratics on a uniform grid, making eta at a user-supplied evaluation
# budget analytic.
tvb_fixture_quadratic <- function(n = 100L) {
  delta <- seq(0, 1, by = 0.01)
  lower <- 0.5 - delta^2
  upper <- 0.5 + delta^2
  bounds <- data.frame(delta = delta, lower = lower, upper = upper,
                       lower_se = 0.1, upper_se = 0.1,
                       ci_lower = lower - 0.15, ci_upper = upper + 0.15)
  new_tvbounds("attrition", bounds, point = 0.5, n = n,
               neighborhood = "tv", level = 0.95, B = 500L,
               estimand_label = "treatment effect")
}
