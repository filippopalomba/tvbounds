# Tests for tvbounds_plot() and the plot() method: structure checks only
# (no snapshot dependencies), on objects with and without confidence bands.

# Number of layers actually built, robust across ggplot2 versions.
n_built_layers <- function(p) length(ggplot2::ggplot_build(p)$data)

test_that("plot returns a buildable ggplot for objects with a confidence band", {
  obj <- tvb_fixture_linear()
  p <- tvbounds_plot(obj)
  expect_s3_class(p, "ggplot")
  expect_no_error(ggplot2::ggplot_build(p))
})

test_that("bands add layers only when CI columns exist", {
  with_ci <- tvb_fixture_linear()
  no_ci <- tvb_fixture_noinf()

  p_bands <- tvbounds_plot(with_ci, bands = TRUE)
  p_nobands <- tvbounds_plot(with_ci, bands = FALSE)
  expect_gt(n_built_layers(p_bands), n_built_layers(p_nobands))

  # bands = TRUE on an object without CI columns is a no-op, not an error.
  p1 <- tvbounds_plot(no_ci, bands = TRUE)
  p2 <- tvbounds_plot(no_ci, bands = FALSE)
  expect_identical(n_built_layers(p1), n_built_layers(p2))
})

test_that("breakdown and baseline marks are optional and skipped when not drawable", {
  obj <- tvb_fixture_linear()
  p_all <- tvbounds_plot(obj, breakdown = TRUE, baseline = TRUE)
  p_min <- tvbounds_plot(obj, breakdown = FALSE, baseline = FALSE)
  expect_gt(n_built_layers(p_all), n_built_layers(p_min))

  # Censored breakdown (no interior crossing): no vline layer, still builds.
  cens <- tvb_fixture_censored()
  expect_no_error(ggplot2::ggplot_build(tvbounds_plot(cens)))
  expect_identical(
    n_built_layers(tvbounds_plot(cens, baseline = FALSE)),
    n_built_layers(tvbounds_plot(cens, breakdown = FALSE, baseline = FALSE)))

  # A missing baseline point disables both marks without error.
  nopt <- tvb_fixture_linear()
  nopt$point <- NA_real_
  expect_no_error(ggplot2::ggplot_build(tvbounds_plot(nopt)))
})

test_that("log_x handles counterfactual grids exceeding 1", {
  cf <- tvb_fixture_counterfactual()
  p_lin <- tvbounds_plot(cf)
  expect_s3_class(p_lin, "ggplot")

  # The grid contains delta = 0, which a log axis cannot show: dropped with
  # a warning, and the plot still builds.
  expect_warning(p_log <- tvbounds_plot(cf, log_x = TRUE), "log_x")
  expect_s3_class(p_log, "ggplot")
  expect_no_error(ggplot2::ggplot_build(p_log))

  # Without nonpositive budgets there is nothing to drop and no warning.
  cf_pos <- tvb_fixture_counterfactual()
  cf_pos$bounds <- cf_pos$bounds[cf_pos$bounds$delta > 0, , drop = FALSE]
  expect_no_warning(tvbounds_plot(cf_pos, log_x = TRUE))
})

test_that("labels default to the budget and the estimand label", {
  obj <- tvb_fixture_noinf()
  p <- tvbounds_plot(obj)
  expect_no_error(ggplot2::ggplot_build(p))
  p2 <- tvbounds_plot(obj, xlab = "budget", ylab = "beta",
                      title = "sensitivity")
  expect_no_error(ggplot2::ggplot_build(p2))
})

test_that("tau_star moves the reference line without error", {
  obj <- tvb_fixture_linear()
  expect_no_error(ggplot2::ggplot_build(tvbounds_plot(obj, tau_star = 0.2)))
})

test_that("plot() dispatches to tvbounds_plot()", {
  obj <- tvb_fixture_linear()
  p <- plot(obj, bands = FALSE)
  expect_s3_class(p, "ggplot")
})

test_that("plot input validation fails early", {
  obj <- tvb_fixture_linear()
  expect_error(tvbounds_plot(42), "tvbounds")
  expect_error(tvbounds_plot(obj, bands = NA), "bands")
  expect_error(tvbounds_plot(obj, log_x = "yes"), "log_x")
  expect_error(tvbounds_plot(obj, tau_star = c(0, 1)), "tau_star")
  expect_error(tvbounds_plot(obj, color = 3), "color")
})
