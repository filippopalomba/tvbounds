# Tests for tvbounds_summary() and its print method, on fixtures with
# analytic breakdown budgets, shadow prices, and frontier values.

test_that("linear fixture reproduces all analytic summary measures", {
  obj <- tvb_fixture_linear()          # point 0.5, slope 1, se 0.1, gap 0.15
  s <- tvbounds_summary(obj)
  m <- s$measures
  zc <- qnorm(0.975)

  expect_s3_class(s, "tvbounds_summary")
  expect_identical(nrow(m), 1L)
  expect_identical(m$direction, "lower")   # point 0.5 > tau* = 0

  # Breakdown budgets: plug-in at 0.5, certified at 0.35, normal floor at
  # 0.5 - zc * 0.1.
  expect_equal(m$delta_b, 0.5, tolerance = 1e-6)
  expect_false(m$censored)
  expect_equal(m$delta_b_ci, 0.35, tolerance = 1e-6)
  expect_false(m$censored_ci)
  expect_equal(m$delta_b_ci_norm, 0.5 - zc * 0.1, tolerance = 1e-6)

  # Shadow price and robustness standard error at the plug-in breakdown: eta = slope =
  # 1, se = 0.1, n = 100 -> varsigma = 0.1 * 10 / 1 = 1.
  expect_equal(m$delta_eval, 0.5, tolerance = 1e-6)
  expect_equal(m$eta, 1, tolerance = 1e-6)
  expect_equal(m$se, 0.1, tolerance = 1e-6)
  expect_equal(m$varsigma, 1, tolerance = 1e-6)
  expect_equal(m$varsigma_sc, 0.1, tolerance = 1e-6)

  # Certification frontier priced at the certified breakdown 0.35 and at
  # 0.35 + jump = 0.40: path values 0.15 and 0.10, sigma^2 = se^2 * n = 1.
  expect_equal(m$frontier_at, 0.35, tolerance = 1e-6)
  expect_equal(m$n_cur, zc^2 / 0.15^2, tolerance = 1e-6)
  expect_identical(as.numeric(m$n_star), floor(zc^2 / 0.10^2) + 1)  # 385
  expect_equal(m$delta_n, m$n_star - 100)
  expect_equal(m$cost_per_pp, m$delta_n * 50 / (100 * 0.05),
               tolerance = 1e-12)
})

test_that("direction auto uses the upper path when point < tau_star", {
  lo <- tvbounds_summary(tvb_fixture_linear())
  up <- tvbounds_summary(tvb_fixture_linear_upper())
  expect_identical(up$measures$direction, "upper")
  # The mirrored object has the same signed path, hence identical measures.
  same <- c("delta_b", "delta_b_ci", "delta_b_ci_norm", "eta", "se",
            "varsigma", "varsigma_sc", "n_cur", "n_star", "delta_n",
            "cost_per_pp")
  for (col in same) {
    expect_equal(up$measures[[col]], lo$measures[[col]], tolerance = 1e-6,
                 label = paste0("upper$", col),
                 expected.label = paste0("lower$", col))
  }
})

test_that("forced directions are respected", {
  obj <- tvb_fixture_linear()
  s_lo <- tvbounds_summary(obj, direction = "lower")
  expect_identical(s_lo$measures$direction, "lower")
  # Forcing the upper path on an object whose point exceeds tau_star gives
  # a path that is already nonpositive at the smallest budget: the
  # breakdown collapses to the left endpoint of the grid.
  s_up <- tvbounds_summary(obj, direction = "upper")
  expect_identical(s_up$measures$direction, "upper")
  expect_equal(s_up$measures$delta_b, 0, tolerance = 1e-12)
})

test_that("censored plug-in breakdown is reported at the right endpoint and anchored at the certified budget", {
  obj <- tvb_fixture_censored()        # lower = 0.5 - 0.3 delta, never crosses
  s <- tvbounds_summary(obj)
  m <- s$measures

  expect_true(m$censored)
  expect_equal(m$delta_b, 1)           # right endpoint, not NA / "> 1"
  expect_equal(m$delta_b_ci, 0.25 / 0.3, tolerance = 1e-6)
  expect_false(m$censored_ci)
  # Anchored at the certified budget: eta = 0.3 there, varsigma = 0.1 * 10 / 0.3.
  expect_equal(m$delta_eval, 0.25 / 0.3, tolerance = 1e-6)
  expect_equal(m$eta, 0.3, tolerance = 1e-6)
  expect_equal(m$varsigma, 1 / 0.3, tolerance = 1e-5)
  expect_true(any(grepl("censored", s$notes)))
})

test_that("objects without inference degrade gracefully", {
  obj <- tvb_fixture_noinf()
  s <- tvbounds_summary(obj)
  m <- s$measures

  expect_equal(m$delta_b, 0.3, tolerance = 1e-6)
  expect_false(s$has_se)
  expect_false(s$has_ci)
  expect_true(is.na(m$delta_b_ci))
  expect_true(is.na(m$delta_b_ci_norm))
  expect_true(is.na(m$se))
  expect_true(is.na(m$varsigma))
  expect_true(is.na(m$n_cur))
  expect_true(is.na(m$n_star))
  # eta is a plug-in quantity and survives: slope of the lower path is 1.
  expect_equal(m$eta, 1, tolerance = 1e-6)
  expect_true(length(s$notes) >= 1L)
  expect_true(any(grepl("no inference by design|no confidence band", s$notes)))
})

test_that("counterfactual grids exceeding 1 are handled, including censoring at the right endpoint", {
  s <- tvbounds_summary(tvb_fixture_counterfactual(slope = 0.4))
  # 1 - 0.4 delta crosses zero at 2.5, interior to the uneven grid.
  expect_equal(s$measures$delta_b, 2.5, tolerance = 1e-6)
  expect_false(s$measures$censored)

  s2 <- tvbounds_summary(tvb_fixture_counterfactual(slope = 0.1))
  # 1 - 0.1 delta stays positive up to delta = 5: censored at the right
  # endpoint of THIS grid (5), not at 1.
  expect_true(s2$measures$censored)
  expect_equal(s2$measures$delta_b, 5)
})

test_that("a user-supplied evaluation budget overrides the anchor", {
  obj <- tvb_fixture_quadratic()       # lower = 0.5 - delta^2, eta = 2 delta
  zc <- qnorm(0.975)
  s <- tvbounds_summary(obj, delta = 0.3)
  m <- s$measures

  expect_equal(m$delta_eval, 0.3)
  expect_equal(m$eta, 0.6, tolerance = 1e-8)
  expect_equal(m$varsigma, 0.1 * 10 / 0.6, tolerance = 1e-6)
  # Frontier priced at the evaluation budget and at budget + jump.
  expect_equal(m$frontier_at, 0.3)
  expect_equal(m$n_cur, zc^2 / (0.5 - 0.09)^2, tolerance = 1e-6)
  expect_identical(as.numeric(m$n_star),
                   floor(zc^2 / (0.5 - 0.35^2)^2) + 1)
})

test_that("nonzero tau_star shifts the breakdown budget", {
  obj <- tvb_fixture_linear()          # lower = 0.5 - delta
  s <- tvbounds_summary(obj, tau_star = 0.2)
  expect_equal(s$measures$delta_b, 0.3, tolerance = 1e-6)
})

test_that("input validation fails early with informative errors", {
  obj <- tvb_fixture_linear()
  expect_error(tvbounds_summary(42), "tvbounds")
  expect_error(tvbounds_summary(obj, delta = c(0.1, 0.2)), "delta")
  expect_error(tvbounds_summary(obj, delta = -0.1), "delta")
  expect_error(tvbounds_summary(obj, level = 1.2), "level")
  expect_error(tvbounds_summary(obj, jump = 0), "jump")
  expect_error(tvbounds_summary(obj, cost_per_unit = -1), "cost_per_unit")
  expect_error(tvbounds_summary(obj, tau_star = c(0, 1)), "tau_star")
  expect_error(tvbounds_summary(obj, direction = "sideways"))

  # direction = "auto" needs a baseline point estimate.
  nopt <- tvb_fixture_linear()
  nopt$point <- NA_real_
  expect_error(tvbounds_summary(nopt), "direction")
  expect_s3_class(tvbounds_summary(nopt, direction = "lower"),
                  "tvbounds_summary")
})

test_that("evaluation budgets beyond the grid warn and extrapolate flat", {
  obj <- tvb_fixture_linear()
  expect_warning(s <- tvbounds_summary(obj, delta = 1.5), "extrapolation")
  # Flat extrapolation: eta at 1.5 equals eta at the right endpoint.
  expect_equal(s$measures$eta, 1, tolerance = 1e-6)
})

test_that("summary() dispatches to tvbounds_summary()", {
  obj <- tvb_fixture_linear()
  s1 <- summary(obj)
  s2 <- tvbounds_summary(obj)
  expect_s3_class(s1, "tvbounds_summary")
  expect_equal(s1$measures, s2$measures, tolerance = 1e-12)
})

test_that("print methods render compactly and return invisibly", {
  obj <- tvb_fixture_linear()
  expect_output(print(obj), "tvbounds")
  expect_output(print(obj), "total-variation neighborhood")
  expect_output(print(obj), "bootstrap")

  s <- tvbounds_summary(obj)
  expect_output(print(s), "breakdown budget")
  expect_output(print(s), "shadow price")
  expect_output(print(s), "certification frontier")
  # print methods return their argument invisibly.
  dump <- capture.output(vis <- withVisible(print(s)))
  expect_false(vis$visible)
  expect_identical(vis$value, s)

  # Censored and no-inference objects print without error.
  expect_output(print(tvbounds_summary(tvb_fixture_censored())), "censored")
  expect_output(print(tvbounds_summary(tvb_fixture_noinf())), "note")
  expect_output(print(tvb_fixture_noinf()), "none attached")
})
