# Tests for tvbounds_counterfactual(), tvbounds_control(), and the
# Julia bridge. Pure-R argument handling is tested unconditionally;
# everything that touches Julia/KNITRO is guarded by skip_on_cran()
# and tvb_julia_available() (which returns FALSE, without error, when
# JuliaCall, Julia, or a licensed KNITRO installation is missing).

# ---------------------------------------------------------------------
# tvbounds_control(): defaults and validation (pure R)
# ---------------------------------------------------------------------

test_that("tvbounds_control() returns the documented defaults", {
  ctrl <- tvbounds_control()
  expect_s3_class(ctrl, "tvbounds_control")
  expect_identical(ctrl$maxsolves, 10L)
  expect_identical(ctrl$startptrange, 0.01)
  expect_false(ctrl$use_optim)
  expect_identical(ctrl$time_limit, 60)
  expect_identical(ctrl$iterations, 100L)
  expect_identical(ctrl$outer_iterations, 3L)
  expect_null(ctrl$inner_opt)
  expect_null(ctrl$outer_opt)
  expect_identical(ctrl$knitro_options, list())
  expect_identical(ctrl$eta_min, 1e-120)
  expect_identical(ctrl$lower_limit, -10)
  expect_identical(ctrl$psi_tv_eps, 1e-4)
  expect_identical(ctrl$tvac_tau, 1e-3)
  expect_identical(ctrl$tvmix_tau, 1e-3)
  expect_identical(ctrl$purekl_acap, 500)
  expect_null(ctrl$tvmix_kappa)
})

test_that("tvbounds_control() validates its arguments", {
  expect_error(tvbounds_control(maxsolves = 0), "maxsolves")
  expect_error(tvbounds_control(maxsolves = 2.5), "maxsolves")
  expect_error(tvbounds_control(startptrange = -1), "startptrange")
  expect_error(tvbounds_control(use_optim = NA), "use_optim")
  expect_error(tvbounds_control(time_limit = 0), "time_limit")
  expect_error(tvbounds_control(iterations = -3), "iterations")
  expect_error(tvbounds_control(inner_opt = "no/such/file.opt"),
               "does not exist")
  expect_error(tvbounds_control(knitro_options = list(1, 2)), "named")
  expect_error(tvbounds_control(knitro_options = list(maxit = 1:2)),
               "length-one")
  expect_error(tvbounds_control(knitro_options = "maxit 5"), "named list")
  expect_error(tvbounds_control(eta_min = 0), "eta_min")
  expect_error(tvbounds_control(lower_limit = Inf), "lower_limit")
  expect_error(tvbounds_control(psi_tv_eps = 0), "psi_tv_eps")
  expect_error(tvbounds_control(tvmix_kappa = 1.5), "tvmix_kappa")
  expect_error(tvbounds_control(tvmix_kappa = -0.1), "tvmix_kappa")
})

test_that(".tvb_merge_opt_file() overrides and appends options", {
  base <- tempfile(fileext = ".opt")
  writeLines(c("# comment", "maxit  3000", "outlev 0", "feastol 1e-08"),
             base)
  merged <- .tvb_merge_opt_file(
    base, list(maxit = 500, outlev = TRUE, newopt = "yes"))
  expect_true(file.exists(merged))
  expect_true(startsWith(basename(merged), "tvbounds_"))
  lines <- readLines(merged)
  expect_true("maxit 500" %in% lines)
  expect_true("outlev 1" %in% lines)          # logical written as 0/1
  expect_true("newopt yes" %in% lines)        # appended
  expect_true("feastol 1e-08" %in% lines)     # untouched
  expect_true("# comment" %in% lines)         # comments preserved
  # the base file is never modified
  expect_identical(readLines(base)[2], "maxit  3000")
  # names that prefix other names are not clobbered
  merged2 <- .tvb_merge_opt_file(base, list(feastol_abs = "1e-3"))
  lines2 <- readLines(merged2)
  expect_true("feastol 1e-08" %in% lines2)
  expect_true("feastol_abs 1e-3" %in% lines2)
})

# ---------------------------------------------------------------------
# Moments / gradient specification parsing (pure R)
# ---------------------------------------------------------------------

test_that(".tvb_parse_moments() accepts the three documented forms", {
  f <- tempfile(fileext = ".jl")
  writeLines("f() = 1", f)

  s1 <- .tvb_parse_moments(c(f, "my_moments!"))
  expect_identical(s1$type, "file")
  expect_identical(s1$name, "my_moments!")

  s2 <- .tvb_parse_moments("my_moments!")
  expect_identical(s2$type, "name")

  s3 <- .tvb_parse_moments(function(theta, U, gamma) NULL)
  expect_identical(s3$type, "rfun")

  expect_error(.tvb_parse_moments(c("no/such/file.jl", "g")),
               "does not exist")
  expect_error(.tvb_parse_moments(c(f, "not a name")), "valid Julia")
  expect_error(.tvb_parse_moments("bad name"), "valid Julia")
  expect_error(.tvb_parse_moments(c("a", "b", "c")), "length-2")
  expect_error(.tvb_parse_moments(42), "moments")
  expect_error(.tvb_parse_moments(function(theta) NULL), "theta, U")
})

test_that(".tvb_parse_gradient() resolves the differentiation mode", {
  rfun_spec  <- list(type = "rfun")
  julia_spec <- list(type = "name")

  expect_identical(.tvb_parse_gradient(NULL, julia_spec)$mode, "forwarddiff")
  # R moments cannot be differentiated by ForwardDiff -> FD fallback
  expect_identical(.tvb_parse_gradient(NULL, rfun_spec)$mode, "fd")
  expect_identical(.tvb_parse_gradient("fd", julia_spec)$mode, "fd")

  g <- .tvb_parse_gradient("my_jac", julia_spec)
  expect_identical(g$mode, "user")
  expect_identical(g$julia_name, "my_jac")

  g2 <- .tvb_parse_gradient(function(theta, U, gamma) NULL, julia_spec)
  expect_identical(g2$mode, "user")
  expect_null(g2$julia_name)

  expect_error(.tvb_parse_gradient("my_jac", rfun_spec), "R function")
  expect_error(.tvb_parse_gradient("bad name", julia_spec), "Julia")
  expect_error(.tvb_parse_gradient(1L, julia_spec), "gradient")
})

# ---------------------------------------------------------------------
# tvbounds_counterfactual(): argument validation (pure R; all of these
# errors trigger before any Julia code runs)
# ---------------------------------------------------------------------

test_that("tvbounds_counterfactual() validates arguments before Julia", {
  mom <- function(theta, U, gamma) NULL     # never called

  expect_error(
    tvbounds_counterfactual(42, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = 0.5, u_dim = 1),
    "moments")
  expect_error(
    tvbounds_counterfactual(mom, d = 0, theta_lb = 0, theta_ub = 1,
                            delta = 0.5, u_dim = 1),
    "`d`")
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = c(0, 0), theta_ub = 1,
                            delta = 0.5, u_dim = 1),
    "equal")
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 1, theta_ub = 0,
                            delta = 0.5, u_dim = 1),
    "theta_lb")
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = 0.5, u_dim = 1, theta_init = 2),
    "theta_init")
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = numeric(0), u_dim = 1),
    "delta")
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = c(0, 0.5), u_dim = 1),
    "strictly positive")
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = c(0.5, 0.5), u_dim = 1),
    "duplicated")
  # TV-family budgets are capped at 1; KL-family budgets are not
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = c(0.5, 2), divergence = "TV",
                            u_dim = 1),
    "\\(0, 1\\]")
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = 0.5, divergence = "nope", u_dim = 1))
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = 0.5, side = "nope", u_dim = 1))
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = 0.5),
    "u_dim")
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = 0.5, u_dim = 16),
    "u_dim <= 15")
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = 0.5, u_dim = 1, M = 1),
    "M")
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = 0.5, U = "not a matrix"),
    "matrix")
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = 0.5, U = matrix(1:6 / 7, 3, 2),
                            u_dim = 3),
    "ncol")
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = 0.5, u_dim = 1, gamma = "x"),
    "gamma")
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = 0.5, u_dim = 1, gradient = "my_jac"),
    "R function")
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = 0.5, u_dim = 1, control = list()),
    "tvbounds_control")
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = 0.5, u_dim = 1, seed = 1.5),
    "seed")
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = 0.5, u_dim = 1, verbose = "yes"),
    "verbose")
  # the reduced TVmix program needs kappa >= 1 - delta
  expect_error(
    tvbounds_counterfactual(mom, d = 1, theta_lb = 0, theta_ub = 1,
                            delta = c(0.2, 0.9), divergence = "TVmix",
                            u_dim = 1,
                            control = tvbounds_control(tvmix_kappa = 0.3)),
    "TVmixC")
})

test_that("validation failures leave the caller's RNG stream untouched", {
  set.seed(42)
  before <- .Random.seed
  expect_error(
    tvbounds_counterfactual(function(theta, U, gamma) NULL, d = 1,
                            theta_lb = 0, theta_ub = 1, delta = -1,
                            u_dim = 1, seed = 99),
    "strictly positive")
  expect_identical(.Random.seed, before)
})

test_that("the withr seed scope round-trips the RNG state", {
  # The RNG state is scoped through withr::local_seed() /
  # withr::local_preserve_seed(); confirm the pattern used by
  # tvbounds_counterfactual() leaves the caller's stream untouched.
  set.seed(7)
  before <- .Random.seed
  local({
    withr::local_seed(123456)
    runif(10)
  })
  expect_identical(.Random.seed, before)
  local({
    withr::local_preserve_seed()
    runif(10)
  })
  expect_identical(.Random.seed, before)
})

# ---------------------------------------------------------------------
# Guarded integration tests (need Julia + JuliaCall + licensed KNITRO)
# ---------------------------------------------------------------------

test_that("toy Julia model: TVmix bounds equal the theta box endpoints", {
  skip_on_cran()
  skip_if_not(tvb_julia_available(),
              "Julia/KNITRO backend not available")

  toy <- system.file("julia", "examples", "toy.jl", package = "tvbounds")
  skip_if_not(nzchar(toy) && file.exists(toy), "toy.jl not found")

  # Model: U ~ Uniform(0,1), moment E[U] = theta, counterfactual K = U.
  # Under "TVmix" (mixture weight 1 - delta) the inner value equals
  # theta whenever the moment is satisfiable, so the outer bounds are
  # the box endpoints (up to O(tau log M) smoothing).
  fit <- tvbounds_counterfactual(
    moments  = c(toy, "tvb_toy_moments!"),
    d        = 1,
    theta_lb = 0.4, theta_ub = 0.6,
    delta    = c(0.5, 1),
    divergence = "TVmix",
    M = 500, u_dim = 1,
    theta_init = 0.5,
    control = tvbounds_control(maxsolves = 2),
    seed = 1234, verbose = FALSE)

  expect_s3_class(fit, "tvbounds_counterfactual")
  expect_s3_class(fit, "tvbounds")
  expect_identical(fit$application, "counterfactual")
  expect_identical(fit$divergence, "TVmix")
  expect_null(fit$neighborhood)
  expect_identical(fit$bounds$delta, c(0.5, 1))
  expect_equal(fit$bounds$lower, c(0.4, 0.4), tolerance = 0.02)
  expect_equal(fit$bounds$upper, c(0.6, 0.6), tolerance = 0.02)
  # baseline point = mean of the Halton draws ~ 1/2
  expect_equal(fit$point, 0.5, tolerance = 0.02)
  # monotone in the budget by construction
  expect_true(all(diff(fit$bounds$lower) <= 1e-8))
  expect_true(all(diff(fit$bounds$upper) >= -1e-8))
  # diagnostics present
  expect_true(is.data.frame(fit$details$solver))
  expect_identical(dim(fit$details$theta_lower), c(1L, 2L))
})

test_that("toy Julia model: fixed-theta TV bounds at delta = 1", {
  skip_on_cran()
  skip_if_not(tvb_julia_available(),
              "Julia/KNITRO backend not available")

  toy <- system.file("julia", "examples", "toy.jl", package = "tvbounds")
  # Degenerate box: inner-only fixed-theta path. At delta = 1 the TV
  # ball contains every distribution on the draws, so the bound solves
  # the Manski/Lee linear program with the moment E[U] = 0.5 pinned:
  # both bounds equal theta = 0.5.
  fit <- tvbounds_counterfactual(
    moments  = c(toy, "tvb_toy_moments!"),
    d        = 1,
    theta_lb = 0.5, theta_ub = 0.5,
    delta    = 1,
    divergence = "TV",
    M = 500, u_dim = 1,
    theta_init = 0.5,
    seed = 1234, verbose = FALSE)

  expect_true(fit$details$fixed_theta)
  expect_equal(fit$bounds$lower, 0.5, tolerance = 0.02)
  expect_equal(fit$bounds$upper, 0.5, tolerance = 0.02)
})

test_that("R moments bridge reproduces the toy fixed-theta bounds", {
  skip_on_cran()
  skip_if_not(tvb_julia_available(),
              "Julia/KNITRO backend not available")

  r_mom <- function(theta, U, gamma) {
    list(K = U[, 1], G = matrix(U[, 1] - theta[1], ncol = 1))
  }
  fit <- tvbounds_counterfactual(
    moments  = r_mom,
    d        = 1,
    theta_lb = 0.5, theta_ub = 0.5,
    delta    = 1,
    divergence = "TVmix",
    M = 300, u_dim = 1,
    theta_init = 0.5,
    seed = 1234, verbose = FALSE)

  expect_equal(fit$bounds$lower, 0.5, tolerance = 0.02)
  expect_equal(fit$bounds$upper, 0.5, tolerance = 0.02)
  expect_identical(fit$details$gradient, "fd")
})

test_that("seeded counterfactual runs are reproducible", {
  skip_on_cran()
  skip_if_not(tvb_julia_available(),
              "Julia/KNITRO backend not available")

  toy <- system.file("julia", "examples", "toy.jl", package = "tvbounds")
  run <- function() {
    tvbounds_counterfactual(
      moments = c(toy, "tvb_toy_moments!"), d = 1,
      theta_lb = 0.5, theta_ub = 0.5, delta = 1,
      divergence = "TVmix", M = 300, u_dim = 1, theta_init = 0.5,
      seed = 2024, verbose = FALSE)
  }
  f1 <- run()
  f2 <- run()
  expect_identical(f1$bounds$lower, f2$bounds$lower)
  expect_identical(f1$bounds$upper, f2$bounds$upper)
})
