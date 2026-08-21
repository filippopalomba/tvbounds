# Tests for tvbounds_attrition(). All data are simulated in-memory; no files.

# Hand-checkable design: 20 treated units, all respond, outcomes 1..20;
# 20 control units, 10 respond, outcomes all zero. Then p1 = 1, p0 = 0.5,
# p_star = 0.5, mu0 = 0, and (atom-safe fractional trimming):
#   Lee lower  = mean(1:10) = 5.5,  Lee upper  = mean(11:20) = 15.5
#   TV  at 0.5 = [6.75, 14.25],  contamination at 0.5 = [8, 13].
make_hand_data <- function() {
  data.frame(
    y = c(1:20, rep(0, 10), rep(NA_real_, 10)),
    d = c(rep(1L, 20), rep(0L, 20)),
    s = c(rep(1L, 20), rep(1L, 10), rep(0L, 10)))
}

make_sim_data <- function(n = 300, seed = 42, effect = 0.4) {
  set.seed(seed)
  d <- rbinom(n, 1, 0.5)
  s <- rbinom(n, 1, ifelse(d == 1, 0.9, 0.7))
  y <- ifelse(s == 1, rnorm(n, mean = effect * d), NA)
  x <- rbinom(n, 1, 0.5)
  g <- sample(seq_len(50), n, replace = TRUE)
  data.frame(y = y, d = d, s = s, x = x, g = g)
}

test_that("delta = 0 collapses to the baseline difference in means", {
  dat <- make_sim_data()
  fit <- tvbounds_attrition(dat, "y", "d", "s", delta = c(0, 0.5, 1),
                            bootstrap = FALSE)
  dim_val <- mean(dat$y[dat$d == 1 & dat$s == 1], na.rm = TRUE) -
             mean(dat$y[dat$d == 0 & dat$s == 1], na.rm = TRUE)
  expect_equal(fit$bounds$lower[1], dim_val, tolerance = 1e-12)
  expect_equal(fit$bounds$upper[1], dim_val, tolerance = 1e-12)
  expect_equal(fit$point, dim_val, tolerance = 1e-12)
})

test_that("delta = 1 reproduces hand-computed Lee bounds", {
  dat <- make_hand_data()
  for (nb in c("tv", "contamination")) {
    fit <- tvbounds_attrition(dat, "y", "d", "s", delta = c(0, 1),
                              neighborhood = nb, bootstrap = FALSE)
    expect_equal(fit$details$p_star, 0.5, tolerance = 1e-12)
    i1 <- which(fit$bounds$delta == 1)
    expect_equal(fit$bounds$lower[i1], 5.5, tolerance = 1e-10)
    expect_equal(fit$bounds$upper[i1], 15.5, tolerance = 1e-10)
    expect_equal(fit$details$lee$lower, 5.5, tolerance = 1e-10)
    expect_equal(fit$details$lee$upper, 15.5, tolerance = 1e-10)
  }
})

test_that("interior budgets match hand-computed closed forms", {
  dat <- make_hand_data()
  fit_tv <- tvbounds_attrition(dat, "y", "d", "s", delta = 0.5,
                               bootstrap = FALSE)
  expect_equal(fit_tv$bounds$lower, 6.75, tolerance = 1e-10)
  expect_equal(fit_tv$bounds$upper, 14.25, tolerance = 1e-10)
  fit_c <- tvbounds_attrition(dat, "y", "d", "s", delta = 0.5,
                              neighborhood = "contamination",
                              bootstrap = FALSE)
  expect_equal(fit_c$bounds$lower, 8, tolerance = 1e-10)
  expect_equal(fit_c$bounds$upper, 13, tolerance = 1e-10)
})

test_that("bounds are monotone in the budget", {
  dat <- make_sim_data(seed = 7)
  grid <- seq(0, 1, by = 0.05)
  for (nb in c("tv", "contamination")) {
    fit <- tvbounds_attrition(dat, "y", "d", "s", delta = grid,
                              neighborhood = nb, bootstrap = FALSE)
    expect_true(all(diff(fit$bounds$lower) <= 1e-10))
    expect_true(all(diff(fit$bounds$upper) >= -1e-10))
    fitx <- tvbounds_attrition(dat, "y", "d", "s", covariates = "x",
                               delta = grid, neighborhood = nb,
                               bootstrap = FALSE)
    expect_true(all(diff(fitx$bounds$lower) <= 1e-10))
    expect_true(all(diff(fitx$bounds$upper) >= -1e-10))
  }
})

test_that("contamination bounds lie weakly inside total variation bounds", {
  grid <- seq(0, 1, by = 0.1)
  for (sd_ in c(1, 11)) {
    dat <- make_sim_data(seed = sd_)
    tv <- tvbounds_attrition(dat, "y", "d", "s", delta = grid,
                             bootstrap = FALSE)
    cc <- tvbounds_attrition(dat, "y", "d", "s", delta = grid,
                             neighborhood = "contamination",
                             bootstrap = FALSE)
    expect_true(all(cc$bounds$lower >= tv$bounds$lower - 1e-10))
    expect_true(all(cc$bounds$upper <= tv$bounds$upper + 1e-10))
    # with covariates: cell-level contamination inside the pooled TV default
    tvx <- tvbounds_attrition(dat, "y", "d", "s", covariates = "x",
                              delta = grid, bootstrap = FALSE)
    ccx <- tvbounds_attrition(dat, "y", "d", "s", covariates = "x",
                              delta = grid, neighborhood = "contamination",
                              bootstrap = FALSE)
    expect_true(all(ccx$bounds$lower >= tvx$bounds$lower - 1e-10))
    expect_true(all(ccx$bounds$upper <= tvx$bounds$upper + 1e-10))
  }
})

test_that("no differential attrition collapses bounds to the baseline", {
  set.seed(3)
  n <- 200
  d <- rbinom(n, 1, 0.5)
  s <- rep(1L, n)                     # everyone responds: p_star = 0
  y <- rnorm(n, mean = 0.2 * d)
  dat <- data.frame(y = y, d = d, s = s)
  fit <- tvbounds_attrition(dat, "y", "d", "s", delta = c(0, 0.5, 1),
                            bootstrap = FALSE)
  expect_equal(fit$details$p_star, 0)
  expect_true(all(abs(fit$bounds$lower - fit$point) < 1e-12))
  expect_true(all(abs(fit$bounds$upper - fit$point) < 1e-12))
})

test_that("covariate pooled bounds nest the within-stratum reference", {
  dat <- make_sim_data(seed = 5, n = 500)
  grid <- seq(0, 1, by = 0.1)
  fit <- tvbounds_attrition(dat, "y", "d", "s", covariates = "x",
                            delta = grid, bootstrap = FALSE)
  pw <- fit$details$pooled$pw
  expect_s3_class(pw, "data.frame")
  expect_true(all(fit$bounds$upper >= pw$upper - 1e-8))
  expect_true(all(fit$bounds$lower <= pw$lower + 1e-8))
  # endpoints coincide (delta = 0 baseline and delta = 1 covariate Lee)
  expect_equal(fit$bounds$upper[1], pw$upper[1], tolerance = 1e-8)
  expect_equal(fit$bounds$lower[grid == 1], pw$lower[grid == 1],
               tolerance = 1e-8)
  # per-stratum detail present with weights summing to one
  st <- fit$details$pooled$strata
  expect_s3_class(st, "data.frame")
  expect_equal(sum(st$weight), 1, tolerance = 1e-12)
})

test_that("small covariate cells are dropped with a warning", {
  dat <- make_sim_data(seed = 9, n = 200)
  dat$rare <- 0L
  # a cell with control respondents but < min_obs treated observed outcomes
  dat$rare[which(dat$d == 0 & dat$s == 1)[1:3]] <- 1L
  expect_warning(
    tvbounds_attrition(dat, "y", "d", "s", covariates = "rare",
                       delta = c(0, 0.5, 1), bootstrap = FALSE),
    regexp = "dropped")
})

test_that("bootstrap is reproducible under seed and restores the RNG", {
  dat <- make_sim_data(seed = 21, n = 200)
  grid <- c(0, 0.25, 0.5, 0.75, 1)
  f1 <- tvbounds_attrition(dat, "y", "d", "s", delta = grid, B = 30,
                           seed = 99)
  f2 <- tvbounds_attrition(dat, "y", "d", "s", delta = grid, B = 30,
                           seed = 99)
  expect_identical(f1$bounds, f2$bounds)
  f3 <- tvbounds_attrition(dat, "y", "d", "s", delta = grid, B = 30,
                           seed = 100)
  expect_false(identical(f1$bounds$lower_se, f3$bounds$lower_se))
  # RNG state is restored: draws after the call match draws without it
  set.seed(123); before <- runif(3)
  set.seed(123)
  invisible(tvbounds_attrition(dat, "y", "d", "s", delta = grid, B = 10,
                               seed = 7))
  after <- runif(3)
  expect_identical(before, after)
})

test_that("bootstrap output has the documented schema", {
  dat <- make_sim_data(seed = 13, n = 200)
  grid <- c(0, 0.5, 1)
  fit <- tvbounds_attrition(dat, "y", "d", "s", delta = grid, B = 30,
                            seed = 1, level = 0.90)
  expect_true(all(c("delta", "lower", "upper", "lower_se", "upper_se",
                    "ci_lower", "ci_upper") %in% names(fit$bounds)))
  expect_equal(fit$level, 0.90)
  expect_equal(fit$B, 30L)
  # the outer band contains the point bounds
  expect_true(all(fit$bounds$ci_lower <= fit$bounds$lower + 1e-10))
  expect_true(all(fit$bounds$ci_upper >= fit$bounds$upper - 1e-10))
  expect_true(is.matrix(fit$details$boot$draws$lower))
  expect_equal(dim(fit$details$boot$draws$lower), c(30L, length(grid)))
})

test_that("cluster bootstrap runs and is reproducible", {
  dat <- make_sim_data(seed = 17, n = 200)
  grid <- c(0, 0.5, 1)
  f1 <- tvbounds_attrition(dat, "y", "d", "s", delta = grid, B = 20,
                           cluster = "g", seed = 5)
  f2 <- tvbounds_attrition(dat, "y", "d", "s", delta = grid, B = 20,
                           cluster = "g", seed = 5)
  expect_identical(f1$bounds, f2$bounds)
  expect_equal(f1$details$n_clusters, length(unique(dat$g)))
  # same seed, unclustered resampling gives a different bootstrap
  f3 <- tvbounds_attrition(dat, "y", "d", "s", delta = grid, B = 20,
                           seed = 5)
  expect_false(identical(f1$bounds$lower_se, f3$bounds$lower_se))
})

test_that("endpoints are computed even when absent from the budget grid", {
  dat <- make_sim_data(seed = 23)
  fit <- tvbounds_attrition(dat, "y", "d", "s", delta = c(0.3, 0.6),
                            bootstrap = FALSE)
  expect_equal(nrow(fit$bounds), 2L)
  expect_true(is.finite(fit$point))
  expect_true(is.finite(fit$details$lee$lower))
  expect_true(is.finite(fit$details$lee$upper))
})

test_that("returned object satisfies the shared contract", {
  dat <- make_sim_data(seed = 29)
  fit <- tvbounds_attrition(dat, "y", "d", "s", delta = c(0, 0.5, 1),
                            bootstrap = FALSE)
  expect_s3_class(fit, "tvbounds_attrition")
  expect_s3_class(fit, "tvbounds")
  expect_identical(fit$application, "attrition")
  expect_identical(fit$neighborhood, "tv")
  expect_identical(fit$estimand_label, "treatment effect")
  expect_identical(fit$n, nrow(dat))
  expect_true(is.na(fit$level) && is.na(fit$B))
  expect_true(all(fit$bounds$upper - fit$bounds$lower >= -1e-10))
})

test_that("input validation fails early with informative errors", {
  dat <- make_sim_data(seed = 31)
  expect_error(tvbounds_attrition(dat, "nope", "d", "s"), "outcome")
  expect_error(tvbounds_attrition(dat, "y", "d", "s", cluster = "nope"),
               "cluster")
  expect_error(tvbounds_attrition(dat, "y", "d", "s", covariates = "nope"),
               "covariates")
  bad <- dat; bad$d[1] <- 2
  expect_error(tvbounds_attrition(bad, "y", "d", "s"), "treatment")
  expect_error(tvbounds_attrition(dat, "y", "d", "s", delta = c(-0.1, 0.5)),
               "delta")
  expect_error(tvbounds_attrition(dat, "y", "d", "s", delta = 1.5), "delta")
  expect_error(tvbounds_attrition(dat, "y", "d", "s", level = 1.2), "level")
  expect_error(tvbounds_attrition(dat, "y", "d", "s", B = 1), "B")
  expect_error(tvbounds_attrition(dat, "y", "d", "s", min_obs = 1),
               "min_obs")
  badx <- dat; badx$x[5] <- NA
  expect_error(tvbounds_attrition(badx, "y", "d", "s", covariates = "x"),
               "missing")
  expect_error(tvbounds_attrition(as.list(dat), "y", "d", "s"), "data frame")
})

test_that("item non-response among respondents is tolerated", {
  dat <- make_sim_data(seed = 37)
  resp <- which(dat$s == 1)
  dat$y[resp[1:4]] <- NA              # respondents with missing outcome
  fit <- tvbounds_attrition(dat, "y", "d", "s", delta = c(0, 1),
                            bootstrap = FALSE)
  expect_true(all(is.finite(fit$bounds$lower)))
  expect_true(all(is.finite(fit$bounds$upper)))
})
