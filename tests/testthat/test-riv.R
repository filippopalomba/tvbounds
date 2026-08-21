# Tests for tvbounds_riv() and the formula-instrument kernels.
# Every independent check below recomputes a quantity by a route that shares
# no code with the kernels (hand-computed closed forms, brute-force simplex
# grids, random-search containment, direct two-stage least squares).

# ---------------------------------------------------------------------------
# Local helpers (test-only, no package code)
# ---------------------------------------------------------------------------

# A reduced criterion object with prescribed criterion values, mirroring the
# output contract of fi_criteria() on an already-centered design.
fake_cr <- function(gy, gx, gy_v, gx_v) {
  gy <- gy - mean(gy); gx <- gx - mean(gx); S <- length(gy)
  list(gy_s = gy, gx_s = gx, gy_v = gy_v, gx_v = gx_v, p = rep(1 / S, S),
       Egy = 0, Egx = 0, Gy0 = gy_v, Gx0 = gx_v,
       beta_hat = gy_v / gx_v, flip = 1, S = S, n = 1L)
}

# Simulated formula-instrument design with a strong or weak first stage.
sim_design <- function(n = 60, S = 12, seed = 1, strength = 1) {
  set.seed(seed)
  Fmat <- matrix(rnorm(n * S), n, S)
  z <- rowMeans(Fmat) + rnorm(n)
  x <- strength * z + rnorm(n)
  y <- 0.4 * x + rnorm(n)
  list(y = y, x = x, z = z, Fmat = Fmat, n = n, S = S)
}

# beta(P) computed directly from a criterion object and a candidate
# distribution q over the draws.
beta_at <- function(cr, q) {
  (cr$gy_v - sum(q * cr$gy_s)) / (cr$gx_v - sum(q * cr$gx_s))
}

# ---------------------------------------------------------------------------
# 1. Criterion bounds: closed form vs hand computation and a simplex grid
# ---------------------------------------------------------------------------

test_that("upper tail integral and TV criterion bounds match hand-computed values", {
  h <- c(-1.5, -0.5, 0.5, 1.5)
  p <- rep(0.25, 4)
  # keep the top 0.7 mass: 0.25*1.5 + 0.25*0.5 + 0.20*(-0.5) = 0.40
  expect_equal(.upper_tail_integral(h, p, 0.3), 0.40)
  expect_equal(.upper_tail_integral(h, p, 0), sum(p * h))
  expect_equal(.upper_tail_integral(h, p, 1), 0)
  # delta * max(h) + integral: 0.3 * 1.5 + 0.40 = 0.85
  expect_equal(tv_bound_upper(h, p, 0.3), 0.85)
  # small-budget branch: only the bottom atom is trimmed
  expect_equal(tv_bound_upper(h, p, 0.2), 0.6)
  # reflection identity for the lower bound
  expect_equal(tv_bound_lower(h, p, 0.3), -0.85)
  expect_equal(tv_bound_lower(h, p, 0.2), -0.6)
  # degenerate budgets
  expect_equal(tv_bound_upper(h, p, 0), sum(p * h))
  expect_equal(tv_bound_upper(h, p, 1), max(h))
  expect_equal(tv_bound_lower(h, p, 1), min(h))
})

test_that("TV criterion bounds agree with a brute-force simplex grid (S = 3)", {
  # Enumerate distributions on a fine grid over the 2-simplex, keep those in
  # the TV ball, and maximize/minimize the linear criterion directly.
  brute <- function(h, q, d, step = 1 / 200) {
    g <- seq(0, 1, by = step)
    grid <- expand.grid(p1 = g, p2 = g)
    grid <- grid[grid$p1 + grid$p2 <= 1 + 1e-12, ]
    p3 <- 1 - grid$p1 - grid$p2
    tv <- 0.5 * (abs(grid$p1 - q[1]) + abs(grid$p2 - q[2]) + abs(p3 - q[3]))
    keep <- tv <= d + 1e-12
    val <- grid$p1[keep] * h[1] + grid$p2[keep] * h[2] + p3[keep] * h[3]
    c(lo = min(val), up = max(val))
  }
  set.seed(42)
  for (rep in 1:5) {
    h <- rnorm(3, sd = 2)
    q <- if (rep %% 2) rep(1 / 3, 3) else { w <- runif(3); w / sum(w) }
    d <- runif(1, 0.05, 0.95)
    b <- brute(h, q, d)
    tol <- 4 * (1 / 200) * diff(range(h))   # grid discretization error
    # the closed form dominates every feasible grid point ...
    expect_gte(tv_bound_upper(h, q, d), b[["up"]] - 1e-10)
    expect_lte(tv_bound_lower(h, q, d), b[["lo"]] + 1e-10)
    # ... and the grid gets within discretization tolerance of it
    expect_lte(tv_bound_upper(h, q, d), b[["up"]] + tol)
    expect_gte(tv_bound_lower(h, q, d), b[["lo"]] - tol)
  }
})

# ---------------------------------------------------------------------------
# 2. First-stage breakdown budgets: hand-computed piecewise-linear example
# ---------------------------------------------------------------------------

test_that("first-stage breakdown budgets match a hand-solved example", {
  # Centered gx values (-1.5, -0.5, 0.5, 1.5), uniform baseline, gx_v = 1.
  # TV margin: 1 - tv_bound_upper = 1 - 3d on [0, .25], 0.75 - 2d on
  # (.25, .5]; root at d = 0.375.  Contamination: 1 - 1.5 d, root at 2/3.
  cr <- fake_cr(gy = c(0, 0, 0, 0), gx = c(-1.5, -0.5, 0.5, 1.5),
                gy_v = 0.3, gx_v = 1)
  d_tv <- fi_delta_fs_tv(cr)
  d_ct <- fi_delta_fs_cont(cr)
  expect_equal(as.numeric(d_tv), 0.375, tolerance = 1e-9)
  expect_false(attr(d_tv, "censored"))
  expect_equal(as.numeric(d_ct), 2 / 3, tolerance = 1e-12)
  expect_false(attr(d_ct, "censored"))
  # TV breaks first: its neighborhood is the larger one
  expect_lte(as.numeric(d_tv), as.numeric(d_ct) + 1e-12)
  # margins change sign exactly at the breakdown budgets
  expect_gt(.fs_margin_tv(cr, 0.374), 0)
  expect_lt(.fs_margin_tv(cr, 0.376), 0)
  expect_gt(.fs_margin_cont(cr, 0.66), 0)
  expect_lt(.fs_margin_cont(cr, 0.67), 0)
})

test_that("a first stage dominating every draw is censored at 1", {
  cr <- fake_cr(gy = c(-1, 0, 1), gx = c(-1, 0, 1), gy_v = 2, gx_v = 10)
  d_tv <- fi_delta_fs_tv(cr)
  d_ct <- fi_delta_fs_cont(cr)
  expect_equal(as.numeric(d_tv), 1)
  expect_true(attr(d_tv, "censored"))
  expect_equal(as.numeric(d_ct), 1)
  expect_true(attr(d_ct, "censored"))
  # both bound routes stay finite on the whole budget range
  expect_true(all(is.finite(fi_bounds_tv(cr, 1))))
  expect_true(all(is.finite(fi_bounds_cont(cr, 1)[1:2])))
})

# ---------------------------------------------------------------------------
# 3. Bounds on the estimate: containment, attainment, nesting, cross-check
# ---------------------------------------------------------------------------

test_that("random feasible distributions never escape the TV bounds, and the least favorable distribution attains them", {
  set.seed(7)
  for (rep in 1:10) {
    S <- sample(4:9, 1)
    gx <- rnorm(S)
    cr <- fake_cr(rnorm(S), gx, rnorm(1), abs(rnorm(1)) + 3 * max(abs(gx)))
    d <- runif(1, 0.05, 0.9)
    if (.fs_margin_tv(cr, d) <= 1e-6) next
    bb <- fi_bounds_tv(cr, d)
    expect_lte(bb[["lower"]], cr$beta_hat + 1e-10)
    expect_gte(bb[["upper"]], cr$beta_hat - 1e-10)
    # containment: mixtures (1-a) p + a w have TV distance at most a <= d
    for (k in 1:200) {
      w <- rexp(S); w <- w / sum(w)
      a <- runif(1, 0, d)
      q <- (1 - a) * cr$p + a * w
      val <- beta_at(cr, q)
      expect_gte(val, bb[["lower"]] - 1e-9)
      expect_lte(val, bb[["upper"]] + 1e-9)
    }
    # attainment: the least favorable distribution is a probability
    # distribution, lies within budget, and attains its bound
    for (tag in c("lower", "upper")) {
      q <- fi_lf_law_tv(cr, d, bb[[tag]],
                        direction = if (tag == "lower") "upper" else "lower")
      expect_equal(sum(q), 1, tolerance = 1e-12)
      expect_true(all(q >= -1e-15))
      expect_lte(0.5 * sum(abs(q - cr$p)), d + 1e-9)
      expect_equal(beta_at(cr, q), bb[[tag]], tolerance = 1e-7)
    }
  }
})

test_that("random contaminating distributions never escape the contamination bounds, and degenerate contamination attains them", {
  set.seed(11)
  for (rep in 1:10) {
    S <- sample(4:9, 1)
    gx <- rnorm(S)
    cr <- fake_cr(rnorm(S), gx, rnorm(1), abs(rnorm(1)) + 3 * max(abs(gx)))
    d <- runif(1, 0.05, 0.9)
    if (.fs_margin_cont(cr, d) <= 1e-6) next
    bb <- fi_bounds_cont(cr, d)
    for (k in 1:200) {
      w <- rexp(S); w <- w / sum(w)
      q <- (1 - d) * cr$p + d * w
      val <- beta_at(cr, q)
      expect_gte(val, bb[["lower"]] - 1e-9)
      expect_lte(val, bb[["upper"]] + 1e-9)
    }
    # each bound is attained by contamination degenerate at its arg index
    for (tag in c("lower", "upper")) {
      s0 <- bb[[paste0("arg_", tag)]]
      w <- rep(0, S); w[s0] <- 1
      q <- (1 - d) * cr$p + d * w
      expect_equal(beta_at(cr, q), bb[[tag]], tolerance = 1e-10)
    }
  }
})

test_that("contamination bounds are nested in TV bounds, and the two routes agree at delta = 1 when censored", {
  set.seed(3)
  S <- 8
  gx <- rnorm(S)
  cr <- fake_cr(rnorm(S), gx, rnorm(1), abs(rnorm(1)) + 3 * max(abs(gx)))
  for (d in c(0.1, 0.3, 0.6, 0.9)) {
    if (.fs_margin_tv(cr, d) <= 0) next
    tv <- fi_bounds_tv(cr, d)
    ct <- fi_bounds_cont(cr, d)
    expect_gte(ct[["lower"]], tv[["lower"]] - 1e-8)
    expect_lte(ct[["upper"]], tv[["upper"]] + 1e-8)
  }
  # at delta = 1 both neighborhoods are the whole simplex over the draws, so
  # the trimming/root-finding route and the ratio-enumeration route (which
  # share no code) must coincide
  if (isTRUE(attr(fi_delta_fs_tv(cr), "censored"))) {
    b_tv <- fi_bounds_tv(cr, 1)
    b_ct <- fi_bounds_cont(cr, 1)
    expect_equal(b_tv[["lower"]], b_ct[["lower"]], tolerance = 1e-8)
    expect_equal(b_tv[["upper"]], b_ct[["upper"]], tolerance = 1e-8)
  }
})

test_that("shift invariance and sign normalization hold on a simulated design", {
  des <- sim_design(n = 40, S = 10, seed = 5)
  y <- des$y - mean(des$y)
  x <- des$x - mean(des$x)
  cr0 <- fi_criteria(y, x, des$z, des$Fmat)
  # unit-specific shifts of the formula and the realized instrument together
  set.seed(6)
  cc <- rnorm(des$n, sd = 10)
  cr1 <- fi_criteria(y, x, des$z - cc, des$Fmat - cc)
  for (d in c(0.1, 0.4, 0.8, 1)) {
    expect_equal(fi_bounds_tv(cr0, d), fi_bounds_tv(cr1, d),
                 tolerance = 1e-10)
    expect_equal(fi_bounds_cont(cr0, d)[1:2], fi_bounds_cont(cr1, d)[1:2],
                 tolerance = 1e-10)
  }
  # replacing (y, x) by (-y, -x) flips the sign normalization, not the bounds
  cr2 <- fi_criteria(-y, -x, des$z, des$Fmat)
  expect_equal(cr2$flip, -cr0$flip)
  expect_equal(cr0$beta_hat, cr2$beta_hat, tolerance = 1e-12)
  for (d in c(0.2, 0.6)) {
    expect_equal(fi_bounds_tv(cr0, d), fi_bounds_tv(cr2, d),
                 tolerance = 1e-9)
  }
})

# ---------------------------------------------------------------------------
# 4. Breakdown budget for a reference value
# ---------------------------------------------------------------------------

test_that("fi_breakdown locates the first covering budget and handles censoring", {
  # synthetic monotone bounds with a known crossing: lower(d) = 1 - 4d covers
  # tau_star = 0 first, at d = 0.25
  lo <- function(d) 1 - 4 * d
  up <- function(d) 1 + d
  expect_equal(fi_breakdown(lo, up, tau_star = 0), 0.25, tolerance = 1e-8)
  # covered already at the baseline
  expect_equal(fi_breakdown(lo, up, tau_star = 1), 0)
  # never covered on [0, 1]
  expect_true(is.na(fi_breakdown(lo, up, tau_star = 5)))
})

# ---------------------------------------------------------------------------
# 5. The user-facing function: schema, collapse at 0, NA encoding, controls
# ---------------------------------------------------------------------------

test_that("tvbounds_riv returns the common object with the required schema", {
  des <- sim_design(seed = 2)
  fit <- tvbounds_riv(des$y, des$x, des$z, des$Fmat,
                      delta = seq(0, 1, by = 0.1))
  expect_s3_class(fit, c("tvbounds_riv", "tvbounds"), exact = TRUE)
  expect_identical(fit$application, "riv")
  expect_identical(fit$neighborhood, "tv")
  expect_identical(names(fit$bounds), c("delta", "lower", "upper"))
  expect_identical(nrow(fit$bounds), 11L)
  expect_identical(fit$n, length(des$y))
  expect_true(is.na(fit$level))
  expect_true(is.na(fit$B))
  expect_identical(fit$estimand_label, "IV coefficient")
  # delta = 0 row collapses to the baseline estimate
  expect_equal(fit$bounds$lower[1], fit$point)
  expect_equal(fit$bounds$upper[1], fit$point)
  # monotone in the budget on the finite range
  fin <- stats::complete.cases(fit$bounds)
  expect_true(all(diff(fit$bounds$lower[fin]) <= 1e-8))
  expect_true(all(diff(fit$bounds$upper[fin]) >= -1e-8))
  # deterministic: the same call reproduces the same object
  fit2 <- tvbounds_riv(des$y, des$x, des$z, des$Fmat,
                       delta = seq(0, 1, by = 0.1))
  expect_equal(fit$bounds, fit2$bounds)
  expect_equal(fit$details$criteria, fit2$details$criteria)
})

test_that("bounds rows are NA exactly where the first stage can vanish", {
  # weak first stage so that the breakdown budget is interior
  des <- sim_design(n = 50, S = 10, seed = 8, strength = 0.05)
  grid <- seq(0, 1, by = 0.02)
  fit <- tvbounds_riv(des$y, des$x, des$z, des$Fmat, delta = grid)
  dfs <- fit$details$delta_fs
  expect_false(fit$details$delta_fs_censored)
  expect_gt(dfs, 0)
  expect_lt(dfs, 1)
  na_rows <- is.na(fit$bounds$lower)
  expect_identical(na_rows, is.na(fit$bounds$upper))
  # finite strictly below the breakdown budget, NA strictly above it
  expect_true(all(!na_rows[grid < dfs - 0.02]))
  expect_true(all(na_rows[grid > dfs + 0.02]))
  # the reported breakdown budget for the sign is covered by the bounds
  db <- fit$details$delta_breakdown
  if (!fit$details$delta_breakdown_censored && db < dfs) {
    lo <- fi_bounds_tv(fit$details$criteria, min(db + 1e-6, 1))
    expect_lte(lo[["lower"]], fit$details$tau_star + 1e-6)
    expect_gte(lo[["upper"]], fit$details$tau_star - 1e-6)
  }
})

test_that("contamination bounds are nested in TV bounds through the user-facing interface", {
  des <- sim_design(seed = 4)
  grid <- seq(0, 1, by = 0.1)
  fit_tv <- tvbounds_riv(des$y, des$x, des$z, des$Fmat, delta = grid)
  fit_ct <- tvbounds_riv(des$y, des$x, des$z, des$Fmat, delta = grid,
                         neighborhood = "contamination")
  expect_identical(fit_ct$neighborhood, "contamination")
  expect_equal(fit_tv$point, fit_ct$point)
  fin <- stats::complete.cases(fit_tv$bounds) &
    stats::complete.cases(fit_ct$bounds)
  expect_true(any(fin))
  expect_true(all(fit_ct$bounds$lower[fin] >= fit_tv$bounds$lower[fin] - 1e-8))
  expect_true(all(fit_ct$bounds$upper[fin] <= fit_tv$bounds$upper[fin] + 1e-8))
  # the TV first stage breaks (weakly) before the contamination one
  expect_lte(fit_tv$details$delta_fs, fit_ct$details$delta_fs + 1e-12)
})

test_that("FWL residualization replicates a direct two-stage least squares fit", {
  des <- sim_design(n = 70, S = 8, seed = 9)
  W <- cbind(w1 = rnorm(des$n), w2 = runif(des$n))
  fit <- tvbounds_riv(des$y, des$x, des$z, des$Fmat, controls = W,
                      delta = c(0, 0.1))
  # direct just-identified 2SLS of y on (x, 1, W) with instruments
  # (z - F p, 1, W), no residualization anywhere
  zr <- des$z - rowMeans(des$Fmat)
  Zmat <- cbind(zr, 1, W)
  Xmat <- cbind(des$x, 1, W)
  b_direct <- as.numeric(solve(crossprod(Zmat, Xmat), crossprod(Zmat, des$y)))[1]
  expect_equal(fit$point, b_direct, tolerance = 1e-9)
  expect_identical(fit$details$n_controls, 2L)
  # a data-frame controls argument gives the same result
  fit_df <- tvbounds_riv(des$y, des$x, des$z, des$Fmat,
                         controls = as.data.frame(W), delta = c(0, 0.1))
  expect_equal(fit_df$point, fit$point)
  expect_equal(fit_df$bounds, fit$bounds)
})

test_that("censoring flags propagate to details for a strong first stage", {
  des <- sim_design(seed = 10, strength = 2)
  fit <- tvbounds_riv(des$y, des$x, des$z, des$Fmat,
                      delta = seq(0, 1, by = 0.25))
  expect_true(fit$details$delta_fs_censored)
  expect_equal(fit$details$delta_fs, 1)
  expect_true(isTRUE(attr(fit$details$delta_fs_tv, "censored")))
  expect_true(isTRUE(attr(fit$details$delta_fs_cont, "censored")))
  # no NA rows: the bounds stay finite over the whole budget range
  expect_true(all(stats::complete.cases(fit$bounds)))
  # criteria stored in details reproduce the bounds without the design
  cr <- fit$details$criteria
  redo <- t(vapply(fit$bounds$delta, function(d) fi_bounds_tv(cr, d),
                   numeric(2)))
  expect_equal(fit$bounds$lower, redo[, 1])
  expect_equal(fit$bounds$upper, redo[, 2])
})

test_that("a supplied baseline distribution p is honored", {
  des <- sim_design(n = 40, S = 6, seed = 12)
  p <- c(0.4, 0.2, 0.1, 0.1, 0.1, 0.1)
  fit <- tvbounds_riv(des$y, des$x, des$z, des$Fmat, p = p,
                      delta = c(0, 0.2, 0.5))
  expect_equal(fit$details$criteria$p, p)
  # the point estimate matches the direct recentering under p
  mu <- as.numeric(des$Fmat %*% p)
  yr <- des$y - mean(des$y); xr <- des$x - mean(des$x)
  b_direct <- sum(yr * (des$z - mu)) / sum(xr * (des$z - mu))
  expect_equal(fit$point, b_direct, tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# 6. Input validation
# ---------------------------------------------------------------------------

test_that("tvbounds_riv validates its inputs with informative errors", {
  des <- sim_design(n = 20, S = 5, seed = 13)
  ok <- function(...) tvbounds_riv(des$y, des$x, des$z, des$Fmat, ...)
  expect_error(tvbounds_riv(des$y[-1], des$x, des$z, des$Fmat), "same length")
  expect_error(tvbounds_riv(c(des$y[-1], NA), des$x, des$z, des$Fmat), "`y`")
  expect_error(tvbounds_riv(des$y, des$x, des$z, des$Fmat[-1, ]),
               "one row per observation")
  expect_error(tvbounds_riv(des$y, des$x, des$z, "not a matrix"), "`F`")
  expect_error(tvbounds_riv(des$y, des$x, des$z, des$Fmat[, 1, drop = FALSE]),
               "at least 2")
  expect_error(ok(p = rep(1, 5)), "sum to 1")
  expect_error(ok(p = c(-0.2, 0.3, 0.3, 0.3, 0.3)), "nonnegative")
  expect_error(ok(p = rep(0.25, 4)), "one probability per column")
  expect_error(ok(delta = c(-0.1, 0.5)), "\\[0, 1\\]")
  expect_error(ok(delta = c(0.2, 1.5)), "\\[0, 1\\]")
  expect_error(ok(delta = c(0.2, 0.2)), "duplicated")
  expect_error(ok(delta = numeric(0)), "non-empty")
  expect_error(ok(delta = c(0.1, NA)), "missing")
  expect_error(ok(tau_star = c(0, 1)), "`tau_star`")
  expect_error(ok(tau_star = Inf), "`tau_star`")
  expect_error(ok(verbose = NA), "`verbose`")
  expect_error(ok(controls = matrix(rnorm(10), 5, 2)),
               "one row per observation")
  W_bad <- data.frame(w1 = rnorm(20), w2 = letters[1:20])
  expect_error(ok(controls = W_bad), "numeric columns")
  expect_error(ok(controls = matrix(c(rnorm(19), NA), 20, 1)),
               "missing or non-finite")
})

test_that("verbose = TRUE emits progress messages, verbose = FALSE is silent", {
  des <- sim_design(n = 30, S = 5, seed = 14)
  msgs <- capture_messages(
    tvbounds_riv(des$y, des$x, des$z, des$Fmat, delta = c(0, 0.5),
                 verbose = TRUE))
  expect_true(any(grepl("Recentered IV design", msgs)))
  expect_true(any(grepl("First-stage breakdown budget", msgs)))
  expect_true(any(grepl("Breakdown budget for tau_star", msgs)))
  expect_silent(
    tvbounds_riv(des$y, des$x, des$z, des$Fmat, delta = c(0, 0.5)))
})
