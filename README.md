# tvbounds

Sensitivity analysis and bounds under total variation neighborhoods, the
companion R package to Palomba (2026), "Sensitivity Analysis in Population
Shares."

The [package vignette](https://cran.r-project.org/web/packages/tvbounds/vignettes/tvbounds.html)
explains what the package does and walks through every functionality in
detail; the examples below are condensed from it.

## Installation

```r
# install.packages("remotes")
remotes::install_github("filippopalomba/tvbounds", build_vignettes = TRUE)
```

While this repository is private, authenticate first — for instance store a
GitHub token with `gitcreds::gitcreds_set()`, or set the `GITHUB_PAT`
environment variable. Only `ggplot2` and `withr` (plus base R) are required
at runtime; `build_vignettes = TRUE` additionally needs `knitr` and
`rmarkdown`.

## KNITRO requirement for structural counterfactuals

`tvbounds_counterfactual()` — and only that function — solves its
optimization problems in Julia through the commercial
[Artelys KNITRO](https://www.artelys.com/solvers/knitro/) solver, so it
requires Julia (>= 1.9), the `JuliaCall` R package, and a valid KNITRO
license. **Students can request a free one-year KNITRO license** through
Artelys' academic program on the same page. The package checks for KNITRO
once per R session, on the first call; every other function is pure R and
needs none of this.

## Quick tour

The commands below reproduce, in brief, what the vignette develops in
detail.

### Randomized experiments with attrition

```r
library(tvbounds)

## a small experiment with village-level assignment and selective attrition
set.seed(20260820)
n <- 500
village <- rep(1:50, each = 10)
d <- as.integer(rbinom(50, 1, 0.5)[village])
x <- rbinom(n, 1, 0.4)
ability <- rnorm(n)
s <- as.integer(runif(n) < plogis(0.2 + 1.2 * d + 0.5 * ability))
y <- ifelse(s == 1, 1 + 0.35 * d + 0.5 * x + ability + 0.5 * rnorm(n), NA)
rct <- data.frame(y = y, d = d, s = s, x = x, village = village)

## total variation bounds with a bootstrap confidence band
fit_tv <- tvbounds_attrition(rct,
  outcome = "y", treatment = "d", response = "s",
  delta = seq(0, 1, by = 0.05), B = 200, seed = 1)
fit_tv
plot(fit_tv)

## contamination neighborhood: weakly tighter bounds, same endpoints
fit_ct <- tvbounds_attrition(rct,
  outcome = "y", treatment = "d", response = "s",
  delta = seq(0, 1, by = 0.05), neighborhood = "contamination",
  bootstrap = FALSE)

## cluster bootstrap and covariate-pooled bounds
fit_cl <- tvbounds_attrition(rct,
  outcome = "y", treatment = "d", response = "s",
  delta = seq(0, 1, by = 0.05), B = 200, cluster = "village", seed = 1)
fit_x <- tvbounds_attrition(rct,
  outcome = "y", treatment = "d", response = "s", covariates = "x",
  delta = seq(0, 1, by = 0.05), B = 200, seed = 1)

## summary measures: breakdown budgets, shadow price, robustness
## standard error, certification frontier
summary(fit_tv)
tvbounds_summary(fit_tv, delta = 0.1)
```

### Recentered instrumental variables

```r
## a shift-share design: exposure shares W, S counterfactual shock draws
set.seed(1901)
n <- 150; K <- 10; S <- 80
W <- matrix(rexp(n * K)^2, n, K)
W <- W / rowSums(W)
g0   <- rnorm(K, mean = 0.3)
G    <- matrix(rnorm(K * S), K, S)
z    <- as.vector(W %*% g0)                # realized formula instrument
Fmat <- W %*% G                            # n x S counterfactual draws
x    <- z + rnorm(n)
y    <- 0.5 * x + 0.4 * as.vector(W %*% rnorm(K)) + 0.5 * rnorm(n)

## bounds under both neighborhoods (deterministic; no inference by design)
riv_tv <- tvbounds_riv(y, x, z, Fmat, delta = seq(0, 1, by = 0.01))
riv_ct <- tvbounds_riv(y, x, z, Fmat, delta = seq(0, 1, by = 0.01),
                       neighborhood = "contamination")
riv_tv
plot(riv_tv)

## first-stage breakdown budget and its censoring flag
c(delta_fs = riv_tv$details$delta_fs,
  censored = riv_tv$details$delta_fs_censored)
```

### Counterfactuals in structural models (requires KNITRO)

```r
## a toy model shipped with the package: moments in Julia, Bundle signature
toy <- system.file("julia", "examples", "toy.jl", package = "tvbounds")

fit_cf <- tvbounds_counterfactual(
  moments    = c(toy, "tvb_toy_moments!"),  # Julia file + function name
  d          = 1,                           # number of moment conditions
  theta_lb   = 0.4, theta_ub = 0.6,         # box for the structural parameter
  delta      = c(0.05, 0.1, 0.25, 0.5, 1),  # budgets (strictly positive)
  divergence = "TVmix",
  side       = "both",
  M          = 5000, u_dim = 1,             # scrambled-Halton draws
  theta_init = 0.5,
  seed       = 1234,
  control    = tvbounds_control(maxsolves = 5))

fit_cf$bounds
plot(fit_cf)
```

## Citation

If you use `tvbounds`, please cite:

Palomba, F. (2026). "Sensitivity Analysis in Population Shares." Working
paper. <https://filippopalomba.github.io/#jmp>

## License

MIT © Filippo Palomba. See `LICENSE.md`.
