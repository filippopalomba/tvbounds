# tvbounds

Sensitivity analysis and bounds under total variation neighborhoods, the
companion software to Palomba (2026), "Sensitivity Analysis in Population
Shares." The methods are available **both as an R package and as a Python
module**, with the same functions, the same options and the same manual.

## Installation

The R package is on [CRAN](https://cran.r-project.org/package=tvbounds) and
the Python module on [PyPI](https://pypi.org/project/tvbounds/):

```r
install.packages("tvbounds")
```

```bash
pip install tvbounds
```

The development versions live in this repository, the R package in
[`R/`](R) and the Python module in [`python/`](python), so each is installed
from its own subdirectory:

```r
# install.packages("remotes")
remotes::install_github("filippopalomba/tvbounds", subdir = "R",
                        build_vignettes = TRUE)
```

```bash
pip install "git+https://github.com/filippopalomba/tvbounds.git#subdirectory=python"
```

Only `ggplot2` and `withr` (plus base R) are required at runtime by the R
package; `build_vignettes = TRUE` additionally needs `knitr` and
`rmarkdown`. The Python module requires only `numpy`, `pandas` and
`matplotlib`.

The [package vignette](https://cran.r-project.org/web/packages/tvbounds/vignettes/tvbounds.html)
explains what the software does and walks through every functionality in
detail. The examples below are a condensed version of the ones in the
vignette, and are given in both languages.

## KNITRO requirement for structural counterfactuals

`tvbounds_counterfactual()` solves its
optimization problems in Julia through the commercial
[Artelys KNITRO](https://www.artelys.com/solvers/knitro/) solver, so it
requires Julia (>= 1.9), a valid KNITRO license, and the `JuliaCall` R
package or the `juliacall` Python package. **Students can request a free
one-year KNITRO license** through Artelys' academic program on the same
page. The package checks for KNITRO once per session, on the first call.

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

```python
import numpy as np
import pandas as pd
from tvbounds import tvbounds_attrition, tvbounds_summary

## a small experiment with village-level assignment and selective attrition
rng = np.random.default_rng(20260820)
n = 500
village = np.repeat(np.arange(1, 51), 10)
d = rng.binomial(1, 0.5, 50)[village - 1]
x = rng.binomial(1, 0.4, n)
ability = rng.normal(size=n)
s = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.2 + 1.2 * d + 0.5 * ability)))).astype(int)
y = np.where(s == 1, 1 + 0.35 * d + 0.5 * x + ability + 0.5 * rng.normal(size=n), np.nan)
rct = pd.DataFrame({"y": y, "d": d, "s": s, "x": x, "village": village})

## total variation bounds with a bootstrap confidence band
fit_tv = tvbounds_attrition(rct,
    outcome="y", treatment="d", response="s",
    delta=np.linspace(0, 1, 21), B=200, seed=1)
print(fit_tv)
fit_tv.plot()

## contamination neighborhood: weakly tighter bounds, same endpoints
fit_ct = tvbounds_attrition(rct,
    outcome="y", treatment="d", response="s",
    delta=np.linspace(0, 1, 21), neighborhood="contamination",
    bootstrap=False)

## cluster bootstrap and covariate-pooled bounds
fit_cl = tvbounds_attrition(rct,
    outcome="y", treatment="d", response="s",
    delta=np.linspace(0, 1, 21), B=200, cluster="village", seed=1)
fit_x = tvbounds_attrition(rct,
    outcome="y", treatment="d", response="s", covariates="x",
    delta=np.linspace(0, 1, 21), B=200, seed=1)

## summary measures: breakdown budgets, shadow price, robustness
## standard error, certification frontier
print(fit_tv.summary())
print(tvbounds_summary(fit_tv, delta=0.1))
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

```python
import numpy as np
from tvbounds import tvbounds_riv

## a shift-share design: exposure shares W, S counterfactual shock draws
rng = np.random.default_rng(1901)
n, K, S = 150, 10, 80
W = rng.exponential(size=(n, K)) ** 2
W = W / W.sum(axis=1, keepdims=True)
g0 = rng.normal(loc=0.3, size=K)
G = rng.normal(size=(K, S))
z = W @ g0                                 # realized formula instrument
Fmat = W @ G                               # n x S counterfactual draws
x = z + rng.normal(size=n)
y = 0.5 * x + 0.4 * (W @ rng.normal(size=K)) + 0.5 * rng.normal(size=n)

## bounds under both neighborhoods (deterministic; no inference by design)
riv_tv = tvbounds_riv(y, x, z, Fmat, delta=np.linspace(0, 1, 101))
riv_ct = tvbounds_riv(y, x, z, Fmat, delta=np.linspace(0, 1, 101),
                      neighborhood="contamination")
print(riv_tv)
riv_tv.plot()

## first-stage breakdown budget and its censoring flag
riv_tv.details["delta_fs"], riv_tv.details["delta_fs_censored"]
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

```python
import tvbounds
from tvbounds import tvbounds_counterfactual, tvbounds_control

## a toy model shipped with the package: moments in Julia, Bundle signature
toy = tvbounds.julia_file("examples", "toy.jl")

fit_cf = tvbounds_counterfactual(
    moments    = (toy, "tvb_toy_moments!"),   # Julia file + function name
    d          = 1,                           # number of moment conditions
    theta_lb   = 0.4, theta_ub = 0.6,         # box for the structural parameter
    delta      = [0.05, 0.1, 0.25, 0.5, 1],   # budgets (strictly positive)
    divergence = "TVmix",
    side       = "both",
    M          = 5000, u_dim = 1,             # scrambled-Halton draws
    theta_init = 0.5,
    seed       = 1234,
    control    = tvbounds_control(maxsolves = 5))

fit_cf.bounds
fit_cf.plot()
```

## Citation

If you use `tvbounds`, please cite:

Palomba, F. (2026). "Sensitivity Analysis in Population Shares." Working
paper.

## License

MIT © Filippo Palomba. See `LICENSE`.
