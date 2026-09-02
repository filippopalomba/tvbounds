# tvbounds

Sensitivity analysis and bounds under total variation neighborhoods, the
companion Python module to Palomba (2026), "Sensitivity Analysis in
Population Shares." The reference manual is `docs/tvbounds-manual.pdf`.

## Installation

```bash
pip install tvbounds
```

Only `numpy`, `pandas`, and `matplotlib` are required at runtime. The
structural-counterfactual solver additionally needs the `juliacall`
package (`pip install "tvbounds[julia]"`), a Julia installation, and the
Artelys KNITRO solver (see below).

## KNITRO requirement for structural counterfactuals

`tvbounds_counterfactual()` solves its optimization problems in Julia
through the commercial [Artelys KNITRO](https://www.artelys.com/solvers/knitro/)
solver, so it requires Julia (>= 1.9), the `juliacall` Python package, and
a valid KNITRO license. **Students can request a free one-year KNITRO
license** through Artelys' academic program on the same page. The package
checks for KNITRO once per Python session, on the first call.

The Julia dependencies (KNITRO.jl, ForwardDiff, Optim, ...) are declared in
`tvbounds/juliapkg.json` and installed by `juliacall` into its own Julia
environment on first use. KNITRO.jl downloads the KNITRO library through
`KNITRO_jll`, whose newest release may be a KNITRO version that your license
does not cover (KNITRO licenses are bound to a release date; a license for
KNITRO 15.1 rejects KNITRO 16 with error `-520`). In that case pin the
release your license covers before the first call, for example

```bash
python -m juliapkg add KNITRO_jll --uuid 0e6b36f8-8e90-4eb5-b54e-06f667ea875c --version "=15.1.0"
```

or point KNITRO.jl at a local installation through the `KNITRODIR`
environment variable, as described in the KNITRO.jl documentation.

## Quick tour

The commands below reproduce, in brief, what the manual develops in
detail.

### Randomized experiments with attrition

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

## Reproducibility

Every estimator is deterministic given its inputs, and the bootstrap in
`tvbounds_attrition()` is reproducible through its `seed` argument: the
same seed gives the same resamples, hence the same standard errors,
percentile bands and summary measures. The seeded stream is local to the
call, so it never disturbs the state of `numpy.random` or of any other
generator in your session.

## Citation

If you use `tvbounds`, please cite:

Palomba, F. (2026). "Sensitivity Analysis in Population Shares." Working
paper.

## License

MIT © Filippo Palomba. See `LICENSE`.
