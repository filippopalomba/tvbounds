# Generated documentation for the public functions of tvbounds.
# DO NOT EDIT BY HAND: regenerate with `python tools/rd2py.py`.

DOCS = {
    'tvbounds_attrition': """\
Sensitivity bounds for randomized experiments with attrition

Description
-----------
Computes sensitivity bounds on the average treatment effect for the
always-observed subpopulation of a randomized experiment with attrition,
following Palomba (2026). Writing P_0 for the
distribution of the data, the estimand is

    tau_0 := E_{P_0}[Y(1) - Y(0) | S(0) = 1, S(1) = 1],

the average treatment effect on the units that respond under either arm.
The bounds are indexed by a budget delta in [0, 1],
supplied through `delta`, which caps the total variation distance between
the outcome distribution of the compliers (units that respond only under
treatment) and that of the always-observed units,
TV(P_C || P_AO) <= delta.
At delta = 0 the two distributions coincide and the
bounds collapse to the baseline difference in means among respondents (the
estimand under missingness completely at random); at
delta = 1 the restriction is vacuous and they equal the
trimming bounds of Lee (2009). Optionally computes a nonparametric
bootstrap (with clustering) for standard errors and a percentile
confidence band, and covariate-pooled bounds that allocate a single budget
optimally across covariate cells.

Usage
-----
tvbounds_attrition(
  data,
  outcome,
  treatment,
  response,
  covariates = None,
  delta = np.linspace(0, 1, 101),
  neighborhood = "tv",
  bootstrap = True,
  B = 1000,
  cluster = None,
  level = 0.95,
  min_obs = 5,
  seed = None,
  verbose = False
)

Arguments
---------
data
    A data frame containing the columns named by `outcome`,
    `treatment`, `response`, and (optionally) `covariates` and `cluster`.
outcome
    String; name of the numeric column holding the outcome
    Y. May be `NaN` for non-respondents (and for the occasional
    respondent with item non-response, which is dropped from the outcome
    samples).
treatment
    String; name of the binary 0/1 column holding the
    treatment D (1 = treated).
response
    String; name of the binary 0/1 column holding the response
    indicator S (1 = outcome observed).
covariates
    Optional list of column names to stratify
    on, forming the discrete covariate X; cells are formed by their
    interaction. The covariate columns must be free of missing values —
    recode missing values into an explicit category first. Default `None`
    (no stratification).
delta
    Numeric vector of budget values
    delta in [0, 1] at which the bounds are
    evaluated; sorted and de-duplicated internally. The baseline
    (delta = 0) and Lee (delta = 1)
    endpoints are always computed internally even when absent from `delta`.
    Default `np.linspace(0, 1, 101)`.
neighborhood
    Either `"tv"` (total variation, default) or
    `"contamination"`; see the Neighborhoods section.
bootstrap
    Logical; compute bootstrap standard errors and the
    percentile confidence band. Default `True`.
B
    Number of bootstrap replications B. Default `1000`.
cluster
    Optional string; name of a cluster identifier column.
    When supplied, the bootstrap resamples whole clusters with
    replacement. Default `None` (units resampled independently).
level
    Confidence level of the percentile band,
    1 - alpha in the notation of the Inference section.
    Default `0.95`.
min_obs
    Minimum number of observed outcomes per arm for a
    covariate cell to be retained. Default `5`.
seed
    Optional integer seed for the bootstrap. When non-`None`,
    the random-number-generator state is set locally and restored on exit,
    so the call has no side effect on the caller's RNG. Default `None`.
verbose
    Logical; emit progress messages. Default `False`.

Value
-----
An object of class `TVBounds`: an object
with attributes

* `application`: `"attrition"`.
* `bounds`: data frame with one row per requested budget and columns
  `delta`, `lower`, `upper`, holding the sensitivity bounds
  tau-lower(delta) and
  tau-upper(delta) (or
  their pooled covariate counterparts
  tau_X-lower(delta)
  and
  tau_X-upper(delta)
  when `covariates` is supplied), plus, when `bootstrap = True`,
  `lower_se`, `upper_se`, `ci_lower`, `ci_upper` (outer percentile
  band endpoints at `level`).
* `point`: the estimate of tau_0
  at delta = 0, where the bounds collapse to the
  difference in means among respondents
  tau_MCAR(P_0) (with `covariates`,
  to its covariate-weighted analogue).
* `n`: number of rows of `data` used.
* `neighborhood`, `level`, `B`, `estimand_label`, `call` as documented
  in the package overview (`level` and `B` are `NaN` without
  inference).
* `details`: dict with `p_star` (the complier share pi,
  which also sets the trimming mass), `response_rate` (the arm-specific
  response rates (r_0-hat, r_1-hat))
  and `n_by_arm` / `n_respondents_by_arm` (named, control and treated),
  `lee` (dict with the Lee-endpoint `lower` / `upper`,
  tau_Lee-lower and
  tau_Lee-upper, of the estimated
  specification), `lee_nocov` (when covariates are used), `breakdown`
  (dict with the point-estimate breakdown budget
  delta_b at the reference value
  tau_star = 0 and, with inference, the
  confidence-band breakdown budget; `NaN` when censored beyond one),
  `pooled` (with covariates: per-stratum table, coverage, dropped cells,
  and the within-stratum reference curve `pw`,
  tau_X^pw-lower(delta)
  and
  tau_X^pw-upper(delta),
  under total variation), `n_clusters` (with `cluster`), and `boot`
  (replicate counts, width standard errors, and — when
  memory-reasonable — the matrices of bound draws).

Setup and estimand
------------------
Let D in {0, 1} be the binary treatment and, for
d in {0, 1}, let Y(d) be the potential outcome
and S(d) in {0, 1} the potential response
indicator. Their realized counterparts are

    Y = D Y(1) + (1 - D) Y(0),  S = D S(1) + (1 - D) S(0),

so that the observed data are (YS, S, D): the outcome is recorded
only when S = 1. The response pattern (S(0), S(1)) partitions
the population into always-observed units
(S(0) = 1, S(1) = 1), never-observed units
(S(0) = 0, S(1) = 0), compliers (S(0) = 0, S(1) = 1) and
defiers (S(0) = 1, S(1) = 0); the estimand
tau_0 is the average treatment
effect on the first of these groups.

Two assumptions are maintained. Random assignment enters as
(S(0), S(1)) independent of (Y(0), Y(1)),
labelled (MCAR), and the monotonicity condition of Lee (2009),
S(1) >= S(0) almost surely, labelled (Mono) —
treatment never causes a unit that would respond under control to attrit —
rules out defiers. Under (Mono) the outcome distribution of the observed
treated, P_T, is a mixture of the complier and
always-observed outcome distributions,

    P_T = pi P_C + (1 - pi) P_AO,  pi = 1 - r_0 / r_1,

where r_1 := P_0[S = 1 | D = 1]
and r_0 := P_0[S = 1 | D = 0] are
the arm-specific response rates and pi is the complier share. The
always-observed control mean is identified,
mu^AO(0) = E_{P_0}[Y | D = 0, S = 1],
whereas the always-observed treated mean
mu^AO(1) is only partially
identified; the bounds on tau_0
follow by subtracting the identified control mean.

The complier share is estimated by
pi-hat = max{1 - r_0-hat / r_1-hat, 0}
and returned as `details["p_star"]`, and the estimated response rates
(r_0-hat, r_1-hat) as
`details["response_rate"]`. A negative unconstrained estimate of pi is
sampling noise under (Mono) and is projected to zero, in which case the
bounds collapse to the baseline difference in means at every budget.

Neighborhoods
-------------
Two robustness sets are available through `neighborhood`. Both are indexed
by the budget delta and both restrict the unobserved
complier outcome distribution P_C relative to
the unobserved always-observed outcome distribution
P_AO:

* `"tv"` (default): the total variation neighborhood of the paper,
  TV(P_C || P_AO) <= delta,
  so the two distributions may disagree on at most a
  delta fraction of their mass.
* `"contamination"`: a one-sided strengthening in which
  P_C lies in the contamination neighborhood
  of P_AO (Huber 1964), that is
  P_C = (1 - delta) P_AO + delta R
  for some distribution R, equivalently
  P_C >= (1 - delta) P_AO
  as measures: a (1 - delta)-share of the compliers has
  outcomes distributed exactly like the always-observed units, and only the
  remaining delta-share may differ arbitrarily.

Under total variation the rescaling identity
TV(P_C || P_T) = (1 - pi) TV(P_C || P_AO)
recenters the restriction on the identified distribution
P_T, so that the candidate complier distributions
Q, among which P_C lies, range over the robustness set

    Q_C(delta) := {Q in Delta(Y) : TV(Q || P_T) <= (1 - pi) delta, pi Q <= P_T},

where Delta(Y) denotes the distributions on the
outcome space and the second restriction is the mixture structure of the
observed treated arm. The resulting sensitivity bounds
tau-lower(delta) and
tau-upper(delta) on
tau_0 are available in closed
form as trimmed means of P_T. Writing
F^{-1}_{P_T} for the quantile
function of the observed treated outcomes and

    s_L(delta) := F^{-1}_{P_T}(pi delta),  s_U(delta) := F^{-1}_{P_T}(1 - (1 - pi) delta),

the upper bound is

    tau-upper(delta) = E_{P_T}[Y 1{s_L(delta) < Y < s_U(delta)}] + (1 / (1 - pi)) E_{P_T}[Y 1{Y >= s_U(delta)}] - mu^AO(0),

and the lower bound is obtained symmetrically, trimming at
t_L(delta) := F^{-1}_{P_T}((1 - pi) delta)
and
t_U(delta) := F^{-1}_{P_T}(1 - pi delta).

Under contamination, combining the same mixture identity with
P_C >= (1 - delta) P_AO
pins the density
w := dP_C / dP_T
between (1 - delta) / (1 - delta pi)
and 1 / pi, and the bounds are again trimmed means of
P_T, now with effective trimming mass
delta pi. The contamination neighborhood is contained
in the total variation one at every budget, so its bounds are weakly
tighter, and the two families share the same endpoints: the baseline at
delta = 0 and, at delta = 1, the Lee
(2009) bounds
tau_Lee-lower = E_{P_T}[Y | Y <= y_{1 - pi}] - mu^AO(0)
and
tau_Lee-upper = E_{P_T}[Y | Y >= y_{pi}] - mu^AO(0),
where y_u := F^{-1}_{P_T}(u).

Both bound families are monotone in the budget by construction: the
robustness sets are nested in delta.

Covariates
----------
When `covariates` is supplied, units are stratified on the interaction of
the covariate columns, which plays the role of a discrete covariate
X with support calX. Cells with fewer than
`min_obs` observed outcomes in either arm are dropped with a warning, and the
retained cells are weighted by their control-respondent shares, which
under (Mono) are the covariate distribution of the always-observed
population,
P_{X | D = 0, S = 1} = P_{X | AO}.
Within a cell the complier share pi(x) and the observed
treated outcome distribution
P_T(x) are identified, and the cell-level
construction is the one above.

Two ways of spending the budget across cells are distinguished. The
within-stratum ("pointwise") restriction imposes
TV(P_C(x) || P_AO(x)) <= delta
in every cell separately, giving one robustness set
Q_C^pw(delta; x)
per cell, whereas the pooled restriction caps only the average departure,

    integral over calX of TV(P_C(x) || P_AO(x)) dP_{X | AO}(x) <= delta,

and so allows heterogeneity across cells inside the single robustness set
Q_{C,X}(delta).
The pooled restriction is the weaker of the two, so
tau_X-lower(delta) <= tau_X^pw-lower(delta)
and
tau_X^pw-upper(delta) <= tau_X-upper(delta),
with equality at delta = 0 and at
delta = 1, where both collapse to the covariate Lee
(2009) bounds
tau_{Lee,X}-lower and
tau_{Lee,X}-upper.

For `neighborhood = "tv"` the reported bounds are the pooled (joint)
bounds tau_X-lower(delta)
and tau_X-upper(delta):
the budget allocation
t : calX -> R_+ subject to
E_{P_T}[t(X)] <= (1 - pi) delta
is solved exactly by a greedy fill over the stratum value functions
V(t; x), which are piecewise linear in the cell budget and
concave for the upper bound and convex for the lower one. The
within-stratum bounds
tau_X^pw-lower(delta)
and
tau_X^pw-upper(delta)
are returned in `details["pooled"]["pw"]` for reference. For
`neighborhood = "contamination"` the reported bounds impose the common
budget `delta` within every retained cell and aggregate; they remain
weakly inside the total variation bounds at every budget.

Inference
---------
The bootstrap resamples the full observation
W = (YS, S, D), together with the covariates, with
replacement — whole clusters when `cluster` is supplied — and recomputes
the entire bounds curve on each replicate, yielding
tau-upper-hat^(b)(delta),
b = 1, ..., B. Each replicate therefore redraws the
arm-specific response rates and hence pi-hat, so the
reported standard errors carry the estimation uncertainty in
pi, which the naive variance
sigma^2_naive(delta)
omits by treating pi-hat as known; in the paper's
influence function
psi_full(W; delta) this uncertainty
is the term
varkappa(delta) psi_{pi}(W). The reported
`upper_se` is the standard deviation of the draws
tau-upper-hat^(b)(delta)
across replicates, that is the paper's
sigma_boot(delta) divided
by sqrt(n) for a sample of size n, and
`lower_se` is its counterpart for the lower bound. The reported confidence
band is the percentile band: writing alpha for the value of
`1 - level`, `ci_lower` is the alpha / 2 quantile of the
lower-bound draws and `ci_upper` the
1 - alpha / 2 quantile of the upper-bound draws, the
outer envelope of the identified set. Replicates on which the bounds
cannot be computed are dropped and counted (a warning reports their
number).

References
----------
Palomba, F. (2026). "Sensitivity Analysis in Population Shares." Working
paper.

Lee, D. S. (2009). "Training, Wages, and Sample Selection: Estimating
Sharp Bounds on Treatment Effects." *Review of Economic Studies*, 76(3),
1071-1102.

Huber, P. J. (1964). "Robust Estimation of a Location Parameter."
*Annals of Mathematical Statistics*, 35(1), 73-101.

See Also
--------
`tvbounds_plot()` and `tvbounds_summary()` for reporting.

Examples
--------
import numpy as np
import pandas as pd
from tvbounds import tvbounds_attrition

rng = np.random.default_rng(123)
n = 400
d = rng.binomial(1, 0.5, n)
s = rng.binomial(1, np.where(d == 1, 0.9, 0.7))
y = np.where(s == 1, rng.normal(loc = 0.3 * d), np.nan)
x = rng.binomial(1, 0.5, n)
dat = pd.DataFrame({"y": y, "d": d, "s": s, "x": x})

## Total variation bounds with a small bootstrap
fit = tvbounds_attrition(dat, outcome = "y", treatment = "d",
  response = "s", delta = np.linspace(0, 1, 11), B = 50, seed = 1)
fit.bounds
fit.details["lee"]

## Contamination neighborhood, no inference: weakly tighter bounds
fit_c = tvbounds_attrition(dat, outcome = "y", treatment = "d",
  response = "s", delta = np.linspace(0, 1, 11),
  neighborhood = "contamination", bootstrap = False)
(fit_c.bounds["lower"] >= fit.bounds["lower"] - 1e-12).all()

## Covariate-pooled bounds
fit_x = tvbounds_attrition(dat, outcome = "y", treatment = "d",
  response = "s", covariates = "x", delta = np.linspace(0, 1, 11),
  bootstrap = False)
""",
    'tvbounds_counterfactual': """\
Sensitivity bounds for counterfactual predictions in structural models

Description
-----------
Computes the lower and upper sensitivity bounds
k_lower(delta) and
k_upper(delta) on a counterfactual
E_P[g(U; theta)], when the distribution
P of the latent variables U ranges over a divergence
neighborhood of the simulated baseline P_* with budget
`delta`, and the structural parameter theta ranges over the
values compatible with the moment conditions
E_P[m(U; theta)] in
M(rho), following Palomba (2026) and Christensen and Connault (2023).
This is the only function in the package that supports general
phi-divergences beyond total variation (spelled out at
first use; `"TV"` below) and contamination.

Usage
-----
tvbounds_counterfactual(
  moments,
  d,
  theta_lb,
  theta_ub,
  delta,
  divergence = "KL_chi2",
  side = "both",
  U = None,
  M = 50000,
  u_dim = None,
  gamma = None,
  gradient = None,
  theta_init = None,
  control = tvbounds_control(),
  seed = None,
  verbose = True
)

Arguments
---------
moments
    The moment/counterfactual function: a length-2
    sequence of strings `(file, fname)`, a single string naming a Julia
    function, or a Python function. See Details.
d
    Integer, the number of moment conditions
    d_m (columns of `G`).
theta_lb, theta_ub
    Numeric vectors of equal length: the box for
    the structural parameter theta over which the outer
    problems optimize. Their common length fixes the dimension of
    theta.
delta
    Numeric vector of strictly positive budgets
    delta (the sensitivity parameter of the neighborhood),
    without duplicates. For the total-variation family the budget must lie
    in (0, 1]; for `"KL_chi2"`, `"KL"`, and `"chi2"` any
    positive value is allowed.
divergence
    Divergence keyword; one of `"KL_chi2"` (default),
    `"KL"`, `"chi2"`, `"TV"`, `"TVmix"`, `"TVmixC"`, `"TVac"`. See
    Details.
side
    `"both"` (default), `"lower"`, or `"upper"`: which bound
    problems to solve at each budget.
U
    Optional `M x u_dim` numeric matrix of latent draws
    U^(1), ..., U^(M), one per row. When
    `None`, scrambled-Halton uniforms are generated (see Details) and
    `M`, `u_dim` are required; when supplied, `M` and `u_dim` are taken
    from its dimensions.
M
    Integer, the number M of simulated draws when
    `U = None` (default `50000`, the setting of the paper).
u_dim
    Integer, the dimension d_z of the
    latent draw when `U = None` (at most 15).
gamma
    Optional dict: an arbitrary payload forwarded to the
    moments function (as `obj.gamma` for Julia moments, as the third
    argument for Python moments).
gradient
    How to differentiate the moments with respect to
    theta in the outer optimization: `None` (default; automatic
    differentiation for Julia moments, finite differences for Python
    moments), the string `"fd"` (finite differences), the name of a
    Julia function `jac(theta, U, obj)`, or a Python function
    `f(theta, U, gamma)`. A user-supplied gradient must return
    either the stacked `(M*(d+1)) x l` Jacobian of `(K, G)` (K rows
    first, then `G` in column-major order) or a dict with components
    `K` (`M x l`) and `G` (`M x d x l`), where `l = len(theta_lb)`.
theta_init
    Optional numeric vector, the initial structural
    parameter theta for the outer optimization (defaults to
    the midpoint of the box). Set it to the baseline estimate of the
    model: the reported baseline `point` is the plug-in counterfactual
    k(theta; P_*) = E_{P_*}[g(U; theta)]
    at `theta_init`, and is only returned when `theta_init` is supplied.
control
    A dict created by `tvbounds_control()` with solver
    tuning options.
seed
    Optional integer seed for the scrambled-Halton draws.
verbose
    Logical; print solver progress (default `True`).

Details
-------
**Setup.** Let U in Z collect the latent variables
of the structural model (taste shocks, unobserved heterogeneity,
measurement errors) and let P_* be the baseline distribution
the econometrician postulates for them. At a structural parameter
theta in Theta, a distribution P is
compatible with the model when
E_P[m(U; theta)] in
M(rho), with m the moment function and
M(rho) the moment constraint set; the object of
interest is the counterfactual
E_P[g(U; theta)], the expectation of a
known criterion g that is linear in P at fixed
theta. The robustness set collects the distributions that
are compatible with the model and within budget of the baseline,

    P_phi(theta; rho, P_*, delta) := {P : D_phi(P || P_*) <= delta,
    E_P[m(U; theta)] in M(rho)},

where D_phi(P || P_*) is the
phi-divergence selected by `divergence` and
delta the budget, and the reported bounds are the nested
extrema

    k_lower(delta) = inf_theta inf_P E_P[g(U; theta)],
    k_upper(delta) = sup_theta sup_P E_P[g(U; theta)],
    both over P in P_phi(theta; rho, P_*, delta).

The optimization over P at a fixed theta is the
inner problem, solved in its dual form; the optimization over
theta is the outer problem.

**Correspondence between the paper and the code.** The paper writes the
counterfactual integrand as g and the moment function as
m. The interfaces below fill an array named `K` with the values
of g and an array named `G` with the values of m: read
`K` as g and `G` as m throughout. The solver takes the
moment conditions in centered equality form,
E_P[m(U; theta)] = 0, so a nonzero
target rho is absorbed by centering the moment function. The
argument `d` is the number of moment conditions,
d_m; the common length of `theta_lb` and `theta_ub`
is the dimension of theta; and `u_dim` is the dimension
d_z of a single latent draw.

**Moments specification.** `moments` can be supplied in three forms:

1. a length-2 sequence of strings `(file, fname)`: `file` is the path
   of a Julia source file that is included into the session, and
   `fname` the name of a function defined there (at the top level, in
   `Main`) with the in-place signature
   `moments!(K, G, theta, U, obj)`. The function must fill `K` (an
   `M`-vector holding the counterfactual values
   g(U^(j); theta)) and `G` (an `M x d` matrix
   holding the moment functions
   m(U^(j); theta), one row per draw) and may
   read the user payload as `obj.gamma` (a Python dict arrives in Julia as
   a dictionary keyed by symbols, so `obj.gamma[:name]`);
2. a single string naming a Julia function with the same signature
   that is already defined in the session;
3. a Python function `f(theta, U, gamma)` returning a dict with
   components `K` (numeric array of length `M`) and `G` (numeric
   `M x d` array). This path is **much slower** (every objective
   evaluation crosses the Python/Julia boundary), and because ForwardDiff
   cannot differentiate through Python code the outer optimization needs
   either a user-supplied `gradient` or finite differences (the
   default for this path).

For Julia moments the outer envelope-theorem gradient differentiates
the moments by automatic differentiation (ForwardDiff), so the Julia
function should be written generically in the element type of
`theta`; pass `gradient = "fd"` for a non-generic function.

**Latent draws.** `U` is the `M x u_dim` matrix of latent draws
U^(1), ..., U^(M) that discretizes the
baseline distribution P_*, row `j` holding the draw
U^(j). When `U = None` the package generates `M`
scrambled-Halton points in the unit cube
(0, 1)^d_z of dimension `u_dim` (Owen,
2017), seeded by `seed`; the moments function is then responsible for
mapping the uniform coordinates into baseline draws (e.g. through
quantile transforms). The Halton generator supports `u_dim <= 15`;
supply `U` directly for higher-dimensional draws.

**Divergences.** `divergence` selects the entropy function
phi whose divergence
D_phi(P || P_*) defines the neighborhood;
the budget grid `delta` must be strictly positive, and must lie in
(0, 1] for the total-variation family:

* `"KL_chi2"` (default): the hybrid Kullback-Leibler/chi-square
  divergence of Christensen and Connault (2023); any `delta > 0`.
* `"KL"`, `"chi2"`: the pure Kullback-Leibler entropy
  phi_KL(s) = s log s -
    s + 1 and the Pearson chi-square entropy; any `delta > 0`.
* `"TV"`: total variation,
  phi_TV(s) = |s - 1| / 2, for
  which D_phi_TV(P || P_*) = TV(P, P_*). Its recession
  function is finite,
  phi_TV^inf(1) = 1/2, so
  the dual keeps the pointwise constraint
  zeta + eta/2 >= sup_u {g(u; theta) -
    lambda' m(u; theta)}, which discretizes into `M` linear feasibility
  constraints (one per draw).
* `"TVmix"`: total variation intersected with the mixture
  (contamination) constraint P >= kappa * P_*,
  that is, the perturbed distribution contains the baseline as a mixing
  component with weight kappa = 1 - delta (or
  `control["tvmix_kappa"]`); solved in the exact reduced form (Palomba,
  2026).
* `"TVmixC"`: the literal dual of the same program, kept as a
  cross-check of `"TVmix"`; it is slower, and it is the only mode
  supporting a mixing weight
  kappa < 1 - delta.
* `"TVac"`: total variation restricted to distributions absolutely
  continuous with respect to the baseline,
  P << P_*.

The kinked total-variation conjugates
phi_TV^* are Huber-smoothed and the
per-draw maxima in the dual objective log-sum-exp-smoothed (scales in
`tvbounds_control()`); both smoothings lie above the exact functions,
so computed bounds are outward-conservative (wider, never narrower) at
order `1e-3`.

**Optimization.** For each budget (in increasing order) and each
side, the solver runs `control["maxsolves"]` multi-start outer
optimizations over theta (KNITRO, or Optim.jl when
`control["use_optim"] = True`), warm-started at the previous budget's
optimum; the reported bound is the inner (dual) value re-solved at
the best candidate, which makes the bound curves monotone in the
budget by construction. A degenerate box (`theta_lb == theta_ub`)
skips the outer optimization and reports the bounds at the fixed
theta supplied through `theta_init`. Failed budgets are
reported as `NaN` (for `"TVmix"` an `NaN` typically signals an
infeasible moment condition at every theta in the box,
that is an empty robustness set at that budget).

Value
-----
An object of class `TVBounds`:
an object with the fields described in the package overview, in
particular `bounds` (data frame with columns `delta`, `lower`,
`upper`, holding delta,
k_lower(delta) and
k_upper(delta); no
standard-error or confidence-band columns, since this application
currently carries no inference), `point` (the plug-in counterfactual
E_{P_*}[g(U; theta)] at
`theta_init`, or `NaN` when `theta_init` was not supplied),
`divergence`, and `details`, a dict with:

`solver`
    data frame of per-budget diagnostics: outer
    multi-start flags, inner KNITRO status codes, and timings for
    each side (`-999` marks entries that do not apply, e.g. outer
    flags in fixed-theta mode).
`theta_lower`, `theta_upper`
    `l x length(delta)` matrices
    of the outer-optimal structural parameters theta at
    each budget, for the lower and the upper bound respectively.
`M`, `u_dim`
    the number M of simulated draws and
    their dimension d_z.
`theta_lb`, `theta_ub`, `theta_init`, `fixed_theta`
    the
    parameter box, the initial point, and whether the box was
    degenerate.
`control`
    the resolved control dict, including the option
    files actually used.
`moments`
    a short description of the moments
    specification.

KNITRO requirement
------------------
This function relies on Julia (>= 1.9) and on the commercial
**Artelys KNITRO** solver, accessed through the 'juliacall' package
and KNITRO.jl. A valid KNITRO license is required (free academic
trials are available from Artelys at
https://www.artelys.com/solvers/knitro/). On the first call in each
Python session the package initializes Julia, instantiates its Julia
environment, and checks that KNITRO.jl loads and that a KNITRO solver
context can be created (which exercises the license); a one-time
message reports the outcome, and the call stops with installation and
license guidance when the check fails. The check runs once per Python
session.

References
----------
Palomba, F. (2026). "Sensitivity Analysis in Population Shares."
Working paper.

Christensen, T. and B. Connault (2023). "Counterfactual Sensitivity
and Robustness." *Econometrica*, 91(1), 263-298.

Owen, A. B. (2017). "A randomized Halton algorithm in R."
arXiv:1706.02808.

See Also
--------
`tvbounds_control()`, `tvbounds_plot()`, `tvbounds_summary()`

Examples
--------
# The full solver requires Julia and a licensed KNITRO installation,
# so a complete run cannot be executed without them:
## Not run:

import tvbounds
from tvbounds import tvbounds_counterfactual, tvbounds_control

# Toy model (shipped with the package): U ~ Uniform(0, 1), one moment
# condition m(U; theta) = U - theta and counterfactual g(U; theta) = U,
# so under the "TVmix" neighborhood the bounds equal the endpoints of
# the theta box. In the Julia file, m is written into G and g into K.
toy = tvbounds.julia_file("examples", "toy.jl")
fit = tvbounds_counterfactual(
  moments  = (toy, "tvb_toy_moments!"),
  d        = 1,
  theta_lb = 0.4, theta_ub = 0.6,
  delta    = [0.5, 1],
  divergence = "TVmix",
  M = 500, u_dim = 1,
  theta_init = 0.5,
  control = tvbounds_control(maxsolves = 2),
  seed = 1234)
fit.bounds
## End(Not run)

# The control constructor is pure Python and always available:
from tvbounds import tvbounds_control
tvbounds_control(maxsolves = 3)["maxsolves"]
""",
    'tvbounds_riv': """\
Sensitivity bounds for recentered instrumental variables

Description
-----------
Computes sensitivity bounds on a recentered (formula) instrumental-variables
estimate when the postulated distribution of the shocks is allowed to vary
within a total variation or a contamination neighborhood of the baseline
assignment distribution P_*, as in Palomba (2026).  The leading
use case is the recentered instruments of Borusyak and Hull (2023), whose
validity rests on a researcher-postulated distribution for the shock
process: the bounds quantify how far the estimate can move when up to a
fraction `delta` of that postulated probability mass is misspecified.

Usage
-----
tvbounds_riv(
  y,
  x,
  z,
  Fmat,
  p = None,
  controls = None,
  delta = np.linspace(0, 1, 501),
  neighborhood = "tv",
  tau_star = 0,
  verbose = False
)

Arguments
---------
y
    Numeric n-vector, the outcome y_i.
x
    Numeric n-vector, the endogenous regressor x_i.
z
    Numeric n-vector, the realized (un-recentered) candidate
    instrument: `z[i]` is the formula of unit `i` evaluated at the realized
    shocks, z_i = f_i(v; w).
Fmat
    Numeric n x S matrix of counterfactual instrument draws:
    `Fmat[i, s]` is the formula of unit `i` evaluated at the `s`-th
    counterfactual shock configuration, f_i(v^(s); w).
    These are the draws from the postulated assignment process used for
    recentering (in Borusyak and Hull (2023), permuted or re-simulated shock
    allocations).
p
    Optional numeric S-vector holding the probabilities that the
    postulated assignment distribution P_* attaches to the
    columns of `Fmat`, that is to the configurations
    v^(1), ..., v^(S).  Defaults to `None`,
    meaning the uniform distribution `1/S` on each draw, which is the
    postulated assignment distribution of Borusyak and Hull (2023).  Must be
    nonnegative and sum to one.
controls
    Optional numeric matrix or data frame of control
    variables (n rows) to be partialled out of `y`, `x`, `z`, and each
    column of `Fmat`.  A constant is always included.  Default `None`.
delta
    Numeric vector of sensitivity budgets
    delta in [0, 1] at which the bounds are traced.
    Default `np.linspace(0, 1, 501)`.
neighborhood
    Either `"tv"` (the total variation ball
    P^FI_TV(delta),
    the default) or `"contamination"` (the contamination neighborhood
    P^FI_cont(delta)).
tau_star
    Numeric scalar reference value tau_star
    for the breakdown budget delta_b(tau_star):
    the smallest budget at which the bounds cover
    tau_star.  The default `0` gives the breakdown budget
    for the sign of beta.
verbose
    Logical; if `True`, print progress messages.  Default
    `False`.

Details
-------
The exercise is conducted conditionally on the realized sample, so every
bound is a deterministic function of the data and of the budget `delta`;
accordingly, and by design (as in the paper), **no standard errors or
confidence bands are produced** for this application.

**The model and the recentered estimate.**  For units
i in [n] the structural equation is

    y_i = beta * x_i + eps_i,

with beta the parameter of interest, y_i the
outcome, x_i the endogenous regressor and
eps_i the unobserved residual.  Let v denote
the vector of exogenous shocks, taking values in a space
V, let w collect predetermined covariates, and
let f_i( . ; w) : V -> R
be the known formula that maps a shock configuration into the instrument of
unit i, so that z_i := f_i(v; w) is the
candidate instrument at the realized shocks.  For a distribution P
on V, the expected instrument and the recentered
instrument are

    mu_i(P) := E_P[f_i(v; w) | w] = Int_V f_i(v'; w) dP(v'),
    ztilde_i(P) := z_i - mu_i(P).

Borusyak and Hull (2023) postulate an assignment distribution
P_* for the shocks and recenter at it.

**The two criterion functions.**  The formula enters only through the two
sample aggregates

    g_y( . ) := sum_i y_i f_i( . ; w),   g_x( . ) := sum_i x_i f_i( . ; w),

whose recentered values are the reduced form
G_y(P) := g_y(v) - E_P[g_y] and
the first stage
G_x(P) := g_x(v) - E_P[g_x].
The estimate at a candidate assignment distribution is the ratio

    betahat(P) := G_y(P) / G_x(P)
    = sum_i ztilde_i(P) y_i / sum_i ztilde_i(P) x_i,

and the reported estimate is
betahat_* = betahat(P_*).

**How the arguments encode the shock space.**  The package represents
P_* by `S` shock configurations
v^(1), ..., v^(S): entry `Fmat[i, s]` holds
f_i(v^(s); w), the formula of unit `i` at the `s`-th
configuration, `p[s]` holds the probability that P_* assigns to
that configuration, and `z[i]` holds z_i = f_i(v; w).
The shock space is therefore taken to be the finite set
V = {v^(1), ..., v^(S)},
over which the candidate distributions P range.

**Neighborhoods.**  The bounds report the range of
betahat(P) as P ranges over the chosen
neighborhood of P_*, intersected with the set
P_nonzero := {P in Delta(V) : G_x(P) != 0} of distributions at which
the ratio is defined:

* `neighborhood = "tv"` (the default): the total variation ball

      P^FI_TV(delta) := {P in Delta(V) : TV(P, P_*) <= delta},

  which delivers the bounds
  beta_TV_lower(delta) and
  beta_TV_upper(delta).  They
  are computed from the closed form the paper gives for the criterion
  bounds
  g_TV_lower(h; delta)
  and
  g_TV_upper(h; delta),
  the smallest and the largest value of
  E_P[h] over the ball: writing q_h for
  the quantile function of a criterion h under P_*,
  and h_sup, h_inf for its
  extremes over V,

      g_TV_upper(h; delta) = delta * h_sup + Int_delta^1 q_h(u) du,
      g_TV_lower(h; delta) = delta * h_inf + Int_0^(1 - delta) q_h(u) du,

  that is, the baseline mean of h once a tail of mass
  delta has been trimmed and relocated to the most (least)
  favorable configuration.  Applied to the one-parameter family of criteria
  g_b := g_y - b * g_x, for which
  betahat(P) = b holds exactly when
  E_P[g_b] = g_b(v), each bound is the
  unique root in b of a strictly monotone function and is located
  by bracketed root finding;
* `neighborhood = "contamination"`: the contamination neighborhood

      P^FI_cont(delta) := {P in Delta(V) : P = delta * R + (1 - delta) * P_*,
      R in Delta(V)},

  in which the contamination share is the budget delta itself,
  delivering the bounds
  beta_cont_lower(delta)
  and beta_cont_upper(delta).
  These are attained by contaminating distributions R degenerate at
  a single shock configuration, so they are obtained by enumerating the `S`
  configurations rather than by optimization.

Every member of
P^FI_cont(delta)
lies in
P^FI_TV(delta), so
the contamination bounds are weakly tighter at every budget.

**First-stage breakdown.**  Once the budget is large enough that some
distribution in the neighborhood makes the recentered first stage
G_x(P) vanish, betahat(P) is no
longer well defined over the whole neighborhood and the identified set is
the entire real line.  The smallest such budget is the first-stage
breakdown budget,
delta^TV_FS for the total
variation ball and
delta^cont_FS for the
contamination neighborhood; since the contamination neighborhood is the
smaller of the two,
delta^TV_FS <= delta^cont_FS.  The
one for the neighborhood in use is reported in `details["delta_fs"]`.
Following the paper, the infimum over an empty set is set to 1, so a
reported value of 1 carrying `delta_fs_censored = True` means that the
first stage never breaks down over the budget range, not that breakdown
occurs at 1.  Rows of `bounds` at budgets where the bounds are vacuous
carry `NaN`.

**Controls.**  When `controls` is supplied, `y`, `x`, `z`, and every column
of `Fmat` are residualized on the controls and a constant (when
`controls = None`, on the constant alone) before the bounds are computed.
By the Frisch--Waugh--Lovell theorem this leaves the just-identified
two-stage least squares coefficient on `x` unchanged, because the
projection matrix is idempotent; the baseline estimate then replicates the
estimate from the full regression with controls.

Value
-----
An object of class `TVBounds`, an object with
attributes

* `application`: `"riv"`.
* `bounds`: data frame with one row per budget and columns `delta`,
  `lower`, `upper`.  Rows at budgets where the neighborhood contains a
  distribution collapsing the recentered first stage carry `NaN` (the
  bounds are vacuous there).  No standard error or confidence interval
  columns are attached: this application carries no inference by design.
* `point`: the reported recentered IV estimate
  betahat_* =
    betahat(P_*), which is the common value of the two bounds at
  `delta = 0`.
* `n`: number of observations.
* `neighborhood`: the neighborhood used.
* `level`, `B`: `NaN` (no inference).
* `estimand_label`: `"IV coefficient"`.
* `call`: the matched call.
* `details`: a dict with

  * `criteria`: the reduced criterion object consumed by the internal
    kernels, holding the realized values g_y(v) and
    g_x(v), the two S-vectors
    g_y(v^(s)) and g_x(v^(s)),
    and the baseline aggregates
    E_{P_*}[g_y] and
    E_{P_*}[g_x]; bounds at additional
    budgets can be recomputed from it without touching the n x S design
    again.
  * `delta_fs`, `delta_fs_censored`: the first-stage breakdown budget
    for the neighborhood used,
    delta^TV_FS or
    delta^cont_FS, and
    whether it is censored at 1 (`True` when the first stage never
    breaks down on [0,1], so 1 is the empty-set convention rather
    than an actual breakdown).
  * `delta_fs_tv`, `delta_fs_cont`: both first-stage breakdown budgets,
    delta^TV_FS and
    delta^cont_FS (each
    carrying its `censored` attribute).
  * `delta_breakdown`, `delta_breakdown_censored`: the smallest budget
    delta_b(tau_star) at which the bounds
    for the chosen neighborhood cover tau_star,
    reported as 1 with `delta_breakdown_censored = True` when
    tau_star is never covered on [0,1].
  * `tau_star`: the reference value tau_star used.
  * `n_controls`: number of control columns partialled out (0 when
    `controls = None`).

References
----------
Palomba, F. (2026). "Sensitivity Analysis in Population Shares."
Working paper.

Borusyak, K. and Hull, P. (2023). "Nonrandom Exposure to Exogenous
Shocks." *Econometrica*, 91(6), 2155--2185.

Examples
--------
import numpy as np
import pandas as pd
from tvbounds import tvbounds_riv

# A small simulated formula-instrument design: S counterfactual shock
# configurations, a realized instrument, a first stage, and an outcome.
rng = np.random.default_rng(123)
n = 80
S = 50
Fmat = rng.normal(size = (n, S))              # Fmat[i, s] = f_i(v^(s); w)
z = Fmat.mean(axis = 1) + rng.normal(size = n) # realized instrument z_i = f_i(v; w)
x = z + 0.5 * rng.normal(size = n)            # endogenous regressor
y = 0.4 * x + rng.normal(size = n)            # outcome

fit = tvbounds_riv(y, x, z, Fmat, delta = np.linspace(0, 1, 21))
fit.point                                     # recentered IV estimate at P_*
fit.bounds.head()                             # bounds along the budget grid
fit.details["delta_fs"]                       # first-stage breakdown budget
fit.details["delta_breakdown"]                # breakdown budget for the sign

# Contamination neighborhood, with controls partialled out.
W = pd.DataFrame({"w1": rng.normal(size = n), "w2": rng.normal(size = n)})
fit_cont = tvbounds_riv(y, x, z, Fmat, controls = W,
                        delta = np.linspace(0, 1, 21),
                        neighborhood = "contamination")
fit_cont.point
""",
    'tvbounds_plot': """\
Plot sensitivity bounds against the budget

Description
-----------
Displays the bounds stored in a `TVBounds` object as functions of the
budget delta of the total-variation (or contamination, or
divergence) neighborhood, supplied through the `delta` column of
`x.bounds` and drawn on the horizontal axis. The region between the
lower bound path tau_lower(delta) and
the upper bound path tau_upper(delta) is
shaded, the outer confidence band (when the object carries one) is drawn
as a lighter ribbon delimited by dashed lines, a dashed horizontal line
marks the reference value tau_star, a point marks
the baseline estimate at delta = 0, and a dotted
vertical line marks the plug-in breakdown budget
delta_b_hat when it is interior to the
budget grid.

Usage
-----
x.plot(**kwargs)

tvbounds_plot(
  x,
  bands = True,
  baseline = True,
  breakdown = True,
  tau_star = 0,
  color = "#1F4E79",
  xlab = None,
  ylab = None,
  title = None,
  log_x = False,
  **kwargs
)

Arguments
---------
x
    A `TVBounds` object returned by `tvbounds_attrition()`,
    `tvbounds_counterfactual()`, or `tvbounds_riv()`.
**kwargs
    For `tvbounds_plot()`: currently unused, accepted for
    compatibility with the `.plot()` method. For the `.plot()`
    method: further arguments forwarded to `tvbounds_plot()`.
bands
    Logical; draw the outer confidence band when the object
    carries one (columns `ci_lower`/`ci_upper` of `x.bounds`). Sides
    without a band are drawn without one. Default `True`.
baseline
    Logical; mark the baseline point estimate, that is, the
    value of the estimand under P_*, at
    delta = 0. Ignored when `log_x = True`, since
    delta = 0 cannot be placed on a logarithmic axis.
    Default `True`.
breakdown
    Logical; draw a dotted vertical line, with a label, at
    the plug-in breakdown budget
    delta_b_hat when it is interior to the
    budget grid. Default `True`.
tau_star
    Reference value tau_star of the
    estimand against which robustness is judged; drawn as a dashed
    horizontal line (default `0`).
color
    Colour of the bounds, ribbons, and breakdown mark (default
    `"#1F4E79"`, the paper's blue).
xlab, ylab, title
    Axis labels and plot title. `None` (the default)
    uses `r"$\\delta$"` for the horizontal axis, the object's
    `estimand_label` for the vertical axis, and no title.
log_x
    Logical; use a logarithmic budget axis. Rows with
    `delta <= 0` are dropped with a warning. Default `False`.

Details
-------
The breakdown budget drawn by `breakdown = True` is the plug-in
breakdown budget of `tvbounds_summary()`, the estimated counterpart of
delta_b(tau_star) = inf{delta : tau_lower(delta) <= tau_star <= tau_upper(delta)}:
the first budget at which the bound path adjacent to
tau_star reaches it — the lower path when the
baseline point estimate exceeds `tau_star`, the upper path otherwise.
The line is annotated with the value of
delta_b_hat. No line is drawn when the
breakdown budget is censored at the right endpoint of the grid, when it
sits at the left endpoint, or when the baseline point estimate is
missing.

Rows of `x.bounds` with missing bound values (e.g. censored or
infeasible budgets) are omitted from the corresponding layer. For the
counterfactual application, whose divergence budgets may exceed one,
`log_x = True` switches to a logarithmic budget axis.

Value
-----
A matplotlib Figure object, which displays in the active graphics
backend and can be modified further with matplotlib calls.

References
----------
Palomba, F. (2026). "Sensitivity Analysis in Population Shares."
Working paper.

See Also
--------
`tvbounds_summary()` for the numerical summary measures;
`.plot()` dispatches here for `TVBounds` objects.

Examples
--------
import numpy as np
import pandas as pd
from tvbounds import tvbounds_attrition, tvbounds_plot

rng = np.random.default_rng(123)
n = 400
d = rng.binomial(1, 0.5, n)
s = rng.binomial(1, 1 / (1 + np.exp(-(0.5 + 0.5 * d))))
y = np.where(s == 1, 0.3 * d + rng.normal(size = n), np.nan)
dat = pd.DataFrame({"y": y, "d": d, "s": s})
fit = tvbounds_attrition(dat, outcome = "y", treatment = "d",
                         response = "s", delta = np.linspace(0, 1, 11),
                         B = 100, seed = 1)
tvbounds_plot(fit)

# Bounds only, no confidence band and no breakdown mark:
tvbounds_plot(fit, bands = False, breakdown = False)
""",
    'tvbounds_summary': """\
Summary measures for total-variation sensitivity bounds

Description
-----------
Computes the summary measures of Palomba (2026) for a `TVBounds` object:
the plug-in and certified breakdown budgets, the shadow price of
robustness, the robustness standard error, and the certification
frontier. The budget `delta` is the sensitivity parameter
delta of the total-variation (or contamination, or
divergence) neighborhood over which the bounds were computed.

Usage
-----
object.summary(**kwargs)

tvbounds_summary(
  object,
  delta = None,
  tau_star = 0,
  level = 0.95,
  cost_per_unit = 50,
  jump = 0.05,
  direction = "auto"
)

Arguments
---------
object
    A `TVBounds` object returned by `tvbounds_attrition()`,
    `tvbounds_counterfactual()`, or `tvbounds_riv()`.
**kwargs
    For the `.summary()` method: further arguments forwarded to
    `tvbounds_summary()`.
delta
    Optional evaluation budget delta (a single
    nonnegative number). `None` (default) evaluates the measures at the
    plug-in breakdown budget
    delta_b(tau_star), with the censoring
    fallback described in Details.
tau_star
    Reference value tau_star of the
    estimand against which robustness is judged (default `0`).
level
    Confidence level 1 - alpha for the
    two-sided critical value z_{1-alpha/2} used by
    the normal-approximation breakdown budget and the certification
    frontier (default `0.95`, that is, `qnorm(0.975)`). The certified
    breakdown itself is read off the band stored in the object.
cost_per_unit
    Marginal cost of one additional sampled unit, used
    for the cost equivalent of the certification frontier (default `50`,
    the paper's USD benchmark from the 3ie closed-grant portfolio).
jump
    Increase in the certified budget that the frontier prices,
    so that `n_star` is evaluated at
    delta + jump (default `0.05`).
direction
    One of `"auto"`, `"lower"`, `"upper"`; see Details.

Details
-------
Write tau_lower(delta) and
tau_upper(delta) for the lower and upper
sensitivity bounds on the estimand at budget delta. The
first is nonincreasing and the second nondecreasing in the budget, and
both collapse at delta = 0 to the value of the
estimand under the baseline distribution P_*. Robustness is
judged against a reference value tau_star of the
estimand, supplied through `tau_star`, and typically zero when it is the
sign of an effect rather than its magnitude that is of interest.

All measures are computed on the signed bound path adjacent to
tau_star, namely
tau_lower(delta) - tau_star
when `direction` is `"lower"` and
tau_star - tau_upper(delta)
when it is `"upper"`, so that in both cases the path starts positive
when the conclusion holds at the baseline. The two directions are thus
treated symmetrically, as the paper treats them by replacing the
integrand g by -g; the displays below are written for
the lower path. With `direction = "auto"` (the default) the lower path
is used when the baseline point estimate exceeds `tau_star` and the
upper path otherwise.

The breakdown budget is the smallest budget at which the bounds cease to
exclude the reference value,

    delta_b(tau_star) = inf{delta in [0, 1] : tau_lower(delta) <= tau_star <= tau_upper(delta)},

with the convention that the infimum over the empty set equals one. That
convention is exactly the package's censoring rule: when the estimated
path never reaches tau_star on the supplied budget
grid, the breakdown is reported at the right endpoint of the grid, which
is one for the total-variation and contamination neighborhoods, and is
flagged by `censored = True` rather than recorded as an inequality.

Sampling uncertainty is accounted for by the certified breakdown budget,

    delta_b^C(alpha) = inf{delta in [0, 1] : tau_lower_hat(delta) - z_{1-alpha/2} sigma_hat(delta) / sqrt(n) <= tau_star},

the largest budget at which the conclusion survives sampling
uncertainty. Here tau_lower_hat(delta)
estimates the lower bound path, sigma(delta) is
the asymptotic standard deviation of that estimator in units of the
estimand and sigma_hat(delta) its
estimator, and z_{1-alpha/2} is the two-sided
normal critical value at the confidence level `level`.

The shadow price of robustness is the marginal cost, in units of the
estimand, of one further unit of budget,
eta(delta) = -tau_lower'(delta);
it is nonnegative and nonincreasing, and the package estimates it by
central differences on the budget grid. Dividing the sampling standard
deviation of the bound by the shadow price converts it from units of the
estimand into units of the budget, which yields the robustness standard
error

    varsigma_b = sigma(delta_b) / eta(delta_b).

The certification frontier is the sample size at which the population
counterpart of the certified-breakdown inequality just clears the
reference value at budget delta,

    n*(delta; alpha) = z_{1-alpha/2}^2 sigma^2(delta) / (tau_lower(delta) - tau_star)^2.

It is real-valued, so the smallest certifying integer sample size is
floor(n*(delta; alpha)) + 1.
The frontier diverges as the budget approaches the breakdown budget and
is meaningful only below it: at and past the breakdown budget no sample
size certifies the conclusion, and the frontier is reported as `NaN`. Its
semi-elasticity
d log n*(delta; alpha) / d delta,
the certification elasticity, gives the rate at which the required
sample size grows with the budget; `cost_per_pp` prices a discrete
version of it.

The one-row data frame `measures` reports the following, alongside the
symbol each one corresponds to in the paper:

* `delta_b` — the plug-in breakdown budget
  delta_b(tau_star), obtained as the
  first crossing of the estimated signed path with zero, linearly
  interpolated between grid points; `censored` flags the empty-set
  convention described above.
* `delta_b_ci` — the certified breakdown budget
  delta_b^C(alpha),
  read off the outer confidence limit stored in the object (columns
  `ci_lower`/`ci_upper` of `object.bounds`), that is, off the same band
  the figures draw, so that tables and figures agree on one number. `NaN`
  when the object carries no band, and `censored_ci` flags censoring as
  above.
* `delta_b_ci_norm` — the same certified breakdown budget in its
  normal-approximation form, the first crossing of
  tau_lower_hat(delta) - z_{1-alpha/2} sigma_hat(delta) / sqrt(n)
  with tau_star. It estimates the same population
  quantity as `delta_b_ci` and is retained as a diagnostic: the two
  differ only when the bootstrap distribution of the bound is
  asymmetric.
* `delta_eval` — the budget delta at which the local
  measures `eta`, `se`, `varsigma`, and `varsigma_sc` are evaluated.
* `eta` — the shadow price of robustness
  eta(delta) at `delta_eval`.
* `se` — the estimated standard error of the bound in units of the
  estimand at `delta_eval`, that is,
  sigma_hat(delta) / sqrt(n).
* `varsigma` — the robustness standard error
  varsigma_b = sigma(delta_b) / eta(delta_b),
  computed as `se * sqrt(n) / eta` and therefore expressed in units of
  the budget rather than in units of the estimand.
* `varsigma_sc` — the finite-sample analogue `se / eta`, equal to
  varsigma_b / sqrt(n), which is the
  sampling standard deviation of the breakdown budget itself at the
  realized sample size.
* `frontier_at`, `n_cur`, `n_star`, `delta_n`, `cost_per_pp` — the
  certification frontier. `frontier_at` is the budget at which the
  frontier is priced and `n_cur` the real-valued frontier
  n*(delta; alpha) there, while `n_star`
  is the smallest certifying integer sample size
  floor(n*(delta; alpha)) + 1
  at the budget raised by `jump`. `delta_n` is that sample size net of
  the realized `n`, and `cost_per_pp` the implied cost of raising the
  certified budget by one percentage point, at `cost_per_unit` per
  sampled unit.
* `label`, `direction`, `n`, `point`, `tau_star` — the estimand label,
  the resolved direction, the sample size n, the baseline
  estimate of the estimand at delta = 0, and the
  reference value tau_star.

The evaluation budget for `eta`, `se`, and `varsigma` is the `delta`
argument when supplied; when `delta = None` (the default) it is the
plug-in breakdown budget, replaced by the certified breakdown budget
when the plug-in breakdown is censored, as in the paper. The frontier is
priced at the certified breakdown (default) or at the supplied `delta`.

Objects without inference (the recentered-IV and counterfactual
applications carry none by design) degrade gracefully: the plug-in
measures are reported, the certified and frontier measures are `NaN`, and
the `notes` field explains why.

Value
-----
An object of class `TVBoundsSummary`: an object with a one-row
data frame `measures` (columns `label`, `direction`, `n`, `point`,
`tau_star`, `delta_b`, `censored`, `delta_b_ci`, `censored_ci`,
`delta_b_ci_norm`, `delta_eval`, `eta`, `se`, `varsigma`,
`varsigma_sc`, `frontier_at`, `n_cur`, `n_star`, `delta_n`,
`cost_per_pp`) and metadata fields (`application`, `estimand_label`,
`neighborhood`, `divergence`, `direction`, `tau_star`, `delta_eval`,
`level`, `band_level`, `zc`, `cost_per_unit`, `jump`, `has_se`,
`has_ci`, `notes`, `call`). Details gives the symbol of the paper each
column of `measures` corresponds to. Printed compactly by
`print()`.

References
----------
Palomba, F. (2026). "Sensitivity Analysis in Population Shares."
Working paper.

See Also
--------
`tvbounds_plot()` to display the bounds; `.summary()` dispatches
here for `TVBounds` objects.

Examples
--------
import numpy as np
import pandas as pd
from tvbounds import tvbounds_attrition, tvbounds_summary

rng = np.random.default_rng(123)
n = 400
d = rng.binomial(1, 0.5, n)
s = rng.binomial(1, 1 / (1 + np.exp(-(0.5 + 0.5 * d))))
y = np.where(s == 1, 0.3 * d + rng.normal(size = n), np.nan)
dat = pd.DataFrame({"y": y, "d": d, "s": s})
fit = tvbounds_attrition(dat, outcome = "y", treatment = "d",
                         response = "s", delta = np.linspace(0, 1, 11),
                         B = 100, seed = 1)
tvbounds_summary(fit)

# Evaluate the measures at a chosen budget instead of the breakdown:
tvbounds_summary(fit, delta = 0.2)
""",
    'tvbounds_control': """\
Control options for the counterfactual solver

Description
-----------
Constructs the list of tuning options consumed by
`tvbounds_counterfactual()`. The defaults reproduce the settings used
for the counterfactual application in Palomba (2026).

Usage
-----
tvbounds_control(
  maxsolves = 10,
  startptrange = 0.01,
  use_optim = False,
  time_limit = 60,
  iterations = 100,
  outer_iterations = 3,
  inner_opt = None,
  outer_opt = None,
  knitro_options = {},
  eta_min = 1e-120,
  lower_limit = -10,
  psi_tv_eps = 1e-04,
  tvac_tau = 0.001,
  tvmix_tau = 0.001,
  purekl_acap = 500,
  tvmix_kappa = None
)

Arguments
---------
maxsolves
    Integer, number of multi-start restarts of the outer
    optimization over the structural parameter theta for each
    budget and side.
startptrange
    Positive scalar; restarts after the first perturb the
    initial theta uniformly on
    `[-startptrange, startptrange]` coordinate-wise (clipped to the
    parameter box).
use_optim
    Logical; if `True` the outer optimization uses
    Optim.jl (projected L-BFGS with the analytic envelope-theorem
    gradient) instead of a nested KNITRO solve. The Optim fallback
    avoids nested KNITRO contexts, which segfault with some KNITRO.jl
    versions.
time_limit
    Positive scalar, wall-clock limit in seconds per outer
    Optim run (used only when `use_optim = True`).
iterations
    Positive integer, inner iteration limit per outer
    Optim run (used only when `use_optim = True`).
outer_iterations
    Positive integer, number of Fminbox outer
    iterations per Optim run (used only when `use_optim = True`).
inner_opt
    Path to a KNITRO option file for the inner (dual)
    problem, or `None` for the shipped default `inner.opt`.
outer_opt
    Path to a KNITRO option file for the outer problem
    (over the structural parameter theta), or `None` for the
    shipped default
    `outer.opt`. The package also ships `outer_fast.opt` (analytic
    envelope gradient, looser tolerances, suited to plotting grids) and
    `outer_boot_tv.opt`; point this argument at
    `tvbounds.julia_file("opt", "outer_fast.opt")`
    to use them.
knitro_options
    Dict of individual KNITRO options (e.g.
    `dict(maxit = 500, outlev = 2)`). These are merged into *both* the
    inner and the outer option files by writing merged copies to
    the temporary directory (`tempfile.gettempdir()`); entries override options already present and are
    appended otherwise. Values must be scalar strings, numbers, or
    booleans (booleans are written as 0/1).
eta_min
    Positive scalar, lower bound on the dual variable
    eta, the multiplier pricing the divergence budget (kept
    strictly positive so the perspective function in the dual objective is
    well defined).
lower_limit
    Scalar; inner (dual) objective values at or below
    this threshold are treated as unbounded below (the inner solver's
    infeasibility guard).
psi_tv_eps
    Positive scalar, Huber smoothing scale for the kinked
    total-variation conjugate phi_TV^*. The
    smoothed conjugate lies above the exact one, so computed bounds remain
    outward-conservative.
tvac_tau
    Positive scalar, soft-max temperature for the
    log-sum-exp term of the `"TVac"` divergence, that is, total variation
    restricted to distributions with P << P_*.
tvmix_tau
    Positive scalar, soft-max temperature for the
    log-sum-exp term of the `"TVmix"` divergence, that is, total variation
    intersected with the mixture constraint
    P >= kappa * P_*.
purekl_acap
    Positive scalar, overflow clamp on the exponent of
    the conjugate of the pure Kullback-Leibler entropy
    phi_KL(s) = s log s - s
      + 1, which increases exponentially in its argument.
tvmix_kappa
    `None` or a scalar in [0, 1]: the
    contamination weight kappa of the mixture constraint
    P >= kappa * P_* used by the `"TVmix"` and
    `"TVmixC"` divergences (the perturbed distribution must contain the
    baseline as a mixing component with weight kappa).
    `None` ties kappa to the budget as
    kappa = 1 - delta, matching Palomba (2026).
    An explicit value decouples the two parameters; the reduced `"TVmix"`
    program requires kappa >= 1 - delta (so
    that the total-variation constraint is redundant), while `"TVmixC"`
    supports any kappa in [0, 1).

Details
-------
The options are stated in the notation of the paper. A candidate
distribution P is measured against the baseline
P_* by the divergence
D_phi(P || P_*) generated by an entropy
function phi, and the budget delta caps it.
At a fixed structural parameter theta the inner problem is
solved in its dual form, over the multipliers
(zeta, eta, lambda) attached respectively to
the total-mass constraint, to the divergence budget -- so that
eta is the shadow price of robustness -- and to the moment
restrictions. The dual objective integrates against the baseline the
perspective of the convex conjugate phi^*,

    (phi^*)^pi( g(U; theta) - lambda' m(U; theta) - zeta,  eta ),

with g the counterfactual criterion and m the moment
function. Under total variation,
phi_TV(s) = |s - 1| / 2, the
conjugate phi_TV^* is piecewise linear and
the perspective collapses to the kinked
max{g(U; theta) - lambda' m(U; theta) - zeta, -eta/2}; the
smoothing options below round those kinks off, always from above, so the
computed bounds stay outward-conservative. The contamination weight
kappa of the mixture constraint
P >= kappa * P_* is set by `tvmix_kappa`.

Value
-----
A dict of class `TVBoundsControl` with the (validated)
options above.

References
----------
Palomba, F. (2026). "Sensitivity Analysis in Population Shares."
Working paper.

Christensen, T. and B. Connault (2023). "Counterfactual Sensitivity
and Robustness." *Econometrica*, 91(1), 263-298.

See Also
--------
`tvbounds_counterfactual()`

Examples
--------
from tvbounds import tvbounds_control

ctrl = tvbounds_control()
ctrl["maxsolves"]

# a faster configuration for exploratory grids
tvbounds_control(maxsolves = 3, knitro_options = dict(maxit = 200))
""",
    'print.tvbounds': """\
Print a tvbounds object

Description
-----------
Compact display of a `TVBounds` object: the application, the
neighborhood over which the sensitivity bounds were computed, the sample
size n, the baseline point estimate — the value of the estimand
under the baseline distribution P_*, reported at
delta = 0 — the grid of budgets delta on
which the bound paths tau_lower(delta)
and tau_upper(delta) were evaluated, and
whether inference is attached.

Usage
-----
print(x)

x.print(digits = 3, **kwargs)

Arguments
---------
x
    A `TVBounds` object.
digits
    Number of significant digits (default `3`).
**kwargs
    Further arguments; ignored.

Value
-----
`x`, invisibly.

References
----------
Palomba, F. (2026). "Sensitivity Analysis in Population Shares."
Working paper.

Examples
--------
import numpy as np
import pandas as pd
from tvbounds import tvbounds_attrition

rng = np.random.default_rng(1)
n = 200
d = rng.binomial(1, 0.5, n)
s = rng.binomial(1, 1 / (1 + np.exp(-(0.4 + 0.4 * d))))
y = np.where(s == 1, 0.3 * d + rng.normal(size = n), np.nan)
fit = tvbounds_attrition(pd.DataFrame({"y": y, "d": d, "s": s}),
                         outcome = "y", treatment = "d", response = "s",
                         delta = np.linspace(0, 1, 11), bootstrap = False)
print(fit)
""",
    'print.tvbounds_summary': """\
Print a tvbounds summary

Description
-----------
Compact display of the summary measures computed by
`tvbounds_summary()`: the plug-in and certified breakdown budgets
delta_b(tau_star) and
delta_b^C(alpha), the
shadow price of robustness
eta(delta) = -tau_lower'(delta),
the robustness standard error
varsigma_b = sigma(delta_b) / eta(delta_b),
and the certification frontier
n*(delta; alpha), followed by any notes
on measures that could not be computed. The reference value of the
estimand, tau_star, is shown as `tau_star`,
matching the argument name.

Usage
-----
print(x)

x.print(digits = 3, **kwargs)

Arguments
---------
x
    A `TVBoundsSummary` object.
digits
    Number of significant digits (default `3`).
**kwargs
    Further arguments; ignored.

Value
-----
`x`, invisibly.

References
----------
Palomba, F. (2026). "Sensitivity Analysis in Population Shares."
Working paper.

Examples
--------
import numpy as np
import pandas as pd
from tvbounds import tvbounds_attrition

rng = np.random.default_rng(1)
n = 200
d = rng.binomial(1, 0.5, n)
s = rng.binomial(1, 1 / (1 + np.exp(-(0.4 + 0.4 * d))))
y = np.where(s == 1, 0.3 * d + rng.normal(size = n), np.nan)
fit = tvbounds_attrition(pd.DataFrame({"y": y, "d": d, "s": s}),
                         outcome = "y", treatment = "d", response = "s",
                         delta = np.linspace(0, 1, 11), bootstrap = False)
# Without bootstrap draws the certified measures degrade to NaN, with a
# note explaining why:
print(fit.summary())
""",
    'tvbounds-package': """\
tvbounds: Sensitivity Analysis and Bounds under Total Variation Neighborhoods

Description
-----------
Implements the sensitivity analysis framework of Palomba (2026),
"Sensitivity Analysis in Population Shares". The estimand is an
expectation E_P[g(Z; theta)] of a
known integrand g under a distribution P of the data
Z. Rather than committing to a single baseline distribution
P_*, the package lets P range over a robustness
set

    P_phi(theta; rho, P_*, delta) = {P : E_P[m(Z; theta)] in M(rho), D_phi(P || P_*) <= delta},

collecting the distributions that remain compatible with the moment
restrictions and lie within a divergence budget delta of
the baseline, and reports the resulting sensitivity bounds

    inf_theta inf_P E_P[g(Z; theta)]  and  sup_theta sup_P E_P[g(Z; theta)].

Details
-------
The sensitivity parameter is the budget delta, supplied
through the `delta` argument. Under the total variation entropy
phi_TV(s) = |s - 1| / 2 the
divergence is the total variation distance
TV(P, P_*) and delta
in [0, 1] bounds the fraction of baseline probability mass that may be
misspecified; under the contamination neighborhood
C_kappa(P_*) = {P = kappa P_* + (1 - kappa) R}
the perturbed distribution is a mixture of the baseline distribution and
an arbitrary distribution R.

Applications
------------
Three ready-made interfaces cover the paper's empirical applications:

* `tvbounds_attrition()` — randomized experiments with attrition:
  total variation and contamination bounds on the treatment effect,
  bootstrap inference, and optional covariate-pooled bounds. At budget
  delta = 1 the bounds reproduce the Lee (2009)
  worst-case bounds.
* `tvbounds_counterfactual()` — counterfactual predictions in structural
  models, through an interface to Julia and the Artelys KNITRO solver;
  the only function supporting general entropy functions
  phi, following Christensen and Connault (2023).
* `tvbounds_riv()` — recentered instrumental variables / formula
  instruments, as in Borusyak and Hull (2023), with first-stage
  breakdown budgets and no bootstrap inference (by design).

Reporting
---------
All estimators return a common `TVBounds` object carrying the bound
paths tau_lower(delta) and
tau_upper(delta) over a grid of budgets.
`tvbounds_plot()` (or `.plot()`) displays the bounds against the budget,
and `tvbounds_summary()` (or `.summary()`) computes the summary measures
of the paper: the breakdown budget
delta_b(tau_star) and its certified
counterpart, the shadow price of robustness
eta(delta), the robustness standard
error varsigma_b = sigma(delta_b) / eta(delta_b),
and the certification frontier
n*(delta; alpha).

KNITRO requirement
------------------
`tvbounds_counterfactual()` relies on Julia and the commercial Artelys
KNITRO solver, which requires a valid license; see that function's help
page for details.

References
----------
Palomba, F. (2026). "Sensitivity Analysis in Population Shares."
Working paper.

Borusyak, K. and Hull, P. (2023). "Nonrandom Exposure to Exogenous
Shocks." *Econometrica*, 91(6), 2155--2185.

Christensen, T. and Connault, B. (2023). "Counterfactual Sensitivity and
Robustness." *Econometrica*, 91(1), 263--298.

Lee, D. S. (2009). "Training, Wages, and Sample Selection: Estimating
Sharp Bounds on Treatment Effects." *Review of Economic Studies*,
76(3), 1071--1102.

Author(s)
---------
**Maintainer**: Filippo Palomba <fpalomba@princeton.edu> [copyright holder]
""",
}
