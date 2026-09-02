# Counterfactual predictions in structural models: sensitivity
# bounds through the Julia/KNITRO backend (TVBoundsJulia).

#' Sensitivity bounds for counterfactual predictions in structural models
#'
#' Computes the lower and upper sensitivity bounds
#' \eqn{\underline{\mathsf{k}}(\delta)}{k_lower(delta)} and
#' \eqn{\overline{\mathsf{k}}(\delta)}{k_upper(delta)} on a counterfactual
#' \eqn{\mathbb{E}_P[g(U;\theta)]}{E_P[g(U; theta)]}, when the distribution
#' \eqn{P}{P} of the latent variables \eqn{U}{U} ranges over a divergence
#' neighborhood of the simulated baseline \eqn{P_{*}}{P_*} with budget
#' `delta`, and the structural parameter \eqn{\theta}{theta} ranges over the
#' values compatible with the moment conditions
#' \eqn{\mathbb{E}_P[m(U;\theta)] \in \mathcal{M}(\rho)}{E_P[m(U; theta)] in
#' M(rho)}, following Palomba (2026) and Christensen and Connault (2023).
#' This is the only function in the package that supports general
#' \eqn{\phi}{phi}-divergences beyond total variation (spelled out at
#' first use; `"TV"` below) and contamination.
#'
#' @section KNITRO requirement:
#' This function relies on Julia (>= 1.9) and on the commercial
#' **Artelys KNITRO** solver, accessed through the 'JuliaCall' package
#' and KNITRO.jl. A valid KNITRO license is required (free academic
#' trials are available from Artelys at
#' <https://www.artelys.com/solvers/knitro/>). On the first call in each
#' R session the package initializes Julia, instantiates its Julia
#' environment, and checks that KNITRO.jl loads and that a KNITRO solver
#' context can be created (which exercises the license); a one-time
#' message reports the outcome, and the call stops with installation and
#' license guidance when the check fails. The check runs once per R
#' session.
#'
#' @details
#' **Setup.** Let \eqn{U \in \mathcal{Z}}{U in Z} collect the latent variables
#' of the structural model (taste shocks, unobserved heterogeneity,
#' measurement errors) and let \eqn{P_{*}}{P_*} be the baseline distribution
#' the econometrician postulates for them. At a structural parameter
#' \eqn{\theta \in \Theta}{theta in Theta}, a distribution \eqn{P}{P} is
#' compatible with the model when
#' \eqn{\mathbb{E}_P[m(U;\theta)] \in \mathcal{M}(\rho)}{E_P[m(U; theta)] in
#' M(rho)}, with \eqn{m}{m} the moment function and
#' \eqn{\mathcal{M}(\rho)}{M(rho)} the moment constraint set; the object of
#' interest is the counterfactual
#' \eqn{\mathbb{E}_P[g(U;\theta)]}{E_P[g(U; theta)]}, the expectation of a
#' known criterion \eqn{g}{g} that is linear in \eqn{P}{P} at fixed
#' \eqn{\theta}{theta}. The robustness set collects the distributions that
#' are compatible with the model and within budget of the baseline,
#' \deqn{\mathcal{P}_{\phi}(\theta;\rho,P_{*},\delta) := \{P :
#'   D_{\phi}(P \| P_{*}) \leq \delta, \;
#'   \mathbb{E}_P[m(U;\theta)] \in \mathcal{M}(\rho)\},}{
#'   P_phi(theta; rho, P_*, delta) := {P : D_phi(P || P_*) <= delta,
#'   E_P[m(U; theta)] in M(rho)},}
#' where \eqn{D_{\phi}(P \| P_{*})}{D_phi(P || P_*)} is the
#' \eqn{\phi}{phi}-divergence selected by `divergence` and
#' \eqn{\delta}{delta} the budget, and the reported bounds are the nested
#' extrema
#' \deqn{\underline{\mathsf{k}}(\delta) = \inf_{\theta \in \Theta}
#'   \inf_{P \in \mathcal{P}_{\phi}(\theta;\rho,P_{*},\delta)}
#'   \mathbb{E}_P[g(U;\theta)], \qquad
#'   \overline{\mathsf{k}}(\delta) = \sup_{\theta \in \Theta}
#'   \sup_{P \in \mathcal{P}_{\phi}(\theta;\rho,P_{*},\delta)}
#'   \mathbb{E}_P[g(U;\theta)].}{
#'   k_lower(delta) = inf_theta inf_P E_P[g(U; theta)],
#'   k_upper(delta) = sup_theta sup_P E_P[g(U; theta)],
#'   both over P in P_phi(theta; rho, P_*, delta).}
#' The optimization over \eqn{P}{P} at a fixed \eqn{\theta}{theta} is the
#' inner problem, solved in its dual form; the optimization over
#' \eqn{\theta}{theta} is the outer problem.
#'
#' **Correspondence between the paper and the code.** The paper writes the
#' counterfactual integrand as \eqn{g}{g} and the moment function as
#' \eqn{m}{m}. The interfaces below fill an array named `K` with the values
#' of \eqn{g}{g} and an array named `G` with the values of \eqn{m}{m}: read
#' `K` as \eqn{g}{g} and `G` as \eqn{m}{m} throughout. The solver takes the
#' moment conditions in centered equality form,
#' \eqn{\mathbb{E}_P[m(U;\theta)] = 0}{E_P[m(U; theta)] = 0}, so a nonzero
#' target \eqn{\rho}{rho} is absorbed by centering the moment function. The
#' argument `d` is the number of moment conditions,
#' \eqn{\mathsf{d}_m}{d_m}; the common length of `theta_lb` and `theta_ub`
#' is the dimension of \eqn{\theta}{theta}; and `u_dim` is the dimension
#' \eqn{\mathsf{d}_z}{d_z} of a single latent draw.
#'
#' **Moments specification.** `moments` can be supplied in three forms:
#'
#' 1. a length-2 character vector `c(file, fname)`: `file` is the path
#'    of a Julia source file that is included into the session, and
#'    `fname` the name of a function defined there (at the top level, in
#'    `Main`) with the in-place signature
#'    `moments!(K, G, theta, U, obj)`. The function must fill `K` (an
#'    `M`-vector holding the counterfactual values
#'    \eqn{g(U^{(j)};\theta)}{g(U^(j); theta)}) and `G` (an `M x d` matrix
#'    holding the moment functions
#'    \eqn{m(U^{(j)};\theta)}{m(U^(j); theta)}, one row per draw) and may
#'    read the user payload as `obj.gamma` (an R list arrives in Julia as
#'    an ordered dictionary keyed by symbols, so `obj.gamma[:name]`);
#' 2. a single string naming a Julia function with the same signature
#'    that is already defined in the session;
#' 3. an R function `function(theta, U, gamma)` returning a list with
#'    components `K` (numeric of length `M`) and `G` (numeric
#'    `M x d` matrix). This path is **much slower** (every objective
#'    evaluation crosses the R/Julia boundary), and because ForwardDiff
#'    cannot differentiate through R code the outer optimization needs
#'    either a user-supplied `gradient` or finite differences (the
#'    default for this path).
#'
#' For Julia moments the outer envelope-theorem gradient differentiates
#' the moments by automatic differentiation (ForwardDiff), so the Julia
#' function should be written generically in the element type of
#' `theta`; pass `gradient = "fd"` for a non-generic function.
#'
#' **Latent draws.** `U` is the `M x u_dim` matrix of latent draws
#' \eqn{U^{(1)}, \dots, U^{(M)}}{U^(1), ..., U^(M)} that discretizes the
#' baseline distribution \eqn{P_{*}}{P_*}, row `j` holding the draw
#' \eqn{U^{(j)}}{U^(j)}. When `U = NULL` the package generates `M`
#' scrambled-Halton points in the unit cube
#' \eqn{(0,1)^{\mathsf{d}_z}}{(0, 1)^d_z} of dimension `u_dim` (Owen,
#' 2017), seeded by `seed`; the moments function is then responsible for
#' mapping the uniform coordinates into baseline draws (e.g. through
#' quantile transforms). The Halton generator supports `u_dim <= 15`;
#' supply `U` directly for higher-dimensional draws.
#'
#' **Divergences.** `divergence` selects the entropy function
#' \eqn{\phi}{phi} whose divergence
#' \eqn{D_{\phi}(P \| P_{*})}{D_phi(P || P_*)} defines the neighborhood;
#' the budget grid `delta` must be strictly positive, and must lie in
#' \eqn{(0,1]}{(0, 1]} for the total-variation family:
#'
#' * `"KL_chi2"` (default): the hybrid Kullback-Leibler/chi-square
#'   divergence of Christensen and Connault (2023); any `delta > 0`.
#' * `"KL"`, `"chi2"`: the pure Kullback-Leibler entropy
#'   \eqn{\phi_{\mathsf{KL}}(s) = s \log s - s + 1}{phi_KL(s) = s log s -
#'   s + 1} and the Pearson chi-square entropy; any `delta > 0`.
#' * `"TV"`: total variation,
#'   \eqn{\phi_{\mathsf{TV}}(s) = |s-1|/2}{phi_TV(s) = |s - 1| / 2}, for
#'   which \eqn{D_{\phi_{\mathsf{TV}}}(P \| P_{*}) =
#'   \mathsf{TV}(P,P_{*})}{D_phi_TV(P || P_*) = TV(P, P_*)}. Its recession
#'   function is finite,
#'   \eqn{\phi^{\infty}_{\mathsf{TV}}(1) = 1/2}{phi_TV^inf(1) = 1/2}, so
#'   the dual keeps the pointwise constraint
#'   \eqn{\zeta + \eta/2 \geq \sup_{u \in \mathcal{Z}} \{g(u;\theta) -
#'   \lambda^{\top} m(u;\theta)\}}{zeta + eta/2 >= sup_u {g(u; theta) -
#'   lambda' m(u; theta)}}, which discretizes into `M` linear feasibility
#'   constraints (one per draw).
#' * `"TVmix"`: total variation intersected with the mixture
#'   (contamination) constraint \eqn{P \geq \kappa P_{*}}{P >= kappa * P_*},
#'   that is, the perturbed distribution contains the baseline as a mixing
#'   component with weight \eqn{\kappa = 1 - \delta}{kappa = 1 - delta} (or
#'   `control$tvmix_kappa`); solved in the exact reduced form (Palomba,
#'   2026).
#' * `"TVmixC"`: the literal dual of the same program, kept as a
#'   cross-check of `"TVmix"`; it is slower, and it is the only mode
#'   supporting a mixing weight
#'   \eqn{\kappa < 1 - \delta}{kappa < 1 - delta}.
#' * `"TVac"`: total variation restricted to distributions absolutely
#'   continuous with respect to the baseline,
#'   \eqn{P \ll P_{*}}{P << P_*}.
#'
#' The kinked total-variation conjugates
#' \eqn{\phi^{*}_{\mathsf{TV}}}{phi_TV^*} are Huber-smoothed and the
#' per-draw maxima in the dual objective log-sum-exp-smoothed (scales in
#' [tvbounds_control()]); both smoothings lie above the exact functions,
#' so computed bounds are outward-conservative (wider, never narrower) at
#' order `1e-3`.
#'
#' **Optimization.** For each budget (in increasing order) and each
#' side, the solver runs `control$maxsolves` multi-start outer
#' optimizations over \eqn{\theta}{theta} (KNITRO, or Optim.jl when
#' `control$use_optim = TRUE`), warm-started at the previous budget's
#' optimum; the reported bound is the inner (dual) value re-solved at
#' the best candidate, which makes the bound curves monotone in the
#' budget by construction. A degenerate box (`theta_lb == theta_ub`)
#' skips the outer optimization and reports the bounds at the fixed
#' \eqn{\theta}{theta} supplied through `theta_init`. Failed budgets are
#' reported as `NA` (for `"TVmix"` an `NA` typically signals an
#' infeasible moment condition at every \eqn{\theta}{theta} in the box,
#' that is an empty robustness set at that budget).
#'
#' @param moments The moment/counterfactual function: a length-2
#'   character vector `c(file, fname)`, a single string naming a Julia
#'   function, or an R function. See Details.
#' @param d Integer, the number of moment conditions
#'   \eqn{\mathsf{d}_m}{d_m} (columns of `G`).
#' @param theta_lb,theta_ub Numeric vectors of equal length: the box for
#'   the structural parameter \eqn{\theta}{theta} over which the outer
#'   problems optimize. Their common length fixes the dimension of
#'   \eqn{\theta}{theta}.
#' @param delta Numeric vector of strictly positive budgets
#'   \eqn{\delta}{delta} (the sensitivity parameter of the neighborhood),
#'   without duplicates. For the total-variation family the budget must lie
#'   in \eqn{(0,1]}{(0, 1]}; for `"KL_chi2"`, `"KL"`, and `"chi2"` any
#'   positive value is allowed.
#' @param divergence Divergence keyword; one of `"KL_chi2"` (default),
#'   `"KL"`, `"chi2"`, `"TV"`, `"TVmix"`, `"TVmixC"`, `"TVac"`. See
#'   Details.
#' @param side `"both"` (default), `"lower"`, or `"upper"`: which bound
#'   problems to solve at each budget.
#' @param U Optional `M x u_dim` numeric matrix of latent draws
#'   \eqn{U^{(1)}, \dots, U^{(M)}}{U^(1), ..., U^(M)}, one per row. When
#'   `NULL`, scrambled-Halton uniforms are generated (see Details) and
#'   `M`, `u_dim` are required; when supplied, `M` and `u_dim` are taken
#'   from its dimensions.
#' @param M Integer, the number \eqn{M}{M} of simulated draws when
#'   `U = NULL` (default `50000`, the setting of the paper).
#' @param u_dim Integer, the dimension \eqn{\mathsf{d}_z}{d_z} of the
#'   latent draw when `U = NULL` (at most 15).
#' @param gamma Optional R list: an arbitrary payload forwarded to the
#'   moments function (as `obj.gamma` for Julia moments, as the third
#'   argument for R moments).
#' @param gradient How to differentiate the moments with respect to
#'   \eqn{\theta}{theta} in the outer optimization: `NULL` (default; automatic
#'   differentiation for Julia moments, finite differences for R
#'   moments), the string `"fd"` (finite differences), the name of a
#'   Julia function `jac(theta, U, obj)`, or an R function
#'   `function(theta, U, gamma)`. A user-supplied gradient must return
#'   either the stacked `(M*(d+1)) x l` Jacobian of `c(K, G)` (K rows
#'   first, then `G` in column-major order) or a list with components
#'   `K` (`M x l`) and `G` (`M x d x l`), where `l = length(theta_lb)`.
#' @param theta_init Optional numeric vector, the initial structural
#'   parameter \eqn{\theta}{theta} for the outer optimization (defaults to
#'   the midpoint of the box). Set it to the baseline estimate of the
#'   model: the reported baseline `point` is the plug-in counterfactual
#'   \eqn{\mathsf{k}(\theta;P_{*}) =
#'   \mathbb{E}_{P_{*}}[g(U;\theta)]}{k(theta; P_*) = E_{P_*}[g(U; theta)]}
#'   at `theta_init`, and is only returned when `theta_init` is supplied.
#' @param control A list created by [tvbounds_control()] with solver
#'   tuning options.
#' @param seed Optional integer seed for the scrambled-Halton draws (and
#'   for R's RNG, which is saved and restored so the call has no side
#'   effect on the caller's random-number stream).
#' @param verbose Logical; print solver progress (default `TRUE`).
#'
#' @return An object of class `c("tvbounds_counterfactual", "tvbounds")`:
#'   a list with the fields described in the package overview, in
#'   particular `bounds` (data frame with columns `delta`, `lower`,
#'   `upper`, holding \eqn{\delta}{delta},
#'   \eqn{\underline{\mathsf{k}}(\delta)}{k_lower(delta)} and
#'   \eqn{\overline{\mathsf{k}}(\delta)}{k_upper(delta)}; no
#'   standard-error or confidence-band columns, since this application
#'   currently carries no inference), `point` (the plug-in counterfactual
#'   \eqn{\mathbb{E}_{P_{*}}[g(U;\theta)]}{E_{P_*}[g(U; theta)]} at
#'   `theta_init`, or `NA` when `theta_init` was not supplied),
#'   `divergence`, and `details`, a list with:
#'   \describe{
#'     \item{`solver`}{data frame of per-budget diagnostics: outer
#'       multi-start flags, inner KNITRO status codes, and timings for
#'       each side (`-999` marks entries that do not apply, e.g. outer
#'       flags in fixed-theta mode).}
#'     \item{`theta_lower`, `theta_upper`}{`l x length(delta)` matrices
#'       of the outer-optimal structural parameters \eqn{\theta}{theta} at
#'       each budget, for the lower and the upper bound respectively.}
#'     \item{`M`, `u_dim`}{the number \eqn{M}{M} of simulated draws and
#'       their dimension \eqn{\mathsf{d}_z}{d_z}.}
#'     \item{`theta_lb`, `theta_ub`, `theta_init`, `fixed_theta`}{the
#'       parameter box, the initial point, and whether the box was
#'       degenerate.}
#'     \item{`control`}{the resolved control list, including the option
#'       files actually used.}
#'     \item{`moments`}{a short description of the moments
#'       specification.}
#'   }
#'
#' @references
#' Palomba, F. (2026). "Sensitivity Analysis in Population Shares."
#' Working paper.
#'
#' Christensen, T. and B. Connault (2023). "Counterfactual Sensitivity
#' and Robustness." *Econometrica*, 91(1), 263-298.
#'
#' Owen, A. B. (2017). "A randomized Halton algorithm in R."
#' arXiv:1706.02808.
#'
#' @examples
#' # The full solver requires Julia and a licensed KNITRO installation,
#' # so a complete run cannot be executed on CRAN or in checks:
#' \dontrun{
#' # Toy model (shipped with the package): U ~ Uniform(0, 1), one moment
#' # condition m(U; theta) = U - theta and counterfactual g(U; theta) = U,
#' # so under the "TVmix" neighborhood the bounds equal the endpoints of
#' # the theta box. In the Julia file, m is written into G and g into K.
#' toy <- system.file("julia", "examples", "toy.jl", package = "tvbounds")
#' fit <- tvbounds_counterfactual(
#'   moments  = c(toy, "tvb_toy_moments!"),
#'   d        = 1,
#'   theta_lb = 0.4, theta_ub = 0.6,
#'   delta    = c(0.5, 1),
#'   divergence = "TVmix",
#'   M = 500, u_dim = 1,
#'   theta_init = 0.5,
#'   control = tvbounds_control(maxsolves = 2),
#'   seed = 1234)
#' fit$bounds
#' }
#'
#' # The control constructor is pure R and always available:
#' tvbounds_control(maxsolves = 3)$maxsolves
#'
#' @seealso [tvbounds_control()], [tvbounds_plot()], [tvbounds_summary()]
#' @export
tvbounds_counterfactual <- function(moments, d, theta_lb, theta_ub,
                                    delta,
                                    divergence = c("KL_chi2", "KL", "chi2",
                                                   "TV", "TVmix", "TVmixC",
                                                   "TVac"),
                                    side = c("both", "lower", "upper"),
                                    U = NULL, M = 50000, u_dim = NULL,
                                    gamma = NULL,
                                    gradient = NULL, theta_init = NULL,
                                    control = tvbounds_control(),
                                    seed = NULL, verbose = TRUE) {
  cl <- match.call()
  divergence <- match.arg(divergence)
  side <- match.arg(side)
  .tvb_check_flag(verbose, "verbose")

  # ---- moments specification (pure R, testable without Julia) --------
  spec <- .tvb_parse_moments(moments)

  # ---- dimensions and boxes ------------------------------------------
  if (!is.numeric(d) || length(d) != 1L || !is.finite(d) ||
      d < 1 || d != as.integer(d)) {
    stop("`d` must be a positive integer (the number of moment ",
         "conditions).", call. = FALSE)
  }
  d <- as.integer(d)
  if (!is.numeric(theta_lb) || !is.numeric(theta_ub) ||
      length(theta_lb) == 0L || length(theta_lb) != length(theta_ub)) {
    stop("`theta_lb` and `theta_ub` must be numeric vectors of equal, ",
         "positive length.", call. = FALSE)
  }
  if (any(!is.finite(theta_lb)) || any(!is.finite(theta_ub))) {
    stop("`theta_lb` and `theta_ub` must be finite.", call. = FALSE)
  }
  if (any(theta_lb > theta_ub)) {
    stop("`theta_lb` must not exceed `theta_ub` in any coordinate.",
         call. = FALSE)
  }
  l <- length(theta_lb)
  theta_init_supplied <- !is.null(theta_init)
  if (is.null(theta_init)) {
    theta_init <- (theta_lb + theta_ub) / 2
  }
  if (!is.numeric(theta_init) || length(theta_init) != l ||
      any(!is.finite(theta_init))) {
    stop(sprintf("`theta_init` must be a finite numeric vector of length %d.",
                 l), call. = FALSE)
  }
  if (any(theta_init < theta_lb) || any(theta_init > theta_ub)) {
    stop("`theta_init` must lie inside the [theta_lb, theta_ub] box.",
         call. = FALSE)
  }

  # ---- budgets -------------------------------------------------------
  if (!is.numeric(delta) || length(delta) == 0L || any(!is.finite(delta))) {
    stop("`delta` must be a non-empty numeric vector of finite budgets.",
         call. = FALSE)
  }
  if (any(delta <= 0)) {
    stop("all budgets in `delta` must be strictly positive; the budget-",
         "zero baseline is reported separately as the `point` field ",
         "(the plug-in counterfactual at `theta_init`).", call. = FALSE)
  }
  if (anyDuplicated(delta)) {
    stop("`delta` must not contain duplicated budget values.",
         call. = FALSE)
  }
  tv_family <- divergence %in% c("TV", "TVmix", "TVmixC", "TVac")
  if (tv_family && any(delta > 1)) {
    stop(sprintf(
      "for divergence \"%s\" all budgets must lie in (0, 1].", divergence),
      call. = FALSE)
  }
  delta <- sort(as.numeric(delta))

  # ---- latent draws --------------------------------------------------
  if (!is.null(U)) {
    if (!is.matrix(U) || !is.numeric(U) || any(!is.finite(U))) {
      stop("`U` must be a finite numeric matrix (M x u_dim), or NULL.",
           call. = FALSE)
    }
    if (nrow(U) < 2L) {
      stop("`U` must have at least 2 rows.", call. = FALSE)
    }
    if (!is.null(u_dim) && u_dim != ncol(U)) {
      stop("`u_dim` disagrees with ncol(U); when `U` is supplied, its ",
           "dimensions are used and `M`/`u_dim` need not be given.",
           call. = FALSE)
    }
    M <- nrow(U)
    u_dim <- ncol(U)
  } else {
    if (is.null(u_dim)) {
      stop("`u_dim` is required when `U` is NULL (the dimension of the ",
           "latent draw for the scrambled-Halton generator).",
           call. = FALSE)
    }
    if (!is.numeric(u_dim) || length(u_dim) != 1L || !is.finite(u_dim) ||
        u_dim < 1 || u_dim != as.integer(u_dim)) {
      stop("`u_dim` must be a positive integer.", call. = FALSE)
    }
    if (u_dim > 15) {
      stop("the scrambled-Halton generator supports u_dim <= 15 (one ",
           "prime base per dimension); supply `U` directly for higher-",
           "dimensional draws.", call. = FALSE)
    }
    if (!is.numeric(M) || length(M) != 1L || !is.finite(M) ||
        M < 2 || M != as.integer(M)) {
      stop("`M` must be an integer >= 2.", call. = FALSE)
    }
  }
  M <- as.integer(M)
  u_dim <- as.integer(u_dim)

  # ---- payload, gradient, control, seed ------------------------------
  if (!is.null(gamma) && !is.list(gamma)) {
    stop("`gamma` must be NULL or a list.", call. = FALSE)
  }
  grad_spec <- .tvb_parse_gradient(gradient, spec)
  if (!inherits(control, "tvbounds_control")) {
    stop("`control` must be created by tvbounds_control().", call. = FALSE)
  }
  if (!is.null(control$tvmix_kappa)) {
    if (!divergence %in% c("TVmix", "TVmixC")) {
      warning("`control$tvmix_kappa` is ignored for divergence \"",
              divergence, "\".", call. = FALSE)
    } else if (divergence == "TVmix" &&
               control$tvmix_kappa < 1 - min(delta)) {
      stop("for divergence \"TVmix\" the mixing weight `tvmix_kappa` must ",
           "satisfy kappa >= 1 - delta at every budget (the reduced ",
           "program requires the total-variation constraint to be ",
           "redundant); use divergence \"TVmixC\" for kappa < 1 - delta.",
           call. = FALSE)
    }
  }
  if (!is.null(seed)) {
    if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed) ||
        seed != as.integer(seed)) {
      stop("`seed` must be NULL or a single integer.", call. = FALSE)
    }
    seed <- as.integer(seed)
  }

  # ---- KNITRO option files -------------------------------------------
  opts <- .tvb_resolve_opt(control)

  # ---- local RNG (no global side effects) ----------------------------
  if (!is.null(seed)) {
    withr::local_seed(seed)
  } else {
    withr::local_preserve_seed()
  }

  # ---- Julia session (lazy; once per R session) ----------------------
  .tvb_julia_setup(verbose = verbose)

  # ---- put the moments (and gradient) in place in Julia --------------
  moments_name <- .tvb_stage_moments(spec, gamma)
  jac_name <- .tvb_stage_gradient(grad_spec, gamma)

  # ---- solve ---------------------------------------------------------
  args <- list(
    moments_name = moments_name,
    delta        = as.numeric(delta),
    d            = d,
    theta_lb     = as.numeric(theta_lb),
    theta_ub     = as.numeric(theta_ub),
    theta_init   = as.numeric(theta_init),
    inner_opt    = opts$inner,
    outer_opt    = opts$outer,
    U            = U,
    M            = M,
    u_dim        = u_dim,
    seed         = seed,
    gamma        = if (spec$type == "rfun") NULL else gamma,
    divergence   = divergence,
    side         = side,
    jac_mode     = grad_spec$mode,
    jac_name     = jac_name,
    maxsolves        = control$maxsolves,
    startptrange     = control$startptrange,
    use_optim        = control$use_optim,
    time_limit       = control$time_limit,
    iterations       = control$iterations,
    outer_iterations = control$outer_iterations,
    lower_limit  = control$lower_limit,
    eta_min      = control$eta_min,
    psi_tv_eps   = control$psi_tv_eps,
    tvac_tau     = control$tvac_tau,
    tvmix_tau    = control$tvmix_tau,
    purekl_acap  = control$purekl_acap,
    tvmix_kappa  = control$tvmix_kappa %||% NaN,
    verbose      = isTRUE(verbose)
  )
  JuliaCall::julia_assign("_tvb_args", args)
  res <- JuliaCall::julia_eval("Main.TVBoundsJulia.tvb_solve(Main._tvb_args)")

  # ---- assemble the common return object -----------------------------
  clean <- function(x) {
    x <- as.numeric(x)
    x[!is.finite(x) | abs(x) >= 1e9] <- NA_real_
    x
  }
  delta_out <- as.numeric(res$delta)
  bounds <- data.frame(
    delta = delta_out,
    lower = clean(res$lower),
    upper = clean(res$upper)
  )
  solver <- data.frame(
    delta              = delta_out,
    status_outer_lower = as.integer(res$status_outer_lower),
    status_inner_lower = as.integer(res$status_inner_lower),
    time_lower         = clean(res$time_lower),
    status_outer_upper = as.integer(res$status_outer_upper),
    status_inner_upper = as.integer(res$status_inner_upper),
    time_upper         = clean(res$time_upper)
  )
  control_resolved <- control
  control_resolved$inner_opt <- opts$inner
  control_resolved$outer_opt <- opts$outer

  details <- list(
    side        = side,
    M           = as.integer(res$M),
    u_dim       = as.integer(res$u_dim),
    theta_lb    = theta_lb,
    theta_ub    = theta_ub,
    theta_init  = theta_init,
    theta_init_supplied = theta_init_supplied,
    fixed_theta = isTRUE(as.logical(res$fixed_theta)),
    solver      = solver,
    theta_lower = matrix(as.numeric(res$theta_lower), nrow = l),
    theta_upper = matrix(as.numeric(res$theta_upper), nrow = l),
    control     = control_resolved,
    moments     = spec$label,
    gradient    = grad_spec$mode,
    seed        = seed
  )

  new_tvbounds(
    application    = "counterfactual",
    bounds         = bounds,
    point          = if (theta_init_supplied &&
                         is.finite(as.numeric(res$point)[1L])) {
                       as.numeric(res$point)[1L]
                     } else {
                       NA_real_
                     },
    n              = NA_integer_,
    neighborhood   = NULL,
    divergence     = divergence,
    level          = NA_real_,
    B              = NA_integer_,
    estimand_label = "counterfactual prediction",
    call           = cl,
    details        = details
  )
}

# ---------------------------------------------------------------------
# Moments / gradient specification parsing (pure R; no Julia needed).
# ---------------------------------------------------------------------

#' Is a string a plausible Julia identifier?
#'
#' Restricting Julia names to this pattern also keeps the interpolated
#' `isdefined` checks free of code injection.
#' @keywords internal
#' @noRd
.tvb_is_julia_name <- function(x) {
  is.character(x) && length(x) == 1L &&
    grepl("^[A-Za-z_][A-Za-z0-9_!]*$", x)
}

#' Parse the `moments` argument of tvbounds_counterfactual()
#'
#' @param moments A length-2 character vector `c(file, fname)`, a single
#'   string naming a Julia function, or an R function.
#' @return A list with elements `type` (`"file"`, `"name"`, or
#'   `"rfun"`), `file`, `name`, `fun`, and a human-readable `label`.
#' @keywords internal
#' @noRd
.tvb_parse_moments <- function(moments) {
  if (is.function(moments)) {
    if (length(formals(moments)) < 2L) {
      stop("an R `moments` function must accept (theta, U, gamma) ",
           "(at least theta and U).", call. = FALSE)
    }
    return(list(type = "rfun", file = NULL, name = NULL, fun = moments,
                label = "R function"))
  }
  if (is.character(moments) && length(moments) == 2L) {
    file <- moments[[1L]]
    name <- moments[[2L]]
    if (!file.exists(file)) {
      stop("the Julia moments file does not exist: ", file, call. = FALSE)
    }
    if (!.tvb_is_julia_name(name)) {
      stop("`moments[2]` must be a valid Julia function name.",
           call. = FALSE)
    }
    return(list(type = "file", file = file, name = name, fun = NULL,
                label = sprintf("Julia function %s (from %s)", name,
                                basename(file))))
  }
  if (is.character(moments) && length(moments) == 1L) {
    if (!.tvb_is_julia_name(moments)) {
      stop("`moments` must be a valid Julia function name.", call. = FALSE)
    }
    return(list(type = "name", file = NULL, name = moments, fun = NULL,
                label = sprintf("Julia function %s", moments)))
  }
  stop("`moments` must be an R function, a single Julia function name, ",
       "or a length-2 character vector c(file, fname).", call. = FALSE)
}

#' Parse the `gradient` argument of tvbounds_counterfactual()
#'
#' @param gradient `NULL`, the string `"fd"`, the name of a Julia
#'   function, or an R function.
#' @param spec The parsed moments specification.
#' @return A list with elements `mode` (`"forwarddiff"`, `"fd"`, or
#'   `"user"`), `julia_name` (or `NULL`), `fun` (or `NULL`).
#' @keywords internal
#' @noRd
.tvb_parse_gradient <- function(gradient, spec) {
  if (is.null(gradient)) {
    if (spec$type == "rfun") {
      # ForwardDiff cannot differentiate through R code.
      return(list(mode = "fd", julia_name = NULL, fun = NULL))
    }
    return(list(mode = "forwarddiff", julia_name = NULL, fun = NULL))
  }
  if (is.function(gradient)) {
    return(list(mode = "user", julia_name = NULL, fun = gradient))
  }
  if (is.character(gradient) && length(gradient) == 1L) {
    if (identical(gradient, "fd")) {
      return(list(mode = "fd", julia_name = NULL, fun = NULL))
    }
    if (spec$type == "rfun") {
      stop("with an R `moments` function, `gradient` must be NULL, ",
           "\"fd\", or an R function.", call. = FALSE)
    }
    if (!.tvb_is_julia_name(gradient)) {
      stop("`gradient` must be NULL, \"fd\", the name of a Julia ",
           "function, or an R function.", call. = FALSE)
    }
    return(list(mode = "user", julia_name = gradient, fun = NULL))
  }
  stop("`gradient` must be NULL, \"fd\", the name of a Julia function, ",
       "or an R function.", call. = FALSE)
}

# ---------------------------------------------------------------------
# Julia staging helpers (require an initialized Julia session).
# ---------------------------------------------------------------------

#' Stage the moments function in the Julia session
#'
#' Includes the user's file and/or checks that the named function is
#' defined; for R moments, assigns the (gamma-currying) closure to Julia
#' and wraps it into a Bundle-compatible in-place function.
#'
#' @return The name (in `Main`) of the Julia-side moments function.
#' @keywords internal
#' @noRd
.tvb_stage_moments <- function(spec, gamma) {
  if (spec$type == "file") {
    JuliaCall::julia_assign("_tvb_moments_file",
                            normalizePath(spec$file, winslash = "/"))
    JuliaCall::julia_command("Base.include(Main, Main._tvb_moments_file);")
  }
  if (spec$type %in% c("file", "name")) {
    ok <- JuliaCall::julia_eval(
      sprintf('isdefined(Main, Symbol("%s"))', spec$name))
    if (!isTRUE(ok)) {
      stop(sprintf(
        "the Julia function `%s` is not defined in Main%s.", spec$name,
        if (spec$type == "file") {
          sprintf(" after including %s (it must be defined at the top level)",
                  basename(spec$file))
        } else {
          ""
        }), call. = FALSE)
    }
    return(spec$name)
  }
  # R function: curry gamma on the R side so the payload never crosses
  # the bridge, then wrap into the in-place Bundle signature in Julia.
  fn <- spec$fun
  wrapped <- if (length(formals(fn)) >= 3L) {
    function(theta, U) fn(theta, U, gamma)
  } else {
    function(theta, U) fn(theta, U)
  }
  JuliaCall::julia_assign("_tvb_r_moments_fun", wrapped)
  JuliaCall::julia_command(paste0(
    "_tvb_moments_rbridge = ",
    "Main.TVBoundsJulia.make_r_moments_wrapper(Main._tvb_r_moments_fun);"))
  "_tvb_moments_rbridge"
}

#' Stage the user gradient (if any) in the Julia session
#'
#' @return The name (in `Main`) of the Julia-side Jacobian function, or
#'   `""` when the gradient mode needs none.
#' @keywords internal
#' @noRd
.tvb_stage_gradient <- function(grad_spec, gamma) {
  if (grad_spec$mode != "user") return("")
  if (!is.null(grad_spec$julia_name)) {
    ok <- JuliaCall::julia_eval(
      sprintf('isdefined(Main, Symbol("%s"))', grad_spec$julia_name))
    if (!isTRUE(ok)) {
      stop(sprintf("the Julia gradient function `%s` is not defined in Main.",
                   grad_spec$julia_name), call. = FALSE)
    }
    return(grad_spec$julia_name)
  }
  fn <- grad_spec$fun
  wrapped <- if (length(formals(fn)) >= 3L) {
    function(theta, U) fn(theta, U, gamma)
  } else {
    function(theta, U) fn(theta, U)
  }
  JuliaCall::julia_assign("_tvb_r_gradient_fun", wrapped)
  JuliaCall::julia_command(paste0(
    "_tvb_jac_rbridge = ",
    "Main.TVBoundsJulia.make_r_jac_wrapper(Main._tvb_r_gradient_fun);"))
  "_tvb_jac_rbridge"
}
