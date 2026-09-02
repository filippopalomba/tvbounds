# Package-level documentation and centralized imports.

#' tvbounds: Sensitivity Analysis and Bounds under Total Variation
#' Neighborhoods
#'
#' Implements the sensitivity analysis framework of Palomba (2026),
#' "Sensitivity Analysis in Population Shares". The estimand is an
#' expectation \eqn{\mathbb{E}_{P}[g(Z;\theta)]}{E_P[g(Z; theta)]} of a
#' known integrand \eqn{g}{g} under a distribution \eqn{P}{P} of the data
#' \eqn{Z}{Z}. Rather than committing to a single baseline distribution
#' \eqn{P_{*}}{P_*}, the package lets \eqn{P}{P} range over a robustness
#' set
#' \deqn{\mathcal{P}_{\phi}(\theta;\rho,P_{*},\delta) = \{P : \mathbb{E}_{P}[m(Z;\theta)] \in \mathcal{M}(\rho),\ D_{\phi}(P \| P_{*}) \le \delta\},}{P_phi(theta; rho, P_*, delta) = {P : E_P[m(Z; theta)] in M(rho), D_phi(P || P_*) <= delta},}
#' collecting the distributions that remain compatible with the moment
#' restrictions and lie within a divergence budget \eqn{\delta}{delta} of
#' the baseline, and reports the resulting sensitivity bounds
#' \deqn{\inf_{\theta \in \Theta}\ \inf_{P \in \mathcal{P}_{\phi}(\theta;\rho,P_{*},\delta)} \mathbb{E}_{P}[g(Z;\theta)] \qquad \mathrm{and} \qquad \sup_{\theta \in \Theta}\ \sup_{P \in \mathcal{P}_{\phi}(\theta;\rho,P_{*},\delta)} \mathbb{E}_{P}[g(Z;\theta)].}{inf_theta inf_P E_P[g(Z; theta)]  and  sup_theta sup_P E_P[g(Z; theta)].}
#'
#' The sensitivity parameter is the budget \eqn{\delta}{delta}, supplied
#' through the `delta` argument. Under the total variation entropy
#' \eqn{\phi_{\mathsf{TV}}(s) = |s - 1| / 2}{phi_TV(s) = |s - 1| / 2} the
#' divergence is the total variation distance
#' \eqn{\mathsf{TV}(P,P_{*})}{TV(P, P_*)} and \eqn{\delta \in [0,1]}{delta
#' in [0, 1]} bounds the fraction of baseline probability mass that may be
#' misspecified; under the contamination neighborhood
#' \eqn{\mathcal{C}_{\kappa}(P_{*}) = \{P = \kappa P_{*} + (1 - \kappa) R\}}{C_kappa(P_*) = {P = kappa P_* + (1 - kappa) R}}
#' the perturbed distribution is a mixture of the baseline distribution and
#' an arbitrary distribution \eqn{R}{R}.
#'
#' @section Applications:
#' Three ready-made interfaces cover the paper's empirical applications:
#' * [tvbounds_attrition()] — randomized experiments with attrition:
#'   total variation and contamination bounds on the treatment effect,
#'   bootstrap inference, and optional covariate-pooled bounds. At budget
#'   \eqn{\delta = 1}{delta = 1} the bounds reproduce the Lee (2009)
#'   worst-case bounds.
#' * [tvbounds_counterfactual()] — counterfactual predictions in structural
#'   models, through an interface to Julia and the Artelys KNITRO solver;
#'   the only function supporting general entropy functions
#'   \eqn{\phi}{phi}, following Christensen and Connault (2023).
#' * [tvbounds_riv()] — recentered instrumental variables / formula
#'   instruments, as in Borusyak and Hull (2023), with first-stage
#'   breakdown budgets and no bootstrap inference (by design).
#'
#' @section Reporting:
#' All estimators return a common `tvbounds` object carrying the bound
#' paths \eqn{\underline{\tau}(\delta)}{tau_lower(delta)} and
#' \eqn{\overline{\tau}(\delta)}{tau_upper(delta)} over a grid of budgets.
#' [tvbounds_plot()] (or `plot()`) displays the bounds against the budget,
#' and [tvbounds_summary()] (or `summary()`) computes the summary measures
#' of the paper: the breakdown budget
#' \eqn{\delta_{b}(\tau_{\star})}{delta_b(tau_star)} and its certified
#' counterpart, the shadow price of robustness
#' \eqn{\underline{\eta}(\delta)}{eta(delta)}, the robustness standard
#' error \eqn{\varsigma_{b} = \sigma(\delta_{b}) / \underline{\eta}(\delta_{b})}{varsigma_b = sigma(delta_b) / eta(delta_b)},
#' and the certification frontier
#' \eqn{n^{\star}(\delta;\alpha)}{n*(delta; alpha)}.
#'
#' @section KNITRO requirement:
#' [tvbounds_counterfactual()] relies on Julia and the commercial Artelys
#' KNITRO solver, which requires a valid license; see that function's help
#' page for details.
#'
#' @references
#' Palomba, F. (2026). "Sensitivity Analysis in Population Shares."
#' Working paper.
#'
#' Borusyak, K. and Hull, P. (2023). "Nonrandom Exposure to Exogenous
#' Shocks." \emph{Econometrica}, 91(6), 2155--2185.
#'
#' Christensen, T. and Connault, B. (2023). "Counterfactual Sensitivity and
#' Robustness." \emph{Econometrica}, 91(1), 263--298.
#'
#' Lee, D. S. (2009). "Training, Wages, and Sample Selection: Estimating
#' Sharp Bounds on Treatment Effects." \emph{Review of Economic Studies},
#' 76(3), 1071--1102.
#'
#' @keywords internal
#' @importFrom stats approx qnorm quantile sd
#' @importFrom utils globalVariables
"_PACKAGE"

# Column names of the `bounds` data frame that appear inside ggplot2::aes()
# calls in R/plot.R; declared so R CMD check does not flag them as unbound.
globalVariables(c("delta", "lower", "upper", "ci_lower", "ci_upper"))
