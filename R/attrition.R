# User-facing estimator for randomized experiments with attrition.
# Computational kernels live in R/attrition-kernels.R.

#' Sensitivity bounds for randomized experiments with attrition
#'
#' Computes sensitivity bounds on the average treatment effect for the
#' always-observed subpopulation of a randomized experiment with attrition,
#' following Palomba (2026). Writing \eqn{P_{*}}{P_*} for the baseline
#' distribution of the data, the estimand is
#' \deqn{\tau_{\mathsf{RCT}}(P_{*}) := \mathbb{E}_{P_{*}}[Y(1) - Y(0) \mid S(0) = 1, S(1) = 1],}{tau_RCT(P_*) := E_{P_*}[Y(1) - Y(0) | S(0) = 1, S(1) = 1],}
#' the average treatment effect on the units that respond under either arm.
#' The bounds are indexed by a budget \eqn{\delta \in [0, 1]}{delta in [0, 1]},
#' supplied through `delta`, which caps the total variation distance between
#' the outcome distribution of the compliers (units that respond only under
#' treatment) and that of the always-observed units,
#' \eqn{\mathsf{TV}(P_{*,\mathsf{C}}, P_{*,\mathsf{AO}}) \le \delta}{TV(P_{*,C}, P_{*,AO}) <= delta}.
#' At \eqn{\delta = 0}{delta = 0} the two distributions coincide and the
#' bounds collapse to the baseline difference in means among respondents (the
#' estimand under missingness completely at random); at
#' \eqn{\delta = 1}{delta = 1} the restriction is vacuous and they equal the
#' trimming bounds of Lee (2009). Optionally computes a nonparametric
#' bootstrap (with clustering) for standard errors and a percentile
#' confidence band, and covariate-pooled bounds that allocate a single budget
#' optimally across covariate cells.
#'
#' @section Setup and estimand:
#' Let \eqn{D \in \{0, 1\}}{D in {0, 1}} be the binary treatment and, for
#' \eqn{d \in \{0, 1\}}{d in {0, 1}}, let \eqn{Y(d)} be the potential outcome
#' and \eqn{S(d) \in \{0, 1\}}{S(d) in {0, 1}} the potential response
#' indicator. Their realized counterparts are
#' \deqn{Y = D Y(1) + (1 - D) Y(0), \qquad S = D S(1) + (1 - D) S(0),}{Y = D Y(1) + (1 - D) Y(0),  S = D S(1) + (1 - D) S(0),}
#' so that the observed data are \eqn{(YS, S, D)}: the outcome is recorded
#' only when \eqn{S = 1}. The response pattern \eqn{(S(0), S(1))} partitions
#' the population into always-observed units
#' \eqn{(S(0) = 1, S(1) = 1)}, never-observed units
#' \eqn{(S(0) = 0, S(1) = 0)}, compliers \eqn{(S(0) = 0, S(1) = 1)} and
#' defiers \eqn{(S(0) = 1, S(1) = 0)}; the estimand
#' \eqn{\tau_{\mathsf{RCT}}(P_{*})}{tau_RCT(P_*)} is the average treatment
#' effect on the first of these groups.
#'
#' Two assumptions are maintained. Random assignment enters as
#' \eqn{(S(0), S(1)) \perp\!\!\!\perp (Y(0), Y(1))}{(S(0), S(1)) independent of (Y(0), Y(1))},
#' labelled (MCAR), and the monotonicity condition of Lee (2009),
#' \eqn{S(1) \ge S(0)}{S(1) >= S(0)} almost surely, labelled (Mono) —
#' treatment never causes a unit that would respond under control to attrit —
#' rules out defiers. Under (Mono) the outcome distribution of the observed
#' treated, \eqn{P_{*,\mathsf{T}}}{P_{*,T}}, is a mixture of the complier and
#' always-observed outcome distributions,
#' \deqn{P_{*,\mathsf{T}} = p_0 P_{*,\mathsf{C}} + (1 - p_0) P_{*,\mathsf{AO}}, \qquad p_0 = 1 - \frac{\pi_0}{\pi_1},}{P_{*,T} = p_0 P_{*,C} + (1 - p_0) P_{*,AO},  p_0 = 1 - pi_0 / pi_1,}
#' where \eqn{\pi_1 := P_{*}[S = 1 \mid D = 1]}{pi_1 := P_*[S = 1 | D = 1]}
#' and \eqn{\pi_0 := P_{*}[S = 1 \mid D = 0]}{pi_0 := P_*[S = 1 | D = 0]} are
#' the arm-specific response rates and \eqn{p_0} is the complier share. The
#' always-observed control mean is identified,
#' \eqn{\mu_0^{\mathsf{AO}}(P_{*}) = \mathbb{E}_{P_{*}}[Y \mid D = 0, S = 1]}{mu_0^AO(P_*) = E_{P_*}[Y | D = 0, S = 1]},
#' whereas the always-observed treated mean
#' \eqn{\mu_1^{\mathsf{AO}}(P_{*})}{mu_1^AO(P_*)} is only partially
#' identified; the bounds on \eqn{\tau_{\mathsf{RCT}}(P_{*})}{tau_RCT(P_*)}
#' follow by subtracting the identified control mean.
#'
#' The complier share is estimated by
#' \eqn{\widehat{p}_0 = \max\{1 - \widehat{\pi}_0 / \widehat{\pi}_1, 0\}}{p_0-hat = max{1 - pi_0-hat / pi_1-hat, 0}}
#' and returned as `details$p_star`, and the estimated response rates
#' \eqn{(\widehat{\pi}_0, \widehat{\pi}_1)}{(pi_0-hat, pi_1-hat)} as
#' `details$response_rate`. A negative unconstrained estimate of \eqn{p_0} is
#' sampling noise under (Mono) and is projected to zero, in which case the
#' bounds collapse to the baseline difference in means at every budget.
#'
#' @section Neighborhoods:
#' Two robustness sets are available through `neighborhood`. Both are indexed
#' by the budget \eqn{\delta}{delta} and both restrict the unobserved
#' complier outcome distribution \eqn{P_{*,\mathsf{C}}}{P_{*,C}} relative to
#' the unobserved always-observed outcome distribution
#' \eqn{P_{*,\mathsf{AO}}}{P_{*,AO}}:
#'
#' * `"tv"` (default): the total variation neighborhood of the paper,
#'   \eqn{\mathsf{TV}(P_{*,\mathsf{C}}, P_{*,\mathsf{AO}}) \le \delta}{TV(P_{*,C}, P_{*,AO}) <= delta},
#'   so the two distributions may disagree on at most a
#'   \eqn{\delta}{delta} fraction of their mass.
#' * `"contamination"`: a one-sided strengthening in which
#'   \eqn{P_{*,\mathsf{C}}}{P_{*,C}} lies in the contamination neighborhood
#'   of \eqn{P_{*,\mathsf{AO}}}{P_{*,AO}} (Huber 1964), that is
#'   \eqn{P_{*,\mathsf{C}} = (1 - \delta) P_{*,\mathsf{AO}} + \delta R}{P_{*,C} = (1 - delta) P_{*,AO} + delta R}
#'   for some distribution \eqn{R}, equivalently
#'   \eqn{P_{*,\mathsf{C}} \ge (1 - \delta) P_{*,\mathsf{AO}}}{P_{*,C} >= (1 - delta) P_{*,AO}}
#'   as measures: a \eqn{(1 - \delta)}{(1 - delta)}-share of the compliers has
#'   outcomes distributed exactly like the always-observed units, and only the
#'   remaining \eqn{\delta}{delta}-share may differ arbitrarily.
#'
#' Under total variation the rescaling identity
#' \eqn{\mathsf{TV}(P_{*,\mathsf{C}}, P_{*,\mathsf{T}}) = (1 - p_0)\,\mathsf{TV}(P_{*,\mathsf{C}}, P_{*,\mathsf{AO}})}{TV(P_{*,C}, P_{*,T}) = (1 - p_0) TV(P_{*,C}, P_{*,AO})}
#' recenters the restriction on the identified distribution
#' \eqn{P_{*,\mathsf{T}}}{P_{*,T}}, so that
#' \eqn{P_{*,\mathsf{C}}}{P_{*,C}} ranges over the robustness set
#' \deqn{\mathcal{P}_{\mathsf{RCT}}(\delta, P_{*,\mathsf{T}}) := \{P \in \Delta(\mathcal{Y}) : \mathsf{TV}(P, P_{*,\mathsf{T}}) \le \delta(1 - p_0), \ p_0 P \le P_{*,\mathsf{T}}\},}{P_RCT(delta, P_{*,T}) := {P in Delta(Y) : TV(P, P_{*,T}) <= delta (1 - p_0), p_0 P <= P_{*,T}},}
#' where \eqn{\Delta(\mathcal{Y})}{Delta(Y)} denotes the distributions on the
#' outcome space and the second restriction is the mixture structure of the
#' observed treated arm. The resulting sensitivity bounds
#' \eqn{\underline{\tau}_{\mathsf{TV}}(\delta)}{tau_TV-lower(delta)} and
#' \eqn{\overline{\tau}_{\mathsf{TV}}(\delta)}{tau_TV-upper(delta)} on
#' \eqn{\tau_{\mathsf{RCT}}(P_{*})}{tau_RCT(P_*)} are available in closed
#' form as trimmed means of \eqn{P_{*,\mathsf{T}}}{P_{*,T}}. Writing
#' \eqn{F^{-1}_{P_{*,\mathsf{T}}}}{F^{-1}_{P_{*,T}}} for the quantile
#' function of the observed treated outcomes and
#' \deqn{s_{\mathsf{L}}(\delta) := F^{-1}_{P_{*,\mathsf{T}}}(p_0 \delta), \qquad s_{\mathsf{U}}(\delta) := F^{-1}_{P_{*,\mathsf{T}}}(1 - (1 - p_0) \delta),}{s_L(delta) := F^{-1}_{P_{*,T}}(p_0 delta),  s_U(delta) := F^{-1}_{P_{*,T}}(1 - (1 - p_0) delta),}
#' the upper bound is
#' \deqn{\overline{\tau}_{\mathsf{TV}}(\delta) = \mathbb{E}_{P_{*,\mathsf{T}}}[Y \mathbf{1}\{s_{\mathsf{L}}(\delta) < Y < s_{\mathsf{U}}(\delta)\}] + \frac{1}{1 - p_0} \mathbb{E}_{P_{*,\mathsf{T}}}[Y \mathbf{1}\{Y \ge s_{\mathsf{U}}(\delta)\}] - \mu_0^{\mathsf{AO}}(P_{*}),}{tau_TV-upper(delta) = E_{P_{*,T}}[Y 1{s_L(delta) < Y < s_U(delta)}] + (1 / (1 - p_0)) E_{P_{*,T}}[Y 1{Y >= s_U(delta)}] - mu_0^AO(P_*),}
#' and the lower bound is obtained symmetrically, trimming at
#' \eqn{t_{\mathsf{L}}(\delta) := F^{-1}_{P_{*,\mathsf{T}}}((1 - p_0) \delta)}{t_L(delta) := F^{-1}_{P_{*,T}}((1 - p_0) delta)}
#' and
#' \eqn{t_{\mathsf{U}}(\delta) := F^{-1}_{P_{*,\mathsf{T}}}(1 - p_0 \delta)}{t_U(delta) := F^{-1}_{P_{*,T}}(1 - p_0 delta)}.
#'
#' Under contamination, combining the same mixture identity with
#' \eqn{P_{*,\mathsf{C}} \ge (1 - \delta) P_{*,\mathsf{AO}}}{P_{*,C} >= (1 - delta) P_{*,AO}}
#' pins the density
#' \eqn{r := \mathrm{d}P_{*,\mathsf{C}} / \mathrm{d}P_{*,\mathsf{T}}}{r := dP_{*,C} / dP_{*,T}}
#' between \eqn{(1 - \delta) / (1 - \delta p_0)}{(1 - delta) / (1 - delta p_0)}
#' and \eqn{1 / p_0}{1 / p_0}, and the bounds are again trimmed means of
#' \eqn{P_{*,\mathsf{T}}}{P_{*,T}}, now with effective trimming mass
#' \eqn{\delta p_0}{delta p_0}. The contamination neighborhood is contained
#' in the total variation one at every budget, so its bounds are weakly
#' tighter, and the two families share the same endpoints: the baseline at
#' \eqn{\delta = 0}{delta = 0} and, at \eqn{\delta = 1}{delta = 1}, the Lee
#' (2009) bounds
#' \eqn{\underline{\tau}_{\mathsf{Lee}} = \mathbb{E}_{P_{*,\mathsf{T}}}[Y \mid Y \le y_{1 - p_0}] - \mu_0^{\mathsf{AO}}(P_{*})}{tau_Lee-lower = E_{P_{*,T}}[Y | Y <= y_{1 - p_0}] - mu_0^AO(P_*)}
#' and
#' \eqn{\overline{\tau}_{\mathsf{Lee}} = \mathbb{E}_{P_{*,\mathsf{T}}}[Y \mid Y \ge y_{p_0}] - \mu_0^{\mathsf{AO}}(P_{*})}{tau_Lee-upper = E_{P_{*,T}}[Y | Y >= y_{p_0}] - mu_0^AO(P_*)},
#' where \eqn{y_u := F^{-1}_{P_{*,\mathsf{T}}}(u)}{y_u := F^{-1}_{P_{*,T}}(u)}.
#'
#' Both bound families are monotone in the budget by construction: the
#' robustness sets are nested in \eqn{\delta}{delta}.
#'
#' @section Covariates:
#' When `covariates` is supplied, units are stratified on the interaction of
#' the covariate columns, which plays the role of a discrete covariate
#' \eqn{X}{X} with support \eqn{\mathcal{X}}{calX}. Cells with fewer than
#' `min_obs` observed outcomes in either arm are dropped with a warning, and the
#' retained cells are weighted by their control-respondent shares, which
#' under (Mono) are the covariate distribution of the always-observed
#' population,
#' \eqn{P_{*,X \mid D = 0, S = 1} = P_{*,X \mid \mathsf{AO}}}{P_{*,X | D = 0, S = 1} = P_{*,X | AO}}.
#' Within a cell the complier share \eqn{p_0(x)}{p_0(x)} and the observed
#' treated outcome distribution
#' \eqn{P_{*,\mathsf{T}}(x)}{P_{*,T}(x)} are identified, and the cell-level
#' construction is the one above.
#'
#' Two ways of spending the budget across cells are distinguished. The
#' within-stratum ("pointwise") restriction imposes
#' \eqn{\mathsf{TV}(P_{*,\mathsf{C}}(x), P_{*,\mathsf{AO}}(x)) \le \delta}{TV(P_{*,C}(x), P_{*,AO}(x)) <= delta}
#' in every cell separately, giving one robustness set
#' \eqn{\mathcal{P}^{\mathsf{pw}}_{\mathsf{RCT}}(\delta, P_{*,\mathsf{T}}(x))}{P_RCT^pw(delta, P_{*,T}(x))}
#' per cell, whereas the pooled restriction caps only the average departure,
#' \deqn{\int_{\mathcal{X}} \mathsf{TV}(P_{*,\mathsf{C}}(x), P_{*,\mathsf{AO}}(x)) \, \mathrm{d}P_{*,X \mid \mathsf{AO}}(x) \le \delta,}{integral over calX of TV(P_{*,C}(x), P_{*,AO}(x)) dP_{*,X | AO}(x) <= delta,}
#' and so allows heterogeneity across cells inside the single robustness set
#' \eqn{\mathcal{P}_{\mathsf{RCT},X}(\delta, P_{*,\mathsf{T}})}{P_{RCT,X}(delta, P_{*,T})}.
#' The pooled restriction is the weaker of the two, so
#' \eqn{\underline{\tau}_{\mathsf{TV},X}(\delta) \le \underline{\tau}^{\mathsf{pw}}_{\mathsf{TV},X}(\delta)}{tau_{TV,X}-lower(delta) <= tau_{TV,X}^pw-lower(delta)}
#' and
#' \eqn{\overline{\tau}^{\mathsf{pw}}_{\mathsf{TV},X}(\delta) \le \overline{\tau}_{\mathsf{TV},X}(\delta)}{tau_{TV,X}^pw-upper(delta) <= tau_{TV,X}-upper(delta)},
#' with equality at \eqn{\delta = 0}{delta = 0} and at
#' \eqn{\delta = 1}{delta = 1}, where both collapse to the covariate Lee
#' (2009) bounds
#' \eqn{\underline{\tau}_{\mathsf{Lee},X}}{tau_{Lee,X}-lower} and
#' \eqn{\overline{\tau}_{\mathsf{Lee},X}}{tau_{Lee,X}-upper}.
#'
#' For `neighborhood = "tv"` the reported bounds are the pooled (joint)
#' bounds \eqn{\underline{\tau}_{\mathsf{TV},X}(\delta)}{tau_{TV,X}-lower(delta)}
#' and \eqn{\overline{\tau}_{\mathsf{TV},X}(\delta)}{tau_{TV,X}-upper(delta)}:
#' the budget allocation
#' \eqn{t : \mathcal{X} \to \mathbb{R}_+}{t : calX -> R_+} subject to
#' \eqn{\mathbb{E}_{P_{*,\mathsf{T}}}[t(X)] \le (1 - p_0) \delta}{E_{P_{*,T}}[t(X)] <= (1 - p_0) delta}
#' is solved exactly by a greedy fill over the stratum value functions
#' \eqn{V(t; x)}{V(t; x)}, which are piecewise linear in the cell budget and
#' concave for the upper bound and convex for the lower one. The
#' within-stratum bounds
#' \eqn{\underline{\tau}^{\mathsf{pw}}_{\mathsf{TV},X}(\delta)}{tau_{TV,X}^pw-lower(delta)}
#' and
#' \eqn{\overline{\tau}^{\mathsf{pw}}_{\mathsf{TV},X}(\delta)}{tau_{TV,X}^pw-upper(delta)}
#' are returned in `details$pooled$pw` for reference. For
#' `neighborhood = "contamination"` the reported bounds impose the common
#' budget `delta` within every retained cell and aggregate; they remain
#' weakly inside the total variation bounds at every budget.
#'
#' @section Inference:
#' The bootstrap resamples the full observation
#' \eqn{W = (YS, S, D)}{W = (YS, S, D)}, together with the covariates, with
#' replacement — whole clusters when `cluster` is supplied — and recomputes
#' the entire bounds curve on each replicate, yielding
#' \eqn{\widehat{\overline{\tau}}^{(b)}(\delta)}{tau-upper-hat^(b)(delta)},
#' \eqn{b = 1, \dots, B}{b = 1, ..., B}. Each replicate therefore redraws the
#' arm-specific response rates and hence \eqn{\widehat{p}_0}{p_0-hat}, so the
#' reported standard errors carry the estimation uncertainty in
#' \eqn{p_0}{p_0}, which the naive variance
#' \eqn{\widehat{\sigma}^2_{\mathsf{naive}}(\delta)}{sigma^2_naive(delta)}
#' omits by treating \eqn{\widehat{p}_0}{p_0-hat} as known; in the paper's
#' influence function
#' \eqn{\psi_{\mathsf{full}}(W; \delta)}{psi_full(W; delta)} this uncertainty
#' is the term
#' \eqn{\varkappa(\delta) \psi_{p_0}(W)}{varkappa(delta) psi_{p_0}(W)}. The reported
#' `upper_se` is the standard deviation of the draws
#' \eqn{\widehat{\overline{\tau}}^{(b)}(\delta)}{tau-upper-hat^(b)(delta)}
#' across replicates, that is the paper's
#' \eqn{\widehat{\sigma}_{\mathsf{boot}}(\delta)}{sigma_boot(delta)} divided
#' by \eqn{\sqrt{n}}{sqrt(n)} for a sample of size \eqn{n}{n}, and
#' `lower_se` is its counterpart for the lower bound. The reported confidence
#' band is the percentile band: writing \eqn{\alpha}{alpha} for the value of
#' `1 - level`, `ci_lower` is the \eqn{\alpha / 2}{alpha / 2} quantile of the
#' lower-bound draws and `ci_upper` the
#' \eqn{1 - \alpha / 2}{1 - alpha / 2} quantile of the upper-bound draws, the
#' outer envelope of the identified set. Replicates on which the bounds
#' cannot be computed are dropped and counted (a warning reports their
#' number).
#'
#' @param data A data frame containing the columns named by `outcome`,
#'   `treatment`, `response`, and (optionally) `covariates` and `cluster`.
#' @param outcome String; name of the numeric column holding the outcome
#'   \eqn{Y}. May be `NA` for non-respondents (and for the occasional
#'   respondent with item non-response, which is dropped from the outcome
#'   samples).
#' @param treatment String; name of the binary 0/1 column holding the
#'   treatment \eqn{D} (1 = treated).
#' @param response String; name of the binary 0/1 column holding the response
#'   indicator \eqn{S} (1 = outcome observed).
#' @param covariates Optional character vector of column names to stratify
#'   on, forming the discrete covariate \eqn{X}; cells are formed by their
#'   interaction. The covariate columns must be free of missing values —
#'   recode missing values into an explicit category first. Default `NULL`
#'   (no stratification).
#' @param delta Numeric vector of budget values
#'   \eqn{\delta \in [0, 1]}{delta in [0, 1]} at which the bounds are
#'   evaluated; sorted and de-duplicated internally. The baseline
#'   (\eqn{\delta = 0}{delta = 0}) and Lee (\eqn{\delta = 1}{delta = 1})
#'   endpoints are always computed internally even when absent from `delta`.
#'   Default `seq(0, 1, by = 0.01)`.
#' @param neighborhood Either `"tv"` (total variation, default) or
#'   `"contamination"`; see the Neighborhoods section.
#' @param bootstrap Logical; compute bootstrap standard errors and the
#'   percentile confidence band. Default `TRUE`.
#' @param B Number of bootstrap replications \eqn{B}. Default `1000`.
#' @param cluster Optional string; name of a cluster identifier column.
#'   When supplied, the bootstrap resamples whole clusters with
#'   replacement. Default `NULL` (units resampled independently).
#' @param level Confidence level of the percentile band,
#'   \eqn{1 - \alpha}{1 - alpha} in the notation of the Inference section.
#'   Default `0.95`.
#' @param min_obs Minimum number of observed outcomes per arm for a
#'   covariate cell to be retained. Default `5`.
#' @param seed Optional integer seed for the bootstrap. When non-`NULL`,
#'   the random-number-generator state is set locally and restored on exit,
#'   so the call has no side effect on the caller's RNG. Default `NULL`.
#' @param verbose Logical; emit progress messages. Default `FALSE`.
#'
#' @return An object of class `c("tvbounds_attrition", "tvbounds")`: a list
#'   with components
#'   * `application`: `"attrition"`.
#'   * `bounds`: data frame with one row per requested budget and columns
#'     `delta`, `lower`, `upper`, holding the sensitivity bounds
#'     \eqn{\underline{\tau}_{\mathsf{TV}}(\delta)}{tau_TV-lower(delta)} and
#'     \eqn{\overline{\tau}_{\mathsf{TV}}(\delta)}{tau_TV-upper(delta)} (or
#'     their pooled covariate counterparts
#'     \eqn{\underline{\tau}_{\mathsf{TV},X}(\delta)}{tau_{TV,X}-lower(delta)}
#'     and
#'     \eqn{\overline{\tau}_{\mathsf{TV},X}(\delta)}{tau_{TV,X}-upper(delta)}
#'     when `covariates` is supplied), plus, when `bootstrap = TRUE`,
#'     `lower_se`, `upper_se`, `ci_lower`, `ci_upper` (outer percentile
#'     band endpoints at `level`).
#'   * `point`: the estimate of \eqn{\tau_{\mathsf{RCT}}(P_{*})}{tau_RCT(P_*)}
#'     at \eqn{\delta = 0}{delta = 0}, where the bounds collapse to the
#'     difference in means among respondents
#'     \eqn{\tau_{\mathsf{MCAR}}(P_{*})}{tau_MCAR(P_*)} (with `covariates`,
#'     to its covariate-weighted analogue).
#'   * `n`: number of rows of `data` used.
#'   * `neighborhood`, `level`, `B`, `estimand_label`, `call` as documented
#'     in the package overview (`level` and `B` are `NA` without
#'     inference).
#'   * `details`: list with `p_star` (the complier share \eqn{p_0}{p_0},
#'     which also sets the trimming mass), `response_rate` (the arm-specific
#'     response rates \eqn{(\widehat{\pi}_0, \widehat{\pi}_1)}{(pi_0-hat, pi_1-hat)})
#'     and `n_by_arm` / `n_respondents_by_arm` (named, control and treated),
#'     `lee` (list with the Lee-endpoint `lower` / `upper`,
#'     \eqn{\underline{\tau}_{\mathsf{Lee}}}{tau_Lee-lower} and
#'     \eqn{\overline{\tau}_{\mathsf{Lee}}}{tau_Lee-upper}, of the estimated
#'     specification), `lee_nocov` (when covariates are used), `breakdown`
#'     (list with the point-estimate breakdown budget
#'     \eqn{\delta_b}{delta_b} at the reference value
#'     \eqn{\tau_{\star} = 0}{tau_star = 0} and, with inference, the
#'     confidence-band breakdown budget; `NA` when censored beyond one),
#'     `pooled` (with covariates: per-stratum table, coverage, dropped cells,
#'     and the within-stratum reference curve `pw`,
#'     \eqn{\underline{\tau}^{\mathsf{pw}}_{\mathsf{TV},X}(\delta)}{tau_{TV,X}^pw-lower(delta)}
#'     and
#'     \eqn{\overline{\tau}^{\mathsf{pw}}_{\mathsf{TV},X}(\delta)}{tau_{TV,X}^pw-upper(delta)},
#'     under total variation), `n_clusters` (with `cluster`), and `boot`
#'     (replicate counts, width standard errors, and — when
#'     memory-reasonable — the matrices of bound draws).
#'
#' @examples
#' set.seed(123)
#' n <- 400
#' d <- rbinom(n, 1, 0.5)
#' s <- rbinom(n, 1, ifelse(d == 1, 0.9, 0.7))
#' y <- ifelse(s == 1, rnorm(n, mean = 0.3 * d), NA)
#' x <- rbinom(n, 1, 0.5)
#' dat <- data.frame(y = y, d = d, s = s, x = x)
#'
#' ## Total variation bounds with a small bootstrap
#' fit <- tvbounds_attrition(dat, outcome = "y", treatment = "d",
#'   response = "s", delta = seq(0, 1, by = 0.1), B = 50, seed = 1)
#' fit$bounds
#' fit$details$lee
#'
#' ## Contamination neighborhood, no inference: weakly tighter bounds
#' fit_c <- tvbounds_attrition(dat, outcome = "y", treatment = "d",
#'   response = "s", delta = seq(0, 1, by = 0.1),
#'   neighborhood = "contamination", bootstrap = FALSE)
#' all(fit_c$bounds$lower >= fit$bounds$lower - 1e-12)
#'
#' ## Covariate-pooled bounds
#' fit_x <- tvbounds_attrition(dat, outcome = "y", treatment = "d",
#'   response = "s", covariates = "x", delta = seq(0, 1, by = 0.1),
#'   bootstrap = FALSE)
#'
#' @references
#' Palomba, F. (2026). "Sensitivity Analysis in Population Shares." Working
#' paper.
#'
#' Lee, D. S. (2009). "Training, Wages, and Sample Selection: Estimating
#' Sharp Bounds on Treatment Effects." *Review of Economic Studies*, 76(3),
#' 1071-1102.
#'
#' Huber, P. J. (1964). "Robust Estimation of a Location Parameter."
#' *Annals of Mathematical Statistics*, 35(1), 73-101.
#'
#' @seealso [tvbounds_plot()] and [tvbounds_summary()] for reporting.
#' @export
tvbounds_attrition <- function(data, outcome, treatment, response,
                               covariates = NULL,
                               delta = seq(0, 1, by = 0.01),
                               neighborhood = c("tv", "contamination"),
                               bootstrap = TRUE, B = 1000, cluster = NULL,
                               level = 0.95, min_obs = 5, seed = NULL,
                               verbose = FALSE) {

  cl <- match.call()
  neighborhood <- match.arg(neighborhood)

  # ---- input validation ----------------------------------------------------
  if (!is.data.frame(data))
    stop("`data` must be a data frame.", call. = FALSE)
  chk_col <- function(x, arg) {
    if (!is.character(x) || length(x) != 1L || is.na(x))
      stop(sprintf("`%s` must be a single column name (string).", arg),
           call. = FALSE)
    if (!x %in% names(data))
      stop(sprintf("`%s` names a column (\"%s\") not found in `data`.",
                   arg, x), call. = FALSE)
    x
  }
  outcome   <- chk_col(outcome, "outcome")
  treatment <- chk_col(treatment, "treatment")
  response  <- chk_col(response, "response")
  if (!is.null(cluster)) cluster <- chk_col(cluster, "cluster")
  if (!is.null(covariates)) {
    if (!is.character(covariates) || length(covariates) < 1L)
      stop("`covariates` must be a character vector of column names.",
           call. = FALSE)
    miss <- setdiff(covariates, names(data))
    if (length(miss) > 0L)
      stop("`covariates` names columns not found in `data`: ",
           paste(miss, collapse = ", "), call. = FALSE)
    if (anyNA(data[covariates]))
      stop("`covariates` columns must not contain missing values; ",
           "recode missing values into an explicit category first.",
           call. = FALSE)
  }

  as_binary <- function(v, arg) {
    if (is.logical(v)) v <- as.integer(v)
    if (!is.numeric(v) || anyNA(v) || !all(v %in% c(0, 1)))
      stop(sprintf("`%s` must be a binary 0/1 column with no missing values.",
                   arg), call. = FALSE)
    as.integer(v)
  }
  D <- as_binary(data[[treatment]], "treatment")
  S <- as_binary(data[[response]], "response")
  Y <- data[[outcome]]
  if (!is.numeric(Y))
    stop("`outcome` must be a numeric column.", call. = FALSE)
  if (length(unique(D)) < 2L)
    stop("`treatment` must contain both treated (1) and control (0) units.",
         call. = FALSE)
  if (anyNA(Y[S == 1]) && verbose)
    message("Some respondents have a missing outcome (item non-response); ",
            "they are dropped from the outcome samples.")
  n_obs_t <- sum(D == 1 & S == 1 & !is.na(Y))
  n_obs_c <- sum(D == 0 & S == 1 & !is.na(Y))
  if (n_obs_t < 2L || n_obs_c < 2L)
    stop("`data` must contain at least two observed outcomes in each arm.",
         call. = FALSE)

  if (!is.numeric(delta) || length(delta) < 1L || anyNA(delta) ||
      any(delta < 0) || any(delta > 1))
    stop("`delta` must be a numeric vector of budget values in [0, 1].",
         call. = FALSE)
  delta <- sort(unique(as.numeric(delta)))
  if (!is.numeric(level) || length(level) != 1L || is.na(level) ||
      level <= 0 || level >= 1)
    stop("`level` must be a single number strictly between 0 and 1.",
         call. = FALSE)
  if (!is.numeric(min_obs) || length(min_obs) != 1L || is.na(min_obs) ||
      min_obs < 2)
    stop("`min_obs` must be a single number greater than or equal to 2.",
         call. = FALSE)
  min_obs <- as.integer(min_obs)
  if (!is.logical(bootstrap) || length(bootstrap) != 1L || is.na(bootstrap))
    stop("`bootstrap` must be TRUE or FALSE.", call. = FALSE)
  if (bootstrap) {
    if (!is.numeric(B) || length(B) != 1L || is.na(B) || B < 2)
      stop("`B` must be a single integer greater than or equal to 2.",
           call. = FALSE)
    B <- as.integer(B)
  }

  # ---- assemble the internal slim data frame -------------------------------
  slim <- data.frame(Y = as.numeric(Y), S = S, D = D)
  if (!is.null(covariates))
    slim$.tvb_stratum <- interaction(data[covariates], drop = TRUE)
  cluster_id <- if (!is.null(cluster)) data[[cluster]] else NULL
  if (!is.null(cluster_id) && anyNA(cluster_id))
    stop("`cluster` column must not contain missing values.", call. = FALSE)

  # The baseline (delta = 0) and Lee (delta = 1) endpoints are always
  # computed, even when the user's grid omits them.
  grid_full <- sort(unique(c(0, delta, 1)))
  idx_user  <- match(delta, grid_full)

  # ---- point-estimate bounds curve ----------------------------------------
  if (verbose)
    message(sprintf("Computing %s bounds on %d budget values ...",
                    if (neighborhood == "tv") "total variation"
                    else "contamination", length(grid_full)))
  pt <- tv_bounds(slim, grid_full, covs = covariates,
                  neighborhood = neighborhood, min_obs = min_obs,
                  verbose = verbose)
  if (all(is.na(pt$tau_lower)))
    stop("The bounds could not be computed on `data` (degenerate sample: ",
         "too few observed outcomes, an inadmissible complier share, or no ",
         "retained covariate cell).", call. = FALSE)
  info <- attr(pt, "info")

  if (!is.null(covariates) && length(info$dropped_small) > 0L)
    warning(sprintf(paste0("%d of %d covariate cell(s) with fewer than ",
                           "min_obs = %d observed outcomes per arm were ",
                           "dropped: %s"),
                    length(info$dropped_small), info$n_strata_total, min_obs,
                    paste(info$dropped_small, collapse = ", ")),
            call. = FALSE)

  # ---- headline quantities -------------------------------------------------
  p1 <- mean(S[D == 1]); p0 <- mean(S[D == 0])
  p_star <- max((p1 - p0) / p1, 0)
  point <- pt$tau_lower[grid_full == 0]
  lee <- list(lower = pt$tau_lower[grid_full == 1],
              upper = pt$tau_upper[grid_full == 1])

  details <- list(
    p_star = p_star,
    response_rate = c(control = p0, treated = p1),
    n_by_arm = c(control = sum(D == 0), treated = sum(D == 1)),
    n_respondents_by_arm = c(control = sum(D == 0 & S == 1),
                             treated = sum(D == 1 & S == 1)),
    lee = lee,
    breakdown = list(
      point = breakdown_delta(grid_full, pt$tau_lower, pt$tau_upper)))
  if (!is.null(covariates)) {
    details$lee_nocov <- lee_bounds(slim, covs = NULL, min_obs = min_obs)
    details$pooled <- list(
      strata = info$strata,
      coverage = info$coverage,
      mu0 = info$mu0_AT,
      n_strata_total = info$n_strata_total,
      dropped_small = info$dropped_small,
      dropped_pstar = info$dropped_pstar)
    if (neighborhood == "tv")
      details$pooled$pw <- data.frame(delta = delta,
                                      lower = pt$tau_lower_pw[idx_user],
                                      upper = pt$tau_upper_pw[idx_user])
  } else {
    details$mu0 <- info$mu0
  }
  if (!is.null(cluster_id))
    details$n_clusters <- length(unique(cluster_id))

  bounds <- data.frame(delta = delta,
                       lower = pt$tau_lower[idx_user],
                       upper = pt$tau_upper[idx_user])

  # ---- bootstrap -----------------------------------------------------------
  if (bootstrap) {
    if (!is.null(seed)) {
      if (!is.numeric(seed) || length(seed) != 1L || is.na(seed))
        stop("`seed` must be a single number (or NULL).", call. = FALSE)
      # Set the RNG locally: save and restore .Random.seed so the call has
      # no side effect on the caller's RNG state.
      if (!exists(".Random.seed", envir = globalenv(), inherits = FALSE))
        stats::runif(1)
      old_seed <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
      on.exit(assign(".Random.seed", old_seed, envir = globalenv()),
              add = TRUE)
      set.seed(seed)
    }
    if (verbose)
      message(sprintf("Bootstrapping (B = %d%s) ...", B,
                      if (is.null(cluster)) ""
                      else sprintf(", %d clusters", details$n_clusters)))
    bt <- boot_tv_bounds(slim, grid_full, covs = covariates,
                         neighborhood = neighborhood, B = B,
                         min_obs = min_obs, cluster = cluster_id)
    if (bt$n_fail > 0L)
      warning(sprintf("%d of %d bootstrap replicates failed and were dropped.",
                      bt$n_fail, B), call. = FALSE)
    if (all(is.na(bt$lo)))
      stop("All bootstrap replicates failed or were degenerate; no inference ",
           "is available. Consider a larger sample or `bootstrap = FALSE`.",
           call. = FALSE)
    bs <- boot_summary(bt, level = level)

    bounds$lower_se <- bs$lo_est_se[idx_user]
    bounds$upper_se <- bs$up_est_se[idx_user]
    bounds$ci_lower <- bs$lo_ci_low[idx_user]
    bounds$ci_upper <- bs$up_ci_high[idx_user]

    details$breakdown$ci <- breakdown_delta(grid_full, bs$lo_ci_low,
                                            bs$up_ci_high)
    details$boot <- list(
      n_fail = bt$n_fail,
      n_eff = bs$n_eff[idx_user],
      width_se = bs$w_est_se[idx_user])
    # Store the raw bound draws (at the requested budgets) only when
    # memory-reasonable; otherwise the band and standard errors above are
    # the record.
    if (B * length(delta) <= 1e6)
      details$boot$draws <- list(lower = bt$lo[, idx_user, drop = FALSE],
                                 upper = bt$up[, idx_user, drop = FALSE])
  }

  new_tvbounds(
    application    = "attrition",
    bounds         = bounds,
    point          = point,
    n              = nrow(data),
    neighborhood   = neighborhood,
    level          = if (bootstrap) level else NA_real_,
    B              = if (bootstrap) B else NA_integer_,
    estimand_label = "treatment effect",
    call           = cl,
    details        = details
  )
}
