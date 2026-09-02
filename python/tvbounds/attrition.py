"""User-facing estimator for randomized experiments with attrition (port of R/attrition.R).

Computational kernels live in ``tvbounds/_attrition_kernels.py``.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from ._attrition_kernels import (_Slim, _r_interaction, boot_summary,
                                 boot_tv_bounds, breakdown_delta, lee_bounds,
                                 tv_bounds)
from ._object import NA, format_call, message, new_tvbounds, warn
from ._rcompat import rmean_int
from ._rrng import RRNG

_NEIGHBORHOODS = ("tv", "contamination")


def _is_number(x) -> bool:
    """R's ``is.numeric(x) && length(x) == 1L`` for a Python scalar."""
    if isinstance(x, (bool, np.bool_)):
        return False
    if isinstance(x, (int, float, np.integer, np.floating)):
        return True
    if isinstance(x, np.ndarray) and x.size == 1 and \
            np.issubdtype(x.dtype, np.number):
        return True
    return False


def _scalar(x) -> float:
    return float(np.asarray(x).ravel()[0])


def _is_logical(x) -> bool:
    return isinstance(x, (bool, np.bool_))


def tvbounds_attrition(data, outcome, treatment, response,
                       covariates=None,
                       delta=np.linspace(0, 1, 101),
                       neighborhood="tv",
                       bootstrap=True, B=1000, cluster=None,
                       level=0.95, min_obs=5, seed=None,
                       verbose=False):
    """See the tvbounds manual."""
    cl = format_call("tvbounds_attrition", {
        "data": data, "outcome": outcome, "treatment": treatment,
        "response": response, "covariates": covariates, "delta": delta,
        "neighborhood": neighborhood, "bootstrap": bootstrap, "B": B,
        "cluster": cluster, "level": level, "min_obs": min_obs, "seed": seed,
        "verbose": verbose})
    # match.arg(neighborhood): the full choice vector (or None, as
    # match.arg(NULL)) selects its first element
    if neighborhood is None or (isinstance(neighborhood, (list, tuple)) and
                                tuple(neighborhood) == _NEIGHBORHOODS):
        neighborhood = _NEIGHBORHOODS[0]
    if isinstance(neighborhood, str) and neighborhood not in _NEIGHBORHOODS:
        # match.arg: a unique prefix is accepted ("cont" -> "contamination")
        hits = [c for c in _NEIGHBORHOODS if neighborhood and c.startswith(neighborhood)]
        if len(hits) == 1:
            neighborhood = hits[0]
    if not isinstance(neighborhood, str) or neighborhood not in _NEIGHBORHOODS:
        raise ValueError("`neighborhood` must be one of \"tv\", \"contamination\".")

    # ---- input validation ----------------------------------------------------
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a data frame.")

    def chk_col(x, arg):
        if not isinstance(x, str):
            raise ValueError(f"`{arg}` must be a single column name (string).")
        if x not in data.columns:
            raise ValueError(f"`{arg}` names a column (\"{x}\") not found in `data`.")
        return x

    outcome = chk_col(outcome, "outcome")
    treatment = chk_col(treatment, "treatment")
    response = chk_col(response, "response")
    if cluster is not None:
        cluster = chk_col(cluster, "cluster")
    if covariates is not None:
        if isinstance(covariates, str):
            covariates = [covariates]
        if not isinstance(covariates, (list, tuple)) or len(covariates) < 1 or \
                not all(isinstance(c, str) for c in covariates):
            raise ValueError("`covariates` must be a character vector of column names.")
        covariates = list(covariates)
        miss = [c for c in covariates if c not in data.columns]
        if len(miss) > 0:
            raise ValueError("`covariates` names columns not found in `data`: "
                             + ", ".join(miss))
        if data[covariates].isna().to_numpy().any():
            raise ValueError("`covariates` columns must not contain missing values; "
                             "recode missing values into an explicit category first.")

    def as_binary(v: pd.Series, arg):
        if pd.api.types.is_bool_dtype(v.dtype):
            if v.isna().any():
                raise ValueError(f"`{arg}` must be a binary 0/1 column with no missing values.")
            v = v.astype("int64")
        if not pd.api.types.is_numeric_dtype(v.dtype) or v.isna().any():
            raise ValueError(f"`{arg}` must be a binary 0/1 column with no missing values.")
        arr = v.to_numpy(dtype=np.float64)
        if not np.all((arr == 0) | (arr == 1)):
            raise ValueError(f"`{arg}` must be a binary 0/1 column with no missing values.")
        return arr.astype(np.int64)

    D = as_binary(data[treatment], "treatment")
    S = as_binary(data[response], "response")
    Ycol = data[outcome]
    if pd.api.types.is_bool_dtype(Ycol.dtype) or \
            not pd.api.types.is_numeric_dtype(Ycol.dtype):
        raise ValueError("`outcome` must be a numeric column.")
    Y = np.asarray(Ycol.astype("float64"), dtype=np.float64)
    if np.unique(D).size < 2:
        raise ValueError("`treatment` must contain both treated (1) and control (0) units.")
    if np.isnan(Y[S == 1]).any() and verbose:
        message("Some respondents have a missing outcome (item non-response); "
                "they are dropped from the outcome samples.")
    n_obs_t = int(np.sum((D == 1) & (S == 1) & ~np.isnan(Y)))
    n_obs_c = int(np.sum((D == 0) & (S == 1) & ~np.isnan(Y)))
    if n_obs_t < 2 or n_obs_c < 2:
        raise ValueError("`data` must contain at least two observed outcomes in each arm.")

    delta_msg = "`delta` must be a numeric vector of budget values in [0, 1]."
    if _is_logical(delta):
        raise ValueError(delta_msg)
    try:
        d_arr = np.atleast_1d(np.asarray(delta, dtype=np.float64)).ravel()
    except (TypeError, ValueError):
        raise ValueError(delta_msg) from None
    if np.asarray(delta).dtype == bool or d_arr.size < 1 or np.isnan(d_arr).any() or \
            np.any(d_arr < 0) or np.any(d_arr > 1):
        raise ValueError(delta_msg)
    delta = np.unique(d_arr)
    if not _is_number(level) or math.isnan(_scalar(level)) or \
            _scalar(level) <= 0 or _scalar(level) >= 1:
        raise ValueError("`level` must be a single number strictly between 0 and 1.")
    level = _scalar(level)
    if not _is_number(min_obs) or math.isnan(_scalar(min_obs)) or _scalar(min_obs) < 2:
        raise ValueError("`min_obs` must be a single number greater than or equal to 2.")
    min_obs = int(_scalar(min_obs))
    if not _is_logical(bootstrap):
        raise ValueError("`bootstrap` must be True or False.")
    bootstrap = bool(bootstrap)
    if bootstrap:
        if not _is_number(B) or math.isnan(_scalar(B)) or _scalar(B) < 2:
            raise ValueError("`B` must be a single integer greater than or equal to 2.")
        B = int(_scalar(B))

    # ---- assemble the internal slim data frame -------------------------------
    slim = _Slim(Y, S, D)
    if covariates is not None:
        st, levels = _r_interaction(data[covariates])
        slim.st = st
        slim.levels = levels
    cluster_id = data[cluster].to_numpy() if cluster is not None else None
    if cluster_id is not None and pd.isna(cluster_id).any():
        raise ValueError("`cluster` column must not contain missing values.")

    # The baseline (delta = 0) and Lee (delta = 1) endpoints are always
    # computed, even when the user's grid omits them.
    grid_full = np.unique(np.concatenate(([0.0], delta, [1.0])))
    idx_user = np.searchsorted(grid_full, delta)

    # ---- point-estimate bounds curve ----------------------------------------
    if verbose:
        message("Computing %s bounds on %d budget values ..."
                % ("total variation" if neighborhood == "tv" else "contamination",
                   grid_full.size))
    pt, info = tv_bounds(slim, grid_full, covs=covariates,
                         neighborhood=neighborhood, min_obs=min_obs,
                         verbose=verbose)
    tau_lower = pt["tau_lower"].to_numpy(dtype=np.float64)
    tau_upper = pt["tau_upper"].to_numpy(dtype=np.float64)
    if np.all(np.isnan(tau_lower)):
        raise RuntimeError("The bounds could not be computed on `data` (degenerate sample: "
                           "too few observed outcomes, an inadmissible complier share, or no "
                           "retained covariate cell).")

    if covariates is not None and len(info["dropped_small"]) > 0:
        warn(("%d of %d covariate cell(s) with fewer than min_obs = %d observed "
              "outcomes per arm were dropped: %s")
             % (len(info["dropped_small"]), info["n_strata_total"], min_obs,
                ", ".join(info["dropped_small"])))

    # ---- headline quantities -------------------------------------------------
    p1 = rmean_int(S[D == 1])
    p0 = rmean_int(S[D == 0])
    p_star = (p1 - p0) / p1
    p_star = p_star if p_star >= 0 else 0.0          # max((p1 - p0) / p1, 0)
    point = float(tau_lower[grid_full == 0][0])
    lee = {"lower": float(tau_lower[grid_full == 1][0]),
           "upper": float(tau_upper[grid_full == 1][0])}

    details = {
        "p_star": p_star,
        "response_rate": {"control": p0, "treated": p1},
        "n_by_arm": {"control": int(np.sum(D == 0)), "treated": int(np.sum(D == 1))},
        "n_respondents_by_arm": {"control": int(np.sum((D == 0) & (S == 1))),
                                 "treated": int(np.sum((D == 1) & (S == 1)))},
        "lee": lee,
        "breakdown": {
            "point": breakdown_delta(grid_full, tau_lower, tau_upper)}}
    if covariates is not None:
        details["lee_nocov"] = lee_bounds(slim, covs=None, min_obs=min_obs)
        details["pooled"] = {
            "strata": info["strata"],
            "coverage": info["coverage"],
            "mu0": info["mu0_AT"],
            "n_strata_total": info["n_strata_total"],
            "dropped_small": info["dropped_small"],
            "dropped_pstar": info["dropped_pstar"]}
        if neighborhood == "tv":
            details["pooled"]["pw"] = pd.DataFrame({
                "delta": delta,
                "lower": pt["tau_lower_pw"].to_numpy(dtype=np.float64)[idx_user],
                "upper": pt["tau_upper_pw"].to_numpy(dtype=np.float64)[idx_user]})
    else:
        details["mu0"] = info["mu0"]
    if cluster_id is not None:
        details["n_clusters"] = int(len(pd.unique(cluster_id)))

    bounds = pd.DataFrame({"delta": delta,
                           "lower": tau_lower[idx_user],
                           "upper": tau_upper[idx_user]})

    # ---- bootstrap -----------------------------------------------------------
    if bootstrap:
        if seed is not None:
            if not _is_number(seed) or math.isnan(_scalar(seed)):
                raise ValueError("`seed` must be a single number (or None).")
            if abs(_scalar(seed)) > 2147483647:
                # set.seed(): the seed must be representable as an R integer
                raise ValueError("supplied seed is not a valid integer")
            # Set the RNG locally: the R-compatible stream is an instance, so
            # the call has no side effect on any global RNG state.
            rng = RRNG(int(_scalar(seed)))
        else:
            rng = RRNG(int.from_bytes(os.urandom(4), "little"))
        if verbose:
            message("Bootstrapping (B = %d%s) ..."
                    % (B, "" if cluster is None
                       else ", %d clusters" % details["n_clusters"]))
        bt = boot_tv_bounds(slim, grid_full, covs=covariates,
                            neighborhood=neighborhood, B=B,
                            min_obs=min_obs, cluster=cluster_id, rng=rng)
        if bt["n_fail"] > 0:
            warn("%d of %d bootstrap replicates failed and were dropped."
                 % (bt["n_fail"], B))
        if np.all(np.isnan(bt["lo"])):
            raise RuntimeError("All bootstrap replicates failed or were degenerate; no inference "
                               "is available. Consider a larger sample or `bootstrap = False`.")
        bs = boot_summary(bt, level=level)

        bounds["lower_se"] = bs["lo_est_se"].to_numpy()[idx_user]
        bounds["upper_se"] = bs["up_est_se"].to_numpy()[idx_user]
        bounds["ci_lower"] = bs["lo_ci_low"].to_numpy()[idx_user]
        bounds["ci_upper"] = bs["up_ci_high"].to_numpy()[idx_user]

        details["breakdown"]["ci"] = breakdown_delta(
            grid_full, bs["lo_ci_low"].to_numpy(), bs["up_ci_high"].to_numpy())
        details["boot"] = {
            "n_fail": int(bt["n_fail"]),
            "n_eff": bs["n_eff"].to_numpy()[idx_user],
            "width_se": bs["w_est_se"].to_numpy()[idx_user]}
        # Store the raw bound draws (at the requested budgets) only when
        # memory-reasonable; otherwise the band and standard errors above are
        # the record.
        if B * delta.size <= 1e6:
            details["boot"]["draws"] = {"lower": bt["lo"][:, idx_user],
                                        "upper": bt["up"][:, idx_user]}

    return new_tvbounds(
        application="attrition",
        bounds=bounds,
        point=point,
        n=int(len(data)),
        neighborhood=neighborhood,
        level=level if bootstrap else NA,
        B=B if bootstrap else NA,
        estimand_label="treatment effect",
        call=cl,
        details=details,
    )
