"""End-to-end analysis pipeline.

Loads the committed synthetic data, runs the detector, the duration guidance,
and the naive-vs-debiased estimator on every experiment-metric series, and
returns structured results plus aggregate scores. Both run_demo.py and the
figure generator call analyze() so that numbers and figures always agree.
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

from . import config, io_utils, detector, duration, estimator, evaluate, projection


def load_series(cfg) -> tuple[list[dict], dict]:
    """Return the ground-truth rows and a dict keyed by (experiment, metric) of
    day-sorted numpy arrays (days, y, se)."""
    gt = io_utils.read_csv(os.path.join(config.DATA_DIR, "ground_truth.csv"))
    daily = io_utils.read_csv(os.path.join(config.DATA_DIR, "daily_observations.csv"))

    grouped = defaultdict(list)
    for r in daily:
        grouped[(r["experiment_id"], r["metric"])].append(r)

    series = {}
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda r: int(r["day"]))
        days = np.array([int(r["day"]) for r in rows], dtype=float)
        y = np.array([io_utils.to_float(r["obs_lift"]) for r in rows])
        se = np.array([io_utils.to_float(r["obs_lift_se"]) for r in rows])
        true_eff = np.array([io_utils.to_float(r["true_effect"]) for r in rows])
        series[key] = (days, y, se, true_eff)
    return gt, series


def analyze(cfg) -> dict:
    """Run the full analysis and return a results dictionary."""
    gt, series = load_series(cfg)
    rng = np.random.default_rng(cfg["seed"] + 101)  # separate stream for bootstrap
    K = cfg["estimator"]["decision_day"]
    k_grid = cfg["estimator"]["decision_day_grid"]

    per_series = []
    true_flags, pred_flags, true_cats, pred_cats = [], [], [], []

    # Per-K error accumulators for naive and debiased (all series and novelty-only).
    err_by_k = {k: {"naive": [], "debiased": []} for k in k_grid}
    err_by_k_nov = {k: {"naive": [], "debiased": []} for k in k_grid}

    # Per-category error accumulators at the primary decision day K.
    err_by_cat = defaultdict(lambda: {"naive": [], "debiased": []})

    dur_in_window = 0
    dur_validated = 0

    # Addition 3: shippable lift at the recommended run length.
    ship_err = []                              # debiased error at the recommended day, all series
    ship_cov = []                              # 1 if true L inside CI, else 0
    ship_err_by_cat = defaultdict(list)
    ship_cov_by_cat = defaultdict(list)

    # Addition 2: genuine-ramp long-term projections.
    ramp_projections = []

    # Addition 4: seasonality floor demonstration on seasonal series.
    seasonal_partial_err = []                  # |partial-week read - L_true|
    seasonal_full_err = []                     # |full-week read - L_true|

    rng_ship = np.random.default_rng(cfg["seed"] + 202)
    rng_proj = np.random.default_rng(cfg["seed"] + 303)
    floor_days = duration.seasonality_floor_days(cfg)

    def cum_naive(d):
        d = int(max(1, d))
        return estimator.naive_estimate(y[:d], se[:d])

    for g in gt:
        key = (g["experiment_id"], g["metric"])
        days, y, se, _ = series[key]
        L_true = float(g["L_true"])
        cat_true = g["category"]
        nov_true = int(g["is_novelty"])
        seasonal_true = int(g.get("is_seasonal", 0))

        # Detection on the FULL series (post-run assessment of novelty shape).
        det = detector.classify(days, y, se, cfg)
        true_flags.append(nov_true)
        pred_flags.append(int(det.is_novelty))
        true_cats.append(cat_true)
        pred_cats.append(det.category)

        # Duration guidance and validation.
        dg = duration.evaluate(days, y, se, det, L_true, cfg)
        if dg.reached_within_window:
            dur_in_window += 1
            dur_validated += int(dg.observed_within_tol)

        # Estimator at the primary decision day K.
        est = estimator.estimate_at(days[:K], y[:K], se[:K], cfg, rng)
        err_by_cat[cat_true]["naive"].append(est.naive - L_true)
        err_by_cat[cat_true]["debiased"].append(est.debiased - L_true)

        # Addition 3: the lift you would actually ship on, read at the
        # recommended run length (capped at the collected horizon).
        T = len(y)
        ship_day = int(min(dg.recommended_days, T))
        est_ship = estimator.estimate_at(days[:ship_day], y[:ship_day], se[:ship_day], cfg, rng_ship)
        ship_error = est_ship.debiased - L_true
        ship_covered = int(est_ship.debiased_lo <= L_true <= est_ship.debiased_hi)
        ship_err.append(ship_error)
        ship_cov.append(ship_covered)
        ship_err_by_cat[cat_true].append(ship_error)
        ship_cov_by_cat[cat_true].append(ship_covered)

        # Addition 2: project the long-term impact for series the framework flags
        # as still-climbing genuine ramps.
        proj_record = None
        if det.category == "genuine_ramp":
            pr = projection.project(days, y, se, cfg, rng_proj)
            H = pr["business_horizon_days"]
            true_H = float(projection.true_effect_at(H, L_true, float(g["A_true"]), float(g["tau_true"])))
            proj_record = {
                "experiment_id": g["experiment_id"],
                "metric": g["metric"],
                "category_true": cat_true,
                "asymptote_hat": pr["asymptote"],
                "asymptote_ci": list(pr["asymptote_ci"]),
                "L_true": L_true,
                "projected_at_horizon": {str(h): pr["point"][h] for h in pr["horizons"]},
                "projected_ci": {str(h): list(pr["ci"][h]) for h in pr["horizons"]},
                "business_horizon_days": H,
                "projected_yearly": pr["point"][H],
                "projected_yearly_ci": list(pr["ci"][H]),
                "true_yearly": true_H,
                "yearly_abs_error": abs(pr["point"][H] - true_H),
                "asymptote_abs_error": abs(pr["asymptote"] - L_true),
                "yearly_covered": int(pr["ci"][H][0] <= true_H <= pr["ci"][H][1]),
            }
            ramp_projections.append(proj_record)

        # Addition 4: on flat seasonal series (no novelty transient to confound
        # the comparison), contrast a partial-week read with a full-week read to
        # isolate the day-of-week bias the seasonality floor removes.
        if seasonal_true and cat_true == "flat" and floor_days > 1:
            partial_day = max(2, floor_days // 2)   # a mid-week partial read
            seasonal_partial_err.append(abs(cum_naive(partial_day) - L_true))
            seasonal_full_err.append(abs(cum_naive(floor_days) - L_true))

        # Estimator across the decision-day grid (point estimates only, fast).
        tau_cap_mult = cfg["estimator"].get("asymptote_tau_cap_mult", 2.0)
        for k in k_grid:
            e_naive = estimator.naive_estimate(y[:k], se[:k])
            dk = detector.classify(days[:k], y[:k], se[:k], cfg,
                                   max_tau=tau_cap_mult * (k))
            e_deb = e_naive if dk.category == "flat" else dk.L_hat
            err_by_k[k]["naive"].append(e_naive - L_true)
            err_by_k[k]["debiased"].append(e_deb - L_true)
            if nov_true:
                err_by_k_nov[k]["naive"].append(e_naive - L_true)
                err_by_k_nov[k]["debiased"].append(e_deb - L_true)

        per_series.append({
            "experiment_id": g["experiment_id"],
            "metric": g["metric"],
            "category_true": cat_true,
            "category_pred": det.category,
            "is_novelty_true": nov_true,
            "is_novelty_pred": int(det.is_novelty),
            "L_true": L_true,
            "A_true": float(g["A_true"]),
            "tau_true": float(g["tau_true"]),
            "L_hat": det.L_hat,
            "A_hat": det.A_hat,
            "tau_hat": det.tau_hat,
            "se_L": det.se_L,
            "f_pvalue": det.f_pvalue,
            "saturation": det.saturation,
            "detector_reason": det.reason,
            "is_seasonal_true": seasonal_true,
            "seasonal_amp_true": float(g.get("seasonal_amp_true", 0.0)),
            "naive_at_K": est.naive,
            "debiased_at_K": est.debiased,
            "debiased_ci_lo": est.debiased_lo,
            "debiased_ci_hi": est.debiased_hi,
            "naive_err_at_K": est.naive - L_true,
            "debiased_err_at_K": est.debiased - L_true,
            "recommended_days": dg.recommended_days,
            "decay_days": dg.decay_days,
            "seasonality_floor_days": dg.seasonality_floor_days,
            "floor_binds": dg.floor_binds,
            "recommended_in_window": dg.reached_within_window,
            "duration_validated": dg.observed_within_tol,
            "naive_cumulative_settle_days": dg.naive_cumulative_settle_days,
            "ship_day": ship_day,
            "shippable_debiased": est_ship.debiased,
            "shippable_ci_lo": est_ship.debiased_lo,
            "shippable_ci_hi": est_ship.debiased_hi,
            "shippable_err": ship_error,
            "shippable_covered": ship_covered,
            "low_confidence": est.low_confidence,
        })

    detection = {
        "binary": evaluate.binary_detection_metrics(true_flags, pred_flags),
        "category_accuracy": evaluate.category_accuracy(true_cats, pred_cats),
        "confusion_matrix": evaluate.confusion_matrix(true_cats, pred_cats),
    }

    # Estimator comparison: overall, per category, and across the K grid.
    all_naive = [r["naive_err_at_K"] for r in per_series]
    all_deb = [r["debiased_err_at_K"] for r in per_series]
    nov_naive = [r["naive_err_at_K"] for r in per_series if r["is_novelty_true"]]
    nov_deb = [r["debiased_err_at_K"] for r in per_series if r["is_novelty_true"]]

    estimator_cmp = {
        "decision_day": K,
        "overall": {
            "naive": evaluate.error_summary(all_naive),
            "debiased": evaluate.error_summary(all_deb),
        },
        "novelty_only": {
            "naive": evaluate.error_summary(nov_naive),
            "debiased": evaluate.error_summary(nov_deb),
        },
        "by_category": {
            cat: {
                "naive": evaluate.error_summary(err_by_cat[cat]["naive"]),
                "debiased": evaluate.error_summary(err_by_cat[cat]["debiased"]),
            }
            for cat in evaluate.CATEGORIES
        },
        "by_decision_day": {
            str(k): {
                "naive": evaluate.error_summary(err_by_k[k]["naive"]),
                "debiased": evaluate.error_summary(err_by_k[k]["debiased"]),
            }
            for k in k_grid
        },
        "by_decision_day_novelty_only": {
            str(k): {
                "naive": evaluate.error_summary(err_by_k_nov[k]["naive"]),
                "debiased": evaluate.error_summary(err_by_k_nov[k]["debiased"]),
            }
            for k in k_grid
        },
    }

    duration_summary = {
        "tol_abs": cfg["duration"]["tol_abs"],
        "seasonality_floor_days": floor_days,
        "n_series_where_floor_binds": sum(1 for r in per_series if r["floor_binds"]),
        "n_recommendations_in_window": dur_in_window,
        "n_validated": dur_validated,
        "validation_pass_rate": (dur_validated / dur_in_window) if dur_in_window else None,
        "median_recommended_days_by_category": {},
        "median_decay_days_by_category": {},
        "median_naive_settle_by_category": {},
    }
    for cat in evaluate.CATEGORIES:
        recs = [r["recommended_days"] for r in per_series if r["category_true"] == cat]
        decays = [r["decay_days"] for r in per_series if r["category_true"] == cat]
        naives = [r["naive_cumulative_settle_days"] for r in per_series if r["category_true"] == cat]
        if recs:
            duration_summary["median_recommended_days_by_category"][cat] = int(np.median(recs))
            duration_summary["median_decay_days_by_category"][cat] = int(np.median(decays))
            duration_summary["median_naive_settle_by_category"][cat] = int(np.median(naives))

    # Addition 3: shippable-lift accuracy at the recommended run length.
    shippable_summary = {
        "overall": {
            "error": evaluate.error_summary(ship_err),
            "interval_coverage": float(np.mean(ship_cov)) if ship_cov else None,
            "n": len(ship_err),
        },
        "by_category": {
            cat: {
                "error": evaluate.error_summary(ship_err_by_cat[cat]),
                "interval_coverage": (float(np.mean(ship_cov_by_cat[cat]))
                                      if ship_cov_by_cat[cat] else None),
                "n": len(ship_err_by_cat[cat]),
            }
            for cat in evaluate.CATEGORIES
        },
    }

    # Addition 2: ramp projection accuracy at the business horizon.
    yearly_errs = [p["yearly_abs_error"] for p in ramp_projections]
    asy_errs = [p["asymptote_abs_error"] for p in ramp_projections]
    covered = [p["yearly_covered"] for p in ramp_projections]
    projection_summary = {
        "business_horizon_days": cfg["projection"]["business_horizon_days"],
        "n_ramps_projected": len(ramp_projections),
        "yearly_projection": evaluate.error_summary(yearly_errs) if yearly_errs else None,
        "asymptote_projection": evaluate.error_summary(asy_errs) if asy_errs else None,
        "yearly_interval_coverage": float(np.mean(covered)) if covered else None,
        "projections": ramp_projections,
    }

    # Addition 4: seasonality floor demonstration.
    seasonality_summary = {
        "n_seasonal_series": int(sum(r["is_seasonal_true"] for r in per_series)),
        "floor_days": floor_days,
        "partial_read_day": max(2, floor_days // 2) if floor_days > 1 else 1,
        "comparison_scope": "flat seasonal series (novelty-free)",
        "mean_abs_error_partial_week": (float(np.mean(seasonal_partial_err))
                                        if seasonal_partial_err else None),
        "mean_abs_error_full_week": (float(np.mean(seasonal_full_err))
                                     if seasonal_full_err else None),
        "n_compared": len(seasonal_partial_err),
    }

    return {
        "per_series": per_series,
        "detection": detection,
        "estimator": estimator_cmp,
        "duration": duration_summary,
        "shippable": shippable_summary,
        "projection": projection_summary,
        "seasonality": seasonality_summary,
    }
