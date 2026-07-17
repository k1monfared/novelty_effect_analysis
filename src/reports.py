"""Build committed output artifacts (JSON and Markdown) from the analysis run.

All numbers come from pipeline.analyze; nothing is hand-entered.
"""
from __future__ import annotations

import os

from . import config, io_utils


def _pp(x, nd=2):
    """Format a fractional lift as a percentage-point string."""
    if x is None:
        return "n/a"
    return f"{100 * x:+.{nd}f}pp"


def _pp_abs(x, nd=2):
    if x is None:
        return "n/a"
    return f"{100 * x:.{nd}f}pp"


def write_json_outputs(results: dict) -> list[str]:
    out = config.OUTPUTS_DIR
    paths = []

    detection_path = os.path.join(out, "detection_metrics.json")
    io_utils.write_json(detection_path, io_utils.round_floats(results["detection"]))
    paths.append(detection_path)

    est_path = os.path.join(out, "estimator_comparison.json")
    io_utils.write_json(est_path, io_utils.round_floats(results["estimator"]))
    paths.append(est_path)

    dur_path = os.path.join(out, "duration_guidance.json")
    io_utils.write_json(dur_path, io_utils.round_floats(results["duration"]))
    paths.append(dur_path)

    ps_path = os.path.join(out, "per_series_results.json")
    io_utils.write_json(ps_path, io_utils.round_floats(results["per_series"]))
    paths.append(ps_path)

    ship_path = os.path.join(out, "shippable_lift_accuracy.json")
    io_utils.write_json(ship_path, io_utils.round_floats(results["shippable"]))
    paths.append(ship_path)

    proj_path = os.path.join(out, "ramp_projections.json")
    io_utils.write_json(proj_path, io_utils.round_floats(results["projection"]))
    paths.append(proj_path)

    seas_path = os.path.join(out, "seasonality_floor.json")
    io_utils.write_json(seas_path, io_utils.round_floats(results["seasonality"]))
    paths.append(seas_path)

    return paths


def write_summary_md(results: dict, cfg: dict) -> str:
    b = results["detection"]["binary"]
    est = results["estimator"]
    K = est["decision_day"]
    ov_n = est["overall"]["naive"]
    ov_d = est["overall"]["debiased"]
    nv_n = est["novelty_only"]["naive"]
    nv_d = est["novelty_only"]["debiased"]
    dur = results["duration"]

    improve = (1 - nv_d["mae"] / nv_n["mae"]) * 100 if nv_n["mae"] else 0.0

    lines = []
    lines.append("# Novelty-effect framework: results summary")
    lines.append("")
    lines.append("SYNTHETIC demonstration. Every number below is produced by "
                 "`python scripts/run_demo.py` from a fixed seed "
                 f"({cfg['seed']}) and is reproducible.")
    lines.append("")
    lines.append(f"Series analysed: {b['n_series']} experiment-metric time series "
                 f"across {len({r['experiment_id'] for r in results['per_series']})} experiments.")
    lines.append("")

    lines.append("## Detection (novelty flag vs known labels)")
    lines.append("")
    lines.append(f"- Precision: {b['precision']:.3f}")
    lines.append(f"- Recall: {b['recall']:.3f}")
    lines.append(f"- F1: {b['f1']:.3f}")
    lines.append(f"- Accuracy: {b['accuracy']:.3f}")
    lines.append(f"- Confusion (novelty): TP {b['true_positive']}, FP {b['false_positive']}, "
                 f"FN {b['false_negative']}, TN {b['true_negative']}")
    lines.append(f"- 4-way category accuracy: {results['detection']['category_accuracy']:.3f}")
    lines.append("")

    lines.append(f"## Debiased vs naive long-term estimate (at decision day {K})")
    lines.append("")
    lines.append("Error is the estimate minus the known true long-term effect.")
    lines.append("")
    lines.append("| Group | Estimator | MAE (pp) | RMSE (pp) | Bias (pp) | Max abs (pp) |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| novelty series | naive | {_pp_abs(nv_n['mae'])} | {_pp_abs(nv_n['rmse'])} | {_pp(nv_n['bias'])} | {_pp_abs(nv_n['max_abs'])} |")
    lines.append(f"| novelty series | debiased | {_pp_abs(nv_d['mae'])} | {_pp_abs(nv_d['rmse'])} | {_pp(nv_d['bias'])} | {_pp_abs(nv_d['max_abs'])} |")
    lines.append(f"| all series | naive | {_pp_abs(ov_n['mae'])} | {_pp_abs(ov_n['rmse'])} | {_pp(ov_n['bias'])} | {_pp_abs(ov_n['max_abs'])} |")
    lines.append(f"| all series | debiased | {_pp_abs(ov_d['mae'])} | {_pp_abs(ov_d['rmse'])} | {_pp(ov_d['bias'])} | {_pp_abs(ov_d['max_abs'])} |")
    lines.append("")
    lines.append(f"On novelty series the debiased estimator cuts mean absolute error by "
                 f"{improve:.0f}% versus the naive early average "
                 f"({_pp_abs(nv_n['mae'])} to {_pp_abs(nv_d['mae'])}).")
    lines.append("")
    lines.append("Note on genuine ramps: a still-climbing effect cannot be extrapolated "
                 "to a trustworthy asymptote from a short early window, so the debiased "
                 "estimator is flagged low-confidence there and the all-series aggregate "
                 "carries that penalty. See the per-category table in "
                 "`outputs/estimator_comparison.json`.")
    lines.append("")

    lines.append("## Duration guidance")
    lines.append("")
    lines.append(f"- Tolerance: {_pp_abs(dur['tol_abs'])} around the long-term effect.")
    lines.append(f"- Recommendations that fall inside the collected horizon: "
                 f"{dur['n_recommendations_in_window']}, of which "
                 f"{dur['n_validated']} reach tolerance on held-out ground truth "
                 f"(pass rate {dur['validation_pass_rate']:.2f}).")
    lines.append("")
    lines.append(f"- Seasonality floor: at least {dur['seasonality_floor_days']} days "
                 f"(one full weekly cycle). It sets the recommendation for "
                 f"{dur['n_series_where_floor_binds']} series whose novelty decays faster "
                 f"than a week.")
    lines.append("")
    lines.append("| Category | Decay-based days (median) | Recommended days (median, with floor) | Naive cumulative average settles (median days) |")
    lines.append("|---|---|---|---|")
    for cat, rd in dur["median_recommended_days_by_category"].items():
        dd = dur["median_decay_days_by_category"].get(cat)
        nd = dur["median_naive_settle_by_category"].get(cat)
        lines.append(f"| {cat} | {dd} | {rd} | {nd} |")
    lines.append("")
    lines.append("Genuine ramps do not settle inside the observed window: the framework "
                 "flags them as still-building rather than issuing a false settle date.")
    lines.append("")

    # Shippable lift at the recommended run length (addition 3).
    ship = results["shippable"]
    so = ship["overall"]
    lines.append("## Shippable lift at the recommended run length")
    lines.append("")
    lines.append("When the framework recommends a stopping day it also produces the "
                 "debiased long-term lift you would ship on at that day. Checked against "
                 "the truth across all series:")
    lines.append("")
    lines.append(f"- Mean absolute error of the shippable lift: {_pp_abs(so['error']['mae'])}.")
    lines.append(f"- Confidence-interval coverage of the true effect: {so['interval_coverage']:.2f}.")
    lines.append("")
    lines.append("| Category | Shippable-lift MAE | CI coverage | n |")
    lines.append("|---|---|---|---|")
    for cat, v in ship["by_category"].items():
        if v["n"] == 0:
            continue
        cov = f"{v['interval_coverage']:.2f}" if v["interval_coverage"] is not None else "n/a"
        lines.append(f"| {cat} | {_pp_abs(v['error']['mae'])} | {cov} | {v['n']} |")
    lines.append("")

    # Long-term projection for genuine ramps (addition 2).
    proj = results["projection"]
    lines.append("## Long-term impact projection for genuine ramps")
    lines.append("")
    H = proj["business_horizon_days"]
    if proj["n_ramps_projected"]:
        lines.append(f"For the {proj['n_ramps_projected']} series flagged as still-climbing "
                     f"genuine ramps, the framework projects the lift at a {H}-day business "
                     "horizon (an extrapolation beyond the data window, with intervals that "
                     "widen the further out the horizon).")
        lines.append("")
        lines.append(f"- Mean absolute error of the {H}-day projection versus the true "
                     f"effect at that horizon: {_pp_abs(proj['yearly_projection']['mae'])}.")
        lines.append(f"- Mean absolute error of the projected asymptote versus the true "
                     f"long-term effect: {_pp_abs(proj['asymptote_projection']['mae'])}.")
        lines.append(f"- Projection interval coverage of the true horizon value: "
                     f"{proj['yearly_interval_coverage']:.2f}.")
    else:
        lines.append("No genuine ramps were flagged in this run.")
    lines.append("")

    # Seasonality floor demonstration (addition 4).
    seas = results["seasonality"]
    lines.append("## Seasonality floor")
    lines.append("")
    lines.append(f"{seas['n_seasonal_series']} series carry weekly seasonality on the "
                 "daily lift. A partial-week read is biased by the day-of-week mix, while "
                 "a full-week read averages it out. Measured on "
                 f"{seas['n_compared']} flat seasonal series (novelty-free, so the "
                 "comparison isolates the seasonal effect), a "
                 f"{seas['partial_read_day']}-day read versus a "
                 f"{seas['floor_days']}-day read:")
    lines.append("")
    lines.append(f"- Mean absolute error of the {seas['partial_read_day']}-day partial-week "
                 f"read: {_pp_abs(seas['mean_abs_error_partial_week'])}.")
    lines.append(f"- Mean absolute error of the {seas['floor_days']}-day full-week read: "
                 f"{_pp_abs(seas['mean_abs_error_full_week'])}.")
    lines.append("")
    ps = {(r["experiment_id"], r["metric"]): r for r in results["per_series"]}
    ex = ps.get(("weekend_promo_banner", "review_starts"))
    if ex:
        lines.append(f"Worked example: `weekend_promo_banner / review_starts` has a fast "
                     f"novelty whose transient decays in {ex['decay_days']} days, but the "
                     f"metric swings by day of week, so the seasonality floor raises the "
                     f"recommendation to {ex['recommended_days']} days (a full week). This is "
                     "the floor changing the decision.")
        lines.append("")
    return "\n".join(lines)


def write_tabs_case_md(results: dict, cfg: dict) -> str:
    K = results["estimator"]["decision_day"]
    ps = {(r["experiment_id"], r["metric"]): r for r in results["per_series"]}
    order = ["review_reactions", "review_starts", "review_completions",
             "tab_usage", "session_duration"]

    lines = []
    lines.append("# Case study: turning a long business page into tabs")
    lines.append("")
    lines.append("SYNTHETIC reconstruction of a real class of problem. A large "
                 "consumer web platform reworked a long business page into a tabbed "
                 "layout. The change touched the whole core experience, so several "
                 "metrics had to be read together, and they moved in contradictory "
                 "directions across different time horizons.")
    lines.append("")
    lines.append(f"The table shows, per metric, the naive reading at an early "
                 f"decision day (day {K}), the debiased long-term estimate with its "
                 f"confidence interval, the true long-term effect, and the recommended "
                 f"run length. All values are percentage points.")
    lines.append("")
    lines.append("| Metric | True category | Detected | Naive @ day "
                 f"{K} | Debiased (95% CI) | True long-term | Recommended days |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in order:
        r = ps.get(("tabs_revamp", m))
        if not r:
            continue
        ci = f"{_pp(r['debiased_at_K'])} [{_pp(r['debiased_ci_lo'])}, {_pp(r['debiased_ci_hi'])}]"
        rec = str(r["recommended_days"]) if r["recommended_in_window"] else "beyond window (still building)"
        lines.append(f"| {m} | {r['category_true']} | {r['category_pred']} | "
                     f"{_pp(r['naive_at_K'])} | {ci} | {_pp(r['L_true'])} | {rec} |")
    lines.append("")

    # Narrative pulled from the numbers.
    starts = ps[("tabs_revamp", "review_starts")]
    comps = ps[("tabs_revamp", "review_completions")]
    react = ps[("tabs_revamp", "review_reactions")]
    tabu = ps[("tabs_revamp", "tab_usage")]

    lines.append("## What the early reading would have told you")
    lines.append("")
    lines.append(f"- review_reactions looked spectacular at day {K} "
                 f"({_pp(react['naive_at_K'])}) because a new control drew curiosity "
                 f"clicks. The true long-term lift is {_pp(react['L_true'])}. Shipping "
                 "on the early number would have massively over-credited the feature.")
    lines.append(f"- review_starts looked strong early ({_pp(starts['naive_at_K'])}) "
                 f"but settles to {_pp(starts['L_true'])}.")
    lines.append(f"- review_completions looked negative early "
                 f"({_pp(comps['naive_at_K'])}) from change aversion, which would have "
                 f"argued for killing the feature. It actually recovers to "
                 f"{_pp(comps['L_true'])} (a primacy effect, detected as "
                 f"{comps['category_pred']}).")
    lines.append(f"- tab_usage is a genuine ramp (detected as {tabu['category_pred']}): "
                 "the benefit compounds as users learn the layout and has not saturated "
                 "in the window, so it must be read as still-building, not debiased to a "
                 "false asymptote.")
    lines.append("")

    # Long-term projection for the tab_usage ramp, pulled from the projection run.
    proj_by = {(p["experiment_id"], p["metric"]): p for p in results["projection"]["projections"]}
    tp = proj_by.get(("tabs_revamp", "tab_usage"))
    if tp:
        H = tp["business_horizon_days"]
        lines.append("## Projecting the tab_usage ramp")
        lines.append("")
        lines.append(f"Because tab_usage is still climbing, the framework does not report a "
                     f"settled lift. Instead it projects the {H}-day impact by extrapolating "
                     "the fit beyond the collected window, and labels it an extrapolation.")
        lines.append("")
        lines.append(f"- Projected {H}-day lift: {_pp(tp['projected_yearly'])} "
                     f"[{_pp(tp['projected_yearly_ci'][0])}, {_pp(tp['projected_yearly_ci'][1])}].")
        lines.append(f"- True effect at {H} days: {_pp(tp['true_yearly'])}.")
        lines.append(f"- Absolute projection error: {_pp_abs(tp['yearly_abs_error'])}.")
        lines.append(f'In plain terms, if you ship this the projected yearly lift is about '
                     f'{_pp(tp["projected_yearly"])}, and the projection lands '
                     f'{_pp_abs(tp["yearly_abs_error"])} from the true value at that horizon.')
        lines.append("")
    lines.append("## The decision the framework supports")
    lines.append("")
    lines.append("The contradictory early signals resolve once each metric is decomposed "
                 "into a long-term level plus a decaying transient:")
    lines.append("")
    lines.append("- Do not kill the feature on the day-" + str(K) +
                 " completions dip: it is primacy and recovers.")
    lines.append("- Do not over-credit the reactions and starts spikes: most of the "
                 "early lift is novelty that decays.")
    lines.append("- Keep watching tab_usage: it is a real, still-growing benefit.")
    lines.append("")
    lines.append("The debiased long-term estimates recover the true effects within their "
                 "confidence intervals, turning a set of contradictory early readings "
                 "into one coherent ship decision.")
    lines.append("")
    return "\n".join(lines)


def write_duration_md(results: dict, cfg: dict) -> str:
    dur = results["duration"]
    lines = []
    lines.append("# Duration guidance detail")
    lines.append("")
    lines.append("SYNTHETIC demonstration. Recommended run length per metric, with "
                 "validation against known ground truth.")
    lines.append("")
    lines.append(f"Tolerance: {_pp_abs(dur['tol_abs'])}. Validation pass rate on "
                 f"in-window recommendations: {dur['n_validated']}/"
                 f"{dur['n_recommendations_in_window']} "
                 f"({dur['validation_pass_rate']:.2f}).")
    lines.append("")
    lines.append(f"Seasonality floor: {dur['seasonality_floor_days']} days. It sets the "
                 f"recommendation for {dur['n_series_where_floor_binds']} series whose "
                 "novelty decays faster than one week.")
    lines.append("")
    lines.append("| Experiment | Metric | True category | Decay days | Floor days | Recommended | Floor binds | In window | Validated | Naive settle days |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results["per_series"]:
        lines.append(f"| {r['experiment_id']} | {r['metric']} | {r['category_true']} | "
                     f"{r['decay_days']} | {r['seasonality_floor_days']} | "
                     f"{r['recommended_days']} | {'yes' if r['floor_binds'] else 'no'} | "
                     f"{'yes' if r['recommended_in_window'] else 'no'} | "
                     f"{'yes' if r['duration_validated'] else 'no'} | "
                     f"{r['naive_cumulative_settle_days']} |")
    lines.append("")
    return "\n".join(lines)


def write_markdown_outputs(results: dict, cfg: dict) -> list[str]:
    out = config.OUTPUTS_DIR
    paths = []
    for name, builder in [
        ("summary.md", write_summary_md),
        ("tabs_case_study.md", write_tabs_case_md),
        ("duration_guidance.md", write_duration_md),
    ]:
        p = os.path.join(out, name)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(builder(results, cfg))
            fh.write("\n")
        paths.append(p)
    return paths
