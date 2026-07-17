"""Figure generation (matplotlib, Agg backend).

Every figure is drawn from the actual analysis run: observed daily lifts,
fitted transients, and the aggregate scores computed in pipeline.analyze.
No emojis, a restrained palette, and colorblind-safe accent colors.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, model
from .pipeline import load_series

# Restrained, colorblind-safe palette.
C_OBS = "#4C6EF5"      # observed points
C_FIT = "#E8590C"      # fitted transient
C_ASYM = "#2B8A3E"     # asymptote / long-term
C_NAIVE = "#B197FC"    # naive early average
C_TRUE = "#495057"     # ground-truth curve
C_GRID = "#DEE2E6"

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 10,
    "axes.edgecolor": "#adb5bd",
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def _fit_curve(days, y, se, cfg):
    fit = model.fit(days, y, se, cfg)
    tt = np.linspace(0, days.max(), 200)
    curve = fit.L + fit.A * np.exp(-tt / fit.tau) if fit.tau > 0 else np.full_like(tt, fit.L)
    return fit, tt, curve


def fig_tabs_case(cfg, results, path):
    """Small multiples of the tabs_revamp metrics: contradictory novelty shapes."""
    _, series = load_series(cfg)
    K = cfg["estimator"]["decision_day"]
    metrics = ["review_reactions", "review_starts", "review_completions",
               "tab_usage", "session_duration"]
    pretty = {r["experiment_id"] + "|" + r["metric"]: r for r in results["per_series"]}

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        key = ("tabs_revamp", metric)
        if key not in series:
            ax.axis("off"); continue
        days, y, se, true_eff = series[key]
        rec = pretty["tabs_revamp|" + metric]
        fit, tt, curve = _fit_curve(days, y, se, cfg)

        ax.axhline(0, color="#ced4da", lw=1)
        ax.errorbar(days, 100 * y, yerr=100 * 1.96 * se, fmt="o", ms=3.2,
                    color=C_OBS, ecolor="#ced4da", elinewidth=0.7, capsize=0,
                    alpha=0.8, label="observed daily lift")
        ax.plot(tt, 100 * curve, color=C_FIT, lw=2.2, label="fitted transient")
        ax.axhline(100 * fit.L, color=C_ASYM, lw=1.6, ls="--",
                   label="debiased long-term")
        naive_K = 100 * rec["naive_at_K"]
        ax.plot([0, K], [naive_K, naive_K], color=C_NAIVE, lw=1.6, ls=":",
                label=f"naive avg @ day {K}")
        ax.axvline(K, color=C_NAIVE, lw=0.8, ls=":", alpha=0.6)

        ax.set_title(f"{metric}\n(true: {rec['category_true']}  |  detected: {rec['category_pred']})",
                     fontsize=9.5)
        ax.set_xlabel("day")
        ax.set_ylabel("lift (%)")

    # Legend in the last (sixth) empty axis.
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", fontsize=10, frameon=False,
                    title="tabs_revamp case study")

    fig.suptitle("Tabs revamp: one feature, contradictory novelty shapes across metrics",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path)
    plt.close(fig)


def fig_category_examples(cfg, results, path):
    """One representative series per category with the fitted transient."""
    _, series = load_series(cfg)
    want = ["flat", "novelty_overshoot", "primacy_dip", "genuine_ramp"]
    chosen = {}
    # Prefer clean, correctly-classified examples with a decent amplitude.
    for cat in want:
        cands = [r for r in results["per_series"]
                 if r["category_true"] == cat and r["category_pred"] == cat]
        if not cands:
            cands = [r for r in results["per_series"] if r["category_true"] == cat]
        cands.sort(key=lambda r: -abs(r["A_true"]))
        chosen[cat] = cands[0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    axes = axes.ravel()
    for ax, cat in zip(axes, want):
        rec = chosen[cat]
        key = (rec["experiment_id"], rec["metric"])
        days, y, se, true_eff = series[key]
        fit, tt, curve = _fit_curve(days, y, se, cfg)
        ax.axhline(0, color="#ced4da", lw=1)
        ax.errorbar(days, 100 * y, yerr=100 * 1.96 * se, fmt="o", ms=3,
                    color=C_OBS, ecolor="#e9ecef", elinewidth=0.7, alpha=0.8,
                    label="observed")
        ax.plot(days, 100 * true_eff, color=C_TRUE, lw=1.4, ls="-.",
                label="ground-truth effect")
        ax.plot(tt, 100 * curve, color=C_FIT, lw=2.2, label="fitted")
        ax.axhline(100 * fit.L, color=C_ASYM, lw=1.6, ls="--", label="asymptote")
        ax.set_title(f"{cat}  ({rec['experiment_id']} / {rec['metric']})", fontsize=10)
        ax.set_xlabel("day"); ax.set_ylabel("lift (%)")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Detector behaviour by category (synthetic examples)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path)
    plt.close(fig)


def fig_confusion(cfg, results, path):
    """Category confusion matrix heatmap."""
    from .evaluate import CATEGORIES
    mat = results["detection"]["confusion_matrix"]
    M = np.array([[mat[a][b] for b in CATEGORIES] for a in CATEGORIES], dtype=int)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(CATEGORIES))); ax.set_yticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticklabels(CATEGORIES, fontsize=8.5)
    ax.set_xlabel("detected category"); ax.set_ylabel("true category")
    thresh = M.max() / 2 if M.max() else 0.5
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            ax.text(j, i, str(M[i, j]), ha="center", va="center",
                    color="white" if M[i, j] > thresh else "#212529", fontsize=11)
    b = results["detection"]["binary"]
    ax.set_title("Category confusion matrix\n"
                 f"novelty flag: precision {b['precision']:.2f}  recall {b['recall']:.2f}  F1 {b['f1']:.2f}",
                 fontsize=11)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_error_vs_day(cfg, results, path):
    """Naive vs debiased estimation error as a function of decision day."""
    ed = results["estimator"]["by_decision_day"]
    edn = results["estimator"]["by_decision_day_novelty_only"]
    ks = sorted(int(k) for k in ed.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
    for ax, table, title in [
        (axes[0], ed, "all series"),
        (axes[1], edn, "novelty series only"),
    ]:
        naive = [100 * table[str(k)]["naive"]["mae"] for k in ks]
        deb = [100 * table[str(k)]["debiased"]["mae"] for k in ks]
        ax.plot(ks, naive, "o-", color=C_NAIVE, lw=2, label="naive early average")
        ax.plot(ks, deb, "o-", color=C_ASYM, lw=2, label="debiased asymptote")
        ax.set_xlabel("decision day (days of data used)")
        ax.set_ylabel("MAE vs true long-term effect (pp)")
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False)
    fig.suptitle("Debiased estimator reaches the truth sooner than the naive average",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path)
    plt.close(fig)


def fig_estimator_scatter(cfg, results, path):
    """Estimated vs true long-term effect, naive and debiased, at decision day K."""
    ps = results["per_series"]
    L = np.array([r["L_true"] for r in ps]) * 100
    naive = np.array([r["naive_at_K"] for r in ps]) * 100
    deb = np.array([r["debiased_at_K"] for r in ps]) * 100
    K = results["estimator"]["decision_day"]

    lim = [min(L.min(), naive.min(), deb.min()) - 1,
           max(L.max(), naive.max(), deb.max()) + 1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, vals, title, col in [
        (axes[0], naive, "naive early average", C_NAIVE),
        (axes[1], deb, "debiased asymptote", C_ASYM),
    ]:
        ax.plot(lim, lim, color="#adb5bd", lw=1, ls="--", label="perfect")
        ax.scatter(L, vals, color=col, s=28, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("true long-term effect (%)")
        ax.set_ylabel("estimate (%)")
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Estimated vs true long-term effect at decision day {K}",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path)
    plt.close(fig)


def fig_duration(cfg, results, path):
    """Recommended run length vs naive cumulative settle time, by category."""
    from .evaluate import CATEGORIES
    ds = results["duration"]
    cats = [c for c in CATEGORIES if c in ds["median_recommended_days_by_category"]]
    rec = [ds["median_recommended_days_by_category"][c] for c in cats]
    naive = [ds["median_naive_settle_by_category"][c] for c in cats]

    x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - w / 2, rec, w, color=C_ASYM, label="recommended (transient worn off)")
    ax.bar(x + w / 2, naive, w, color=C_NAIVE, label="naive cumulative average settles")
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("median days")
    ax.set_title("Duration guidance by category (median across series)", fontsize=11.5)
    for xi, v in zip(x - w / 2, rec):
        ax.text(xi, v + 0.5, str(v), ha="center", fontsize=8.5)
    for xi, v in zip(x + w / 2, naive):
        ax.text(xi, v + 0.5, str(v), ha="center", fontsize=8.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_ramp_projection(cfg, results, path):
    """Projected long-term lift versus horizon for a genuine ramp, with the
    interval widening the further out we extrapolate, against ground truth."""
    projs = results["projection"]["projections"]
    ps = {(r["experiment_id"], r["metric"]): r for r in results["per_series"]}
    if not projs:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "no genuine ramps flagged", ha="center")
        ax.axis("off"); fig.savefig(path); plt.close(fig); return

    pick = next((p for p in projs if (p["experiment_id"], p["metric"]) == ("tabs_revamp", "tab_usage")), projs[0])
    rec = ps[(pick["experiment_id"], pick["metric"])]
    A_true, tau_true, L_true = rec["A_true"], rec["tau_true"], rec["L_true"]
    horizons = sorted(int(h) for h in pick["projected_at_horizon"].keys())

    proj = [100 * pick["projected_at_horizon"][str(h)] for h in horizons]
    lo = [100 * pick["projected_ci"][str(h)][0] for h in horizons]
    hi = [100 * pick["projected_ci"][str(h)][1] for h in horizons]
    true_curve = [100 * (L_true + A_true * np.exp(-h / tau_true)) for h in horizons]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.fill_between(horizons, lo, hi, color=C_ASYM, alpha=0.15, label="95% projection interval")
    ax.plot(horizons, proj, "o-", color=C_ASYM, lw=2, label="projected lift")
    ax.plot(horizons, true_curve, "s--", color=C_TRUE, lw=1.6, label="true effect at horizon")
    ax.set_xlabel("horizon (days out)")
    ax.set_ylabel("projected lift (%)")
    ax.set_title(f"Long-term projection for a genuine ramp\n({pick['experiment_id']} / {pick['metric']}, extrapolation beyond the data)",
                 fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_shippable_accuracy(cfg, results, path):
    """Debiased lift read at the recommended stopping day versus the truth."""
    ps = results["per_series"]
    cats = ["flat", "novelty_overshoot", "primacy_dip", "genuine_ramp"]
    colors = {"flat": "#868e96", "novelty_overshoot": C_FIT,
              "primacy_dip": "#1098AD", "genuine_ramp": C_NAIVE}
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    allv = []
    for cat in cats:
        rows = [r for r in ps if r["category_true"] == cat]
        if not rows:
            continue
        x = np.array([100 * r["L_true"] for r in rows])
        y = np.array([100 * r["shippable_debiased"] for r in rows])
        allv.extend(list(x) + list(y))
        ax.scatter(x, y, s=26, color=colors[cat], alpha=0.8, edgecolor="white",
                   linewidth=0.4, label=cat)
    lim = [min(allv) - 1, max(allv) + 1]
    ax.plot(lim, lim, color="#adb5bd", lw=1, ls="--", label="perfect")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("true long-term effect (%)")
    ax.set_ylabel("shippable debiased lift at recommended day (%)")
    so = results["shippable"]["overall"]
    ax.set_title("What you would ship at the recommended stopping day\n"
                 f"MAE {100*so['error']['mae']:.2f}pp, CI coverage {so['interval_coverage']:.2f}",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_seasonality_floor(cfg, results, path):
    """A seasonal series whose cumulative read swings by day of week until a
    full week is reached, showing why the floor raises the recommendation."""
    _, series = load_series(cfg)
    ps = {(r["experiment_id"], r["metric"]): r for r in results["per_series"]}
    key = ("weekend_promo_banner", "review_starts")
    if key not in series:
        seasonal = [r for r in results["per_series"] if r["is_seasonal_true"] and r["floor_binds"]]
        if not seasonal:
            fig, ax = plt.subplots(figsize=(8, 5)); ax.axis("off")
            ax.text(0.5, 0.5, "no seasonal floor-binding series", ha="center")
            fig.savefig(path); plt.close(fig); return
        key = (seasonal[0]["experiment_id"], seasonal[0]["metric"])
    rec = ps[key]
    days, y, se, _ = series[key]
    w = 1.0 / np.maximum(se ** 2, 1e-12)
    running = np.cumsum(w * y) / np.cumsum(w)
    L_true = rec["L_true"]
    floor = rec["seasonality_floor_days"]
    decay = rec["decay_days"]

    n = min(28, len(days))
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.axhline(100 * L_true, color=C_ASYM, lw=1.6, ls="--", label="true long-term")
    ax.plot(days[:n], 100 * running[:n], "o-", color=C_OBS, lw=1.8, ms=3.5,
            label="cumulative naive read")
    ax.axvline(decay, color=C_FIT, lw=1.4, ls=":", label=f"decay-based day ({decay})")
    ax.axvline(floor, color=C_NAIVE, lw=1.6, ls="-", label=f"seasonality floor ({floor})")
    ax.set_xlabel("day")
    ax.set_ylabel("cumulative lift read (%)")
    ax.set_title("Seasonality floor: a partial-week read swings, a full week settles\n"
                 f"({key[0]} / {key[1]})", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_stakeholder_reactions(cfg, results, path):
    """Stakeholder view of the tabs review_reactions metric: the observed daily
    lift over time, the naive early reading, and the debiased long-term level
    with its 95 percent interval.

    This is what a product manager actually sees during a live experiment, so
    the injected true long-term effect is deliberately not drawn: in real life
    we do not have it. The point is that the early spike overstates and the
    debiased estimate lands well below it.
    """
    _, series = load_series(cfg)
    K = cfg["estimator"]["decision_day"]
    key = ("tabs_revamp", "review_reactions")
    days, y, se, _true = series[key]
    rec = {r["experiment_id"] + "|" + r["metric"]: r
           for r in results["per_series"]}["tabs_revamp|review_reactions"]

    naive = 100 * rec["naive_at_K"]
    deb = 100 * rec["debiased_at_K"]
    lo = 100 * rec["debiased_ci_lo"]
    hi = 100 * rec["debiased_ci_hi"]

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.axhline(0, color="#ced4da", lw=1)

    # Debiased long-term level and its interval (no true line: unknown in life).
    ax.axhspan(lo, hi, color=C_ASYM, alpha=0.14,
               label=f"debiased 95% interval ({lo:.2f} to {hi:.2f}pp)")
    ax.axhline(deb, color=C_ASYM, lw=1.8, ls="--",
               label=f"debiased long-term ({deb:.2f}pp)")

    # Observed daily lift over time.
    ax.errorbar(days, 100 * y, yerr=100 * 1.96 * se, fmt="o", ms=4,
                color=C_OBS, ecolor="#ced4da", elinewidth=0.8, capsize=0,
                alpha=0.85, label="observed daily lift")

    # Naive early reading marked at the decision day.
    ax.plot([0, K], [naive, naive], color=C_NAIVE, lw=2, ls=":",
            label=f"naive reading at day {K} ({naive:.2f}pp)")
    ax.axvline(K, color=C_NAIVE, lw=0.9, ls=":", alpha=0.6)

    # Point out the early spike that overstates the durable effect.
    i_peak = int(np.argmax(100 * y))
    ax.annotate("early spike overstates",
                xy=(days[i_peak], 100 * y[i_peak]),
                xytext=(days[i_peak] + 6, 100 * y[i_peak] + 1.2),
                fontsize=9, color="#495057",
                arrowprops=dict(arrowstyle="->", color="#adb5bd", lw=1))

    ax.set_xlabel("day")
    ax.set_ylabel("lift (%)")
    ax.set_title("What the product manager sees: tabs review_reactions\n"
                 "early spike versus the debiased long-term estimate "
                 "(true effect unknown in a live test)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_stakeholder_completions(cfg, results, path):
    """Stakeholder view of tabs review_completions: an early negative dip that the
    debiased estimate recovers to a positive long-term level. Mirrors the
    review_reactions view. The injected true effect is not drawn, since a live
    test does not reveal it. The point is that the early negative understates the
    durable effect, the mirror image of the review_reactions overshoot.
    """
    _, series = load_series(cfg)
    K = cfg["estimator"]["decision_day"]
    key = ("tabs_revamp", "review_completions")
    days, y, se, _true = series[key]
    rec = {r["experiment_id"] + "|" + r["metric"]: r
           for r in results["per_series"]}["tabs_revamp|review_completions"]

    naive = 100 * rec["naive_at_K"]
    deb = 100 * rec["debiased_at_K"]
    lo = 100 * rec["debiased_ci_lo"]
    hi = 100 * rec["debiased_ci_hi"]

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.axhline(0, color="#ced4da", lw=1)

    ax.axhspan(lo, hi, color=C_ASYM, alpha=0.14,
               label=f"debiased 95% interval ({lo:.2f} to {hi:.2f}pp)")
    ax.axhline(deb, color=C_ASYM, lw=1.8, ls="--",
               label=f"debiased long-term ({deb:.2f}pp)")

    ax.errorbar(days, 100 * y, yerr=100 * 1.96 * se, fmt="o", ms=4,
                color=C_OBS, ecolor="#ced4da", elinewidth=0.8, capsize=0,
                alpha=0.85, label="observed daily lift")

    ax.plot([0, K], [naive, naive], color=C_NAIVE, lw=2, ls=":",
            label=f"naive reading at day {K} ({naive:.2f}pp)")
    ax.axvline(K, color=C_NAIVE, lw=0.9, ls=":", alpha=0.6)

    # Point out the early dip that understates the durable effect.
    i_trough = int(np.argmin((100 * y)[: max(1, K)]))
    ax.annotate("early dip understates",
                xy=(days[i_trough], 100 * y[i_trough]),
                xytext=(days[i_trough] + 5, 100 * y[i_trough] - 1.4),
                fontsize=9, color="#495057",
                arrowprops=dict(arrowstyle="->", color="#adb5bd", lw=1))

    ax.set_xlabel("day")
    ax.set_ylabel("lift (%)")
    ax.set_title("What the product manager sees: tabs review_completions\n"
                 "early negative dip versus the debiased long-term estimate "
                 "(true effect unknown in a live test)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_stakeholder_tab_usage(cfg, results, path):
    """Stakeholder view of tabs tab_usage: a genuine ramp still climbing at the
    decision day, shown with the observed daily lift, the fitted transient, the
    naive early reading, and the projected one-year lift with its interval. The
    injected true effect is not drawn, since a live test does not reveal it. The
    point is that this series has not settled, so it is projected, not debiased to
    a false settled number.
    """
    _, series = load_series(cfg)
    K = cfg["estimator"]["decision_day"]
    key = ("tabs_revamp", "tab_usage")
    days, y, se, _true = series[key]
    rec = {r["experiment_id"] + "|" + r["metric"]: r
           for r in results["per_series"]}["tabs_revamp|tab_usage"]
    naive = 100 * rec["naive_at_K"]
    _fit, tt, curve = _fit_curve(days, y, se, cfg)

    # Projected one-year value and interval for this ramp.
    projs = results["projection"]["projections"]
    pick = next((p for p in projs if (p["experiment_id"], p["metric"]) == key), None)

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.axhline(0, color="#ced4da", lw=1)

    if pick is not None:
        horizons = sorted(int(h) for h in pick["projected_at_horizon"].keys())
        H = 365 if 365 in horizons else horizons[-1]
        proj = 100 * pick["projected_at_horizon"][str(H)]
        plo, phi = (100 * v for v in pick["projected_ci"][str(H)])
        ax.axhspan(plo, phi, color=C_ASYM, alpha=0.14,
                   label=f"projected {H}-day 95% interval ({plo:.2f} to {phi:.2f}pp)")
        ax.axhline(proj, color=C_ASYM, lw=1.8, ls="--",
                   label=f"projected {H}-day lift ({proj:.2f}pp)")

    ax.errorbar(days, 100 * y, yerr=100 * 1.96 * se, fmt="o", ms=4,
                color=C_OBS, ecolor="#ced4da", elinewidth=0.8, capsize=0,
                alpha=0.85, label="observed daily lift")
    ax.plot(tt, 100 * curve, color=C_FIT, lw=2.0, label="fitted transient")

    ax.plot([0, K], [naive, naive], color=C_NAIVE, lw=2, ls=":",
            label=f"naive reading at day {K} ({naive:.2f}pp)")
    ax.axvline(K, color=C_NAIVE, lw=0.9, ls=":", alpha=0.6)

    # Point out that the series is still climbing at the decision day.
    i_last = len(days) - 1
    ax.annotate("still climbing, not settled",
                xy=(days[i_last], 100 * y[i_last]),
                xytext=(max(0, days[i_last] - 9), 100 * y[i_last] + 1.6),
                fontsize=9, color="#495057",
                arrowprops=dict(arrowstyle="->", color="#adb5bd", lw=1))

    ax.set_xlabel("day")
    ax.set_ylabel("lift (%)")
    ax.set_title("What the product manager sees: tabs tab_usage\n"
                 "still climbing at the decision day, projected to a one-year lift "
                 "(true effect unknown in a live test)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def generate_all(cfg, results) -> list[str]:
    config.ensure_dirs()
    d = config.IMAGES_DIR
    jobs = [
        ("fig_stakeholder_reactions.png", fig_stakeholder_reactions),
        ("fig_stakeholder_completions.png", fig_stakeholder_completions),
        ("fig_stakeholder_tab_usage.png", fig_stakeholder_tab_usage),
        ("fig_tabs_case.png", fig_tabs_case),
        ("fig_category_examples.png", fig_category_examples),
        ("fig_detection_confusion.png", fig_confusion),
        ("fig_error_vs_day.png", fig_error_vs_day),
        ("fig_estimator_scatter.png", fig_estimator_scatter),
        ("fig_duration_guidance.png", fig_duration),
        ("fig_ramp_projection.png", fig_ramp_projection),
        ("fig_shippable_accuracy.png", fig_shippable_accuracy),
        ("fig_seasonality_floor.png", fig_seasonality_floor),
    ]
    paths = []
    for name, fn in jobs:
        p = os.path.join(d, name)
        fn(cfg, results, p)
        paths.append(p)
    return paths
