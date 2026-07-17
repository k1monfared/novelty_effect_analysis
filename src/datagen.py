"""Synthetic data generation for the novelty-effect demo.

WHAT THIS PRODUCES
------------------
For a catalogue of experiment-metric pairs we simulate daily A/B readings:
a control-arm daily mean and a treatment-arm daily mean, each with sampling
noise driven by realistic daily traffic. From those we derive the observed
daily lift (treatment / control - 1) and its standard error via the delta
method. That daily-lift series is exactly what an experimenter reads off a
dashboard.

THE GROUND-TRUTH MODEL
----------------------
Every experiment-metric has a KNOWN noise-free daily effect

    effect(t) = L + A * exp(-t / tau)          for t = 0, 1, ... , T-1

where
    L    the true long-term effect (the asymptote the effect settles to),
    A    the transient amplitude (0 when there is no novelty),
    tau  the decay time-constant in days.

Four categories are generated:

    flat               A = 0. The effect is constant; the early reading is
                       already unbiased. (label: no novelty)
    novelty_overshoot  A has the same sign as L, so the effect starts inflated
                       and decays down to L. Early readings OVER-estimate.
                       (label: novelty)
    primacy_dip        A has the opposite sign to L, so the effect starts
                       depressed (often the wrong sign) and recovers up to L.
                       Early readings UNDER-estimate. (label: novelty)
    genuine_ramp       A large tau relative to a short window: the effect is a
                       real benefit that accumulates and has NOT saturated
                       inside the observed window. It must not be mistaken for
                       a decaying novelty transient. (label: no novelty)

WHY THIS IS REALISTIC
---------------------
New features attract curiosity clicks that fade (novelty), or trigger change
aversion that users get over (primacy), or unlock value that compounds as
users learn (genuine ramp). Different metrics on the SAME feature can move in
contradictory directions over different horizons. The hand-authored
"tabs_revamp" experiment reproduces that situation explicitly.

GROUND TRUTH is written to data/ground_truth.csv and is used only for scoring.
The analysis code consumes only data/daily_observations.csv.
"""
from __future__ import annotations

import hashlib

import numpy as np


# The hand-authored narrative experiments are pinned to a fixed base seed so
# that their worked-example numbers stay stable when the master seed is changed
# to reshape the random population. Only the random population and its noise move
# with the master seed.
NAMED_SEED = 20260703
NAMED_EXPERIMENT_IDS = {"tabs_revamp", "draft_autosave", "weekend_promo_banner"}


def _series_seed(base_seed: int, exp: str, metric: str) -> int:
    """A stable per-series seed derived from the experiment and metric names.

    Giving each series its own RNG makes its data independent of every other
    series, so adding or removing experiments never shifts the hand-authored
    narrative series. This keeps the documented worked-example numbers stable.
    """
    h = hashlib.md5(f"{exp}|{metric}".encode("utf-8")).hexdigest()
    return (int(base_seed) + int(h[:8], 16)) % (2 ** 32)


# --------------------------------------------------------------------------
# Metric templates: each metric has a kind and a baseline level.
#   kind "rate"  metric is a proportion in (0,1); per-user variance p(1-p).
#   kind "mean"  metric is a positive mean; per-user variance (cv*mu)^2.
# --------------------------------------------------------------------------
METRIC_TEMPLATES = {
    "review_starts":     {"kind": "rate", "base": 0.220},
    "review_completions":{"kind": "rate", "base": 0.140},
    "review_reactions":  {"kind": "rate", "base": 0.090},
    "review_length":     {"kind": "mean", "base": 48.0, "cv": 0.9},
    "tab_usage":         {"kind": "mean", "base": 2.30, "cv": 0.8},
    "session_duration":  {"kind": "mean", "base": 310.0, "cv": 1.0},
    "d7_retention":      {"kind": "rate", "base": 0.360},
    "photo_review_starts":{"kind": "rate", "base": 0.060},
    "draft_abandonment": {"kind": "rate", "base": 0.310},
}


def _effect(t: np.ndarray, L: float, A: float, tau: float) -> np.ndarray:
    """Noise-free daily effect L + A*exp(-t/tau); A=0 gives a flat effect."""
    if A == 0.0 or tau <= 0.0:
        return np.full_like(t, L, dtype=float)
    return L + A * np.exp(-t / tau)


def _simulate_series(rng, template, L, A, tau, T, n_min, n_max,
                     seasonal_amp=0.0, seasonal_phase=0.0):
    """Simulate daily control and treatment readings for one experiment-metric.

    Returns a list of per-day dicts with the observed lift and its SE, plus
    the noise-free true effect for reference.

    seasonal_amp adds a weekly swing S*sin(2*pi*dow/7 + phase) directly to the
    daily lift. It averages to zero over any whole week, so it does not change
    the true long-term effect L, but it biases any partial-week read. This is
    what the seasonality floor in the duration guidance protects against.
    """
    t = np.arange(T, dtype=float)
    eff = _effect(t, L, A, tau)

    kind = template["kind"]
    base = template["base"]

    # Daily traffic per arm varies mildly day to day around a per-series level.
    n_level = rng.integers(n_min, n_max)
    n_ctrl = rng.integers(int(n_level * 0.9), int(n_level * 1.1), size=T)
    n_trt = rng.integers(int(n_level * 0.9), int(n_level * 1.1), size=T)

    # Shared weekly seasonality on the LEVELS cancels in the lift and only adds
    # realistic wiggle. The seasonal swing on the EFFECT (below) does not cancel.
    dow = (t % 7).astype(int)
    season = 1.0 + 0.06 * np.sin(2 * np.pi * dow / 7.0)

    # Weekly seasonal swing added to the effect itself (does not cancel in the
    # lift). Mean zero over a full week.
    seas = seasonal_amp * np.sin(2 * np.pi * dow / 7.0 + seasonal_phase)
    total_eff = eff + seas

    rows = []
    for i in range(T):
        if kind == "rate":
            p_c = np.clip(base * season[i], 1e-4, 0.999)
            p_t = np.clip(p_c * (1.0 + total_eff[i]), 1e-4, 0.999)
            sd_c = np.sqrt(p_c * (1.0 - p_c) / n_ctrl[i])
            sd_t = np.sqrt(p_t * (1.0 - p_t) / n_trt[i])
            c_mean = float(rng.normal(p_c, sd_c))
            t_mean = float(rng.normal(p_t, sd_t))
            c_mean = max(c_mean, 1e-4)
            t_mean = max(t_mean, 1e-4)
            c_se = np.sqrt(max(c_mean * (1.0 - c_mean), 1e-8) / n_ctrl[i])
            t_se = np.sqrt(max(t_mean * (1.0 - t_mean), 1e-8) / n_trt[i])
        else:  # mean
            cv = template["cv"]
            mu_c = base * season[i]
            mu_t = mu_c * (1.0 + total_eff[i])
            sd_c = (cv * mu_c) / np.sqrt(n_ctrl[i])
            sd_t = (cv * mu_t) / np.sqrt(n_trt[i])
            c_mean = float(rng.normal(mu_c, sd_c))
            t_mean = float(rng.normal(mu_t, sd_t))
            c_mean = max(c_mean, 1e-6)
            t_mean = max(t_mean, 1e-6)
            c_se = (cv * c_mean) / np.sqrt(n_ctrl[i])
            t_se = (cv * t_mean) / np.sqrt(n_trt[i])

        lift = t_mean / c_mean - 1.0
        # Delta-method SE of the ratio treatment/control.
        lift_se = np.sqrt((t_se / c_mean) ** 2 + (t_mean * c_se / c_mean ** 2) ** 2)

        rows.append({
            "day": i,
            "control_n": int(n_ctrl[i]),
            "control_mean": c_mean,
            "control_se": c_se,
            "treatment_n": int(n_trt[i]),
            "treatment_mean": t_mean,
            "treatment_se": t_se,
            "obs_lift": lift,
            "obs_lift_se": lift_se,
            "true_effect": float(eff[i]),
        })
    return rows


# --------------------------------------------------------------------------
# Hand-authored experiments (fixed parameters) that carry the narrative.
# --------------------------------------------------------------------------
def _named_specs():
    """Return the hand-authored experiment-metric specifications.

    tabs_revamp reproduces the "turn a long business page into tabs" case:
    metrics move in contradictory directions across horizons.
    draft_autosave is a contrast case with an immediate, stable, real effect
    and no novelty, so it can be shipped on early data with confidence.
    """
    specs = []

    # tabs_revamp: the worked case study.
    specs.append(dict(experiment_id="tabs_revamp", metric="review_starts",
                      category="novelty_overshoot", L=0.025, A=0.075, tau=5.0, T=42, K=14,
                      story="Curiosity from the new tabbed layout inflates review starts early (+10pp) then settles to a modest real lift."))
    specs.append(dict(experiment_id="tabs_revamp", metric="review_completions",
                      category="primacy_dip", L=0.012, A=-0.060, tau=7.0, T=42, K=14,
                      story="Change aversion: users cannot find the review action at first so completions dip negative early, then recover to a small positive."))
    specs.append(dict(experiment_id="tabs_revamp", metric="review_reactions",
                      category="novelty_overshoot", L=0.040, A=0.140, tau=3.0, T=42, K=14,
                      story="New reaction controls draw a large curiosity spike that decays fast to a solid long-term lift."))
    specs.append(dict(experiment_id="tabs_revamp", metric="tab_usage",
                      category="genuine_ramp", L=0.120, A=-0.108, tau=45.0, T=28, K=14,
                      story="Users progressively learn to navigate tabs, so the benefit compounds and has not saturated inside the window."))
    specs.append(dict(experiment_id="tabs_revamp", metric="session_duration",
                      category="flat", L=0.004, A=0.0, tau=0.0, T=42, K=14,
                      story="Session duration is a guardrail, and the change is essentially neutral."))

    # draft_autosave: immediate real effect, no novelty (ship-early contrast).
    specs.append(dict(experiment_id="draft_autosave", metric="review_completions",
                      category="flat", L=0.030, A=0.0, tau=0.0, T=42, K=14,
                      story="Autosave removes a real friction from day one, so the lift is immediate and stable and no waiting is needed."))
    specs.append(dict(experiment_id="draft_autosave", metric="draft_abandonment",
                      category="flat", L=-0.045, A=0.0, tau=0.0, T=42, K=14,
                      story="Abandonment drops immediately and stays down, a clean no-novelty win."))

    # weekend_promo_banner: a fast novelty whose decay-based duration is only a
    # few days, but the metric has strong weekly seasonality, so the seasonality
    # floor raises the recommendation to a full week.
    specs.append(dict(experiment_id="weekend_promo_banner", metric="review_starts",
                      category="novelty_overshoot", L=0.018, A=0.028, tau=1.5, T=42, K=14,
                      seasonal_amp=0.010, seasonal_phase=0.0,
                      story="A promo banner drives a short curiosity spike that decays in a few days, but starts swing by day of week, so a partial-week read is misleading and at least one full week is required."))
    specs.append(dict(experiment_id="weekend_promo_banner", metric="review_reactions",
                      category="flat", L=0.006, A=0.0, tau=0.0, T=42, K=14,
                      seasonal_amp=0.015, seasonal_phase=1.4,
                      story="Reactions are essentially neutral in the long run but swing by day of week, so a partial-week read can look falsely positive or negative."))

    return specs


# --------------------------------------------------------------------------
# Randomly generated population (for detection precision / recall power).
# --------------------------------------------------------------------------
_RANDOM_METRIC_POOL = [
    "review_starts", "review_completions", "review_reactions",
    "review_length", "d7_retention", "photo_review_starts",
    "draft_abandonment", "session_duration", "tab_usage",
]


def _sample_random_specs(rng, cfg):
    """Sample a population of experiment-metric specs from config ranges."""
    dcfg = cfg["data"]
    cats = list(dcfg["category_weights"].keys())
    weights = np.array([dcfg["category_weights"][c] for c in cats], dtype=float)
    weights = weights / weights.sum()
    ranges = dcfg["ranges"]

    specs = []
    n_exp = dcfg["n_random_experiments"]
    m_per = dcfg["metrics_per_random_experiment"]

    for e in range(n_exp):
        exp_id = f"exp_{e+1:02d}"
        metrics = list(rng.choice(_RANDOM_METRIC_POOL, size=m_per, replace=False))
        for metric in metrics:
            cat = str(rng.choice(cats, p=weights))
            r = ranges[cat]
            L = float(rng.uniform(r["L"][0], r["L"][1]))
            if cat == "flat":
                A, tau = 0.0, 0.0
            elif cat == "novelty_overshoot":
                A = float(rng.uniform(r["A"][0], r["A"][1])) * np.sign(L if L != 0 else 1.0)
                tau = float(rng.uniform(r["tau"][0], r["tau"][1]))
            elif cat == "primacy_dip":
                A = -float(rng.uniform(r["A"][0], r["A"][1])) * np.sign(L if L != 0 else 1.0)
                tau = float(rng.uniform(r["tau"][0], r["tau"][1]))
            else:  # genuine_ramp
                A = -L * float(rng.uniform(r["A_frac"][0], r["A_frac"][1]))
                tau = float(rng.uniform(r["tau"][0], r["tau"][1]))

            # A fraction of series carry weekly seasonality on the lift.
            scfg = dcfg.get("seasonality", {})
            seasonal_amp, seasonal_phase = 0.0, 0.0
            if rng.random() < scfg.get("fraction_seasonal", 0.0):
                seasonal_amp = float(rng.uniform(scfg["amp_min"], scfg["amp_max"]))
                seasonal_phase = float(rng.uniform(0.0, 2 * np.pi))

            specs.append(dict(experiment_id=exp_id, metric=metric, category=cat,
                              L=L, A=A, tau=tau, T=int(r["T_days"]), K=int(r["K_early_days"]),
                              seasonal_amp=seasonal_amp, seasonal_phase=seasonal_phase,
                              story=""))
    return specs


def generate(cfg) -> tuple[list[dict], list[dict], list[dict]]:
    """Generate the full synthetic dataset.

    Returns three tables (as lists of dicts):
        ground_truth   one row per experiment-metric with the known labels,
        manifest       one row per experiment-metric with generation params,
        daily          one row per experiment-metric-day with observations.
    """
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    dcfg = cfg["data"]

    specs = _named_specs() + _sample_random_specs(rng, cfg)

    ground_truth, manifest, daily = [], [], []
    for spec in specs:
        template = METRIC_TEMPLATES[spec["metric"]]
        seasonal_amp = float(spec.get("seasonal_amp", 0.0))
        seasonal_phase = float(spec.get("seasonal_phase", 0.0))
        # Each series draws from its own RNG so its data is independent of the
        # rest of the population. Hand-authored narrative series are pinned to a
        # fixed base seed so the master seed only reshapes the random population.
        base = NAMED_SEED if spec["experiment_id"] in NAMED_EXPERIMENT_IDS else seed
        srng = np.random.default_rng(_series_seed(base, spec["experiment_id"], spec["metric"]))
        rows = _simulate_series(
            srng, template, spec["L"], spec["A"], spec["tau"], spec["T"],
            dcfg["daily_users_min"], dcfg["daily_users_max"],
            seasonal_amp=seasonal_amp, seasonal_phase=seasonal_phase,
        )
        is_novelty = spec["category"] in ("novelty_overshoot", "primacy_dip")

        ground_truth.append({
            "experiment_id": spec["experiment_id"],
            "metric": spec["metric"],
            "category": spec["category"],
            "is_novelty": int(is_novelty),
            "is_seasonal": int(seasonal_amp > 0.0),
            "L_true": round(spec["L"], 6),
            "A_true": round(spec["A"], 6),
            "tau_true": round(spec["tau"], 6),
            "seasonal_amp_true": round(seasonal_amp, 6),
            "T_days": spec["T"],
            "K_early_days": spec["K"],
        })
        manifest.append({
            "experiment_id": spec["experiment_id"],
            "metric": spec["metric"],
            "metric_kind": template["kind"],
            "baseline_level": template["base"],
            "category": spec["category"],
            "is_seasonal": int(seasonal_amp > 0.0),
            "L_true": round(spec["L"], 6),
            "A_true": round(spec["A"], 6),
            "tau_true": round(spec["tau"], 6),
            "seasonal_amp_true": round(seasonal_amp, 6),
            "T_days": spec["T"],
            "K_early_days": spec["K"],
            "story": spec["story"],
        })
        for row in rows:
            out = {"experiment_id": spec["experiment_id"], "metric": spec["metric"]}
            out.update(row)
            # Round for stable committed files.
            for k in ("control_mean", "control_se", "treatment_mean", "treatment_se",
                      "obs_lift", "obs_lift_se", "true_effect"):
                out[k] = round(out[k], 8)
            daily.append(out)

    return ground_truth, manifest, daily
