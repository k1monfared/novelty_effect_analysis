"""Long-term impact projection, aimed at still-climbing genuine ramps.

A genuine ramp has not saturated inside the collected window, so there is no
settled reading to report. Instead we project forward. From the transient fit at
the end of the collected window we read:

    the fitted asymptote L (the value the ramp is heading toward), and
    the projected lift at a business horizon h days out,
        effect(h) = L + A * exp(-h / tau).

Both come with a bootstrap confidence interval. The interval widens with the
horizon, because extrapolating further beyond the data is less certain. Every
projection is an extrapolation beyond the observed window and is labeled as such.

Because the simulation has ground truth we can score the projection: the true
effect at horizon h is L_true + A_true * exp(-h / tau_true), which for a ramp
with a decay constant of a few weeks is essentially L_true at a one-year horizon.
We report how far the projection lands from that truth.
"""
from __future__ import annotations

import numpy as np

from . import model


def true_effect_at(h, L_true, A_true, tau_true):
    """Ground-truth effect at horizon h days."""
    if A_true == 0.0 or tau_true <= 0.0:
        return L_true
    return L_true + A_true * np.exp(-h / tau_true)


def _fit_cap(days, cfg):
    window = float(np.max(days)) + 1.0
    return cfg["estimator"].get("asymptote_tau_cap_mult", 2.0) * window


def project(days, y, se, cfg, rng) -> dict:
    """Project the lift at the configured horizons with bootstrap intervals."""
    days = np.asarray(days, dtype=float)
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)

    pcfg = cfg["projection"]
    horizons = list(pcfg["horizon_grid_days"])
    if pcfg["business_horizon_days"] not in horizons:
        horizons.append(pcfg["business_horizon_days"])
    horizons = sorted(set(horizons))

    max_tau = _fit_cap(days, cfg)
    fit = model.fit(days, y, se, cfg, max_tau=max_tau)

    def eff_at(h, L, A, tau):
        return L + A * np.exp(-h / tau) if tau > 0 else L

    point = {h: float(eff_at(h, fit.L, fit.A, fit.tau)) for h in horizons}
    asymptote = float(fit.L)

    B = pcfg["bootstrap_draws"]
    draws = {h: np.empty(B) for h in horizons}
    asy_draws = np.empty(B)
    lo_pct = cfg["estimator"]["ci_lower_pct"]
    hi_pct = cfg["estimator"]["ci_upper_pct"]
    for b in range(B):
        yb = rng.normal(y, se)
        fb = model.fit(days, yb, se, cfg, max_tau=max_tau)
        asy_draws[b] = fb.L
        for h in horizons:
            draws[h][b] = eff_at(h, fb.L, fb.A, fb.tau)

    ci = {h: (float(np.percentile(draws[h], lo_pct)),
              float(np.percentile(draws[h], hi_pct))) for h in horizons}
    asy_ci = (float(np.percentile(asy_draws, lo_pct)),
              float(np.percentile(asy_draws, hi_pct)))

    return {
        "horizons": horizons,
        "point": point,
        "ci": ci,
        "asymptote": asymptote,
        "asymptote_ci": asy_ci,
        "business_horizon_days": pcfg["business_horizon_days"],
    }
