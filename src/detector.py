"""Novelty detector.

Given a daily-lift series it fits the transient model and classifies the series
into one of four categories, mirroring the ground-truth labels:

    flat               no statistically significant, practically large dynamics.
    novelty_overshoot  a significant decaying transient that has saturated
                       inside the window and whose early reading is INFLATED
                       relative to the asymptote.
    primacy_dip        a significant decaying transient that has saturated but
                       whose early reading is DEPRESSED relative to the
                       asymptote.
    genuine_ramp       significant dynamics that have NOT saturated in the
                       window (large tau, or a linear trend fits better): a real
                       effect that is still building, not a novelty artefact.

The binary novelty flag is True for novelty_overshoot and primacy_dip.

The saturation test is the crux of telling a decaying transient apart from a
genuine ramp: for a transient that has settled, exp(-T/tau) is small (little of
the transient remains at the end of the window); for a still-climbing ramp it is
large.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import model


@dataclass
class Detection:
    category: str
    is_novelty: bool
    L_hat: float
    A_hat: float
    tau_hat: float
    se_L: float
    f_pvalue: float
    saturation: float          # exp(-T/tau_hat): fraction of transient left at window end
    tau_ratio: float           # tau_hat / T
    early_effect: float        # L_hat + A_hat (fitted effect on day 0)
    reason: str


def classify(days, y, se, cfg: dict, max_tau: float | None = None) -> Detection:
    days = np.asarray(days, dtype=float)
    fit = model.fit(days, y, se, cfg, max_tau=max_tau)
    dcfg = cfg["detector"]

    T = float(np.max(days))
    tau = fit.tau
    saturation = float(np.exp(-T / tau)) if tau > 0 else 1.0
    tau_ratio = tau / T if T > 0 else np.inf
    early_effect = fit.L + fit.A

    significant = fit.f_pvalue < dcfg["alpha"]
    large_enough = abs(fit.A) >= dcfg["amp_min_abs"]

    if not (significant and large_enough):
        return Detection(
            category="flat", is_novelty=False, L_hat=fit.L, A_hat=fit.A,
            tau_hat=tau, se_L=fit.se_L, f_pvalue=fit.f_pvalue,
            saturation=saturation, tau_ratio=tau_ratio, early_effect=early_effect,
            reason=("transient not significant (p={:.3f})".format(fit.f_pvalue)
                    if not significant else
                    "transient too small (|A|={:.4f} < {:.4f})".format(abs(fit.A), dcfg["amp_min_abs"])),
        )

    linear_better = fit.aicc2 < (fit.aicc1 - dcfg["linear_margin_aicc"])
    not_saturated = (tau_ratio > dcfg["tau_max_ratio"]) or (saturation > dcfg["saturation_max"])

    if not_saturated or linear_better:
        return Detection(
            category="genuine_ramp", is_novelty=False, L_hat=fit.L, A_hat=fit.A,
            tau_hat=tau, se_L=fit.se_L, f_pvalue=fit.f_pvalue,
            saturation=saturation, tau_ratio=tau_ratio, early_effect=early_effect,
            reason=("not saturated in window (exp(-T/tau)={:.2f}, tau/T={:.2f})".format(saturation, tau_ratio)
                    if not_saturated else "linear trend fits better (AICc)"),
        )

    # Decaying transient that has saturated: novelty vs primacy by whether the
    # transient pushes the early reading in the SAME direction as the asymptote
    # (overshoot) or AGAINST it (a dip / change aversion that recovers).
    same_direction = (fit.A * fit.L) >= 0.0
    if same_direction:
        category = "novelty_overshoot"
        reason = "transient reinforces the asymptote (A={:.4f}, L={:.4f}); early reading inflated".format(fit.A, fit.L)
    else:
        category = "primacy_dip"
        reason = "transient opposes the asymptote (A={:.4f}, L={:.4f}); early reading depressed".format(fit.A, fit.L)

    return Detection(
        category=category, is_novelty=True, L_hat=fit.L, A_hat=fit.A,
        tau_hat=tau, se_L=fit.se_L, f_pvalue=fit.f_pvalue,
        saturation=saturation, tau_ratio=tau_ratio, early_effect=early_effect,
        reason=reason,
    )
