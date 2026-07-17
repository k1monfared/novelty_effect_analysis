"""Naive vs debiased long-term effect estimation.

At a decision day K an experimenter has the daily lifts for days 0..K-1 and must
estimate the long-term effect L.

    naive     the inverse-variance-weighted average of the daily lifts over the
              window. This is what a dashboard "overall lift" reports and it is
              biased whenever a transient is present.

    debiased  fit the transient model on the same window and report its
              asymptote L_hat. When the detector calls the series flat the two
              estimates coincide (the debiased estimator does not manufacture a
              transient that is not there).

A parametric bootstrap resamples the daily lifts from Normal(lift, se) and
refits to give a confidence interval on the debiased asymptote.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import model
from .detector import classify


@dataclass
class Estimate:
    naive: float
    debiased: float
    debiased_lo: float
    debiased_hi: float
    category: str
    is_novelty: bool
    low_confidence: bool


def naive_estimate(y: np.ndarray, se: np.ndarray) -> float:
    """Inverse-variance-weighted average daily lift."""
    w = 1.0 / np.maximum(se ** 2, 1e-12)
    return float(np.sum(w * y) / np.sum(w))


def estimate_at(days, y, se, cfg: dict, rng) -> Estimate:
    """Compute naive and debiased estimates on a given window with a CI."""
    days = np.asarray(days, dtype=float)
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)

    ecfg = cfg["estimator"]
    # Cap the transient timescale at a small multiple of the observed window:
    # an asymptote cannot be credibly extrapolated from a transient far slower
    # than the data we hold.
    window = float(np.max(days)) + 1.0
    max_tau = ecfg.get("asymptote_tau_cap_mult", 2.0) * window

    naive = naive_estimate(y, se)
    det = classify(days, y, se, cfg, max_tau=max_tau)

    if det.category == "flat":
        debiased = naive
    else:
        debiased = det.L_hat

    # A genuine ramp seen only through a short, unsaturated window cannot be
    # extrapolated to a trustworthy asymptote; flag it rather than pretend.
    low_conf = det.category == "genuine_ramp"

    # Parametric bootstrap for the debiased asymptote.
    B = ecfg["bootstrap_draws"]
    draws = np.empty(B)
    for b in range(B):
        yb = rng.normal(y, se)
        db = classify(days, yb, se, cfg, max_tau=max_tau)
        draws[b] = naive_estimate(yb, se) if db.category == "flat" else db.L_hat
    lo = float(np.percentile(draws, ecfg["ci_lower_pct"]))
    hi = float(np.percentile(draws, ecfg["ci_upper_pct"]))

    return Estimate(
        naive=naive, debiased=debiased, debiased_lo=lo, debiased_hi=hi,
        category=det.category, is_novelty=det.is_novelty, low_confidence=low_conf,
    )
