"""Per-metric duration (run-length) guidance.

The recommended run length has two components and is the maximum of the two.

1. Decay-based duration. For the fitted transient effect(t) = L + A*exp(-t/tau)
   the per-day deviation from the asymptote is |A|*exp(-t/tau), which falls
   within a tolerance tol when

       t >= tau * ln(|A| / tol).

   For a flat metric (no transient) this component is 1 day: nothing needs to
   wear off.

2. Seasonality floor. Businesses commonly require running at least one full
   seasonal cycle (by default one week) so that a partial-week day-of-week mix
   cannot dominate the read. The floor is season_length_days * min_cycles.

The final recommendation is max(decay-based, seasonality floor). A fast novelty
whose transient decays in three days still needs a full week if the metric has
weekly seasonality.

We validate the recommendation against ground truth: the average of the OBSERVED
daily lifts from the recommended day to the end of the window should agree with
the true long-term effect L_true, within tolerance plus a sampling-noise band.

We also report, for contrast, how long the NAIVE cumulative average takes to
settle. Because an early spike leaves the cumulative mean only as fast as
~ A*tau / d, that number is often much larger than the recommended duration:
the concrete cost of reading the raw dashboard number instead of debiasing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Z = 1.96  # normal quantile for the sampling-noise band used in validation


@dataclass
class DurationGuidance:
    recommended_days: int          # final recommendation: max(decay, seasonality floor)
    decay_days: int                # decay-based component (novelty transient wear-off)
    seasonality_floor_days: int    # at least one full seasonal cycle
    floor_binds: bool              # True when the seasonality floor sets the recommendation
    tol: float
    reached_within_window: bool    # recommendation falls inside the collected horizon
    observed_within_tol: bool      # post-recommendation observed mean matches L_true (noise-banded)
    naive_cumulative_settle_days: int  # days for the naive cumulative average to settle (cost of naive)


def _tol_for(L_hat: float, cfg: dict) -> float:
    d = cfg["duration"]
    return max(d["tol_abs"], d["tol_rel"] * abs(L_hat))


def seasonality_floor_days(cfg: dict) -> int:
    """Minimum days from the seasonality floor (whole seasonal cycles)."""
    sc = cfg["duration"].get("seasonality_floor", {})
    if not sc.get("enabled", False):
        return 0
    return int(sc.get("season_length_days", 7)) * int(sc.get("min_cycles", 1))


def decay_duration(L_hat: float, A_hat: float, tau_hat: float, cfg: dict) -> tuple[int, float]:
    """Days until the per-day transient |A|*exp(-t/tau) falls within tolerance."""
    tol = _tol_for(L_hat, cfg)
    max_days = cfg["duration"]["max_days"]
    if abs(A_hat) <= tol or tau_hat <= 0.0:
        return 1, tol
    d = math.ceil(tau_hat * math.log(abs(A_hat) / tol))
    d = int(min(max(d, 1), max_days))
    return d, tol


def naive_cumulative_settle_days(days, y, se, L_true: float, tol: float) -> int:
    """Days for the OBSERVED cumulative average to enter and stay within tol
    (plus a noise band) of the true effect. Returns T+1 if it never settles."""
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)
    w = 1.0 / np.maximum(se ** 2, 1e-12)
    T = len(y)
    cum_w = np.cumsum(w)
    running = np.cumsum(w * y) / cum_w
    se_running = 1.0 / np.sqrt(cum_w)
    within = np.abs(running - L_true) <= (tol + Z * se_running)
    outside = np.where(~within)[0]
    if len(outside) == 0:
        return 1
    settle = int(outside[-1]) + 1
    return settle + 1 if settle < T else T + 1


def evaluate(days, y, se, det, L_true: float, cfg: dict) -> DurationGuidance:
    """Produce guidance from the detector fit and validate it on ground truth."""
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)
    T = len(y)

    if det.category == "flat":
        # No transient to wear off: the decay component is one day.
        decay_days, tol = 1, _tol_for(det.L_hat, cfg)
    else:
        decay_days, tol = decay_duration(det.L_hat, det.A_hat, det.tau_hat, cfg)

    floor = seasonality_floor_days(cfg)
    rec_days = max(decay_days, floor)
    floor_binds = floor > decay_days
    reached = rec_days <= T

    observed_ok = False
    if reached:
        # Average of observed daily lifts from the recommended day to the end:
        # if the transient has truly worn off, this equals L_true.
        idx = rec_days - 1
        yy = y[idx:]
        ss = se[idx:]
        w = 1.0 / np.maximum(ss ** 2, 1e-12)
        wsum = np.sum(w)
        post_mean = float(np.sum(w * yy) / wsum)
        se_mean = 1.0 / np.sqrt(wsum)
        observed_ok = abs(post_mean - L_true) <= (tol + Z * se_mean)

    naive_settle = naive_cumulative_settle_days(days, y, se, L_true, tol)

    return DurationGuidance(
        recommended_days=rec_days, decay_days=decay_days,
        seasonality_floor_days=floor, floor_binds=floor_binds, tol=tol,
        reached_within_window=reached, observed_within_tol=observed_ok,
        naive_cumulative_settle_days=naive_settle,
    )
