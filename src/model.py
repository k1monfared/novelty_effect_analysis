"""Transient model fitting and model-selection statistics.

The effect over time is modelled as

    effect(t) = L + A * exp(-t / tau)

which is LINEAR in (L, A) for a fixed tau. We therefore fit it by profiling
over a grid of tau values and solving a weighted least squares problem for
(L, A) at each grid point. This is far more robust than a black-box nonlinear
fit and gives closed-form standard errors.

Weights are the inverse variances of the daily lifts (1 / se^2), so noisy days
count less. We also fit two nested comparison models:

    M0  constant:  effect(t) = L0            (no dynamics)
    M2  linear:    effect(t) = a + b * t     (an unbounded ongoing trend)

Model M1 (the transient) is compared to M0 with an extra-sum-of-squares F test
and to M2 with AICc. Together these separate "flat", "decaying transient", and
"still-trending ramp".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class FitResult:
    n: int
    # Constant model M0
    L0: float
    sse0: float
    # Transient model M1
    L: float
    A: float
    tau: float
    se_L: float
    se_A: float
    sse1: float
    # Linear model M2
    lin_a: float
    lin_b: float
    se_b: float
    sse2: float
    # Comparison statistics
    f_stat: float
    f_pvalue: float
    aicc0: float
    aicc1: float
    aicc2: float


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted least squares. Returns beta, residual SSE (weighted), and the
    unscaled covariance (X^T W X)^-1."""
    W = w[:, None]
    XtWX = X.T @ (W * X)
    XtWy = X.T @ (w * y)
    XtWX_inv = np.linalg.pinv(XtWX)
    beta = XtWX_inv @ XtWy
    resid = y - X @ beta
    sse = float(np.sum(w * resid ** 2))
    return beta, sse, XtWX_inv


def _aicc(sse_w: float, n: int, k: int) -> float:
    """AICc using the weighted SSE as a chi-square style deviance.

    With inverse-variance weights the weighted SSE is on a comparable scale
    across models, so AIC = SSE_w + 2k ranks models by fit-plus-parsimony.
    The small-sample AICc correction is added on top.
    """
    aic = sse_w + 2 * k
    denom = max(n - k - 1, 1)
    return aic + (2 * k * (k + 1)) / denom


def fit(days: np.ndarray, y: np.ndarray, se: np.ndarray, cfg: dict,
        max_tau: float | None = None) -> FitResult:
    """Fit M0, M1, M2 to a daily-lift series and return all statistics.

    max_tau optionally caps the largest decay constant considered. Detection
    uses the full grid (large tau is how a genuine ramp is recognised), but the
    asymptote estimator caps tau at a small multiple of the data window: a
    transient whose timescale far exceeds the observed window cannot be
    extrapolated to a credible asymptote, and leaving the cap off lets rare
    bootstrap draws degenerate into a near-linear fit with a runaway intercept.
    """
    days = np.asarray(days, dtype=float)
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)
    n = len(y)

    # Inverse-variance weights, guarded against zero.
    var = np.maximum(se ** 2, 1e-12)
    w = 1.0 / var

    # ---- M0 constant ----
    X0 = np.ones((n, 1))
    beta0, sse0, _ = _wls(X0, y, w)
    L0 = float(beta0[0])

    # ---- M1 transient, profiled over tau ----
    dcfg = cfg["detector"]
    tau_max = float(np.max(days)) + dcfg["tau_grid_max_pad"]
    if max_tau is not None:
        tau_max = min(tau_max, max_tau)
    tau_grid = np.geomspace(dcfg["tau_grid_min"], tau_max, dcfg["tau_grid_points"])

    best = None
    for tau in tau_grid:
        X = np.column_stack([np.ones(n), np.exp(-days / tau)])
        beta, sse, cov = _wls(X, y, w)
        if best is None or sse < best[1]:
            best = (beta, sse, cov, tau)
    beta1, sse1, cov1, tau_hat = best
    L, A = float(beta1[0]), float(beta1[1])

    # Dispersion-scaled covariance for honest SEs (quasi-likelihood style).
    dof1 = max(n - 3, 1)  # 3 effective params: L, A, tau
    disp = max(sse1 / dof1, 1.0)
    se_L = float(np.sqrt(cov1[0, 0] * disp))
    se_A = float(np.sqrt(cov1[1, 1] * disp))

    # ---- M2 linear ----
    X2 = np.column_stack([np.ones(n), days])
    beta2, sse2, cov2 = _wls(X2, y, w)
    dof2 = max(n - 2, 1)
    disp2 = max(sse2 / dof2, 1.0)
    se_b = float(np.sqrt(cov2[1, 1] * disp2))

    # ---- F test: M1 vs M0 (extra sum of squares) ----
    # M0 has 1 parameter, M1 has 3 (L, A, tau).
    dfn = 2
    dfd = max(n - 3, 1)
    num = (sse0 - sse1) / dfn
    den = sse1 / dfd
    if den <= 0:
        f_stat = float("inf")
    else:
        f_stat = num / den
    f_stat = max(f_stat, 0.0)
    f_pvalue = float(stats.f.sf(f_stat, dfn, dfd))

    return FitResult(
        n=n,
        L0=L0, sse0=sse0,
        L=L, A=A, tau=float(tau_hat), se_L=se_L, se_A=se_A, sse1=sse1,
        lin_a=float(beta2[0]), lin_b=float(beta2[1]), se_b=se_b, sse2=sse2,
        f_stat=f_stat, f_pvalue=f_pvalue,
        aicc0=_aicc(sse0, n, 1),
        aicc1=_aicc(sse1, n, 3),
        aicc2=_aicc(sse2, n, 2),
    )


def running_average_model(d: np.ndarray, L: float, A: float, tau: float) -> np.ndarray:
    """Analytic cumulative average of the transient over days 0..d.

    mean over [0, d] of (L + A*exp(-t/tau)) approximated in discrete daily form
    but expressed with the continuous integral for smoothness:

        L + A * tau * (1 - exp(-d / tau)) / d
    """
    d = np.asarray(d, dtype=float)
    out = np.full_like(d, L, dtype=float)
    mask = (d > 0) & (tau > 0) & (A != 0.0)
    dd = d[mask]
    out[mask] = L + A * tau * (1.0 - np.exp(-dd / tau)) / dd
    return out
