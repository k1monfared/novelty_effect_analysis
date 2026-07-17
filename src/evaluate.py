"""Scoring against ground truth.

Provides:
  - binary detection metrics (precision, recall, F1) for the novelty flag,
  - a 4-way confusion matrix over categories,
  - error aggregation for naive vs debiased estimates (MAE and RMSE),
    overall and per category.

Nothing here is invented: every number is computed from the actual run.
"""
from __future__ import annotations

import numpy as np

CATEGORIES = ["flat", "novelty_overshoot", "primacy_dip", "genuine_ramp"]


def binary_detection_metrics(true_flags: list[int], pred_flags: list[int]) -> dict:
    """Precision, recall, F1, and the 2x2 counts for the novelty flag."""
    t = np.asarray(true_flags, dtype=int)
    p = np.asarray(pred_flags, dtype=int)
    tp = int(np.sum((t == 1) & (p == 1)))
    fp = int(np.sum((t == 0) & (p == 1)))
    fn = int(np.sum((t == 1) & (p == 0)))
    tn = int(np.sum((t == 0) & (p == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(t) if len(t) else 0.0
    return {
        "n_series": int(len(t)),
        "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "true_negative": tn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def confusion_matrix(true_cats: list[str], pred_cats: list[str]) -> dict:
    """4-way confusion matrix as nested dict [true][pred] = count."""
    mat = {a: {b: 0 for b in CATEGORIES} for a in CATEGORIES}
    for tc, pc in zip(true_cats, pred_cats):
        mat[tc][pc] += 1
    return mat


def category_accuracy(true_cats: list[str], pred_cats: list[str]) -> float:
    t = np.asarray(true_cats)
    p = np.asarray(pred_cats)
    return float(np.mean(t == p)) if len(t) else 0.0


def error_summary(errors: list[float]) -> dict:
    """MAE and RMSE for a list of signed errors."""
    e = np.asarray(errors, dtype=float)
    if len(e) == 0:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "max_abs": None}
    return {
        "n": int(len(e)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "bias": float(np.mean(e)),
        "max_abs": float(np.max(np.abs(e))),
    }
