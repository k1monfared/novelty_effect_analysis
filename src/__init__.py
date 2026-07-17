"""Novelty-effect analysis framework (synthetic demonstration).

Modules:
    config     load the JSON configuration
    io_utils   small CSV read/write helpers (no pandas dependency)
    datagen    synthetic per-experiment daily time-series generator
    model      weighted transient fit and model-selection statistics
    detector   classify a daily-lift series and flag novelty
    duration   per-metric duration (run-length) guidance
    estimator  naive vs debiased long-term effect estimation
    evaluate   scoring against ground truth
    figures    matplotlib figure generation
"""
