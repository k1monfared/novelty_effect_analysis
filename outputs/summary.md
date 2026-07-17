# Novelty-effect framework: results summary

SYNTHETIC demonstration. Every number below is produced by `python scripts/run_demo.py` from a fixed seed (404) and is reproducible.

Series analysed: 159 experiment-metric time series across 33 experiments.

## Detection (novelty flag vs known labels)

- Precision: 0.948
- Recall: 0.973
- F1: 0.961
- Accuracy: 0.962
- Confusion (novelty): TP 73, FP 4, FN 2, TN 80
- 4-way category accuracy: 0.956

## Debiased vs naive long-term estimate (at decision day 14)

Error is the estimate minus the known true long-term effect.

| Group | Estimator | MAE (pp) | RMSE (pp) | Bias (pp) | Max abs (pp) |
|---|---|---|---|---|---|
| novelty series | naive | 2.40pp | 2.61pp | -0.04pp | 4.62pp |
| novelty series | debiased | 0.99pp | 1.95pp | -0.02pp | 11.27pp |
| all series | naive | 2.00pp | 2.91pp | -0.85pp | 9.37pp |
| all series | debiased | 1.12pp | 2.17pp | -0.60pp | 11.27pp |

On novelty series the debiased estimator cuts mean absolute error by 59% versus the naive early average (2.40pp to 0.99pp).

Note on genuine ramps: a still-climbing effect cannot be extrapolated to a trustworthy asymptote from a short early window, so the debiased estimator is flagged low-confidence there and the all-series aggregate carries that penalty. See the per-category table in `outputs/estimator_comparison.json`.

## Duration guidance

- Tolerance: 0.50pp around the long-term effect.
- Recommendations that fall inside the collected horizon: 137, of which 135 reach tolerance on held-out ground truth (pass rate 0.99).

- Seasonality floor: at least 7 days (one full weekly cycle). It sets the recommendation for 67 series whose novelty decays faster than a week.

| Category | Decay-based days (median) | Recommended days (median, with floor) | Naive cumulative average settles (median days) |
|---|---|---|---|
| flat | 1 | 7 | 1 |
| novelty_overshoot | 14 | 14 | 40 |
| primacy_dip | 15 | 15 | 43 |
| genuine_ramp | 90 | 90 | 43 |

Genuine ramps do not settle inside the observed window: the framework flags them as still-building rather than issuing a false settle date.

## Shippable lift at the recommended run length

When the framework recommends a stopping day it also produces the debiased long-term lift you would ship on at that day. Checked against the truth across all series:

- Mean absolute error of the shippable lift: 0.95pp.
- Confidence-interval coverage of the true effect: 0.87.

| Category | Shippable-lift MAE | CI coverage | n |
|---|---|---|---|
| flat | 0.23pp | 0.97 | 62 |
| novelty_overshoot | 1.61pp | 0.83 | 41 |
| primacy_dip | 0.95pp | 0.74 | 34 |
| genuine_ramp | 1.77pp | 0.86 | 22 |

## Long-term impact projection for genuine ramps

For the 23 series flagged as still-climbing genuine ramps, the framework projects the lift at a 365-day business horizon (an extrapolation beyond the data window, with intervals that widen the further out the horizon).

- Mean absolute error of the 365-day projection versus the true effect at that horizon: 1.52pp.
- Mean absolute error of the projected asymptote versus the true long-term effect: 1.55pp.
- Projection interval coverage of the true horizon value: 0.83.

## Seasonality floor

47 series carry weekly seasonality on the daily lift. A partial-week read is biased by the day-of-week mix, while a full-week read averages it out. Measured on 14 flat seasonal series (novelty-free, so the comparison isolates the seasonal effect), a 3-day read versus a 7-day read:

- Mean absolute error of the 3-day partial-week read: 0.55pp.
- Mean absolute error of the 7-day full-week read: 0.09pp.

Worked example: `weekend_promo_banner / review_starts` has a fast novelty whose transient decays in 4 days, but the metric swings by day of week, so the seasonality floor raises the recommendation to 7 days (a full week). This is the floor changing the decision.

