# Case study: turning a long business page into tabs

SYNTHETIC reconstruction of a real class of problem. A large consumer web platform reworked a long business page into a tabbed layout. The change touched the whole core experience, so several metrics had to be read together, and they moved in contradictory directions across different time horizons.

The table shows, per metric, the naive reading at an early decision day (day 14), the debiased long-term estimate with its confidence interval, the true long-term effect, and the recommended run length. All values are percentage points.

| Metric | True category | Detected | Naive @ day 14 | Debiased (95% CI) | True long-term | Recommended days |
|---|---|---|---|---|---|---|
| review_reactions | novelty_overshoot | novelty_overshoot | +7.51pp | +4.87pp [+4.14pp, +5.52pp] | +4.00pp | 9 |
| review_starts | novelty_overshoot | novelty_overshoot | +5.35pp | +1.93pp [-0.27pp, +3.04pp] | +2.50pp | 15 |
| review_completions | primacy_dip | primacy_dip | -1.29pp | +1.14pp [-0.25pp, +8.71pp] | +1.20pp | 18 |
| tab_usage | genuine_ramp | genuine_ramp | +2.63pp | +8.93pp [+5.43pp, +9.70pp] | +12.00pp | beyond window (still building) |
| session_duration | flat | flat | +0.49pp | +0.49pp [+0.32pp, +0.65pp] | +0.40pp | 7 |

## What the early reading would have told you

- review_reactions looked spectacular at day 14 (+7.51pp) because a new control drew curiosity clicks. The true long-term lift is +4.00pp. Shipping on the early number would have massively over-credited the feature.
- review_starts looked strong early (+5.35pp) but settles to +2.50pp.
- review_completions looked negative early (-1.29pp) from change aversion, which would have argued for killing the feature. It actually recovers to +1.20pp (a primacy effect, detected as primacy_dip).
- tab_usage is a genuine ramp (detected as genuine_ramp): the benefit compounds as users learn the layout and has not saturated in the window, so it must be read as still-building, not debiased to a false asymptote.

## Projecting the tab_usage ramp

Because tab_usage is still climbing, the framework does not report a settled lift. Instead it projects the 365-day impact by extrapolating the fit beyond the collected window, and labels it an extrapolation.

- Projected 365-day lift: +10.08pp [+7.90pp, +14.16pp].
- True effect at 365 days: +12.00pp.
- Absolute projection error: 1.92pp.
In plain terms, if you ship this the projected yearly lift is about +10.08pp, and the projection lands 1.92pp from the true value at that horizon.

## The decision the framework supports

The contradictory early signals resolve once each metric is decomposed into a long-term level plus a decaying transient:

- Do not kill the feature on the day-14 completions dip: it is primacy and recovers.
- Do not over-credit the reactions and starts spikes: most of the early lift is novelty that decays.
- Keep watching tab_usage: it is a real, still-growing benefit.

The debiased long-term estimates recover the true effects within their confidence intervals, turning a set of contradictory early readings into one coherent ship decision.

