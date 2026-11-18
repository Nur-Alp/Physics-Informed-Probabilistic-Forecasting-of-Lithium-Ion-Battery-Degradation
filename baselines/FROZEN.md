# Frozen baselines (18 Nov 2026)

The baselines below are fixed. From here on they are not changed in response to development (train + val) results. The locked test sets are scored once, at final evaluation:

```bash
python -m baselines.evaluate severson --set frozen --final
python -m baselines.evaluate kirkaldy --set frozen --final
```

If a baseline must change after this point (a bug, say), the change is a new version: it gets its own entry below, and the frozen version is still the one reported first.

## Why freeze

The train/val split was used to develop the baselines: the failures fixed in v2 were diagnosed on val cells, and variants were compared against published numbers on val. Val numbers are therefore optimistic and are reported as development results only. Grouped cross-validation (below) is the development estimate; the test set is the only unbiased one.

## The frozen sets

Chosen by one rule fixed in advance: for each baseline, the version (original or v2) with the lower capacity CRPS, averaged over the three forecast origins, in grouped 5-fold CV over all development cells. CRPS is the same score already used to tune the naive window.

| Baseline | Severson | Kirkaldy | Source |
|---|---|---|---|
| Naive slope | v2, slope capped at 0 (CRPS 31.9 vs 32.5 mAh) | v2 (121.0 vs 123.5) | Hyndman & Koehler 2006; physical bound |
| Empirical fade fit | v2 `FadeFitPrior`, MAP with a train-population prior (98.8 vs 156.5) | original `FadeFit` (226.9 vs 1078.6) | Richardson et al. 2017; He et al. 2011 |
| Wiener with drift | original linear `Wiener` (25.5 vs 28.0) | v2 `WienerPower`, power-law time scale (97.2 vs 112.7) | Xu et al. 2019; Zhang et al. 2018 |
| Plain GP | unchanged `PlainGP` | unchanged `PlainGP` | Richardson et al. 2017 |
| Cross-cell capacity | `CrossCellCapacity` | not used (most lives censored) | Severson et al. 2019 SI Note 3 |

Fitted on the fitting cells only (train for the final run; the other folds in CV): the naive window (by CRPS), the Wiener prior and power-law exponent, the fade-fit population prior, and the cross-cell elastic net with its fade-shape template. All forecasts get the same physical clip, [0, 1.1 × max observed capacity].

## Development estimate: grouped 5-fold CV, frozen models

Folds are grouped by charging protocol (Severson, 61 groups over 83 cells) and by experiment × temperature (Kirkaldy, 10 groups over 27 cells), so no group appears in both the fitting and the held-out cells. Each cell is scored once, out of fold. Median end-of-life absolute % error with a 95% bootstrap interval over cells, then the EOL 95% coverage.

Severson (origins 100 / 200 / 400 cycles):

| Model | Median EOL error (95% CI) | EOL coverage |
|---|---|---|
| Cross-cell capacity | 23 (18–27) / 25 (20–27) / 12 (9–14)% | 0.93 / 0.95 / 0.94 |
| Fade fit v2 | 33 (26–43) / 22 (20–25) / 11 (8–15)% | 0.83 / 0.33 / 0.13 |
| Wiener (original) | 123 (111–138) / 190 (154–202) / 125 (110–170)% | 0.42 / 0.16 / 0.05 |
| Naive slope v2 | 317 / 246 / 40% | 0.00 / 0.00 / 0.00 |
| Plain GP | 437 / 449 / 314% | 0.02 / 0.02 / 0.04 |

Kirkaldy (origins 4 / 6 / 8 reference tests, 14 cells with an observed EOL):

| Model | Median EOL error (95% CI) | EOL coverage |
|---|---|---|
| Naive slope v2 | 36 (33–39) / 5.3 (4.2–12) / 4.7 (1.5–9.0)% | 0.07 / 0.07 / 0.29 |
| Wiener v2 | 28 (12–34) / 17 (8–19) / 9.0 (6.4–16)% | 1.00 / 1.00 / 0.64 |
| Fade fit (original) | 47 (43–50) / 28 (19–106) / 13 (7–97)% | 0.00 / 0.57 / 0.21 |
| Plain GP | 185 / 123 / 16% | 0.86 / 0.93 / 0.50 |

Files: `results/baselines/{severson,kirkaldy}_{original,v2}_cv5_{summary,per_cell}.csv`.

### What the CV changed

The single val split had overstated two v2 gains: Kirkaldy fade fit v2 (val 14 / 14 / 17%, CV 57 / 39 / 16%, worse than the original) and Severson Wiener v2 at origin 100 (val 29%, CV 185%). Neither is in the frozen set.

## Literature comparisons (not part of the frozen forecasting set)

`baselines/literature_check.py` re-runs each paper's own protocol; `baselines/benchmark.py` keeps the scoreboard. The Thelen-protocol bias correction for the fade fit (shift and log-normal error fitted on train, `run_thelen_recal`) was also checked by grouped CV (`cv_thelen_recal`): fold fits a = −0.15 to −0.20, s = 0.52 to 0.60; out of fold, RUL RMSE 53.0 cycles, calibration error 0.166, alpha-accuracy 25.2%, 95% coverage 0.99 (Thelen 2022 primary / secondary: 147.4 / 105.0, 0.06 / 0.20, 32.9 / 15.4%).

## Variants tried before freezing (about 20)

Literature checks: fade fit original, + residual GP, + residual GP scaled ×4 / ×8 / ×16, with population prior, + bias recalibration (spread-scaling version abandoned when its scale hit the grid edge twice, replaced by log-normal dressing); plain GP original, true zero mean, Bayesian with GPax priors, linear mean (Kenney protocol); Liu basic GPR; Zou linear persistence (two window readings). Own protocol: original set, v2 set (naive cap, fade prior, power-law Wiener, cross-cell). All runs are in `results/baselines/literature_check/experiment_log.jsonl` (earlier runs: their summary CSVs) and in the Action plan.
