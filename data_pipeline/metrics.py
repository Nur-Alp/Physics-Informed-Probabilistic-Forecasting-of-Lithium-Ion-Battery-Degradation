"""Evaluation metrics, shared by every model.

All probabilistic metrics work on Monte Carlo samples (array shape
[n_samples, n_points]), so any model that can draw sample trajectories
plugs in, whatever its internal form.

* RMSE of the predictive mean against observed capacity.
* EOL error: predicted EOL (median over sampled trajectories of each
  trajectory's first crossing of the EOL threshold) minus true EOL.
* 95% coverage: fraction of observed points inside the central 95%
  predictive interval (pointwise), plus whether the true EOL lies inside
  the 95% interval of the sampled EOL distribution.
* CRPS (Gneiting & Raftery 2007, eq. 21, the "energy" form):
  CRPS(F, y) = E|X - y| - 0.5 E|X - X'|, X, X' ~ F independent.
  Computed exactly for the empirical sample distribution using sorted
  samples. ``crps_gaussian`` is the closed form, used to test it.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def rmse(y: np.ndarray, mean: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(mean)) ** 2)))


def crps_samples(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Pointwise CRPS of the empirical distribution of ``samples``
    ([S, P]) at observations ``y`` ([P])."""
    s = np.sort(np.asarray(samples, float), axis=0)
    n = s.shape[0]
    term1 = np.mean(np.abs(s - np.asarray(y, float)[None, :]), axis=0)
    w = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    term2 = np.sum(w * s, axis=0) / n**2  # = 0.5 * E|X - X'|
    return term1 - term2


def crps_gaussian(y, mu, sigma) -> np.ndarray:
    z = (np.asarray(y) - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def interval(samples: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    a = (1 - level) / 2
    return np.quantile(samples, a, axis=0), np.quantile(samples, 1 - a, axis=0)


def coverage(y: np.ndarray, samples: np.ndarray, level: float = 0.95) -> float:
    lo, hi = interval(samples, level)
    return float(np.mean((y >= lo) & (y <= hi)))


def eol_samples(x: np.ndarray, samples: np.ndarray, threshold: float) -> np.ndarray:
    """First threshold crossing of each sampled trajectory, linearly
    interpolated. Trajectories that never cross within ``x`` get +inf
    (right-censored at the forecast horizon)."""
    below = samples < threshold
    has = below.any(axis=1)
    i = np.argmax(below, axis=1)
    out = np.full(samples.shape[0], np.inf)
    rows = np.flatnonzero(has & (i > 0))
    i1 = i[rows]
    q0, q1 = samples[rows, i1 - 1], samples[rows, i1]
    out[rows] = x[i1 - 1] + (q0 - threshold) * (x[i1] - x[i1 - 1]) / (q0 - q1)
    out[has & (i == 0)] = x[0]
    return out


def eol_metrics(eol_true: float, eols: np.ndarray, horizon: float,
                level: float = 0.95) -> dict:
    """EOL error / coverage / CRPS from sampled EOLs. Censored samples
    (inf) are set to ``horizon`` for the CRPS and the median, which makes
    the EOL error a conservative (under-)estimate whenever more than half
    the samples are censored; ``eol_censored_frac`` reports how often."""
    e = np.where(np.isfinite(eols), eols, horizon)
    pred = float(np.median(e))
    # quantiles with +inf present give NaN under interpolation; use a
    # finite stand-in far beyond the horizon, so an upper bound past the
    # horizon still counts as "beyond any true EOL".
    big = np.where(np.isfinite(eols), eols, 1e12)
    lo, hi = np.quantile(big, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {
        "eol_pred": pred,
        "eol_error": pred - eol_true,
        "eol_abs_pct_error": abs(pred - eol_true) / eol_true * 100,
        "eol_covered": bool(lo <= eol_true <= hi),
        "eol_crps": float(crps_samples(np.array([eol_true]), e[:, None])[0]),
        "eol_censored_frac": float(np.mean(~np.isfinite(eols))),
    }
