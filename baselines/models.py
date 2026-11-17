"""The four Phase-2 baselines (Action plan item 11).

Common interface, so every model goes through the same scoring code:

    m.fit_population(train_curves)   # optional, train cells only
    m.fit(x_hist, q_hist)            # one cell's first-N observations
    m.sample(x_new, n, rng) -> [n, len(x_new)] capacity trajectories (Ah)

Samples are of *observed* capacity (they include measurement noise), so
they are scored directly against the measured capacities.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import curve_fit
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


class Forecaster:
    name = "base"

    def fit_population(self, curves: list[tuple[np.ndarray, np.ndarray]]) -> None:
        pass

    def fit(self, x: np.ndarray, q: np.ndarray) -> None:
        raise NotImplementedError

    def sample(self, x_new: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------- 1. naive
class NaiveSlope(Forecaster):
    """"Current slope continues": ordinary least-squares line through the
    last ``window`` observations, extrapolated. Uncertainty = OLS
    parameter covariance + residual noise (Gaussian). ``window`` is tuned
    on training cells only (see evaluate.py)."""
    name = "naive_slope"

    def __init__(self, window: int):
        self.window = window

    def fit(self, x, q):
        x, q = x[-self.window:], q[-self.window:]
        self.x0 = x[-1]
        X = np.column_stack([np.ones_like(x), x - self.x0])
        beta, *_ = np.linalg.lstsq(X, q, rcond=None)
        dof = max(len(q) - 2, 1)
        s2 = float(np.sum((q - X @ beta) ** 2) / dof)
        if len(q) <= 2:  # exact fit: no residual information
            s2 = 1e-6
        self.beta, self.cov, self.s2 = beta, s2 * np.linalg.pinv(X.T @ X), s2

    def sample(self, x_new, n, rng):
        b = rng.multivariate_normal(self.beta, self.cov, size=n)
        mean = b[:, :1] + b[:, 1:] * (x_new - self.x0)[None, :]
        return mean + rng.normal(0, np.sqrt(self.s2), size=mean.shape)


# ------------------------------------------------------------ 2. fade fit
def _exp_model(x, a1, a2, a3):
    return a1 + a2 * np.exp(np.clip(a3 * x, -50, 50))


class FadeFit(Forecaster):
    """Empirical fade fit q = a1 + a2 exp(a3 x): the explicit mean
    function of Richardson, Osborne & Howey (2017), equivalent to Goebel
    et al.'s degradation model. a2 < 0, a3 > 0 gives accelerating
    (knee-like) fade; a3 < 0 gives saturating fade; both allowed.
    Uncertainty: Laplace approximation (parameters ~ N(popt, pcov)) +
    residual noise. Falls back to a straight line through all history if
    the fit fails or there are fewer than MIN_POINTS observations."""
    name = "fade_fit"

    MIN_POINTS = 6  # 3 parameters: below this the Laplace covariance is
    # meaningless and sampled exponentials blow up (seen on Kirkaldy with 4 RPTs)

    def fit(self, x, q):
        self.xs = float(max(x.max(), 1.0))
        u = x / self.xs
        best = None
        if len(x) < self.MIN_POINTS:
            self.fallback = NaiveSlope(len(x))
            self.fallback.fit(x, q)
            return
        for k in (-2.0, 0.5, 1.0, 2.0, 4.0, 8.0):
            dq = q[-1] - q[0]
            a2 = dq / (np.exp(k) - 1) if k != 0 else dq
            p0 = (q[0] - a2, a2, k)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p, cov = curve_fit(_exp_model, u, q, p0=p0, maxfev=5000,
                                       bounds=([0, -10, -20], [10, 10, 20]))
            except (RuntimeError, ValueError):
                continue
            sse = float(np.sum((q - _exp_model(u, *p)) ** 2))
            if np.all(np.isfinite(cov)) and (best is None or sse < best[0]):
                best = (sse, p, cov)
        if best is None:
            self.fallback = NaiveSlope(len(x))
            self.fallback.fit(x, q)
            return
        self.fallback = None
        sse, self.p, self.cov = best
        self.s2 = sse / max(len(q) - 3, 1)

    def sample(self, x_new, n, rng):
        if self.fallback is not None:
            return self.fallback.sample(x_new, n, rng)
        P = rng.multivariate_normal(self.p, self.cov, size=n, check_valid="ignore")
        u = x_new / self.xs
        mean = _exp_model(u[None, :], P[:, :1], P[:, 1:2], P[:, 2:3])
        return mean + rng.normal(0, np.sqrt(self.s2), size=mean.shape)


# ------------------------------------------------------------- 3. Wiener
class Wiener(Forecaster):
    """Linear Wiener process with random drift, Xu et al. (2019) Sec. 3:
    degradation X(t) = Q0 - Q(t) = lambda t + sigma_B B(t),
    lambda ~ N(mu_lambda, sigma_lambda^2) across cells.

    Population (train cells only): two-step MLE, Xu et al. eqs. (6)-(7)
    generalised to unequal time steps: per-cell lambda_n = total
    increment / total time, pooled sigma_B^2 from the residual increments,
    then mu_lambda, sigma_lambda^2 from the lambda_n (variance corrected
    for the sigma_B^2/T_n estimation noise in each lambda_n).

    Per cell: conjugate normal update of lambda from that cell's own
    observed increments, then sample paths with drift from the posterior
    plus Brownian increments.

    Xu et al.'s relaxation-effect removal is NOT applied: it targets
    capacity regeneration after rest periods in the NASA data, which
    neither dataset here has between consecutive points.

    ``block``: Severson measures every cycle and per-cycle capacity noise
    would dominate the Brownian term, so capacities are averaged over
    ``block``-cycle bins first (block=1 for Kirkaldy's RPT points).
    """
    name = "wiener"

    def __init__(self, block: int = 1):
        self.block = block

    def _binned(self, x, q):
        if self.block <= 1:
            return x, q
        k = len(x) // self.block
        if k < 2:
            return x, q
        xb = x[: k * self.block].reshape(k, self.block).mean(1)
        qb = q[: k * self.block].reshape(k, self.block).mean(1)
        return xb, qb

    def fit_population(self, curves):
        lams, resid, m_tot, inv_T = [], 0.0, 0, []
        for x, q in curves:
            x, q = self._binned(x, q)
            X = q[0] - q
            dt, dx = np.diff(x), np.diff(X)
            T = x[-1] - x[0]
            lam = (X[-1] - X[0]) / T
            lams.append(lam)
            resid += np.sum((dx - lam * dt) ** 2 / dt)
            m_tot += len(dx)
            inv_T.append(1 / T)
        lams = np.array(lams)
        self.sB2 = resid / m_tot
        self.mu_l = float(lams.mean())
        self.s_l2 = float(max(lams.var(ddof=1) - self.sB2 * np.mean(inv_T),
                              1e-3 * lams.var(ddof=1)))

    def fit(self, x, q):
        xb, qb = self._binned(x, q)
        self.q0 = qb[0]
        T = xb[-1] - xb[0]
        Xk = self.q0 - qb[-1]
        prec = 1 / self.s_l2 + T / self.sB2
        self.post_var = 1 / prec
        self.post_mu = self.post_var * (self.mu_l / self.s_l2 + Xk / self.sB2)
        self.t_k, self.X_k = xb[-1], Xk
        # measurement noise of a single raw observation, for scoring
        # against raw (unbinned) capacities
        r = q - np.interp(x, xb, qb)
        self.noise = float(np.std(r)) if len(r) > 2 else 0.0

    def sample(self, x_new, n, rng):
        order = np.argsort(x_new)
        xs = x_new[order]
        lam = rng.normal(self.post_mu, np.sqrt(self.post_var), size=(n, 1))
        dt = np.diff(np.concatenate([[self.t_k], xs]))
        dt = np.clip(dt, 0, None)
        dB = rng.normal(0, 1, size=(n, len(xs))) * np.sqrt(self.sB2 * dt)[None, :]
        X = self.X_k + lam * (xs - self.t_k)[None, :] + np.cumsum(dB, axis=1)
        out = np.empty_like(X)
        out[:, order] = self.q0 - X
        return out + rng.normal(0, self.noise, size=out.shape)


# ------------------------------------------------------------- 4. plain GP
class PlainGP(Forecaster):
    """Plain GP regression on capacity vs x: Richardson, Osborne & Howey
    (2017)'s best zero-mean model (their model (c)), kernel
    Matern-5/2 + Matern-3/2 + white noise, hyperparameters by maximising
    the marginal likelihood (scikit-learn). ``normalize_y`` makes the
    prior mean the training-data mean rather than 0 Ah. x is scaled by
    the history span so length-scale bounds are dataset-independent.

    Known failure mode to watch for (Richardson et al. 2017, Action plan
    item 23): a stationary zero-mean GP reverts to its mean away from the
    data, so long-horizon forecasts flatten out and EOL can be missed."""
    name = "plain_gp"

    def __init__(self, max_points: int = 300, restarts: int = 2):
        self.max_points, self.restarts = max_points, restarts

    def fit(self, x, q):
        if len(x) > self.max_points:  # thin evenly, keep last point
            idx = np.unique(np.linspace(0, len(x) - 1, self.max_points).round().astype(int))
            x, q = x[idx], q[idx]
        self.xs = float(max(x.max(), 1.0))
        k = (C(1.0, (1e-3, 1e3)) * Matern(0.5, (1e-2, 1e2), nu=2.5)
             + C(0.1, (1e-4, 1e3)) * Matern(0.1, (1e-3, 1e2), nu=1.5)
             + WhiteKernel(1e-2, (1e-8, 1.0)))
        self.gp = GaussianProcessRegressor(k, normalize_y=True,
                                           n_restarts_optimizer=self.restarts,
                                           random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit((x / self.xs)[:, None], q)

    def sample(self, x_new, n, rng):
        Xn = (x_new / self.xs)[:, None]
        mu, cov = self.gp.predict(Xn, return_cov=True)
        cov = 0.5 * (cov + cov.T) + 1e-10 * np.eye(len(mu))
        L = np.linalg.cholesky(cov)
        return mu[None, :] + rng.normal(size=(n, len(mu))) @ L.T
