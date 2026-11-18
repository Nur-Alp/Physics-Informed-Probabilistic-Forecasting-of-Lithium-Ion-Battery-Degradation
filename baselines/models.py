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

    def __init__(self, max_points: int = 300, restarts: int = 2,
                 normalize_y: bool = True):
        self.max_points, self.restarts = max_points, restarts
        self.normalize_y = normalize_y  # False = a true zero prior mean

    def fit(self, x, q):
        if len(x) > self.max_points:  # thin evenly, keep last point
            idx = np.unique(np.linspace(0, len(x) - 1, self.max_points).round().astype(int))
            x, q = x[idx], q[idx]
        self.xs = float(max(x.max(), 1.0))
        k = (C(1.0, (1e-3, 1e3)) * Matern(0.5, (1e-2, 1e2), nu=2.5)
             + C(0.1, (1e-4, 1e3)) * Matern(0.1, (1e-3, 1e2), nu=1.5)
             + WhiteKernel(1e-2, (1e-8, 1.0)))
        self.gp = GaussianProcessRegressor(k, normalize_y=self.normalize_y,
                                           n_restarts_optimizer=self.restarts,
                                           random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit((x / self.xs)[:, None], q)

    def sample(self, x_new, n, rng, latent=False):
        Xn = (x_new / self.xs)[:, None]
        mu, cov = self.gp.predict(Xn, return_cov=True)
        if latent:  # drop the white-noise term (smooth trajectories, plots only)
            cov = cov - self.gp.kernel_.k2.noise_level * np.eye(len(mu))
            w, V = np.linalg.eigh(0.5 * (cov + cov.T))  # near-singular: eigen sqrt
            L = V * np.sqrt(np.clip(w, 0, None))[None, :]
            return mu[None, :] + rng.normal(size=(n, len(mu))) @ L.T
        cov = 0.5 * (cov + cov.T) + 1e-10 * np.eye(len(mu))
        L = np.linalg.cholesky(cov)
        return mu[None, :] + rng.normal(size=(n, len(mu))) @ L.T


# ------------------------------------------------- improved variants
# Added 18 Nov 2026 to bring the baselines up to published performance
# (Action plan item 11 follow-up). The four originals above are kept
# unchanged so earlier results stay reproducible.

class FadeFitGP(Forecaster):
    """Fade fit with a GP on its residuals, Thelen et al. (2022) Sec.
    2.2.2: "the empirical model is first curve fit to the available data,
    and then the GP is fit to the curve fitting residuals". Forecast =
    exponential trend (Laplace parameter samples) + GP residual samples.
    Far from the data the residual GP reverts to its prior variance,
    which is what widens the forecast compared with FadeFit.

    ``scale`` multiplies the parameter-covariance and GP standard
    deviations; 1.0 is the plain method, other values come only from a
    recalibration fitted on training cells (Kuleshov et al. 2018)."""
    name = "fade_fit_gp"

    def __init__(self, max_points: int = 150, scale: float = 1.0):
        self.max_points, self.scale = max_points, scale

    def fit(self, x, q):
        self.trend = FadeFit()
        self.trend.fit(x, q)
        r = q - self._trend_mean(x)
        if len(x) > self.max_points:
            idx = np.unique(np.linspace(0, len(x) - 1, self.max_points).round().astype(int))
            x, r = x[idx], r[idx]
        self.xs = float(max(x.max(), 1.0))
        k = (C(1e-4, (1e-8, 1e-1)) * Matern(0.2, (1e-2, 1e1), nu=1.5)
             + WhiteKernel(1e-5, (1e-9, 1e-2)))
        self.gp = GaussianProcessRegressor(k, normalize_y=False, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit((x / self.xs)[:, None], r)
        self.noise = float(np.sqrt(self.gp.kernel_.k2.noise_level))

    def _trend_mean(self, x):
        t = self.trend
        if t.fallback is not None:
            b = t.fallback.beta
            return b[0] + b[1] * (x - t.fallback.x0)
        return _exp_model(x / t.xs, *t.p)

    def _trend_samples(self, x_new, n, rng):
        t = self.trend
        if t.fallback is not None:
            f = t.fallback
            b = rng.multivariate_normal(f.beta, self.scale ** 2 * f.cov, size=n)
            return b[:, :1] + b[:, 1:] * (x_new - f.x0)[None, :]
        P = rng.multivariate_normal(t.p, self.scale ** 2 * t.cov, size=n, check_valid="ignore")
        u = x_new / t.xs
        return _exp_model(u[None, :], P[:, :1], P[:, 1:2], P[:, 2:3])

    def sample_latent(self, x_new, n, rng):
        mu, cov = self.gp.predict((x_new / self.xs)[:, None], return_cov=True)
        # predict() includes the white-noise term on the diagonal; remove it
        cov = 0.5 * (cov + cov.T) - self.noise ** 2 * np.eye(len(mu))
        w, V = np.linalg.eigh(cov)  # robust square root (cov is near-singular)
        L = V * np.sqrt(np.clip(w, 0, None))[None, :]
        g = mu[None, :] + self.scale * rng.normal(size=(n, len(mu))) @ L.T
        return self._trend_samples(x_new, n, rng) + g

    def sample(self, x_new, n, rng):
        S = self.sample_latent(x_new, n, rng)
        return S + rng.normal(0, self.noise, size=S.shape)


class PlainGPLinearMean(PlainGP):
    """Same kernel as PlainGP, but with an explicit linear mean fitted by
    least squares and the GP on the residuals: the "LGPFR" idea of Liu
    et al. (2013), whose linear mean cut RMSE from 13-23 to 1.4-6.9 SOH
    points. Not a plain GP any more; kept as a labelled variant."""
    name = "gp_linear_mean"

    def fit(self, x, q):
        self.b = np.polyfit(x, q, 1)
        super().fit(x, q - np.polyval(self.b, x))

    def predict(self, x_new):
        mu, sd = self.gp.predict((x_new / self.xs)[:, None], return_std=True)
        return mu + np.polyval(self.b, x_new), sd

    def sample(self, x_new, n, rng):
        return super().sample(x_new, n, rng) + np.polyval(self.b, x_new)[None, :]


def _matern52(a, b, length, scale):
    r = np.abs(a[:, None] - b[None, :]) / length
    s5 = np.sqrt(5.0) * r
    return scale * (1 + s5 + 5.0 / 3.0 * r ** 2) * np.exp(-s5)


class PlainGPBayes(Forecaster):
    """Zero-mean Matern-5/2 GP with fully Bayesian hyperparameters, as in
    Kenney et al. (2024) via GPax 0.1.8 (defaults read from its source
    18 Nov 2026: k_length, k_scale (a variance) and noise variance each
    LogNormal(0, 1); zero mean; Matern-5/2). GPax samples with NUTS; here
    a random-walk Metropolis sampler on the three log-hyperparameters
    does the same job (3 dimensions, <= a few hundred points).

    Still a plain GP: no mean function, no physics. The difference from
    PlainGP is the prior on hyperparameters (which favours length scales
    near the x unit) and averaging over them instead of one ML point.

    ``x_unit``: x is divided by this before fitting. None = history span
    (as PlainGP); Kenney's own protocol passes x already in [0, 1]."""
    name = "plain_gp_bayes"

    def __init__(self, x_unit=None, n_steps=3000, burn=1000, thin=20,
                 max_points=200, seed=0):
        self.x_unit, self.n_steps, self.burn, self.thin = x_unit, n_steps, burn, thin
        self.max_points, self.seed = max_points, seed

    def _loglik(self, th):
        ell, sc, nz = np.exp(th)
        K = _matern52(self.u, self.u, ell, sc) + (nz + 1e-6) * np.eye(len(self.u))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -np.inf
        a = np.linalg.solve(L, self.y)
        ll = -0.5 * a @ a - np.log(np.diag(L)).sum()
        return ll - 0.5 * np.sum(th ** 2)  # LogNormal(0,1) priors on log scale

    def fit(self, x, q):
        if len(x) > self.max_points:
            idx = np.unique(np.linspace(0, len(x) - 1, self.max_points).round().astype(int))
            x, q = x[idx], q[idx]
        self.xs = float(self.x_unit or max(x.max(), 1.0))
        self.u, self.y = x / self.xs, q
        rng = np.random.default_rng(self.seed)
        th = np.zeros(3)
        lp = self._loglik(th)
        step, chain, acc = 0.3, [], 0
        for i in range(self.n_steps):
            prop = th + step * rng.normal(size=3)
            lp2 = self._loglik(prop)
            if np.log(rng.random()) < lp2 - lp:
                th, lp, acc = prop, lp2, acc + 1
            if i < self.burn and (i + 1) % 100 == 0:  # tune step during burn-in
                step *= 1.5 if acc / 100 > 0.35 else 0.7
                acc = 0
            if i >= self.burn and (i - self.burn) % self.thin == 0:
                chain.append(th.copy())
        self.chain = np.exp(np.array(chain))

    def _posterior(self, x_new):
        un = x_new / self.xs
        mus, covs = [], []
        for ell, sc, nz in self.chain:
            K = _matern52(self.u, self.u, ell, sc) + (nz + 1e-6) * np.eye(len(self.u))
            Ks = _matern52(un, self.u, ell, sc)
            L = np.linalg.cholesky(K)
            A = np.linalg.solve(L, Ks.T)
            mus.append(A.T @ np.linalg.solve(L, self.y))
            covs.append(_matern52(un, un, ell, sc) - A.T @ A + nz * np.eye(len(un)))
        return np.array(mus), np.array(covs)

    def predict(self, x_new):
        """Mixture mean and sd over hyperparameter samples (incl. noise)."""
        mus, covs = self._posterior(x_new)
        var = np.mean([np.diag(c) for c in covs], 0) + mus.var(0)
        return mus.mean(0), np.sqrt(np.clip(var, 1e-12, None))

    def sample(self, x_new, n, rng):
        mus, covs = self._posterior(x_new)
        pick = rng.integers(len(mus), size=n)
        out = np.empty((n, len(x_new)))
        for j in np.unique(pick):
            w, V = np.linalg.eigh(0.5 * (covs[j] + covs[j].T))
            Lj = V * np.sqrt(np.clip(w, 0, None))[None, :]
            k = pick == j
            out[k] = mus[j][None, :] + rng.normal(size=(k.sum(), len(x_new))) @ Lj.T
        return out


# ------------------------------------------------------- v2 baselines
# Added 18 Nov 2026 after the origin-200 example plots showed avoidable
# failures (rising naive lines, per-cell fade fits that always invent a
# knee, a linear Wiener drift that cannot bend). Each fix is tied to a
# source; train cells only ever set the population quantities.

from scipy.optimize import least_squares  # noqa: E402


class NaiveSlopeNonIncreasing(NaiveSlope):
    """NaiveSlope with the slope capped at 0: capacity cannot keep rising.
    3 of 40 Severson val cells at origin 200 got a rising line from noise
    in a 10-cycle window."""
    name = "naive_slope_v2"

    def sample(self, x_new, n, rng, latent=False):
        b = rng.multivariate_normal(self.beta, self.cov, size=n)
        b[:, 1] = np.minimum(b[:, 1], 0.0)
        mean = b[:, :1] + b[:, 1:] * (x_new - self.x0)[None, :]
        if latent:
            return mean
        return mean + rng.normal(0, np.sqrt(self.s2), size=mean.shape)


def _knee_model(x, c, logA, logtau):
    return c - np.exp(logA) * np.exp(np.clip(x / np.exp(logtau), -50, 50))


class FadeFitPrior(Forecaster):
    """Exponential fade q = c - A exp(x / tau) fitted per cell by MAP,
    with a Gaussian prior on (c, log A, log tau) estimated from the train
    cells' whole curves. He et al. (2011) show a cross-cell parameter
    prior is what makes early extrapolation of this model work (RUL error
    0-7 cycles with the prior, 31 without). Uncertainty: Laplace
    (Gauss-Newton) around the MAP, plus residual noise whose size is the
    train cells' typical misfit of this model, not the history's noise."""
    name = "fade_fit_v2"

    def _fit_one(self, x, q):
        best = None
        for tau in np.quantile(x, [0.5, 1.0]) * np.array([1.0, 0.5]):
            for k in (1e-3, 1e-2):
                th0 = [q[:5].mean(), np.log(k * max(q[:5].mean() - q.min(), 1e-3)),
                       np.log(max(tau, 1.0))]
                try:
                    r = least_squares(lambda th: _knee_model(x, *th) - q, th0)
                except ValueError:
                    continue
                if best is None or r.cost < best.cost:
                    best = r
        return best

    def fit_population(self, curves):
        thetas, resid = [], []
        for x, q in curves:
            r = self._fit_one(x, q)
            if r is None or not np.all(np.isfinite(r.x)):
                continue
            thetas.append(r.x)
            resid.append(np.sqrt(2 * r.cost / len(x)))
        T = np.array(thetas)
        self.mu = T.mean(0)
        self.Sigma = np.cov(T.T) + 1e-6 * np.eye(3)
        self.Linv = np.linalg.inv(np.linalg.cholesky(self.Sigma))
        self.sigma = float(np.median(resid))
        self.n_pop = len(T)

    def fit(self, x, q):
        def res(th):
            return np.concatenate([(_knee_model(x, *th) - q) / self.sigma,
                                   self.Linv @ (th - self.mu)])
        th0 = self.mu.copy()
        th0[0] = q[: min(5, len(q))].mean()
        r = least_squares(res, th0)
        J = r.jac
        self.th = r.x
        self.cov = np.linalg.pinv(J.T @ J)

    def sample(self, x_new, n, rng, latent=False):
        T = rng.multivariate_normal(self.th, self.cov, size=n, check_valid="ignore")
        m = _knee_model(x_new[None, :], T[:, :1], T[:, 1:2], T[:, 2:3])
        if latent:
            return m
        return m + rng.normal(0, self.sigma, size=m.shape)


class WienerPower(Wiener):
    """Wiener process with random drift on a power-law time scale,
    X(t) = lambda L(t) + sigma_B B(L(t)), L(t) = (t / s)^b: the time-scale
    transformation of Zhang et al. (2018) Eq. 7, which lets the mean
    degradation accelerate (b > 1). b is chosen by maximum profile
    likelihood on the train cells; everything else is the linear model
    (Xu et al. 2019) on transformed time."""
    name = "wiener_v2"

    def __init__(self, block=1, scale=1000.0, b_grid=np.arange(0.5, 4.01, 0.1)):
        super().__init__(block)
        self.s, self.b_grid = scale, b_grid

    def _L(self, x):
        return (np.clip(x, 0, None) / self.s) ** self.b

    def fit_population(self, curves):
        best = None
        for b in self.b_grid:
            self.b = b
            ll = 0.0
            for x, q in curves:
                xb, qb = self._binned(self._L(x), q)
                X = qb[0] - qb
                dt, dx = np.diff(xb), np.diff(X)
                ok = dt > 0
                dt, dx = dt[ok], dx[ok]
                lam = dx.sum() / dt.sum()
                s2 = max(np.mean((dx - lam * dt) ** 2 / dt), 1e-12)
                ll += -0.5 * np.sum(np.log(2 * np.pi * s2 * dt) + (dx - lam * dt) ** 2 / (s2 * dt))
            if best is None or ll > best[0]:
                best = (ll, b)
        self.b = best[1]
        super().fit_population([(self._L(x), q) for x, q in curves])

    def fit(self, x, q):
        super().fit(self._L(x), q)

    def sample(self, x_new, n, rng, latent=False):
        noise = self.noise
        if latent:
            self.noise = 0.0
        try:
            return super().sample(self._L(x_new), n, rng)
        finally:
            self.noise = noise


class CrossCellCapacity(Forecaster):
    """Severson et al. (2019) SI Note 3 "multivariate discharge curve,
    cycle N" benchmark: elastic net on log10 cycle life using seven
    features of the first N cycles' discharge capacity (SI Table 2):
    slope and intercept of a line through cycles 2..N, slope and
    intercept through the last 10 cycles, Q(2), max Q - Q(2), Q(N).
    Trained on train cells for each origin N (cells already past EOL at
    N excluded). Published mean % error on cycle life: N=100, 18.8 train
    / 78.5 primary (23.1 without one outlier) / 50.0 secondary; N=300,
    12.5 / 26.9 / 45.5.

    It predicts a cycle life, not a trajectory. To score it like the
    others, a trajectory is drawn from each sampled life L with the
    train cells' median post-origin fade shape: q(x) = q_N - (q_N - EOL)
    g((x - N) / (L - N)). Life uncertainty = Gaussian in log10 life with
    the train cross-validated residual s.d."""
    name = "cross_cell_capacity"

    def __init__(self, eol_ah, min_train=10):
        self.eol_ah, self.min_train = eol_ah, min_train
        self._cache = {}

    @staticmethod
    def features(x, q):
        h = x >= 2
        x2, q2 = x[h], q[h]
        b1, a1 = np.polyfit(x2, q2, 1)
        b2, a2 = np.polyfit(x2[-10:], q2[-10:], 1)
        return np.array([b1, a1, b2, a2, q2[0], q2.max() - q2[0], q2[-10:].mean()])

    def fit_population(self, curves):
        self.curves = curves  # truncated at EOL, so EOL = last x

    def _train(self, N):
        from sklearn.linear_model import ElasticNetCV
        from sklearn.model_selection import cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        F, y, shapes = [], [], []
        u = np.linspace(0, 1, 51)
        for x, q in self.curves:
            if x[-1] <= N + 10:
                continue
            h = x <= N
            F.append(self.features(x[h], q[h]))
            y.append(np.log10(x[-1]))
            qN = q[h][-10:].mean()
            uu = (x[~h] - N) / (x[-1] - N)
            vv = (qN - q[~h]) / max(qN - self.eol_ah, 1e-6)
            shapes.append(np.interp(u, np.concatenate([[0], uu]), np.concatenate([[0], vv])))
        F, y = np.array(F), np.array(y)
        model = make_pipeline(StandardScaler(), ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=4, max_iter=50000, random_state=0))
        resid = y - cross_val_predict(model, F, y, cv=4)
        model.fit(F, y)
        g = np.median(np.array(shapes), 0)
        g = np.maximum.accumulate(np.clip(g, 0, None))
        g[-1] = 1.0
        return model, float(np.std(resid, ddof=1)), u, g, len(y)

    def fit(self, x, q):
        N = float(x[-1])
        if N not in self._cache:
            self._cache[N] = self._train(N)
        self.model, self.s_log, self.u, self.g, self.n_train = self._cache[N]
        self.N = N
        self.mu_log = float(self.model.predict(self.features(x, q)[None, :])[0])
        self.qN = q[-10:].mean()
        self.noise = float(np.std(q[-20:] - np.polyval(np.polyfit(x[-20:], q[-20:], 1), x[-20:])))

    def _shape(self, uu):
        slope = (self.g[-1] - self.g[-3]) / (self.u[-1] - self.u[-3])
        return np.where(uu <= 1, np.interp(uu, self.u, self.g), 1 + slope * (uu - 1))

    def sample(self, x_new, n, rng, latent=False):
        L = 10 ** rng.normal(self.mu_log, self.s_log, size=(n, 1))
        L = np.maximum(L, self.N + 5)
        uu = np.clip((x_new[None, :] - self.N) / (L - self.N), 0, None)
        m = self.qN - (self.qN - self.eol_ah) * self._shape(uu)
        if latent:
            return m
        return m + rng.normal(0, self.noise, size=m.shape)
