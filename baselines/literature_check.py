"""Like-for-like checks of two baselines against published numbers.

These runs copy each paper's own protocol, NOT ours, so the numbers can
be put next to the published ones. They are sanity checks of the model
code; they are not used to choose anything. Train and val cells only:
the locked Severson test batch (2018-04-12) is never loaded here.

1. thelen: empirical exponential fade fit, Thelen et al. (2022) Applied
   Energy 324 119624, Sec. 2.2.2 and 3:
   * capacity normalised by each cell's FIRST measurement;
   * first prediction time (FPT) = first cycle with normalised capacity
     < 0.97; predictions every 5th cycle from FPT until EOL;
   * EOL = first cycle with normalised capacity < 0.80. Cells that stop
     before 0.80 are "artificially extended by 100 cycles by linearly
     extrapolating the previous 50 normalized capacity measurements";
   * RUL = EOL - t; score = RMSE of predicted vs true RUL in cycles
     (their eq. 21). Eq. 21 does not say whether predictions are pooled
     across cells or RMSE is taken per cell and then averaged, so both
     are reported. Point prediction = crossing of the fitted mean curve.
   Published (169 LFP, exponential model-based GP, Table 4 / thesis
   Table 3.5): 67.9 train, 147.4 primary test, 105.0 secondary test.
   Difference: their model is a GP with the exponential fit as its trend
   (the GP absorbs residuals); ours is the parametric fit alone. Their
   exact exponential form is in their Supplementary Note 1 (not
   available here); ours is a1 + a2 exp(a3 t).

2. liu: zero-mean squared-exponential GP, Liu et al. (2013)
   Microelectron. Reliab. 53 832, Sec. 4: train on one cell's capacity
   at cycles 1..100, forecast the next 68 (NASA B0005-7 have 168),
   SOH = C_i / C_0 x 100 (their eq. 2), RMSE and MAPE in SOH %.
   Published basic GPR: RMSE 13.03 / 22.51 / 20.70 SOH points, MAPE
   12.1 / 27.0 / 19.2 % (B0005 / B0006 / B0007, their Tables 1-3).
   Severson cells barely fade in 168 cycles, so the literal 100 -> 68
   setup is also run in Liu's PROPORTIONS: train on the first 100/168 of
   the cell's life, forecast the rest up to EOL. That origin uses the
   true lifetime, which our own protocol forbids; it is acceptable only
   here, as a reproduction check.

Usage:
    python -m baselines.literature_check thelen
    python -m baselines.literature_check liu
"""
from __future__ import annotations

import os

# One BLAS thread per worker process: with 8 workers each running a
# multi-threaded BLAS, small GP fits ran ~10x slower (measured 18 Nov 2026).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C

from baselines.models import (FadeFit, FadeFitGP, FadeFitPrior, PlainGP, PlainGPBayes,
                             PlainGPLinearMean, _exp_model, _knee_model)
from data_pipeline import severson, splits

OUT = Path(__file__).resolve().parents[1] / "results" / "baselines" / "literature_check"


# ---------------------------------------------------------------- data
def untruncated_dev_curves():
    """Train+val Severson cells with their FULL raw curves (not cut at
    0.88 Ah), merged across continuation files exactly as severson.load()
    does. Thelen's 80%-of-first-measurement EOL can lie below 0.88 Ah."""
    ds = severson.load()
    sp = splits.get("severson")  # test not returned without final_evaluation
    dev = {c: p for p in ("train", "val") for c in sp[p]}
    _, cyc = severson.load_raw(clean=True)
    out = {}
    for cid, part in dev.items():
        pieces, offset = [], 0
        for fid in cid.split("+"):  # load() keeps file order (earlier run first)
            c = cyc[cyc.cell_id == fid].sort_values("cycle")
            pieces.append(c["cycle"].to_numpy(float) + offset)
            offset = int(pieces[-1].max())
            pieces[-1] = (pieces[-1], c["capacity_ah"].to_numpy(float))
        x = np.concatenate([p[0] for p in pieces])
        q = np.concatenate([p[1] for p in pieces])
        out[cid] = (part, x, q)
    return ds, out


# -------------------------------------------------------------- Thelen
def thelen_eol(x, qn):
    below = np.flatnonzero(qn < 0.80)
    if below.size:
        return float(x[below[0]]), "observed"
    # Thelen: extend 100 cycles by a line through the last 50 points
    b, a = np.polyfit(x[-50:], qn[-50:], 1)
    if b >= 0:
        return np.nan, "no crossing"
    t = (0.80 - a) / b
    return float(t), ("extended" if t <= x[-1] + 100 else "extended>100")



def _crossing(x_grid, curve, thr):
    i = np.flatnonzero(curve < thr)
    return float(x_grid[i[0]]) if i.size else np.inf


RUL_LEVELS = np.linspace(0.01, 0.99, 49)  # stored RUL quantiles, for recalibration
RUL_CAP = 1e5  # forecasts that never reach 80% within the grid


def make_model(variant, prior=None):
    """Variants tried against the published numbers. Every one is logged
    in experiment_log.jsonl, including the ones that did not help."""
    if variant == "fadefit":
        return FadeFit()
    if variant.startswith("fadefit_gp"):
        scale = float(variant.split("_s")[1]) if "_s" in variant else 1.0
        return FadeFitGP(scale=scale)
    if variant == "fadefit_prior":
        m = FadeFitPrior()
        m.mu, m.Sigma, m.sigma = prior
        m.Linv = np.linalg.inv(np.linalg.cholesky(m.Sigma))
        return m
    raise ValueError(variant)


def latent_samples(m, grid, n, rng):
    """Noise-free forecast trajectories, used for EOL (measurement noise
    would trigger spurious early crossings)."""
    if isinstance(m, FadeFitGP):
        return m.sample_latent(grid, n, rng)
    if isinstance(m, FadeFitPrior):
        return m.sample(grid, n, rng, latent=True)
    if m.fallback is not None:
        f = m.fallback
        b = rng.multivariate_normal(f.beta, f.cov, size=n)
        return b[:, :1] + b[:, 1:] * (grid - f.x0)[None, :]
    P = rng.multivariate_normal(m.p, m.cov, size=n, check_valid="ignore")
    return _exp_model((grid / m.xs)[None, :], P[:, :1], P[:, 1:2], P[:, 2:3])


def mean_curve(m, grid):
    if isinstance(m, FadeFitGP):
        return m._trend_mean(grid) + m.gp.predict((grid / m.xs)[:, None])
    if isinstance(m, FadeFitPrior):
        return _knee_model(grid, *m.th)
    if m.fallback is not None:
        b = m.fallback.beta
        return b[0] + b[1] * (grid - m.fallback.x0)
    return _exp_model(grid / m.xs, *m.p)


def thelen_cell(args):
    cid, part, x, q, variant, n_samples, seed, prior = args
    qn = q / q[0]
    eol, how = thelen_eol(x, qn)
    below97 = np.flatnonzero(qn < 0.97)
    rows = []
    if np.isnan(eol) or not below97.size:
        return rows
    fpt = x[below97[0]]
    rng = np.random.default_rng(seed)
    for t in [t for t in x if fpt <= t < eol and t % 5 == 0]:
        h = x <= t
        m = make_model(variant, prior)
        m.fit(x[h], qn[h])
        fine = np.arange(t + 1, t + 6000, 1.0)
        coarse = np.arange(t + 5, t + 4000, 5.0)
        e_mean = _crossing(fine, mean_curve(m, fine), 0.80)
        S = latent_samples(m, coarse, n_samples, rng)
        hit = S < 0.80
        e_s = np.where(hit.any(1), coarse[hit.argmax(1)], np.inf)
        rul_s = np.minimum(e_s - t, RUL_CAP)
        rq = np.quantile(rul_s, RUL_LEVELS)
        row = dict(cell_id=cid, part=part, t=t, fpt=fpt, eol=eol, eol_how=how,
                   rul_true=eol - t, rul_mean_curve=e_mean - t,
                   # PIT: predicted P(EOL <= true EOL); uniform if calibrated
                   pit=float(np.mean(e_s <= eol)),
                   lo95=float(np.quantile(rul_s, 0.025)), hi95=float(np.quantile(rul_s, 0.975)))
        row.update({f"rq{i:02d}": float(v) for i, v in enumerate(rq)})
        rows.append(row)
    return rows


def thelen_priors(curves):
    """Population prior for FadeFitPrior in Thelen's normalised units:
    fitted on train cells' curves (capacity / first measurement, up to the
    80% crossing). Val cells get the prior from all train cells; each train
    cell gets a leave-one-out prior so its own curve never informs it."""
    def prep(x, q):
        qn = q / q[0]
        k = np.flatnonzero(qn < 0.80)
        end = k[0] + 1 if k.size else len(x)
        return x[:end], qn[:end]
    train = {c: prep(x, q) for c, (part, x, q) in curves.items() if part == "train"}

    def fit(ids):
        m = FadeFitPrior()
        m.fit_population([train[c] for c in ids])
        return (m.mu, m.Sigma, m.sigma)
    allp = fit(list(train))
    return {c: (fit([d for d in train if d != c]) if part == "train" else allp)
            for c, (part, x, q) in curves.items()}


def calibration_error(pit, levels=np.linspace(0, 1, 20)):
    """Thelen et al. (2022) eq. 22-23 (after Kuleshov et al. 2018): share
    of true values at or below each predicted quantile level, compared
    with the level itself, mean absolute gap over 20 levels."""
    pit = np.asarray(pit)
    return float(np.mean([abs(np.mean(pit <= p) - p) for p in levels]))


def _rmse(e):
    e = np.asarray(e, float)
    e = np.where(np.isfinite(e), e, np.nan)
    return float(np.sqrt(np.nanmean(e ** 2)))


def summarise_thelen(df):
    df = df.copy()
    df["err"] = df.rul_mean_curve - df.rul_true
    df["alpha_ok"] = np.abs(df.err) <= 0.1 * df.rul_true
    df["cov95"] = (df.lo95 <= df.rul_true) & (df.rul_true <= df.hi95)
    out = []
    for part, d in list(df.groupby("part")) + [("train+val", df)]:
        g = d.groupby("cell_id")
        merged = d.cell_id.str.contains("+", regex=False)
        out.append(dict(
            part=part, n_cells=d.cell_id.nunique(), n_predictions=len(d),
            rmse_cellavg=float(g["err"].apply(_rmse).mean()),
            rmse_cellmedian=float(g["err"].apply(_rmse).median()),
            rmse_pooled=_rmse(d.err), rmse_pooled_no_merged=_rmse(d.err[~merged]),
            alpha_acc_pct=100 * float(g["alpha_ok"].mean().mean()),
            calib_err_cellavg=float(g["pit"].apply(calibration_error).mean()),
            calib_err_pooled=calibration_error(d.pit),
            coverage95=float(d.cov95.mean()),
            n_never_crossing=int((~np.isfinite(d.err)).sum())))
    return pd.DataFrame(out).round(3)


def code_version():
    """Short hash of the model and protocol code, stored with every result
    so a cached result is only reused if the code that made it is unchanged."""
    import hashlib
    here = Path(__file__).resolve().parent
    h = hashlib.sha1()
    for f in ("models.py", "literature_check.py"):
        h.update((here / f).read_bytes())
    return h.hexdigest()[:10]


def log_experiment(check, variant, summ, note=""):
    """Append one JSON record per summary row to experiment_log.jsonl, and
    save the summary with a sidecar recording the code version."""
    import json
    OUT.mkdir(parents=True, exist_ok=True)
    meta = dict(run_at=pd.Timestamp.now().isoformat(timespec="seconds"),
                check=check, variant=variant, code=code_version(), note=note)
    with open(OUT / "experiment_log.jsonl", "a") as f:
        for rec in summ.reset_index(drop=True).to_dict("records"):
            f.write(json.dumps({**meta, **rec}, default=float) + "\n")
    summ.to_csv(OUT / f"{check}_{variant}_summary.csv", index=False)
    (OUT / f"{check}_{variant}_summary.json").write_text(json.dumps(meta))


def run_thelen(variant="fadefit", workers=8, n_samples=200, note=""):
    _, curves = untruncated_dev_curves()
    priors = thelen_priors(curves) if variant == "fadefit_prior" else {}
    jobs = [(cid, part, x, q, variant, n_samples, i, priors.get(cid))
            for i, (cid, (part, x, q)) in enumerate(curves.items())]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rs in ex.map(thelen_cell, jobs) for r in rs]
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"thelen_{variant}_predictions.csv", index=False)
    s = summarise_thelen(df)
    s.to_csv(OUT / f"thelen_{variant}_summary.csv", index=False)
    log_experiment("thelen", variant, s, note)
    return s


def _pit_from_quantiles(LQ, y):
    """PIT of y (n,) under each row's quantile function, given log-RUL
    quantiles LQ (n, len(RUL_LEVELS)); linear between stored levels,
    0 / 1 outside them."""
    k = (LQ < y[:, None]).sum(1)
    out = np.where(k == 0, 0.0, 1.0)
    mid = (k > 0) & (k < LQ.shape[1])
    i = k[mid]
    lo, hi = LQ[mid, i - 1], LQ[mid, i]
    w = np.where(hi > lo, (y[mid] - lo) / np.where(hi > lo, hi - lo, 1), 0.5)
    out[mid] = RUL_LEVELS[i - 1] + w * (RUL_LEVELS[i] - RUL_LEVELS[i - 1])
    return out


def run_thelen_recal(base, note="", seed=0):
    """Post-hoc recalibration of a Thelen-check variant (in the spirit of
    Kuleshov et al. 2018, the reference Thelen use for calibration). Each
    forecast's RUL distribution (its 49 stored quantiles, treated as
    equally weighted samples) is dressed with log-normal error:
        log r' = log r + a + s * eps,   eps ~ N(0, 1).
    a removes a systematic late/early bias; s adds the spread the
    parametric fit cannot see (model misfit). (a, s) minimise the
    cell-averaged calibration error on TRAIN predictions only; val is
    scored with them unchanged. Point forecast = median of r'.
    (A first version scaled the model's own spread instead; the scale ran
    to the edge of its grid twice because that spread is near zero.)"""
    from scipy.stats import norm
    df = pd.read_csv(OUT / f"thelen_{base}_predictions.csv")
    cols = [f"rq{i:02d}" for i in range(len(RUL_LEVELS))]
    LQ = np.log(np.maximum(df[cols].to_numpy(), 1.0))
    ly = np.log(df.rul_true.to_numpy())
    tr = (df.part == "train").to_numpy()
    cells = df.cell_id.to_numpy()

    def pit(a, s):
        return norm.cdf((ly[:, None] - a - LQ) / s).mean(1)

    def calib(p, mask):
        d = pd.DataFrame({"c": cells[mask], "p": p[mask]})
        return float(d.groupby("c")["p"].apply(calibration_error).mean())

    best = None
    for s_ in np.geomspace(0.01, 3.0, 40):
        for a in np.linspace(-1.5, 1.5, 61):
            e = calib(pit(a, s_), tr)
            if best is None or e < best[0]:
                best = (e, a, s_)
    _, a, s_ = best
    rng = np.random.default_rng(seed)
    draws = LQ[:, :, None] + a + s_ * rng.normal(size=(1, 1, 40))
    draws = draws.reshape(len(df), -1)
    out = df.copy()
    out["pit"] = pit(a, s_)
    out["rul_mean_curve"] = np.exp(np.median(draws, 1))
    out["lo95"] = np.exp(np.quantile(draws, 0.025, axis=1))
    out["hi95"] = np.exp(np.quantile(draws, 0.975, axis=1))
    variant = f"{base}_recal"
    out.to_csv(OUT / f"thelen_{variant}_predictions.csv", index=False)
    summ = summarise_thelen(out)
    summ.to_csv(OUT / f"thelen_{variant}_summary.csv", index=False)
    log_experiment("thelen", variant, summ, f"a={a:.3f}, s={s_:.3f} fitted on train. {note}")
    edge = " (AT GRID EDGE)" if s_ in (0.01, 3.0) or abs(a) == 1.5 else ""
    print(f"recalibration from train: shift a = {a:.3f} (x{np.exp(a):.2f} on RUL), "
          f"log-RUL error s.d. s = {s_:.3f}{edge}")
    return summ


def cv_thelen_recal(base, k=5):
    """Grouped k-fold CV of run_thelen_recal's (a, s): fitted on k-1
    folds of the 83 dev cells, scored on the held-out fold, folds grouped
    by charging protocol. Each cell is scored once, out of fold."""
    from scipy.stats import norm
    from sklearn.model_selection import GroupKFold
    df = pd.read_csv(OUT / f"thelen_{base}_predictions.csv")
    grp = severson.load().cells.set_index("cell_id")["group"]
    cols = [f"rq{i:02d}" for i in range(len(RUL_LEVELS))]
    LQ = np.log(np.maximum(df[cols].to_numpy(), 1.0))
    ly = np.log(df.rul_true.to_numpy())
    cells = df.cell_id.to_numpy()
    uc = np.unique(cells)
    out = df.copy()
    fits = []
    for i, (tr_i, te_i) in enumerate(GroupKFold(k).split(uc, groups=grp.loc[uc].to_numpy())):
        tr, te = np.isin(cells, uc[tr_i]), np.isin(cells, uc[te_i])
        best = None
        for s_ in np.geomspace(0.01, 3.0, 40):
            for a in np.linspace(-1.5, 1.5, 61):
                p_ = norm.cdf((ly[tr, None] - a - LQ[tr]) / s_).mean(1)
                e = float(pd.DataFrame({"c": cells[tr], "p": p_}).groupby("c")["p"]
                          .apply(calibration_error).mean())
                if best is None or e < best[0]:
                    best = (e, a, s_)
        _, a, s_ = best
        fits.append((i, round(a, 3), round(s_, 3)))
        rng = np.random.default_rng(i)
        draws = (LQ[te][:, :, None] + a + s_ * rng.normal(size=(1, 1, 40))).reshape(te.sum(), -1)
        out.loc[te, "pit"] = norm.cdf((ly[te, None] - a - LQ[te]) / s_).mean(1)
        out.loc[te, "rul_mean_curve"] = np.exp(np.median(draws, 1))
        out.loc[te, "lo95"] = np.exp(np.quantile(draws, 0.025, axis=1))
        out.loc[te, "hi95"] = np.exp(np.quantile(draws, 0.975, axis=1))
    out["part"] = "cv_out_of_fold"
    summ = summarise_thelen(out).iloc[:1]
    summ.to_csv(OUT / f"thelen_{base}_recal_cv{k}_summary.csv", index=False)
    print("fold fits (fold, a, s):", fits)
    return summ


# ---------------------------------------------------------------- Liu
def liu_basic_gpr(x, q):
    """Liu et al. 2013 basic GPR: zero mean (no normalisation of y),
    squared-exponential kernel + noise, initial [l, sigma_f] = [1, 1],
    hyperparameters by maximum likelihood, raw cycle number as input."""
    k = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e4)) + WhiteKernel(1e-4, (1e-8, 1.0))
    gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=2,
                                  random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(x[:, None], q)
    return gp


def liu_cell(args):
    cid, part, x, q, eol = args
    soh = lambda c: 100 * c / q[0]
    rows = []
    setups = {"literal_100_to_168": (100, 168),
              "proportional_60pct_of_life": (eol * 100 / 168, eol)}
    for name, (n_train, n_end) in setups.items():
        h, f = x <= n_train, (x > n_train) & (x <= n_end)
        if h.sum() < 10 or f.sum() < 2:
            continue
        xf, qf = x[f], q[f]
        for model in ("liu_basic_gpr", "our_plain_gp"):
            if model == "liu_basic_gpr":
                mu, sd = liu_basic_gpr(x[h], q[h]).predict(xf[:, None], return_std=True)
            else:
                m = PlainGP()
                m.fit(x[h], q[h])
                mu, sd = m.gp.predict((xf / m.xs)[:, None], return_std=True)
            err = soh(mu) - soh(qf)
            rows.append(dict(cell_id=cid, part=part, setup=name, model=model,
                             n_train=int(h.sum()), n_forecast=int(f.sum()),
                             rmse_soh_pts=float(np.sqrt(np.mean(err ** 2))),
                             mape_pct=float(100 * np.mean(np.abs(mu - qf) / qf)),
                             cov95=float(np.mean(np.abs(mu - qf) <= 1.96 * sd)),
                             soh_drop_true=float(soh(qf[-1]) - soh(q[h][-1])),
                             soh_drop_pred=float(soh(mu[-1]) - soh(q[h][-1]))))
    return rows


def run_liu(workers=8):
    ds = severson.load()
    sp = splits.get("severson")
    eol = ds.cells.set_index("cell_id")["eol_x"]
    jobs = []
    for part in ("train", "val"):
        for cid in sp[part]:
            x, q = ds.curve(cid)
            jobs.append((cid, part, x, q, float(eol[cid])))
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rs in ex.map(liu_cell, jobs) for r in rs]
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "liu_gp_per_cell.csv", index=False)
    s = df.groupby(["setup", "model"]).agg(
        n_cells=("cell_id", "count"),
        rmse_soh_mean=("rmse_soh_pts", "mean"), rmse_soh_median=("rmse_soh_pts", "median"),
        mape_mean=("mape_pct", "mean"), cov95=("cov95", "mean"),
        soh_drop_true=("soh_drop_true", "median"),
        soh_drop_pred=("soh_drop_pred", "median")).round(3)
    for model, g in s.reset_index().groupby("model"):
        log_experiment("liu", model, g.drop(columns="model"), "Liu protocol")
    return s



# -------------------------------------------------------------- Kenney
KENNEY_PUBLISHED_GP = {0.5: 5.45, 0.6: 6.08, 0.7: 5.86, 0.8: 3.67}  # Table 4, % MAPE


def kenney_curve(x, q, n=50):
    """Kenney et al. (2024) Sec. 2.5: capacity normalised to the curve's
    maximum, resampled to 50 points over x in [0, 1]. The x axis is the
    cell's own life, so the context fraction leaks EOL (their protocol,
    not ours)."""
    xn = (x - x[0]) / (x[-1] - x[0])
    g = np.linspace(0, 1, n)
    return g, np.interp(g, xn, q / q.max())


def make_kenney_model(variant):
    if variant == "plain_gp":
        return PlainGP(restarts=2)
    if variant == "plain_gp_bayes":
        return PlainGPBayes(x_unit=1.0)
    if variant == "plain_gp_zeromean":
        return PlainGP(restarts=2, normalize_y=False)
    if variant == "gp_linear_mean":
        return PlainGPLinearMean(restarts=2, normalize_y=False)
    raise ValueError(variant)


def kenney_cell(args):
    cid, part, x, q, variant = args
    g, s = kenney_curve(x, q)
    rows = []
    for ctx, pub in KENNEY_PUBLISHED_GP.items():
        k = int(round(ctx * len(g)))
        m = make_kenney_model(variant)
        m.fit(g[:k], s[:k])
        mu, sd = m.predict(g[-5:]) if hasattr(m, "predict") else (
            m.gp.predict((g[-5:] / m.xs)[:, None], return_std=True))
        y = s[-5:]
        rows.append(dict(cell_id=cid, part=part, context=ctx,
                         mape=float(100 * np.mean(np.abs(y - mu) / y)),
                         nlpd=float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2)
                                            + (y - mu) ** 2 / (2 * sd ** 2)))))
    return rows


def run_kenney(variant="plain_gp", workers=8, note=""):
    ds = severson.load()
    sp = splits.get("severson")
    jobs = [(c, p, *ds.curve(c), variant) for p in ("train", "val") for c in sp[p]]
    with ProcessPoolExecutor(workers) as ex:
        df = pd.DataFrame([r for rs in ex.map(kenney_cell, jobs) for r in rs])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"kenney_{variant}_per_cell.csv", index=False)
    s = df.groupby(["part", "context"]).agg(
        n_cells=("cell_id", "count"), mape=("mape", "mean"),
        mape_median=("mape", "median"), nlpd=("nlpd", "mean")).reset_index()
    s["published_gp_mape"] = s.context.map(KENNEY_PUBLISHED_GP)
    s = s.round(3)
    log_experiment("kenney", variant, s, note)
    return s


# ---------------------------------------------------------------- Zou
ZOU_PUBLISHED = {50: 1636.56, 100: 1562.07}  # linear persistence, EOL MAE cycles


def run_zou(note=""):
    """Zou et al. (2026) Sec. 2.1 and 3: SOH = Q(t)/Q(1), EOL = first cycle
    with SOH <= 0.8, given the first N cycles. Their "linear-persistence
    extrapolation" is not specified further, so two readings are run:
    a line through all N cycles, and through the last 10. Lines with no
    downward slope never reach 0.8; they are counted and given the
    longest EOL seen in the train cells (so the MAE stays finite)."""
    _, curves = untruncated_dev_curves()
    eols = {}
    for cid, (part, x, q) in curves.items():
        e, _ = thelen_eol(x, q / q[0])  # same 80%-of-first rule, same extension
        eols[cid] = e
    cap = max(e for c, e in eols.items() if curves[c][0] == "train" and np.isfinite(e))
    rows = []
    for cid, (part, x, q) in curves.items():
        if not np.isfinite(eols[cid]):
            continue
        s = q / q[0]
        for N in ZOU_PUBLISHED:
            h = x <= N
            for win, xs, ss in (("all_N", x[h], s[h]), ("last_10", x[h][-10:], s[h][-10:])):
                b, a = np.polyfit(xs, ss, 1)
                pred = (0.8 - a) / b if b < 0 else np.inf
                rows.append(dict(cell_id=cid, part=part, N=N, window=win,
                                 eol_true=eols[cid], eol_pred=pred,
                                 never=not np.isfinite(pred) or pred > cap))
    df = pd.DataFrame(rows)
    df["abs_err"] = np.abs(np.minimum(df.eol_pred, cap) - df.eol_true)
    df["abs_err_uncapped"] = np.abs(df.eol_pred - df.eol_true)
    s = df.groupby(["window", "N"]).agg(
        n_cells=("cell_id", "count"), mae_capped=("abs_err", "mean"),
        median_ae_uncapped=("abs_err_uncapped", "median"),
        frac_beyond_cap=("never", "mean")).reset_index()
    s["published_mae"] = s.N.map(ZOU_PUBLISHED)
    s = s.round(2)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "zou_naive_per_cell.csv", index=False)
    log_experiment("zou", "linear_persistence", s, note)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["thelen", "liu", "kenney", "zou"])
    ap.add_argument("--variant", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.check == "thelen" and a.variant and a.variant.endswith("_recal"):
        s = run_thelen_recal(a.variant[:-len("_recal")], note=a.note)
    elif a.check == "thelen":
        s = run_thelen(a.variant or "fadefit", a.workers, note=a.note)
    elif a.check == "kenney":
        s = run_kenney(a.variant or "plain_gp", a.workers, note=a.note)
    elif a.check == "zou":
        s = run_zou(note=a.note)
    else:
        s = run_liu(a.workers)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(s)


if __name__ == "__main__":
    main()
