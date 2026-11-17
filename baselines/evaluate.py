"""Forecast-origin evaluation of the baselines.

Protocol (fixed here, reused for every later model):

* Forecast origins are FIXED amounts of observed history, never a
  fraction of each cell's life: choosing the origin from the true
  lifetime would leak the answer. Severson: first N in {100, 200, 400}
  cycles. Kirkaldy: first k in {4, 6, 8} RPT measurements (BoL included).
  Multiple origins per cell follow Tashman's recommendation as discussed
  by Cerqueira et al. (2020).
* A (cell, origin) pair is skipped when the cell has already reached EOL
  inside its history (Severson) or has < 2 future points (Kirkaldy);
  skip counts are reported.
* Scored on: future observed capacities (Severson thinned to cycles that
  are multiples of 5, up to EOL) for RMSE / CRPS / 95% coverage; and on
  sampled EOL (forecast grid out to ``horizon``) for EOL error, EOL 95%
  coverage and EOL CRPS, for cells whose EOL was observed.
* Every model's samples are clipped to [0, 1.1 x max observed capacity]
  before scoring (a physical bound, applied identically to all models).
* Only train and val cells are evaluated here. The locked test sets are
  touched only with ``--final`` (not used yet, by design).

Usage:
    python -m baselines.evaluate severson
    python -m baselines.evaluate kirkaldy
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.models import FadeFit, NaiveSlope, PlainGP, Wiener
from data_pipeline import kirkaldy, metrics as M, severson, splits

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "baselines"

CONFIG = {
    "severson": dict(loader=severson.load, origins=[100, 200, 400], origin_kind="x",
                     horizon=3000.0, grid_step=5.0, thin=5,
                     naive_windows=[10, 25, 50, 100], wiener_block=10),
    "kirkaldy": dict(loader=kirkaldy.load, origins=[4, 6, 8], origin_kind="points",
                     horizon=2000.0, grid_step=5.0, thin=None,
                     naive_windows=[2, 3, 4], wiener_block=1),
}
N_SAMPLES = 300


def _history(x, q, origin, kind):
    m = (x <= origin) if kind == "x" else (np.arange(len(x)) < origin)
    return m


def evaluate_cell(model, x, q, eol_true, origin, cfg, ds, rng):
    hist = _history(x, q, origin, cfg["origin_kind"])
    xh, qh = x[hist], q[hist]
    xf, qf = x[~hist], q[~hist]
    if cfg["origin_kind"] == "x" and not np.isnan(eol_true) and eol_true <= origin:
        return None
    if len(xf) < 2:
        return None
    if cfg["thin"]:
        keep = (xf % cfg["thin"]) == 0
        xf, qf = xf[keep], qf[keep]
    grid = np.arange(xh[-1] + cfg["grid_step"], cfg["horizon"] + 1e-9, cfg["grid_step"])
    x_all = np.union1d(xf, grid)
    t0 = time.perf_counter()
    model.fit(xh, qh)
    S = model.sample(x_all, N_SAMPLES, rng)
    secs = time.perf_counter() - t0
    # Same physical bound for every model: capacity in [0, 1.1 x max seen].
    S = np.clip(S, 0.0, 1.1 * qh.max())
    idx = np.searchsorted(x_all, xf)
    Sf = S[:, idx]
    row = dict(n_hist=len(xh), n_future=len(xf), fit_seconds=secs,
               rmse=M.rmse(qf, Sf.mean(0)),
               crps=float(M.crps_samples(qf, Sf).mean()),
               coverage95=M.coverage(qf, Sf))
    if not np.isnan(eol_true):
        g = x_all >= xh[-1]
        eols = M.eol_samples(x_all[g], S[:, g], ds.eol_capacity_ah)
        row.update(M.eol_metrics(eol_true, eols, cfg["horizon"]))
    return row, (xh, qh, xf, qf, x_all, S)


def run_part(models, ds, cfg, cell_ids, part, seed=0, keep_examples=()):
    rows, examples = [], {}
    eol = ds.cells.set_index("cell_id")["eol_x"]
    for mi, model in enumerate(models):
        for ci, cid in enumerate(cell_ids):
            x, q = ds.curve(cid)
            for origin in cfg["origins"]:
                rng = np.random.default_rng([seed, mi, ci, origin])
                res = evaluate_cell(model, x, q, eol[cid], origin, cfg, ds, rng)
                if res is None:
                    rows.append(dict(model=model.name, part=part, cell_id=cid,
                                     origin=origin, skipped=True))
                    continue
                r, ex = res
                rows.append(dict(model=model.name, part=part, cell_id=cid,
                                 origin=origin, skipped=False, eol_true=eol[cid], **r))
                if cid in keep_examples:
                    examples[(model.name, cid, origin)] = ex
    return pd.DataFrame(rows), examples


def summarise(df):
    d = df[~df.skipped]
    agg = d.groupby(["part", "model", "origin"]).agg(
        n_cells=("cell_id", "count"),
        rmse_mah=("rmse", lambda s: 1000 * s.mean()),
        rmse_mah_median=("rmse", lambda s: 1000 * s.median()),
        crps_mah=("crps", lambda s: 1000 * s.mean()),
        coverage95=("coverage95", "mean"),
        eol_n=("eol_error", "count"),
        eol_rmse=("eol_error", lambda s: float(np.sqrt(np.nanmean(s**2)))),
        eol_mape=("eol_abs_pct_error", "mean"),
        eol_ape_median=("eol_abs_pct_error", "median"),
        eol_cov95=("eol_covered", "mean"),
        eol_crps=("eol_crps", "mean"),
        eol_censored=("eol_censored_frac", "mean"),
    )
    skipped = df[df.skipped].groupby(["part", "model", "origin"]).size().rename("n_skipped")
    return agg.join(skipped).fillna({"n_skipped": 0}).round(3)


def tune_naive_window(ds, cfg, train_ids):
    """Pick the naive-slope window by mean CRPS on TRAIN cells only."""
    scores = {}
    for w in cfg["naive_windows"]:
        df, _ = run_part([NaiveSlope(w)], ds, cfg, train_ids, "train")
        scores[w] = df.loc[~df.skipped, "crps"].mean()
    best = min(scores, key=scores.get)
    return best, scores


def plot_examples(examples, ds, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted(examples)
    model_names = sorted({k[0] for k in keys})
    cells = sorted({(k[1], k[2]) for k in keys})
    fig, axes = plt.subplots(len(cells), len(model_names),
                             figsize=(3.2 * len(model_names), 2.4 * len(cells)),
                             sharey="row", squeeze=False)
    for i, (cid, origin) in enumerate(cells):
        x, q = ds.curve(cid)
        for j, mname in enumerate(model_names):
            ax = axes[i, j]
            ex = examples.get((mname, cid, origin))
            ax.plot(x, q, color="0.6", lw=0.8, label="observed")
            if ex is not None:
                xh, qh, xf, qf, x_all, S = ex
                lo, hi = M.interval(S)
                ax.fill_between(x_all, lo, hi, color="C0", alpha=0.25, lw=0, label="95% PI")
                ax.plot(x_all, np.median(S, 0), color="C0", lw=1.2, label="median")
                ax.axvline(xh[-1], color="k", ls=":", lw=0.8)
            ax.axhline(ds.eol_capacity_ah, color="C3", lw=0.6, ls="--")
            ax.set_xlim(0, max(x.max() * 1.3, xh[-1] * 2 if ex else 1))
            ax.set_ylim(ds.eol_capacity_ah * 0.9, q.max() * 1.02)
            if i == 0:
                ax.set_title(mname, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{cid[-12:]}\norigin {origin}", fontsize=7)
            if i == len(cells) - 1:
                ax.set_xlabel(ds.x_name, fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(CONFIG))
    ap.add_argument("--final", action="store_true",
                    help="ALSO score the locked test set. Final evaluation only.")
    a = ap.parse_args()
    cfg = CONFIG[a.dataset]
    ds = cfg["loader"]()
    sp = splits.get(a.dataset, final_evaluation=a.final)
    OUT.mkdir(parents=True, exist_ok=True)

    w, wscores = tune_naive_window(ds, cfg, sp["train"])
    print(f"naive window by train CRPS: {w}  ({ {k: round(v*1000, 2) for k, v in wscores.items()} } mAh)")

    wiener = Wiener(block=cfg["wiener_block"])
    wiener.fit_population([ds.curve(c) for c in sp["train"]])
    print(f"Wiener prior from {len(sp['train'])} train cells: mu_lambda={wiener.mu_l:.3e}, "
          f"sigma_lambda={np.sqrt(wiener.s_l2):.3e}, sigma_B={np.sqrt(wiener.sB2):.3e} (Ah / sqrt({ds.x_name}))")

    models = [NaiveSlope(w), FadeFit(), wiener, PlainGP()]
    parts = ["train", "val"] + (["test"] if a.final else [])
    eol = ds.cells.set_index("cell_id")["eol_x"]
    val_ex = [c for c in sp["val"] if not np.isnan(eol[c])]
    # three example cells: shortest, median, longest observed life in val
    ex = sorted(val_ex, key=lambda c: eol[c])
    ex = [ex[0], ex[len(ex) // 2], ex[-1]] if len(ex) >= 3 else ex
    frames, examples = [], {}
    for part in parts:
        t0 = time.time()
        df, exm = run_part(models, ds, cfg, sp[part], part,
                           keep_examples=ex if part == "val" else ())
        frames.append(df)
        examples.update(exm)
        print(f"{part}: {len(sp[part])} cells, {time.time()-t0:.0f}s")
    per_cell = pd.concat(frames, ignore_index=True)
    per_cell.to_csv(OUT / f"{a.dataset}_per_cell.csv", index=False)
    summ = summarise(per_cell)
    summ.to_csv(OUT / f"{a.dataset}_summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ)
    # examples at the middle origin
    mid = cfg["origins"][1]
    plot_examples({k: v for k, v in examples.items() if k[2] == mid}, ds, cfg,
                  OUT / f"{a.dataset}_val_examples_origin{mid}.png")


if __name__ == "__main__":
    main()
