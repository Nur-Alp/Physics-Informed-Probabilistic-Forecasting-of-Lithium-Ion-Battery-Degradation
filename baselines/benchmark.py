"""Benchmark runner: baseline variants against published numbers.

One place for the loop "try a variant -> run the paper's protocol ->
compare with the paper". It wraps baselines/literature_check.py:

    python -m baselines.benchmark run thelen:fadefit_gp_s4 kenney:plain_gp_bayes
    python -m baselines.benchmark run --all        # every registered variant
    python -m baselines.benchmark board            # scoreboard vs targets

* Results are cached: a run is skipped if its summary exists and was made
  by the current code version (hash of models.py + literature_check.py).
  ``--force`` re-runs anyway.
* Selection rule, fixed in advance so the val numbers stay honest: for
  every target the "best" variant is the one with the best TRAIN value;
  its VAL value is the one compared with the paper. The Severson test
  batch is never touched (literature_check only loads train+val).
* Every run, including variants that did not help, is appended to
  results/baselines/literature_check/experiment_log.jsonl.

The scoreboard is written to results/baselines/literature_check/scoreboard.md.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from baselines import literature_check as L

OUT = L.OUT

# Variants per check. Add new ones here (and in literature_check's
# make_model / make_kenney_model).
VARIANTS = {
    "thelen": ["fadefit", "fadefit_gp", "fadefit_gp_s4", "fadefit_gp_s8", "fadefit_gp_s16",
               "fadefit_prior", "fadefit_prior_recal"],  # _recal reads its base's predictions
    "kenney": ["plain_gp", "plain_gp_zeromean", "plain_gp_bayes", "gp_linear_mean"],
    "zou": ["linear_persistence"],
    "liu": ["liu_basic_gpr"],  # one run produces liu_basic_gpr and our_plain_gp
}

# Published numbers. ``where`` selects the summary row; ``better`` says
# which direction wins. Values re-read from the PDFs on 18 Nov 2026.
TARGETS = [
    dict(check="thelen", metric="rmse_cellavg", where={}, value=147.4, better="lower",
         label="Fade fit: RUL RMSE (cycles)",
         source="Thelen 2022 Table 5, exponential model-based, primary test (secondary 105.0)"),
    dict(check="thelen", metric="calib_err_cellavg", where={}, value=0.06, better="lower",
         label="Fade fit: calibration error",
         source="Thelen 2022 Table 5, primary test (secondary 0.20)"),
    dict(check="thelen", metric="alpha_acc_pct", where={}, value=32.9, better="higher",
         label="Fade fit: alpha-accuracy (%)",
         source="Thelen 2022 Table 5, primary test (secondary 15.4)"),
    *[dict(check="kenney", metric="mape", where={"context": c}, value=v, better="lower",
           label=f"Plain GP: MAPE last 5 pts, {int(c*100)}% context",
           source="Kenney 2024 Table 4, GP without pre-training")
      for c, v in L.KENNEY_PUBLISHED_GP.items()],
    *[dict(check="zou", metric="mae_capped", where={"N": n, "window": "all_N"}, value=v,
           better="lower", label=f"Naive: EOL MAE (cycles), first {n} cycles",
           source="Zou 2026 Sec. 3, linear persistence (cap rule not stated)")
      for n, v in L.ZOU_PUBLISHED.items()],
    dict(check="liu", metric="rmse_soh_mean", where={"setup": "literal_100_to_168"}, value=13.03,
         better="lower", label="GP: RMSE SOH pts, train 100 -> forecast 68",
         source="Liu 2013 Tables 1-3, basic GPR, best of 13.03 / 22.51 / 20.70"),
]


def _meta(check, variant):
    p = OUT / f"{check}_{variant}_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def run(specs, force=False, workers=8):
    code = L.code_version()
    for spec in specs:
        check, variant = spec.split(":")
        m = _meta(check, variant)
        if not force and m and m.get("code") == code:
            print(f"skip {spec} (cached, code {code})")
            continue
        print(f"run  {spec}")
        if check == "thelen" and variant.endswith("_recal"):
            L.run_thelen_recal(variant[:-len("_recal")], note="benchmark")
        elif check == "thelen":
            L.run_thelen(variant, workers, note="benchmark")
        elif check == "kenney":
            L.run_kenney(variant, workers, note="benchmark")
        elif check == "zou":
            L.run_zou(note="benchmark")
        elif check == "liu":
            L.run_liu(workers)
        else:
            raise ValueError(check)


def _value(df, t, part):
    d = df
    for k, v in t["where"].items():
        if k in d:
            d = d[np.isclose(d[k], v)] if isinstance(v, float) else d[d[k] == v]
    if "part" in d:
        d = d[d.part == part]
    return float(d[t["metric"]].iloc[0]) if len(d) else np.nan


def board():
    code = L.code_version()
    lines = ["| Target | Published | Best variant (picked on train) | Train | Val | Meets? | All variants (val) |",
             "|---|---|---|---|---|---|---|"]
    for t in TARGETS:
        rows = []
        for f in sorted(OUT.glob(f"{t['check']}_*_summary.csv")):
            variant = f.name[len(t["check"]) + 1:-len("_summary.csv")]
            df = pd.read_csv(f)
            m = _meta(t["check"], variant)
            stale = "" if (m and m.get("code") == code) else "*"
            has_part = "part" in df
            rows.append(dict(variant=variant + stale,
                             train=_value(df, t, "train") if has_part else np.nan,
                             val=_value(df, t, "val") if has_part else _value(df, t, None)))
        if not rows:
            continue
        r = pd.DataFrame(rows)
        sign = 1 if t["better"] == "lower" else -1
        key = r.train if r.train.notna().any() else r.val
        best = r.loc[(sign * key).idxmin()]
        ok = sign * best.val <= sign * t["value"]
        allv = ", ".join(f"{v.variant} {v.val:.3g}" for v in r.itertuples())
        lines.append(f"| {t['label']} | {t['value']} | {best.variant} | {best.train:.3g} | "
                     f"{best.val:.3g} | {'yes' if ok else 'no'} | {allv} |")
    txt = "\n".join(lines)
    txt += ("\n\n`*` = result made by an older code version (re-run with `run`). "
            "Sources: " + "; ".join(sorted({t['source'] for t in TARGETS})) + ".\n")
    (OUT / "scoreboard.md").write_text(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("specs", nargs="*")
    r.add_argument("--all", action="store_true")
    r.add_argument("--force", action="store_true")
    r.add_argument("--workers", type=int, default=8)
    sub.add_parser("board")
    a = ap.parse_args()
    if a.cmd == "run":
        specs = a.specs or []
        if a.all:
            specs = [f"{c}:{v}" for c, vs in VARIANTS.items() for v in vs]
        run(specs, a.force, a.workers)
        board()
    else:
        board()


if __name__ == "__main__":
    main()
