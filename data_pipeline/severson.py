"""Severson et al. (2019) fast-charge dataset loader.

Source: data.matr.io "BEEP structured data" zip (FastCharge.zip). Checked
17 Nov 2026 against the real file: the zip holds 140 BEEP
``ProcessedCyclerRun`` JSON files (``FastCharge_<id>_CH<ch>_structure.json``),
NOT MATLAB structs. Each file has a per-cycle ``summary`` dict
(cycle_index, discharge_capacity, ...) and a ``protocol`` string whose
prefix gives the batch date (2017-05-12 / 2017-06-30 / 2018-04-12).

Parsing the full JSON is slow (~200 MB each, 28 GB total), so
``build_cache`` reads each file straight from the zip once and keeps only
the per-cycle summary, written to ``data/severson/cache/``. ``load`` then
reads the small cache.

Usage:
    python -m data_pipeline.severson          # build cache (slow, once)
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.common import Dataset

REPO = Path(__file__).resolve().parents[1]
ZIP_PATH = REPO / "data" / "severson" / "FastCharge.zip"
CACHE_DIR = REPO / "data" / "severson" / "cache"
CYCLES_PATH = CACHE_DIR / "severson_cycles.parquet"
CELLS_PATH = CACHE_DIR / "severson_cells.csv"

NOMINAL_CAPACITY_AH = 1.1  # A123 APR18650M1A, Severson et al. 2019
EOL_FRACTION = 0.8         # EOL defined as 80% of nominal (0.88 Ah)
# The 2018-04-12 batch stops a few mAh ABOVE 0.88 Ah in BEEP's recomputed
# capacity (final values 0.8808-0.8843 Ah): the cycler's own stop rule
# fired first. A test ending within this tolerance of 0.88 Ah counts as
# reaching EOL at its last cycle.
STOP_TOLERANCE_AH = 0.005

SUMMARY_FIELDS = [
    "cycle_index", "discharge_capacity", "charge_capacity",
    "dc_internal_resistance", "temperature_average", "temperature_maximum",
    "charge_throughput", "date_time_iso",
]


def _parse_one(name: str) -> tuple[dict, pd.DataFrame]:
    with zipfile.ZipFile(ZIP_PATH) as z:
        d = json.loads(z.read(name))
    s = d["summary"]
    df = pd.DataFrame({k: s[k] for k in SUMMARY_FIELDS if k in s})
    file_id = name.removesuffix("_structure.json")
    df.insert(0, "file_id", file_id)
    protocol = d.get("protocol") or ""
    m = re.match(r"(\d{4}-\d{2}-\d{2})", protocol)
    meta = {
        "file_id": file_id,
        "barcode": d.get("barcode"),
        "channel_id": d.get("channel_id"),
        "protocol": protocol,
        "batch_date": m.group(1) if m else None,
        "n_cycles": len(df),
    }
    return meta, df


def build_cache(workers: int = 4) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = sorted(n for n in z.namelist() if n.endswith("_structure.json"))
    metas, frames = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (meta, df) in enumerate(ex.map(_parse_one, names), 1):
            metas.append(meta)
            frames.append(df)
            print(f"[{i}/{len(names)}] {meta['file_id']} {meta['batch_date']} "
                  f"{meta['n_cycles']} cycles", flush=True)
    pd.concat(frames, ignore_index=True).to_parquet(CYCLES_PATH)
    pd.DataFrame(metas).to_csv(CELLS_PATH, index=False)


def load_raw(clean: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-FILE tables straight from the cache (140 files, before
    merging continuation files). Mostly for inspection; use ``load``."""
    cells = pd.read_csv(CELLS_PATH).rename(columns={"file_id": "cell_id"})
    cyc = pd.read_parquet(CYCLES_PATH).rename(
        columns={"file_id": "cell_id", "cycle_index": "cycle",
                 "discharge_capacity": "capacity_ah"})
    if clean:
        cyc = _clean(cyc)
    return cells, cyc.reset_index(drop=True)


def normalise_protocol(protocol: str) -> str:
    """'2017-05-12_TESTS\\20170512-4_8C-80PER_4_8C.SDU' and
    '2018-04-12_batch8\\20180412-4_8C_80per_4_8C_newstructure.sdu' both
    -> '4.8C_80per_4.8C', so identical charging protocols group together
    across batches."""
    p = protocol.split("\\")[-1].lower()
    p = re.sub(r"\.sdu$", "", p)
    p = re.sub(r"_newstructure$", "", p)
    p = re.sub(r"^\d{8}-", "", p)
    p = p.replace("-", "_")
    p = re.sub(r"(\d)_(\d)", r"\1.\2", p)
    return p


def load() -> Dataset:
    """Severson dataset in the shared ``Dataset`` format, one row per
    physical cell (124-ish after inclusion rules, see below).

    Processing, each step checked against the real files 17 Nov 2026:

    1. Continuation files merged by barcode. Five 2017-05-12 cells
       (channels 0,1,2,4,5) were cycled in two runs and appear as two
       files with the same barcode. The later run (lower starting
       capacity) is appended after the earlier one, cycle index offset
       by the earlier run's last index. This reproduces the published
       cycle lives exactly where checked (merged max 2237/2238 = the
       2,237-cycle maximum quoted by Geslin et al. 2023).
    2. Cycle 0 dropped and logging glitches removed (``_clean``).
    3. EOL = first cycle with capacity < 0.88 Ah (80% of 1.1 Ah nominal,
       Severson et al. 2019), or the last cycle if the test stopped
       within STOP_TOLERANCE_AH of 0.88 Ah (``eol_rule`` records which).
       Curves truncated at EOL: data past it is
       outside the defined lifetime (the 2017-06-30 batch kept cycling to
       ~0.83 Ah, the other two stopped at ~0.88).
    4. Inclusion: cell must have an observed EOL. Excluded cells keep a
       row in ``cells`` with ``exclude_reason`` so the count is auditable.
    """
    files, cyc = load_raw(clean=True)
    files["first_time"] = files["cell_id"].map(
        cyc.groupby("cell_id")["date_time_iso"].min())

    cell_rows, curve_parts = [], []
    for barcode, fg in files.groupby(files["barcode"].str.lower(), sort=False):
        fg = fg.sort_values("first_time", na_position="last")
        parts, offset = [], 0
        for fid in fg["cell_id"]:
            c = cyc[cyc.cell_id == fid].sort_values("cycle")
            if c.empty:
                continue
            c = c.assign(cycle=c["cycle"] + offset)
            offset = int(c["cycle"].max())
            parts.append(c)
        first = fg.iloc[0]
        cid = first["cell_id"] if len(fg) == 1 else "+".join(fg["cell_id"])
        row = dict(cell_id=cid, barcode=barcode, channel_id=first["channel_id"],
                   batch=first["batch_date"], protocol_raw=first["protocol"],
                   group=normalise_protocol(first["protocol"]),
                   n_files=len(fg), included=True, exclude_reason="")
        if not parts:
            row.update(included=False, exclude_reason="no cycles after cleaning",
                       eol_x=np.nan, n_points=0)
            cell_rows.append(row)
            continue
        c = pd.concat(parts)
        x = c["cycle"].to_numpy(float)
        q = c["capacity_ah"].to_numpy(float)
        eol_ah = EOL_FRACTION * NOMINAL_CAPACITY_AH
        below = np.flatnonzero(q < eol_ah)
        if below.size:
            eol, rule = float(x[below[0]]), "first cycle < 0.88 Ah"
        elif np.median(q[-5:]) < eol_ah + STOP_TOLERANCE_AH:
            eol, rule = float(x[-1]), "test stopped at 80% (last cycle)"
        else:
            eol, rule = np.nan, ""
        if np.isnan(eol):
            row.update(included=False, exclude_reason="EOL (0.88 Ah) never reached")
        else:
            c = c[c["cycle"] <= eol]
        row.update(eol_x=eol, eol_rule=rule, n_points=len(c))
        cell_rows.append(row)
        curve_parts.append(pd.DataFrame({
            "cell_id": cid, "x": c["cycle"].to_numpy(float),
            "capacity_ah": c["capacity_ah"].to_numpy(float)}))

    cells = pd.DataFrame(cell_rows)
    curves = pd.concat(curve_parts, ignore_index=True)
    curves["soh"] = curves["capacity_ah"] / NOMINAL_CAPACITY_AH
    return Dataset("severson", cells, curves, "cycle", NOMINAL_CAPACITY_AH,
                   EOL_FRACTION)


def _clean(cyc: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cid, g in cyc.groupby("cell_id", sort=False):
        g = g.sort_values("cycle")
        g = g[g["cycle"] >= 1]
        # Physically bounded: LFP cell rated 1.1 Ah; capacity outside
        # (0.5, 1.3) Ah is a logging glitch, not degradation.
        g = g[(g["capacity_ah"] > 0.5) & (g["capacity_ah"] < 1.3)]
        # Single-cycle spikes: deviation from a 5-cycle rolling median.
        med = g["capacity_ah"].rolling(5, center=True, min_periods=1).median()
        g = g[(g["capacity_ah"] - med).abs() < 0.03]
        out.append(g)
    return pd.concat(out)


if __name__ == "__main__":
    build_cache(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
