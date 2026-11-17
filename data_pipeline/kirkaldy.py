"""Kirkaldy et al. (2024) LG M50T ageing dataset loader.

Source: Zenodo 10.5281/zenodo.10637534, five experiment zips in
``data/kirkaldy/``. Only the small ``Summary Data`` folders are needed and
are extracted to ``data/kirkaldy/extracted/`` by ``extract()``.

Checked against the real files 17 Nov 2026:

* The SoH signal is the C/10 discharge capacity measured at each
  Reference Performance Test (RPT), column ``C/10 Capacity [mA h]`` in
  ``Summary Data/Performance Summary/Expt N - cell X (TdegC) - Processed
  Data.csv``: 10-16 RPTs per cell, 40 cells total.
* The per-cycle ``Discharge Capacity`` in ``Summary per Cycle`` is NOT a
  SoH measure: most experiments cycle a partial SoC window (e.g. 0.73 Ah
  per cycle on a 4.9 Ah cell in Expt 3, 85-100% SoC).
* ``Ageing Cycles`` is not comparable across experiments (ageing sets
  were scaled to the SoC-window size), so x is equivalent full cycles,
  EFC = Charge Throughput / (2 * 4.865 Ah). ``Charge Throughput`` counts
  charge plus discharge: for Expt 3 cell A, summing per-cycle charge +
  discharge capacity over the first set gives 766 Ah against the
  summary's 760 Ah.
* Nominal capacity 4.865 Ah = mean BoL C/10 capacity (Kirkaldy et al.
  2024 Sec. 2.3). EOL threshold here = 80% of that (3.892 Ah), mirroring
  Severson's "80% of nominal". The study is fixed-duration, so only some
  cells reach it.

The ``batteryDAT`` package suggested in the plan was not needed: the
summary files are plain CSV.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.common import Dataset, first_crossing

REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "kirkaldy"
EXTRACT_DIR = RAW_DIR / "extracted"

NOMINAL_CAPACITY_AH = 4.865
EOL_FRACTION = 0.8

# SoC window / cycling type per experiment, Kirkaldy et al. 2024 Sec. 2.3.
EXPERIMENTS = {
    "1": "0-30% SoC (Si-based degradation)",
    "2,2": "70-85% SoC (C-based degradation, repeat)",
    "3": "85-100% SoC (cathode degradation / Li plating)",
    "4": "0-100% SoC, WLTP drive-cycle discharge (control)",
    "5": "0-100% SoC, 1C CC discharge (control)",
}

_NAME = re.compile(r"Expt ([\d,]+) - cell (\w) \((\d+)degC\)")


def extract() -> None:
    """Unzip only the Summary Data and Settings folders (~MBs, vs ~160 GB
    for the full raw data)."""
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    for z in sorted(RAW_DIR.glob("*.zip")):
        subprocess.run(["unzip", "-oq", str(z), "*/Summary Data/*",
                        "*/Settings/*", "-d", str(EXTRACT_DIR)], check=False)


def load() -> Dataset:
    files = sorted(EXTRACT_DIR.glob("*/Summary Data/Performance Summary/*.csv"))
    if not files:
        raise FileNotFoundError(f"No summary CSVs under {EXTRACT_DIR}; run extract()")
    cell_rows, curve_parts = [], []
    eol_ah = EOL_FRACTION * NOMINAL_CAPACITY_AH
    for f in files:
        m = _NAME.search(f.name)
        expt, letter, temp = m.group(1), m.group(2), int(m.group(3))
        d = pd.read_csv(f)
        d = d[d["C/10 Capacity [mA h]"].notna()]
        x = d["Charge Throughput [A h]"].to_numpy(float) / (2 * NOMINAL_CAPACITY_AH)
        q = d["C/10 Capacity [mA h]"].to_numpy(float) / 1000.0
        cid = f"E{expt.replace(',', '.')}_{letter}"
        cell_rows.append(dict(
            cell_id=cid, batch=f"Expt {expt}", experiment=expt, temperature_c=temp,
            group=f"Expt{expt}_{temp}C", condition=EXPERIMENTS[expt],
            n_points=len(d), eol_x=first_crossing(x, q, eol_ah),
            included=True, exclude_reason=""))
        curve_parts.append(pd.DataFrame({
            "cell_id": cid, "x": x, "capacity_ah": q,
            "ageing_cycles": d["Ageing Cycles"].to_numpy(),
            "days": d["Days of degradation"].to_numpy()}))
    cells = pd.DataFrame(cell_rows)
    curves = pd.concat(curve_parts, ignore_index=True)
    curves["soh"] = curves["capacity_ah"] / NOMINAL_CAPACITY_AH
    return Dataset("kirkaldy", cells, curves, "efc", NOMINAL_CAPACITY_AH,
                   EOL_FRACTION)
