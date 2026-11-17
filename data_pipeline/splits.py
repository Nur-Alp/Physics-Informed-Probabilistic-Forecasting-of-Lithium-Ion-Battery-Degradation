"""Leakage-free, locked train / validation / test splits.

Design (read Geslin et al. 2023 and Cerqueira et al. 2020 first):

* Whole cells only. A cell is in exactly one of train / val / test; no
  cycles of a test cell are ever used to fit anything except that cell's
  own forecast, and then only its first N cycles (``forecast_origins``).
* Grouped by cycling protocol / condition. Geslin et al. show that when
  identically-cycled cells sit on both sides of a split, models can score
  well by recognising the protocol instead of the cell. Train and val
  therefore never share a protocol (Severson) or condition (Kirkaldy).
* Severson test = the 2018-04-12 batch, i.e. Severson et al.'s own
  "secondary test" set, generated after their model development: the
  closest thing to a temporally out-of-sample check (Cerqueira et al.:
  for non-stationary series only order-preserving holdout is reliable).
  Known caveat: one protocol (4.8C_80per_4.8C) appears in both the dev
  batches and the test batch (5 dev cells, 8 test cells). Kept, because
  the batch-level split is the published benchmark and the baselines use
  no protocol information; flagged here so later protocol-aware models
  account for it.
* Kirkaldy: 15 conditions (5 experiments x 3 temperatures). Each split
  gets one temperature per experiment, in a Latin-square rotation, so
  every SoC window and every temperature appears in every split but no
  condition is shared.

Locking: ``build()`` writes ``splits/<dataset>.json`` with a SHA-256 of
the test-cell list. ``get()`` re-checks the hash on every load and only
returns the test cells when called with ``final_evaluation=True``. To
change a split you have to delete the JSON on purpose.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from data_pipeline.common import Dataset

SPLIT_DIR = Path(__file__).resolve().parent / "splits"

SEVERSON_TEST_BATCH = "2018-04-12"
# (experiment, temperature) -> split. Latin square over 5 expts x 3 temps.
KIRKALDY_ASSIGNMENT = {
    "1": {10: "test", 25: "val", 40: "train"},
    "2,2": {25: "test", 40: "val", 10: "train"},
    "3": {40: "test", 10: "val", 25: "train"},
    "4": {10: "test", 25: "val", 40: "train"},
    "5": {25: "test", 40: "val", 10: "train"},
}


def _hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _stable_key(s: str) -> str:
    # Deterministic, library-version-independent ordering of groups.
    return hashlib.sha256(s.encode()).hexdigest()


def _severson(ds: Dataset) -> dict[str, list[str]]:
    c = ds.cells[ds.cells.included]
    test = c[c.batch == SEVERSON_TEST_BATCH]
    dev = c[c.batch != SEVERSON_TEST_BATCH]
    # Greedy grouped halving: walk protocols in hashed order, add each to
    # whichever of train/val currently has fewer cells.
    train, val = [], []
    for g in sorted(dev.group.unique(), key=_stable_key):
        ids = dev[dev.group == g].cell_id.tolist()
        (train if len(train) <= len(val) else val).extend(ids)
    return {"train": train, "val": val, "test": test.cell_id.tolist()}


def _kirkaldy(ds: Dataset) -> dict[str, list[str]]:
    out = {"train": [], "val": [], "test": []}
    for _, r in ds.cells[ds.cells.included].iterrows():
        out[KIRKALDY_ASSIGNMENT[r.experiment][r.temperature_c]].append(r.cell_id)
    return out


def build(ds: Dataset) -> Path:
    path = SPLIT_DIR / f"{ds.name}.json"
    if path.exists():
        raise FileExistsError(f"{path} exists and is locked; delete it deliberately "
                              "if the split really must change.")
    split = {"severson": _severson, "kirkaldy": _kirkaldy}[ds.name](ds)
    groups = ds.cells.set_index("cell_id")["group"]
    tr, va = set(groups[split["train"]]), set(groups[split["val"]])
    assert not (tr & va), "train/val share a protocol"
    record = {
        "dataset": ds.name,
        "created": date.today().isoformat(),
        "locked": True,
        "test_sha256": _hash(split["test"]),
        "counts": {k: len(v) for k, v in split.items()},
        "train": sorted(split["train"]),
        "val": sorted(split["val"]),
        "test": sorted(split["test"]),
    }
    SPLIT_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    return path


def get(name: str, final_evaluation: bool = False) -> dict[str, list[str]]:
    rec = json.loads((SPLIT_DIR / f"{name}.json").read_text())
    if _hash(rec["test"]) != rec["test_sha256"]:
        raise RuntimeError(f"{name} test set was edited after locking")
    out = {"train": rec["train"], "val": rec["val"]}
    if final_evaluation:
        out["test"] = rec["test"]
    return out


if __name__ == "__main__":
    from data_pipeline import kirkaldy, severson
    for mod in (severson, kirkaldy):
        ds = mod.load()
        p = build(ds)
        print(p, json.loads(p.read_text())["counts"])
