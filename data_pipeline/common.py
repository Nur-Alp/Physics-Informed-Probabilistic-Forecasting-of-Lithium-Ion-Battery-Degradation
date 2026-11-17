"""Shared table format so both datasets plug into one downstream pipeline.

Every loader returns a ``Dataset``:

cells  : one row per cell. Required columns
         cell_id, group (cycling protocol / condition, used for grouped
         splitting), batch, included (bool), exclude_reason, eol_x
         (x at which capacity first drops below eol_capacity_ah, NaN if
         never observed).
curves : long table, one row per capacity measurement. Required columns
         cell_id, x (cycle for Severson, equivalent full cycles for
         Kirkaldy), capacity_ah, soh (= capacity_ah / nominal_ah).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    name: str
    cells: pd.DataFrame
    curves: pd.DataFrame
    x_name: str
    nominal_ah: float
    eol_fraction: float = 0.8

    @property
    def eol_capacity_ah(self) -> float:
        return self.eol_fraction * self.nominal_ah

    def curve(self, cell_id: str) -> tuple[np.ndarray, np.ndarray]:
        g = self.curves[self.curves.cell_id == cell_id].sort_values("x")
        return g["x"].to_numpy(float), g["capacity_ah"].to_numpy(float)


def first_crossing(x: np.ndarray, q: np.ndarray, threshold: float) -> float:
    """x where q first drops below threshold, linearly interpolated
    between the bracketing points. NaN if it never does."""
    below = np.flatnonzero(q < threshold)
    if below.size == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return float(x[0])
    x0, x1, q0, q1 = x[i - 1], x[i], q[i - 1], q[i]
    return float(x0 + (q0 - threshold) * (x1 - x0) / (q0 - q1))
