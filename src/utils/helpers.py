# helpers.py
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd


def to_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def bool_to_int(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(int)


def safe_divide(numerator, denominator, fill_value: float = 0.0):
    numerator = np.asarray(numerator)
    denominator = np.asarray(denominator)
    return np.where(denominator == 0, fill_value, numerator / denominator)


def create_missing_flag(series: pd.Series, sentinel=None) -> pd.Series:
    if sentinel is None:
        return series.isna().astype(int)
    return (series.isna() | (series == sentinel)).astype(int)
