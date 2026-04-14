# encode_categoricals.py
from __future__ import annotations

import pandas as pd


def one_hot_encode(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    existing = [c for c in categorical_cols if c in df.columns]
    return pd.get_dummies(df, columns=existing, dummy_na=False, drop_first=False)
