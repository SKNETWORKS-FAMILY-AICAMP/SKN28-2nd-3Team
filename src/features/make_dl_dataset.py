from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.io import save_csv


def make_dl_ready_table(df: pd.DataFrame, output_path) -> pd.DataFrame:
    out = df.copy()
    target = None
    if "churn_flag" in out.columns:
        target = out["churn_flag"].copy()
        out = out.drop(columns=["churn_flag"])

    numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
    scaler = StandardScaler()
    out[numeric_cols] = scaler.fit_transform(out[numeric_cols])

    if target is not None:
        out["churn_flag"] = target.values

    save_csv(out, output_path)
    return out
