from __future__ import annotations

import pandas as pd

from src.utils.io import save_csv


def save_local_explanations(shap_values, X: pd.DataFrame, output_prefix) -> None:
    sample_idx = [0, min(1, len(X)-1)]
    for i, idx in enumerate(sample_idx, start=1):
        local = pd.DataFrame({
            "feature": X.columns,
            "feature_value": X.iloc[idx].values,
            "shap_value": shap_values[idx],
        }).sort_values("shap_value", key=lambda s: s.abs(), ascending=False)
        save_csv(local, output_prefix.parent / f"local_explanation_sample_{i}.csv")
