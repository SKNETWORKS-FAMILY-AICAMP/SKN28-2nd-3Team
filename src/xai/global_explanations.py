# global_explanations.py
from __future__ import annotations

import pandas as pd

from src.utils.io import save_csv


def save_global_shap_summary(mean_abs_shap, feature_names, output_path) -> pd.DataFrame:
    summary = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    save_csv(summary, output_path)
    return summary
