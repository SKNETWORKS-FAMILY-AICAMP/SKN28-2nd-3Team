from __future__ import annotations

import pandas as pd

from src.utils.io import save_csv


def build_reason_mapping_report(global_summary: pd.DataFrame, analysis_table: pd.DataFrame, output_path) -> None:
    reason_counts = analysis_table["latest_reason_code"].fillna("unknown").value_counts().rename_axis("reason_code").reset_index(name="count")
    top_features = global_summary.head(10).copy()
    top_features["note"] = "모델이 중요하게 본 이탈 전 신호"
    reason_counts["note"] = "실제 churn_events 기반 사후 사유 요약"
    combined = pd.concat([
        top_features.rename(columns={"feature": "item", "mean_abs_shap": "score"})[["item", "score", "note"]],
        reason_counts.rename(columns={"reason_code": "item", "count": "score"})[["item", "score", "note"]],
    ], ignore_index=True)
    save_csv(combined, output_path)
