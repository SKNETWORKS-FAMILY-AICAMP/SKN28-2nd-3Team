# eda_numeric.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config.settings import KEY_NUMERIC_FEATURES
from src.utils.io import save_csv
from src.utils.plot_utils import save_figure


def _numeric_columns(df: pd.DataFrame, target_col: str = "churn_flag") -> list[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in numeric_cols if c not in {target_col}]


def run_numeric_eda(
    df: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
    target_col: str = "churn_flag",
) -> None:
    numeric_cols = _numeric_columns(df, target_col)
    numeric_summary = df[numeric_cols].describe().T.reset_index().rename(columns={"index": "feature"})
    save_csv(numeric_summary, tables_dir / "numeric_summary.csv")

    skewness = df[numeric_cols].skew(numeric_only=True).rename("skewness").reset_index().rename(columns={"index": "feature"})
    save_csv(skewness, tables_dir / "skewness_summary.csv")

    corr_cols = [c for c in KEY_NUMERIC_FEATURES if c in df.columns]
    if corr_cols:
        corr = df[corr_cols].corr(numeric_only=True)
        save_csv(corr.reset_index().rename(columns={"index": "feature"}), tables_dir / "correlation_matrix_key_features.csv")
        plt.figure(figsize=(12, 8))
        plt.imshow(corr, aspect="auto")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation Heatmap (Key Features)")
        plt.colorbar()
        plt.tight_layout()
        save_figure(plots_dir / "correlation_heatmap_key_features.png")
        plt.close()

    if target_col in df.columns:
        corr_target = (
            df[numeric_cols + [target_col]].corr(numeric_only=True)[target_col]
            .drop(target_col)
            .sort_values(ascending=False)
            .rename("correlation_with_churn")
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        save_csv(corr_target, tables_dir / "correlation_with_churn.csv")
        save_csv(corr_target.rename(columns={"correlation_with_churn": "score"}), tables_dir / "feature_importance_precheck.csv")

    hist_targets = [
        ("account_age_days", "hist_account_age_days.png"),
        ("total_subscriptions", "hist_total_subscriptions.png"),
        ("avg_mrr_amount", "hist_avg_mrr_amount.png"),
        ("days_since_last_usage", "hist_days_since_last_usage.png"),
        ("health_score", "hist_health_score.png"),
    ]
    for feature, filename in hist_targets:
        if feature not in df.columns:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(df[feature].dropna(), bins=30)
        plt.title(f"{feature} Distribution")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.tight_layout()
        save_figure(plots_dir / filename)
        plt.close()
