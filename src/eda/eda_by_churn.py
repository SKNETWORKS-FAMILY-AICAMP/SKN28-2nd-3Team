# eda_by_churn.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import save_csv
from src.utils.plot_utils import save_figure


def _safe_numeric_columns(df: pd.DataFrame, target_col: str = "churn_flag") -> list[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [col for col in numeric_cols if col != target_col]


def _make_group_mean_table(df: pd.DataFrame, target_col: str = "churn_flag") -> pd.DataFrame:
    rows = []
    for col in _safe_numeric_columns(df, target_col=target_col):
        non_churn_mean = df.loc[df[target_col] == 0, col].mean()
        churn_mean = df.loc[df[target_col] == 1, col].mean()
        rows.append({
            "feature": col,
            "non_churn_mean": non_churn_mean,
            "churn_mean": churn_mean,
            "diff_churn_minus_nonchurn": churn_mean - non_churn_mean,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("diff_churn_minus_nonchurn", ascending=False).reset_index(drop=True)
    return result


def _plot_mean_comparison(df: pd.DataFrame, feature: str, output_path: Path, target_col: str = "churn_flag") -> None:
    group_means = df.groupby(target_col, dropna=False)[feature].mean().rename(index={0: "Non-Churn", 1: "Churn"})
    plt.figure(figsize=(8, 5))
    plt.bar(group_means.index.astype(str), group_means.values)
    plt.title(f"{feature} 평균 비교")
    plt.ylabel(feature)
    plt.xlabel("Churn Group")
    plt.tight_layout()
    save_figure(output_path)
    plt.close()


def run_churn_comparison_eda(
    df: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
    target_col: str = "churn_flag",
) -> None:
    if target_col not in df.columns:
        raise KeyError(f"'{target_col}' 컬럼이 데이터프레임에 없습니다.")
    group_mean_df = _make_group_mean_table(df, target_col=target_col)
    save_csv(group_mean_df, tables_dir / "group_mean_by_churn.csv")

    preferred = [
        ("total_usage_count", "bar_mean_by_churn_usage.png"),
        ("error_rate", "bar_mean_by_churn_error_rate.png"),
        ("avg_satisfaction_score", "bar_mean_by_churn_satisfaction.png"),
        ("health_score", "bar_mean_by_churn_health_score.png"),
    ]
    for feature, filename in preferred:
        if feature in df.columns:
            _plot_mean_comparison(df, feature, plots_dir / filename, target_col=target_col)
