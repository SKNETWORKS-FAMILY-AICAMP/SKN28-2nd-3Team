# eda_missingness.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import save_csv
from src.utils.plot_utils import save_figure


def run_missingness_eda(
    df: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
    target_col: str = "churn_flag",
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    missing_counts = df.isna().sum().sort_values(ascending=False).rename("missing_count").reset_index()
    missing_counts.columns = ["column", "missing_count"]
    save_csv(missing_counts, tables_dir / "missing_counts.csv")

    missing_ratio = (df.isna().mean() * 100).round(2).rename("missing_ratio_pct").reset_index()
    missing_ratio.columns = ["column", "missing_ratio_pct"]
    save_csv(missing_ratio, tables_dir / "missing_ratio.csv")

    if target_col not in df.columns:
        raise KeyError(f"'{target_col}' 컬럼이 데이터프레임에 없습니다.")

    records: list[pd.DataFrame] = []
    for col in df.columns:
        if col == target_col:
            continue
        tmp = pd.DataFrame({
            "column": col,
            "missing_flag": df[col].isna().astype(int),
            target_col: df[target_col],
        })
        summary = (
            tmp.groupby(["column", "missing_flag"], dropna=False)[target_col]
            .agg(customer_count="count", churn_rate="mean")
            .reset_index()
        )
        records.append(summary)

    missing_vs_churn = pd.concat(records, ignore_index=True) if records else pd.DataFrame(
        columns=["column", "missing_flag", "customer_count", "churn_rate"]
    )
    save_csv(missing_vs_churn, tables_dir / "missing_vs_churn.csv")

    heatmap_df = df.isna().astype(int)
    if heatmap_df.shape[1] > 60:
        heatmap_df = heatmap_df.iloc[:, :60]
    plt.figure(figsize=(12, 6))
    plt.imshow(heatmap_df.T, aspect="auto")
    plt.yticks(range(len(heatmap_df.columns)), heatmap_df.columns)
    plt.xticks([])
    plt.title("Missing Pattern Heatmap")
    plt.tight_layout()
    save_figure(plots_dir / "missing_pattern_heatmap.png")
    plt.close()
