from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import save_csv
from src.utils.plot_utils import save_figure


def summarize_dummy_group(df: pd.DataFrame, prefix: str, target_col: str = "churn_flag") -> pd.DataFrame:
    dummy_cols = [col for col in df.columns if col.startswith(f"{prefix}_")]
    rows: list[dict] = []
    for col in dummy_cols:
        subset = df[df[col] == 1].copy()
        if subset.empty:
            continue
        rows.append({
            "group": col.replace(f"{prefix}_", ""),
            "customer_count": len(subset),
            "churn_rate": subset[target_col].mean(),
        })
    if not rows:
        return pd.DataFrame(columns=["group", "customer_count", "churn_rate"])
    return pd.DataFrame(rows).sort_values(["churn_rate", "customer_count"], ascending=[False, False]).reset_index(drop=True)


def plot_dummy_group_summary(summary_df: pd.DataFrame, title: str, output_path: Path) -> None:
    if summary_df.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df["group"], summary_df["churn_rate"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Churn Rate")
    plt.title(title)
    plt.tight_layout()
    save_figure(output_path)
    plt.close()


def run_categorical_eda(
    df: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
    target_col: str = "churn_flag",
) -> None:
    mapping = {
        "country": "dummy_group_summary_country.csv",
        "industry": "dummy_group_summary_industry.csv",
        "referral_source": "dummy_group_summary_referral.csv",
    }
    for prefix, output_name in mapping.items():
        summary = summarize_dummy_group(df, prefix=prefix, target_col=target_col)
        save_csv(summary, tables_dir / output_name)
        plot_dummy_group_summary(summary, f"{prefix} 그룹별 Churn Rate", plots_dir / output_name.replace('.csv', '.png'))
