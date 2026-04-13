from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.plot_utils import apply_plot_style


def save_histograms(df: pd.DataFrame, features: list[str], output_dir) -> None:
    apply_plot_style()
    for col in features:
        plt.figure()
        df[col].dropna().hist(bins=30)
        plt.title(f"{col} distribution")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_{col}.png")
        plt.close()


def save_bar_means(df: pd.DataFrame, features: list[str], target_col: str, output_dir) -> None:
    apply_plot_style()
    grouped = df.groupby(target_col)[features].mean()
    for col in features:
        plt.figure()
        grouped[col].plot(kind="bar")
        plt.title(f"mean {col} by churn")
        plt.xlabel(target_col)
        plt.ylabel("mean")
        plt.tight_layout()
        plt.savefig(output_dir / f"bar_mean_by_churn_{col}.png")
        plt.close()


def save_target_distribution(df: pd.DataFrame, target_col: str, output_dir) -> None:
    apply_plot_style()
    plt.figure()
    df[target_col].value_counts().sort_index().plot(kind="bar")
    plt.title("target distribution overall")
    plt.xlabel(target_col)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution_overall.png")
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame, features: list[str], target_col: str, output_dir) -> None:
    import numpy as np
    apply_plot_style()
    corr = df[features + [target_col]].corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap_key_features.png")
    plt.close()

    missing_matrix = df[features].isna().astype(int)
    plt.figure(figsize=(12, 6))
    plt.imshow(missing_matrix.T, aspect="auto")
    plt.yticks(range(len(features)), features)
    plt.tight_layout()
    plt.savefig(output_dir / "missing_pattern_heatmap.png")
    plt.close()
