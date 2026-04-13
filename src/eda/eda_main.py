from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.config.paths import PROCESSED_DIR, EDA_TABLES_DIR, EDA_PLOTS_DIR
from src.eda.eda_missingness import run_missingness_eda
from src.eda.eda_numeric import run_numeric_eda
from src.eda.eda_categoricals import run_categorical_eda
from src.eda.eda_by_churn import run_churn_comparison_eda
from src.utils.plot_utils import apply_plot_style


def ensure_output_dirs() -> None:
    EDA_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    EDA_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data() -> pd.DataFrame:
    file_path = PROCESSED_DIR / "train_table_ml.csv"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {file_path}\n"
            "먼저 make_train_table.py를 실행해서 train_table_ml.csv를 생성하세요."
        )
    return pd.read_csv(file_path)


def build_target_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{
        "dataset": "overall",
        "row_count": len(df),
        "churn_count": int(df["churn_flag"].sum()),
        "churn_rate": float(df["churn_flag"].mean()),
    }]
    return pd.DataFrame(rows)


def main() -> None:
    apply_plot_style()
    ensure_output_dirs()
    df = load_processed_data()

    target_summary = build_target_distribution_summary(df)
    target_summary.to_csv(EDA_TABLES_DIR / "target_distribution_summary.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(6,4))
    counts = df["churn_flag"].value_counts().sort_index()
    plt.bar(["Non-Churn", "Churn"], counts.values)
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR / "target_distribution_overall.png", bbox_inches="tight")
    plt.close()

    print("[1/4] Missingness EDA 실행 중...")
    run_missingness_eda(df, EDA_TABLES_DIR, EDA_PLOTS_DIR)

    print("[2/4] Numeric EDA 실행 중...")
    run_numeric_eda(df, EDA_TABLES_DIR, EDA_PLOTS_DIR)

    print("[3/4] Categorical EDA 실행 중...")
    run_categorical_eda(df, EDA_TABLES_DIR, EDA_PLOTS_DIR)

    print("[4/4] Churn comparison EDA 실행 중...")
    run_churn_comparison_eda(df, EDA_TABLES_DIR, EDA_PLOTS_DIR)

    summary = pd.DataFrame([{
        "row_count": len(df),
        "column_count": df.shape[1],
        "churn_rate": float(df["churn_flag"].mean()),
    }])
    summary.to_csv(EDA_TABLES_DIR / "eda_summary_report.csv", index=False, encoding="utf-8-sig")

    print("EDA 완료")
    print(f"- tables: {EDA_TABLES_DIR}")
    print(f"- plots : {EDA_PLOTS_DIR}")


if __name__ == "__main__":
    main()
