# shap_analysis.py
from __future__ import annotations

import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.config.paths import PROCESSED_DIR, MODELS_OUTPUT_DIR, XAI_OUTPUT_DIR
from src.config.settings import TARGET_COL
from src.xai.global_explanations import save_global_shap_summary
from src.xai.local_explanations import save_local_explanations
from src.xai.reason_mapping import build_reason_mapping_report
from src.utils.io import read_csv
from src.utils.plot_utils import apply_plot_style
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    model = joblib.load(MODELS_OUTPUT_DIR / "best_model.pkl")
    train = read_csv(PROCESSED_DIR / "train.csv")
    analysis = read_csv(PROCESSED_DIR / "analysis_table.csv")

    X = train.drop(columns=[TARGET_COL, "account_id"], errors="ignore")
    if len(X) > 200:
        X = X.sample(200, random_state=42)

    apply_plot_style()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_matrix = shap_values[1]
    else:
        shap_matrix = shap_values
    if getattr(shap_matrix, "ndim", 0) == 3:
        shap_matrix = shap_matrix[:, :, 1]

    mean_abs = abs(shap_matrix).mean(axis=0)
    global_summary = save_global_shap_summary(mean_abs, X.columns, XAI_OUTPUT_DIR / "xai_summary_report.csv")
    save_local_explanations(shap_matrix, X.reset_index(drop=True), XAI_OUTPUT_DIR / "local_explanation_sample_0.csv")
    build_reason_mapping_report(global_summary, analysis, XAI_OUTPUT_DIR / "reason_mapping_report.csv")

    shap.summary_plot(shap_matrix, X, show=False)
    plt.tight_layout()
    plt.savefig(XAI_OUTPUT_DIR / "shap_summary.png", bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_matrix, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(XAI_OUTPUT_DIR / "shap_bar.png", bbox_inches="tight")
    plt.close()

    for feature, filename in [("total_usage_count", "shap_dependence_usage.png"), ("error_rate", "shap_dependence_error.png")]:
        if feature in X.columns:
            shap.dependence_plot(feature, shap_matrix, X, show=False)
            plt.tight_layout()
            plt.savefig(XAI_OUTPUT_DIR / filename, bbox_inches="tight")
            plt.close()

    logger.info("saved shap outputs")


if __name__ == "__main__":
    main()
