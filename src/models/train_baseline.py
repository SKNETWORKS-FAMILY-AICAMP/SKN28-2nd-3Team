from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from src.config.paths import PROCESSED_DIR, MODELS_OUTPUT_DIR
from src.config.settings import TARGET_COL, RANDOM_STATE
from src.models.evaluate import evaluate_binary_classifier
from src.models.threshold_tuning import tune_threshold
from src.models.save_model import save_model
from src.utils.io import read_csv, save_csv
from src.utils.plot_utils import apply_plot_style
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _plot_confusion(cm_df: pd.DataFrame, output_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm_df.values, aspect="auto")
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            plt.text(j, i, int(cm_df.iloc[i, j]), ha="center", va="center")
    plt.xticks(range(cm_df.shape[1]), cm_df.columns)
    plt.yticks(range(cm_df.shape[0]), cm_df.index)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    train = read_csv(PROCESSED_DIR / "train.csv")
    valid = read_csv(PROCESSED_DIR / "valid.csv")
    X_train = train.drop(columns=[TARGET_COL, "account_id"], errors="ignore")
    y_train = train[TARGET_COL]
    X_valid = valid.drop(columns=[TARGET_COL, "account_id"], errors="ignore")
    y_valid = valid[TARGET_COL]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), X_train.columns.tolist())
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    model.fit(X_train, y_train)
    valid_proba = model.predict_proba(X_valid)[:, 1]
    best_threshold = tune_threshold(y_valid, valid_proba, MODELS_OUTPUT_DIR / "threshold_metrics.csv")
    valid_pred = (valid_proba >= best_threshold).astype(int)
    result = evaluate_binary_classifier(y_valid, valid_pred, valid_proba)

    metrics_df = pd.DataFrame([{"model": "logistic_regression", "best_threshold": best_threshold, **result.metrics}])
    save_csv(metrics_df, MODELS_OUTPUT_DIR / "baseline_metrics.csv")
    save_csv(result.confusion_matrix_df.reset_index(), MODELS_OUTPUT_DIR / "confusion_matrix.csv")
    save_csv(result.roc_curve_df, MODELS_OUTPUT_DIR / "roc_curve_points.csv")
    save_csv(result.pr_curve_df, MODELS_OUTPUT_DIR / "pr_curve_points.csv")
    save_model(model, MODELS_OUTPUT_DIR / "baseline_model.pkl")

    apply_plot_style()
    _plot_confusion(result.confusion_matrix_df, MODELS_OUTPUT_DIR / "confusion_matrix.png")

    plt.figure()
    plt.plot(result.roc_curve_df["fpr"], result.roc_curve_df["tpr"])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(MODELS_OUTPUT_DIR / "roc_curve.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(result.pr_curve_df["recall"], result.pr_curve_df["precision"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.tight_layout()
    plt.savefig(MODELS_OUTPUT_DIR / "pr_curve.png", bbox_inches="tight")
    plt.close()

    logger.info("saved baseline model outputs")


if __name__ == "__main__":
    main()
