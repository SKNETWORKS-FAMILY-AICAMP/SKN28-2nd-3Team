from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config.paths import PROCESSED_DIR, MODELS_OUTPUT_DIR
from src.config.settings import TARGET_COL, RANDOM_STATE
from src.models.evaluate import evaluate_binary_classifier
from src.models.threshold_tuning import tune_threshold
from src.models.save_model import save_model
from src.utils.io import read_csv, save_csv
from src.utils.plot_utils import apply_plot_style
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    train = read_csv(PROCESSED_DIR / "train.csv")
    valid = read_csv(PROCESSED_DIR / "valid.csv")

    X_train = train.drop(columns=[TARGET_COL, "account_id"], errors="ignore")
    y_train = train[TARGET_COL]
    X_valid = valid.drop(columns=[TARGET_COL, "account_id"], errors="ignore")
    y_valid = valid[TARGET_COL]

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    valid_proba = model.predict_proba(X_valid)[:, 1]
    best_threshold = tune_threshold(y_valid, valid_proba, MODELS_OUTPUT_DIR / "threshold_metrics.csv")
    valid_pred = (valid_proba >= best_threshold).astype(int)

    result = evaluate_binary_classifier(y_valid, valid_pred, valid_proba)
    current = pd.DataFrame([{"model": "random_forest", "best_threshold": best_threshold, **result.metrics}])

    baseline_path = MODELS_OUTPUT_DIR / "baseline_metrics.csv"
    comparison = pd.concat([read_csv(baseline_path), current], ignore_index=True) if baseline_path.exists() else current
    save_csv(comparison, MODELS_OUTPUT_DIR / "model_comparison.csv")
    save_csv(result.confusion_matrix_df.reset_index(), MODELS_OUTPUT_DIR / "confusion_matrix_tree.csv")
    save_csv(result.roc_curve_df, MODELS_OUTPUT_DIR / "roc_curve_points_tree.csv")
    save_csv(result.pr_curve_df, MODELS_OUTPUT_DIR / "pr_curve_points_tree.csv")

    importances = pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    save_csv(importances, MODELS_OUTPUT_DIR / "feature_importance.csv")
    save_model(model, MODELS_OUTPUT_DIR / "best_model.pkl")

    apply_plot_style()
    plt.figure(figsize=(10, 6))
    plt.barh(importances.head(20).sort_values("importance")["feature"], importances.head(20).sort_values("importance")["importance"])
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(MODELS_OUTPUT_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close()

    logger.info("saved tree model outputs")


if __name__ == "__main__":
    main()
