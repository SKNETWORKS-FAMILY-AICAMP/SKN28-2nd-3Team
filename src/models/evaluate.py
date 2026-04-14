# evaluate.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, roc_curve
)


@dataclass
class EvaluationResult:
    metrics: dict
    confusion_matrix_df: pd.DataFrame
    roc_curve_df: pd.DataFrame
    pr_curve_df: pd.DataFrame


def evaluate_binary_classifier(y_true, y_pred, y_proba) -> EvaluationResult:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"])
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": list(roc_thresholds) + [None]*(len(fpr)-len(roc_thresholds))})
    pr_df = pd.DataFrame({"precision": precision, "recall": recall})
    return EvaluationResult(metrics=metrics, confusion_matrix_df=cm_df, roc_curve_df=roc_df, pr_curve_df=pr_df)
