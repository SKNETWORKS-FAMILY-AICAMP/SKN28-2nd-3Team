# threshold_tuning.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from src.utils.io import save_csv


def tune_threshold(y_true, y_proba, output_path) -> float:
    rows = []
    best_threshold = 0.5
    best_f1 = -1
    for threshold in np.arange(0.1, 0.95, 0.05):
        pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        rows.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    save_csv(pd.DataFrame(rows), output_path)
    return float(best_threshold)
