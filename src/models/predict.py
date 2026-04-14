# predict.py
from __future__ import annotations

import pandas as pd
import joblib


def load_model(path):
    return joblib.load(path)


def predict_proba(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    decision = model.decision_function(X)
    return decision
