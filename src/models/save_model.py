from __future__ import annotations

import joblib


def save_model(model, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
