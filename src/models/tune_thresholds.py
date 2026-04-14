# tune_thresholds.py
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def load_xy():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0]
    return X_train, X_test, y_train, y_test


def prepare_ml_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    drop_cols = [c for c in ["account_id", "customer_id", "id"] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype(int)

    # 숫자형만 사용
    X_num = X.select_dtypes(include=["number"]).copy()
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())
    return X_num


def load_dl_artifacts():
    feature_names = pd.read_csv(MODEL_DIR / "dl_feature_columns.csv")["feature"].tolist()
    with open(MODEL_DIR / "dl_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model = MLP(input_dim=len(feature_names))
    model.load_state_dict(torch.load(MODEL_DIR / "dl_model.pth", map_location="cpu"))
    model.eval()
    return model, scaler, feature_names


def get_dl_proba(X_test: pd.DataFrame) -> np.ndarray:
    model, scaler, feature_names = load_dl_artifacts()
    X_num = prepare_ml_features(X_test)

    missing_cols = [c for c in feature_names if c not in X_num.columns]
    if missing_cols:
        raise ValueError(f"DL 입력에 필요한 컬럼이 없습니다: {missing_cols}")

    X_num = X_num[feature_names]
    X_scaled = scaler.transform(X_num)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        proba = model(X_tensor).squeeze().numpy()
    return np.array(proba)


def get_lr_proba(X_train, X_test, y_train) -> np.ndarray:
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def get_rf_proba(X_train, X_test, y_train) -> np.ndarray:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def evaluate_at_threshold(y_true, proba, threshold: float, model_name: str) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "model": model_name,
        "threshold": round(float(threshold), 3),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pred_positive_rate": float(pred.mean()),
    }


def search_best_threshold(y_true, proba, model_name: str, objective: str = "f1") -> tuple[dict, pd.DataFrame]:
    rows = []
    thresholds = np.arange(0.05, 0.96, 0.05)

    for th in thresholds:
        row = evaluate_at_threshold(y_true, proba, th, model_name)
        rows.append(row)

    df = pd.DataFrame(rows)

    if objective == "f1":
        best = df.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0].to_dict()
    elif objective == "balanced_recall_precision":
        df["balance_gap"] = (df["recall"] - df["precision"]).abs()
        best = df.sort_values(["f1", "balance_gap"], ascending=[False, True]).iloc[0].to_dict()
    else:
        best = df.sort_values(["f1"], ascending=[False]).iloc[0].to_dict()

    return best, df


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_xy()
    X_train_num = prepare_ml_features(X_train)
    X_test_num = prepare_ml_features(X_test)

    lr_proba = get_lr_proba(X_train_num, X_test_num, y_train)
    rf_proba = get_rf_proba(X_train_num, X_test_num, y_train)
    dl_proba = get_dl_proba(X_test)

    best_lr, curve_lr = search_best_threshold(y_test, lr_proba, "logistic_regression")
    best_rf, curve_rf = search_best_threshold(y_test, rf_proba, "random_forest")
    best_dl, curve_dl = search_best_threshold(y_test, dl_proba, "DL_MLP")

    tuned_df = pd.DataFrame([best_lr, best_rf, best_dl]).sort_values("f1", ascending=False).reset_index(drop=True)
    tuned_df.to_csv(MODEL_DIR / "model_comparison_tuned.csv", index=False)

    curve_all = pd.concat([curve_lr, curve_rf, curve_dl], ignore_index=True)
    curve_all.to_csv(MODEL_DIR / "threshold_metrics_all_models.csv", index=False)

    print("저장 완료:")
    print(MODEL_DIR / "model_comparison_tuned.csv")
    print(MODEL_DIR / "threshold_metrics_all_models.csv")
    print("\nBest thresholds:")
    print(tuned_df)


if __name__ == "__main__":
    main()