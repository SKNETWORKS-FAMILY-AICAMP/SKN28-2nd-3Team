# predict_dl_model.py
from pathlib import Path
import pickle

import pandas as pd
import torch
import torch.nn as nn
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


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    drop_candidates = ["account_id", "customer_id", "id"]
    existing_drop_cols = [col for col in drop_candidates if col in X.columns]
    if existing_drop_cols:
        X = X.drop(columns=existing_drop_cols)

    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype(int)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    X_num = X[numeric_cols].copy()

    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())

    if X_num.empty:
        raise ValueError("예측에 사용할 숫자형 컬럼이 없습니다.")

    return X_num


def load_scaler():
    scaler_path = MODEL_DIR / "dl_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            "dl_scaler.pkl이 없습니다.\n"
            "먼저 python -m src.models.train_dl_model 을 다시 실행하세요."
        )

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return scaler


def load_feature_columns():
    feature_path = MODEL_DIR / "dl_feature_columns.csv"
    if not feature_path.exists():
        raise FileNotFoundError(
            "dl_feature_columns.csv가 없습니다.\n"
            "먼저 python -m src.models.train_dl_model 을 다시 실행하세요."
        )

    feature_df = pd.read_csv(feature_path)
    return feature_df["feature"].tolist()


def load_model(input_dim: int):
    model_path = MODEL_DIR / "dl_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            "dl_model.pth가 없습니다.\n"
            "먼저 python -m src.models.train_dl_model 을 실행하세요."
        )

    model = MLP(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def evaluate_predictions(y_true: pd.Series, pred_proba, threshold: float = 0.5) -> pd.DataFrame:
    pred_label = (pred_proba >= threshold).astype(int)

    metrics = {
        "model": "DL_MLP",
        "accuracy": accuracy_score(y_true, pred_label),
        "precision": precision_score(y_true, pred_label, zero_division=0),
        "recall": recall_score(y_true, pred_label, zero_division=0),
        "f1": f1_score(y_true, pred_label, zero_division=0),
        "roc_auc": roc_auc_score(y_true, pred_proba),
    }
    return pd.DataFrame([metrics])


def main():
    x_test_path = PROCESSED_DIR / "X_test.csv"
    y_test_path = PROCESSED_DIR / "y_test.csv"

    if not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            "X_test.csv 또는 y_test.csv가 없습니다.\n"
            "먼저 python -m src.data.split_dataset 를 실행하세요."
        )

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    if y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    account_ids = X_test["account_id"].copy() if "account_id" in X_test.columns else None

    X_test_num = prepare_features(X_test)

    feature_names = load_feature_columns()
    missing_cols = [col for col in feature_names if col not in X_test_num.columns]
    if missing_cols:
        raise ValueError(f"X_test에 필요한 feature가 없습니다: {missing_cols}")

    X_test_num = X_test_num[feature_names]

    scaler = load_scaler()
    X_test_scaled = scaler.transform(X_test_num)

    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    model = load_model(input_dim=X_test_scaled.shape[1])

    with torch.no_grad():
        pred_proba = model(X_tensor).squeeze().numpy()

    pred_df = pd.DataFrame({
        "pred_proba": pred_proba
    })

    if account_ids is not None:
        pred_df.insert(0, "account_id", account_ids.values)

    pred_df["pred_label"] = (pred_df["pred_proba"] >= 0.5).astype(int)

    pred_save_path = MODEL_DIR / "dl_test_predictions.csv"
    pred_df.to_csv(pred_save_path, index=False)

    metric_df = evaluate_predictions(y_test, pred_proba, threshold=0.5)
    metric_save_path = MODEL_DIR / "dl_metrics.csv"
    metric_df.to_csv(metric_save_path, index=False)

    print(f"예측 완료: {pred_save_path}")
    print(f"DL 평가 지표 저장 완료: {metric_save_path}")


if __name__ == "__main__":
    main()