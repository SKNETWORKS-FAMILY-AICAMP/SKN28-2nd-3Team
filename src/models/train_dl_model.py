from pathlib import Path
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn


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


def load_data():
    x_path = PROCESSED_DIR / "X_train.csv"
    y_path = PROCESSED_DIR / "y_train.csv"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "X_train.csv 또는 y_train.csv가 없습니다.\n"
            "먼저 python -m src.data.split_dataset 를 실행하세요."
        )

    X_train = pd.read_csv(x_path)
    y_train = pd.read_csv(y_path)

    if y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]

    return X_train, y_train


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    DL 입력용 feature 정리
    - 식별자 제거
    - bool -> int 변환
    - 숫자형 컬럼만 사용
    - 결측은 중앙값 대체
    """
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
        raise ValueError("DL 학습에 사용할 숫자형 컬럼이 없습니다.")

    return X_num


def preprocess_and_fit_scaler(X_train: pd.DataFrame):
    X_num = prepare_features(X_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    return X_scaled, X_num.columns.tolist(), scaler


def train_model(X_train, y_train):
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    model = MLP(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

    return model


def save_artifacts(model, feature_names: list[str], scaler: StandardScaler):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "dl_model.pth"
    feature_path = MODEL_DIR / "dl_feature_columns.csv"
    scaler_path = MODEL_DIR / "dl_scaler.pkl"

    torch.save(model.state_dict(), model_path)
    pd.DataFrame({"feature": feature_names}).to_csv(feature_path, index=False)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"DL 모델 저장 완료: {model_path}")
    print(f"DL feature 목록 저장 완료: {feature_path}")
    print(f"DL scaler 저장 완료: {scaler_path}")


def main():
    print("DL 모델 학습 시작")

    X_train, y_train = load_data()
    X_scaled, feature_names, scaler = preprocess_and_fit_scaler(X_train)

    print(f"사용 feature 수: {len(feature_names)}")
    print("DL 입력 컬럼 예시:", feature_names[:10])

    model = train_model(X_scaled, y_train)
    save_artifacts(model, feature_names, scaler)

    print("DL 모델 학습 종료")


if __name__ == "__main__":
    main()