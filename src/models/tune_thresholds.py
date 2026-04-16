# tune_thresholds.py

"""
이 파일은 여러 모델(Logistic Regression, Random Forest, DL MLP)에 대해
threshold를 변화시키며 성능을 비교하고, 각 모델별 최적 threshold를 탐색하는 모듈이다.

주요 역할:
- train/test 데이터 로드 및 공통 전처리 수행
- ML 모델(Logistic Regression, Random Forest) 학습 및 확률 예측
- 저장된 DL 모델을 불러와 확률 예측 수행
- 다양한 threshold 구간에서 precision, recall, f1 등 성능 평가
- 각 모델별 최적 threshold 선택
- 전체 threshold 성능 곡선 및 최종 비교표를 CSV로 저장
"""
from pathlib import Path  # 파일 및 폴더 경로 처리를 위한 객체
import pickle  # scaler 객체 로드를 위한 직렬화 라이브러리
import numpy as np  # 수치 계산 라이브러리
import pandas as pd  # 데이터프레임 처리 라이브러리
import torch  # PyTorch 프레임워크
import torch.nn as nn  # 신경망 모듈

from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀 모델
from sklearn.ensemble import RandomForestClassifier  # 랜덤 포레스트 모델
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  # 평가 지표


# 프로젝트 루트 및 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


class MLP(nn.Module):
    # DL 모델 구조 정의 (학습 시 사용한 구조와 동일해야 함)
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
    # 학습 및 테스트 데이터 로드
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0]
    return X_train, X_test, y_train, y_test


def prepare_ml_features(X: pd.DataFrame) -> pd.DataFrame:
    # 모델 입력용 feature 전처리
    X = X.copy()

    # 식별자 컬럼 제거
    drop_cols = [c for c in ["account_id", "customer_id", "id"] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    # bool → int 변환
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype(int)

    # 숫자형 컬럼만 사용
    X_num = X.select_dtypes(include=["number"]).copy()

    # 결측값 중앙값 대체
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())

    return X_num


def load_dl_artifacts():
    # DL 모델 관련 파일 로드 (feature, scaler, model)
    feature_names = pd.read_csv(MODEL_DIR / "dl_feature_columns.csv")["feature"].tolist()

    with open(MODEL_DIR / "dl_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model = MLP(input_dim=len(feature_names))
    model.load_state_dict(torch.load(MODEL_DIR / "dl_model.pth", map_location="cpu"))
    model.eval()

    return model, scaler, feature_names


def get_dl_proba(X_test: pd.DataFrame) -> np.ndarray:
    # DL 모델로 확률 예측
    model, scaler, feature_names = load_dl_artifacts()

    X_num = prepare_ml_features(X_test)

    # 필요한 feature가 없는 경우 에러
    missing_cols = [c for c in feature_names if c not in X_num.columns]
    if missing_cols:
        raise ValueError(f"DL 입력에 필요한 컬럼이 없습니다: {missing_cols}")

    # feature 순서 맞추기
    X_num = X_num[feature_names]

    # 스케일링 적용
    X_scaled = scaler.transform(X_num)

    # tensor 변환 후 예측
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        proba = model(X_tensor).squeeze().numpy()

    return np.array(proba)


def get_lr_proba(X_train, X_test, y_train) -> np.ndarray:
    # Logistic Regression 학습 및 예측
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def get_rf_proba(X_train, X_test, y_train) -> np.ndarray:
    # Random Forest 학습 및 예측
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
    # threshold 기준으로 확률 → 라벨 변환
    pred = (proba >= threshold).astype(int)

    # 성능 지표 계산
    return {
        "model": model_name,
        "threshold": round(float(threshold), 3),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pred_positive_rate": float(pred.mean()),  # 양성 예측 비율
    }


def search_best_threshold(y_true, proba, model_name: str, objective: str = "f1") -> tuple[dict, pd.DataFrame]:
    # 다양한 threshold에서 성능 평가
    rows = []
    thresholds = np.arange(0.05, 0.96, 0.05)

    for th in thresholds:
        row = evaluate_at_threshold(y_true, proba, th, model_name)
        rows.append(row)

    df = pd.DataFrame(rows)

    # 최적 threshold 선택 기준
    if objective == "f1":
        best = df.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0].to_dict()

    elif objective == "balanced_recall_precision":
        df["balance_gap"] = (df["recall"] - df["precision"]).abs()
        best = df.sort_values(["f1", "balance_gap"], ascending=[False, True]).iloc[0].to_dict()

    else:
        best = df.sort_values(["f1"], ascending=[False]).iloc[0].to_dict()

    return best, df


def main():
    # 결과 저장 폴더 생성
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 데이터 로드 및 전처리
    X_train, X_test, y_train, y_test = load_xy()
    X_train_num = prepare_ml_features(X_train)
    X_test_num = prepare_ml_features(X_test)

    # 모델별 확률 예측
    lr_proba = get_lr_proba(X_train_num, X_test_num, y_train)
    rf_proba = get_rf_proba(X_train_num, X_test_num, y_train)
    dl_proba = get_dl_proba(X_test)

    # 모델별 최적 threshold 탐색
    best_lr, curve_lr = search_best_threshold(y_test, lr_proba, "logistic_regression")
    best_rf, curve_rf = search_best_threshold(y_test, rf_proba, "random_forest")
    best_dl, curve_dl = search_best_threshold(y_test, dl_proba, "DL_MLP")

    # 최종 비교표 생성 및 저장
    tuned_df = pd.DataFrame([best_lr, best_rf, best_dl]) \
        .sort_values("f1", ascending=False) \
        .reset_index(drop=True)

    tuned_df.to_csv(MODEL_DIR / "model_comparison_tuned.csv", index=False)

    # 모든 threshold 결과 저장
    curve_all = pd.concat([curve_lr, curve_rf, curve_dl], ignore_index=True)
    curve_all.to_csv(MODEL_DIR / "threshold_metrics_all_models.csv", index=False)

    # 결과 출력
    print("저장 완료:")
    print(MODEL_DIR / "model_comparison_tuned.csv")
    print(MODEL_DIR / "threshold_metrics_all_models.csv")
    print("\nBest thresholds:")
    print(tuned_df)


if __name__ == "__main__":
    main()