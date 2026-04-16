# predict_dl_model.py

"""
이 파일은 저장된 딥러닝 MLP 모델을 불러와 테스트 데이터에 대한 예측을 수행하고,
예측 결과 및 성능 평가 지표를 파일로 저장하는 모듈이다.

주요 역할:
- 테스트용 입력 데이터(X_test, y_test) 로드
- 예측에 필요한 수치형 feature 정리 및 결측값 보정
- 학습 시 저장한 feature 목록과 scaler 불러오기
- 저장된 딥러닝 MLP 모델 가중치 로드
- 테스트 데이터에 대한 예측 확률 및 예측 라벨 생성
- 예측 결과와 성능 지표를 CSV 파일로 저장
"""
from pathlib import Path  # 파일 및 폴더 경로 처리를 위한 객체
import pickle  # 저장된 scaler 객체 로드를 위한 직렬화 라이브러리

import pandas as pd  # 데이터프레임 처리 라이브러리
import torch  # PyTorch 딥러닝 프레임워크
import torch.nn as nn  # 신경망 레이어 모듈
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  # 분류 성능 지표


# 현재 파일 위치를 기준으로 프로젝트 루트 경로 계산
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 전처리된 데이터 및 모델 산출물 경로 설정
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


class MLP(nn.Module):
    # 다층 퍼셉트론(MLP) 모델 정의
    def __init__(self, input_dim: int):
        super().__init__()

        # 순차형 신경망 구성
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),  # 입력층 → 은닉층1
            nn.ReLU(),                 # 활성화 함수
            nn.Dropout(0.3),           # 과적합 방지용 dropout
            nn.Linear(64, 32),         # 은닉층1 → 은닉층2
            nn.ReLU(),                 # 활성화 함수
            nn.Dropout(0.2),           # 과적합 방지용 dropout
            nn.Linear(32, 1),          # 은닉층2 → 출력층
            nn.Sigmoid(),              # 이진 분류용 확률 출력
        )

    def forward(self, x):
        # 입력 데이터를 신경망에 통과시켜 출력 반환
        return self.model(x)


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    # 원본 데이터 손상을 막기 위해 복사본 생성
    X = X.copy()

    # -----------------------------------
    # 1. ID 성격 컬럼 제거
    # -----------------------------------
    # 모델 입력에 불필요할 수 있는 식별자 컬럼 후보 정의
    drop_candidates = ["account_id", "customer_id", "id"]

    # 실제 존재하는 식별자 컬럼만 선택
    existing_drop_cols = [col for col in drop_candidates if col in X.columns]

    # 존재하는 식별자 컬럼 제거
    if existing_drop_cols:
        X = X.drop(columns=existing_drop_cols)

    # -----------------------------------
    # 2. bool 타입을 int 타입으로 변환
    # -----------------------------------
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype(int)

    # -----------------------------------
    # 3. 수치형 컬럼만 선택
    # -----------------------------------
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    X_num = X[numeric_cols].copy()

    # -----------------------------------
    # 4. 결측값 보정
    # -----------------------------------
    # 각 수치형 컬럼의 결측값을 중앙값으로 대체
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())

    # 수치형 컬럼이 하나도 없으면 예측 불가
    if X_num.empty:
        raise ValueError("예측에 사용할 숫자형 컬럼이 없습니다.")

    # 전처리된 수치형 feature 반환
    return X_num


def load_scaler():
    # 저장된 scaler 파일 경로
    scaler_path = MODEL_DIR / "dl_scaler.pkl"

    # scaler 파일이 없으면 에러 발생
    if not scaler_path.exists():
        raise FileNotFoundError(
            "dl_scaler.pkl이 없습니다.\n"
            "먼저 python -m src.models.train_dl_model 을 다시 실행하세요."
        )

    # pickle로 scaler 객체 로드
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return scaler


def load_feature_columns():
    # 학습 시 사용한 feature 목록 파일 경로
    feature_path = MODEL_DIR / "dl_feature_columns.csv"

    # feature 목록 파일이 없으면 에러 발생
    if not feature_path.exists():
        raise FileNotFoundError(
            "dl_feature_columns.csv가 없습니다.\n"
            "먼저 python -m src.models.train_dl_model 을 다시 실행하세요."
        )

    # feature 목록 로드 후 리스트로 반환
    feature_df = pd.read_csv(feature_path)
    return feature_df["feature"].tolist()


def load_model(input_dim: int):
    # 저장된 모델 가중치 파일 경로
    model_path = MODEL_DIR / "dl_model.pth"

    # 모델 파일이 없으면 에러 발생
    if not model_path.exists():
        raise FileNotFoundError(
            "dl_model.pth가 없습니다.\n"
            "먼저 python -m src.models.train_dl_model 을 실행하세요."
        )

    # 동일한 구조의 MLP 모델 생성
    model = MLP(input_dim=input_dim)

    # 저장된 가중치 불러오기 (CPU 환경 기준)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    # 평가 모드로 전환
    model.eval()

    return model


def evaluate_predictions(y_true: pd.Series, pred_proba, threshold: float = 0.5) -> pd.DataFrame:
    # -----------------------------------
    # 1. 확률값을 라벨(0/1)로 변환
    # -----------------------------------
    pred_label = (pred_proba >= threshold).astype(int)

    # -----------------------------------
    # 2. 핵심 평가 지표 계산
    # -----------------------------------
    metrics = {
        "model": "DL_MLP",  # 모델명
        "accuracy": accuracy_score(y_true, pred_label),                      # 정확도
        "precision": precision_score(y_true, pred_label, zero_division=0),  # 정밀도
        "recall": recall_score(y_true, pred_label, zero_division=0),        # 재현율
        "f1": f1_score(y_true, pred_label, zero_division=0),                # F1-score
        "roc_auc": roc_auc_score(y_true, pred_proba),                       # ROC-AUC
    }

    # 결과를 DataFrame 형태로 반환
    return pd.DataFrame([metrics])


def main():
    # -----------------------------------
    # 1. 테스트 데이터 경로 설정
    # -----------------------------------
    x_test_path = PROCESSED_DIR / "X_test.csv"
    y_test_path = PROCESSED_DIR / "y_test.csv"

    # 테스트 데이터가 없으면 에러 발생
    if not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            "X_test.csv 또는 y_test.csv가 없습니다.\n"
            "먼저 python -m src.data.split_dataset 를 실행하세요."
        )

    # -----------------------------------
    # 2. 테스트 데이터 로드
    # -----------------------------------
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    # y_test가 1컬럼 DataFrame이면 Series로 변환
    if y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    # account_id가 있으면 예측 결과와 함께 저장하기 위해 따로 보관
    account_ids = X_test["account_id"].copy() if "account_id" in X_test.columns else None

    # -----------------------------------
    # 3. 예측용 feature 전처리
    # -----------------------------------
    X_test_num = prepare_features(X_test)

    # 학습 시 사용한 feature 순서/목록 불러오기
    feature_names = load_feature_columns()

    # 테스트 데이터에 필요한 feature가 모두 있는지 확인
    missing_cols = [col for col in feature_names if col not in X_test_num.columns]
    if missing_cols:
        raise ValueError(f"X_test에 필요한 feature가 없습니다: {missing_cols}")

    # 학습 때와 동일한 컬럼 순서로 정렬
    X_test_num = X_test_num[feature_names]

    # -----------------------------------
    # 4. 스케일링 적용
    # -----------------------------------
    scaler = load_scaler()
    X_test_scaled = scaler.transform(X_test_num)

    # -----------------------------------
    # 5. 모델 로드 및 예측 수행
    # -----------------------------------
    # numpy 배열을 torch tensor로 변환
    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    # 저장된 모델 로드
    model = load_model(input_dim=X_test_scaled.shape[1])

    # gradient 계산 없이 추론만 수행
    with torch.no_grad():
        pred_proba = model(X_tensor).squeeze().numpy()

    # -----------------------------------
    # 6. 예측 결과 테이블 생성
    # -----------------------------------
    pred_df = pd.DataFrame({
        "pred_proba": pred_proba
    })

    # account_id가 있으면 맨 앞 컬럼으로 추가
    if account_ids is not None:
        pred_df.insert(0, "account_id", account_ids.values)

    # threshold 0.5 기준으로 예측 라벨 생성
    pred_df["pred_label"] = (pred_df["pred_proba"] >= 0.5).astype(int)

    # 예측 결과 저장
    pred_save_path = MODEL_DIR / "dl_test_predictions.csv"
    pred_df.to_csv(pred_save_path, index=False)

    # -----------------------------------
    # 7. 성능 평가 및 저장
    # -----------------------------------
    metric_df = evaluate_predictions(y_test, pred_proba, threshold=0.5)

    metric_save_path = MODEL_DIR / "dl_metrics.csv"
    metric_df.to_csv(metric_save_path, index=False)

    # 완료 메시지 출력
    print(f"예측 완료: {pred_save_path}")
    print(f"DL 평가 지표 저장 완료: {metric_save_path}")


# 스크립트를 직접 실행할 때 main 함수 호출
if __name__ == "__main__":
    main()