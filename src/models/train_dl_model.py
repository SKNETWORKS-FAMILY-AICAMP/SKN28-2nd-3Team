# train_dl_model.py

"""
이 파일은 전처리된 학습 데이터를 이용해 딥러닝 MLP 모델을 학습하고,
예측에 필요한 모델 가중치, 입력 feature 목록, scaler를 함께 저장하는 모듈이다.

주요 역할:
- 학습용 입력 데이터(X_train, y_train) 로드
- 딥러닝 입력에 맞게 feature 전처리 및 표준화 수행
- MLP(Multi-Layer Perceptron) 모델 학습
- 학습된 모델 가중치, feature 목록, scaler 저장
- 이후 예측 단계에서 동일한 입력 구조를 재현할 수 있도록 학습 산출물 보존
"""
from pathlib import Path  # 파일 및 폴더 경로 처리를 위한 객체
import pickle  # scaler 객체 저장용 직렬화 라이브러리

import pandas as pd  # 데이터프레임 처리 라이브러리
from sklearn.preprocessing import StandardScaler  # 수치형 변수 표준화 도구

import torch  # PyTorch 딥러닝 프레임워크
import torch.nn as nn  # 신경망 레이어 모듈


# 현재 파일 위치를 기준으로 프로젝트 루트 경로 계산
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 전처리된 데이터 및 모델 저장 경로 설정
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


class MLP(nn.Module):
    # 다층 퍼셉트론(MLP) 모델 정의
    def __init__(self, input_dim: int):
        super().__init__()

        # 순차형 신경망 구조 정의
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),  # 입력층 → 은닉층1
            nn.ReLU(),                 # 활성화 함수
            nn.Dropout(0.3),           # 과적합 방지용 dropout
            nn.Linear(64, 32),         # 은닉층1 → 은닉층2
            nn.ReLU(),                 # 활성화 함수
            nn.Dropout(0.2),           # 과적합 방지용 dropout
            nn.Linear(32, 1),          # 은닉층2 → 출력층
            nn.Sigmoid(),              # 이진 분류 확률 출력
        )

    def forward(self, x):
        # 입력 데이터를 모델에 통과시켜 예측값 반환
        return self.model(x)


def load_data():
    # 학습용 입력/타겟 데이터 경로 설정
    x_path = PROCESSED_DIR / "X_train.csv"
    y_path = PROCESSED_DIR / "y_train.csv"

    # 필요한 파일이 없으면 에러 발생
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "X_train.csv 또는 y_train.csv가 없습니다.\n"
            "먼저 python -m src.data.split_dataset 를 실행하세요."
        )

    # CSV 파일 로드
    X_train = pd.read_csv(x_path)
    y_train = pd.read_csv(y_path)

    # y_train이 1컬럼 DataFrame이면 Series로 변환
    if y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]

    # 입력 데이터와 타겟 반환
    return X_train, y_train


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    DL 입력용 feature 정리
    - 식별자 제거
    - bool -> int 변환
    - 숫자형 컬럼만 사용
    - 결측은 중앙값 대체
    """
    # 원본 데이터 손상을 막기 위해 복사본 생성
    X = X.copy()

    # -----------------------------------
    # 1. 식별자 컬럼 제거
    # -----------------------------------
    drop_candidates = ["account_id", "customer_id", "id"]
    existing_drop_cols = [col for col in drop_candidates if col in X.columns]
    if existing_drop_cols:
        X = X.drop(columns=existing_drop_cols)

    # -----------------------------------
    # 2. bool 타입을 int로 변환
    # -----------------------------------
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype(int)

    # -----------------------------------
    # 3. 숫자형 컬럼만 선택
    # -----------------------------------
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    X_num = X[numeric_cols].copy()

    # -----------------------------------
    # 4. 결측값을 중앙값으로 대체
    # -----------------------------------
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())

    # 숫자형 컬럼이 하나도 없으면 학습 불가
    if X_num.empty:
        raise ValueError("DL 학습에 사용할 숫자형 컬럼이 없습니다.")

    # 전처리된 숫자형 feature 반환
    return X_num


def preprocess_and_fit_scaler(X_train: pd.DataFrame):
    # 딥러닝 입력용 feature 전처리 수행
    X_num = prepare_features(X_train)

    # 표준화 객체 생성
    scaler = StandardScaler()

    # 학습 데이터 기준으로 평균/표준편차를 학습하고 transform 수행
    X_scaled = scaler.fit_transform(X_num)

    # 스케일된 데이터, feature 이름 목록, scaler 반환
    return X_scaled, X_num.columns.tolist(), scaler


def train_model(X_train, y_train):
    # numpy 배열/Series를 torch tensor로 변환
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    # 모델 생성
    model = MLP(input_dim=X_train.shape[1])

    # 이진 분류 손실 함수 설정
    criterion = nn.BCELoss()

    # Adam optimizer 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # -----------------------------------
    # 학습 루프
    # -----------------------------------
    for epoch in range(20):
        # 순전파: 예측값 계산
        pred = model(X_tensor)

        # 손실 계산
        loss = criterion(pred, y_tensor)

        # 이전 gradient 초기화
        optimizer.zero_grad()

        # 역전파
        loss.backward()

        # 가중치 업데이트
        optimizer.step()

        # 5 epoch마다 loss 출력
        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

    # 학습 완료된 모델 반환
    return model


def save_artifacts(model, feature_names: list[str], scaler: StandardScaler):
    # 모델 저장 폴더가 없으면 생성
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 저장할 파일 경로 정의
    model_path = MODEL_DIR / "dl_model.pth"
    feature_path = MODEL_DIR / "dl_feature_columns.csv"
    scaler_path = MODEL_DIR / "dl_scaler.pkl"

    # 모델 가중치 저장
    torch.save(model.state_dict(), model_path)

    # 학습에 사용한 feature 목록 저장
    pd.DataFrame({"feature": feature_names}).to_csv(feature_path, index=False)

    # scaler 객체 저장
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # 저장 완료 메시지 출력
    print(f"DL 모델 저장 완료: {model_path}")
    print(f"DL feature 목록 저장 완료: {feature_path}")
    print(f"DL scaler 저장 완료: {scaler_path}")


def main():
    # 학습 시작 메시지 출력
    print("DL 모델 학습 시작")

    # -----------------------------------
    # 1. 데이터 로드 및 전처리
    # -----------------------------------
    X_train, y_train = load_data()
    X_scaled, feature_names, scaler = preprocess_and_fit_scaler(X_train)

    # 입력 feature 정보 출력
    print(f"사용 feature 수: {len(feature_names)}")
    print("DL 입력 컬럼 예시:", feature_names[:10])

    # -----------------------------------
    # 2. 모델 학습
    # -----------------------------------
    model = train_model(X_scaled, y_train)

    # -----------------------------------
    # 3. 산출물 저장
    # -----------------------------------
    save_artifacts(model, feature_names, scaler)

    # 학습 종료 메시지 출력
    print("DL 모델 학습 종료")


# 스크립트를 직접 실행할 때 main 함수 호출
if __name__ == "__main__":
    main()