# split_dataset.py

"""
이 파일은 모델 학습에 사용할 데이터를 train / validation / test 세트로 분할하는 모듈이다.

주요 역할:
- 전처리 및 feature engineering이 완료된 학습용 테이블 로드
- 타겟 변수(y)와 입력 변수(X) 분리
- stratified sampling을 적용하여 클래스 비율을 유지한 데이터 분할
- train / validation / test 데이터셋 생성
- 모델 학습 및 평가에 사용할 다양한 형태의 데이터셋을 CSV로 저장
"""
from __future__ import annotations  # 타입 힌트 지연 평가 (forward reference 지원)

import pandas as pd  # 데이터프레임 처리 라이브러리
from sklearn.model_selection import train_test_split  # 데이터 분할 함수

# 경로 설정 (processed 데이터 위치)
from src.config.paths import PROCESSED_DIR

# 설정값 (난수 고정값, 타겟 변수명, validation/test 비율)
from src.config.settings import RANDOM_STATE, TARGET_COL, VALID_SIZE, TEST_SIZE

# CSV 입출력 및 로깅 유틸
from src.utils.io import read_csv, save_csv
from src.utils.logger import get_logger

# 현재 파일 기준 logger 생성
logger = get_logger(__name__)


def split_dataset() -> None:
    # 학습용 전체 테이블 로드 (feature engineering 완료된 데이터)
    df = read_csv(PROCESSED_DIR / "train_table_ml.csv")

    # 타겟 변수(y) 분리 (예: churn 여부)
    y = df[TARGET_COL]

    # ID 컬럼 정의 (현재는 account_id만 존재 시 포함)
    id_cols = [c for c in ["account_id"] if c in df.columns]

    # 입력 변수(X) 생성 (타겟 변수 제거)
    X = df.drop(columns=[TARGET_COL])

    # -----------------------------------
    # 1차 분할: train+valid vs test
    # -----------------------------------
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,              # test 비율
        random_state=RANDOM_STATE,        # 재현성 확보
        stratify=y                        # 클래스 비율 유지 (중요)
    )

    # -----------------------------------
    # 2차 분할: train vs valid
    # -----------------------------------
    # 전체에서 valid 비율을 맞추기 위해 비율 보정
    valid_ratio_adjusted = VALID_SIZE / (1 - TEST_SIZE)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid,
        test_size=valid_ratio_adjusted,   # 보정된 validation 비율
        random_state=RANDOM_STATE,
        stratify=y_train_valid            # 클래스 비율 유지
    )

    # -----------------------------------
    # train / valid / test 데이터 생성
    # -----------------------------------
    # X와 y를 다시 결합하여 최종 dataset 생성
    train = X_train.copy(); train[TARGET_COL] = y_train.values
    valid = X_valid.copy(); valid[TARGET_COL] = y_valid.values
    test = X_test.copy(); test[TARGET_COL] = y_test.values

    # -----------------------------------
    # 데이터 저장
    # -----------------------------------
    # 전체 데이터셋 저장
    save_csv(train, PROCESSED_DIR / "train.csv")
    save_csv(valid, PROCESSED_DIR / "valid.csv")
    save_csv(test, PROCESSED_DIR / "test.csv")

    # 입력 변수(X)만 저장
    save_csv(X_train, PROCESSED_DIR / "X_train.csv")
    save_csv(X_valid, PROCESSED_DIR / "X_valid.csv")
    save_csv(X_test, PROCESSED_DIR / "X_test.csv")

    # 타겟 변수(y)만 저장 (DataFrame 형태로 변환)
    save_csv(y_train.to_frame(name=TARGET_COL), PROCESSED_DIR / "y_train.csv")
    save_csv(y_valid.to_frame(name=TARGET_COL), PROCESSED_DIR / "y_valid.csv")
    save_csv(y_test.to_frame(name=TARGET_COL), PROCESSED_DIR / "y_test.csv")

    # 저장 완료 로그 출력
    logger.info("saved train/valid/test splits")


def main() -> None:
    # 데이터 분할 실행
    split_dataset()


# 스크립트 실행 시 main 함수 호출
if __name__ == "__main__":
    main()