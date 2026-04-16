# preprocess_accounts.py

"""
이 파일은 accounts(고객 계정) 원천 데이터를 전처리하여
모델링의 기준이 되는 account 단위의 기본 정보와 타겟 변수를 정제하는 모듈이다.

주요 역할:
- 원천 accounts 데이터 로드
- 가입일(sign-up date) 컬럼 datetime 변환
- trial 여부 및 churn 여부 boolean 값을 0/1로 변환
- account_id 기준 중복 제거
- 주요 범주형 컬럼 결측값 처리
- 분석 및 모델링에 불필요한 컬럼 제거
- 전처리된 결과를 interim 영역에 저장
"""
from __future__ import annotations  # 타입 힌트에서 forward reference 사용 가능하게 설정

import pandas as pd  # 데이터프레임 처리 라이브러리

# 경로 설정 (raw → interim)
from src.config.paths import RAW_DIR, INTERIM_DIR

# 설정값 (파일명, 타겟 변수)
from src.config.settings import ACCOUNT_FILE, TARGET_COL

# 유틸 함수 (날짜 변환, bool → int 변환)
from src.utils.helpers import to_datetime, bool_to_int

# 입출력 및 로깅
from src.utils.io import read_csv, save_csv
from src.utils.logger import get_logger

# logger 생성
logger = get_logger(__name__)


# -----------------------------------
# accounts 테이블 전처리
# -----------------------------------
def preprocess_accounts() -> pd.DataFrame:

    # 원천 accounts 데이터 로드
    df = read_csv(RAW_DIR / ACCOUNT_FILE)

    # signup_date 컬럼을 datetime으로 변환
    df = to_datetime(df, ["signup_date"])

    # boolean 컬럼 정의 (trial 여부, churn 여부)
    bool_cols = ["is_trial", TARGET_COL]

    # boolean → int (0/1) 변환
    for col in bool_cols:
        df[col] = bool_to_int(df[col])

    # account_id 기준 중복 제거 (중복 계정 방지)
    df = df.drop_duplicates(subset=["account_id"]).copy()

    # 범주형 결측값 처리 (Unknown으로 통일)
    df["industry"] = df["industry"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")
    df["referral_source"] = df["referral_source"].fillna("Unknown")

    # 분석/모델링에 필요 없는 컬럼 제거
    if "account_name" in df.columns:
        df = df.drop(columns=["account_name"])

    # 정제된 데이터 반환
    return df


# -----------------------------------
# 메인 실행 함수
# -----------------------------------
def main() -> None:
    df = preprocess_accounts()  # 전처리 수행

    # interim 경로에 저장
    save_csv(df, INTERIM_DIR / "accounts_clean.csv")

    # 로그 출력 (데이터 shape 포함)
    logger.info("saved accounts_clean.csv shape=%s", df.shape)


# 스크립트 실행 시 main 함수 호출
if __name__ == "__main__":
    main()