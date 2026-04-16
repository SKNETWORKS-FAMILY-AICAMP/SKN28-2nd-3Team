# preprocess_subscriptions.py

"""
이 파일은 subscriptions(구독) 원천 데이터를 전처리하여
모델링에 사용할 수 있는 정제된 형태로 변환하는 모듈이다.

주요 역할:
- 원천 CSV 데이터 로드
- 날짜 컬럼 datetime 변환
- boolean 컬럼을 0/1로 변환
- 범주형 결측값 처리
- 전처리된 데이터를 interim 영역에 저장
"""
from __future__ import annotations  # 타입 힌트에서 forward reference 사용 가능하게 설정

import pandas as pd  # 데이터프레임 처리 라이브러리

# 경로 설정 (raw → interim)
from src.config.paths import RAW_DIR, INTERIM_DIR

# 설정값 (파일명)
from src.config.settings import SUBSCRIPTIONS_FILE

# 유틸 함수 (날짜 변환, bool → int 변환)
from src.utils.helpers import to_datetime, bool_to_int

# 입출력 및 로깅
from src.utils.io import read_csv, save_csv
from src.utils.logger import get_logger

# logger 생성
logger = get_logger(__name__)


# -----------------------------------
# subscriptions 데이터 전처리
# -----------------------------------
def preprocess_subscriptions() -> pd.DataFrame:

    # 원천 subscription 데이터 로드
    df = read_csv(RAW_DIR / SUBSCRIPTIONS_FILE)

    # start_date, end_date를 datetime으로 변환
    df = to_datetime(df, ["start_date", "end_date"])

    # boolean 컬럼들을 0/1로 변환
    for col in ["is_trial", "upgrade_flag", "downgrade_flag", "churn_flag", "auto_renew_flag"]:
        df[col] = bool_to_int(df[col])

    # 구독 등급 결측값 처리
    df["plan_tier"] = df["plan_tier"].fillna("Unknown")

    # 결제 주기 결측값 처리
    df["billing_frequency"] = df["billing_frequency"].fillna("Unknown")

    # 정제된 데이터 반환
    return df


# -----------------------------------
# 메인 실행 함수
# -----------------------------------
def main() -> None:
    df = preprocess_subscriptions()  # 전처리 수행

    # interim 경로에 저장
    save_csv(df, INTERIM_DIR / "subscriptions_clean.csv")

    # 로그 출력 (데이터 shape 포함)
    logger.info("saved subscriptions_clean.csv shape=%s", df.shape)


# 스크립트 실행 시 main 함수 호출
if __name__ == "__main__":
    main()