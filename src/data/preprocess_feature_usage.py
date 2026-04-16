# preprocess_feature_usage.py

"""
이 파일은 feature_usage(서비스 기능 사용 로그) 데이터를 전처리하고,
subscription 정보를 활용하여 account 단위의 사용 패턴 feature로 집계하는 모듈이다.

주요 역할:
- 원천 usage 데이터 로드 및 날짜/boolean 타입 정제
- subscription_id → account_id 매핑을 통한 grain 변환
- 기준 시점(reference_date) 대비 사용 이력 파생 변수 생성
- account 단위로 사용량, 사용 다양성, 최근 사용 시점 등의 feature 집계
- 집계 결과를 모델 입력용 데이터로 저장
"""
from __future__ import annotations  # 타입 힌트에서 forward reference 사용 가능하게 설정

import pandas as pd  # 데이터프레임 처리 라이브러리

# 경로 설정 (raw → interim)
from src.config.paths import RAW_DIR, INTERIM_DIR

# 설정값 (파일명)
from src.config.settings import FEATURE_USAGE_FILE

# 유틸 함수 (날짜 변환, bool → int 변환)
from src.utils.helpers import to_datetime, bool_to_int

# 입출력 및 로깅
from src.utils.io import read_csv, save_csv
from src.utils.logger import get_logger

# logger 생성
logger = get_logger(__name__)


# -----------------------------------
# feature_usage 원천 데이터 전처리
# -----------------------------------
def preprocess_feature_usage() -> pd.DataFrame:

    # 원천 usage 데이터 로드
    df = read_csv(RAW_DIR / FEATURE_USAGE_FILE)

    # usage_date를 datetime으로 변환
    df = to_datetime(df, ["usage_date"])

    # beta feature 여부 컬럼이 존재하면 0/1로 변환
    if "is_beta_feature" in df.columns:
        df["is_beta_feature"] = bool_to_int(df["is_beta_feature"])

    # 정제된 데이터 반환
    return df


# -----------------------------------
# usage → account 단위 집계
# -----------------------------------
def aggregate_feature_usage(df: pd.DataFrame, subscriptions: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:

    # subscription_id 기준으로 account_id 연결 (grain 맞추기)
    merged = df.merge(
        subscriptions[["subscription_id", "account_id"]],
        on="subscription_id",
        how="left",
    )

    # 기준 날짜 대비 사용 경과 일수 계산
    merged["days_since_usage"] = (reference_date - merged["usage_date"]).dt.days

    # account 단위로 집계 (핵심 feature 생성)
    agg = merged.groupby("account_id").agg(
        total_usage_count=("usage_count", "sum"),               # 전체 사용 횟수
        total_usage_duration_secs=("usage_duration_secs", "sum"),  # 전체 사용 시간
        total_error_count=("error_count", "sum"),               # 전체 오류 수
        unique_feature_count=("feature_name", "nunique"),       # 사용한 기능 종류 수
        beta_feature_usage_count=("is_beta_feature", "sum"),    # 베타 기능 사용 횟수
        last_usage_date=("usage_date", "max"),                  # 마지막 사용일
        days_since_last_usage=("days_since_usage", "min"),      # 마지막 사용 이후 경과일
    ).reset_index()

    # 집계 결과 반환
    return agg


# -----------------------------------
# 메인 실행 함수
# -----------------------------------
def main() -> None:

    # usage 데이터 전처리
    usage = preprocess_feature_usage()

    # subscription 데이터 로드 (account 연결용)
    subs = read_csv(INTERIM_DIR / "subscriptions_clean.csv")

    # 날짜 컬럼 datetime 변환 (안전하게 처리)
    subs["start_date"] = pd.to_datetime(subs["start_date"], errors="coerce")
    subs["end_date"] = pd.to_datetime(subs["end_date"], errors="coerce")

    # 기준 날짜 설정 (가장 최근 시점)
    reference_date = max(
        usage["usage_date"].max(),
        subs["end_date"].max(),
        subs["start_date"].max(),
    )

    # account 단위 usage feature 생성
    agg = aggregate_feature_usage(usage, subs, reference_date)

    # 결과 저장
    save_csv(agg, INTERIM_DIR / "feature_usage_agg.csv")

    # 로그 출력
    logger.info("saved feature_usage_agg.csv shape=%s", agg.shape)


# 스크립트 실행 시 main 함수 호출
if __name__ == "__main__":
    main()