# data_check.py

"""
이 파일은 프로젝트에서 사용하는 모든 원천(raw) 데이터 테이블에 대해
컬럼 단위의 데이터 품질을 점검하고 요약 리포트를 생성하는 모듈이다.

주요 역할:
- accounts, subscriptions, feature_usage, support_tickets, churn_events 원천 데이터 로드
- 주요 날짜 컬럼을 datetime 타입으로 변환하여 일관성 확보
- 각 컬럼별 데이터 타입, 결측치 개수/비율, 고유값 개수 등의 통계 요약 생성
- 테이블별 요약 정보를 하나의 데이터프레임으로 통합
- 데이터 품질 점검 결과를 docs 영역에 CSV 파일로 저장
"""
from __future__ import annotations  # 타입 힌트에서 forward reference 사용 가능하게 설정

import pandas as pd  # 데이터프레임 처리 라이브러리

# 경로 설정 (raw 데이터 위치, 결과 저장 위치)
from src.config.paths import RAW_DIR, DOCS_DIR

# 원천 데이터 파일명 설정
from src.config.settings import (
    ACCOUNT_FILE, SUBSCRIPTIONS_FILE, FEATURE_USAGE_FILE,
    SUPPORT_TICKETS_FILE, CHURN_EVENTS_FILE
)

# 날짜 컬럼을 datetime으로 변환하는 유틸 함수
from src.utils.helpers import to_datetime

# CSV 입출력 유틸
from src.utils.io import read_csv, save_csv

# 로깅 설정
from src.utils.logger import get_logger

# 현재 파일 기준 logger 생성
logger = get_logger(__name__)


# 데이터프레임의 컬럼별 요약 정보를 생성하는 함수
def summarize_df(name: str, df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "column": df.columns,                          # 컬럼명
        "dtype": df.dtypes.astype(str).values,         # 데이터 타입
        "missing_count": df.isna().sum().values,       # 결측치 개수
        "missing_ratio": (df.isna().mean() * 100).round(2).values,  # 결측치 비율(%)
        "nunique": df.nunique(dropna=False).values,    # 고유값 개수 (NaN 포함)
    })
    summary.insert(0, "table_name", name)  # 어떤 테이블인지 구분하기 위한 컬럼 추가
    return summary  # 요약 데이터 반환


# 메인 실행 함수
def main() -> None:
    # 분석 대상 원천 데이터 파일 목록 정의
    files = {
        "accounts": ACCOUNT_FILE,
        "subscriptions": SUBSCRIPTIONS_FILE,
        "feature_usage": FEATURE_USAGE_FILE,
        "support_tickets": SUPPORT_TICKETS_FILE,
        "churn_events": CHURN_EVENTS_FILE,
    }

    all_summaries = []  # 각 테이블 요약 결과를 담을 리스트

    # 각 파일을 순회하며 데이터 점검 수행
    for name, filename in files.items():
        path = RAW_DIR / filename  # 파일 경로 생성

        df = read_csv(path)  # CSV 파일 읽기

        # 테이블별 날짜 컬럼을 datetime으로 변환
        if name == "accounts":
            df = to_datetime(df, ["signup_date"])
        elif name == "subscriptions":
            df = to_datetime(df, ["start_date", "end_date"])
        elif name == "feature_usage":
            df = to_datetime(df, ["usage_date"])
        elif name == "support_tickets":
            df = to_datetime(df, ["submitted_at", "closed_at"])
        elif name == "churn_events":
            df = to_datetime(df, ["churn_date"])

        # 데이터 shape 로그 출력 (행, 열 개수 확인)
        logger.info("%s shape=%s", name, df.shape)

        # 요약 정보 생성 후 리스트에 추가
        all_summaries.append(summarize_df(name, df))

    # 모든 테이블 요약을 하나의 데이터프레임으로 결합
    summary_df = pd.concat(all_summaries, ignore_index=True)

    # 결과를 CSV로 저장
    save_csv(summary_df, DOCS_DIR / "raw_data_check_summary.csv")

    # 저장 완료 로그 출력
    logger.info("saved raw data check summary")


# 스크립트 실행 시 main 함수 실행
if __name__ == "__main__":
    main()