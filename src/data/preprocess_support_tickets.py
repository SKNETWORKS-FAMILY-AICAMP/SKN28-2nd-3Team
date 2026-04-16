# preprocess_support_tickets.py

"""
이 파일은 support_tickets 원천 데이터를 전처리하고,
account 단위로 집계(aggregation)하여 모델 입력용 feature를 생성하는 모듈이다.

주요 역할:
1. 원천 CSV 데이터 로드
2. 날짜 컬럼 datetime 변환
3. boolean → numeric 변환
4. account 기준 집계(feature engineering)
5. 중간 결과(interim 데이터) 저장
"""
from __future__ import annotations  # Python 3.7+에서 타입 힌트를 지연 평가 (순환참조 방지 등)

import pandas as pd  # 데이터 처리 라이브러리

# 프로젝트 내부 설정 및 유틸 import
from src.config.paths import RAW_DIR, INTERIM_DIR  # 데이터 경로 상수
from src.config.settings import SUPPORT_TICKETS_FILE  # 파일명 설정
from src.utils.helpers import to_datetime, bool_to_int  # 전처리 helper 함수
from src.utils.io import read_csv, save_csv  # 입출력 함수
from src.utils.logger import get_logger  # 로깅 설정

logger = get_logger(__name__)  # 현재 모듈 기준 logger 생성


def preprocess_support_tickets() -> pd.DataFrame:
    """
    support_tickets 원천 데이터를 전처리하는 함수

    수행 작업:
    - CSV 파일 로드
    - 날짜 컬럼 datetime 변환
    - boolean 값을 0/1로 변환

    Returns:
        pd.DataFrame: 전처리된 티켓 데이터
    """

    # 원천 데이터 로드 (data/raw/ 경로 기준)
    df = read_csv(RAW_DIR / SUPPORT_TICKETS_FILE)

    # 날짜 컬럼을 datetime 타입으로 변환
    # → 이후 시간 계산 및 최신값 추출 등에 활용
    df = to_datetime(df, ["submitted_at", "closed_at"])

    # escalation_flag (True/False)를 1/0으로 변환
    # → 모델 입력 feature로 사용 가능하도록 numeric화
    df["escalation_flag"] = bool_to_int(df["escalation_flag"])

    return df  # 전처리된 데이터 반환


def aggregate_support_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """
    support_tickets 데이터를 account_id 기준으로 집계하는 함수

    핵심 개념:
    - ticket 단위 데이터를 account 단위 feature로 변환 (grain 변환)

    생성되는 주요 feature:
    - total_tickets: 총 티켓 수
    - avg_resolution_time_hours: 평균 해결 시간
    - avg_first_response_time_minutes: 평균 최초 응답 시간
    - avg_satisfaction_score: 평균 만족도
    - escalation_count: escalation 발생 횟수
    - latest_ticket_date: 가장 최근 티켓 날짜
    - escalation_ratio: escalation 비율

    Args:
        df (pd.DataFrame): 전처리된 티켓 데이터

    Returns:
        pd.DataFrame: account 단위 집계 데이터
    """

    # account_id 기준으로 그룹화 후 다양한 통계 집계 수행
    agg = df.groupby("account_id").agg(

        # ticket 개수 (count)
        total_tickets=("ticket_id", "count"),

        # 평균 해결 시간 (mean)
        avg_resolution_time_hours=("resolution_time_hours", "mean"),

        # 평균 최초 응답 시간 (mean)
        avg_first_response_time_minutes=("first_response_time_minutes", "mean"),

        # 평균 고객 만족도 점수 (mean)
        avg_satisfaction_score=("satisfaction_score", "mean"),

        # escalation 발생 횟수 (sum → 1/0 합)
        escalation_count=("escalation_flag", "sum"),

        # 가장 최근 티켓 생성 날짜 (max)
        latest_ticket_date=("submitted_at", "max"),

    ).reset_index()  # index를 컬럼으로 복원

    # escalation 비율 계산
    # total_tickets가 0인 경우 division by zero 방지 위해 clip 사용
    agg["escalation_ratio"] = agg["escalation_count"] / agg["total_tickets"].clip(lower=1)

    return agg  # 집계 결과 반환


def main() -> None:
    """
    전체 실행 함수

    실행 흐름:
    1. 원천 데이터 전처리
    2. account 단위 집계
    3. CSV 파일로 저장
    4. 로그 출력
    """

    # 1. 전처리 수행
    tickets = preprocess_support_tickets()

    # 2. account 단위 집계
    agg = aggregate_support_tickets(tickets)

    # 3. 결과 저장 (data/interim/ 경로)
    save_csv(agg, INTERIM_DIR / "support_tickets_agg.csv")

    # 4. 저장 결과 로그 출력 (shape 확인)
    logger.info("saved support_tickets_agg.csv shape=%s", agg.shape)


# 해당 파일을 직접 실행할 경우 main() 실행
if __name__ == "__main__":
    main()