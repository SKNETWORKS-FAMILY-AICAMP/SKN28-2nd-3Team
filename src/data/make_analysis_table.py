# make_analysis_table.py

"""
이 파일은 모델 학습용으로 통합된 base 테이블에 churn 이벤트 정보를 추가 결합하여,
이탈 원인과 재활성화 이력까지 함께 볼 수 있는 분석용 최종 테이블을 생성하는 모듈이다.

주요 역할:
- account 단위로 통합된 base 테이블 로드
- churn 이벤트 데이터 로드 및 날짜 컬럼 파싱
- account별 이탈 횟수, 최근 이탈일, 최근 이탈 사유, 환불 금액, 재활성화 횟수 집계
- base 테이블과 churn 요약 정보를 병합하여 분석용 테이블 생성
- 최종 분석 테이블을 processed 영역에 저장
"""
from __future__ import annotations  # 타입 힌트에서 forward reference 사용 가능하게 설정

import pandas as pd  # 데이터프레임 처리 라이브러리

# 경로 설정 (interim 데이터, 최종 processed 데이터)
from src.config.paths import INTERIM_DIR, PROCESSED_DIR

# CSV 입출력 유틸
from src.utils.io import read_csv, save_csv

# 로깅 설정
from src.utils.logger import get_logger

# 현재 파일 기준 logger 생성
logger = get_logger(__name__)


# 분석용 테이블 생성 함수
def make_analysis_table() -> pd.DataFrame:
    # 기본 통합 테이블 로드 (이미 여러 테이블이 join된 상태)
    base = read_csv(INTERIM_DIR / "merged_base_table.csv")

    # churn 이벤트 데이터 로드 (churn_date는 datetime으로 파싱)
    churn = read_csv(INTERIM_DIR / "churn_events_clean.csv", parse_dates=["churn_date"])

    # account 단위로 churn 관련 정보 집계
    reason_summary = churn.groupby("account_id").agg(
        churn_event_count=("churn_event_id", "count"),          # 이탈 이벤트 발생 횟수
        latest_churn_date=("churn_date", "max"),                # 가장 최근 이탈 발생일
        latest_reason_code=("reason_code", "last"),             # 가장 최근 이탈 사유 코드
        total_refund_amount_usd=("refund_amount_usd", "sum"),   # 환불 총액
        reactivation_count=("is_reactivation", "sum"),          # 재활성화 횟수
    ).reset_index()  # groupby 결과를 DataFrame으로 변환

    # base 테이블과 churn 요약 테이블을 account_id 기준으로 병합
    analysis = base.merge(reason_summary, on="account_id", how="left")

    # 최종 분석 테이블 저장
    save_csv(analysis, PROCESSED_DIR / "analysis_table.csv")

    # 저장 완료 로그 출력 (데이터 shape 포함)
    logger.info("saved analysis_table.csv shape=%s", analysis.shape)

    # 생성된 테이블 반환
    return analysis


# 메인 실행 함수
def main() -> None:
    make_analysis_table()  # 분석 테이블 생성 실행


# 스크립트 실행 시 main 함수 호출
if __name__ == "__main__":
    main()