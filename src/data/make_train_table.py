# make_train_table.py

"""
이 파일은 전처리 및 집계가 완료된 여러 테이블을 account 단위로 통합하여,
머신러닝(ML)과 딥러닝(DL) 학습에 사용할 최종 학습용 테이블을 생성하는 모듈이다.

주요 역할:
- accounts, subscriptions, feature usage, support tickets 데이터를 로드
- 전체 데이터에서 가장 최근 시점을 기준 날짜(reference_date)로 추론
- 구독 변화 이력 및 공통 파생 변수(feature) 생성
- 여러 테이블을 account_id 기준으로 병합하여 단일 학습 테이블 구성
- 숫자형 결측치 처리 및 범주형 변수 인코딩 수행
- 모델 입력에 불필요한 날짜 컬럼 제거
- ML용 / DL용 학습 테이블과 분석용 base 테이블 저장
"""
from __future__ import annotations  # 타입 힌트에서 forward reference 사용 가능하게 설정

import pandas as pd  # 데이터프레임 처리 라이브러리

# 경로 설정
from src.config.paths import INTERIM_DIR, PROCESSED_DIR

# 타겟 변수명 설정
from src.config.settings import TARGET_COL

# feature 생성 관련 모듈
from src.features.subscription_change_features import build_subscription_change_features  # 구독 변화 feature 생성
from src.features.build_features import build_common_features  # 공통 feature 생성
from src.features.encode_categoricals import one_hot_encode  # 범주형 변수 one-hot encoding
from src.features.make_dl_dataset import make_dl_ready_table  # DL 입력용 데이터 생성

# 입출력 및 로깅
from src.utils.io import read_csv, save_csv
from src.utils.logger import get_logger

# logger 생성
logger = get_logger(__name__)


# -----------------------------------
# 기준 날짜(reference_date) 자동 추론 함수
# -----------------------------------
def infer_reference_date(accounts: pd.DataFrame, subs: pd.DataFrame, usage: pd.DataFrame, tickets: pd.DataFrame) -> pd.Timestamp:
    dates = []  # 모든 날짜를 모을 리스트

    # accounts 테이블에서 signup_date 최대값
    for col in ["signup_date"]:
        if col in accounts.columns:
            dates.append(accounts[col].max())

    # subscriptions 테이블에서 start/end 날짜 최대값
    for col in ["start_date", "end_date"]:
        if col in subs.columns:
            dates.append(subs[col].max())

    # usage 데이터에서 usage_date 최대값
    for col in ["usage_date"]:
        if col in usage.columns:
            dates.append(usage[col].max())

    # tickets 데이터에서 주요 날짜 컬럼 최대값
    for col in ["submitted_at", "closed_at", "latest_ticket_date"]:
        if col in tickets.columns:
            dates.append(tickets[col].max())

    # 가장 최신 날짜를 기준(reference_date)으로 반환
    return max(pd.to_datetime(d) for d in dates if pd.notna(d))


# -----------------------------------
# 학습용 테이블 생성
# -----------------------------------
def make_train_table() -> pd.DataFrame:

    # 각 전처리된 테이블 로드
    accounts = read_csv(INTERIM_DIR / "accounts_clean.csv", parse_dates=["signup_date"])
    subs = read_csv(INTERIM_DIR / "subscriptions_clean.csv", parse_dates=["start_date", "end_date"])
    usage_agg = read_csv(INTERIM_DIR / "feature_usage_agg.csv", parse_dates=["last_usage_date"])
    tickets_agg = read_csv(INTERIM_DIR / "support_tickets_agg.csv", parse_dates=["latest_ticket_date"])

    # 기준 날짜 생성 (전체 데이터 중 가장 최근 시점)
    reference_date = infer_reference_date(
        accounts,
        subs,
        usage_agg.rename(columns={"last_usage_date": "usage_date"}),  # 컬럼명 맞추기
        tickets_agg.rename(columns={"latest_ticket_date": "submitted_at"}),  # 컬럼명 맞추기
    )

    # 구독 변화 기반 feature 생성
    sub_agg = build_subscription_change_features(subs, reference_date)

    # -----------------------------------
    # 테이블 병합 (account 단위 통합)
    # -----------------------------------
    merged = accounts.merge(sub_agg, on="account_id", how="left")  # 구독 feature 병합
    merged = merged.merge(usage_agg, on="account_id", how="left")  # 사용량 feature 병합
    merged = merged.merge(tickets_agg, on="account_id", how="left")  # 티켓 feature 병합

    # 기준 날짜 컬럼 추가
    merged["reference_date"] = reference_date

    # 공통 feature 생성 (파생 변수)
    merged = build_common_features(merged, reference_date)

    # -----------------------------------
    # 결측치 처리
    # -----------------------------------
    # 숫자형 컬럼에 대해 결측치를 0으로 채움
    for col in [
        c for c in merged.columns
        if c not in ["account_id", "signup_date", "reference_date", TARGET_COL]
        and merged[c].dtype != "O"
    ]:
        merged[col] = merged[col].fillna(0)

    # -----------------------------------
    # 범주형 변수 인코딩
    # -----------------------------------
    categorical_cols = [
        "industry", "country", "referral_source",
        "plan_tier", "latest_plan_tier", "latest_billing_frequency",
    ]

    # one-hot encoding 수행
    train_table = one_hot_encode(merged, categorical_cols=categorical_cols)

    # -----------------------------------
    # 날짜 컬럼 제거 (모델 입력에서 제외)
    # -----------------------------------
    date_cols = [
        c for c in ["signup_date", "last_usage_date", "latest_ticket_date", "reference_date"]
        if c in train_table.columns
    ]
    train_table = train_table.drop(columns=date_cols)

    # -----------------------------------
    # 결과 저장
    # -----------------------------------

    # ML 모델용 테이블 저장
    save_csv(train_table, PROCESSED_DIR / "train_table_ml.csv")

    # DL 모델용 테이블 생성 및 저장
    make_dl_ready_table(train_table, PROCESSED_DIR / "train_table_dl.csv")

    # 분석용 base 테이블도 함께 저장
    save_csv(merged, INTERIM_DIR / "merged_base_table.csv")

    # 로그 출력
    logger.info("saved train_table_ml.csv and train_table_dl.csv")

    # 최종 테이블 반환
    return train_table


# 메인 실행 함수
def main() -> None:
    make_train_table()


# 스크립트 실행 시 main 호출
if __name__ == "__main__":
    main()