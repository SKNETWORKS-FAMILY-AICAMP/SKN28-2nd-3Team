# build_features.py

"""
이 파일은 여러 원천/집계 테이블을 병합한 base 데이터에서
모델링에 활용할 공통 파생 변수(feature)를 생성하는 모듈이다.

주요 역할:
- 기준 시점(reference_date) 기준 계정 연령(account age) 계산
- 결측 여부를 보존하기 위한 missing flag 생성
- 주요 지원(support) 관련 연속형 변수의 결측값 보정
- 사용량, 오류, 티켓 관련 기본 수치형 변수 보완
- subscription 대비 사용량·오류·티켓 비율 파생
- 서비스 이용 건전성을 나타내는 health score 생성
"""
from __future__ import annotations  # 타입 힌트 지연 평가

import pandas as pd  # 데이터 처리 라이브러리
import numpy as np  # 수치 계산 라이브러리

from src.features.missing_flags import add_missing_flags  # 결측 여부 flag 생성 함수


def build_common_features(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    # 원본 데이터 손상을 막기 위해 복사본 생성
    out = df.copy()

    # -----------------------------------
    # 1. 계정 연령(account age) 계산
    # -----------------------------------
    # signup_date가 있으면 기준 시점 대비 가입 후 경과일 계산
    if "signup_date" in out.columns:
        out["account_age_days"] = (reference_date - out["signup_date"]).dt.days.clip(lower=0)

    # -----------------------------------
    # 2. 결측 여부 flag 생성
    # -----------------------------------
    # 연속형 변수의 결측값을 대체하기 전에, 원래 결측이었는지 표시하는 flag 추가
    out = add_missing_flags(out)

    # -----------------------------------
    # 3. support 관련 연속형 변수 결측값 보정
    # -----------------------------------
    # 중앙값(median)으로 대체하여 극단값 영향을 줄임
    for col in ["avg_resolution_time_hours", "avg_first_response_time_minutes", "avg_satisfaction_score"]:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].median())

    # -----------------------------------
    # 4. 주요 수치형 컬럼 기본값 보정
    # -----------------------------------
    # 컬럼이 없거나 결측이면 기본값으로 채워 계산 안정성 확보
    out["total_usage_count"] = out.get("total_usage_count", 0).fillna(0)
    out["total_usage_duration_secs"] = out.get("total_usage_duration_secs", 0).fillna(0)
    out["total_error_count"] = out.get("total_error_count", 0).fillna(0)
    out["unique_feature_count"] = out.get("unique_feature_count", 0).fillna(0)
    out["days_since_last_usage"] = out.get("days_since_last_usage", 999).fillna(999)
    out["total_tickets"] = out.get("total_tickets", 0).fillna(0)
    out["escalation_ratio"] = out.get("escalation_ratio", 0).fillna(0)

    # -----------------------------------
    # 5. subscription 기준 비율 파생 변수 생성
    # -----------------------------------
    # total_subscriptions가 0일 수 있으므로 clip(lower=1)로 0 나눗셈 방지
    out["usage_per_subscription"] = out["total_usage_count"] / out["total_subscriptions"].clip(lower=1)
    out["ticket_per_subscription"] = out["total_tickets"] / out["total_subscriptions"].clip(lower=1)
    out["error_per_subscription"] = out["total_error_count"] / out["total_subscriptions"].clip(lower=1)

    # 전체 사용량 대비 오류 비율 계산
    out["error_rate"] = out["total_error_count"] / out["total_usage_count"].clip(lower=1)

    # -----------------------------------
    # 6. health score 생성
    # -----------------------------------
    # 사용량과 기능 다양성, 만족도는 가산
    # 오류율, 티켓 비율, escalation 비율은 감산
    out["health_score"] = (
        np.log1p(out["total_usage_count"])
        + np.log1p(out["unique_feature_count"])
        + 0.25 * out["avg_satisfaction_score"]
        - 5 * out["error_rate"]
        - 0.25 * out["ticket_per_subscription"]
        - 0.5 * out["escalation_ratio"]
    )

    # 파생 변수가 추가된 데이터 반환
    return out