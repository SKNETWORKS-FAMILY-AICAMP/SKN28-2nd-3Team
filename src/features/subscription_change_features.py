# subscription_change_features.py

"""
이 파일은 subscription 이력 데이터를 바탕으로
계정(account) 단위의 구독 변화 관련 feature를 생성하는 모듈이다.

주요 역할:
- 구독 시작일·종료일을 기준으로 최근 구독 변경 시점 계산
- 최근 90일 이내 업그레이드/다운그레이드 여부 생성
- 종료된 구독 여부 및 활성 구독 비율 계산
- 계정별 구독 수, 좌석 수, 매출 관련 통계 집계
- 가장 최근 구독 상태(plan, billing, auto-renew, trial) 추출
- 최종적으로 account 단위 구독 변화 feature 테이블 반환
"""
from __future__ import annotations  # 타입 힌트 지연 평가

import pandas as pd  # 데이터 처리 라이브러리


def build_subscription_change_features(sub: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    # 원본 데이터 손상을 막기 위해 복사본 생성
    sub = sub.copy()

    # -----------------------------------
    # 1. 구독 이벤트 기준일 생성
    # -----------------------------------
    # end_date가 있으면 종료일, 없으면 start_date를 이벤트 시점으로 사용
    sub["event_date"] = sub["end_date"].fillna(sub["start_date"])

    # 기준 날짜 대비 구독 변경 후 경과 일수 계산
    sub["days_since_sub_change"] = (reference_date - sub["event_date"]).dt.days

    # -----------------------------------
    # 2. 최근 90일 이내 업그레이드/다운그레이드 여부 생성
    # -----------------------------------
    sub["recent_upgrade_90d"] = (
        (sub["upgrade_flag"] == 1) & (sub["days_since_sub_change"] <= 90)
    ).astype(int)

    sub["recent_downgrade_90d"] = (
        (sub["downgrade_flag"] == 1) & (sub["days_since_sub_change"] <= 90)
    ).astype(int)

    # 종료일이 존재하면 종료된 구독으로 표시
    sub["ended_subscription_flag"] = sub["end_date"].notna().astype(int)

    # -----------------------------------
    # 3. 계정별 가장 최근 구독 정보 추출
    # -----------------------------------
    # account_id별로 날짜순 정렬 후 마지막 구독 레코드의 index 추출
    latest_idx = (
        sub.sort_values(["account_id", "start_date", "end_date"])
        .groupby("account_id")
        .tail(1)
        .index
    )

    # 최근 구독의 핵심 상태 변수만 선택
    latest = sub.loc[latest_idx, [
        "account_id", "plan_tier", "billing_frequency", "auto_renew_flag",
        "is_trial", "days_since_sub_change"
    ]].copy()

    # 최근 상태임을 명확히 하기 위해 컬럼명 변경
    latest = latest.rename(columns={
        "plan_tier": "latest_plan_tier",
        "billing_frequency": "latest_billing_frequency",
        "auto_renew_flag": "latest_auto_renew_flag",
        "is_trial": "latest_is_trial",
        "days_since_sub_change": "days_since_last_subscription_change",
    })

    # -----------------------------------
    # 4. account 단위 기본 집계
    # -----------------------------------
    agg = sub.groupby("account_id").agg(
        total_subscriptions=("subscription_id", "count"),                  # 전체 구독 수
        active_subscriptions=("end_date", lambda s: s.isna().sum()),       # 종료일이 없는 활성 구독 수
        avg_subscription_duration_days=("start_date", lambda s: 0),        # 이후 별도 계산값으로 대체 예정
        avg_sub_seats=("seats", "mean"),                                   # 평균 좌석 수
        max_sub_seats=("seats", "max"),                                    # 최대 좌석 수
        avg_mrr_amount=("mrr_amount", "mean"),                             # 평균 월 반복 매출
        max_mrr_amount=("mrr_amount", "max"),                              # 최대 월 반복 매출
        avg_arr_amount=("arr_amount", "mean"),                             # 평균 연 반복 매출
        total_arr_amount=("arr_amount", "sum"),                            # 총 연 반복 매출
        trial_subscription_count=("is_trial", "sum"),                      # trial 구독 수
        auto_renew_count=("auto_renew_flag", "sum"),                       # 자동갱신 구독 수
        upgrade_count=("upgrade_flag", "sum"),                             # 업그레이드 횟수
        downgrade_count=("downgrade_flag", "sum"),                         # 다운그레이드 횟수
        recent_upgrade_90d=("recent_upgrade_90d", "sum"),                  # 최근 90일 업그레이드 횟수
        recent_downgrade_90d=("recent_downgrade_90d", "sum"),              # 최근 90일 다운그레이드 횟수
        ended_subscriptions_count=("ended_subscription_flag", "sum"),      # 종료된 구독 수
    ).reset_index()

    # -----------------------------------
    # 5. 구독 지속 기간(duration) 계산
    # -----------------------------------
    # 종료일이 없으면 아직 활성 상태이므로 reference_date를 종료 시점으로 간주
    filled_end = sub["end_date"].fillna(reference_date)

    # 구독 시작일부터 종료일까지의 기간 계산
    duration = (filled_end - sub["start_date"]).dt.days.clip(lower=0)

    # account 단위 평균/최대 구독 기간 집계
    duration_agg = pd.DataFrame({
        "account_id": sub["account_id"],
        "subscription_duration_days": duration,
    }).groupby("account_id").agg(
        avg_subscription_duration_days=("subscription_duration_days", "mean"),
        max_subscription_duration_days=("subscription_duration_days", "max"),
    ).reset_index()

    # -----------------------------------
    # 6. 기본 집계 + duration 집계 병합
    # -----------------------------------
    out = agg.merge(duration_agg, on="account_id", how="left", suffixes=("", "_durfix"))

    # 아래 루프는 현재 별도 동작은 하지 않지만,
    # 후속 컬럼 처리 대상으로 의도된 자리표시자 역할
    for col in ["avg_subscription_duration_days_durfix", "max_subscription_duration_days"]:
        pass

    # 기존 placeholder 컬럼을 실제 duration 계산값으로 대체
    if "avg_subscription_duration_days_durfix" in out.columns:
        out["avg_subscription_duration_days"] = out["avg_subscription_duration_days_durfix"]
        out = out.drop(columns=["avg_subscription_duration_days_durfix"])

    # -----------------------------------
    # 7. 비율형 파생 변수 생성
    # -----------------------------------
    # 전체 구독 대비 활성 구독 비율
    out["active_subscription_ratio"] = out["active_subscriptions"] / out["total_subscriptions"].clip(lower=1)

    # 전체 구독 대비 자동 갱신 비율
    out["auto_renew_ratio"] = out["auto_renew_count"] / out["total_subscriptions"].clip(lower=1)

    # -----------------------------------
    # 8. 최근 구독 상태 정보 병합
    # -----------------------------------
    out = out.merge(latest, on="account_id", how="left")

    # 최종 feature 테이블 반환
    return out