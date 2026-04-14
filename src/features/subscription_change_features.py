# subscription_change_features.py
from __future__ import annotations

import pandas as pd


def build_subscription_change_features(sub: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    sub = sub.copy()
    sub["event_date"] = sub["end_date"].fillna(sub["start_date"])
    sub["days_since_sub_change"] = (reference_date - sub["event_date"]).dt.days

    sub["recent_upgrade_90d"] = ((sub["upgrade_flag"] == 1) & (sub["days_since_sub_change"] <= 90)).astype(int)
    sub["recent_downgrade_90d"] = ((sub["downgrade_flag"] == 1) & (sub["days_since_sub_change"] <= 90)).astype(int)
    sub["ended_subscription_flag"] = sub["end_date"].notna().astype(int)

    latest_idx = sub.sort_values(["account_id", "start_date", "end_date"]).groupby("account_id").tail(1).index
    latest = sub.loc[latest_idx, [
        "account_id", "plan_tier", "billing_frequency", "auto_renew_flag",
        "is_trial", "days_since_sub_change"
    ]].copy()
    latest = latest.rename(columns={
        "plan_tier": "latest_plan_tier",
        "billing_frequency": "latest_billing_frequency",
        "auto_renew_flag": "latest_auto_renew_flag",
        "is_trial": "latest_is_trial",
        "days_since_sub_change": "days_since_last_subscription_change",
    })

    agg = sub.groupby("account_id").agg(
        total_subscriptions=("subscription_id", "count"),
        active_subscriptions=("end_date", lambda s: s.isna().sum()),
        avg_subscription_duration_days=("start_date", lambda s: 0),
        avg_sub_seats=("seats", "mean"),
        max_sub_seats=("seats", "max"),
        avg_mrr_amount=("mrr_amount", "mean"),
        max_mrr_amount=("mrr_amount", "max"),
        avg_arr_amount=("arr_amount", "mean"),
        total_arr_amount=("arr_amount", "sum"),
        trial_subscription_count=("is_trial", "sum"),
        auto_renew_count=("auto_renew_flag", "sum"),
        upgrade_count=("upgrade_flag", "sum"),
        downgrade_count=("downgrade_flag", "sum"),
        recent_upgrade_90d=("recent_upgrade_90d", "sum"),
        recent_downgrade_90d=("recent_downgrade_90d", "sum"),
        ended_subscriptions_count=("ended_subscription_flag", "sum"),
    ).reset_index()

    filled_end = sub["end_date"].fillna(reference_date)
    duration = (filled_end - sub["start_date"]).dt.days.clip(lower=0)
    duration_agg = pd.DataFrame({
        "account_id": sub["account_id"],
        "subscription_duration_days": duration,
    }).groupby("account_id").agg(
        avg_subscription_duration_days=("subscription_duration_days", "mean"),
        max_subscription_duration_days=("subscription_duration_days", "max"),
    ).reset_index()

    out = agg.merge(duration_agg, on="account_id", how="left", suffixes=("", "_durfix"))
    for col in ["avg_subscription_duration_days_durfix", "max_subscription_duration_days"]:
        pass
    if "avg_subscription_duration_days_durfix" in out.columns:
        out["avg_subscription_duration_days"] = out["avg_subscription_duration_days_durfix"]
        out = out.drop(columns=["avg_subscription_duration_days_durfix"])
    out["active_subscription_ratio"] = out["active_subscriptions"] / out["total_subscriptions"].clip(lower=1)
    out["auto_renew_ratio"] = out["auto_renew_count"] / out["total_subscriptions"].clip(lower=1)
    out = out.merge(latest, on="account_id", how="left")
    return out
