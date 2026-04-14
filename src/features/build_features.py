# build_features.py
from __future__ import annotations

import pandas as pd
import numpy as np

from src.features.missing_flags import add_missing_flags


def build_common_features(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    if "signup_date" in out.columns:
        out["account_age_days"] = (reference_date - out["signup_date"]).dt.days.clip(lower=0)

    # impute continuous support metrics with median while preserving missing flags
    out = add_missing_flags(out)
    for col in ["avg_resolution_time_hours", "avg_first_response_time_minutes", "avg_satisfaction_score"]:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].median())

    out["total_usage_count"] = out.get("total_usage_count", 0).fillna(0)
    out["total_usage_duration_secs"] = out.get("total_usage_duration_secs", 0).fillna(0)
    out["total_error_count"] = out.get("total_error_count", 0).fillna(0)
    out["unique_feature_count"] = out.get("unique_feature_count", 0).fillna(0)
    out["days_since_last_usage"] = out.get("days_since_last_usage", 999).fillna(999)
    out["total_tickets"] = out.get("total_tickets", 0).fillna(0)
    out["escalation_ratio"] = out.get("escalation_ratio", 0).fillna(0)

    out["usage_per_subscription"] = out["total_usage_count"] / out["total_subscriptions"].clip(lower=1)
    out["ticket_per_subscription"] = out["total_tickets"] / out["total_subscriptions"].clip(lower=1)
    out["error_per_subscription"] = out["total_error_count"] / out["total_subscriptions"].clip(lower=1)
    out["error_rate"] = out["total_error_count"] / out["total_usage_count"].clip(lower=1)

    out["health_score"] = (
        np.log1p(out["total_usage_count"])
        + np.log1p(out["unique_feature_count"])
        + 0.25 * out["avg_satisfaction_score"]
        - 5 * out["error_rate"]
        - 0.25 * out["ticket_per_subscription"]
        - 0.5 * out["escalation_ratio"]
    )

    return out
