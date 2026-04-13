from __future__ import annotations

import pandas as pd

from src.config.paths import RAW_DIR, INTERIM_DIR
from src.config.settings import FEATURE_USAGE_FILE
from src.utils.helpers import to_datetime, bool_to_int
from src.utils.io import read_csv, save_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_feature_usage() -> pd.DataFrame:
    df = read_csv(RAW_DIR / FEATURE_USAGE_FILE)
    df = to_datetime(df, ["usage_date"])
    if "is_beta_feature" in df.columns:
        df["is_beta_feature"] = bool_to_int(df["is_beta_feature"])
    return df


def aggregate_feature_usage(df: pd.DataFrame, subscriptions: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    merged = df.merge(
        subscriptions[["subscription_id", "account_id"]],
        on="subscription_id",
        how="left",
    )
    merged["days_since_usage"] = (reference_date - merged["usage_date"]).dt.days
    agg = merged.groupby("account_id").agg(
        total_usage_count=("usage_count", "sum"),
        total_usage_duration_secs=("usage_duration_secs", "sum"),
        total_error_count=("error_count", "sum"),
        unique_feature_count=("feature_name", "nunique"),
        beta_feature_usage_count=("is_beta_feature", "sum"),
        last_usage_date=("usage_date", "max"),
        days_since_last_usage=("days_since_usage", "min"),
    ).reset_index()
    return agg


def main() -> None:
    usage = preprocess_feature_usage()
    subs = read_csv(INTERIM_DIR / "subscriptions_clean.csv")
    subs["start_date"] = pd.to_datetime(subs["start_date"], errors="coerce")
    subs["end_date"] = pd.to_datetime(subs["end_date"], errors="coerce")
    reference_date = max(
        usage["usage_date"].max(),
        subs["end_date"].max(),
        subs["start_date"].max(),
    )
    agg = aggregate_feature_usage(usage, subs, reference_date)
    save_csv(agg, INTERIM_DIR / "feature_usage_agg.csv")
    logger.info("saved feature_usage_agg.csv shape=%s", agg.shape)


if __name__ == "__main__":
    main()
