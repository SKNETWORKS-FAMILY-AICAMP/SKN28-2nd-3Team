from __future__ import annotations

import pandas as pd

from src.config.paths import INTERIM_DIR, PROCESSED_DIR
from src.config.settings import TARGET_COL
from src.features.subscription_change_features import build_subscription_change_features
from src.features.build_features import build_common_features
from src.features.encode_categoricals import one_hot_encode
from src.features.make_dl_dataset import make_dl_ready_table
from src.utils.io import read_csv, save_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)


def infer_reference_date(accounts: pd.DataFrame, subs: pd.DataFrame, usage: pd.DataFrame, tickets: pd.DataFrame) -> pd.Timestamp:
    dates = []
    for col in ["signup_date"]:
        if col in accounts.columns:
            dates.append(accounts[col].max())
    for col in ["start_date", "end_date"]:
        if col in subs.columns:
            dates.append(subs[col].max())
    for col in ["usage_date"]:
        if col in usage.columns:
            dates.append(usage[col].max())
    for col in ["submitted_at", "closed_at", "latest_ticket_date"]:
        if col in tickets.columns:
            dates.append(tickets[col].max())
    return max(pd.to_datetime(d) for d in dates if pd.notna(d))


def make_train_table() -> pd.DataFrame:
    accounts = read_csv(INTERIM_DIR / "accounts_clean.csv", parse_dates=["signup_date"])
    subs = read_csv(INTERIM_DIR / "subscriptions_clean.csv", parse_dates=["start_date", "end_date"])
    usage_agg = read_csv(INTERIM_DIR / "feature_usage_agg.csv", parse_dates=["last_usage_date"])
    tickets_agg = read_csv(INTERIM_DIR / "support_tickets_agg.csv", parse_dates=["latest_ticket_date"])

    reference_date = infer_reference_date(accounts, subs, usage_agg.rename(columns={"last_usage_date":"usage_date"}), tickets_agg.rename(columns={"latest_ticket_date":"submitted_at"}))
    sub_agg = build_subscription_change_features(subs, reference_date)

    merged = accounts.merge(sub_agg, on="account_id", how="left")
    merged = merged.merge(usage_agg, on="account_id", how="left")
    merged = merged.merge(tickets_agg, on="account_id", how="left")

    merged["reference_date"] = reference_date
    merged = build_common_features(merged, reference_date)

    # fill subscription aggregations after merge
    for col in [c for c in merged.columns if c not in ["account_id", "signup_date", "reference_date", TARGET_COL] and merged[c].dtype != "O"]:
        merged[col] = merged[col].fillna(0)

    categorical_cols = [
        "industry", "country", "referral_source",
        "plan_tier", "latest_plan_tier", "latest_billing_frequency",
    ]
    train_table = one_hot_encode(merged, categorical_cols=categorical_cols)

    # remove date columns not needed for direct modeling
    date_cols = [c for c in ["signup_date", "last_usage_date", "latest_ticket_date", "reference_date"] if c in train_table.columns]
    train_table = train_table.drop(columns=date_cols)

    # keep id for analysis, but also create a pure feature table later in split_dataset
    save_csv(train_table, PROCESSED_DIR / "train_table_ml.csv")
    make_dl_ready_table(train_table, PROCESSED_DIR / "train_table_dl.csv")
    save_csv(merged, INTERIM_DIR / "merged_base_table.csv")
    logger.info("saved train_table_ml.csv and train_table_dl.csv")
    return train_table


def main() -> None:
    make_train_table()


if __name__ == "__main__":
    main()
