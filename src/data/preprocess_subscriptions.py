# preprocess_feature_usage.py
from __future__ import annotations

import pandas as pd

from src.config.paths import RAW_DIR, INTERIM_DIR
from src.config.settings import SUBSCRIPTIONS_FILE
from src.utils.helpers import to_datetime, bool_to_int
from src.utils.io import read_csv, save_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_subscriptions() -> pd.DataFrame:
    df = read_csv(RAW_DIR / SUBSCRIPTIONS_FILE)
    df = to_datetime(df, ["start_date", "end_date"])
    for col in ["is_trial", "upgrade_flag", "downgrade_flag", "churn_flag", "auto_renew_flag"]:
        df[col] = bool_to_int(df[col])
    df["plan_tier"] = df["plan_tier"].fillna("Unknown")
    df["billing_frequency"] = df["billing_frequency"].fillna("Unknown")
    return df


def main() -> None:
    df = preprocess_subscriptions()
    save_csv(df, INTERIM_DIR / "subscriptions_clean.csv")
    logger.info("saved subscriptions_clean.csv shape=%s", df.shape)


if __name__ == "__main__":
    main()
