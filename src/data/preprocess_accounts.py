from __future__ import annotations

import pandas as pd

from src.config.paths import RAW_DIR, INTERIM_DIR
from src.config.settings import ACCOUNT_FILE, TARGET_COL
from src.utils.helpers import to_datetime, bool_to_int
from src.utils.io import read_csv, save_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_accounts() -> pd.DataFrame:
    df = read_csv(RAW_DIR / ACCOUNT_FILE)
    df = to_datetime(df, ["signup_date"])
    bool_cols = ["is_trial", TARGET_COL]
    for col in bool_cols:
        df[col] = bool_to_int(df[col])
    df = df.drop_duplicates(subset=["account_id"]).copy()
    df["industry"] = df["industry"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")
    df["referral_source"] = df["referral_source"].fillna("Unknown")
    if "account_name" in df.columns:
        df = df.drop(columns=["account_name"])
    return df


def main() -> None:
    df = preprocess_accounts()
    save_csv(df, INTERIM_DIR / "accounts_clean.csv")
    logger.info("saved accounts_clean.csv shape=%s", df.shape)


if __name__ == "__main__":
    main()
