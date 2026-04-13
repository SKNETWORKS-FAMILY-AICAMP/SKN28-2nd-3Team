from __future__ import annotations

import pandas as pd

from src.config.paths import RAW_DIR, DOCS_DIR
from src.config.settings import (
    ACCOUNT_FILE, SUBSCRIPTIONS_FILE, FEATURE_USAGE_FILE,
    SUPPORT_TICKETS_FILE, CHURN_EVENTS_FILE
)
from src.utils.helpers import to_datetime
from src.utils.io import read_csv, save_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)


def summarize_df(name: str, df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_count": df.isna().sum().values,
        "missing_ratio": (df.isna().mean() * 100).round(2).values,
        "nunique": df.nunique(dropna=False).values,
    })
    summary.insert(0, "table_name", name)
    return summary


def main() -> None:
    files = {
        "accounts": ACCOUNT_FILE,
        "subscriptions": SUBSCRIPTIONS_FILE,
        "feature_usage": FEATURE_USAGE_FILE,
        "support_tickets": SUPPORT_TICKETS_FILE,
        "churn_events": CHURN_EVENTS_FILE,
    }
    all_summaries = []
    for name, filename in files.items():
        path = RAW_DIR / filename
        df = read_csv(path)
        if name == "accounts":
            df = to_datetime(df, ["signup_date"])
        elif name == "subscriptions":
            df = to_datetime(df, ["start_date", "end_date"])
        elif name == "feature_usage":
            df = to_datetime(df, ["usage_date"])
        elif name == "support_tickets":
            df = to_datetime(df, ["submitted_at", "closed_at"])
        elif name == "churn_events":
            df = to_datetime(df, ["churn_date"])
        logger.info("%s shape=%s", name, df.shape)
        all_summaries.append(summarize_df(name, df))

    summary_df = pd.concat(all_summaries, ignore_index=True)
    save_csv(summary_df, DOCS_DIR / "raw_data_check_summary.csv")
    logger.info("saved raw data check summary")


if __name__ == "__main__":
    main()
