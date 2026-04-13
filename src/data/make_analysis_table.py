from __future__ import annotations

import pandas as pd

from src.config.paths import INTERIM_DIR, PROCESSED_DIR
from src.utils.io import read_csv, save_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_analysis_table() -> pd.DataFrame:
    base = read_csv(INTERIM_DIR / "merged_base_table.csv")
    churn = read_csv(INTERIM_DIR / "churn_events_clean.csv", parse_dates=["churn_date"])
    reason_summary = churn.groupby("account_id").agg(
        churn_event_count=("churn_event_id", "count"),
        latest_churn_date=("churn_date", "max"),
        latest_reason_code=("reason_code", "last"),
        total_refund_amount_usd=("refund_amount_usd", "sum"),
        reactivation_count=("is_reactivation", "sum"),
    ).reset_index()

    analysis = base.merge(reason_summary, on="account_id", how="left")
    save_csv(analysis, PROCESSED_DIR / "analysis_table.csv")
    logger.info("saved analysis_table.csv shape=%s", analysis.shape)
    return analysis


def main() -> None:
    make_analysis_table()


if __name__ == "__main__":
    main()
