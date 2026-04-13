from __future__ import annotations

import pandas as pd

from src.config.paths import RAW_DIR, INTERIM_DIR
from src.config.settings import SUPPORT_TICKETS_FILE
from src.utils.helpers import to_datetime, bool_to_int
from src.utils.io import read_csv, save_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_support_tickets() -> pd.DataFrame:
    df = read_csv(RAW_DIR / SUPPORT_TICKETS_FILE)
    df = to_datetime(df, ["submitted_at", "closed_at"])
    df["escalation_flag"] = bool_to_int(df["escalation_flag"])
    return df


def aggregate_support_tickets(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("account_id").agg(
        total_tickets=("ticket_id", "count"),
        avg_resolution_time_hours=("resolution_time_hours", "mean"),
        avg_first_response_time_minutes=("first_response_time_minutes", "mean"),
        avg_satisfaction_score=("satisfaction_score", "mean"),
        escalation_count=("escalation_flag", "sum"),
        latest_ticket_date=("submitted_at", "max"),
    ).reset_index()
    agg["escalation_ratio"] = agg["escalation_count"] / agg["total_tickets"].clip(lower=1)
    return agg


def main() -> None:
    tickets = preprocess_support_tickets()
    agg = aggregate_support_tickets(tickets)
    save_csv(agg, INTERIM_DIR / "support_tickets_agg.csv")
    logger.info("saved support_tickets_agg.csv shape=%s", agg.shape)


if __name__ == "__main__":
    main()
