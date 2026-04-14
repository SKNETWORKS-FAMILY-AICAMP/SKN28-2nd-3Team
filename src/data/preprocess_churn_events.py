# preprocess_churn_events.py
from __future__ import annotations

import pandas as pd

from src.config.paths import RAW_DIR, INTERIM_DIR
from src.config.settings import CHURN_EVENTS_FILE
from src.utils.helpers import to_datetime, bool_to_int
from src.utils.io import read_csv, save_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_churn_events() -> pd.DataFrame:
    df = read_csv(RAW_DIR / CHURN_EVENTS_FILE)
    df = to_datetime(df, ["churn_date"])
    for col in ["preceding_upgrade_flag", "preceding_downgrade_flag", "is_reactivation"]:
        df[col] = bool_to_int(df[col])
    df["reason_code"] = df["reason_code"].fillna("unknown")
    df["feedback_text"] = df["feedback_text"].fillna("no_feedback")
    return df


def main() -> None:
    df = preprocess_churn_events()
    save_csv(df, INTERIM_DIR / "churn_events_clean.csv")
    logger.info("saved churn_events_clean.csv shape=%s", df.shape)


if __name__ == "__main__":
    main()
