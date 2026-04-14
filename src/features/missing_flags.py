# missing_flags.py
from __future__ import annotations

import pandas as pd
from src.config.settings import MISSING_FLAG_COLUMNS


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in MISSING_FLAG_COLUMNS:
        if col in out.columns:
            out[f"{col}_missing_flag"] = out[col].isna().astype(int)
    out["is_inactive_user"] = (out.get("total_usage_count", 0) == 0).astype(int)
    out["has_no_ticket_history"] = (out.get("total_tickets", 0) == 0).astype(int)
    return out
