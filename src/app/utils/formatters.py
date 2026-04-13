from __future__ import annotations


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_int(value) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)
