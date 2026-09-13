"""Derive the analysis fields used by every report and chart."""

from __future__ import annotations

import pandas as pd

from . import config


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, revenue and flag columns to a cleaned orders table."""
    out = df.copy()

    dates = pd.to_datetime(out[config.DATE_COL], errors="coerce")
    out["order_year"] = dates.dt.year.astype("Int64")
    out["order_quarter"] = dates.dt.quarter.astype("Int64")
    out["order_month"] = dates.dt.month.astype("Int64")
    out["order_month_name"] = dates.dt.strftime("%b")
    out["order_year_month"] = dates.dt.strftime("%Y-%m")
    out["order_day_of_week"] = dates.dt.day_name()

    out["is_revenue"] = out["status"].isin(config.REVENUE_STATUSES)
    # Recognised revenue: only orders that turned into money for the shop.
    out["revenue"] = out["total_price"].where(out["is_revenue"], 0.0)
    # Gross value regardless of status, useful for cancellation impact.
    out["order_value"] = out["total_price"]

    out["has_note"] = out["notes"].ne("none")
    return out


def summarise_features(df: pd.DataFrame) -> dict:
    """Small dict used in logs / dashboard header."""
    return {
        "date_min": str(pd.to_datetime(df[config.DATE_COL]).min().date()),
        "date_max": str(pd.to_datetime(df[config.DATE_COL]).max().date()),
        "years": sorted(df["order_year"].dropna().astype(int).unique().tolist()),
        "products": int(df["product"].nunique()),
        "countries": int(df["country"].nunique()),
    }
