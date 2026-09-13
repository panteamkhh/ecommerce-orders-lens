"""One function per business question.

Each function takes the feature-engineered orders table and returns a small,
tidy DataFrame (or dict) that the visualisation layer and the dashboard both
consume. Keeping the numbers here and the styling elsewhere means a chart can
never disagree with the table it sits next to.
"""

from __future__ import annotations

import pandas as pd


def overview_metrics(df: pd.DataFrame) -> dict:
    """Headline KPIs for the whole period."""
    paid = df[df["is_revenue"]]
    total_revenue = float(paid["revenue"].sum())
    recognised_orders = int(len(paid))
    aov = total_revenue / recognised_orders if recognised_orders else 0.0
    return {
        "total_orders": int(len(df)),
        "recognised_orders": recognised_orders,
        "total_revenue": total_revenue,
        "gross_value": float(df["order_value"].sum()),
        "total_units": int(paid["quantity"].sum()),
        "avg_order_value": aov,
        "completion_rate": 100 * recognised_orders / max(len(df), 1),
        "cancelled_orders": int((df["status"] == "Cancelled").sum()),
        "refunded_orders": int((df["status"] == "Refunded").sum()),
        "pending_orders": int((df["status"] == "Pending").sum()),
        "customers": int(df["customer_name"].nunique()),
        "products": int(df["product"].nunique()),
        "countries": int(df["country"].nunique()),
    }


def revenue_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue, orders, units and average price per product, ranked."""
    paid = df[df["is_revenue"]]
    out = (
        paid.groupby("product")
        .agg(
            revenue=("revenue", "sum"),
            orders=("order_id", "nunique"),
            units=("quantity", "sum"),
            avg_unit_price=("unit_price", "mean"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    total = out["revenue"].sum() or 1.0
    out["share_%"] = (100 * out["revenue"] / total).round(1)
    return out


def revenue_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly recognised revenue and order counts, chronologically ordered."""
    paid = df[df["is_revenue"] & df["order_year_month"].notna()]
    out = (
        paid.groupby("order_year_month")
        .agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"))
        .reset_index()
        .sort_values("order_year_month")
        .reset_index(drop=True)
    )
    return out


def monthly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Average revenue per calendar month (seasonality across years)."""
    paid = df[df["is_revenue"] & df["order_month"].notna()]
    out = (
        paid.groupby(["order_month", "order_month_name"])
        .agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"))
        .reset_index()
    )
    years = max(paid["order_year"].dropna().nunique(), 1)
    out["avg_revenue"] = (out["revenue"] / years).round(2)
    return out.sort_values("order_month").reset_index(drop=True)


def revenue_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue contribution per country."""
    paid = df[df["is_revenue"]]
    out = (
        paid.groupby("country")
        .agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"))
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    total = out["revenue"].sum() or 1.0
    out["share_%"] = (100 * out["revenue"] / total).round(1)
    return out


def revenue_by_payment(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue contribution per payment method."""
    paid = df[df["is_revenue"]]
    out = (
        paid.groupby("payment_method")
        .agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"))
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    total = out["revenue"].sum() or 1.0
    out["share_%"] = (100 * out["revenue"] / total).round(1)
    return out


def status_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Order count and value per status."""
    out = (
        df.groupby("status")
        .agg(orders=("order_id", "nunique"), value=("order_value", "sum"))
        .reset_index()
        .sort_values("orders", ascending=False)
        .reset_index(drop=True)
    )
    total = out["orders"].sum() or 1
    out["share_%"] = (100 * out["orders"] / total).round(1)
    return out


def top_customers(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Customers ranked by recognised revenue."""
    paid = df[df["is_revenue"]]
    out = (
        paid.groupby("customer_name")
        .agg(
            revenue=("revenue", "sum"),
            orders=("order_id", "nunique"),
            units=("quantity", "sum"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return out


def cancellation_impact(df: pd.DataFrame) -> pd.DataFrame:
    """How much money is sitting in non-revenue statuses."""
    non_revenue = df[~df["is_revenue"]]
    rows = [
        {
            "status": status,
            "orders": int((non_revenue["status"] == status).sum()),
            "value": float(non_revenue.loc[non_revenue["status"] == status, "order_value"].sum()),
        }
        for status in ["Cancelled", "Refunded", "Pending"]
    ]
    out = pd.DataFrame(rows)
    total = out["value"].sum() or 1.0
    out["share_of_lost_%"] = (100 * out["value"] / total).round(1)
    return out


def revenue_by_note(df: pd.DataFrame) -> pd.DataFrame:
    """Optional add-ons (gift wrap, rush order) vs. plain orders."""
    paid = df[df["is_revenue"]].copy()
    paid["note"] = paid["notes"].replace({"none": "No add-on"})
    out = (
        paid.groupby("note")
        .agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"))
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    return out
