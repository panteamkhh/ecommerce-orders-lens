"""Tests for the derived analysis fields."""

import pandas as pd

from src import feature_engineering


def test_expected_columns_are_added(featured):
    expected = {
        "order_year",
        "order_quarter",
        "order_month",
        "order_month_name",
        "order_year_month",
        "order_day_of_week",
        "is_revenue",
        "revenue",
        "order_value",
        "has_note",
    }
    assert expected.issubset(featured.columns)


def test_revenue_is_zero_for_non_revenue_statuses(featured):
    non_revenue = featured[~featured["is_revenue"]]
    assert (non_revenue["revenue"] == 0).all()


def test_revenue_equals_total_price_for_revenue_statuses(featured):
    revenue = featured[featured["is_revenue"]]
    pd.testing.assert_series_equal(revenue["revenue"], revenue["total_price"], check_names=False)


def test_order_year_month_is_year_dash_month(featured):
    values = featured["order_year_month"].dropna()
    assert values.str.match(r"^\d{4}-\d{2}$").all()


def test_summarise_features_has_expected_keys(featured):
    summary = feature_engineering.summarise_features(featured)
    assert set(summary) == {"date_min", "date_max", "years", "products", "countries"}
    assert summary["years"]
