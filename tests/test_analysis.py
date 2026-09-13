"""Tests for the business-question functions."""

import pytest

from src import analysis


def test_overview_metrics_keys(featured):
    kpis = analysis.overview_metrics(featured)
    expected = {
        "total_orders",
        "recognised_orders",
        "total_revenue",
        "total_units",
        "avg_order_value",
        "completion_rate",
        "customers",
        "products",
        "countries",
    }
    assert expected.issubset(kpis)
    assert kpis["recognised_orders"] <= kpis["total_orders"]
    assert kpis["total_revenue"] > 0
    assert 0 <= kpis["completion_rate"] <= 100


def test_revenue_by_product_shares_sum_to_100(featured):
    out = analysis.revenue_by_product(featured)
    assert not out.empty
    assert out["revenue"].is_monotonic_decreasing
    assert out["share_%"].sum() == pytest.approx(100, abs=0.5)


def test_revenue_trend_is_chronological(featured):
    out = analysis.revenue_trend(featured)
    assert out["order_year_month"].is_monotonic_increasing
    assert (out["revenue"] >= 0).all()


def test_revenue_by_country_shares_sum_to_100(featured):
    out = analysis.revenue_by_country(featured)
    assert out["share_%"].sum() == pytest.approx(100, abs=0.5)


def test_revenue_by_payment_shares_sum_to_100(featured):
    out = analysis.revenue_by_payment(featured)
    assert out["share_%"].sum() == pytest.approx(100, abs=0.5)


def test_status_breakdown_counts_every_order(featured):
    out = analysis.status_breakdown(featured)
    assert int(out["orders"].sum()) == len(featured)


def test_top_customers_is_limited(featured):
    out = analysis.top_customers(featured, top_n=5)
    assert len(out) <= 5
    assert out["revenue"].is_monotonic_decreasing


def test_cancellation_impact_covers_non_revenue_statuses(featured):
    out = analysis.cancellation_impact(featured)
    assert set(out["status"]) == {"Cancelled", "Refunded", "Pending"}
    assert (out["value"] >= 0).all()
