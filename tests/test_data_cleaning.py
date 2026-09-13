"""Tests for the cleaning pipeline.

These are written as invariants: whatever the source file looks like, the
cleaned output must satisfy them.
"""

import pandas as pd

from src import config, data_cleaning, data_loader


def test_raw_schema_is_valid(raw_df):
    data_loader.validate_schema(raw_df)
    assert len(raw_df) > 0


def test_missing_mask_detects_blanks_and_sentinels():
    series = pd.Series(["a", "", "  ", "N/A", None, "value"])
    mask = data_cleaning.is_missing(series)
    assert mask.tolist() == [False, True, True, True, True, False]


def test_quality_score_is_bounded(raw_df):
    score = data_cleaning.data_quality_score(raw_df)
    assert 0 <= score <= 10


def test_row_count_never_increases(raw_df, cleaned):
    assert len(cleaned) <= len(raw_df)


def test_order_id_is_unique(cleaned):
    assert cleaned[config.ID_COL].is_unique


def test_no_totally_empty_rows(cleaned):
    assert not cleaned.isna().all(axis=1).any()


def test_status_values_are_canonical(cleaned):
    assert set(cleaned["status"]).issubset(set(config.VALID_STATUSES))


def test_country_and_payment_are_canonical(cleaned):
    assert "US" not in set(cleaned["country"])
    assert set(cleaned["payment_method"]).issubset(
        {"Credit Card", "Debit Card", "PayPal", "Apple Pay", "Unknown"}
    )


def test_quantity_is_positive_integer(cleaned):
    qty = cleaned["quantity"]
    assert qty.notna().all()
    assert (qty > 0).all()
    assert qty.dtype == "Int64"


def test_total_price_matches_quantity_times_unit_price(cleaned):
    comparable = cleaned.dropna(subset=["quantity", "unit_price", "total_price"])
    expected = (comparable["quantity"].astype(float) * comparable["unit_price"]).round(2)
    diff = (expected - comparable["total_price"]).abs()
    assert (diff <= config.TOTAL_PRICE_TOLERANCE).all()


def test_order_dates_are_not_in_the_future(cleaned):
    dates = pd.to_datetime(cleaned[config.DATE_COL])
    assert (dates.dropna() <= pd.Timestamp.today().normalize()).all()


def test_emails_are_valid_or_missing(cleaned):
    present = cleaned["email"].dropna()
    assert present.str.match(config.EMAIL_RE).all()


def test_quality_score_improves(raw_df, cleaned, metrics):
    assert metrics["score_after"] >= metrics["score_before"]


def test_cleaning_log_is_complete(cleaning_log):
    assert cleaning_log
    required = {"Phase", "Prompt", "Issue", "Column", "Action", "Reason"}
    assert required.issubset(cleaning_log[0].keys())
    assert all(entry["Reason"] for entry in cleaning_log)
