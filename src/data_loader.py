"""Read the raw e-commerce orders file and validate its *structure*.

Reading everything as text is deliberate: it stops pandas from silently
guessing dtypes and hiding problems such as numbers stored as text, currency
symbols, or mixed date formats. Type coercion is a cleaning decision, not a
loading one.
"""

from __future__ import annotations

import pandas as pd

from . import config

EXPECTED_COLUMNS = [
    "order_id",
    "customer_name",
    "email",
    "order_date",
    "product",
    "quantity",
    "unit_price",
    "total_price",
    "status",
    "country",
    "payment_method",
    "notes",
]


class DataSchemaError(ValueError):
    """Raised when the raw file does not match the expected schema."""


def load_raw(path=config.RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw CSV as all-text, preserving blanks as empty strings."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Fail fast if the expected columns are missing."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise DataSchemaError(f"Missing required columns: {missing}")


def load_and_validate(path=config.RAW_DATA_FILE) -> pd.DataFrame:
    """Convenience wrapper: load then validate."""
    df = load_raw(path)
    validate_schema(df)
    return df
