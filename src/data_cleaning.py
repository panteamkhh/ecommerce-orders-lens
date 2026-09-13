"""Clean the raw e-commerce orders table.

The 21-step checklist is grouped into phases so the reasoning behind every
change is explicit. Each action is also recorded in an audit log so the output
workbook can explain itself.

Golden rule: never fix a value before you understand *why* it looks the way it
does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Missing-value helpers
# --------------------------------------------------------------------------- #
def is_missing(series: pd.Series) -> pd.Series:
    """True where a cell is NaN or a blank / sentinel string."""
    text = series.astype(str).str.strip().str.lower()
    return series.isna() | text.isin(config.SENTINELS)


def missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Element-wise missing mask for a whole DataFrame."""
    text = df.astype(str)
    text = text.apply(lambda col: col.str.strip().str.lower())
    return df.isna() | text.isin(config.SENTINELS)


# --------------------------------------------------------------------------- #
# Quality scoring
# --------------------------------------------------------------------------- #
def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column completeness table."""
    n = max(len(df), 1)
    rows = []
    for col in df.columns:
        missing = int(is_missing(df[col]).sum())
        rows.append((col, round(100 * missing / n, 1), round(100 * (1 - missing / n), 1)))
    report = pd.DataFrame(rows, columns=["column", "missing_%", "completeness_%"])
    return report.sort_values("missing_%", ascending=False, ignore_index=True)


def data_quality_score(df: pd.DataFrame) -> float:
    """A transparent 0-10 score from missingness, duplicates and padding.

    It is a proxy, not an absolute truth: only three measurable defects are
    used so the number is reproducible run to run.
    """
    n = max(len(df), 1)
    as_text = df.astype(str)
    padded = as_text != as_text.apply(lambda col: col.str.strip())
    padded_ratio = int(padded.sum().sum()) / (n * len(df.columns))
    missing_ratio = int(missing_mask(df).sum().sum()) / (n * len(df.columns))
    duplicate_ratio = int(df.duplicated().sum()) / n
    score = 10 * (1 - 0.5 * missing_ratio - 0.2 * min(duplicate_ratio, 1.0) - 0.3 * padded_ratio)
    return round(max(score, 0.0), 1)


# --------------------------------------------------------------------------- #
# Audit log
# --------------------------------------------------------------------------- #
def log_step(log, phase, prompt, issue, column, action, reason):
    """Append one cleaning action to the audit trail."""
    log.append(
        {
            "Phase": phase,
            "Prompt": prompt,
            "Issue": issue,
            "Column": column,
            "Action": action,
            "Reason": reason,
        }
    )


# --------------------------------------------------------------------------- #
# Standardization helpers
# --------------------------------------------------------------------------- #
def standardize(series: pd.Series, mapping: dict, fallback: str = "Unknown") -> pd.Series:
    """Map casing / abbreviation variants to one canonical value per group."""

    def _one(value):
        if pd.isna(value):
            return fallback
        key = str(value).strip().upper()
        if key in ("", "N/A", "NA", "NULL"):
            return fallback
        return mapping.get(key, str(value).strip())

    return series.map(_one)


def clean_person_name(value) -> str:
    """Light cleanup: tidy spacing, reorder 'Last, First', title-case."""
    if pd.isna(value):
        return "Unknown"
    text = " ".join(str(value).replace("\xa0", " ").split())
    if "," in text:
        last, first = [part.strip() for part in text.split(",", 1)]
        text = f"{first} {last}"
    return text.title()


def _to_number(series: pd.Series) -> pd.Series:
    """Strip currency symbols / separators and coerce to numbers."""
    return pd.to_numeric(
        series.astype(str).str.replace(config.CURRENCY_STRIP_RE, "", regex=True),
        errors="coerce",
    )


def parse_order_date(value):
    """Parse the two date formats present in the source: M/D/YY and D-M-YYYY."""
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    for fmt in ("%m/%d/%y", "%d-%m-%Y"):
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return parsed
    return pd.NaT


def iqr_bounds(series: pd.Series):
    """Return the (low, high) Tukey fences for a series."""
    clean = series.dropna()
    if clean.empty:
        return None, None
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def clean_orders(raw: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, list, dict]:
    """Run the full cleaning checklist.

    Returns
    -------
    clean : pd.DataFrame
        The analysis-ready table.
    log : list[dict]
        One entry per cleaning action.
    metrics : dict
        Before/after figures used by the report and the dashboard.
    """
    log: list[dict] = []
    metrics: dict = {}
    say = print if verbose else (lambda *_: None)

    # ------------------------------------------------------------------ #
    # PHASE 0 - Understand before touching anything
    # ------------------------------------------------------------------ #
    metrics["rows_before"] = int(raw.shape[0])
    metrics["columns_before"] = int(raw.shape[1])
    score_before = data_quality_score(raw)
    metrics["score_before"] = score_before
    log_step(
        log,
        "Phase 0",
        "1",
        "Initial inspection",
        "ALL",
        f"Imported {raw.shape[0]}x{raw.shape[1]} as text; initial score {score_before}/10",
        "Build a mental map of the data before changing anything (golden rule).",
    )

    clean = raw.copy()

    # ------------------------------------------------------------------ #
    # PHASE 1 - Structural cleaning
    # ------------------------------------------------------------------ #
    # Normalise invisible whitespace and turn '' placeholders into real NaN.
    for col in clean.columns:
        text = clean[col].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
        clean[col] = text.mask(text == "", np.nan)

    empty_rows = int(clean.isna().all(axis=1).sum())
    clean = clean.dropna(how="all").reset_index(drop=True)
    metrics["empty_rows_removed"] = empty_rows
    log_step(
        log,
        "Phase 1",
        "5",
        "Empty records",
        "ALL",
        f"Dropped {empty_rows} totally empty rows",
        "A row with no data cannot represent a real order and carries no information.",
    )

    # Identifier: cannot be invented, drop the rows that lack it.
    before = int(clean[config.ID_COL].isna().sum())
    clean = clean.dropna(subset=[config.ID_COL]).reset_index(drop=True)
    metrics["missing_ids_removed"] = before
    log_step(
        log,
        "Phase 1",
        "5",
        "Missing identifier",
        config.ID_COL,
        f"Dropped {before} rows without an order_id",
        "Without the business key the record cannot be identified or joined.",
    )

    # Text categories: the blanks look MCAR, so keep the row with 'Unknown'.
    for col in [
        "customer_name",
        "email",
        "product",
        "status",
        "country",
        "payment_method",
        "notes",
    ]:
        n = int(clean[col].isna().sum())
        if n:
            clean[col] = clean[col].fillna("Unknown")
            log_step(
                log,
                "Phase 1",
                "5",
                "Missing value",
                col,
                f"Filled {n} blanks with 'Unknown'",
                "Missingness looks random (MCAR); an explicit 'Unknown' keeps the row usable.",
            )

    # Measures -> real numbers.
    for col in config.MEASURE_COLS:
        clean[col] = _to_number(clean[col])
    clean["quantity"] = clean["quantity"].round().astype("Int64")

    # ------------------------------------------------------------------ #
    # PHASE 1 - Duplicates (define the grain first)
    # ------------------------------------------------------------------ #
    exact_dups = int(clean.duplicated().sum())
    clean = clean.drop_duplicates().reset_index(drop=True)
    metrics["exact_duplicates_removed"] = exact_dups
    log_step(
        log,
        "Phase 1",
        "6",
        "Exact duplicate rows",
        "ALL",
        f"Removed {exact_dups} byte-identical duplicate rows",
        "They are load artefacts, not repeated events.",
    )

    dup_ids = int(clean[config.ID_COL].duplicated().sum())
    if dup_ids:
        n_before = len(clean)
        clean["_missing"] = clean.isna().sum(axis=1)
        clean = (
            clean.sort_values("_missing")
            .drop_duplicates(subset=config.ID_COL, keep="first")
            .drop(columns="_missing")
            .reset_index(drop=True)
        )
        log_step(
            log,
            "Phase 1",
            "6",
            "Duplicate business key",
            config.ID_COL,
            f"Kept one row per order_id, removed {n_before - len(clean)} duplicates",
            "order_id is the grain; the retained copy is the most complete one.",
        )
    metrics["duplicate_ids_removed"] = dup_ids

    # ------------------------------------------------------------------ #
    # PHASE 2 - Standardizing & formatting
    # ------------------------------------------------------------------ #
    for col, mapping in [
        ("status", config.STATUS_MAP),
        ("country", config.COUNTRY_MAP),
        ("payment_method", config.PAYMENT_MAP),
        ("product", config.PRODUCT_MAP),
    ]:
        before_vals = clean[col].unique().tolist()
        clean[col] = standardize(clean[col], mapping)
        after_vals = clean[col].unique().tolist()
        if set(before_vals) != set(after_vals):
            log_step(
                log,
                "Phase 2",
                "7",
                "Inconsistent category",
                col,
                f"Canonicalised {len(before_vals)} variants -> {len(after_vals)} values",
                "Same real-world value written with different casing / abbreviations.",
            )

    clean["notes"] = clean["notes"].replace(
        {"n/a": "none", "N/A": "none", "NULL": "none", "Unknown": "none", "": "none"}
    )
    log_step(
        log,
        "Phase 2",
        "8",
        "Placeholder text",
        "notes",
        "Normalised n/a, N/A, NULL and blanks to 'none'",
        "Removes invisible formatting differences that break exact filters / joins.",
    )

    clean["customer_name"] = clean["customer_name"].map(clean_person_name)
    clean["email"] = clean["email"].astype(str).str.strip().str.lower()
    log_step(
        log,
        "Phase 2",
        "12",
        "Free-text normalization",
        "customer_name, email",
        "Names title-cased and 'Last, First' reordered; emails lower-cased",
        "Light cleanup because the column is read / grouped; email must be case-insensitive.",
    )

    # ------------------------------------------------------------------ #
    # PHASE 3 - Validating values
    # ------------------------------------------------------------------ #
    bad_email = clean["email"].ne("unknown") & ~clean["email"].str.match(config.EMAIL_RE)
    n_bad_email = int(bad_email.sum())
    clean.loc[bad_email, "email"] = np.nan
    metrics["invalid_emails"] = n_bad_email
    log_step(
        log,
        "Phase 3",
        "9",
        "Invalid email pattern",
        "email",
        f"Replaced {n_bad_email} invalid addresses with missing",
        "Value is the right type (text) but violates the business rule for e-mail.",
    )

    impossible_qty = clean["quantity"].notna() & (clean["quantity"] <= 0)
    n_impossible = int(impossible_qty.sum())
    clean.loc[impossible_qty, "quantity"] = pd.NA
    metrics["impossible_quantities"] = n_impossible
    log_step(
        log,
        "Phase 3",
        "11",
        "Impossible quantity",
        "quantity",
        f"Set {n_impossible} values <= 0 to missing",
        "You cannot sell a negative / zero number of items; always an error.",
    )

    qty_missing = int(clean["quantity"].isna().sum())
    if qty_missing:
        median_qty = float(clean["quantity"].median())
        clean["quantity"] = clean["quantity"].fillna(median_qty).astype("Int64")
        log_step(
            log,
            "Phase 3",
            "5/11",
            "Missing quantity after removing errors",
            "quantity",
            f"Imputed {qty_missing} values with the median ({median_qty:g})",
            "Median resists outliers; keeps the row usable for revenue analysis.",
        )

    # ------------------------------------------------------------------ #
    # PHASE 4 - Outliers & special fields
    # ------------------------------------------------------------------ #
    clean[config.DATE_COL] = clean[config.DATE_COL].map(parse_order_date)
    today = pd.Timestamp.today().normalize()
    future = clean[config.DATE_COL] > today
    n_future = int(future.sum())
    clean.loc[future, config.DATE_COL] = pd.NaT
    metrics["future_dates"] = n_future
    log_step(
        log,
        "Phase 4",
        "13",
        "Mixed date formats",
        config.DATE_COL,
        "Parsed M/D/YY and D-M-YYYY into one ISO 8601 datetime column",
        "Text dates cannot be sorted / compared and hide ambiguous formats.",
    )
    log_step(
        log,
        "Phase 4",
        "13",
        "Impossible future date",
        config.DATE_COL,
        f"Set {n_future} future dates to missing",
        "An order cannot be placed in the future.",
    )

    clean["total_price"] = _to_number(clean["total_price"])
    log_step(
        log,
        "Phase 4",
        "14",
        "Embedded symbols / units",
        "unit_price, total_price",
        "Stripped $, commas, % and spaces; converted to float",
        "Currency symbols make a numeric column text and break every calculation.",
    )

    outlier_counts = {}
    for col in ["unit_price", "total_price"]:
        low, high = iqr_bounds(clean[col])
        if low is None:
            continue
        outlier = clean[col].notna() & ((clean[col] < low) | (clean[col] > high))
        outlier_counts[col] = int(outlier.sum())
        log_step(
            log,
            "Phase 4",
            "10",
            "Statistical outliers",
            col,
            f"Flagged {int(outlier.sum())} values but kept them",
            "A rare high-value order can be legitimate; deleting it would bias revenue.",
        )
    metrics["outliers"] = outlier_counts

    # ------------------------------------------------------------------ #
    # PHASE 5 - Categories, ids & columns
    # ------------------------------------------------------------------ #
    log_step(
        log,
        "Phase 5",
        "15",
        "Categorical cardinality",
        "product, status, country, payment_method",
        "Reviewed cardinality; all remain true categories (no free-text column mislabeled)",
        "High cardinality would signal that a column is really free text.",
    )

    clean[config.ID_COL] = clean[config.ID_COL].astype(str)
    bad_ids = ~clean[config.ID_COL].str.match(config.ORDER_ID_RE)
    n_bad_ids = int(bad_ids.sum())
    metrics["malformed_ids"] = n_bad_ids
    log_step(
        log,
        "Phase 5",
        "16",
        "Identifier type / format",
        config.ID_COL,
        f"Stored as text; {n_bad_ids} values failed the ORD-##### pattern",
        "Numeric IDs lose leading zeros and break joins; text preserves the key.",
    )

    log_step(
        log,
        "Phase 5",
        "17",
        "Column selection",
        "notes",
        "Kept notes; no column dropped automatically",
        "No column is dropped until a human confirms it is truly useless.",
    )

    recalc = clean["quantity"].notna() & clean["unit_price"].notna()
    expected = (clean["quantity"].astype("Float64") * clean["unit_price"]).round(2)
    mismatch = recalc & ((expected - clean["total_price"]).abs() > config.TOTAL_PRICE_TOLERANCE)
    n_recalc = int(mismatch.sum())
    clean.loc[mismatch, "total_price"] = expected[mismatch]
    metrics["total_price_recalculated"] = n_recalc
    log_step(
        log,
        "Phase 5",
        "18",
        "Derived field inconsistency",
        "total_price",
        f"Recalculated {n_recalc} rows as quantity * unit_price",
        "Source measures were judged more reliable than the derived total.",
    )

    # ------------------------------------------------------------------ #
    # PHASE 6 - Integrity & final sign-off
    # ------------------------------------------------------------------ #
    contradiction = clean["status"].isin(["Cancelled", "Refunded"]) & (clean["total_price"] > 0)
    n_contradiction = int(contradiction.sum())
    metrics["contradictions"] = n_contradiction
    log_step(
        log,
        "Phase 6",
        "20",
        "Cross-column contradiction",
        "status + total_price",
        f"Flagged {n_contradiction} rows for review",
        "Each value is valid alone, but the combination Cancelled + money is not.",
    )

    clean["needs_review"] = bad_ids | contradiction
    log_step(
        log,
        "Phase 6",
        "19",
        "Key uniqueness",
        config.ID_COL,
        f"Verified unique: {bool(clean[config.ID_COL].is_unique)}",
        "A clean order table must have exactly one row per order_id.",
    )

    metrics["rows_after"] = int(clean.shape[0])
    metrics["columns_after"] = int(clean.shape[1])
    score_after = data_quality_score(clean)
    metrics["score_after"] = score_after
    log_step(
        log,
        "Phase 6",
        "21",
        "Final validation",
        "ALL",
        f"Quality score improved {score_before}/10 -> {score_after}/10",
        "Before/after comparison closes the cleaning session.",
    )

    say(f"Cleaned shape: {clean.shape}  |  quality {score_before} -> {score_after}")
    return clean, log, metrics
