"""Project-wide paths, constants and business rules.

Everything that another module might want to tune (file locations, canonical
category values, which statuses count as realised revenue) lives here so the
rest of the package stays declarative.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_FILE = DATA_DIR / "Ecommerce_Orders.csv"

CLEANED_DIR = PROJECT_ROOT / "cleaned_data"
CLEANED_CSV = CLEANED_DIR / "Ecommerce_Orders_cleaned.csv"
CLEANED_XLSX = CLEANED_DIR / "Ecommerce_Orders_cleaned.xlsx"
CLEANING_LOG_FILE = CLEANED_DIR / "cleaning_log.csv"

SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
REPORTS_DIR = PROJECT_ROOT / "reports"
DASHBOARD_FILE = REPORTS_DIR / "dashboard.html"

for _directory in (CLEANED_DIR, SCREENSHOTS_DIR, REPORTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
ID_COL = "order_id"
DATE_COL = "order_date"
MEASURE_COLS = ["quantity", "unit_price", "total_price"]
CATEGORY_COLS = ["product", "status", "country", "payment_method"]

SENTINELS = {"", "nan", "nat", "none", "n/a", "na", "null"}

# --------------------------------------------------------------------------- #
# Business rules
# --------------------------------------------------------------------------- #
# "Recognised" revenue = orders that actually turned into money for the shop.
REVENUE_STATUSES = ["Completed", "Shipped"]
# Terminal states where no money should be recognised.
NON_REVENUE_STATUSES = ["Cancelled", "Refunded", "Pending"]

VALID_STATUSES = ["Cancelled", "Completed", "Pending", "Refunded", "Shipped"]

STATUS_MAP = {
    "COMPLETED": "Completed",
    "PENDING": "Pending",
    "SHIPPED": "Shipped",
    "CANCELLED": "Cancelled",
    "REFUNDED": "Refunded",
}

COUNTRY_MAP = {
    "US": "United States",
    "USA": "United States",
    "U.S.A": "United States",
    "UNITED STATES": "United States",
    "AUS": "Australia",
    "AUSTRALIA": "Australia",
    "CA": "Canada",
    "CANADA": "Canada",
    "UK": "United Kingdom",
    "ENGLAND": "United Kingdom",
    "UNITED KINGDOM": "United Kingdom",
}

PAYMENT_MAP = {
    "CREDIT CARD": "Credit Card",
    "DEBIT": "Debit Card",
    "DEBIT CARD": "Debit Card",
    "PAYPAL": "PayPal",
    "APPLE PAY": "Apple Pay",
}

PRODUCT_MAP = {"UNKNOWN": "Unknown"}

# A derived total is allowed to differ from quantity * unit_price by this much
# (rounding on the source side) before we consider it wrong.
TOTAL_PRICE_TOLERANCE = 0.05

# --------------------------------------------------------------------------- #
# Machine learning
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5
MAX_CLUSTERS = 5

ML_METRICS_FILE = REPORTS_DIR / "ml_metrics.json"

CURRENCY_STRIP_RE = r"[$,%\s]"
ORDER_ID_RE = r"^ORD-\d{5}$"
EMAIL_RE = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
