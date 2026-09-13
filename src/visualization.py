"""Render one chart per business question into ``screenshots/``.

The plotting layer is deliberately dumb: it receives already-computed
DataFrames from :mod:`src.analysis` and only worries about presentation.

Every chart shares one visual language: the same fonts, title weight, grid,
tick sizes, colour meaning and value-label style. That consistency is enforced
through the small helpers below rather than repeated in each function.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config

# --------------------------------------------------------------------------- #
# Shared theme
# --------------------------------------------------------------------------- #
PRIMARY = "#2563eb"  # single-series bars / lines
GOOD = "#10b981"  # realised revenue
BAD = "#ef4444"  # losses / cancellations
MUTED = "#94a3b8"  # reference lines
INK = "#334155"  # value labels
CATEGORICAL = [PRIMARY, GOOD, "#f59e0b", "#8b5cf6", "#0ea5e9", BAD, MUTED]

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.labelcolor": "#1f2937",
        "axes.edgecolor": "#cbd5e1",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.autolayout": True,
    }
)


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _int(value: float) -> str:
    return f"{value:,.0f}"


def _save(fig, filename: str):
    path = config.SCREENSHOTS_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def _title(ax, text: str) -> None:
    ax.set_title(text)


def _money_axis(ax, which: str = "y") -> None:
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_formatter(lambda value, _: _money(value))


def _vlabels(ax, values, fmt=_money) -> None:
    """Value labels on top of vertical bars."""
    for x, value in enumerate(values):
        ax.text(x, value, fmt(value), ha="center", va="bottom", fontsize=8, color=INK)


def _hlabels(ax, values, fmt=_money, suffix: str = "") -> None:
    """Value labels at the end of horizontal bars."""
    for y, value in enumerate(values):
        ax.text(value, y, f"  {fmt(value)}{suffix}", va="center", fontsize=8, color=INK)


# --------------------------------------------------------------------------- #
# Business charts
# --------------------------------------------------------------------------- #
def chart_top_bottom_products(product_df: pd.DataFrame, top_n: int = 5):
    """Top sellers vs. the products that barely move the needle."""
    top = product_df.head(top_n)
    bottom = product_df.tail(top_n)
    view = pd.concat([top, bottom]).drop_duplicates(subset="product").sort_values("revenue")

    colors = [GOOD if r >= product_df["revenue"].median() else BAD for r in view["revenue"]]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(view["product"], view["revenue"], color=colors)
    _title(ax, "Revenue by product (top and bottom performers)")
    ax.set_xlabel("Recognised revenue")
    _money_axis(ax, "x")
    _hlabels(ax, view["revenue"])
    return _save(fig, "01_top_bottom_products.png")


def chart_revenue_trend(trend_df: pd.DataFrame):
    """Monthly recognised revenue over the whole history."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(trend_df["order_year_month"], trend_df["revenue"], marker="o", color=PRIMARY)
    ax.fill_between(trend_df["order_year_month"], trend_df["revenue"], color=PRIMARY, alpha=0.12)
    _title(ax, "Monthly revenue trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    _money_axis(ax, "y")
    step = max(len(trend_df) // 10, 1)
    ax.set_xticks(range(0, len(trend_df), step))
    ax.set_xticklabels(trend_df["order_year_month"][::step], rotation=45, ha="right")
    return _save(fig, "02_revenue_trend.png")


def chart_revenue_by_country(country_df: pd.DataFrame):
    """Revenue share per country."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(country_df["country"], country_df["revenue"], color=PRIMARY)
    _title(ax, "Revenue by country")
    ax.set_ylabel("Revenue")
    _money_axis(ax, "y")
    _vlabels(ax, country_df["revenue"])
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, "03_revenue_by_country.png")


def chart_revenue_by_payment(payment_df: pd.DataFrame):
    """Revenue split by payment method."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.pie(
        payment_df["revenue"],
        labels=payment_df["payment_method"],
        autopct="%1.1f%%",
        startangle=90,
        colors=CATEGORICAL,
        textprops={"fontsize": 9, "color": "#1f2937"},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    _title(ax, "Revenue by payment method")
    return _save(fig, "04_revenue_by_payment.png")


def chart_status_breakdown(status_df: pd.DataFrame):
    """Order count per status, coloured by realised vs. not."""
    colors = [GOOD if s in config.REVENUE_STATUSES else BAD for s in status_df["status"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(status_df["status"], status_df["orders"], color=colors)
    _title(ax, "Orders by status")
    ax.set_ylabel("Orders")
    _vlabels(ax, status_df["orders"], fmt=_int)
    return _save(fig, "05_status_breakdown.png")


def chart_top_customers(customer_df: pd.DataFrame):
    """Top customers by recognised revenue."""
    view = customer_df.sort_values("revenue")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(view["customer_name"], view["revenue"], color=PRIMARY)
    _title(ax, "Top customers by revenue")
    ax.set_xlabel("Revenue")
    _money_axis(ax, "x")
    _hlabels(ax, view["revenue"])
    return _save(fig, "06_top_customers.png")


def chart_seasonality(season_df: pd.DataFrame):
    """Average revenue per calendar month."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(season_df["order_month_name"], season_df["avg_revenue"], color=PRIMARY)
    _title(ax, "Average revenue by calendar month")
    ax.set_ylabel("Avg. revenue")
    _money_axis(ax, "y")
    return _save(fig, "07_monthly_seasonality.png")


def chart_cancellation_impact(impact_df: pd.DataFrame):
    """Money sitting in Cancelled / Refunded / Pending."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.bar(impact_df["status"], impact_df["value"], color=BAD)
    _title(ax, "Value locked in non-revenue statuses")
    ax.set_ylabel("Order value")
    _money_axis(ax, "y")
    _vlabels(ax, impact_df["value"])
    return _save(fig, "08_cancellation_impact.png")


def chart_quality_before_after(metrics: dict):
    """The data-quality score before and after cleaning."""
    fig, ax = plt.subplots(figsize=(5, 4.4))
    bars = ax.bar(
        ["Before", "After"],
        [metrics["score_before"], metrics["score_after"]],
        color=[BAD, GOOD],
        width=0.55,
    )
    ax.set_ylim(0, 10)
    _title(ax, "Data-quality score")
    ax.set_ylabel("Score (out of 10)")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )
    return _save(fig, "09_data_quality.png")


# --------------------------------------------------------------------------- #
# Machine-learning charts
# --------------------------------------------------------------------------- #
def chart_ml_confusion_matrix(matrix, model_name: str, labels=("Not revenue", "Revenue")):
    """Confusion matrix of the winning classifier on the held-out test set."""
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        ax=ax,
    )
    _title(ax, f"Confusion matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    return _save(fig, "10_ml_confusion_matrix.png")


def chart_ml_feature_importance(importance_df: pd.DataFrame, top_n: int = 12):
    """Which signals drive the order-outcome prediction."""
    view = importance_df.head(top_n).sort_values("weight")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(view["feature"], view["importance_%"], color=PRIMARY)
    _title(ax, "What predicts whether an order converts")
    ax.set_xlabel("Relative importance (%)")
    _hlabels(ax, view["importance_%"], fmt=lambda v: f"{v:.1f}%")
    return _save(fig, "11_ml_feature_importance.png")


def chart_ml_roc(roc: dict, model_name: str):
    """ROC curve of the winning classifier."""
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.plot(
        roc["fpr"],
        roc["tpr"],
        color=PRIMARY,
        linewidth=2,
        label=f"{model_name} (AUC={roc['auc']:.2f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, label="Random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _title(ax, "ROC curve - order outcome")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    return _save(fig, "12_ml_roc_curve.png")


def chart_customer_segments(segment_summary: pd.DataFrame):
    """Revenue and customer count per RFM segment."""
    view = segment_summary.sort_values("total_revenue")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(view["segment"], view["total_revenue"], color=PRIMARY)
    _title(ax, "Revenue by customer segment")
    ax.set_xlabel("Recognised revenue")
    _money_axis(ax, "x")
    for y, (value, count) in enumerate(zip(view["total_revenue"], view["customers"], strict=True)):
        ax.text(
            value,
            y,
            f"  {_money(value)}  ({_int(count)} customers)",
            va="center",
            fontsize=8,
            color=INK,
        )
    return _save(fig, "13_customer_segments.png")
