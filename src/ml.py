"""Two machine-learning algorithms on the cleaned orders table.

1. **Order-outcome classification** - predict whether an order will convert to
   recognised revenue (Completed / Shipped) or not, using only information
   known at order time. Comparing a linear and a tree-based model keeps the
   result honest: whichever generalises better on cross-validation wins.

2. **RFM customer segmentation** - unsupervised KMeans over Recency,
   Frequency and Monetary value, with the number of clusters chosen by
   silhouette score rather than picked by hand.

Both are deliberately small and reproducible (fixed random seed) because the
dataset has fewer than 200 rows: the goal is an explainable signal, not a
leaderboard score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

NUMERIC_FEATURES = ["quantity", "unit_price"]
CATEGORICAL_FEATURES = ["product"]

SEGMENT_LABELS = ["Champions", "Loyal", "Occasional", "At Risk", "Dormant"]


# --------------------------------------------------------------------------- #
# 1. Order-outcome classification
# --------------------------------------------------------------------------- #
def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)]
    )


def _candidate_models() -> dict:
    seed = config.RANDOM_STATE
    return {
        "Baseline (prior)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=0.5, class_weight="balanced", random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=3,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
        ),
    }


def _original_feature(transformed_name: str) -> str:
    """Map 'cat__product_Smart Watch' back to 'product'."""
    base = transformed_name.split("__", 1)[-1]
    for feature in CATEGORICAL_FEATURES:
        if base == feature or base.startswith(f"{feature}_"):
            return feature
    for feature in NUMERIC_FEATURES:
        if base == feature:
            return feature
    return base


def _feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    pre = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]
    names = pre.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        weights = np.asarray(model.feature_importances_)
    else:
        weights = np.abs(np.asarray(model.coef_).ravel())

    out = pd.DataFrame({"encoded_feature": names, "weight": weights})
    out["feature"] = out["encoded_feature"].map(_original_feature)
    agg = (
        out.groupby("feature", as_index=False)["weight"]
        .sum()
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    total = agg["weight"].sum() or 1.0
    agg["importance_%"] = (100 * agg["weight"] / total).round(1)
    return agg


def train_order_outcome_model(df: pd.DataFrame) -> dict:
    """Train and evaluate models that predict `is_revenue`.

    Returns a dict with the winning model, cross-validation table, held-out
    metrics, confusion matrix, feature importance and ROC-curve points.
    """
    data = df.copy()
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data["is_revenue"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    cv_rows = []
    fitted = {}
    for name, model in _candidate_models().items():
        pipeline = Pipeline([("pre", _preprocessor()), ("model", model)])
        auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        acc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        cv_rows.append(
            {
                "model": name,
                "cv_roc_auc": round(float(auc.mean()), 3),
                "cv_roc_auc_std": round(float(auc.std()), 3),
                "cv_accuracy": round(float(acc.mean()), 3),
            }
        )
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline

    cv_table = (
        pd.DataFrame(cv_rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)
    )
    # The baseline is shown for context but never "wins".
    challengers = cv_table[cv_table["model"] != "Baseline (prior)"]
    best_name = challengers.iloc[0]["model"]
    best = fitted[best_name]
    best_cv_auc = float(challengers.iloc[0]["cv_roc_auc"])
    baseline_auc = float(
        cv_table.loc[cv_table["model"] == "Baseline (prior)", "cv_roc_auc"].iloc[0]
    )

    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    test_auc = round(float(roc_auc_score(y_test, y_prob)), 3)

    metrics = {
        "best_model": best_name,
        "baseline_auc": round(baseline_auc, 3),
        "cv_roc_auc": round(best_cv_auc, 3),
        "beats_baseline": best_cv_auc > baseline_auc,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
        "roc_auc": test_auc,
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 3),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": round(float(y.mean()), 3),
    }

    return {
        "cv_table": cv_table,
        "metrics": metrics,
        "confusion": confusion_matrix(y_test, y_pred),
        "feature_importance": _feature_importance(best),
        "roc": {"fpr": fpr, "tpr": tpr, "auc": metrics["roc_auc"]},
    }


# --------------------------------------------------------------------------- #
# 2. RFM customer segmentation
# --------------------------------------------------------------------------- #
def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """One row per customer with Recency, Frequency and Monetary value."""
    dated = df.dropna(subset=[config.DATE_COL])
    reference = pd.to_datetime(dated[config.DATE_COL]).max() + pd.Timedelta(days=1)

    grouped = df.groupby("customer_name")
    rfm = grouped.agg(
        last_order=(config.DATE_COL, "max"),
        frequency=(config.ID_COL, "nunique"),
        monetary=("revenue", "sum"),
    ).reset_index()
    rfm["recency"] = (reference - pd.to_datetime(rfm["last_order"])).dt.days
    rfm["recency"] = rfm["recency"].fillna(rfm["recency"].max())
    rfm["monetary"] = rfm["monetary"].round(2)
    rfm = rfm[["customer_name", "recency", "frequency", "monetary"]]
    return rfm.sort_values("monetary", ascending=False).reset_index(drop=True)


def segment_customers(df: pd.DataFrame, n_clusters: int | None = None) -> dict:
    """Cluster customers on scaled RFM and give each cluster a business label."""
    rfm = build_rfm(df)
    features = rfm[["recency", "frequency", "monetary"]].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(features)

    if n_clusters is None:
        k_scores = []
        upper = min(config.MAX_CLUSTERS, max(len(rfm) - 1, 2))
        for k in range(2, upper + 1):
            labels = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit_predict(
                scaled
            )
            k_scores.append(
                {"k": k, "silhouette": round(float(silhouette_score(scaled, labels)), 3)}
            )
        k_table = (
            pd.DataFrame(k_scores).sort_values("silhouette", ascending=False).reset_index(drop=True)
        )
        chosen_k = int(k_table.loc[0, "k"])
        best_silhouette = float(k_table.loc[0, "silhouette"])
    else:
        chosen_k = int(n_clusters)
        labels = KMeans(
            n_clusters=chosen_k, n_init=10, random_state=config.RANDOM_STATE
        ).fit_predict(scaled)
        best_silhouette = float(silhouette_score(scaled, labels))
        k_table = pd.DataFrame([{"k": chosen_k, "silhouette": round(best_silhouette, 3)}])

    model = KMeans(n_clusters=chosen_k, n_init=10, random_state=config.RANDOM_STATE)
    rfm["cluster"] = model.fit_predict(scaled)

    # Label clusters from best to worst by monetary value.
    order = rfm.groupby("cluster")["monetary"].mean().sort_values(ascending=False).index.tolist()
    label_map = {
        cluster: SEGMENT_LABELS[i % len(SEGMENT_LABELS)] for i, cluster in enumerate(order)
    }
    rfm["segment"] = rfm["cluster"].map(label_map)

    summary = (
        rfm.groupby("segment")
        .agg(
            customers=("customer_name", "nunique"),
            avg_recency_days=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            total_revenue=("monetary", "sum"),
        )
        .round({"avg_recency_days": 0, "avg_frequency": 1, "total_revenue": 2})
        .reset_index()
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )
    total = summary["total_revenue"].sum() or 1.0
    summary["revenue_share_%"] = (100 * summary["total_revenue"] / total).round(1)

    return {
        "customers": rfm,
        "summary": summary,
        "k_table": k_table,
        "chosen_k": chosen_k,
        "silhouette": round(best_silhouette, 3),
    }
