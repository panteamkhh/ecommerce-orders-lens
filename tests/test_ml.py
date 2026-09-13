"""Tests for the two machine-learning algorithms."""

import numpy as np

from src import config, ml


def test_rfm_one_row_per_customer(featured):
    rfm = ml.build_rfm(featured)
    assert rfm["customer_name"].is_unique
    assert set(rfm.columns) == {"customer_name", "recency", "frequency", "monetary"}
    assert (rfm["recency"] >= 0).all()
    assert (rfm["frequency"] >= 1).all()


def test_classifier_metrics_are_sane(ml_result, featured):
    metrics = ml_result["metrics"]

    assert metrics["best_model"] in {"Logistic Regression", "Random Forest"}
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["n_train"] + metrics["n_test"] == len(featured)
    assert set(ml_result["cv_table"]["model"]) == {
        "Baseline (prior)",
        "Logistic Regression",
        "Random Forest",
    }


def test_confusion_matrix_shape(ml_result):
    assert ml_result["confusion"].shape == (2, 2)
    assert ml_result["confusion"].sum() == ml_result["metrics"]["n_test"]


def test_feature_importance_is_ranked(ml_result):
    importance = ml_result["feature_importance"]
    assert not importance.empty
    assert importance["weight"].is_monotonic_decreasing
    assert np.isclose(importance["importance_%"].sum(), 100, atol=0.5)


def test_roc_points_are_valid(ml_result):
    roc = ml_result["roc"]
    assert len(roc["fpr"]) == len(roc["tpr"])
    assert 0 <= roc["auc"] <= 1


def test_segmentation_labels_every_customer(segmentation):
    assert 2 <= segmentation["chosen_k"] <= config.MAX_CLUSTERS
    assert -1 <= segmentation["silhouette"] <= 1
    assert segmentation["customers"]["segment"].notna().all()


def test_segment_summary_accounts_for_all_customers(segmentation):
    summary = segmentation["summary"]
    assert int(summary["customers"].sum()) == len(segmentation["customers"])
    assert np.isclose(summary["revenue_share_%"].sum(), 100, atol=0.5)
