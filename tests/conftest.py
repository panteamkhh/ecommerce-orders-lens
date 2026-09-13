"""Shared pytest fixtures: the raw table is loaded and cleaned once per session."""

import pandas as pd
import pytest

from src import data_cleaning, data_loader, feature_engineering, ml


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return data_loader.load_and_validate()


@pytest.fixture(scope="session")
def cleaned(raw_df):
    clean, log, metrics = data_cleaning.clean_orders(raw_df)
    return clean


@pytest.fixture(scope="session")
def cleaning_log(raw_df):
    _, log, _ = data_cleaning.clean_orders(raw_df)
    return log


@pytest.fixture(scope="session")
def metrics(raw_df):
    _, _, metrics = data_cleaning.clean_orders(raw_df)
    return metrics


@pytest.fixture(scope="session")
def featured(cleaned) -> pd.DataFrame:
    return feature_engineering.add_features(cleaned)


@pytest.fixture(scope="session")
def ml_result(featured):
    return ml.train_order_outcome_model(featured)


@pytest.fixture(scope="session")
def segmentation(featured):
    return ml.segment_customers(featured)
