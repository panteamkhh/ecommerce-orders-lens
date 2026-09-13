# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-09-13

### Added

- `src/` analysis package: `config`, `data_loader`, `data_cleaning`,
  `feature_engineering`, `analysis`, `visualization`, `dashboard` and the
  `run_analysis` CLI entry point.
- Documented 21-step cleaning checklist grouped into seven phases, with an
  audit log written to `cleaned_data/cleaning_log.csv` and the Excel workbook.
- Business analyses: product, monthly trend, seasonality, country, payment
  method, status, top customers, cancellation impact and add-on impact.
- Two machine-learning algorithms in `src/ml.py`:
  - order-outcome classification (baseline vs. logistic regression vs. random
    forest, 5-fold cross-validation and a held-out test set);
  - KMeans RFM customer segmentation with the cluster count chosen by
    silhouette score.
- Nine business charts plus four ML charts exported to `screenshots/`, all
  sharing one visual theme.
- Self-contained HTML dashboard with embedded charts, KPI cards and tables.
- `pytest` test suite covering cleaning invariants, feature engineering, every
  analysis function and both ML algorithms.
- Tooling: `pyproject.toml`, `ruff`, `black`, `pre-commit`, GitHub Actions CI.
