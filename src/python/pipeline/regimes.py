"""
Pipeline: regime detection

TODO:
- load data/parquet/features/regime (hive)
- select feature columns (config/regimes.yaml)
- split train/validation by date
- standardize (z-score on train only, apply to full)
- fit model(s) via src/python/models/hmm.py
- produce state + probabilities per date
- write outputs to data/parquet/regimes (hive partitioned by year)
- store last processed date in SQLite (optional, like features.py)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "data/parquet/features/assets"
FEATURES_REGIME_DIR = ROOT / "data/parquet/features/regime"
REGIMES_DIR = ROOT / "data/parquet/regimes"


def load_regime_features() -> pd.DataFrame:
    # TODO: load parquet dataset from FEATURES_DIR (hive) and parse date.

    dataset_assets = ds.dataset(FEATURES_DIR, format="parquet", partitioning="hive")
    dataset_regime = ds.dataset(FEATURES_REGIME_DIR, format="parquet", partitioning="hive")
    return dataset_assets.to_table().to_pandas(), dataset_regime.to_table().to_pandas()


def select_regime_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    # TODO: keep only numeric columns in feature_cols, drop rows with NaNs as needed.
    pass


def standardize_train_apply_all(
    df: pd.DataFrame,
    train_end: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    # TODO: compute mean/std on train slice, apply to full df (z-score).
    pass


def fit_regime_model(df_z: pd.DataFrame):
    # TODO: call models.hmm.fit_hmm_features or fit_markov_market depending on config.
    pass


def build_regime_outputs(
    model,
    df_z: pd.DataFrame,
) -> pd.DataFrame:
    # TODO: compute state/proba using models.hmm helpers.
    pass


def write_regimes_dataset(df: pd.DataFrame) -> None:
    # TODO: add year column and write to REGIMES_DIR (hive partitioned).
    pass


def run_regime_pipeline() -> None:
    # TODO: wire all steps: load -> select -> standardize -> fit -> outputs -> write.
    pass


if __name__ == "__main__":
    # TODO: parse args (train_end, model type, feature list, etc.)
    run_regime_pipeline()
