from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import numpy.random as rd
import pyarrow as pa
import pyarrow.dataset as ds
import sqlite3
import time
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
FEATURES_ASSETS_DIR = ROOT / "data/parquet/features/assets"
REGIMES_DIR = ROOT / "data/parquet/regimes"
DB_PATH = ROOT / "data/_meta.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = ROOT / "config/mc.yaml"
MC_DIR = ROOT / "data/parquet/mc"

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def load_regimes() -> pd.DataFrame:
    dataset_regime = ds.dataset(str(REGIMES_DIR), format="parquet", partitioning="hive")
    df_regime = dataset_regime.to_table().to_pandas()
    if "date" in df_regime.columns:
        df_regime["date"] = pd.to_datetime(df_regime["date"], errors="coerce")
    return df_regime


def load_asset_features() -> pd.DataFrame:
    dataset_assets = ds.dataset(str(FEATURES_ASSETS_DIR), format="parquet", partitioning="hive")
    df_assets = dataset_assets.to_table().to_pandas()
    if "date" in df_assets.columns:
        df_assets["date"] = pd.to_datetime(df_assets["date"], errors="coerce")
    return df_assets


def load_mc_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    content = CONFIG_PATH.read_text().strip()
    if not content:
        return {}
    if yaml is None:
        raise ImportError("PyYAML is required to parse config/regimes.yaml.")
    data = yaml.safe_load(content)
    return data if isinstance(data, dict) else {}



def select_universe(df_assets: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    # TODO: filter asset features to a stable universe.
    pass


def build_returns_matrix(df_assets: pd.DataFrame) -> pd.DataFrame:
    # TODO: pivot to date x ticker returns (log-returns).
    pass


def calibrate_regime_params(
    returns: pd.DataFrame,
    regimes: pd.DataFrame,
    window: int,
) -> dict[int, dict[str, np.ndarray]]:
    # TODO: estimate mu/Sigma per regime on rolling window.
    span = len(regimes)
    mu = returns.rolling(regimes.rolling(window)).mean()
    sigma = ""

    out = pd.DataFrame(index=mu.index)
    out["mu"] = mu
    out["sigma"] = sigma
    return out


def simulate_paths(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_sims: int,
    horizon: int,
    dist: str = "gaussian",
) -> np.ndarray:
    # TODO: simulate MC paths (gaussian or t) and return paths array.
    pass


def summarize_paths(paths: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    # TODO: compute VaR/CVaR/quantiles/probabilities from paths.
    pass


def build_mc_outputs(
    returns: pd.DataFrame,
    regimes: pd.DataFrame,
    params: dict[int, dict[str, np.ndarray]],
    n_sims: int,
    horizons: list[int],
    dist: str,
) -> pd.DataFrame:
    # TODO: loop over dates and horizons; assemble summary rows.
    pass


def write_mc_dataset(
    df: pd.DataFrame,
    base_dir: Path,
    partition_cols: list[str],
    existing_data_behavior: str = "overwrite_or_ignore",
    basename_template: str | None = None,
) -> None:
    if df is None or df.empty:
        return
    base_dir.mkdir(parents=True, exist_ok=True)
    if "year" not in df.columns and "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        ok = dt.notna()
        df = df.loc[ok].copy()
        df["year"] = dt.loc[ok].dt.year.astype("int32")
    table = pa.Table.from_pandas(df, preserve_index=False)
    schema = []
    for col in partition_cols:
        if col not in df.columns:
            raise KeyError(f"Missing partition column: {col}")
        if col == "year":
            schema.append((col, pa.int32()))
        else:
            schema.append((col, pa.string()))
    partitioning = ds.partitioning(pa.schema(schema), flavor="hive")
    if existing_data_behavior == "overwrite":
        existing_data_behavior = "delete_matching"
        ds.write_dataset(
        table,
        base_dir=str(base_dir),
        format="parquet",
        partitioning=partitioning,
        existing_data_behavior=existing_data_behavior,
        basename_template=basename_template,
    )


def run_mc_pipeline(existing_data_behavior: str = "overwrite_or_ignore") -> None:
    # TODO: wire all steps: load -> align -> calibrate -> simulate -> summarize -> write.
    pass
