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

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def load_regime() -> pd.DataFrame:
    dataset_regime = ds.dataset(str(REGIMES_DIR), format="parquet", partitioning="hive")
    df_regime = dataset_regime.to_table().to_pandas()
    if "date" in df_regime.columns:
        df_regime["date"] = pd.to_datetime(df_regime["date"], errors="coerce")
    return df_regime


def load_assets_features() -> pd.DataFrame:
    dataset_assets = ds.dataset(str(FEATURES_ASSETS_DIR), format="parquet", partitioning="hive")
    df_assets = dataset_assets.to_table().to_pandas()
    if "date" in df_assets.columns:
        df_assets["date"] = pd.to_datetime(df_assets["date"], errors="coerce")
    return df_assets


def EstimProbaX():
    L=[]
    for i in range(10000):
        L.append(SimulX())
    return L.count(4)/10000