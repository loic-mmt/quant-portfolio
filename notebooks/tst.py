import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.dataset as ds
from pathlib import Path

out_dir = Path("data/parquet/features")
REGIME_DIR = out_dir / "regime"
ASSET_DIR = out_dir / "assets"
DB_PATH = Path("data/_meta.db")

def load_prices_dataset() -> pd.DataFrame:
    dataset = ds.dataset("data/parquet/prices", format="parquet", partitioning="hive")
    return dataset.to_table().to_pandas()

def load_features_dataset() -> pd.DataFrame:
    dataset = ds.dataset("data/parquet/features", format="parquet", partitioning="hive")
    return dataset.to_table().to_pandas()

prices = load_prices_dataset()
features = load_features_dataset()


plt.plot(features['vol_i_60'])