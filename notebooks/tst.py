import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.dataset as ds
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out_dir = ROOT / "data/parquet/features"
REGIME_DIR = out_dir / "regime"
ASSET_DIR = out_dir / "assets"
DB_PATH = Path("data/_meta.db")
REGIMES_DIR = ROOT / "data/parquet/regimes"

def load_prices_dataset() -> pd.DataFrame:
    dataset = ds.dataset(str(ROOT / "data/parquet/prices"), format="parquet", partitioning="hive")
    return dataset.to_table().to_pandas()

def load_asset_features(ticker: str) -> pd.DataFrame:
    dataset = ds.dataset(str(ASSET_DIR), format="parquet", partitioning="hive")
    table = dataset.to_table(filter=ds.field("ticker") == ticker)
    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date")

def load_regimes() -> pd.DataFrame:
    dataset_regime = ds.dataset(str(REGIMES_DIR), format="parquet", partitioning="hive")
    df_regime = dataset_regime.to_table().to_pandas()
    if "date" in df_regime.columns:
        df_regime["date"] = pd.to_datetime(df_regime["date"], errors="coerce")
    return df_regime

ticker = "AEG"
features = load_asset_features(ticker)
regimes = load_regimes()

plt.figure(figsize=(10, 4))
plt.plot(features["date"], features["vol_i_60"])
plt.title(f"{ticker} vol_i_60")
plt.xlabel("Date")
plt.ylabel("vol_i_60")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(regimes["date"], regimes["state"])
plt.title("Market Regimes")
plt.xlabel("Date")
plt.ylabel("Regime")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(regimes["date"], regimes["p_state_0"])
plt.title("Market Regimes")
plt.xlabel("Date")
plt.ylabel("Regime")
plt.tight_layout()
plt.show()

#, regimes["p_state_1"], regimes["p_state_2"]