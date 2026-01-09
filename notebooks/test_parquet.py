import pyarrow.dataset as ds
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGIMES_DIR = ROOT / "data/parquet/regimes"
dataset = ds.dataset(str(REGIMES_DIR), format="parquet", partitioning="hive")
df = dataset.to_table().to_pandas()
print(df.head())
