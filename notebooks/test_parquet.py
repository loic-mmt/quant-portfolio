import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import shutil

# --- Output dataset directory (clean run) ---
out_dir = Path("data/parquet/prices")
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=5),
    "asset": ["AAPL"]*5,
    "close": [1,2,3,4,5],
})
df["year"] = df["date"].dt.year

table = pa.Table.from_pandas(df, preserve_index=False)

# --- Write as a Hive-partitioned Parquet dataset ---
partitioning = ds.partitioning(
    pa.schema([
        ("asset", pa.string()),
        ("year", pa.int32()),
    ]),
    flavor="hive",
)

ds.write_dataset(
    table,
    base_dir=str(out_dir),
    format="parquet",
    partitioning=partitioning,
    existing_data_behavior="overwrite_or_ignore",  # pratique en dev
)

# --- Read dataset (Hive partition discovery adds asset/year as virtual columns) ---
dataset = ds.dataset(str(out_dir), format="parquet", partitioning="hive")
print("Discovered schema:", dataset.schema)

table = dataset.to_table(filter=(ds.field("asset") == "AAPL") & (ds.field("year") == 2025),
                         columns=["date", "close"])
df_aapl = table.to_pandas()
print(df_aapl)
