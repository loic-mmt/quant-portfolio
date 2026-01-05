import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import shutil 


out_dir = Path("../data/parquet/features")
CLEAN_PARQUET = False  # set True only if you want to reset the dataset
if CLEAN_PARQUET and out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # --- Read dataset (Hive partition discovery adds asset/year as virtual columns) ---
    dataset = ds.dataset(str(out_dir), format="parquet", partitioning="hive")
    print("Discovered schema:", dataset.schema)

    table = dataset.to_table(filter=(ds.field("ticker")) & (ds.field("year")),
                            columns=["date", "Adj Close", "Volume"])
    df = table.to_pandas()
    print(df.head())