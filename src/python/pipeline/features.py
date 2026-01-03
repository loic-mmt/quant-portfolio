import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import linear_model
import pyarrow as pa
import pyarrow.dataset as ds
import shutil


out_dir = Path("data/parquet/features")
CLEAN_PARQUET = False  # set True only if you want to reset the dataset
if CLEAN_PARQUET and out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)


def compute_returns(df):
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    prices = df["adj_close"] if "adj_close" in df.columns else df.iloc[:, 0]
    return prices.pct_change()



prices = Path("data/parquet/prices")
def momentum(df, batch: str = "all"):
    batch_params = {
        "20 jours": 20,
        "3 mois": 60,
        "1 an": 252,
        "all": "all",
    }
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)
    if batch not in batch_params:
        raise KeyError("Please choose between: 20 jours, 3 mois, 1 an, all")

    returns = compute_returns(df)
    out = pd.DataFrame(index=df.index)
    out["mom_20"] = returns.rolling(20).sum()
    out["mom_60"] = returns.rolling(60).sum()
    out["mom_252"] = returns.rolling(252).sum()

    if batch == "all":
        return out
    window = batch_params[batch]
    return out[[f"mom_{window}"]]



def trend_slope_60(df):
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    prices = df["adj_close"] if "adj_close" in df.columns else df.iloc[:, 0]
    log_prices = np.log(prices)
    window = 60
    x = np.arange(window).reshape(-1, 1)

    def slope_from_window(y_window):
        lr = linear_model.LinearRegression()
        lr.fit(x, y_window)
        return lr.coef_[0]

    return log_prices.rolling(window).apply(slope_from_window, raw=True)


def dist_ma(df, batch: str = "all"):
    batch_params = {
        "50 jours": 50,
        "200 jours": 200,
        "all": "all",
    }
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)
    if batch not in batch_params:
        raise KeyError("Please choose between: 20 jours, 3 mois, 1 an, all")

    prices = df["adj_close"] if "adj_close" in df.columns else df.iloc[:, 0]
    MA50 = prices.rolling(50).mean()
    MA200 = prices.rolling(200).mean()
    
    out = pd.DataFrame(index=df.index)
    out["dist_ma_50"] = (prices/MA50)-1
    out["dist_ma_200"] = (prices/MA200)-1

    if batch == "all":
        return out
    window = batch_params[batch]
    return out[[f"dist_ma{window}"]]
 


def upsert_features(df: pd.DataFrame) -> int:
    """Append the new batch to a Hive-partitioned Parquet dataset.

    Dataset layout: data/parquet/features/feature=ZZZ/ticker=XXX/year=YYYY/*.parquet
    """
    if df is None or df.empty:
        return 0

    # Ensure required partition columns exist
    if "year" not in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        ok = dt.notna()
        df = df.loc[ok].copy()
        df["year"] = dt.loc[ok].dt.year.astype("int32")

    table = pa.Table.from_pandas(df, preserve_index=False)

    partitioning = ds.partitioning(
        pa.schema(
            [
                ("feature", pa.string()),
                ("ticker", pa.string()),
                ("year", pa.int32()),
            ]
        ),
        flavor="hive",
    )

    # Unique filenames per batch to avoid overwriting older fragments
    ticker = str(df["ticker"].iloc[0])
    feature_name = str(df["feature"].iloc[0])
    dmin = str(df["date"].min())
    dmax = str(df["date"].max())
    basename_template = f"{ticker}_{dmin}_{dmax}_{feature_name}_{{i}}.parquet"

    ds.write_dataset(
        table,
        base_dir=str(out_dir),
        format="parquet",
        partitioning=partitioning,
        basename_template=basename_template,
        existing_data_behavior="overwrite_or_ignore",
    )

    return table.num_rows
