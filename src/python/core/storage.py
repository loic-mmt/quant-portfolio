from __future__ import annotations

from pathlib import Path
from typing import Iterable
import time

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


def read_parquet_dataset(
    base_dir: Path,
    columns: list[str] | None = None,
    filter_expr: ds.Expression | None = None,
) -> pd.DataFrame:
    """Read a Parquet dataset (hive partitioned) into a DataFrame."""
    dataset = ds.dataset(str(base_dir), format="parquet", partitioning="hive")
    table = dataset.to_table(filter=filter_expr, columns=columns)
    return table.to_pandas()


def _ensure_year_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if "year" not in df.columns and date_col in df.columns:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        ok = dt.notna()
        df = df.loc[ok].copy()
        df["year"] = dt.loc[ok].dt.year.astype("int32")
    return df


def write_partitioned_dataset(
    df: pd.DataFrame,
    base_dir: Path,
    partition_cols: list[str],
    existing_data_behavior: str = "overwrite_or_ignore",
    basename_template: str | None = None,
) -> None:
    """Write a DataFrame to a hive-partitioned Parquet dataset."""
    if df is None or df.empty:
        return
    base_dir.mkdir(parents=True, exist_ok=True)

    df = _ensure_year_column(df)

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


def append_prices_dataset(df: pd.DataFrame, base_dir: Path) -> int:
    """Append a batch to the prices dataset (partitioned by ticker/year)."""
    if df is None or df.empty:
        return 0

    df = _ensure_year_column(df)
    table = pa.Table.from_pandas(df, preserve_index=False)

    partitioning = ds.partitioning(
        pa.schema(
            [
                ("ticker", pa.string()),
                ("year", pa.int32()),
            ]
        ),
        flavor="hive",
    )

    ticker = str(df["ticker"].iloc[0])
    dmin = str(df["date"].min())
    dmax = str(df["date"].max())
    basename_template = f"{ticker}_{dmin}_{dmax}_{{i}}.parquet"

    ds.write_dataset(
        table,
        base_dir=str(base_dir),
        format="parquet",
        partitioning=partitioning,
        basename_template=basename_template,
        existing_data_behavior="overwrite_or_ignore",
    )

    return table.num_rows


def append_features_dataset(df: pd.DataFrame, base_dir: Path) -> int:
    """Append a batch to the features dataset (partitioned by feature/ticker/year)."""
    if df is None or df.empty:
        return 0

    df = _ensure_year_column(df)
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

    ticker = str(df["ticker"].iloc[0])
    feature_name = str(df["feature"].iloc[0])
    dmin = str(df["date"].min())
    dmax = str(df["date"].max())
    basename_template = f"{ticker}_{dmin}_{dmax}_{feature_name}_{{i}}.parquet"

    ds.write_dataset(
        table,
        base_dir=str(base_dir),
        format="parquet",
        partitioning=partitioning,
        basename_template=basename_template,
        existing_data_behavior="overwrite_or_ignore",
    )

    return table.num_rows


def load_prices_dataset(
    base_dir: Path,
    tickers: list[str] | None = None,
    columns: list[str] | None = None,
    clean: bool = True,
) -> pd.DataFrame:
    """Load the prices parquet dataset, optionally filtered by tickers."""
    dataset = ds.dataset(str(base_dir), format="parquet", partitioning="hive")

    filt = None
    if tickers:
        tickers = [t.upper().strip() for t in tickers]
        filt = ds.field("ticker").isin(tickers)

    table = dataset.to_table(filter=filt, columns=columns)
    df = table.to_pandas()

    if clean:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        required = [c for c in ["date", "ticker", "adj_close"] if c in df.columns]
        if required:
            df = df.dropna(subset=required)
        sort_cols = [c for c in ["date", "ticker"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
    return df


def load_weights_dataset(base_dir: Path, run_id: str | None = None) -> pd.DataFrame:
    """Load target weights from parquet, optionally filtered by run_id."""
    dataset = ds.dataset(str(base_dir), format="parquet", partitioning="hive")
    schema_names = set(dataset.schema.names)
    cols = ["date", "ticker", "weight"]
    if "run_id" in schema_names:
        cols.append("run_id")
    if run_id is not None:
        if "run_id" not in schema_names:
            raise KeyError("weights dataset has no run_id column.")
        run_id_field = dataset.schema.field("run_id").type
        if pa.types.is_integer(run_id_field):
            filt = ds.field("run_id") == int(run_id)
        else:
            filt = ds.field("run_id") == str(run_id)
        table = dataset.to_table(filter=filt, columns=cols)
    else:
        table = dataset.to_table(columns=cols)
    df = table.to_pandas()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "ticker", "weight"]).sort_values(["date", "ticker"])
    return df


def _load_dataset_with_date(
    base_dir: Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    df = read_parquet_dataset(base_dir, columns=columns)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_regimes_dataset(
    base_dir: Path,
    columns: list[str] | None = None,
    clean: bool = True,
) -> pd.DataFrame:
    """Load the regimes parquet dataset."""
    df = _load_dataset_with_date(base_dir, columns=columns)
    if clean:
        required = [c for c in (columns or ["date"]) if c in df.columns]
        if required:
            df = df.dropna(subset=required)
        sort_cols = [c for c in ["date", "ticker"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
    return df


def load_mc_dataset(
    base_dir: Path,
    columns: list[str] | None = None,
    clean: bool = True,
) -> pd.DataFrame:
    """Load the MC parquet dataset."""
    df = _load_dataset_with_date(base_dir, columns=columns)
    if clean:
        required = [c for c in (columns or ["date"]) if c in df.columns]
        if required:
            df = df.dropna(subset=required)
        sort_cols = [c for c in ["date", "ticker"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
    return df


def load_regime_features(base_dir: Path) -> pd.DataFrame:
    """Load regime feature dataset from parquet."""
    return _load_dataset_with_date(base_dir)


def load_asset_features(base_dir: Path) -> pd.DataFrame:
    """Load asset features dataset from parquet."""
    return _load_dataset_with_date(base_dir)


def write_features_dataset(
    df: pd.DataFrame,
    base_dir: Path,
    partition_cols: list[str],
    existing_data_behavior: str = "overwrite_or_ignore",
    basename_template: str | None = None,
) -> None:
    """Write features to a partitioned parquet dataset."""
    write_partitioned_dataset(
        df,
        base_dir,
        partition_cols,
        existing_data_behavior=existing_data_behavior,
        basename_template=basename_template,
    )


def write_regimes_dataset(
    df: pd.DataFrame,
    base_dir: Path,
    partition_cols: list[str],
    existing_data_behavior: str = "overwrite_or_ignore",
    basename_template: str | None = None,
) -> None:
    """Write regimes to a partitioned parquet dataset."""
    write_partitioned_dataset(
        df,
        base_dir,
        partition_cols,
        existing_data_behavior=existing_data_behavior,
        basename_template=basename_template,
    )


def write_mc_dataset(
    df: pd.DataFrame,
    base_dir: Path,
    partition_cols: list[str],
    existing_data_behavior: str = "overwrite_or_ignore",
    basename_template: str | None = None,
) -> None:
    """Write Monte Carlo summaries to a partitioned parquet dataset."""
    write_partitioned_dataset(
        df,
        base_dir,
        partition_cols,
        existing_data_behavior=existing_data_behavior,
        basename_template=basename_template,
    )


def write_weights_dataset(
    weights: pd.DataFrame,
    base_dir: Path,
    partition_cols: list[str],
    run_id: str | None = None,
    existing_data_behavior: str = "overwrite_or_ignore",
) -> str:
    """Write weights to a partitioned parquet dataset."""
    if weights is None or weights.empty:
        raise ValueError("weights empty")

    df = weights.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "weight"])
    if run_id is None:
        run_id = str(int(time.time()))
    df["run_id"] = str(run_id)
    df = _ensure_year_column(df)

    write_partitioned_dataset(
        df,
        base_dir,
        partition_cols,
        existing_data_behavior=existing_data_behavior,
    )
    return str(run_id)
