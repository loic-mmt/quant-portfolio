from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import sqlite3
import time
import shutil
from typing import Any
from ..models.hmm import (
    fit_markov_market,
    fit_hmm_features,
    hmm_states_from_model,
    hmm_proba_from_model,
)


ROOT = Path(__file__).resolve().parents[3]
FEATURES_REGIME_DIR = ROOT / "data/parquet/features/regime"
REGIMES_DIR = ROOT / "data/parquet/regimes"
DB_PATH = ROOT / "data/_meta.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = ROOT / "config/regimes.yaml"

try:
    import yaml
except Exception:
    yaml = None


def load_regime_features() -> pd.DataFrame:
    """Load regime feature dataset from parquet."""
    dataset_regime = ds.dataset(str(FEATURES_REGIME_DIR), format="parquet", partitioning="hive")
    df_regime = dataset_regime.to_table().to_pandas()
    if "date" in df_regime.columns:
        df_regime["date"] = pd.to_datetime(df_regime["date"], errors="coerce")
    return df_regime


def load_regime_config() -> dict[str, Any]:
    """Load regimes YAML configuration or return empty dict."""
    if not CONFIG_PATH.exists():
        return {}
    content = CONFIG_PATH.read_text().strip()
    if not content:
        return {}
    if yaml is None:
        raise ImportError("PyYAML is required to parse config/regimes.yaml.")
    data = yaml.safe_load(content)
    return data if isinstance(data, dict) else {}


def select_regime_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Select and clean the feature columns used for regime modeling."""
    if df is None or df.empty:
        raise ValueError("X is empty.")
    if "date" in df.columns:
        df = df.sort_values("date").set_index("date")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.dropna()
    if X.empty:
        raise ValueError("X has only NaNs after dropna.")
    return X


def standardize_train_apply_all(
    df: pd.DataFrame,
    train_end: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Z-score features using train window statistics, applied to all dates."""
    train_end_dt = pd.to_datetime(train_end)
    train = df.loc[:train_end_dt]
    mean = train.mean()
    std = train.std().replace(0, np.nan)
    df_z = (df - mean) / std
    return df_z, mean, std


def fit_regime_model(df_z: pd.DataFrame, mkt_returns: pd.Series):
    """Fit the market and feature HMM models for regimes."""
    results_hmm_mkt = fit_markov_market(mkt_returns)
    hmm_features = fit_hmm_features(df_z)
    return results_hmm_mkt, hmm_features


def build_regime_outputs(model, df_z: pd.DataFrame) -> pd.DataFrame:
    """Build regime states and probabilities output frame."""
    states = hmm_states_from_model(model, df_z)
    proba = hmm_proba_from_model(model, df_z)
    return pd.concat([states, proba], axis=1)


def init_features_db(conn: sqlite3.Connection) -> None:
    """Initialize SQLite table storing last regime dates."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS regimes_last_dates (
          feature   TEXT NOT NULL,
          ticker    TEXT NOT NULL,
          date      TEXT NOT NULL,  -- YYYY-MM-DD
          state     INTEGER,
          proba     REAL,
          PRIMARY KEY (feature, ticker)
        );
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_regimes_last_dates_feature ON regimes_last_dates(feature);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_regimes_last_dates_ticker ON regimes_last_dates(ticker);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_regimes_last_dates_date ON regimes_last_dates(date);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_regimes_last_dates_state ON regimes_last_dates(state);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_regimes_last_dates_proba ON regimes_last_dates(proba);"
    )
    conn.commit()


def upsert_regime_last_dates(
    conn: sqlite3.Connection,
    feature: str,
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> None:
    """Upsert the latest available regime date per ticker."""
    if df is None or df.empty:
        return
    if ticker_col not in df.columns or date_col not in df.columns:
        missing = ", ".join(sorted({ticker_col, date_col} - set(df.columns)))
        raise KeyError(f"Missing columns: {missing}")

    dt = pd.to_datetime(df[date_col], errors="coerce")
    ok = dt.notna()
    df = df.loc[ok].copy()
    df[date_col] = dt.loc[ok].dt.strftime("%Y-%m-%d")

    last_by_ticker = df.groupby(ticker_col)[date_col].max()
    sql = """
    INSERT INTO regimes_last_dates (feature, ticker, date)
    VALUES (?, ?, ?)
    ON CONFLICT(feature, ticker) DO UPDATE SET
      date = excluded.date;
    """
    for ticker, last_date in last_by_ticker.items():
        conn.execute(sql, (feature, str(ticker), str(last_date)))
    conn.commit()



def get_last_regime_date(conn: sqlite3.Connection, feature: str, ticker: str) -> str | None:
    """Fetch the last regime date for a given feature/ticker."""
    row = conn.execute(
        "SELECT date FROM regimes_last_dates WHERE feature = ? AND ticker = ?",
        (feature, ticker),
    ).fetchone()
    return row[0] if row else None


def get_all_last_regime_dates(conn: sqlite3.Connection, feature: str) -> dict[str, str]:
    """Return a dict of last regime dates for all tickers."""
    rows = conn.execute(
        "SELECT ticker, date FROM regimes_last_dates WHERE feature = ?",
        (feature,),
    ).fetchall()
    return {ticker: date for ticker, date in rows}


def write_regimes_dataset(
    df: pd.DataFrame,
    base_dir: Path,
    partition_cols: list[str],
    existing_data_behavior: str = "overwrite_or_ignore",
    basename_template: str | None = None,
) -> None:
    """Write regimes to a partitioned parquet dataset."""
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


def run_regime_pipeline(existing_data_behavior: str = "overwrite_or_ignore") -> None:
    """Run the regime computation pipeline end-to-end."""
    conn = sqlite3.connect(DB_PATH)
    init_features_db(conn)

    full_recompute = existing_data_behavior in {"overwrite", "delete_matching"}
    last_regime = get_last_regime_date(conn, "regime", "__MARKET__")
    lookback_days = 260

    print(f'\nLoading Features...')
    df_regimes = load_regime_features()
    if "date" in df_regimes.columns and not full_recompute and last_regime:
        cutoff = pd.to_datetime(last_regime) - pd.Timedelta(days=lookback_days)
        df_regimes = df_regimes[df_regimes["date"] >= cutoff]

    print(f'\nLoading regime configuration...')
    cfg = load_regime_config()
    regime_cols = cfg.get("regime_features")
    if not isinstance(regime_cols, list) or not regime_cols:
        regime_cols = [c for c in df_regimes.columns if c not in {"date", "year"}]

    print(f'\Standardizing features...')
    X_regime = select_regime_features(df_regimes, regime_cols)
    X_regime_z, _, _ = standardize_train_apply_all(X_regime, cfg.get("train_end", "2024-01-01"))

    mkt_returns = df_regimes.set_index("date").get("mom_mkt_20")
    if mkt_returns is None:
        raise KeyError("mom_mkt_20 column is required for market returns.")
    mkt_returns = mkt_returns.dropna()

    print(f'\Computing regimes...')
    _, hmm_features = fit_regime_model(X_regime_z, mkt_returns)
    outputs = build_regime_outputs(hmm_features, X_regime_z)
    outputs = outputs.reset_index().rename(columns={"index": "date"})

    if not full_recompute and last_regime:
        outputs["date"] = pd.to_datetime(outputs["date"], errors="coerce")
        outputs = outputs[outputs["date"] > pd.to_datetime(last_regime)]

    if not outputs.empty:
        outputs["ticker"] = "__MARKET__"
        upsert_regime_last_dates(conn, "regime", outputs, ticker_col="ticker")

    if full_recompute and REGIMES_DIR.exists():
        shutil.rmtree(REGIMES_DIR)
    
    print("outputs shape:", outputs.shape)
    print(outputs.head(3))
    print(outputs.dtypes)


    print(f'\Writing regimes...')
    suffix = str(int(time.time()))
    basename_template = f"regimes_{suffix}_{{i}}.parquet" if not full_recompute else None
    write_regimes_dataset(
        outputs,
        REGIMES_DIR,
        partition_cols=["year"],
        existing_data_behavior=existing_data_behavior,
        basename_template=basename_template,
    )
    conn.close()
    print(f'\Done.')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute and write feature datasets.")
    parser.add_argument(
        "--existing-data-behavior",
        default="overwrite_or_ignore",
        choices=["overwrite_or_ignore", "overwrite", "error", "delete_matching"],
        help="Behavior when target dataset already has data.",
    )
    args = parser.parse_args()

    run_regime_pipeline(existing_data_behavior=args.existing_data_behavior)
