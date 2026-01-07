"""
Pipeline: regime detection

TODO:
- load data/parquet/features/regime (hive)
- select feature columns (config/regimes.yaml)
- split train/validation by date
- standardize (z-score on train only, apply to full)
- fit model(s) via src/python/models/hmm.py
- produce state + probabilities per date
- write outputs to data/parquet/regimes (hive partitioned by year)
- store last processed date in SQLite (optional, like features.py)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import sqlite3
from models.hmm import fit_markov_market, fit_hmm_features, hmm_states_from_model, hmm_proba_from_model


ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "data/parquet/features/assets"
FEATURES_REGIME_DIR = ROOT / "data/parquet/features/regime"
REGIMES_DIR = ROOT / "data/parquet/regimes"
DB_PATH = Path("data/_meta.db")


def load_regime_features() -> pd.DataFrame:
    # TODO: load parquet dataset from FEATURES_DIR (hive) and parse date.

    dataset_assets = ds.dataset(FEATURES_DIR, format="parquet", partitioning="hive")
    dataset_regime = ds.dataset(FEATURES_REGIME_DIR, format="parquet", partitioning="hive")
    return dataset_assets.to_table().to_pandas(), dataset_regime.to_table().to_pandas()


def select_regime_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    # TODO: keep only numeric columns in feature_cols, drop rows with NaNs as needed.
    if df is None or df.empty:
        raise ValueError("X is empty.")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        raise ValueError("X has only NaNs after dropna.")
    if feature_cols not in df.columns:
        raise ValueError("Columns missins.")
    return df


def standardize_train_apply_all(
    df: pd.DataFrame,
    train_end: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    # TODO: compute mean/std on train slice, apply to full df (z-score).
    train = df[: train_end]
    test = df[train_end:]
    zscore = train.mean().std()
    return zscore, train, test


def fit_regime_model(df_z: pd.DataFrame, df_mkt: pd.DataFrame):
    # TODO: call models.hmm.fit_hmm_features or fit_markov_market depending on config.
    results_hmm_mkt = fit_markov_market(df_mkt)
    hmm_features = fit_hmm_features(df_z)
    return results_hmm_mkt, hmm_features


def build_regime_outputs(
    model,
    df_z: pd.DataFrame,
) -> pd.DataFrame:
    # TODO: compute state/proba using models.hmm helpers.
    states = hmm_states_from_model(model, df_z)
    proba = hmm_proba_from_model(model, df_z)
    return states, proba


def init_features_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS regimes_last_dates (
          feature   TEXT NOT NULL,
          ticker    TEXT NOT NULL,
          date      TEXT NOT NULL,  -- YYYY-MM-DD
          state     INTEGER NOT NULL
          proba     INTEGER NOT NULL
          PRIMARY KEY (feature, ticker)
        );
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_regimes_last_dates_feature ON feature_last_dates(feature);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_regimes_last_dates_ticker ON feature_last_dates(ticker);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_regimes_last_dates_date ON feature_last_dates(date);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_regimes_last_dates_state ON feature_last_dates(state);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_regimes_last_dates_proba ON feature_last_dates(proba);"
    )
    conn.commit()


def upsert_regime_last_dates(
    conn: sqlite3.Connection,
    feature: str,
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> None:
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
    row = conn.execute(
        "SELECT date FROM regimes_last_dates WHERE feature = ? AND ticker = ?",
        (feature, ticker),
    ).fetchone()
    return row[0] if row else None


def get_all_last_regime_dates(conn: sqlite3.Connection, feature: str) -> dict[str, str]:
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
    base_dir = REGIMES_DIR
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
    ds.write_dataset(
        table,
        base_dir=str(base_dir),
        format="parquet",
        partitioning=partitioning,
        existing_data_behavior=existing_data_behavior,
        basename_template=basename_template,
    )


def run_regime_pipeline(existing_data_behavior: str = "overwrite_or_ignore") -> None:
    # TODO: wire all steps: load -> select -> standardize -> fit -> outputs -> write.
    conn = sqlite3.connect(DB_PATH)
    init_features_db(conn)

    full_recompute = existing_data_behavior in {"overwrite", "delete_matching"}
    lookback_days = 260

    last_regime = get_last_regime_date(conn, "regime", "__MARKET__")
    last_assets = get_all_last_regime_dates(conn, "assets")

    df_assets, df_regimes = load_regime_features()
    df_assets = select_regime_features(df_assets)
    df_regimes = select_regime_features(df_regimes)

    zscore_assets, train_assets, test_assets = standardize_train_apply_all(df_assets, '01/01/2024')
    zscore_regimes, train_regimes, test_regimes = standardize_train_apply_all(df_regimes, '01/01/2024')

    results_hmm_mkt, hmm_features = fit_regime_model(train_assets, train_regimes)
    states, proba = build_regime_outputs(hmm_features, test_assets)

    upsert_regime_last_dates()
    write_regimes_dataset()

if __name__ == "__main__":
    # TODO: parse args (train_end, model type, feature list, etc.)
    run_regime_pipeline()
