import sqlite3
from datetime import datetime, timedelta
import logging

import pandas as pd
import yfinance as yf

from quant_portfolio.core.db import (
    get_last_price_date,
    init_prices_last_dates_db,
    upsert_price_last_dates,
)
from quant_portfolio.core.settings import load_base_config
from quant_portfolio.core.storage import append_prices_dataset


LOGGER = logging.getLogger(__name__)
BASE_CONFIG = load_base_config()
out_dir = BASE_CONFIG.parquet_dir / "prices"
out_dir.mkdir(parents=True, exist_ok=True)
DB_PATH = BASE_CONFIG.metadata_db
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
tickers_2024 = list(BASE_CONFIG.tickers)  # Compatibility alias for existing notebooks.




def download_one(ticker: str, start: str | None, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV data from yfinance for a ticker."""
    # auto_adjust=False pour garder Adj Close
    # group_by='column' => si MultiIndex, le niveau 0 = champ (Open/High/...), niveau 1 = ticker
    return yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        interval="1d",
        actions=False,
        group_by="column",
    )


def normalize_yf(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize a yfinance OHLCV frame to our canonical schema.

    Handles both:
      - Standard columns: Open/High/Low/Close/Adj Close/Volume
      - MultiIndex columns (when yfinance returns (field, ticker))

    Output columns:
      ticker, date (YYYY-MM-DD), year (int32), open, high, low, close, adj_close, volume (int64)
    """

    cols = ["ticker", "date", "year", "open", "high", "low", "close", "adj_close", "volume"]

    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    # --- Flatten/Select in case yfinance returns MultiIndex columns ---
    if isinstance(df.columns, pd.MultiIndex):
        # Common case: columns are (field, ticker)
        lvl_last = df.columns.get_level_values(-1)
        if ticker in set(lvl_last):
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)
        else:
            # Fallback: keep first level names
            df = df.copy()
            df.columns = [c[0] for c in df.columns]

    # Ensure index is the date
    df = df.copy()
    df.index.name = "date"
    out = df.reset_index()

    # Normalize column names (case/spacing)
    def _norm(c: object) -> str:
        s = str(c).strip()
        s = s.replace("Adj Close", "Adj_Close")
        return s.lower().replace(" ", "_")

    out.columns = [_norm(c) for c in out.columns]

    # Some yfinance variants may use 'adjclose'
    if "adjclose" in out.columns and "adj_close" not in out.columns:
        out = out.rename(columns={"adjclose": "adj_close"})

    required = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    if not required.issubset(set(out.columns)):
        return pd.DataFrame(columns=cols)

    # Parse dates first, drop invalid rows BEFORE casting year
    dt = pd.to_datetime(out["date"], errors="coerce")
    ok = dt.notna()
    out = out.loc[ok].copy()
    dt = dt.loc[ok]

    out["ticker"] = ticker
    out["date"] = dt.dt.strftime("%Y-%m-%d")
    out["year"] = dt.dt.year.astype("int32")

    # Ensure numeric dtypes
    for c in ["open", "high", "low", "close", "adj_close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    vol = out["volume"]
    # If duplicate columns created a DataFrame slice, take the first one
    if isinstance(vol, pd.DataFrame):
        vol = vol.iloc[:, 0]
    out["volume"] = pd.to_numeric(vol, errors="coerce").fillna(0).astype("int64")

    return out[cols]



def run_ingest_pipeline() -> None:
    """Run the ingestion pipeline for the static ticker list."""
    conn = sqlite3.connect(DB_PATH)
    init_prices_last_dates_db(conn)

    total = 0
    for t in BASE_CONFIG.tickers:
        last = get_last_price_date(conn, t)
        if last is None:
            start = BASE_CONFIG.start_date
        else:
            start_dt = datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)
            start = start_dt.strftime("%Y-%m-%d")

        df = download_one(t, start=start, end=BASE_CONFIG.end_date)
        df2 = normalize_yf(df, t)

        inserted = append_prices_dataset(df2, out_dir)
        new_last = upsert_price_last_dates(conn, df2)
        total += inserted

        LOGGER.info(
            "%s: last=%s start=%s rows=%d new_last=%s",
            t,
            last,
            start,
            inserted,
            new_last,
        )

    conn.close()
    LOGGER.info("Ingestion complete: %d rows written", total)


def main() -> None:
    """Backward-compatible script entry point."""
    run_ingest_pipeline()


if __name__ == "__main__":
    main()
