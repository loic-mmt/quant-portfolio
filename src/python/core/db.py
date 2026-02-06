from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd


def init_prices_last_dates_db(conn: sqlite3.Connection) -> None:
    """Create the metadata table for tracking last ingested dates."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS last_dates (
          ticker     TEXT NOT NULL,
          date       TEXT NOT NULL,  -- YYYY-MM-DD
          open       REAL,
          high       REAL,
          low        REAL,
          close      REAL,
          adj_close  REAL,
          volume     INTEGER,
          PRIMARY KEY (ticker, date)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_last_dates_ticker ON last_dates(ticker);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_last_dates_date ON last_dates(date);")
    conn.commit()


def get_last_price_date(conn: sqlite3.Connection, ticker: str) -> str | None:
    """Return the last ingested date (YYYY-MM-DD) for a ticker."""
    row = conn.execute(
        "SELECT MAX(date) FROM last_dates WHERE ticker = ?",
        (ticker,),
    ).fetchone()
    return row[0]


def upsert_price_last_dates(conn: sqlite3.Connection, df: pd.DataFrame) -> str | None:
    """Store ONLY the latest available date for this ticker in SQLite."""
    if df is None or df.empty:
        return None

    df_last = df.sort_values("date").tail(1)
    last_date = df_last.iloc[0]["date"]

    sql = """
    INSERT INTO last_dates (ticker, date, open, high, low, close, adj_close, volume)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(ticker, date) DO UPDATE SET
      open      = excluded.open,
      high      = excluded.high,
      low       = excluded.low,
      close     = excluded.close,
      adj_close = excluded.adj_close,
      volume    = excluded.volume;
    """

    row = (
        df_last.iloc[0]["ticker"],
        df_last.iloc[0]["date"],
        float(df_last.iloc[0]["open"]) if pd.notna(df_last.iloc[0]["open"]) else None,
        float(df_last.iloc[0]["high"]) if pd.notna(df_last.iloc[0]["high"]) else None,
        float(df_last.iloc[0]["low"]) if pd.notna(df_last.iloc[0]["low"]) else None,
        float(df_last.iloc[0]["close"]) if pd.notna(df_last.iloc[0]["close"]) else None,
        float(df_last.iloc[0]["adj_close"]) if pd.notna(df_last.iloc[0]["adj_close"]) else None,
        int(df_last.iloc[0]["volume"]) if pd.notna(df_last.iloc[0]["volume"]) else None,
    )

    conn.execute(sql, row)
    conn.commit()
    return last_date


def init_feature_last_dates_db(conn: sqlite3.Connection) -> None:
    """Initialize SQLite table storing last feature dates."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_last_dates (
          feature   TEXT NOT NULL,
          ticker    TEXT NOT NULL,
          date      TEXT NOT NULL,  -- YYYY-MM-DD
          PRIMARY KEY (feature, ticker)
        );
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_feature_last_dates_feature ON feature_last_dates(feature);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_feature_last_dates_ticker ON feature_last_dates(ticker);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_feature_last_dates_date ON feature_last_dates(date);"
    )
    conn.commit()


def get_last_feature_date(conn: sqlite3.Connection, feature: str, ticker: str) -> str | None:
    """Fetch the last feature date for a given feature/ticker."""
    row = conn.execute(
        "SELECT date FROM feature_last_dates WHERE feature = ? AND ticker = ?",
        (feature, ticker),
    ).fetchone()
    return row[0] if row else None


def get_all_last_feature_dates(conn: sqlite3.Connection, feature: str) -> dict[str, str]:
    """Return a dict of last feature dates for all tickers."""
    rows = conn.execute(
        "SELECT ticker, date FROM feature_last_dates WHERE feature = ?",
        (feature,),
    ).fetchall()
    return {ticker: date for ticker, date in rows}


def upsert_feature_last_dates(
    conn: sqlite3.Connection,
    feature: str,
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> None:
    """Upsert the latest available feature date per ticker."""
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
    INSERT INTO feature_last_dates (feature, ticker, date)
    VALUES (?, ?, ?)
    ON CONFLICT(feature, ticker) DO UPDATE SET
      date = excluded.date;
    """
    for ticker, last_date in last_by_ticker.items():
        conn.execute(sql, (feature, str(ticker), str(last_date)))
    conn.commit()


def init_regime_last_dates_db(conn: sqlite3.Connection) -> None:
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


def init_backtest_db(conn: sqlite3.Connection) -> None:
    """Initialize the backtests metadata table in SQLite."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtests (
          date_debut     TEXT NOT NULL,
          date_fin       TEXT NOT NULL,  -- YYYY-MM-DD
          CAGR                  REAL,
          volatility            REAL,
          Sharpe                REAL,
          max_drawdown          REAL,
          turnover_annualised   REAL,
          turnover_mean         REAL,
          turnover_vol          REAL,
          run_id                TEXT PRIMARY KEY
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_backtests_run_id ON backtests(run_id);")
    conn.commit()


def upsert_backtest_summary(
    conn: sqlite3.Connection,
    summary: dict[str, float],
    run_id: str,
    date_start: str,
    date_end: str,
) -> None:
    """Insert or update a backtest summary row for a given run id."""
    if not summary:
        return
    sql = """
    INSERT INTO backtests (
      date_debut, date_fin, CAGR, volatility, Sharpe, max_drawdown,
      turnover_annualised, turnover_mean, turnover_vol, run_id
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(run_id) DO UPDATE SET
      date_debut = excluded.date_debut,
      date_fin = excluded.date_fin,
      CAGR = excluded.CAGR,
      volatility = excluded.volatility,
      Sharpe = excluded.Sharpe,
      max_drawdown = excluded.max_drawdown,
      turnover_annualised = excluded.turnover_annualised,
      turnover_mean = excluded.turnover_mean,
      turnover_vol = excluded.turnover_vol;
    """
    row = (
        date_start,
        date_end,
        float(summary.get("CAGR", np.nan)),
        float(summary.get("volatility", np.nan)),
        float(summary.get("Sharpe", np.nan)),
        float(summary.get("max_drawdown", np.nan)),
        float(summary.get("turnover_annualised", np.nan)),
        float(summary.get("turnover_mean", np.nan)),
        float(summary.get("turnover_vol", np.nan)),
        str(run_id),
    )
    conn.execute(sql, row)
    conn.commit()
