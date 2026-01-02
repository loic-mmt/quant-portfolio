import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
import sqlite3

# --- Output dataset directory ---
out_dir = Path("data/parquet/prices")
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

DB_PATH = Path("data/_meta.db")

tickers_2024 = [
    "PDD",          # PDD Holdings
    "SDRL",         # Seadrill
    "SMCI",         # Super Micro Computer
    "WHD",          # Cactus
    "DHT",          # DHT Holdings
    "EG",           # Everest Group
    "NFG",          # National Fuel Gas
    "PTEN",         # Patterson-UTI Energy
    "PFS",          # Provident Financial Services
    "LYC.AX",       # Lynas Rare Earths
    "PRN.AX",       # Perenti
    "GSY.TO",       # goeasy
    "BCHN.SW",      # Burckhardt Compression
    "CFR.SW",       # Richemont
    "DOC.VI",       # DO & CO
    "MAIRE.MI",     # Maire
    "MAP.MC",       # Mapfre
    "RHM.DE",       # Rheinmetall
    "RYAAY",        # Ryanair (ADR US)
    "AEG",          # Aegon (ADR US)
    "BRNL.AS",      # Brunel International
    "COK.DE",       # Cancom
    "CBK.DE",       # Commerzbank
    "DIE.BR",       # D'Ieteren
    "FSKRS.HE",     # Fiskars
    "GTT.PA",       # GTT
    "GEST.MC",      # Gestamp
    "KSB.DE",       # KSB
    "SL.MI",        # Sanlorenzo
    "SBO.VI",       # SBO AG
    "HO.PA",        # Thales
    "FR.PA",        # Valeo
    "VCT.PA",       # Vicat
    "WAVE.PA",      # Wavestone
    "MONY.L",       # MONY Group
    "BEZ.L",        # Beazley
    "DRX.L",        # Drax
    "INCH.L",       # Inchcape
    "TBCG.L",       # TBC Bank Group
    "1908.HK",      # C&D International
    "3320.HK",      # China Resources Pharmaceutical
    "1138.HK",      # COSCO Shipping Energy Transportation
    "2367.HK",      # Giant Biogene
    "2005.HK",      # SSY Group
    "1585.HK",      # Yadea
    "RICHTER.BD",   # Richter Gedeon
    "ADMIE.AT",     # Admie Holding
    "6532.T",       # BayCurrent Consulting
    "7267.T",       # Honda
    "ELK.OL",       # Elkem
    "MOWI.OL",      # Mowi
    "SBNOR.OL",     # Sparebanken Norge
    "BFT.WA",       # Benefit Systems
    "UNIB-SDB.ST",  # Kindred Group
    "VOLCAR-B.ST",  # Volvo Car
]



def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS last_dates (
      ticker     TEXT NOT NULL,
      date       TEXT NOT NULL,
      open       REAL,
      high       REAL,
      low        REAL,
      close      REAL,
      adj_close  REAL,
      volume     INTEGER,
      PRIMARY KEY (ticker, date)
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);")
    conn.commit()


def get_last_date(conn: sqlite3.Connection, ticker: str) -> str | None:
    row = conn.execute(
        "SELECT MAX(date) FROM last_dates WHERE ticker = ?",
        (ticker,)
    ).fetchone()
    return row[0]  # "YYYY-MM-DD" ou None


def download_one(ticker: str, start: str | None, end: str | None = None) -> pd.DataFrame:
    # auto_adjust=False pour garder Adj Close
    return yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        interval="1d",
        actions=False,
    )


def upsert_last_dates(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df_max = df.where(datetime.max())
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
    rows = list(df_max.itertuples(index=False, name=None))
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def upsert_prices(df: pd.DataFrame):
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
    return len(table)

def normalize_yf(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    # yfinance -> colonnes standard: Open High Low Close Adj Close Volume
    out = df.reset_index().rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })
    out["ticker"] = ticker
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    # garde uniquement les colonnes utiles
    return out[["ticker","date","open","high","low","close","adj_close","volume"]]


def main():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    total = 0
    for t in tickers_2024:
        last = get_last_date(conn, t)
        if last is None:
            start = "2000-01-01"  # ou une date plus récente si tu veux
        else:
            # on repart du lendemain (évite de recharger inutilement)
            start_dt = datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)
            start = start_dt.strftime("%Y-%m-%d")

        df = download_one(t, start=start)
        df2 = normalize_yf(df, t)
        last_date = upsert_last_dates(conn, df2)
        inserted = upsert_prices(conn, df2)
        total += inserted
        print(f"{t}: last={last} start={start} -> {inserted} rows")

    conn.close()
    print(f"Done. Total rows upserted: {total}")