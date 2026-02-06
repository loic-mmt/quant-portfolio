import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import sys

import pandas as pd
import yfinance as yf

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from core.db import (
    get_last_price_date,
    init_prices_last_dates_db,
    upsert_price_last_dates,
)
from core.storage import append_prices_dataset


# --- Output dataset directory ---
out_dir = Path("data/parquet/prices")
CLEAN_PARQUET = False  # set True only if you want to reset the dataset
if CLEAN_PARQUET and out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

DB_PATH = Path("data/_meta.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


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



def main() -> None:
    """Run the ingestion pipeline for the static ticker list."""
    conn = sqlite3.connect(DB_PATH)
    init_prices_last_dates_db(conn)

    total = 0
    for t in tickers_2024:
        last = get_last_price_date(conn, t)
        if last is None:
            start = "2019-01-01"
        else:
            # on repart du lendemain (évite de recharger inutilement)
            start_dt = datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)
            start = start_dt.strftime("%Y-%m-%d")

        df = download_one(t, start=start)
        df2 = normalize_yf(df, t)

        new_last = upsert_price_last_dates(conn, df2)
        inserted = append_prices_dataset(df2, out_dir)
        total += inserted

        print(f"{t}: last={last} start={start} -> {inserted} rows  |  New last date: {new_last}")

    conn.close()
    print(f"Done. Total rows upserted: {total}")


if __name__ == "__main__":
    main()
