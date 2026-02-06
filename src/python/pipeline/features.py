import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import linear_model
import shutil
import sqlite3
import sys
import time
from scipy.stats import skew, kurtosis

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from core.db import (
    get_all_last_feature_dates,
    get_last_feature_date,
    init_feature_last_dates_db,
    upsert_feature_last_dates,
)
from core.storage import load_prices_dataset, write_features_dataset


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"


out_dir = DATA_DIR / "parquet/features"
CLEAN_PARQUET = False  # set True only if you want to reset the dataset
if CLEAN_PARQUET and out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

REGIME_DIR = out_dir / "regime"
ASSET_DIR = out_dir / "assets"
DB_PATH = DATA_DIR / "_meta.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ==================== Market Regimes ====================
# ========================================================

def compute_returns(df):
    """Compute log returns from a price series or frame."""
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    prices = df["adj_close"] if "adj_close" in df.columns else df.iloc[:, 0]
    return np.log(prices / prices.shift(1))



prices = DATA_DIR / "parquet/prices"
def momentum(df, batch: str = "all"):
    """Compute rolling momentum windows over log returns."""
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
    """Estimate 60-day linear trend slope of log prices."""
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
    """Compute distance to moving averages (50/200)."""
    batch_params = {
        "50 jours": 50,
        "200 jours": 200,
        "all": "all",
    }
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)
    if batch not in batch_params:
        raise KeyError("Please choose between: 50 jours, 200 jours, all")

    prices = df["adj_close"] if "adj_close" in df.columns else df.iloc[:, 0]
    MA20 = prices.rolling(20).mean()
    MA50 = prices.rolling(50).mean()
    MA60 = prices.rolling(60).mean()
    MA200 = prices.rolling(200).mean()
    
    out = pd.DataFrame(index=df.index)
    outMA = pd.DataFrame(index=df.index)
    outMA["ma20"] = MA20
    outMA["ma50"] = MA50
    outMA["ma60"] = MA60
    outMA["ma200"] = MA200

    out["dist_ma_50"] = (prices/MA50)-1
    out["dist_ma_200"] = (prices/MA200)-1

    if batch == "all":
        return out, outMA
    window = batch_params[batch]
    return out[[f"dist_ma_{window}"]], outMA



def volatility(df):
    """Compute rolling volatility and related measures."""
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)

    returns = compute_returns(df)
    vol_20 = returns.rolling(20).std()
    vol_60 = returns.rolling(60).std()
    emwa_vol = returns.ewm(alpha=0.06).std()
    vol_of_vol_20 = vol_20.rolling(20).std()
    
    out = pd.DataFrame(index=df.index)
    out["vol_20"] = vol_20
    out["vol_60"] = vol_60
    out["emwa_vol"] = emwa_vol
    out["vol_of_vol_20"] = vol_of_vol_20

    return out


def drawdown(df):
    """Compute drawdown metrics over rolling windows."""
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)

    prices = df["adj_close"] if "adj_close" in df.columns else df.iloc[:, 0]
    roll_max_60 = prices.rolling(60).max()
    dd_60 = 1 - prices / roll_max_60
    roll_max_252 = prices.rolling(252).max()
    dd_252 = 1 - prices / roll_max_252
    mdd_252 = dd_252.rolling(252).max()
    
    out = pd.DataFrame(index=df.index)
    out["dd_60"] = dd_60
    out["mdd_252"] = mdd_252

    return out


def asymetry(df):
    """Compute rolling skewness and kurtosis of returns."""
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)

    returns = compute_returns(df)

    skew_60 = returns.rolling(60).skew()
    kurt_60 = returns.rolling(60).kurt()
    
    out = pd.DataFrame(index=df.index)
    out["skew_60"] = skew_60
    out["kurt_60"] = kurt_60

    return out


# ==================== Cross-Section =====================
# ========================================================


def _pivot_prices_returns(df: pd.DataFrame, tickers: list[str] | None = None):
    """Pivot long prices to wide price and return matrices."""
    if df is None or df.empty:
        return (
            pd.DataFrame(index=df.index if df is not None else None),
            pd.DataFrame(index=df.index if df is not None else None),
        )
    required = {"date", "ticker", "adj_close"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns: {missing}")

    data = df.copy()
    if tickers is not None:
        data = data[data["ticker"].isin(tickers)]

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    prices = (
        data.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
        .sort_index()
    )
    returns = np.log(prices / prices.shift(1))
    return prices, returns


def _avg_corr_series(returns: pd.DataFrame, window: int) -> pd.Series:
    """Compute rolling average pairwise correlation."""
    if returns is None or returns.empty:
        return pd.Series(dtype="float64")
    
    values = returns.values
    out = np.full(len(returns), np.nan, dtype="float64")

    for idx in range(window - 1, len(returns)):
        window_data = values[idx - window + 1 : idx + 1]
        valid_cols = ~np.all(np.isnan(window_data), axis=0)
        window_data = window_data[:, valid_cols]
        
        if window_data.shape[1] < 2:
            continue
        corr = pd.DataFrame(window_data).corr().to_numpy()
        np.fill_diagonal(corr, np.nan)
        out[idx] = np.nanmean(corr)

    return pd.Series(out, index=returns.index)


def mean_corr(df: pd.DataFrame, tickers: list[str] | None = None):
    """Compute mean correlation measures across assets."""
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)
    _, returns = _pivot_prices_returns(df, tickers)

    out = pd.DataFrame(index=returns.index)
    out["avg_corr_60"] = _avg_corr_series(returns, 60)
    out["avg_corr_20"] = _avg_corr_series(returns, 20)
    return out


def dispersion(df: pd.DataFrame, tickers: list[str] | None = None):
    """Compute cross-sectional return dispersion."""
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)
    _, returns = _pivot_prices_returns(df, tickers)

    daily_disp = returns.std(axis=1, skipna=True)
    out = pd.DataFrame(index=returns.index)
    out["disp_20"] = daily_disp.rolling(20).mean()
    out["disp_60"] = daily_disp.rolling(60).mean()
    return out


def breadth(df: pd.DataFrame, tickers: list[str] | None = None):
    """Compute market breadth measures."""
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)
    prices, returns = _pivot_prices_returns(df, tickers)

    breadth_up = (returns > 0).mean(axis=1, skipna=True)
    breadth_up_20 = breadth_up.rolling(20).mean()
    ma50 = prices.rolling(50).mean()
    breadth_ma50 = (prices > ma50).mean(axis=1, skipna=True)

    out = pd.DataFrame(index=returns.index)
    out["breadth_up"] = breadth_up
    out["breadth_up_20"] = breadth_up_20
    out["breadth_ma50"] = breadth_ma50
    return out


def correlation_shock(df: pd.DataFrame, tickers: list[str] | None = None):
    """Compute short-vs-long average correlation spread."""
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    correlation = mean_corr(df, tickers)
    return correlation["avg_corr_20"] - correlation["avg_corr_60"]


# ==================== Market Proxy ======================
# ========================================================


def build_market_index(df: pd.DataFrame, tickers: list[str] | None = None):
    """Build a simple equal-weight market proxy from prices."""
    prices, _ = _pivot_prices_returns(df, tickers)
    daily_mean_prices = prices.mean(axis=1, skipna=True)
    out = pd.DataFrame(index=prices.index)
    out["adj_close"] = daily_mean_prices
    return out


# ==================== Regime Features ===================
# ========================================================


def regime_features(df: pd.DataFrame, tickers: list[str] | None = None):
    """Compute regime-level features from the market proxy and breadth."""
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)

    mkt = build_market_index(df, tickers)

    mom = momentum(mkt).rename(     # préfixes mkt_ pour éviter les collisions.
        columns={
            "mom_20": "mom_mkt_20",
            "mom_60": "mom_mkt_60",
            "mom_252": "mom_mkt_252",
        }
    )
    slope = trend_slope_60(mkt).rename("trend_slope_60")
    dist, _ = dist_ma(mkt)
    dist = dist.rename(
        columns={
            "dist_ma_50": "dist_mkt_ma_50",
            "dist_ma_200": "dist_mkt_ma_200",
        }
    )
    vol = volatility(mkt).rename(
        columns={
            "vol_20": "vol_mkt_20",
            "vol_60": "vol_mkt_60",
            "emwa_vol": "ewma_vol_mkt",
            "vol_of_vol_20": "vol_of_vol_20",
        }
    )
    dd = drawdown(mkt).rename(
        columns={
            "dd_60": "dd_mkt_60",
            "mdd_252": "mdd_mkt_252",
        }
    )
    asym = asymetry(mkt).rename(
        columns={
            "skew_60": "skew_mkt_60",
            "kurt_60": "kurt_mkt_60",
        }
    )

    avg_corr = mean_corr(df, tickers)
    disp = dispersion(df, tickers)
    br = breadth(df, tickers)
    d_avg_corr = correlation_shock(df, tickers).rename("d_avg_corr")

    out = pd.concat(
        [
            mom,
            slope,
            dist,
            vol,
            dd,
            asym,
            avg_corr,
            disp,
            br,
            d_avg_corr,
        ],
        axis=1,
    ).sort_index()
    return out


# ==================== Per-Asset Features ================
# ========================================================


def asset_features(df: pd.DataFrame, tickers: list[str] | None = None):
    """Compute per-asset feature set for cross-sectional modeling."""
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)

    required = {"date", "ticker", "adj_close"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns: {missing}")

    data = df.copy()
    if tickers is not None:
        data = data[data["ticker"].isin(tickers)]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    data = data.sort_values(["ticker", "date"])

    mkt = build_market_index(data, tickers)
    mkt_returns = compute_returns(mkt)

    out_list = []
    for ticker, g in data.groupby("ticker", sort=False):
        g = g.sort_values("date")
        g = g.set_index("date")

        mom = momentum(g).rename(
            columns={
                "mom_20": "mom_i_20",
                "mom_60": "mom_i_60",
                "mom_252": "mom_i_252",
            }
        )
        slope = trend_slope_60(g).rename("trend_slope_i_60")
        dist, _ = dist_ma(g)
        dist = dist.rename(
            columns={
                "dist_ma_50": "dist_ma_i_50",
                "dist_ma_200": "dist_ma_i_200",
            }
        )
        vol = volatility(g).rename(
            columns={
                "vol_20": "vol_i_20",
                "vol_60": "vol_i_60",
                "emwa_vol": "ewma_vol_i",
                "vol_of_vol_20": "vol_of_vol_i_20",
            }
        )
        dd = drawdown(g).rename(
            columns={
                "dd_60": "dd_i_60",
                "mdd_252": "mdd_i_252",
            }
        )
        asym = asymetry(g).rename(
            columns={
                "skew_60": "skew_i_60",
                "kurt_60": "kurt_i_60",
            }
        )

        returns = compute_returns(g)
        downside_vol = returns.where(returns < 0).rolling(60).std()
        downside_vol = downside_vol.rename("downside_vol_i_60")

        beta_idio = beta_idio_features(g, mkt_returns, window=60).rename(
            columns={
                "beta_60": "beta_i_60",
                "idio_vol_60": "idio_vol_i_60",
            }
        )
        if "volume" in g.columns:
            adv = g["volume"].rolling(20).mean().rename("adv_i_20")
            dollar_volume = (g["adj_close"] * g["volume"]).rolling(20).mean()
            dollar_volume = dollar_volume.rename("dollar_volume_i_20")
            liq = pd.concat([adv, dollar_volume], axis=1)
        else:
            liq = pd.DataFrame(index=g.index)

        feat = pd.concat(
            [mom, slope, dist, vol, dd, asym, downside_vol, beta_idio, liq],
            axis=1,
        )
        feat["ticker"] = ticker
        feat = feat.set_index("ticker", append=True)
        out_list.append(feat)

    out = pd.concat(out_list).sort_index()
    return out


def beta_idio_features(df: pd.DataFrame, mkt_returns: pd.Series, window: int = 60):
    """Compute rolling beta and idiosyncratic volatility."""
    if df is None or df.empty or mkt_returns is None or mkt_returns.empty:
        return pd.DataFrame(index=df.index if df is not None else None)

    returns = compute_returns(df).rename("asset_ret")
    aligned = pd.concat([returns, mkt_returns.rename("mkt_ret")], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame(index=returns.index)

    cov = aligned["asset_ret"].rolling(window).cov(aligned["mkt_ret"])
    var = aligned["mkt_ret"].rolling(window).var()
    beta = cov / var

    resid = aligned["asset_ret"] - beta * aligned["mkt_ret"]
    idio_vol = resid.rolling(window).std()

    out = pd.DataFrame(index=aligned.index)
    out[f"beta_{window}"] = beta
    out[f"idio_vol_{window}"] = idio_vol
    return out


def liquidity_features(df: pd.DataFrame, window: int = 20):
    """Compute ADV and dollar volume liquidity features."""
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)

    required = {"volume", "ticker", "adj_close"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns: {missing}")

    data = df.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"])
        data = data.sort_values(["ticker", "date"])

    adv = (
        data.groupby("ticker", group_keys=False)["volume"]
        .rolling(window)
        .mean()
    )
    dollar_volume = (
        data["adj_close"] * data["volume"]
    ).groupby(data["ticker"], group_keys=False).rolling(window).mean()

    out = pd.DataFrame(index=data.index)
    out[f"adv_{window}"] = adv
    out[f"dollar_volume_{window}"] = dollar_volume
    return out


# ======================= Parquet ========================
# ========================================================

# ======================== Runner ========================
# ========================================================


def run_features_pipeline(existing_data_behavior: str = "overwrite_or_ignore") -> None:
    """Run the feature computation pipeline end-to-end."""
    conn = sqlite3.connect(DB_PATH)
    init_feature_last_dates_db(conn)

    full_recompute = existing_data_behavior in {"overwrite", "delete_matching"}
    lookback_days = 260

    last_regime = get_last_feature_date(conn, "regime", "__MARKET__")
    last_assets = get_all_last_feature_dates(conn, "assets")

    print(f'\nLoading Prices...')
    prices = load_prices_dataset(DATA_DIR / "parquet/prices", clean=False)
    if not full_recompute and (last_regime or last_assets):
        dates = [d for d in [last_regime, *last_assets.values()] if d is not None]
        if dates:
            min_last = pd.to_datetime(min(dates))
            start_date = min_last - pd.Timedelta(days=lookback_days)
            prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
            prices = prices[prices["date"] >= start_date]

    if full_recompute:
        if REGIME_DIR.exists():
            shutil.rmtree(REGIME_DIR)
        if ASSET_DIR.exists():
            shutil.rmtree(ASSET_DIR)

    print(f'\nCalculating regime features...')
    regime = regime_features(prices)
    if not full_recompute and last_regime:
        cutoff = pd.to_datetime(last_regime)
        regime = regime[regime.index > cutoff]
    regime_out = regime.reset_index().rename(columns={"index": "date"})

    print(f'\nCalculating asset features...')
    assets = asset_features(prices)
    assets_out = assets.reset_index()
    if not full_recompute and last_assets:
        assets_out["date"] = pd.to_datetime(assets_out["date"], errors="coerce")
        last_map = pd.Series(last_assets)
        cutoff = assets_out["ticker"].map(last_map).fillna(pd.Timestamp.min)
        assets_out = assets_out[assets_out["date"] > cutoff]

    print(f'\nNumber of NA values in regime data : \n{regime_out.isna().sum()}')
    print(f'\nNumber of NA values in assets data : \n{assets_out.isna().sum()}')
    suffix = str(int(time.time()))
    basename_template = f"features_{suffix}_{{i}}.parquet" if not full_recompute else None

    print(f'\nWriting regime features dataset...')
    write_features_dataset(
        regime_out,
        REGIME_DIR,
        partition_cols=["year"],
        existing_data_behavior=existing_data_behavior,
        basename_template=basename_template,
    )
    print(f'\nWriting assets features dataset...')
    write_features_dataset(
        assets_out,
        ASSET_DIR,
        partition_cols=["ticker", "year"],
        existing_data_behavior=existing_data_behavior,
        basename_template=basename_template,
    )

    print(f'\nWriting lasts features dates in the SQL Database...\n')
    if not regime_out.empty:
        regime_out["ticker"] = "__MARKET__"
        upsert_feature_last_dates(conn, "regime", regime_out, ticker_col="ticker")
    if not assets_out.empty:
        upsert_feature_last_dates(conn, "assets", assets_out, ticker_col="ticker")

    conn.close()


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

    run_features_pipeline(existing_data_behavior=args.existing_data_behavior)
