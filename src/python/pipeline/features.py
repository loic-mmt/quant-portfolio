import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import linear_model
import pyarrow as pa
import pyarrow.dataset as ds
import shutil
from scipy.stats import skew, kurtosis, variation, Covariance


out_dir = Path("data/parquet/features")
CLEAN_PARQUET = False  # set True only if you want to reset the dataset
if CLEAN_PARQUET and out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ==================== Market Regimes ====================
# ========================================================

def compute_returns(df):
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    prices = df["adj_close"] if "adj_close" in df.columns else df.iloc[:, 0]
    return np.log(prices / prices.shift(1))



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
    returns = prices.pct_change()
    return prices, returns


def _avg_corr_series(returns: pd.DataFrame, window: int) -> pd.Series:
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
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)
    _, returns = _pivot_prices_returns(df, tickers)

    out = pd.DataFrame(index=returns.index)
    out["avg_corr_60"] = _avg_corr_series(returns, 60)
    out["avg_corr_20"] = _avg_corr_series(returns, 20)
    return out


def dispersion(df: pd.DataFrame, tickers: list[str] | None = None):
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)
    _, returns = _pivot_prices_returns(df, tickers)

    daily_disp = returns.std(axis=1, skipna=True)
    out = pd.DataFrame(index=returns.index)
    out["disp_20"] = daily_disp.rolling(20).mean()
    out["disp_60"] = daily_disp.rolling(60).mean()
    return out


def breadth(df: pd.DataFrame, tickers: list[str] | None = None):
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
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    correlation = mean_corr(df, tickers)
    return correlation["avg_corr_20"] - correlation["avg_corr_60"]


# ==================== Market Proxy ======================
# ========================================================


def build_market_index(df: pd.DataFrame, tickers: list[str] | None = None):
    prices, _ = _pivot_prices_returns(df, tickers)
    daily_mean_prices = prices.mean(axis=1, skipna=True)
    out = pd.DataFrame(index=prices.index)
    out["adj_close"] = daily_mean_prices
    return out


# ==================== Regime Features ===================
# ========================================================


def regime_features(df: pd.DataFrame, tickers: list[str] | None = None):
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
            [mom, vol, dd, downside_vol, beta_idio, liq],
            axis=1,
        )
        feat["ticker"] = ticker
        feat = feat.set_index("ticker", append=True)
        out_list.append(feat)

    out = pd.concat(out_list).sort_index()
    return out


def beta_idio_features(df: pd.DataFrame, mkt_returns: pd.Series, window: int = 60):
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
