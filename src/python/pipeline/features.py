import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import linear_model
import pyarrow as pa
import pyarrow.dataset as ds
import shutil
from scipy.stats import skew, kurtosis


out_dir = Path("data/parquet/features")
CLEAN_PARQUET = False  # set True only if you want to reset the dataset
if CLEAN_PARQUET and out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ==================== Market Regimes ====================
# ========================================================

def compute_returns(df):  # à modifier
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
    # TODO: construire l'index marché via build_market_index.
    # TODO: calculer les features market regime sur cet index (mom, vol, ewma, dd, skew, etc.).
    # TODO: calculer les features cross-section (avg_corr, disp, breadth) sur l'univers complet.
    # TODO: fusionner toutes les features par date en un seul DataFrame.
    # TODO: renvoyer le DataFrame "regime_features" prêt pour le modèle de régimes.
    pass


# ==================== Per-Asset Features ================
# ========================================================


def asset_features(df: pd.DataFrame, tickers: list[str] | None = None):
    # TODO: utiliser groupby(ticker) pour calculer les features par action.
    # TODO: inclure vol_i_20/60, ewma_vol_i, mom_i_20/60/252, dd_i_60/252.
    # TODO: calculer downside_vol_i_60 (std des retours négatifs).
    # TODO: calculer beta_i_60 et idio_vol_i_60 vs index marché.
    # TODO: ajouter adv_20 et dollar_volume_20 si volume dispo.
    # TODO: retourner un DataFrame indexé par date/ticker avec les features.
    pass


def beta_idio_features(df: pd.DataFrame, mkt_returns: pd.Series, window: int = 60):
    # TODO: calculer beta rolling par ticker via cov(r_i, r_mkt)/var(r_mkt).
    # TODO: calculer idio_vol rolling via std des résidus (r_i - beta*r_mkt).
    # TODO: retourner un DataFrame avec colonnes beta_{window}, idio_vol_{window}.
    pass


def liquidity_features(df: pd.DataFrame, window: int = 20):
    # TODO: vérifier la présence de volume.
    # TODO: calculer adv_{window} (moyenne du volume).
    # TODO: calculer dollar_volume_{window} (price * volume en moyenne).
    # TODO: retourner un DataFrame avec ces colonnes.
    pass


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
