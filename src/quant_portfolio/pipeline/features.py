import numpy as np
import pandas as pd
import sqlite3
import logging

from quant_portfolio.core.db import (
    init_feature_last_dates_db,
    upsert_feature_last_dates,
)
from quant_portfolio.core.settings import load_base_config
from quant_portfolio.core.storage import (
    load_asset_features,
    load_prices_dataset,
    load_regime_features,
    replace_partitioned_dataset,
)
from quant_portfolio.pipeline.data_quality import (
    build_feature_quality_report,
    build_price_quality_report,
    clean_price_data,
    convert_prices_to_reference_currency,
    write_json_report,
)


BASE_CONFIG = load_base_config()
DATA_DIR = BASE_CONFIG.data_dir
out_dir = BASE_CONFIG.parquet_dir / "features"
out_dir.mkdir(parents=True, exist_ok=True)

REGIME_DIR = out_dir / "regime"
ASSET_DIR = out_dir / "assets"
DB_PATH = DATA_DIR / "_meta.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
LOGGER = logging.getLogger(__name__)


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
    x = np.arange(window, dtype="float64")
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))

    def slope_from_window(y_window):
        return float(np.dot(x_centered, y_window - np.mean(y_window)) / denominator)

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

    breadth_up = returns.gt(0).where(returns.notna()).mean(axis=1, skipna=True)
    breadth_up_20 = breadth_up.rolling(20).mean()
    ma50 = prices.rolling(50).mean()
    valid_ma = prices.notna() & ma50.notna()
    breadth_ma50 = prices.gt(ma50).where(valid_ma).mean(axis=1, skipna=True)

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
    """Build scale-invariant equal-weight proxy from daily asset returns."""
    prices, _ = _pivot_prices_returns(df, tickers)
    simple_returns = prices.pct_change(fill_method=None)
    market_returns = simple_returns.mean(axis=1, skipna=True)
    market_returns = market_returns.where(simple_returns.notna().any(axis=1))
    index_level = 100.0 * (1.0 + market_returns.fillna(0.0)).cumprod()
    out = pd.DataFrame(index=prices.index)
    out["adj_close"] = index_level
    out["market_return"] = market_returns
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


def compute_feature_datasets(
    prices_frame: pd.DataFrame,
    tickers: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute complete causal regime and asset feature snapshots."""
    clean = clean_price_data(prices_frame)
    selected = list(tickers) if tickers is not None else None
    regime_out = regime_features(clean, selected).reset_index()
    if "index" in regime_out.columns and "date" not in regime_out.columns:
        regime_out = regime_out.rename(columns={"index": "date"})
    assets_out = asset_features(clean, selected).reset_index()
    regime_out["date"] = pd.to_datetime(regime_out["date"], errors="coerce")
    assets_out["date"] = pd.to_datetime(assets_out["date"], errors="coerce")
    return regime_out.sort_values("date"), assets_out.sort_values(["ticker", "date"])


def merge_feature_snapshots(
    existing: pd.DataFrame,
    computed: pd.DataFrame,
    key_columns: list[str],
    *,
    full_recompute: bool,
) -> pd.DataFrame:
    """Merge only unseen computed rows, producing idempotent feature snapshot."""
    if full_recompute or existing is None or existing.empty:
        result = computed.copy()
    elif key_columns == ["date"]:
        cutoff = pd.to_datetime(existing["date"], errors="coerce").max()
        new_rows = computed[pd.to_datetime(computed["date"], errors="coerce") > cutoff]
        result = pd.concat([existing, new_rows], ignore_index=True)
    else:
        existing_dates = existing.copy()
        existing_dates["date"] = pd.to_datetime(existing_dates["date"], errors="coerce")
        cutoffs = existing_dates.groupby("ticker")["date"].max()
        candidate = computed.copy()
        candidate["date"] = pd.to_datetime(candidate["date"], errors="coerce")
        cutoff = candidate["ticker"].map(cutoffs)
        new_rows = candidate[cutoff.isna() | candidate["date"].gt(cutoff)]
        result = pd.concat([existing, new_rows], ignore_index=True)

    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=key_columns)
    result = result.drop_duplicates(key_columns, keep="last")
    return result.sort_values(key_columns).reset_index(drop=True)


def _dataset_exists(path) -> bool:
    return path.exists() and any(path.rglob("*.parquet"))


def run_features_pipeline(existing_data_behavior: str = "overwrite_or_ignore") -> None:
    """Build exact incremental features from full causal history and audit quality."""
    if existing_data_behavior not in {
        "overwrite_or_ignore",
        "overwrite",
        "delete_matching",
        "error",
    }:
        raise ValueError(f"Unsupported existing_data_behavior: {existing_data_behavior}")

    conn = sqlite3.connect(DB_PATH)
    init_feature_last_dates_db(conn)

    full_recompute = existing_data_behavior in {"overwrite", "delete_matching"}
    if existing_data_behavior == "error" and (
        _dataset_exists(REGIME_DIR) or _dataset_exists(ASSET_DIR)
    ):
        conn.close()
        raise FileExistsError("Feature dataset already exists")

    try:
        LOGGER.info("Loading full price history for exact incremental computation")
        raw_prices = load_prices_dataset(
            BASE_CONFIG.parquet_dir / "prices",
            tickers=list(BASE_CONFIG.tickers),
            clean=False,
        )
        raw_fx = (
            load_prices_dataset(
                BASE_CONFIG.parquet_dir / "fx",
                tickers=[rate.ticker for rate in BASE_CONFIG.universe.fx_rates],
                clean=False,
            )
            if _dataset_exists(BASE_CONFIG.parquet_dir / "fx")
            else pd.DataFrame()
        )
        price_quality = build_price_quality_report(
            raw_prices,
            BASE_CONFIG.tickers,
            configured_statuses={
                asset.ticker: asset.status for asset in BASE_CONFIG.universe.assets
            },
        )
        fx_quality = build_price_quality_report(
            raw_fx,
            [rate.ticker for rate in BASE_CONFIG.universe.fx_rates],
        )
        converted = convert_prices_to_reference_currency(
            raw_prices,
            raw_fx,
            BASE_CONFIG.universe,
        )
        missing_fx_rows = int(converted["adj_close"].isna().sum())
        converted = clean_price_data(converted)
        if converted.empty:
            raise ValueError("No valid reference-currency prices available for features")

        LOGGER.info("Calculating features from %d validated price rows", len(converted))
        regime_computed, assets_computed = compute_feature_datasets(
            converted,
            BASE_CONFIG.tickers,
        )

        regime_existing = (
            load_regime_features(REGIME_DIR) if _dataset_exists(REGIME_DIR) else pd.DataFrame()
        )
        assets_existing = (
            load_asset_features(ASSET_DIR) if _dataset_exists(ASSET_DIR) else pd.DataFrame()
        )
        regime_out = merge_feature_snapshots(
            regime_existing,
            regime_computed,
            ["date"],
            full_recompute=full_recompute,
        )
        assets_out = merge_feature_snapshots(
            assets_existing,
            assets_computed,
            ["ticker", "date"],
            full_recompute=full_recompute,
        )

        LOGGER.info("Replacing idempotent feature snapshots")
        replace_partitioned_dataset(regime_out, REGIME_DIR, ["year"])
        replace_partitioned_dataset(assets_out, ASSET_DIR, ["ticker", "year"])

        regime_freshness = regime_out[["date"]].copy()
        regime_freshness["ticker"] = "__MARKET__"
        upsert_feature_last_dates(conn, "regime", regime_freshness)
        upsert_feature_last_dates(conn, "assets", assets_out)

        feature_quality = build_feature_quality_report(
            regime_out,
            assets_out,
            universe=BASE_CONFIG.universe,
        )
        feature_quality["prices"] = price_quality
        feature_quality["fx"] = fx_quality
        feature_quality["fx"]["missing_converted_rows"] = missing_fx_rows
        write_json_report(
            feature_quality,
            BASE_CONFIG.data_dir / "quality/features_latest.json",
        )
        LOGGER.info(
            "Features complete: regime_rows=%d asset_rows=%d missing_fx_rows=%d",
            len(regime_out),
            len(assets_out),
            missing_fx_rows,
        )
    finally:
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
