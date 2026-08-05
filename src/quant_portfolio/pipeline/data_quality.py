from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd

from quant_portfolio.core.universe import UniverseDefinition

if TYPE_CHECKING:
    from quant_portfolio.core.settings import BaseConfig


PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return canonical, deduplicated prices suitable for research calculations."""
    if df is None or df.empty:
        return pd.DataFrame(columns=list(df.columns) if df is not None else None)

    required = {"date", "ticker", "adj_close"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing price columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out["ticker"].astype("string").str.strip().str.upper()
    for column in [*PRICE_COLUMNS, "volume", "dividends", "stock_splits"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=["date", "ticker", "adj_close"])
    out = out[out["ticker"].ne("") & out["adj_close"].gt(0)]
    for column in PRICE_COLUMNS:
        if column in out.columns:
            out = out[out[column].isna() | out[column].gt(0)]

    out = out.sort_values(["ticker", "date"])
    out = out.drop_duplicates(["ticker", "date"], keep="last")
    return out.reset_index(drop=True)


def build_price_quality_report(
    raw: pd.DataFrame,
    expected_tickers: Iterable[str],
    *,
    configured_statuses: dict[str, str] | None = None,
    stale_after_days: int = 10,
) -> dict[str, Any]:
    """Audit duplicates, gaps, invalid prices, actions, coverage, and availability."""
    expected = tuple(str(ticker).strip().upper() for ticker in expected_tickers)
    statuses = configured_statuses or {}
    data = raw.copy() if raw is not None else pd.DataFrame()
    if data.empty:
        return {
            "summary": {
                "rows_raw": 0,
                "rows_clean": 0,
                "expected_tickers": len(expected),
                "available_tickers": 0,
                "missing_tickers": list(expected),
                "duplicate_rows": 0,
                "invalid_date_rows": 0,
                "non_positive_price_rows": 0,
                "corporate_action_rows": 0,
                "suspicious_adjustment_rows": 0,
            },
            "tickers": {
                ticker: {
                    "status": "delisted" if statuses.get(ticker) == "delisted" else "missing",
                    "configured_status": statuses.get(ticker, "unknown"),
                    "rows": 0,
                    "coverage": 0.0,
                    "missing_dates": 0,
                    "first_date": None,
                    "last_date": None,
                }
                for ticker in expected
            },
        }

    if "date" not in data.columns or "ticker" not in data.columns:
        raise KeyError("Price quality audit requires date and ticker columns")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["ticker"] = data["ticker"].astype("string").str.strip().str.upper()
    invalid_dates = int(data["date"].isna().sum())
    duplicates = int(data.duplicated(["ticker", "date"], keep=False).sum())

    non_positive_mask = pd.Series(False, index=data.index)
    for column in PRICE_COLUMNS:
        if column in data.columns:
            numeric = pd.to_numeric(data[column], errors="coerce")
            non_positive_mask |= numeric.notna() & numeric.le(0)

    action_mask = pd.Series(False, index=data.index)
    if "dividends" in data.columns:
        action_mask |= pd.to_numeric(data["dividends"], errors="coerce").fillna(0).ne(0)
    if "stock_splits" in data.columns:
        action_mask |= pd.to_numeric(data["stock_splits"], errors="coerce").fillna(0).ne(0)

    suspicious = pd.Series(False, index=data.index)
    if {"close", "adj_close"}.issubset(data.columns):
        close = pd.to_numeric(data["close"], errors="coerce")
        adjusted = pd.to_numeric(data["adj_close"], errors="coerce")
        ratio = close / adjusted.replace(0, np.nan)
        ratio_change = ratio.groupby(data["ticker"]).pct_change(fill_method=None).abs()
        suspicious = ratio_change.gt(0.20) & ~action_mask

    clean = clean_price_data(data)
    available = set(clean["ticker"].unique()) if not clean.empty else set()
    global_last = clean["date"].max() if not clean.empty else pd.NaT
    observed_calendar = pd.DatetimeIndex(sorted(clean["date"].unique()))

    ticker_report: dict[str, dict[str, Any]] = {}
    for ticker in expected:
        group = clean[clean["ticker"].eq(ticker)]
        configured = statuses.get(ticker, "unknown")
        if group.empty:
            status = "delisted" if configured == "delisted" else "missing"
            ticker_report[ticker] = {
                "status": status,
                "configured_status": configured,
                "rows": 0,
                "coverage": 0.0,
                "missing_dates": 0,
                "first_date": None,
                "last_date": None,
            }
            continue

        first = group["date"].min()
        last = group["date"].max()
        relevant_calendar = observed_calendar[
            (observed_calendar >= first) & (observed_calendar <= last)
        ]
        present = pd.DatetimeIndex(group["date"].unique())
        missing_dates = int(len(relevant_calendar.difference(present)))
        denominator = len(relevant_calendar)
        coverage = float(len(present) / denominator) if denominator else 1.0
        stale = pd.notna(global_last) and (global_last - last).days > stale_after_days
        status = "stale" if stale else ("partial" if coverage < 0.80 else "ok")
        ticker_report[ticker] = {
            "status": status,
            "configured_status": configured,
            "rows": int(len(group)),
            "coverage": coverage,
            "missing_dates": missing_dates,
            "first_date": first.strftime("%Y-%m-%d"),
            "last_date": last.strftime("%Y-%m-%d"),
        }

    missing_tickers = [ticker for ticker in expected if ticker not in available]
    return {
        "summary": {
            "rows_raw": int(len(data)),
            "rows_clean": int(len(clean)),
            "expected_tickers": len(expected),
            "available_tickers": len(available.intersection(expected)),
            "missing_tickers": missing_tickers,
            "duplicate_rows": duplicates,
            "invalid_date_rows": invalid_dates,
            "non_positive_price_rows": int(non_positive_mask.sum()),
            "corporate_action_rows": int(action_mask.sum()),
            "suspicious_adjustment_rows": int(suspicious.sum()),
        },
        "tickers": ticker_report,
    }


def convert_prices_to_reference_currency(
    prices: pd.DataFrame,
    fx_prices: pd.DataFrame,
    universe: UniverseDefinition,
    *,
    max_fx_staleness_days: int = 7,
) -> pd.DataFrame:
    """Convert local OHLC prices to universe reference currency using past FX only."""
    assets = universe.asset_by_ticker
    data = clean_price_data(prices)
    data = data[data["ticker"].isin(assets)].copy()
    if data.empty:
        return data

    fx = clean_price_data(fx_prices) if fx_prices is not None and not fx_prices.empty else pd.DataFrame()
    outputs: list[pd.DataFrame] = []
    for ticker, group in data.groupby("ticker", sort=False):
        asset = assets[ticker]
        converted = group.sort_values("date").copy()
        converted["currency"] = asset.currency
        converted["reference_currency"] = universe.reference_currency
        converted["price_multiplier"] = asset.price_multiplier
        for column in PRICE_COLUMNS:
            if column in converted.columns:
                converted[f"{column}_local"] = converted[column]
                converted[column] = converted[column] * asset.price_multiplier

        if asset.currency == universe.reference_currency:
            converted["fx_rate"] = 1.0
        else:
            rate_definition = universe.fx_by_currency[asset.currency]
            rate_rows = fx[fx["ticker"].eq(rate_definition.ticker)][["date", "adj_close"]]
            rate_rows = rate_rows.rename(columns={"adj_close": "fx_rate"}).sort_values("date")
            if rate_rows.empty:
                raise ValueError(
                    f"Missing FX data for {asset.currency}: {rate_definition.ticker}"
                )
            converted = pd.merge_asof(
                converted.sort_values("date"),
                rate_rows,
                on="date",
                direction="backward",
                tolerance=pd.Timedelta(days=max_fx_staleness_days),
            )
            for column in PRICE_COLUMNS:
                if column not in converted.columns:
                    continue
                if rate_definition.quote_currency_per_reference:
                    converted[column] = converted[column] / converted["fx_rate"]
                else:
                    converted[column] = converted[column] * converted["fx_rate"]
        outputs.append(converted)

    return pd.concat(outputs, ignore_index=True).sort_values(["ticker", "date"])


def load_reference_price_dataset(config: "BaseConfig") -> pd.DataFrame:
    """Load persisted assets and convert them to configured reference currency."""
    from quant_portfolio.core.storage import load_prices_dataset

    prices_dir = config.parquet_dir / "prices"
    fx_dir = config.parquet_dir / "fx"
    raw_prices = load_prices_dataset(
        prices_dir,
        tickers=list(config.tickers),
        clean=False,
    )
    fx_tickers = [rate.ticker for rate in config.universe.fx_rates]
    raw_fx = (
        load_prices_dataset(fx_dir, tickers=fx_tickers, clean=False)
        if fx_dir.exists() and any(fx_dir.rglob("*.parquet"))
        else pd.DataFrame()
    )
    converted = convert_prices_to_reference_currency(raw_prices, raw_fx, config.universe)
    return clean_price_data(converted)


def build_feature_quality_report(
    regime: pd.DataFrame,
    assets: pd.DataFrame,
    *,
    universe: UniverseDefinition,
) -> dict[str, Any]:
    """Build coverage, NaN, robust-outlier, and distribution-stability diagnostics."""

    def audit(frame: pd.DataFrame, identity_columns: set[str]) -> dict[str, Any]:
        numeric = frame.select_dtypes(include=[np.number]).drop(
            columns=list(identity_columns.intersection(frame.columns)), errors="ignore"
        )
        features: dict[str, Any] = {}
        for column in numeric.columns:
            series = numeric[column].replace([np.inf, -np.inf], np.nan)
            valid = series.dropna()
            median = float(valid.median()) if not valid.empty else None
            mad = float((valid - median).abs().median()) if not valid.empty else None
            outliers = 0
            if valid.size and mad and np.isfinite(mad):
                outliers = int(((valid - median).abs() > 8.0 * 1.4826 * mad).sum())

            if "date" in frame.columns:
                stability_frame = pd.DataFrame(
                    {
                        "date": pd.to_datetime(frame["date"], errors="coerce"),
                        "value": series,
                    }
                ).dropna()
                stability_series = (
                    stability_frame.groupby("date", sort=True)["value"].mean().sort_index()
                )
            else:
                stability_series = series
            stability_split = max(1, len(stability_series) // 2)
            old = stability_series.iloc[:stability_split].dropna()
            recent = stability_series.iloc[stability_split:].dropna()
            baseline_std = float(old.std(ddof=1)) if len(old) > 1 else np.nan
            stability = None
            if old.size and recent.size and np.isfinite(baseline_std) and baseline_std > 0:
                stability = float((recent.mean() - old.mean()) / baseline_std)
            features[column] = {
                "coverage": float(series.notna().mean()) if len(series) else 0.0,
                "nan_count": int(series.isna().sum()),
                "outlier_count": outliers,
                "stability_z_shift": stability,
            }
        dates = (
            pd.to_datetime(frame["date"], errors="coerce")
            if "date" in frame.columns
            else pd.Series(dtype="datetime64[ns]")
        )
        return {
            "rows": int(len(frame)),
            "first_date": dates.min().strftime("%Y-%m-%d") if dates.notna().any() else None,
            "last_date": dates.max().strftime("%Y-%m-%d") if dates.notna().any() else None,
            "features": features,
        }

    return {
        "universe": {
            "id": universe.universe_id,
            "version": universe.version,
            "fingerprint": universe.fingerprint,
            "reference_currency": universe.reference_currency,
        },
        "regime": audit(regime, {"year"}),
        "assets": audit(assets, {"year"}),
    }


def write_json_report(report: dict[str, Any], path: Path) -> None:
    """Write deterministic, human-readable JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
