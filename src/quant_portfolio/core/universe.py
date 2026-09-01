from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml


VALID_ASSET_STATUSES = {"active", "delisted", "unknown"}
VALID_FX_POLICIES = {"convert_to_reference", "single_currency"}


@dataclass(frozen=True)
class AssetDefinition:
    """One versioned universe constituent."""

    ticker: str
    currency: str
    price_multiplier: float = 1.0
    status: str = "unknown"
    sector: str | None = None


@dataclass(frozen=True)
class FxRateDefinition:
    """FX series used to convert one local currency to reference currency."""

    currency: str
    ticker: str
    quote_currency_per_reference: bool = True


@dataclass(frozen=True)
class UniverseDefinition:
    """Immutable, versioned research universe loaded from YAML."""

    universe_id: str
    version: int
    as_of: str
    constituent_source: str
    survivorship_bias: str
    reference_currency: str
    fx_policy: str
    assets: tuple[AssetDefinition, ...]
    fx_rates: tuple[FxRateDefinition, ...]
    source_path: Path
    fingerprint: str

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(asset.ticker for asset in self.assets)

    @property
    def currencies(self) -> tuple[str, ...]:
        return tuple(sorted({asset.currency for asset in self.assets}))

    @property
    def asset_by_ticker(self) -> dict[str, AssetDefinition]:
        return {asset.ticker: asset for asset in self.assets}

    @property
    def fx_by_currency(self) -> dict[str, FxRateDefinition]:
        return {rate.currency: rate for rate in self.fx_rates}


def _read_mapping(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Universe configuration must be a YAML mapping: {path}")
    return dict(data), raw


def _required_text(data: Mapping[str, Any], key: str, path: Path) -> str:
    value = str(data.get(key, "")).strip()
    if not value:
        raise ValueError(f"Missing non-empty '{key}' in {path}")
    return value


def load_universe(path: Path) -> UniverseDefinition:
    """Load and validate a versioned universe definition."""
    if not path.exists():
        raise FileNotFoundError(f"Universe configuration not found: {path}")

    data, raw = _read_mapping(path)
    allowed = {
        "universe_id",
        "version",
        "as_of",
        "constituent_source",
        "survivorship_bias",
        "reference_currency",
        "fx_policy",
        "assets",
        "fx_rates",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown universe fields in {path.name}: {sorted(unknown)}")

    version = int(data.get("version", 0))
    if version < 1:
        raise ValueError("Universe version must be >= 1")

    reference_currency = _required_text(data, "reference_currency", path).upper()
    fx_policy = str(data.get("fx_policy", "convert_to_reference")).strip()
    if fx_policy not in VALID_FX_POLICIES:
        raise ValueError(f"Invalid fx_policy: {fx_policy}")

    raw_assets = data.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ValueError("Universe must define a non-empty 'assets' list")

    assets: list[AssetDefinition] = []
    for index, item in enumerate(raw_assets):
        if not isinstance(item, Mapping):
            raise ValueError(f"assets[{index}] must be a mapping")
        ticker = str(item.get("ticker", "")).strip().upper()
        currency = str(item.get("currency", "")).strip().upper()
        status = str(item.get("status", "unknown")).strip().lower()
        multiplier = float(item.get("price_multiplier", 1.0))
        if not ticker or not currency:
            raise ValueError(f"assets[{index}] requires ticker and currency")
        if status not in VALID_ASSET_STATUSES:
            raise ValueError(f"Invalid status for {ticker}: {status}")
        if multiplier <= 0:
            raise ValueError(f"price_multiplier must be positive for {ticker}")
        sector = item.get("sector")
        if sector is not None and (not isinstance(sector, str) or not sector.strip()):
            raise ValueError(f"sector must be a non-empty string for {ticker}")
        assets.append(AssetDefinition(ticker, currency, multiplier, status, sector.strip() if sector else None))

    tickers = [asset.ticker for asset in assets]
    duplicates = sorted({ticker for ticker in tickers if tickers.count(ticker) > 1})
    if duplicates:
        raise ValueError(f"Duplicate universe tickers: {duplicates}")

    raw_fx = data.get("fx_rates", {})
    if not isinstance(raw_fx, Mapping):
        raise ValueError("fx_rates must be a currency-to-definition mapping")
    fx_rates: list[FxRateDefinition] = []
    for currency, item in raw_fx.items():
        currency_code = str(currency).strip().upper()
        if isinstance(item, str):
            ticker = item.strip().upper()
            quote_per_reference = True
        elif isinstance(item, Mapping):
            ticker = str(item.get("ticker", "")).strip().upper()
            quote_per_reference = bool(item.get("quote_currency_per_reference", True))
        else:
            raise ValueError(f"Invalid FX definition for {currency_code}")
        if not ticker:
            raise ValueError(f"Missing FX ticker for {currency_code}")
        fx_rates.append(FxRateDefinition(currency_code, ticker, quote_per_reference))

    non_reference = {asset.currency for asset in assets if asset.currency != reference_currency}
    configured_fx = {rate.currency for rate in fx_rates}
    if fx_policy == "convert_to_reference":
        missing_fx = sorted(non_reference - configured_fx)
        if missing_fx:
            raise ValueError(f"Missing FX rates for currencies: {missing_fx}")
    elif non_reference:
        raise ValueError(
            "single_currency universe contains currencies other than reference_currency: "
            f"{sorted(non_reference)}"
        )

    return UniverseDefinition(
        universe_id=_required_text(data, "universe_id", path),
        version=version,
        as_of=_required_text(data, "as_of", path),
        constituent_source=_required_text(data, "constituent_source", path),
        survivorship_bias=_required_text(data, "survivorship_bias", path),
        reference_currency=reference_currency,
        fx_policy=fx_policy,
        assets=tuple(assets),
        fx_rates=tuple(fx_rates),
        source_path=path.resolve(),
        fingerprint=hashlib.sha256(raw).hexdigest(),
    )
