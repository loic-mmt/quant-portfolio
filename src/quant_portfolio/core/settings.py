from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "config/base.yaml"
VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


@dataclass(frozen=True)
class BaseConfig:
    """Configuration shared by all pipeline stages."""

    tickers: tuple[str, ...]
    data_dir: Path
    start_date: str
    end_date: str | None
    log_level: str
    random_seed: int

    @property
    def parquet_dir(self) -> Path:
        return self.data_dir / "parquet"

    @property
    def metadata_db(self) -> Path:
        return self.data_dir / "_meta.db"


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML mapping, returning an empty mapping for an empty file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return {}
    data = yaml.safe_load(content)
    if not isinstance(data, Mapping):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return dict(data)


def _normalise_tickers(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("base.yaml must define a non-empty 'tickers' list")
    tickers = tuple(str(item).strip().upper() for item in raw if str(item).strip())
    if not tickers:
        raise ValueError("The configured ticker universe is empty")
    duplicates = sorted({ticker for ticker in tickers if tickers.count(ticker) > 1})
    if duplicates:
        raise ValueError(f"Duplicate tickers in base.yaml: {duplicates}")
    return tickers


def load_base_config(path: Path | None = None) -> BaseConfig:
    """Load and validate the common project configuration."""
    config_path = path or DEFAULT_BASE_CONFIG
    data = load_yaml_mapping(config_path)
    allowed = {"tickers", "data_dir", "start_date", "end_date", "log_level", "random_seed"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown fields in {config_path.name}: {sorted(unknown)}")

    data_dir = Path(str(data.get("data_dir", "data")))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    start_date = str(data.get("start_date", "2019-01-01"))
    end_raw = data.get("end_date")
    end_date = None if end_raw in {None, ""} else str(end_raw)
    log_level = str(data.get("log_level", "INFO")).upper()
    if log_level not in VALID_LOG_LEVELS:
        raise ValueError(f"Invalid log level: {log_level}")

    random_seed = int(data.get("random_seed", 42))
    if random_seed < 0:
        raise ValueError("random_seed must be >= 0")

    return BaseConfig(
        tickers=_normalise_tickers(data.get("tickers")),
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        log_level=log_level,
        random_seed=random_seed,
    )


def configure_logging(level: str = "INFO", verbose: bool = False) -> None:
    """Configure a consistent log format for command-line execution."""
    effective_level = "DEBUG" if verbose else level.upper()
    logging.basicConfig(
        level=effective_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
