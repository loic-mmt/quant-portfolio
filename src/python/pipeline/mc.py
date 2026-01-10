from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import numpy.random as rd
import pyarrow as pa
import pyarrow.dataset as ds
import sqlite3
import time
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
FEATURES_ASSETS_DIR = ROOT / "data/parquet/features/assets"
PRICES_DIR = ROOT / "data/parquet/prices"
REGIMES_DIR = ROOT / "data/parquet/regimes"
DB_PATH = ROOT / "data/_meta.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = ROOT / "config/mc.yaml"
MC_DIR = ROOT / "data/parquet/mc"

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def load_regimes() -> pd.DataFrame:
    dataset_regime = ds.dataset(str(REGIMES_DIR), format="parquet", partitioning="hive")
    df_regime = dataset_regime.to_table().to_pandas()
    if "date" in df_regime.columns:
        df_regime["date"] = pd.to_datetime(df_regime["date"], errors="coerce")
    return df_regime


def load_asset_features() -> pd.DataFrame:
    dataset_assets = ds.dataset(str(FEATURES_ASSETS_DIR), format="parquet", partitioning="hive")
    df_assets = dataset_assets.to_table().to_pandas()
    if "date" in df_assets.columns:
        df_assets["date"] = pd.to_datetime(df_assets["date"], errors="coerce")
    return df_assets


def load_mc_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    content = CONFIG_PATH.read_text().strip()
    if not content:
        return {}
    if yaml is None:
        raise ImportError("PyYAML is required to parse config/regimes.yaml.")
    data = yaml.safe_load(content)
    return data if isinstance(data, dict) else {}



def select_universe(df_assets: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    if df_assets is None or df_assets.empty:
        return pd.DataFrame(index=df_assets.index if df_assets is not None else None)
    if tickers:
        return df_assets[df_assets["ticker"].isin(tickers)].copy()
    return df_assets.copy()


def build_returns_matrix(df_prices: pd.DataFrame) -> pd.DataFrame:
    if df_prices is None or df_prices.empty:
        return pd.DataFrame()
    if "adj_close" not in df_prices.columns:
        raise KeyError("adj_close column is required to build returns matrix.")
    data = df_prices.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    prices = data.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
    returns = np.log(prices / prices.shift(1))
    return returns


def calibrate_regime_params(
    returns: pd.DataFrame,
    regimes: pd.DataFrame,
    window: int,
) -> dict[int, dict[str, np.ndarray]]:
    if returns is None or returns.empty:
        raise ValueError("returns is empty.")
    if regimes is None or regimes.empty:
        raise ValueError("regimes is empty.")
    if "state" not in regimes.columns:
        raise KeyError("regimes must include a 'state' column.")

    if "date" in regimes.columns:
        regimes = regimes.set_index("date")
    if "date" in returns.columns:
        returns = returns.set_index("date")

    aligned_returns = returns.loc[returns.index.intersection(regimes.index)]
    aligned_states = regimes.loc[aligned_returns.index, "state"]

    params: dict[int, dict[str, np.ndarray]] = {}  # Initialise le dictionnaire qui stockera, pour chaque régime (state), les paramètres estimés (mu, sigma).
    for state in sorted(aligned_states.dropna().unique()):  # Parcourt chaque régime unique non-nul, trié, présent dans les états alignés.
        state_dates = aligned_states[aligned_states == state].index  # Récupère les dates correspondant uniquement au régime courant.
        window_returns = aligned_returns.loc[state_dates].tail(window)  # Extrait les rendements de ce régime et ne garde que les `window` dernières observations.
        if window_returns.empty:  # Vérifie qu'il existe bien des données pour ce régime sur la fenêtre sélectionnée.
            continue  # Saute ce régime si aucune observation n'est disponible.
        mu = window_returns.mean().to_numpy()  # Calcule la moyenne (vecteur des espérances) des rendements sur la fenêtre.
        sigma = window_returns.cov().to_numpy()  # Calcule la matrice de covariance des rendements sur la fenêtre.
        params[int(state)] = {"mu": mu, "sigma": sigma}  # Enregistre les paramètres (mu, sigma) pour ce régime dans le dictionnaire, avec une clé entière.

    return params


def simulate_paths(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_sims: int,
    horizon: int,
    dist: str = "gaussian",
) -> np.ndarray:
    if n_sims <= 0 or horizon <= 0:
        raise ValueError("n_sims and horizon must be positive.")
    if dist != "gaussian":
        raise ValueError("Only gaussian dist is supported for now.")

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    n_assets = mu.shape[0]
    paths = np.zeros((n_sims, horizon, n_assets), dtype="float64")
    for t in range(horizon):
        paths[:, t, :] = np.random.multivariate_normal(mu, sigma, size=n_sims)
    return paths


def summarize_paths(paths: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    if paths is None or len(paths) == 0:
        raise ValueError("paths is empty.")
    arr = np.asarray(paths)  # Convertit `paths` en tableau numpy (au cas où ce n’était pas déjà un ndarray).
    if arr.ndim == 1:  # Si le tableau est 1D, on considère qu’il contient déjà directement des PnL par scénario.
        pnl = arr  # Assigne directement le tableau comme distribution de PnL.
    else:  # Sinon, `arr` est multi-dimensionnel (ex: scénarios × temps × actifs, etc.).
        pnl = np.nansum(arr, axis=tuple(range(1, arr.ndim)))  # Agrège chaque scénario en sommant sur toutes les dimensions sauf la première, en ignorant les NaN.
    pnl = np.asarray(pnl, dtype="float64")
    var = np.nanquantile(pnl, alpha)  # Calcule la Value-at-Risk au niveau `alpha` (quantile bas), en ignorant les NaN.
    cvar = np.nanmean(pnl[pnl <= var]) if np.any(pnl <= var) else np.nan  # Calcule la CVaR comme la moyenne des pertes <= VaR, sinon renvoie NaN si aucun point n’est dans la queue.
    q95 = np.nanquantile(pnl, 0.95)  # Calcule le quantile 95% de la distribution de PnL (mesure de “bon scénario” / upside).
    return {"var": float(var), "cvar": float(cvar), "q95": float(q95)}


def build_mc_outputs(
    returns: pd.DataFrame,
    regimes: pd.DataFrame,
    params: dict[int, dict[str, np.ndarray]],
    n_sims: int,
    horizons: list[int],
    dist: str,
) -> pd.DataFrame:
    if returns is None or returns.empty:
        return pd.DataFrame()
    if regimes is None or regimes.empty:
        return pd.DataFrame()
    if "state" not in regimes.columns:
        raise KeyError("regimes must include a 'state' column.")

    regimes = regimes.copy()
    if "date" in regimes.columns:
        regimes["date"] = pd.to_datetime(regimes["date"], errors="coerce")
        regimes = regimes.dropna(subset=["date"])
        regimes = regimes.set_index("date")

    rows: list[dict[str, Any]] = []
    for date, row in regimes.iterrows():
        state = row["state"]
        if state not in params:
            continue
        mu = params[state]["mu"]
        sigma = params[state]["sigma"]
        for horizon in horizons:
            paths = simulate_paths(mu, sigma, n_sims, horizon, dist=dist)
            summary = summarize_paths(paths)
            rows.append(
                {
                    "date": date,
                    "state": int(state),
                    "horizon": int(horizon),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def write_mc_dataset(
    df: pd.DataFrame,
    base_dir: Path,
    partition_cols: list[str],
    existing_data_behavior: str = "overwrite_or_ignore",
    basename_template: str | None = None,
) -> None:
    if df is None or df.empty:
        return
    base_dir.mkdir(parents=True, exist_ok=True)
    if "year" not in df.columns and "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        ok = dt.notna()
        df = df.loc[ok].copy()
        df["year"] = dt.loc[ok].dt.year.astype("int32")
    table = pa.Table.from_pandas(df, preserve_index=False)
    schema = []
    for col in partition_cols:
        if col not in df.columns:
            raise KeyError(f"Missing partition column: {col}")
        if col == "year":
            schema.append((col, pa.int32()))
        else:
            schema.append((col, pa.string()))
    partitioning = ds.partitioning(pa.schema(schema), flavor="hive")
    if existing_data_behavior == "overwrite":
        existing_data_behavior = "delete_matching"
    ds.write_dataset(
        table,
        base_dir=str(base_dir),
        format="parquet",
        partitioning=partitioning,
        existing_data_behavior=existing_data_behavior,
        basename_template=basename_template,
    )


def run_mc_pipeline(existing_data_behavior: str = "overwrite_or_ignore") -> None:
    print("\nLoading configuration...")
    cfg = load_mc_config()
    n_sims = int(cfg.get("n_sims", 2000))
    horizons = [int(h) for h in cfg.get("horizons", [5, 20])]
    window = int(cfg.get("window", 252))
    dist = str(cfg.get("dist", "gaussian"))
    tickers = cfg.get("tickers")

    print("\nLoading regimes...")
    regimes = load_regimes()
    if regimes.empty:
        raise ValueError("No regimes data available.")

    print("\nLoading prices...")
    prices_ds = ds.dataset(str(PRICES_DIR), format="parquet", partitioning="hive")
    df_prices = prices_ds.to_table().to_pandas()
    df_prices = select_universe(df_prices, tickers)

    print("\nBuilding returns matrix...")
    returns = build_returns_matrix(df_prices)
    if returns.empty:
        raise ValueError("No returns matrix available.")

    print("\nCalibrating regime parameters...")
    params = calibrate_regime_params(returns, regimes, window=window)
    print("\nBuilding Monte-Carlo outputs...")
    outputs = build_mc_outputs(returns, regimes, params, n_sims, horizons, dist)

    if outputs.empty:
        return

    print("\nWriting dataset...")
    suffix = str(int(time.time()))
    basename_template = f"mc_{suffix}_{{i}}.parquet"
    write_mc_dataset(
        outputs,
        MC_DIR,
        partition_cols=["year"],
        existing_data_behavior=existing_data_behavior,
        basename_template=basename_template,
    )
    print("\nDone.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute and write mc datasets.")
    parser.add_argument(
        "--existing-data-behavior",
        default="overwrite_or_ignore",
        choices=["overwrite_or_ignore", "overwrite", "error", "delete_matching"],
        help="Behavior when target dataset already has data.",
    )
    args = parser.parse_args()

    run_mc_pipeline(existing_data_behavior=args.existing_data_behavior)
