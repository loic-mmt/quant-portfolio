from __future__ import annotations

import logging
import shutil
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_portfolio.core.db import init_regime_last_dates_db, upsert_regime_last_dates
from quant_portfolio.core.settings import PROJECT_ROOT, load_base_config, load_yaml_mapping
from quant_portfolio.core.storage import (
    load_regime_features,
    replace_partitioned_dataset,
)
from quant_portfolio.models.hmm import (
    ECONOMIC_LABELS,
    economic_state_mapping,
    filtered_hmm_probabilities,
    fit_hmm_features,
    semantic_probabilities,
)
from quant_portfolio.pipeline.data_quality import write_json_report

LOGGER = logging.getLogger(__name__)
BASE_CONFIG = load_base_config()
FEATURES_REGIME_DIR = BASE_CONFIG.parquet_dir / "features/regime"
REGIMES_DIR = BASE_CONFIG.parquet_dir / "regimes"
CALIBRATION_DIR = BASE_CONFIG.data_dir / "artifacts/regimes"
QUALITY_PATH = BASE_CONFIG.data_dir / "quality/regimes_latest.json"
DB_PATH = BASE_CONFIG.metadata_db
CONFIG_PATH = PROJECT_ROOT / "config/regimes.yaml"
SEMANTIC_STATE = {"calm": 0, "choppy": 1, "stress": 2}


@dataclass(frozen=True)
class RegimeConfig:
    """Validated walk-forward HMM configuration."""

    n_states: int
    random_seed: int
    covariance_type: str
    n_iter: int
    tol: float
    min_covar: float
    min_train_size: int
    train_window: int | None
    recalibration_frequency: str
    confidence_threshold: float
    short_confirmation_days: int
    short_confirmation_share: float
    long_confirmation_days: int
    long_confirmation_probability: float
    regime_features: tuple[str, ...]


def load_regime_config(path: Path | None = None) -> RegimeConfig:
    """Load and validate walk-forward regime configuration."""
    config_path = path or CONFIG_PATH
    data = load_yaml_mapping(config_path)
    allowed = {
        "n_states",
        "random_seed",
        "covariance_type",
        "n_iter",
        "tol",
        "min_covar",
        "min_train_size",
        "train_window",
        "recalibration_frequency",
        "confidence_threshold",
        "confirmation",
        "regime_features",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown regime config fields: {sorted(unknown)}")

    confirmation = data.get("confirmation", {})
    if not isinstance(confirmation, dict):
        raise ValueError("confirmation must be a mapping")
    unknown_confirmation = set(confirmation) - {
        "short_days",
        "short_share",
        "long_days",
        "long_probability",
    }
    if unknown_confirmation:
        raise ValueError(f"Unknown confirmation fields: {sorted(unknown_confirmation)}")
    features = data.get("regime_features")
    if not isinstance(features, list) or not features:
        raise ValueError("regime_features must be a non-empty list")

    train_window_raw = data.get("train_window", 1260)
    config = RegimeConfig(
        n_states=int(data.get("n_states", 3)),
        random_seed=int(data.get("random_seed", BASE_CONFIG.random_seed)),
        covariance_type=str(data.get("covariance_type", "diag")),
        n_iter=int(data.get("n_iter", 300)),
        tol=float(data.get("tol", 1e-3)),
        min_covar=float(data.get("min_covar", 1e-6)),
        min_train_size=int(data.get("min_train_size", 504)),
        train_window=None if train_window_raw is None else int(train_window_raw),
        recalibration_frequency=str(data.get("recalibration_frequency", "M")),
        confidence_threshold=float(data.get("confidence_threshold", 0.55)),
        short_confirmation_days=int(confirmation.get("short_days", 20)),
        short_confirmation_share=float(confirmation.get("short_share", 0.60)),
        long_confirmation_days=int(confirmation.get("long_days", 60)),
        long_confirmation_probability=float(confirmation.get("long_probability", 0.40)),
        regime_features=tuple(str(feature) for feature in features),
    )
    if config.n_states < 3:
        raise ValueError("n_states must be >= 3 for calm/choppy/stress mapping")
    if config.random_seed < 0:
        raise ValueError("random_seed must be >= 0")
    if config.covariance_type not in {"diag", "full", "tied", "spherical"}:
        raise ValueError(f"Invalid covariance_type: {config.covariance_type}")
    if config.n_iter < 1 or config.tol <= 0 or config.min_covar <= 0:
        raise ValueError("n_iter, tol, and min_covar must be positive")
    if config.min_train_size <= config.n_states:
        raise ValueError("min_train_size must exceed n_states")
    if config.train_window is not None and config.train_window < config.min_train_size:
        raise ValueError("train_window must be >= min_train_size")
    if not 0 < config.confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in (0, 1]")
    if not 0 < config.short_confirmation_share <= 1:
        raise ValueError("short_share must be in (0, 1]")
    if not 0 < config.long_confirmation_probability <= 1:
        raise ValueError("long_probability must be in (0, 1]")
    if config.short_confirmation_days < 1:
        raise ValueError("short_days must be positive")
    if config.long_confirmation_days < config.short_confirmation_days:
        raise ValueError("long_days must be >= short_days")
    return config


def select_regime_features(df: pd.DataFrame, feature_cols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Select ordered numeric features and remove incomplete dates."""
    if df is None or df.empty:
        raise ValueError("Regime features are empty")
    data = df.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"]).sort_values("date").set_index("date")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Regime features require DatetimeIndex or date column")
    missing = [column for column in feature_cols if column not in data.columns]
    if missing:
        raise ValueError(f"Missing regime features: {missing}")
    selected = data[list(feature_cols)].apply(pd.to_numeric, errors="coerce").dropna()
    selected = selected[~selected.index.duplicated(keep="last")].sort_index()
    if selected.empty:
        raise ValueError("Regime features contain no complete row")
    return selected


def fit_standardizer(train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Fit scaler only on training observations."""
    mean = train.mean()
    scale = train.std(axis=0, ddof=0).replace(0.0, 1.0)
    if mean.isna().any() or scale.isna().any():
        raise ValueError("Training scaler contains NaN")
    return mean, scale


def build_recalibration_dates(
    index: pd.DatetimeIndex,
    min_train_size: int,
    frequency: str,
) -> list[pd.Timestamp]:
    """Return first prediction date of each configured recalibration period."""
    ordered = pd.DatetimeIndex(index).sort_values().unique()
    if len(ordered) <= min_train_size:
        return []
    eligible = ordered[min_train_size:]
    normalized = frequency.strip().upper()
    if normalized in {"D", "DAILY"}:
        return list(eligible)
    if normalized.endswith("B") and normalized[:-1].isdigit():
        step = int(normalized[:-1])
        if step < 1:
            raise ValueError("Business-day recalibration step must be positive")
        return list(eligible[::step])
    try:
        periods = eligible.to_period(frequency)
    except ValueError as exc:
        raise ValueError(f"Invalid recalibration_frequency: {frequency}") from exc
    schedule = pd.Series(eligible, index=periods).groupby(level=0, sort=True).first()
    return [pd.Timestamp(value) for value in schedule]


def _metadata_record(
    *,
    model,
    config: RegimeConfig,
    calibration_date: pd.Timestamp,
    train: pd.DataFrame,
    prediction: pd.DataFrame,
    mean: pd.Series,
    scale: pd.Series,
    mapping: dict[int, str],
    profiles: pd.DataFrame,
    universe_fingerprint: str,
) -> dict[str, Any]:
    profile_records = []
    for raw_state, row in profiles.iterrows():
        profile_records.append(
            {
                "raw_state": int(raw_state),
                **{
                    column: str(value) if column == "regime" else float(value)
                    for column, value in row.items()
                },
            }
        )
    return {
        "calibration_date": calibration_date.strftime("%Y-%m-%d"),
        "train_start": train.index.min().strftime("%Y-%m-%d"),
        "train_end": train.index.max().strftime("%Y-%m-%d"),
        "prediction_start": prediction.index.min().strftime("%Y-%m-%d"),
        "prediction_end": prediction.index.max().strftime("%Y-%m-%d"),
        "train_rows": int(len(train)),
        "prediction_rows": int(len(prediction)),
        "universe_fingerprint": universe_fingerprint,
        "config": asdict(config),
        "scaler": {
            "mean": {column: float(value) for column, value in mean.items()},
            "scale": {column: float(value) for column, value in scale.items()},
        },
        "model": {
            "startprob": np.asarray(model.startprob_).tolist(),
            "transmat": np.asarray(model.transmat_).tolist(),
            "means": np.asarray(model.means_).tolist(),
            "covars": np.asarray(model._covars_).tolist(),
            "log_likelihood": float(model.score(((train - mean) / scale).to_numpy())),
            "converged": bool(model.monitor_.converged),
            "iterations": int(model.monitor_.iter),
        },
        "raw_to_regime": {str(state): label for state, label in mapping.items()},
        "state_profiles": profile_records,
    }


def walk_forward_regimes(
    features: pd.DataFrame,
    config: RegimeConfig,
    *,
    universe_fingerprint: str = "unknown",
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    """Fit scaler/HMM on past windows and predict only next periods."""
    schedule = build_recalibration_dates(
        features.index,
        config.min_train_size,
        config.recalibration_frequency,
    )
    if not schedule:
        raise ValueError("Not enough observations for first regime calibration")

    outputs: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    warmup = pd.DataFrame()
    for position, calibration_date in enumerate(schedule):
        next_date = schedule[position + 1] if position + 1 < len(schedule) else None
        train = features.loc[features.index < calibration_date]
        if config.train_window is not None:
            train = train.tail(config.train_window)
        if len(train) < config.min_train_size:
            raise ValueError(f"Insufficient training rows at {calibration_date.date()}")

        prediction_mask = features.index >= calibration_date
        if next_date is not None:
            prediction_mask &= features.index < next_date
        prediction = features.loc[prediction_mask]
        if prediction.empty:
            continue

        mean, scale = fit_standardizer(train)
        train_z = (train - mean) / scale
        prediction_z = (prediction - mean) / scale
        model = fit_hmm_features(
            train_z,
            n_states=config.n_states,
            covariance_type=config.covariance_type,
            random_seed=config.random_seed,
            n_iter=config.n_iter,
            tol=config.tol,
            min_covar=config.min_covar,
        )
        if not model.monitor_.converged:
            LOGGER.warning("HMM did not converge at calibration %s", calibration_date.date())

        combined_z = pd.concat([train_z, prediction_z])
        raw_all = filtered_hmm_probabilities(model, combined_z)
        raw_train = raw_all.loc[train.index]
        mapping, profiles = economic_state_mapping(train, raw_train)
        semantic_all = semantic_probabilities(raw_all, mapping)

        if warmup.empty:
            warmup = semantic_all.loc[train.index].tail(config.long_confirmation_days)
        raw_prediction = raw_all.loc[prediction.index]
        semantic_prediction = semantic_all.loc[prediction.index]
        period = pd.concat([raw_prediction, semantic_prediction], axis=1)
        period["raw_state"] = raw_prediction.to_numpy().argmax(axis=1)
        period["candidate_regime"] = semantic_prediction.idxmax(axis=1).str.removeprefix("p_")
        period["candidate_state"] = period["candidate_regime"].map(SEMANTIC_STATE).astype("int8")
        period["confidence"] = semantic_prediction.max(axis=1)
        period["calibration_date"] = calibration_date
        period["p_state_0"] = period["p_calm"]
        period["p_state_1"] = period["p_choppy"]
        period["p_state_2"] = period["p_stress"]
        outputs.append(period)
        metadata.append(
            _metadata_record(
                model=model,
                config=config,
                calibration_date=calibration_date,
                train=train,
                prediction=prediction,
                mean=mean,
                scale=scale,
                mapping=mapping,
                profiles=profiles,
                universe_fingerprint=universe_fingerprint,
            )
        )

    if not outputs:
        raise ValueError("Walk-forward regime model produced no prediction")
    unconfirmed = pd.concat(outputs).sort_index()
    confirmed = apply_regime_confirmation(unconfirmed, config, warmup=warmup)
    return confirmed, metadata, warmup


def apply_regime_confirmation(
    probabilities: pd.DataFrame,
    config: RegimeConfig,
    *,
    warmup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply causal confidence, 20/60 confirmation, and transition hysteresis."""
    required = [f"p_{label}" for label in ECONOMIC_LABELS]
    missing = [column for column in required if column not in probabilities.columns]
    if missing:
        raise KeyError(f"Missing semantic probability columns: {missing}")

    data = probabilities.copy().sort_index()
    warm = (
        warmup[required].copy() if warmup is not None and not warmup.empty else pd.DataFrame(columns=required)
    )
    combined_probs = pd.concat([warm, data[required]]) if not warm.empty else data[required].copy()
    candidates = combined_probs.idxmax(axis=1).str.removeprefix("p_")
    confidences = combined_probs.max(axis=1)

    confirmed_labels: list[str] = []
    transitions: list[bool] = []
    reasons: list[str] = []
    current: str | None = None
    for row in range(len(combined_probs)):
        candidate = str(candidates.iloc[row])
        confidence = float(confidences.iloc[row])
        transition = False
        if current is None:
            current = candidate
            reason = "initial"
        elif candidate == current:
            reason = "candidate_same"
        elif confidence < config.confidence_threshold:
            reason = "low_confidence"
        else:
            short_start = max(0, row - config.short_confirmation_days + 1)
            long_start = max(0, row - config.long_confirmation_days + 1)
            short_labels = candidates.iloc[short_start : row + 1]
            long_probability = combined_probs[f"p_{candidate}"].iloc[long_start : row + 1]
            if len(short_labels) < config.short_confirmation_days:
                reason = "short_history"
            elif float(short_labels.eq(candidate).mean()) < config.short_confirmation_share:
                reason = "short_confirmation"
            elif len(long_probability) < config.long_confirmation_days:
                reason = "long_history"
            elif float(long_probability.mean()) < config.long_confirmation_probability:
                reason = "long_confirmation"
            else:
                current = candidate
                transition = True
                reason = "confirmed_20_60"
        confirmed_labels.append(current)
        transitions.append(transition)
        reasons.append(reason)

    offset = len(warm)
    data["regime"] = confirmed_labels[offset:]
    data["state"] = data["regime"].map(SEMANTIC_STATE).astype("int8")
    data["transition"] = transitions[offset:]
    data["transition_reason"] = reasons[offset:]
    data["proba"] = [float(data.iloc[row][f"p_{label}"]) for row, label in enumerate(data["regime"])]
    return data


def compute_regime_diagnostics(
    outputs: pd.DataFrame,
    features: pd.DataFrame,
    calibration_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute occupation, duration, transition matrix, and economic profiles."""
    aligned = outputs[["regime", "confidence", "transition"]].join(features, how="left")
    occupation = aligned["regime"].value_counts(normalize=True)
    transition_counts = pd.crosstab(
        aligned["regime"].shift(1),
        aligned["regime"],
        dropna=False,
    ).reindex(index=ECONOMIC_LABELS, columns=ECONOMIC_LABELS, fill_value=0)
    transition_probabilities = transition_counts.div(
        transition_counts.sum(axis=1).replace(0, 1),
        axis=0,
    )

    run_id = aligned["regime"].ne(aligned["regime"].shift()).cumsum()
    durations = aligned.groupby(run_id)["regime"].agg(["first", "size"])
    duration_report = {
        label: {
            "runs": int((durations["first"] == label).sum()),
            "mean_days": float(durations.loc[durations["first"] == label, "size"].mean())
            if (durations["first"] == label).any()
            else 0.0,
            "max_days": int(durations.loc[durations["first"] == label, "size"].max())
            if (durations["first"] == label).any()
            else 0,
        }
        for label in ECONOMIC_LABELS
    }
    profiles = aligned.groupby("regime")[list(features.columns)].mean(numeric_only=True)
    return {
        "first_prediction_date": outputs.index.min().strftime("%Y-%m-%d"),
        "last_prediction_date": outputs.index.max().strftime("%Y-%m-%d"),
        "recalibrations": len(calibration_metadata),
        "occupation": {label: float(occupation.get(label, 0.0)) for label in ECONOMIC_LABELS},
        "duration": duration_report,
        "transition_counts": {
            source: {target: int(transition_counts.loc[source, target]) for target in ECONOMIC_LABELS}
            for source in ECONOMIC_LABELS
        },
        "transition_probabilities": {
            source: {
                target: float(transition_probabilities.loc[source, target]) for target in ECONOMIC_LABELS
            }
            for source in ECONOMIC_LABELS
        },
        "profiles": {
            label: {column: float(value) for column, value in profiles.loc[label].items()}
            for label in profiles.index
        },
        "mean_confidence": float(aligned["confidence"].mean()),
        "confirmed_transitions": int(aligned["transition"].sum()),
    }


def write_calibration_artifacts(records: list[dict[str, Any]], target: Path) -> None:
    """Atomically replace JSON calibration parameter snapshot."""
    if not records:
        raise ValueError("No calibration metadata to write")
    target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = target.parent / f".{target.name}.tmp-{token}"
    backup = target.parent / f".{target.name}.backup-{token}"
    temporary.mkdir(parents=True)
    try:
        for record in records:
            write_json_report(record, temporary / f"{record['calibration_date']}.json")
        write_json_report(
            {"calibrations": [record["calibration_date"] for record in records]},
            temporary / "index.json",
        )
        if target.exists():
            target.rename(backup)
        temporary.rename(target)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if not target.exists() and backup.exists():
            backup.rename(target)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _dataset_exists(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.parquet"))


def run_regime_pipeline(existing_data_behavior: str = "overwrite_or_ignore") -> None:
    """Run deterministic walk-forward regime estimation and persistence."""
    if existing_data_behavior not in {
        "overwrite_or_ignore",
        "overwrite",
        "delete_matching",
        "error",
    }:
        raise ValueError(f"Unsupported existing_data_behavior: {existing_data_behavior}")
    if existing_data_behavior == "error" and _dataset_exists(REGIMES_DIR):
        raise FileExistsError("Regime dataset already exists")

    config = load_regime_config()
    feature_frame = load_regime_features(FEATURES_REGIME_DIR)
    features = select_regime_features(feature_frame, config.regime_features)
    LOGGER.info(
        "Walk-forward regimes: rows=%d features=%d frequency=%s",
        len(features),
        len(features.columns),
        config.recalibration_frequency,
    )
    outputs, calibrations, _ = walk_forward_regimes(
        features,
        config,
        universe_fingerprint=BASE_CONFIG.universe.fingerprint,
    )
    diagnostics = compute_regime_diagnostics(outputs, features, calibrations)
    diagnostics["config"] = asdict(config)
    diagnostics["universe"] = {
        "id": BASE_CONFIG.universe.universe_id,
        "version": BASE_CONFIG.universe.version,
        "fingerprint": BASE_CONFIG.universe.fingerprint,
    }

    persisted = outputs.reset_index().rename(columns={"index": "date"})
    persisted["ticker"] = "__MARKET__"
    replace_partitioned_dataset(persisted, REGIMES_DIR, ["year"])
    write_calibration_artifacts(calibrations, CALIBRATION_DIR)
    write_json_report(diagnostics, QUALITY_PATH)

    with sqlite3.connect(DB_PATH) as connection:
        init_regime_last_dates_db(connection)
        upsert_regime_last_dates(connection, "regime", persisted)
    LOGGER.info(
        "Regime pipeline complete: predictions=%d recalibrations=%d transitions=%d",
        len(persisted),
        len(calibrations),
        diagnostics["confirmed_transitions"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute walk-forward market regimes.")
    parser.add_argument(
        "--existing-data-behavior",
        default="overwrite_or_ignore",
        choices=["overwrite_or_ignore", "overwrite", "error", "delete_matching"],
    )
    args = parser.parse_args()
    run_regime_pipeline(existing_data_behavior=args.existing_data_behavior)
