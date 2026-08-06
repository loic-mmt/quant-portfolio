"""Deterministic Gaussian-HMM primitives with causal filtering.

No file I/O lives here. Pipeline code owns walk-forward scheduling, persistence,
and semantic regime confirmation.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

ECONOMIC_LABELS = ("calm", "choppy", "stress")


def fit_markov_market(mkt_returns: pd.Series, k_regimes: int = 2):
    """Fit legacy Markov regression, retained for notebook compatibility."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    if mkt_returns is None or mkt_returns.empty:
        raise ValueError("mkt_returns is empty.")
    clean = pd.to_numeric(mkt_returns, errors="coerce").dropna()
    if len(clean) < 10:
        raise ValueError("Data length insufficient.")
    model = MarkovRegression(clean, k_regimes=k_regimes, switching_variance=True)
    return model.fit()


def fit_hmm_features(
    X: pd.DataFrame,
    n_states: int = 3,
    covariance_type: str = "diag",
    *,
    random_seed: int = 42,
    n_iter: int = 300,
    tol: float = 1e-3,
    min_covar: float = 1e-6,
) -> GaussianHMM:
    """Fit deterministic Gaussian HMM on already-scaled training features."""
    if X is None or X.empty:
        raise ValueError("X is empty.")
    clean = X.apply(pd.to_numeric, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("X has only NaNs after dropna.")
    if n_states < 2:
        raise ValueError("n_states must be >= 2")
    if covariance_type not in {"diag", "full", "tied", "spherical"}:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")
    if len(clean) <= n_states:
        raise ValueError("Training data must contain more rows than states")

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        min_covar=min_covar,
        random_state=random_seed,
        implementation="log",
    )
    model.fit(clean.to_numpy(dtype="float64"))
    return model


def _gaussian_log_likelihood(model: GaussianHMM, values: np.ndarray) -> np.ndarray:
    """Evaluate Gaussian emission log-density for supported covariance forms."""
    x = np.asarray(values, dtype="float64")
    means = np.asarray(model.means_, dtype="float64")
    covars = np.asarray(model._covars_, dtype="float64")  # hmmlearn canonical storage.
    n_rows, n_features = x.shape
    result = np.empty((n_rows, model.n_components), dtype="float64")
    constant = n_features * np.log(2.0 * np.pi)

    for state in range(model.n_components):
        delta = x - means[state]
        if model.covariance_type == "diag":
            variance = np.clip(covars[state], 1e-12, None)
            quadratic = np.sum(delta * delta / variance, axis=1)
            log_det = np.log(variance).sum()
        elif model.covariance_type == "spherical":
            variance = float(np.clip(np.ravel(covars[state])[0], 1e-12, None))
            quadratic = np.sum(delta * delta, axis=1) / variance
            log_det = n_features * np.log(variance)
        else:
            covariance = covars if model.covariance_type == "tied" else covars[state]
            covariance = np.asarray(covariance, dtype="float64")
            covariance = covariance + np.eye(n_features) * 1e-12
            sign, log_det = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise ValueError(f"Non-positive covariance determinant for state {state}")
            precision = np.linalg.pinv(covariance)
            quadratic = np.einsum("ij,jk,ik->i", delta, precision, delta)
        result[:, state] = -0.5 * (constant + log_det + quadratic)
    return result


def filtered_hmm_probabilities(model: GaussianHMM, X: pd.DataFrame) -> pd.DataFrame:
    """Compute causal filtered probabilities P(S_t | x_1, ..., x_t)."""
    if X is None or X.empty:
        raise ValueError("X is empty.")
    clean = X.apply(pd.to_numeric, errors="coerce")
    if clean.isna().any().any():
        raise ValueError("Filtered HMM input contains NaN")

    emissions = _gaussian_log_likelihood(model, clean.to_numpy(dtype="float64"))
    log_start = np.log(np.clip(np.asarray(model.startprob_), 1e-300, None))
    log_transition = np.log(np.clip(np.asarray(model.transmat_), 1e-300, None))
    log_alpha = np.empty_like(emissions)
    log_alpha[0] = log_start + emissions[0]
    log_alpha[0] -= logsumexp(log_alpha[0])
    for row in range(1, len(clean)):
        prediction = logsumexp(log_alpha[row - 1][:, None] + log_transition, axis=0)
        log_alpha[row] = prediction + emissions[row]
        log_alpha[row] -= logsumexp(log_alpha[row])

    probabilities = np.exp(log_alpha)
    columns = [f"p_raw_{state}" for state in range(model.n_components)]
    return pd.DataFrame(probabilities, index=clean.index, columns=columns)


def economic_state_mapping(
    features: pd.DataFrame,
    filtered_probabilities: pd.DataFrame,
) -> tuple[dict[int, str], pd.DataFrame]:
    """Map raw states to economic labels from posterior-weighted state profiles."""
    if features.empty or filtered_probabilities.empty:
        raise ValueError("features and probabilities must be non-empty")
    aligned_features, aligned_probabilities = features.align(
        filtered_probabilities,
        join="inner",
        axis=0,
    )
    if aligned_features.empty:
        raise ValueError("features and probabilities have no common dates")

    profile_rows: list[dict[str, float | int]] = []
    for column in aligned_probabilities.columns:
        raw_state = int(column.rsplit("_", maxsplit=1)[-1])
        weights = aligned_probabilities[column].to_numpy(dtype="float64")
        total = weights.sum()
        if total <= 0:
            raise ValueError(f"Raw state {raw_state} has zero posterior mass")
        weighted = aligned_features.mul(weights, axis=0).sum(axis=0) / total
        row: dict[str, float | int] = {
            "raw_state": raw_state,
            "occupation": float(weights.mean()),
        }
        row.update({name: float(value) for name, value in weighted.items()})
        profile_rows.append(row)

    profiles = pd.DataFrame(profile_rows).set_index("raw_state").sort_index()
    feature_columns = [column for column in profiles.columns if column != "occupation"]
    standardized = profiles[feature_columns].copy()
    scale = standardized.std(axis=0, ddof=0).replace(0.0, np.nan)
    standardized = (standardized - standardized.mean(axis=0)) / scale

    directions: dict[str, float] = {}
    for column in feature_columns:
        name = column.lower()
        if any(token in name for token in ("vol", "dd", "corr", "disp", "kurt")):
            directions[column] = 1.0
        elif any(token in name for token in ("mom", "trend", "breadth")):
            directions[column] = -1.0
    usable = [column for column in feature_columns if column in directions]
    if not usable:
        raise ValueError("No economically directed regime feature available")

    direction_vector = pd.Series({column: directions[column] for column in usable})
    risk_score = standardized[usable].mul(direction_vector, axis=1).mean(axis=1, skipna=True)
    if risk_score.isna().any() or float(risk_score.max() - risk_score.min()) <= 1e-10:
        raise ValueError("Economic state profiles are not sufficiently distinct")
    profiles["risk_score"] = risk_score

    calm_state = int(risk_score.idxmin())
    stress_state = int(risk_score.idxmax())
    mapping = {
        int(state): (
            "calm" if int(state) == calm_state else "stress" if int(state) == stress_state else "choppy"
        )
        for state in profiles.index
    }
    profiles["regime"] = pd.Series(mapping)
    return mapping, profiles


def semantic_probabilities(
    raw_probabilities: pd.DataFrame,
    mapping: Mapping[int, str],
) -> pd.DataFrame:
    """Aggregate raw-state probabilities into calm/choppy/stress probabilities."""
    output = pd.DataFrame(0.0, index=raw_probabilities.index, columns=[f"p_{x}" for x in ECONOMIC_LABELS])
    for raw_state, label in mapping.items():
        output[f"p_{label}"] += raw_probabilities[f"p_raw_{raw_state}"]
    return output


def hmm_states_from_model(model: GaussianHMM, X: pd.DataFrame) -> pd.Series:
    """Return causal maximum-filtered raw state for each observation."""
    probabilities = filtered_hmm_probabilities(model, X)
    states = probabilities.to_numpy().argmax(axis=1)
    return pd.Series(states, index=probabilities.index, name="state")


def hmm_proba_from_model(model: GaussianHMM, X: pd.DataFrame) -> pd.DataFrame:
    """Return causal filtered raw-state probabilities using legacy column names."""
    probabilities = filtered_hmm_probabilities(model, X)
    probabilities.columns = [column.replace("p_raw_", "p_state_") for column in probabilities]
    return probabilities
