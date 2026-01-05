"""
Docstring pour python.models.hmm.py
Contient la logique du modèle :

- fit_hmm(X, ...) ou une classe HMMModel
- predict_states(X) / predict_proba(X)
- éventuellement save_model / load_model
- aucun I/O de parquet ici

"""

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from hmmlearn.hmm import GaussianHMM

def fit_markov_market(mkt_returns: pd.Series, k_regimes: int = 2):
    """Fit a Markov switching model on a 1D market return series."""
    if mkt_returns is None or mkt_returns.empty:
        raise ValueError("mkt_returns is empty.")
    mkt_returns = mkt_returns.dropna()
    # TODO: add input checks (frequency, minimum length, dtype).
    model = MarkovRegression(mkt_returns, k_regimes=k_regimes, switching_variance=True)
    results = model.fit()
    return results


def fit_hmm_features(X: pd.DataFrame, n_states: int = 3, covariance_type: str = "diag"):
    """Fit a Gaussian HMM on multivariate features."""
    if X is None or X.empty:
        raise ValueError("X is empty.")
    X = X.dropna()
    if X.empty:
        raise ValueError("X has only NaNs after dropna.")
    # TODO: standardize X outside (pipeline/regimes.py) and pass as numpy array.
    model = GaussianHMM(n_components=n_states, covariance_type=covariance_type, n_iter=200)
    model.fit(X.to_numpy())
    return model


def hmm_states_from_model(model: GaussianHMM, X: pd.DataFrame) -> pd.Series:
    """Return most likely state for each observation."""
    # TODO: handle alignment and NaNs consistently with pipeline.
    states = model.predict(X.to_numpy())
    return pd.Series(states, index=X.index, name="state")


def hmm_proba_from_model(model: GaussianHMM, X: pd.DataFrame) -> pd.DataFrame:
    """Return state probabilities for each observation."""
    # TODO: handle alignment and NaNs consistently with pipeline.
    proba = model.predict_proba(X.to_numpy())
    cols = [f"p_state_{i}" for i in range(proba.shape[1])]
    return pd.DataFrame(proba, index=X.index, columns=cols)
