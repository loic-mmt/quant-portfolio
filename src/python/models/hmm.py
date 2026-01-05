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
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.dataset as ds
from pathlib import Path
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from scipy.stats import multivariate_normal
from ..pipeline.features import build_market_index, compute_returns
from ..pipeline.regimes import load_regime_features


ROOT = Path(__file__).resolve().parents[1]
out_dir = ROOT / "data/parquet/features"
REGIME_DIR = out_dir / "regime"
ASSET_DIR = out_dir / "assets"


def hmm_model(df: pd.DataFrame, tickers: list[str] | None = None):
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)
    mkt = build_market_index(df, tickers)
    mkt_returns = compute_returns(mkt)

    model_mkt = MarkovRegression(mkt_returns, k_regimes=2, switching_variance=True).fit()
    print(model_mkt.summary())
    regime_mkt = pd.Series(model_mkt.smoothed_marginal_probabilities.values.argmax(axis=1)+1, 
                      index=features.index, name='regime')
    df_1_mkt = features.loc[features.index][regime_mkt == 1]
    df_2_mkt = features.loc[features.index][regime_mkt == 2]
    mean_mkt = np.array([features.loc[df_1_mkt.index].mean(), features.loc[df_2_mkt.index].mean()])
    cov_mkt = np.array([[features.loc[df_1_mkt.index].var(), 0], [0, features.loc[df_2_mkt.index].var()]])
    dist_mkt = multivariate_normal(mean=mean_mkt.flatten(), cov=cov_mkt)
    mean_1_mkt, mean_2_mkt = mean_mkt[0], mean_mkt[1]
    sigma_1_mkt, sigma_2_mkt = cov_mkt[0,0], cov_mkt[1,1]

    out_mkt = pd.DataFrame(index=mkt_returns.index)
    out_mkt[f"dist_mkt"] = dist_mkt
    out_mkt[f"mean_1_mkt"] = mean_1_mkt
    out_mkt[f"mean_2_mkt"] = mean_2_mkt
    out_mkt[f"sigma_1_mkt"] = sigma_1_mkt
    out_mkt[f"sigma_2_mkt"] = sigma_2_mkt

    features = load_regime_features()

    return out_mkt
