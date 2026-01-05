"""
Docstring pour python.pipeline.regimes.py
Contenu minimal :
- load data/parquet/features/regime
- normaliser (z‑score sur train)
- fit modèle (HMM ou clustering)
- produire state + proba par date
- écrire dans data/parquet/regimes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.dataset as ds
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from scipy.stats import multivariate_normal
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

ROOT = Path(__file__).resolve().parents[1]
out_dir = ROOT / "data/parquet/features"
REGIME_DIR = out_dir / "regime"
ASSET_DIR = out_dir / "assets"


def load_regime_features() -> pd.DataFrame:
    dataset = ds.dataset(str(REGIME_DIR), format="parquet", partitioning="hive")
    table = dataset.to_table()
    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date")

features = load_regime_features()

model = MarkovRegression(features, k_regimes=4, switching_variance=True).fit()
print(model.summary())

regime = pd.Series(model.smoothed_marginal_probabilities.values.argmax(axis=1)+1, 
                      index=features.index, name='regime')
df_1 = features.loc[features.index][regime == 1]
df_2 = features.loc[features.index][regime == 2]
df_3 = features.loc[features.index][regime == 3]
df_4 = features.loc[features.index][regime == 4]