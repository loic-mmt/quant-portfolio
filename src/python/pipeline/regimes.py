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
from hmmlearn import hmm
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

mean = np.array([features.loc[df_1.index].mean(), features.loc[df_2.index].mean(), features.loc[df_3.index].mean(), features.loc[df_4.index].mean()])
cov = np.array([[features.loc[df_1.index].var(), 0], [0, features.loc[df_2.index].var()], [0, features.loc[df_3.index].var()], [0, features.loc[df_4.index].var()]])

dist = multivariate_normal(mean=mean.flatten(), cov=cov)
mean_1, mean_2, mean_3, mean_4 = mean[0], mean[1], mean[2], mean[3]
sigma_1, sigma_2, sigma_3, sigma_4 = cov[0,0], cov[1,1], cov[2,2], cov[3,3]

x = np.linspace(-0.05, 0.05, num=100)
y = np.linspace(-0.05, 0.05, num=100)
X, Y = np.meshgrid(x,y)
pdf = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pdf[i,j] = dist.pdf([X[i,j], Y[i,j]])


fig, axes = plt.subplots(2, figsize=(15, 10))
ax = axes[0]
ax.plot(model.smoothed_marginal_probabilities[0])
ax.set(title='Smoothed probability of Low Variance Regime')
ax = axes[1]
ax.plot(model.smoothed_marginal_probabilities[1])
ax.set(title='Smoothed probability of High Variance Regime')
fig.tight_layout()
plt.show()


df_1.index = pd.to_datetime(df_1.index)
df_1 = df_1.sort_index()
df_2.index = pd.to_datetime(df_2.index)
df_2 = df_2.sort_index()
plt.figure(figsize=(15, 10))
plt.scatter(df_1.index, df_1, color='blue', label="Low Variance Regime")
plt.scatter(df_2.index, df_2, color='red', label="High Variance Regime")
plt.title("Price series")
plt.ylabel("Price ($)")
plt.xlabel("Date")
plt.legend()
plt.show()


fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(122, projection = '3d')
ax.plot_surface(X, Y, pdf, cmap = 'viridis')
ax.axes.zaxis.set_ticks([])
plt.xlabel("Low Volatility Regime")
plt.ylabel("High Volatility Regime")
plt.title('Bivariate normal distribution of the Regimes')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
plt.contourf(X, Y, pdf, cmap = 'viridis')
plt.xlabel("Low Volatility Regime")
plt.ylabel("High Volatility Regime")
plt.title('Bivariate normal distribution of the Regimes')
plt.tight_layout()
plt.show()