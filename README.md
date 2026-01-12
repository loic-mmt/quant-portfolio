# Regime-Aware Dynamic Equity Portfolio (Python + C++ + SQL cache + Parquet)

Un système quant, reproductible, et orienté “risk-first” pour gérer un portefeuille **long-only** de **40–50 actions** sur une période cible d’**1 an** (01/01 → 31/12), avec une allocation qui **s’adapte aux régimes de marché** et un **contrôle du risque** au quotidien.

L’objectif n’est pas de “prédire le futur” au sens naïf. L’objectif est de **détecter l’état du marché**, **quantifier le risque futur** (distribution), puis **ajuster l’exposition** et les poids de façon disciplinée.

---

## Table of Contents
- [Regime-Aware Dynamic Equity Portfolio (Python + C++ + SQL cache + Parquet)](#regime-aware-dynamic-equity-portfolio-python--c--sql-cache--parquet)
  - [Table of Contents](#table-of-contents)
  - [Key Ideas](#key-ideas)
  - [Project Goals](#project-goals)
  - [System Overview](#system-overview)
  - [Data](#data)
    - [Universe](#universe)
    - [Frequency](#frequency)
    - [Incremental Download Cache (SQL)](#incremental-download-cache-sql)
  - [Pipeline](#pipeline)
  - [Regime Detection](#regime-detection)
  - [Risk Modeling (Monte-Carlo)](#risk-modeling-monte-carlo)
    - [Stored Outputs (summary only)](#stored-outputs-summary-only)
  - [Dynamic Allocation](#dynamic-allocation)
    - [Optimization (typical constraints)](#optimization-typical-constraints)
  - [Risk Overlay](#risk-overlay)
  - [Backtesting Protocol](#backtesting-protocol)
    - [Baselines / Ablations](#baselines--ablations)
  - [Storage Strategy](#storage-strategy)
    - [Minimal SQL DB](#minimal-sql-db)
    - [File Storage](#file-storage)
  - [Outputs](#outputs)
  - [Project Structure (may vary):](#project-structure-may-vary)
  - [How to Run](#how-to-run)
  - [Evaluation Metrics](#evaluation-metrics)
    - [Performance](#performance)
    - [Risk](#risk)
    - [Practicality](#practicality)
  - [Limitations](#limitations)
  - [Roadmap](#roadmap)
  - [License / Disclaimer](#license--disclaimer)

---

## Key Ideas

- **One-year portfolio** does *not* mean one decision horizon.  
  The system uses **multi-horizon logic**:
  - **Regime** estimated on **20d** (confirmed by **60d**).
  - **Rebalancing** weekly / bi-weekly.
  - **Risk overlay** daily (vol targeting + stress cut).

- **Regimes come from real market data**, not from simulations.  
  Monte-Carlo is used to **quantify risk** *conditional on the regime*, not to generate “scenarios” for a DL model.

- **Cross-asset structure matters**.  
  Regimes are heavily driven by:
  - average correlations (spike in crises),
  - dispersion (winners vs losers),
  - volatility & drawdowns.

---

## Project Goals

1. Build a **market regime detector** (interpretable states + probabilities).
2. Build a **risk engine** producing forward risk distributions (VaR/CVaR/probabilities).
3. Implement a **dynamic allocation engine** that changes policy by regime.
4. Add a **daily risk overlay** to control tail risk and stabilize volatility.
5. Produce a **walk-forward backtest** (out-of-sample year as the “project year”).
6. Generate a final report with:
   - regime interpretation,
   - portfolio behavior by regime,
   - performance/risk metrics,
   - ablation tests (with/without regimes, with/without overlay, etc.).

---

## System Overview

**Inputs**
- Equity close prices (adjusted if available) for ~40–50 tickers.
- Market proxy (index / aggregate of universe).
- Optional: sector tags for diversification constraints.

**Core Engines**
- Feature engineering (market + cross-sectional)
- Regime detection (HMM / clustering)
- Risk estimation (regime-conditional Monte-Carlo)
- Optimization / allocation (constrained)
- Risk overlay (vol targeting + stress cut)
- Backtest + analytics

**Outputs**
- Daily regime state + probabilities
- Daily portfolio weights
- Trades + turnover
- Performance curve, risk stats, regime attribution

---

## Data

### Universe
- 40–50 equities (long-only).
- The universe should be fixed for the backtest period (avoid changing constituents if possible).

### Frequency
- Daily closes (close-to-close returns).

### Incremental Download Cache (SQL)
A minimal SQL database is used only as a **freshness cache**:
- stores the last available price date per ticker
- prevents redundant re-downloads

> The heavy time-series (prices/features) are stored in files (Parquet). The SQL DB remains lightweight.

---

## Pipeline

The project is structured as a deterministic pipeline:

1. **Ingest**
   - Download missing price data (incremental using the SQL cache).
   - Align dates, clean missing values, build returns.

2. **Feature Engineering**
   - Build market features (vol, momentum, drawdown, vol-of-vol).
   - Build cross-sectional features (avg correlation, dispersion, breadth).

3. **Regime Detection**
   - Fit/Update regime model on a rolling training window.
   - Output state + state probabilities.

4. **Risk Modeling (Monte-Carlo)**
   - Calibrate regime-conditional covariance (shrinkage).
   - Simulate portfolio return distributions for horizons (5d, 20d).
   - Output VaR/CVaR and stress probabilities.

5. **Allocation**
   - Convert regime → allocation policy.
   - Solve constrained optimization (long-only, caps, turnover, costs).
   - Produce target weights.

6. **Risk Overlay (Daily)**
   - Vol targeting (scale exposure to match volatility target).
   - Stress cut (reduce exposure if tail risk exceeds thresholds).

7. **Backtest**
   - Apply rebalancing schedule, transaction costs, constraints.
   - Produce performance series and all diagnostics.

---

## Regime Detection

The regime model outputs:
- a discrete state: `S_t ∈ {1..K}`
- state probabilities: `P(S_t = k)`

Recommended: **HMM with K=3 states**, interpreted as:
- **State 1 — Calm/Trend**: low vol, positive momentum, moderate correlations
- **State 2 — Choppy/Mean-Revert**: mixed momentum, mid vol, higher dispersion
- **State 3 — Stress/Risk-off**: high vol, drawdowns, high correlations

**Important**: regime detection is trained on multiple years (if available) and tested on the project year out-of-sample.

---

## Risk Modeling (Monte-Carlo)

Monte-Carlo is used to estimate **future risk distributions**, conditional on regime:
- Build regime-conditional parameters:
  - mean `μ_k` (optional / can be set to 0 conservatively)
  - covariance `Σ_k` (with shrinkage for stability)
- Simulate multi-asset returns at horizons:
  - **5 days** (tactical risk)
  - **20 days** (regime horizon)

### Stored Outputs (summary only)
We **do not store paths**. We store summaries:
- VaR(5%), CVaR(5%)
- VaR(1%), CVaR(1%)
- probability of loss worse than X
- probability of drawdown worse than Y
- quantiles of portfolio PnL distribution

This is compact, auditable, and directly usable by the overlay.

---

## Dynamic Allocation

The allocation engine maps `regime → policy`.

Example policies (illustrative):
- **Calm/Trend**: risk-on posture (higher exposure, more concentration, momentum tilt)
- **Choppy**: diversify, reduce concentration, constrain turnover
- **Stress**: reduce exposure (allow cash), tighten caps, minimize tail risk

### Optimization (typical constraints)
- `Σ w_i = 1` (or `≤ 1` if cash is allowed)
- `0 ≤ w_i ≤ w_max`
- turnover cap per rebalance
- transaction costs (bps) + optional slippage
- optional target volatility constraint

---

## Risk Overlay

This layer runs daily and enforces a “risk-first” behavior:

1. **Vol Targeting**
   - scale exposure to hit target annualized volatility (e.g., 10–15%)

2. **Stress Cut**
   - if regime-conditional tail risk exceeds thresholds (via MC summaries),
     reduce exposure automatically

3. **Turnover Governor**
   - limits over-trading (crucial for realism)

This overlay is what makes the system stable and closer to professional portfolio processes.

---

## Backtesting Protocol

To avoid fooling ourselves:

- Train regime/risk parameters on a rolling historical window (multi-year if possible).
- Evaluate on the **project year** out-of-sample (01/01 → 31/12).
- Use **walk-forward**:
  - recalibrate → trade next period → repeat

### Baselines / Ablations
- Equal-weight portfolio
- Minimum variance without regimes
- Strategy without overlay
- Strategy without Monte-Carlo summaries

---

## Storage Strategy

### Minimal SQL DB
Used only for:
- `ticker → last_available_date`

### File Storage
Store heavy objects in files:
- prices, returns
- features
- covariance matrices
- regime outputs
- weights, trades
- backtest results

Parquet is ideal.

---

## Outputs

- `regimes`: state + probabilities
- `weights`: daily portfolio weights
- `trades`: rebalancing deltas + turnover
- `risk`: VaR/CVaR and stress probabilities
- `performance`: equity curve, drawdowns
- `report`: plots and summary tables

---

## Project Structure (may vary):
src/
ingest/
features/
regimes/
risk_mc/
optimize/
backtest/
report/
db/
data/
artifacts/


- **Python** orchestrates and owns the research pipeline.
- **C++** accelerates compute-heavy blocks (MC / optimization), called from Python.
- **SQL** acts as a lightweight cache for price freshness.

---

## How to Run

High-level steps:
1. Configure tickers and parameters in config files.
2. Run ingestion (incremental).
3. Build features.
4. Fit regimes (walk-forward).
5. Run backtest.
6. Generate report.

(Exact CLI commands depend on implementation.)

---

## Evaluation Metrics

### Performance
- CAGR
- Sharpe / Sortino
- hit rate (optional)

### Risk
- Max Drawdown
- VaR / CVaR (1%, 5%)
- downside deviation
- tail-risk frequency

### Practicality
- turnover
- concentration (HHI)
- regime attribution (performance per regime)

---

## Limitations

- Without options, hedging is limited; stress regime response is mainly **de-risking / cash**.
- Regime models can lag; overly frequent state switching is a risk (mitigated by probabilities + smoothing).
- Results depend on realistic costs and turnover constraints.
- If the universe changes over time, survivorship bias may affect the backtest.

---

## Roadmap

1. Ingest + SQL cache
2. Features + diagnostics
3. HMM regimes (K=3) + interpretation
4. Baseline allocation (min-variance) + backtest
5. Vol targeting overlay
6. Regime-conditional Monte-Carlo summaries
7. Full walk-forward + ablations + final report
8. C++ acceleration for MC/optimization if needed

---

## License / Disclaimer

This project is for research and educational purposes. It is not financial advice.