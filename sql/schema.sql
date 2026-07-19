PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
  version     INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS last_dates (
  ticker     TEXT NOT NULL,
  date       TEXT NOT NULL,
  open       REAL,
  high       REAL,
  low        REAL,
  close      REAL,
  adj_close  REAL,
  volume     INTEGER,
  PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_last_dates_ticker ON last_dates(ticker);
CREATE INDEX IF NOT EXISTS idx_last_dates_date ON last_dates(date);

CREATE TABLE IF NOT EXISTS feature_last_dates (
  feature TEXT NOT NULL,
  ticker  TEXT NOT NULL,
  date    TEXT NOT NULL,
  PRIMARY KEY (feature, ticker)
);

CREATE INDEX IF NOT EXISTS idx_feature_last_dates_feature ON feature_last_dates(feature);
CREATE INDEX IF NOT EXISTS idx_feature_last_dates_ticker ON feature_last_dates(ticker);
CREATE INDEX IF NOT EXISTS idx_feature_last_dates_date ON feature_last_dates(date);

CREATE TABLE IF NOT EXISTS regimes_last_dates (
  feature TEXT NOT NULL,
  ticker  TEXT NOT NULL,
  date    TEXT NOT NULL,
  state   INTEGER,
  proba   REAL,
  PRIMARY KEY (feature, ticker)
);

CREATE INDEX IF NOT EXISTS idx_regimes_last_dates_feature ON regimes_last_dates(feature);
CREATE INDEX IF NOT EXISTS idx_regimes_last_dates_ticker ON regimes_last_dates(ticker);
CREATE INDEX IF NOT EXISTS idx_regimes_last_dates_date ON regimes_last_dates(date);

CREATE TABLE IF NOT EXISTS backtests (
  date_debut           TEXT NOT NULL,
  date_fin             TEXT NOT NULL,
  CAGR                 REAL,
  volatility           REAL,
  Sharpe               REAL,
  max_drawdown         REAL,
  turnover_annualised  REAL,
  turnover_mean        REAL,
  turnover_vol         REAL,
  run_id               TEXT PRIMARY KEY
);

CREATE INDEX IF NOT EXISTS idx_backtests_run_id ON backtests(run_id);

INSERT OR IGNORE INTO schema_migrations(version) VALUES (1);
