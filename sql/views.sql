CREATE VIEW IF NOT EXISTS v_latest_price_dates AS
SELECT ticker, MAX(date) AS last_available_date
FROM last_dates
GROUP BY ticker;

CREATE VIEW IF NOT EXISTS v_feature_freshness AS
SELECT feature, ticker, date AS last_available_date
FROM feature_last_dates;

CREATE VIEW IF NOT EXISTS v_latest_backtest AS
SELECT *
FROM backtests
WHERE rowid = (SELECT MAX(rowid) FROM backtests);
