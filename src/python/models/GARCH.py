import numpy as np
import pandas as pd
from arch import arch_model

def train_vol(returns, vol: str = "GARCH"):
    model = arch_model(
        returns,
        vol = vol,
        p=1,
        o=1,
        q=1,
        dist='t',            # Student's t for fat tails
        mean='Constant' 
    )
    res = model.fit(update_freq=10, disp="off")
    return res

def forecast_vol(returns, window, model, horizon):
    forecast = []

    for i in range(window, len(returns)):
        model_fitted = train_vol(returns, "GARCH")
        _forecast = model_fitted.forecast(horizon=horizon)
        vol_forecast = np.sqrt(_forecast.variance.values[-1,:][0])
        forecast.append(vol_forecast)

    forecast_dates = returns.index[window:]
    forecast_series = pd.Series(forecast, index = forecast_dates)
    annualized_forecast = forecast_series * np.sqrt(252)
    monthly_forecast = forecast_series * np.sqrt(30)
    weekly_forecast = forecast_series * np.sqrt(7)
    return forecast_series, annualized_forecast, monthly_forecast, weekly_forecast
