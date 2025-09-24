# utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.clip(np.abs(y_true), 1e-9, None)
    return float(np.mean(np.abs((y_true - y_pred)/denom))*100)

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.clip((np.abs(y_true)+np.abs(y_pred)), 1e-9, None)
    return float(np.mean(2*np.abs(y_pred-y_true)/denom)*100)

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true)-np.asarray(y_pred))))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true)-np.asarray(y_pred))**2)))

def mase(y_true, y_pred, seasonality=52):
    y = np.asarray(y_true, float)
    yhat = np.asarray(y_pred, float)
    if len(y) <= seasonality+1:
        return np.nan
    denom = np.mean(np.abs(y[seasonality:] - y[:-seasonality]))
    denom = max(denom, 1e-9)
    return float(np.mean(np.abs(y - yhat)) / denom)

def score_table(y_true, y_pred, seasonality=52):
    return {
        "MAPE": round(mape(y_true, y_pred), 2),
        "SMAPE": round(smape(y_true, y_pred), 2),
        "MAE": round(mae(y_true, y_pred), 2),
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MASE": round(mase(y_true, y_pred, seasonality), 3),
    }

def combine_stats_row(model_name, y_true, y_pred, seasonality=52):
    s = score_table(y_true, y_pred, seasonality)
    s["Model"] = model_name
    return s

def plot_history_forecast(df_hist, forecast_df, title="History + Forecast"):
    fig = plt.figure(figsize=(10,4))
    plt.plot(df_hist["date"], df_hist["sales"], label="Actual")
    if len(forecast_df):
        plt.plot(forecast_df["date"], forecast_df["forecast"], label=f"Forecast ({len(forecast_df)}w)")
        if "lower" in forecast_df and "upper" in forecast_df:
            plt.fill_between(forecast_df["date"], forecast_df["lower"], forecast_df["upper"], alpha=0.2, label="Interval")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return fig
