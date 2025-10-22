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
    denom = np.clip((np.abs(y_true)+np.abs(y_pred)) / 2.0, 1e-9, None)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)

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
    unit = "pts"
    if len(forecast_df):
        # infer days vs weeks by typical spacing
        dts = pd.to_datetime(forecast_df["date"]).sort_values().to_numpy()
        if len(dts) >= 2:
            delta_days = np.median(np.diff(dts).astype("timedelta64[D]").astype(int))
            unit = "days" if delta_days <= 2 else "weeks"
        plt.plot(forecast_df["date"], forecast_df["forecast"], label=f"Forecast ({len(forecast_df)} {unit})")
        if "lower" in forecast_df and "upper" in forecast_df:
            plt.fill_between(forecast_df["date"], forecast_df["lower"], forecast_df["upper"], alpha=0.2, label="Interval")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return fig

# --- v1.2: Reliability helpers ---
import numpy as _np
import pandas as _pd

def stability_score_from_preds(preds_dict: dict) -> tuple[float, _pd.DataFrame]:
    """
    preds_dict: {"ModelName": DataFrame(date, forecast), ...}
    Returns (score_0_100, per_date_df with columns: date, mean, std, ratio)
    Score is 100 * (1 - mean(std/mean)) across the forecast horizon, clipped [0,100].
    """
    if not preds_dict:
        return 0.0, _pd.DataFrame()
    # align by date
    all_dates = _pd.DatetimeIndex(sorted(_np.unique(_np.concatenate([
        _pd.to_datetime(df["date"]).to_numpy() for df in preds_dict.values()
    ]))))
    aligned = []
    for name, df in preds_dict.items():
        tmp = df.copy()
        tmp["date"] = _pd.to_datetime(tmp["date"])
        tmp = tmp.set_index("date").reindex(all_dates)
        tmp = tmp.rename(columns={"forecast": name})
        aligned.append(tmp[name])
    F = _pd.concat(aligned, axis=1)
    mean = F.mean(axis=1, skipna=True)
    std  = F.std(axis=1, skipna=True)
    ratio = (std / mean.replace(0, _np.nan)).replace([_np.inf, -_np.inf], _np.nan).fillna(0.0)
    score = float(100.0 * (1.0 - ratio.mean()))
    score = float(_np.clip(score, 0.0, 100.0))
    out = _pd.DataFrame({"date": all_dates, "mean": mean.values, "std": std.values, "ratio": ratio.values})
    return score, out
def detect_regime_shift(y: _pd.Series, window_short: int = 12, window_long: int = 26, z: float = 3.0) -> bool:
    """
    Flags a shift if recent mean deviates from longer-term mean by > z * long std,
    or if recent volatility is > z * long volatility.
    """
    s = _pd.Series(_np.asarray(y, float)).dropna()
    if len(s) < max(window_long + 4, window_short + 4):
        return False
    mu_s, mu_l = s.rolling(window_short).mean(), s.rolling(window_long).mean()
    sd_l = s.rolling(window_long).std()
    recent_mu_diff = abs(mu_s.iloc[-1] - mu_l.iloc[-1])
    recent_sd = s.rolling(window_short).std().iloc[-1]
    long_sd = sd_l.iloc[-1]
    cond_level = long_sd > 0 and (recent_mu_diff > z * long_sd)
    cond_vol   = long_sd > 0 and (recent_sd > z * long_sd)
    return bool(cond_level or cond_vol)