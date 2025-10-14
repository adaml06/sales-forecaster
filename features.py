# features.py
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from datetime import timedelta

def prep_timeseries(df, date_col="date", target_col="sales", freq="W"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)  # <-- do NOT dropna here
    # reindex to weekly continuous index (fills internal gaps and KEEPS future dates)
    full_idx = pd.date_range(df[date_col].min(), df[date_col].max(), freq=freq)
    df = df.set_index(date_col).reindex(full_idx).rename_axis(date_col).reset_index()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    return df

def add_calendar(df, date_col="date"):
    d = df.copy()
    woy = d[date_col].dt.isocalendar().week.astype(int)
    d["woy_sin"] = np.sin(2*np.pi*woy/52.0)
    d["woy_cos"] = np.cos(2*np.pi*woy/52.0)
    d["month"]   = d[date_col].dt.month
    d["dow"]     = d[date_col].dt.dayofweek
    d["quarter"] = d[date_col].dt.quarter
    d["eoq"]     = (d[date_col] + pd.offsets.QuarterEnd(0) == d[date_col]).astype(int)
    return d

def add_lags_rolls(d, target="sales", max_lag_list=(1,2,3,4,6,8,12,26,52), roll_windows=(4,8,12)):
    n = len(d)
    lags = [L for L in max_lag_list if n > (L + 8)] or [1,2]
    out = d.copy()

    # <<< NEW: build features from a forward-filled target so FUTURE rows aren't NaN >>>
    s = d[target].ffill()

    for L in lags:
        out[f"lag_{L}"] = s.shift(L)

    windows = [w for w in roll_windows if n > (w + 8)]
    for w in windows:
        out[f"roll_mean_{w}"] = s.shift(1).rolling(w, min_periods=max(2, w//2)).mean()
        out[f"roll_std_{w}"]  = s.shift(1).rolling(w, min_periods=max(2, w//2)).std()

    return out

def add_external(df, price_col="price", promo_col="is_promo"):
    d = df.copy()
    for c in [price_col, promo_col]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "price" in d:
        d["price_pct_change"] = d["price"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if "is_promo" in d:
        d["promo_rolling_4"] = d["is_promo"].rolling(4, min_periods=1).mean()
    return d

def add_holidays(df, country="US", date_col="date"):
    try:
        import holidays
        hol = holidays.country_holidays(country)
        df = df.copy()
        df["is_holiday"] = df[date_col].dt.date.map(lambda x: 1 if x in hol else 0)
    except Exception:
        df = df.copy()
        df["is_holiday"] = 0
    return df

def build_features(df, date_col="date", target_col="sales"):
    d = prep_timeseries(df, date_col, target_col, freq="W")
    d = add_calendar(d, date_col)
    d = add_external(d)
    d = add_holidays(d, country="US", date_col=date_col)
    d = add_lags_rolls(d, target=target_col)

    feature_cols = [c for c in d.columns if c not in [date_col, target_col]]

    # ✅ Only clean historical rows (with known sales)
    mask_hist = d[target_col].notna()
    d.loc[mask_hist, feature_cols] = d.loc[mask_hist, feature_cols].ffill().fillna(0)

    # ✅ Keep future rows even if features are partially NaN
    # Just ensure at least one non-NaN feature exists so we don’t lose the entire block
    keep = mask_hist | d[feature_cols].notna().any(axis=1)
    d = d.loc[keep].reset_index(drop=True)

    return d, feature_cols

def data_quality_report(df, date_col="date", target_col="sales"):
    # simple quality signals for UI
    n = len(df)
    zeros = int((df[target_col]==0).sum())
    missing = int(df[target_col].isna().sum())
    gaps = int(df[date_col].diff().dt.days.fillna(0).ne(7).sum() - 1) if n > 1 else 0
    start, end = df[date_col].min(), df[date_col].max()
    return {
        "rows": n,
        "start": str(start.date()) if pd.notna(start) else "n/a",
        "end": str(end.date()) if pd.notna(end) else "n/a",
        "zeros": zeros,
        "missing": missing,
        "gaps": max(gaps, 0)
    }

def log1p_target(y):
    return np.log1p(np.clip(y, 0, None))

def expm1_target(yhat):
    return np.expm1(yhat)
