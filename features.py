# features.py
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from datetime import timedelta

FREQ_DEFAULTS = {
    "D": {"seasonality": 7},    # weekly pattern in daily data
    "W": {"seasonality": 52},   # yearly pattern in weekly data
}

def infer_freq(dates: pd.Series) -> str:
    s = pd.to_datetime(dates, errors="coerce").dropna().sort_values().unique()
    if len(s) < 3:
        return "W"
    diffs = np.diff(s).astype("timedelta64[D]").astype(int)
    dailyish = ((diffs == 1) | (diffs == 2)).sum()
    weekly   = (diffs == 7).sum()
    return "D" if dailyish >= weekly else "W"

def prep_timeseries(df: pd.DataFrame, date_col: str = "date", target_col: str = "sales", freq: str | None = None) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    if freq is None:
        freq = infer_freq(d[date_col])
    # build continuous index at the inferred cadence
    # build continuous index at the inferred cadence
    rng = pd.date_range(d[date_col].min(), d[date_col].max(), freq=freq)

    # --- NEW: collapse duplicate dates with sensible aggregations ---
    agg = {}
    for c in d.columns:
        if c == date_col:
            continue
        if c == "sales":
            agg[c] = "sum"       # multiple rows same day/week → sum units
        elif c == "price":
            agg[c] = "mean"      # average price across same period
        elif c == "is_promo":
            agg[c] = "max"       # any promo that day/week → 1
        else:
            agg[c] = "first"     # keep first non-null for other cols

    d = (d.groupby(date_col, as_index=False)
        .agg(agg)
        .sort_values(date_col)
        .reset_index(drop=True))

    # now index is unique → safe to reindex the full range
    d = (d.set_index(date_col)
        .reindex(rng)
        .rename_axis(date_col)
        .reset_index()
        .rename(columns={"index": date_col}))

    return d

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
    if "price" in d.columns:
        d["price_lag1"] = d["price"].shift(1)
        d["price_chg"]  = d["price"].pct_change().replace([np.inf, -np.inf], 0)
    if "is_promo" in d.columns:
        d["promo_roll4"] = d["is_promo"].rolling(4, min_periods=1).mean()
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

def _fill_hist_gaps_weekly(d: pd.DataFrame, target_col: str = "sales", seasonality: int = 52) -> pd.DataFrame:
    """
    Fill ONLY historical NaNs in `target_col` on a weekly indexed frame.
    Priority: seasonal carry from t-52 if available, else linear, then ffill/bfill.
    Does NOT touch future rows (which should stay NaN).
    """
    d = d.copy()
    assert "date" in d.columns
    d = d.sort_values("date").reset_index(drop=True)
    # identify history (labels known/unknown by target)
    hist_mask = d[target_col].notna()
    # if no gaps, return quickly
    if d.loc[hist_mask, target_col].isna().sum() == 0 and d.loc[~hist_mask, target_col].isna().all():
        return d

    # seasonal suggestion from t-52
    if len(d) > seasonality:
        s = d[target_col].shift(seasonality)
    else:
        s = pd.Series(index=d.index, dtype=float)

    # build a candidate fill for history rows with NaN
    fill = d[target_col].copy()
    # If seasonal value exists where current is NaN, use it
    use_seasonal = fill.isna() & s.notna()
    fill.loc[use_seasonal] = s.loc[use_seasonal]

    # Interpolate linearly for remaining hist NaNs
    rem = fill.isna() & hist_mask
    if rem.any():
        tmp = fill.copy()
        tmp.loc[~hist_mask] = np.nan  # do not interpolate into future
        tmp = tmp.interpolate("linear", limit_direction="both")
        fill.loc[rem] = tmp.loc[rem]

    # Safety ffill/bfill inside history only
    fill.loc[hist_mask] = fill.loc[hist_mask].ffill().bfill()

    d[target_col] = fill
    return d

def _fill_hist_gaps_daily(d: pd.DataFrame, target_col: str = "sales", seasonality: int = 7) -> pd.DataFrame:
    d = d.copy().sort_values("date").reset_index(drop=True)
    hist_mask = d[target_col].notna()
    if d.loc[hist_mask, target_col].isna().sum() == 0 and d.loc[~hist_mask, target_col].isna().all():
        return d
    s = d[target_col].shift(seasonality) if len(d) > seasonality else pd.Series(index=d.index, dtype=float)

    fill = d[target_col].copy()
    use_seasonal = fill.isna() & s.notna()
    fill.loc[use_seasonal] = s.loc[use_seasonal]

    rem = fill.isna() & hist_mask
    if rem.any():
        tmp = fill.copy()
        tmp.loc[~hist_mask] = np.nan
        tmp = tmp.interpolate("linear", limit_direction="both")
        fill.loc[rem] = tmp.loc[rem]

    fill.loc[hist_mask] = fill.loc[hist_mask].ffill().bfill()
    d[target_col] = fill
    return d

def make_daily_and_fill(df: pd.DataFrame, date_col: str = "date", target_col: str = "sales") -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col)

    agg = {target_col: "sum"}
    if "price" in d.columns:
        agg["price"] = "last"
    if "is_promo" in d.columns:
        agg["is_promo"] = "max"

    # daily aggregate (noop for already-daily)
    d_day = d.groupby(pd.Grouper(key=date_col, freq="D")).agg(agg).reset_index()
    d_day = prep_timeseries(d_day, date_col=date_col, target_col=target_col, freq="D")
    d_day = _fill_hist_gaps_daily(d_day, target_col=target_col, seasonality=7)
    return d_day

def make_weekly_and_fill(df: pd.DataFrame, date_col: str = "date", target_col: str = "sales") -> pd.DataFrame:
    """
    Button-friendly helper:
    1) Resample to weekly frequency (sum 'sales'; keep the last 'price'; use max of 'is_promo')
    2) Reindex to a continuous weekly calendar
    3) Fill ONLY historical gaps in `sales` (future stays NaN)

    Returns a clean weekly df with columns: date, sales, [optional drivers]
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col)

    # 1) Aggregate to weekly
    agg = {target_col: "sum"}
    if "price" in d.columns:
        agg["price"] = "last"
    if "is_promo" in d.columns:
        # treat as a weekly indicator if you had any promo that week
        agg["is_promo"] = "max"

    d_week = (
        d.groupby(pd.Grouper(key=date_col, freq="W")).agg(agg).reset_index()
    )

    # 2) Reindex to continuous weekly calendar
    d_week = prep_timeseries(d_week, date_col=date_col, target_col=target_col, freq="W")

    # 3) Fill historical gaps in sales
    d_week = _fill_hist_gaps_weekly(d_week, target_col=target_col, seasonality=52)

    return d_week

def build_features(df, date_col="date", target_col="sales"):
    # 1) infer cadence from the incoming dates, reindex once to that cadence
    freq = infer_freq(df[date_col])
    d = prep_timeseries(df, date_col, target_col, freq=freq)

    # 2) calendar & externals (these work for both D and W)
    d = add_calendar(d, date_col)
    d = add_external(d)
    d = add_holidays(d, country="US", date_col=date_col)

    # 3) lags/rolls (built off ffilled target so future rows aren’t dropped)
    d = add_lags_rolls(d, target=target_col)

    feature_cols = [c for c in d.columns if c not in [date_col, target_col]]

    # 4) clean FEATURES on history only (labels known)
    mask_hist = d[target_col].notna()
    d.loc[mask_hist, feature_cols] = d.loc[mask_hist, feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    # 5) keep future rows even if partially NaN, as long as there is at least one feature
    keep = mask_hist | d[feature_cols].notna().any(axis=1)
    d = d.loc[keep].reset_index(drop=True)

    return d, feature_cols

def data_quality_report(df, date_col="date", target_col="sales"):
    n = len(df)
    zeros = int((df[target_col]==0).sum())
    missing = int(df[target_col].isna().sum())
    # infer cadence and expected step (1 day for D, 7 days for W)
    freq = infer_freq(df[date_col])
    expected = 1 if freq == "D" else 7
    gaps = int(df[date_col].diff().dt.days.fillna(expected).ne(expected).sum() - 1) if n > 1 else 0
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
