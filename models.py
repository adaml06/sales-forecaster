# models.py — safe backtesting + forecasting helpers
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import numpy as np

def seasonality_from_freq(freq: str | None) -> int:
    if freq == "D":
        return 7
    return 52  # default weekly

def _clean_xy(X, y=None, fill=0.0):
    """Return (X_clean, y_clean) with NaN/±inf handled and columns numeric.
       If y is None, just cleans X."""
    # ensure DataFrame for consistent ops
    if not hasattr(X, "columns"):
        # convert numpy array to DataFrame with generic columns
        X = pd.DataFrame(X)
    X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(fill)
    # some models care about dtypes; force numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(fill)
    if y is None:
        return X
    y = (pd.Series(y)
     .replace([np.inf, -np.inf], np.nan)
     .ffill()
     .bfill()
     .fillna(0.0)
     .astype(float))
    return X, y

def interval_from_backtest_ape(yhat: np.ndarray, ape_percent: float | None, floor: float = 0.05, cap: float = 1.50):
    """
    Build relative (percentage) bands around yhat using a single backtest error number.
    - yhat: 1D array of point forecasts
    - ape_percent: e.g., 18.3 for 18.3% (MAPE/SMAPE). If None/NaN, defaults to 35%.
    - floor/cap clamp the relative half-width (e.g., 5% .. 150%)
    Returns (lower, upper, q) where q is the half-width used as a fraction (0..1.5)
    """
    yhat = np.asarray(yhat, dtype=float)
    if ape_percent is None or not np.isfinite(ape_percent):
        q = 0.35
    else:
        q = float(ape_percent) / 100.0
    q = max(float(floor), min(float(cap), q))
    lower = np.maximum(0.0, yhat * (1.0 - q))
    upper = yhat * (1.0 + q)
    return lower, upper, q

def summarize_change(hist_df: pd.DataFrame, fc_df: pd.DataFrame, weeks: int = 8, col: str = "sales"):
    """
    Compare the median of the last N historical weeks vs the first N forecast weeks.
    Returns (pct_change, base_median, fc_median) with pct clamped to [-95%, +500%].
    """
    if weeks <= 0:
        weeks = 8
    h = hist_df[["date", col]].dropna(subset=[col]).tail(int(weeks))
    f = fc_df[["date", "forecast"]].rename(columns={"forecast": col}).dropna(subset=[col]).head(int(weeks))
    if len(h) == 0 or len(f) == 0:
        return 0.0, 0.0, 0.0
    base = float(h[col].median())
    nxt  = float(f[col].median())
    eps  = 1e-6
    if base < eps:
        # baseline ~0 → report +1/0/-1 as absolute direction (avoid inf%)
        return (1.0 if nxt > 0 else 0.0) - (1.0 if base > 0 else 0.0), base, nxt
    pct = (nxt - base) / base
    pct = max(-0.95, min(pct, 5.0))
    return pct, base, nxt

# scikit-learn: models + metrics
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
# LightGBM regressor
from lightgbm import LGBMRegressor
# Holt-Winters (statsmodels)
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- NumPy 2.0 compatibility shim (must run before importing prophet) ---
# Some libs (including older Prophet stacks) reference deprecated NumPy aliases.
# These assignments are no-ops on older NumPy and prevent AttributeError on 2.0+.
if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = np.int64  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore[attr-defined]
if not hasattr(np, "object"):
    np.object = object  # type: ignore[attr-defined]
if not hasattr(np, "complex"):
    np.complex = np.complex128  # type: ignore[attr-defined]
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
# ------------------------------------------------------------------------

# --- Optional XGBoost (safe import) ---
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# --- Optional Prophet (safe import) ---
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

# --- Prophet readiness check (skip slow first-time bootstrap on cloud) ---
import os

def _prophet_ready() -> bool:
    if not _HAS_PROPHET:
        return False
    try:
        import cmdstanpy  # Prophet >=1.1 uses cmdstanpy
        p = cmdstanpy.cmdstan_path()
        return bool(p) and os.path.exists(p) and len(os.listdir(p)) > 0
    except Exception:
        return False

# --- v1.3: tuned params store -----------------------------------------------
TUNED_PARAMS: dict[str, dict] = {}   # e.g. {"LightGBM": {...}, "XGBoost": {...}}

def set_tuned_params(model_name: str, params: dict | None):
    """Store tuned params to be used by fit_* functions if provided."""
    if not params:
        return
    TUNED_PARAMS[model_name] = dict(params)  # shallow copy
# ---------------------------------------------------------------------------

# -----------------------
# Metrics (lower = better)
# -----------------------
def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-9, None)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip((np.abs(y_true) + np.abs(y_pred)) / 2.0, 1e-9, None)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)

def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))

_METRICS = {
    "MAPE": mape,
    "SMAPE": smape,
    "MAE": mae,
    "RMSE": rmse,
}

# -----------------------
# Model fit/predict helpers
# -----------------------
def fit_naive(train_sales: pd.Series):
    last_val = float(train_sales.iloc[-1])
    return {"last": last_val}

def predict_naive(model, horizon: int) -> np.ndarray:
    return np.full(horizon, model["last"], dtype=float)

def fit_snaive(train_sales: pd.Series, seasonality: int):
    # Needs at least 1 full seasonal cycle to predict next season
    if len(train_sales) < seasonality:
        return None
    last_season = train_sales.iloc[-seasonality:].to_numpy(dtype=float)
    return {"last_season": last_season, "seasonality": seasonality}

def predict_snaive(model, horizon: int) -> Optional[np.ndarray]:
    if model is None:
        return None
    s = model["seasonality"]
    base = model["last_season"]
    reps = int(math.ceil(horizon / s))
    arr = np.tile(base, reps)[:horizon]
    return arr.astype(float)

def fit_holt(train_sales: pd.Series, seasonality: int):
    """
    Safe Holt-Winters:
    - if >= 2 full cycles: trend + seasonality
    - if >= 1 full cycle but < 2: trend only (no seasonality)
    - else: level only (simple exponential smoothing)
    """
    n = len(train_sales)
    y = train_sales.to_numpy(dtype=float)

    if n >= 2 * seasonality:
        m = ExponentialSmoothing(
            y, trend="add", seasonal="add", seasonal_periods=seasonality,
            initialization_method="estimated"
        ).fit(optimized=True)
        kind = "holt_winters_seasonal"
    elif n >= seasonality:
        m = ExponentialSmoothing(
            y, trend="add", seasonal=None,
            initialization_method="estimated"
        ).fit(optimized=True)
        kind = "holt_trend_only"
    else:
        m = ExponentialSmoothing(
            y, trend=None, seasonal=None,
            initialization_method="estimated"
        ).fit(optimized=True)
        kind = "simple_exp"
    return {"model": m, "kind": kind, "seasonality": seasonality}

def predict_holt(model, horizon: int) -> np.ndarray:
    m = model["model"]
    fc = m.forecast(horizon)
    return np.asarray(fc, dtype=float)

def fit_ols(Xtr: pd.DataFrame, ytr: pd.Series):
    ols = LinearRegression().fit(Xtr, ytr)
    return ols

def predict_ols(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X).astype(float)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, ElasticNetCV

def fit_ols(Xtr: pd.DataFrame, ytr: pd.Series):
    pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("ols", LinearRegression())
    ])
    pipe.fit(Xtr, ytr)
    return pipe

def predict_ols(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X).astype(float)

def fit_enet(Xtr: pd.DataFrame, ytr: pd.Series):
    # Slightly smaller grid so it does fewer fits (reduces warnings/time)
    alpha_grid = np.logspace(-4, 2, num=40)

    pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("enet", ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            cv=3,
            alphas=alpha_grid,
            max_iter=20000,   # up from 3000
            tol=1e-2,         # a bit looser tolerance to converge faster
            random_state=0,
            n_jobs=1
        ))
    ])

    # Suppress only ElasticNet convergence warnings inside this fit
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        pipe.fit(Xtr, ytr)

    return pipe

def predict_enet(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X).astype(float)

# Aliases some apps import:
def fit_elastic(Xtr: pd.DataFrame, ytr: pd.Series):
    return fit_enet(Xtr, ytr)

def predict_elastic(model, X: pd.DataFrame) -> np.ndarray:
    return predict_enet(model, X)

def fit_lgbm(Xtr: pd.DataFrame, ytr: pd.Series):
    # Small, stable config for short weekly series; overridden by tuned params if present
    base = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=15,
        min_data_in_leaf=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=0,
        verbose=-1,
        n_jobs=1
    )
    user = TUNED_PARAMS.get("LightGBM", {})
    cfg = {**base, **user}

    lgb = LGBMRegressor(**cfg)
    lgb.fit(Xtr, ytr)
    return lgb

def predict_lgbm(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X).astype(float)

# --- XGBoost helpers ---
def fit_xgb(X_train: pd.DataFrame, y_train: pd.Series):
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed. Install it with: pip install xgboost")
    base = dict(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1
    )
    user = TUNED_PARAMS.get("XGBoost", {})
    cfg = {**base, **user}

    model = XGBRegressor(**cfg)
    model.fit(X_train, y_train)
    return model

def predict_xgb(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X).astype(float)

# --- Prophet helpers ---
def _to_prophet_frame(X_or_df: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a Prophet-ready dataframe with columns:
      - ds (datetime), y (target)
      - optional external regressors (price, is_promo) if present
    Accepts either:
      - a single DataFrame that already has 'date' and 'sales'
      - X (with a 'date' column) + y Series
    """
    if y is None:
        df = X_or_df.copy()
        if "date" in df.columns and "sales" in df.columns:
            ds = pd.to_datetime(df["date"])
            ycol = df["sales"].astype(float)
            work = pd.DataFrame({"ds": ds, "y": ycol})
            # carry over known drivers if present
            regressors = []
            for c in ["price", "is_promo"]:
                if c in df.columns:
                    work[c] = df[c].astype(float)
                    regressors.append(c)
            return work, regressors
        else:
            # maybe it's a Series with datetime index
            s = X_or_df  # type: ignore
            if isinstance(s, pd.Series):
                idx = pd.to_datetime(s.index)
                work = pd.DataFrame({"ds": idx, "y": s.astype(float).values})
                return work, []
            raise ValueError("Prophet input must include 'date'+'sales' columns or a Series with a datetime index.")
    else:
        X = X_or_df.copy()
        if "date" not in X.columns:
            raise ValueError("For Prophet with (X,y), X must include a 'date' column.")
        ds = pd.to_datetime(X["date"])
        work = pd.DataFrame({"ds": ds, "y": y.astype(float).values})
        regressors = []
        for c in ["price", "is_promo"]:
            if c in X.columns:
                work[c] = X[c].astype(float)
                regressors.append(c)
        return work, regressors

def fit_prophet(X_or_df: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    Fit a Prophet model. Accepts either:
      - fit_prophet(df_with_date_and_sales_and_optional_drivers)
      - fit_prophet(X, y) where X has a 'date' column and optional 'price','is_promo'
    """
    if not _HAS_PROPHET:
        raise ImportError("prophet is not installed. Install it with: pip install prophet")
    df, regressors = _to_prophet_frame(X_or_df, y)

    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.1,
    )
    for r in regressors:
        m.add_regressor(r)
    m.fit(df)

    # infer cadence from ds spacing
    ds_sorted = pd.to_datetime(df["ds"]).sort_values().to_numpy()
    freq = "W"
    if len(ds_sorted) >= 3:
        diffs = np.diff(ds_sorted).astype("timedelta64[D]").astype(int)
        dailyish = ((diffs == 1) | (diffs == 2)).sum()
        weekly   = (diffs == 7).sum()
        freq = "D" if dailyish >= weekly else "W"

    bundle = {
        "model": m,
        "regressors": regressors,
        "last_vals": {r: float(df[r].iloc[-1]) for r in regressors},
        "last_date": pd.to_datetime(df["ds"]).max(),
        "freq": freq,
    }
    return bundle

def predict_prophet(bundle, steps: int) -> np.ndarray:
    """
    Forecast next `steps` periods using Prophet. If regressors are used,
    we carry-forward their last observed values.
    """
    m = bundle["model"]
    regressors: List[str] = bundle["regressors"]
    freq = bundle.get("freq", "W")

    future = m.make_future_dataframe(periods=steps, freq=freq, include_history=False)
    # add carried-forward regressors if needed
    for r in regressors:
        future[r] = bundle["last_vals"][r]
    fcst = m.predict(future)
    return fcst["yhat"].to_numpy(dtype=float)

# -----------------------
# Backtest (optional Prophet)
# -----------------------
@dataclass
class BacktestResult:
    errors_by_model: Dict[str, float]
    fold_table: pd.DataFrame
    best_model_name: str

def backtest_models(
    feat: pd.DataFrame,
    feature_cols: List[str],
    folds: int = 5,
    horizon: int = 2,
    metric: str = "MAPE",
    seasonality: int = 52,
    allowed_models: Optional[List[str]] = None,
    freq: str | None = None,  # accepted for interface parity; not used internally
) -> Tuple[Dict[str, float], pd.DataFrame, str]:
    assert metric in _METRICS, f"metric must be one of {list(_METRICS.keys())}"
    score_fn = _METRICS[metric]

    # 1) Only historical rows (labels known)
    feat_hist = feat.loc[feat["sales"].notna()].copy()

    # 2) Drop feature columns that are all-NaN across history (useless & dangerous)
    if feature_cols:
        hist_X = feat_hist[feature_cols]
        bad_cols = [c for c in hist_X.columns if hist_X[c].notna().sum() == 0]
        if bad_cols:
            feature_cols = [c for c in feature_cols if c not in bad_cols]

    # 3) Prefill historical features once (handles stragglers; remove ±inf too)
    if feature_cols:
        feat_hist[feature_cols] = (
            feat_hist[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .fillna(0.0)
        )

    n = len(feat_hist)
    if n <= horizon + 5:
        raise ValueError("Not enough rows to backtest. Add more history or reduce folds/horizon.")

    step = max((n - horizon) // max(folds, 1), 1)
    usable_folds = []
    for i in range(folds):
        end = min((i + 1) * step, n - horizon)
        if end <= 5 or (end + horizon) > n:
            continue
        usable_folds.append(end)
    if not usable_folds:
        raise ValueError("Not enough rows to backtest. Add more history or reduce folds/horizon.")

    models_to_try = ["Naive", "OLS", "ElasticNet", "LightGBM", "XGBoost"]
    if n >= seasonality:
        models_to_try.append("sNaive")
    models_to_try.append("HoltWintersSafe")
    if _HAS_PROPHET and _prophet_ready():
        models_to_try.append("Prophet")
    if allowed_models:
        models_to_try = [m for m in models_to_try if m in set(allowed_models)]

    fold_rows = []
    per_model_errs: Dict[str, List[float]] = {m: [] for m in models_to_try}

    for end in usable_folds:
        # 4) Slice folds from the historical frame only
        train = feat_hist.iloc[:end].copy()
        val   = feat_hist.iloc[end:end + horizon].copy()

        # 5) Reindex to FINAL feature set each fold (drop extras, add missings)
        if feature_cols:
            Xtr = train.reindex(columns=feature_cols, fill_value=0.0)[feature_cols]
            Xv  = val.reindex(columns=feature_cols,   fill_value=0.0)[feature_cols]
        else:
            Xtr = pd.DataFrame(index=train.index)
            Xv  = pd.DataFrame(index=val.index)

        ytr = train["sales"].astype(float)
        yv  = val["sales"].astype(float)

        # 6) Absolute safety: clean X and y (NaN/±inf → finite)
        Xtr, ytr = _clean_xy(Xtr, ytr, fill=0.0)
        Xv,  yv  = _clean_xy(Xv,  yv,  fill=0.0)

        # 6b) Per-fold drop of any columns that STILL contain NaN after cleaning
        if Xtr.shape[1] > 0:
            bad_cols_fold = [c for c in Xtr.columns if Xtr[c].isna().any() or Xv[c].isna().any()]
            if bad_cols_fold:
                keep_cols = [c for c in Xtr.columns if c not in bad_cols_fold]
                Xtr = Xtr[keep_cols]
                Xv  = Xv[keep_cols]

        # 6c) Final guard on labels for XGB (and others)
        ytr = pd.Series(ytr).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0).astype(float)
        yv  = pd.Series(yv ).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0).astype(float)

        # --- Naive ---
        naive_m = fit_naive(ytr)
        naive_pred = predict_naive(naive_m, len(yv))
        per_model_errs["Naive"].append(score_fn(yv, naive_pred))
        fold_rows.append({"fold_end": end, "model": "Naive", "metric": metric, "score": per_model_errs["Naive"][-1]})

        # --- sNaive ---
        if "sNaive" in models_to_try:
            snaive_m = fit_snaive(ytr, seasonality)
            if snaive_m is not None:
                snaive_pred = predict_snaive(snaive_m, len(yv))
                per_model_errs["sNaive"].append(score_fn(yv, snaive_pred))
                fold_rows.append({"fold_end": end, "model": "sNaive", "metric": metric, "score": per_model_errs["sNaive"][-1]})

        # --- OLS ---
        if "OLS" in models_to_try and Xtr.shape[1] > 0:
            try:
                ols_m = fit_ols(Xtr, ytr)
                per_model_errs["OLS"].append(score_fn(yv, predict_ols(ols_m, Xv)))
                fold_rows.append({"fold_end": end, "model": "OLS", "metric": metric, "score": per_model_errs["OLS"][-1]})
            except Exception:
                pass

        # --- ElasticNet ---
        if "ElasticNet" in models_to_try and Xtr.shape[1] > 0:
            try:
                enet_m = fit_enet(Xtr, ytr)
                per_model_errs["ElasticNet"].append(score_fn(yv, predict_enet(enet_m, Xv)))
                fold_rows.append({"fold_end": end, "model": "ElasticNet", "metric": metric, "score": per_model_errs["ElasticNet"][-1]})
            except Exception:
                pass

        # --- LightGBM ---
        if "LightGBM" in models_to_try and Xtr.shape[1] > 0:
            try:
                lgb_m = fit_lgbm(Xtr, ytr)
                per_model_errs["LightGBM"].append(score_fn(yv, predict_lgbm(lgb_m, Xv)))
                fold_rows.append({"fold_end": end, "model": "LightGBM", "metric": metric, "score": per_model_errs["LightGBM"][-1]})
            except Exception:
                pass

        # --- XGBoost ---
        if "XGBoost" in models_to_try and Xtr.shape[1] > 0 and _HAS_XGB:
            try:
                xgb_m = fit_xgb(Xtr, ytr)
                per_model_errs["XGBoost"].append(score_fn(yv, predict_xgb(xgb_m, Xv)))
                fold_rows.append({"fold_end": end, "model": "XGBoost", "metric": metric, "score": per_model_errs["XGBoost"][-1]})
            except Exception:
                pass

        # --- Holt-Winters ---
        try:
            holt_m = fit_holt(ytr, seasonality)
            holt_pred = predict_holt(holt_m, len(yv))
            per_model_errs["HoltWintersSafe"].append(score_fn(yv, holt_pred))
            fold_rows.append({"fold_end": end, "model": "HoltWintersSafe", "metric": metric, "score": per_model_errs["HoltWintersSafe"][-1]})
        except Exception:
            pass

        # --- Prophet (unchanged) ---
        if "Prophet" in models_to_try:
            try:
                df_tr = train[["date", "sales"]].copy()
                for c in ["price", "is_promo"]:
                    if c in train.columns:
                        df_tr[c] = train[c]
                p_m = fit_prophet(df_tr)
                preds = predict_prophet(p_m, steps=len(yv))
                per_model_errs["Prophet"].append(score_fn(yv, preds))
                fold_rows.append({"fold_end": end, "model": "Prophet", "metric": metric, "score": per_model_errs["Prophet"][-1]})
            except Exception:
                pass

    # Aggregate
    errors_by_model: Dict[str, float] = {}
    for m in models_to_try:
        vals = per_model_errs.get(m, [])
        if len(vals):
            errors_by_model[m] = float(np.mean(vals))
    if not errors_by_model:
        raise ValueError("No models could be evaluated. Add more data or relax backtest settings.")

    best_model_name = min(errors_by_model, key=errors_by_model.get)
    fold_table = pd.DataFrame(fold_rows).sort_values(["fold_end", "model"]).reset_index(drop=True)
    return errors_by_model, fold_table, best_model_name

    # Aggregate
    errors_by_model: Dict[str, float] = {}
    for m in models_to_try:
        vals = per_model_errs.get(m, [])
        if len(vals):
            errors_by_model[m] = float(np.mean(vals))
    if not errors_by_model:
        raise ValueError("No models could be evaluated. Add more data or relax backtest settings.")

    best_model_name = min(errors_by_model, key=errors_by_model.get)
    fold_table = pd.DataFrame(fold_rows).sort_values(["fold_end", "model"]).reset_index(drop=True)
    return errors_by_model, fold_table, best_model_name

# -----------------------
# Forecast on full history
# -----------------------

def train_full_and_forecast(
    df_hist: pd.DataFrame,
    make_features_fn,
    feature_cols: list[str],
    model_name: str = "LightGBM",
    steps: int = 12,
    seasonality: int | None = None,
    freq: str | None = None,
):
    # 0) Basic guards
    if df_hist is None or len(df_hist) == 0:
        return pd.DataFrame(columns=["date", "forecast"])

    d = df_hist.copy().sort_values("date").reset_index(drop=True)

    # 1) Infer cadence if not provided
    if freq not in ("D", "W"):
        ds = pd.to_datetime(d["date"]).sort_values().to_numpy()
        if len(ds) >= 3:
            diffs = np.diff(ds).astype("timedelta64[D]").astype(int)
            dailyish = ((diffs == 1) | (diffs == 2)).sum()
            weekly   = (diffs == 7).sum()
            freq = "D" if dailyish >= weekly else "W"
        else:
            freq = "W"

    step = pd.Timedelta(days=1 if freq == "D" else 7)
    _seasonality = seasonality if seasonality is not None else (7 if freq == "D" else 52)

    # 2) FUTURE DATES
    last_date = pd.to_datetime(d["date"]).max()
    future_dates = pd.date_range(last_date + step, periods=int(steps), freq=freq)

    # 3) Prophet path (no engineered features needed)
    if model_name == "Prophet":
        if not _HAS_PROPHET:
            return pd.DataFrame(columns=["date", "forecast"])
        df_tr = df_hist[["date", "sales"]].copy()
        for c in ["price", "is_promo"]:
            if c in df_hist.columns:
                df_tr[c] = df_hist[c]
        p_bundle = fit_prophet(df_tr)
        # Pass cadence to predictor
        p_bundle["freq"] = freq
        preds = predict_prophet(p_bundle, steps=int(steps))
        return pd.DataFrame({"date": future_dates, "forecast": preds})

    # 4) Feature engineering (history only) — we will build future features next
    feat_hist, fcols_hist = make_features_fn(df_hist.copy())
    # choose feature set: either caller’s or what we just built
    fcols = list(feature_cols) if feature_cols else list(fcols_hist)

    # Split to X/y on known history
    hist_mask = feat_hist["sales"].notna()
    X_tr = feat_hist.loc[hist_mask, fcols] if fcols else pd.DataFrame(index=feat_hist.index)
    y_tr = feat_hist.loc[hist_mask, "sales"].astype(float)

    # Safety cleanup
    if fcols:
        X_tr, y_tr = _clean_xy(X_tr, y_tr, fill=0.0)
    else:
        # If no features, the ML models can’t run — fall back to Holt/Naive
        model_name = "HoltWintersSafe"

    # 5) Fit chosen model
    core_model = None
    if model_name == "Naive":
        core_model = fit_naive(y_tr)
    elif model_name == "sNaive":
        core_model = fit_snaive(y_tr, _seasonality) or fit_naive(y_tr)
        if core_model and "last_season" not in core_model:
            model_name = "Naive"  # fell back
    elif model_name == "HoltWintersSafe":
        core_model = fit_holt(y_tr, _seasonality)
    elif model_name in ("OLS", "ElasticNet", "LightGBM", "XGBoost"):
        # must have features
        if len(fcols) == 0:
            core_model = fit_holt(y_tr, _seasonality)
            model_name = "HoltWintersSafe"
        else:
            if model_name == "OLS":
                core_model = fit_ols(X_tr, y_tr)
            elif model_name == "ElasticNet":
                core_model = fit_enet(X_tr, y_tr)
            elif model_name == "LightGBM":
                core_model = fit_lgbm(X_tr, y_tr)
            elif model_name == "XGBoost":
                core_model = fit_xgb(X_tr, y_tr)
    else:
        # Unknown -> safe fallback
        model_name = "HoltWintersSafe"
        core_model = fit_holt(y_tr, _seasonality)

    # 6) Predict
    # (a) Statistical models: direct multi-step forecast
    if model_name in ("Naive", "sNaive", "HoltWintersSafe"):
        if model_name == "Naive":
            yhat = predict_naive(core_model, int(steps))
        elif model_name == "sNaive":
            yhat = predict_snaive(core_model, int(steps))
        else:
            yhat = predict_holt(core_model, int(steps))
        return pd.DataFrame({"date": future_dates, "forecast": yhat})

    # (b) ML models: build future feature frame in batch and predict
    # carry last-known drivers forward so the feature builder can compute future rows
    future_stub = pd.DataFrame({"date": future_dates})
    if "price" in df_hist.columns:
        future_stub["price"] = float(df_hist["price"].dropna().iloc[-1])
    if "is_promo" in df_hist.columns:
        future_stub["is_promo"] = int(df_hist["is_promo"].dropna().iloc[-1]) if df_hist["is_promo"].notna().any() else 0

    hist_plus_future = pd.concat([df_hist, future_stub], ignore_index=True)
    feat_all, fcols_all = make_features_fn(hist_plus_future)
    # Ensure we use the same columns as at train time (drivers first, same order)
    fcols_final = [c for c in fcols if c in feat_all.columns]
    Xf = feat_all.loc[feat_all["date"].isin(future_dates), fcols_final]
    Xf = _clean_xy(Xf, fill=0.0)

    # If for some reason future features are empty, bail safely
    if Xf is None or len(Xf) == 0:
        return pd.DataFrame(columns=["date", "forecast"])

    yhat = predict_lgbm(core_model, Xf) if model_name == "LightGBM" else (
           predict_enet(core_model, Xf) if model_name == "ElasticNet" else (
           predict_ols(core_model, Xf)  if model_name == "OLS"       else (
           predict_xgb(core_model, Xf)  if model_name == "XGBoost"   else None)))
    if yhat is None:
        return pd.DataFrame(columns=["date", "forecast"])

    return pd.DataFrame({"date": future_dates, "forecast": np.asarray(yhat, dtype=float)})


# ===============================
# Aliases expected by app.py
# (thin wrappers that call our unified APIs)
# ===============================

def forecast_naive(
    df_hist,
    make_features_fn,
    feature_cols,
    steps,
    seasonality=None,
    freq=None,
):
    """
    Naive forecast helper. 'seasonality' isn't actually used by Naive itself,
    but we accept it for a consistent interface with other forecasters.
    """
    # compute safe defaults at call time
    _freq = freq or "W"
    _seasonality = seasonality if seasonality is not None else (7 if _freq == "D" else 52)
    return train_full_and_forecast(
        df_hist, make_features_fn, feature_cols, model_name="Naive", steps=steps, seasonality=seasonality
    )

def forecast_snaive(
    df_hist,
    make_features_fn,
    feature_cols,
    steps,
    seasonality=None,
    freq=None,
):
    # compute safe defaults at call time
    _freq = (freq or "W").upper()
    _seasonality = seasonality if seasonality is not None else (7 if _freq == "D" else 52)
    return train_full_and_forecast(
        df_hist, make_features_fn, feature_cols,
        model_name="sNaive", steps=steps, seasonality=_seasonality, freq=_freq,
    )

def forecast_ols(
    df_hist,
    make_features_fn,
    feature_cols,
    steps,
    seasonality=None,
    freq=None,
):
    # compute safe defaults at call time
    _freq = freq or "W"
    _seasonality = seasonality if seasonality is not None else (7 if _freq == "D" else 52)
    return train_full_and_forecast(
        df_hist, make_features_fn, feature_cols,
        model_name="OLS", steps=steps, seasonality=_seasonality,  # <- use _seasonality
    )

# --- v1.3: simple auto-tuners (fast, dependency-free) -----------------------
def _metric_fn(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    metric = (metric or "SMAPE").upper()
    eps = 1e-8
    if metric == "MAPE":
        return float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100)
    if metric == "MAE":
        return float(np.mean(np.abs(y_true - y_pred)))
    if metric == "RMSE":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # SMAPE default
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / np.clip(np.abs(y_true) + np.abs(y_pred), eps, None)) * 100)


def _recent_train_val_split(feat: pd.DataFrame, feature_cols: list[str], val_len: int = 12):
    # Keep temporal order; last val_len rows are validation
    feat = feat.dropna(subset=feature_cols + ["sales"]).copy()
    if len(feat) <= val_len + 10:
        val_len = max(4, len(feat)//5)
    tr = feat.iloc[:-val_len].copy()
    va = feat.iloc[-val_len:].copy()
    Xtr, ytr = tr[feature_cols], tr["sales"].astype(float)
    Xva, yva = va[feature_cols], va["sales"].astype(float)
    return Xtr, ytr, Xva, yva


def tune_lightgbm(feat: pd.DataFrame, feature_cols: list[str], metric: str = "SMAPE", trials: int = 12) -> dict:
    Xtr, ytr, Xva, yva = _recent_train_val_split(feat, feature_cols)
    # small search space; deterministic set for speed
    grid = [
        {"n_estimators": n, "learning_rate": lr, "num_leaves": nl, "min_data_in_leaf": mdl}
        for n in (200, 300, 400, 550)
        for lr in (0.05, 0.08)
        for nl in (15, 31)
        for mdl in (3, 8)
    ][:max(6, min(trials, 24))]

    best = {"score": float("inf"), "params": {}}
    for g in grid:
        m = LGBMRegressor(
            **{**dict(subsample=0.9, colsample_bytree=0.9, random_state=0, verbose=-1, n_jobs=1), **g}
        ).fit(Xtr, ytr)
        pred = m.predict(Xva).astype(float)
        s = _metric_fn(yva.to_numpy(), pred, metric)
        if s < best["score"]:
            best = {"score": s, "params": g}
    return best


def tune_xgboost(feat: pd.DataFrame, feature_cols: list[str], metric: str = "SMAPE", trials: int = 12) -> dict:
    if not _HAS_XGB:
        return {"score": float("inf"), "params": {}}
    Xtr, ytr, Xva, yva = _recent_train_val_split(feat, feature_cols)
    grid = [
        {"n_estimators": n, "learning_rate": lr, "max_depth": md, "subsample": ss, "colsample_bytree": cs}
        for n in (300, 450, 600)
        for lr in (0.05, 0.08)
        for md in (4, 5, 6)
        for ss in (0.7, 0.85)
        for cs in (0.7, 0.9)
    ][:max(6, min(trials, 24))]

    best = {"score": float("inf"), "params": {}}
    for g in grid:
        m = XGBRegressor(
            **{**dict(objective="reg:squarederror", tree_method="hist", random_state=42, n_jobs=1), **g}
        ).fit(Xtr, ytr)
        pred = m.predict(Xva).astype(float)
        s = _metric_fn(yva.to_numpy(), pred, metric)
        if s < best["score"]:
            best = {"score": s, "params": g}
    return best

# Some apps call ElasticNet "elastic"
def forecast_elastic(
    df_hist,
    make_features_fn,
    feature_cols,
    steps,
    seasonality=None,
    freq=None,
):
    # compute safe defaults at call time
    _freq = freq or "W"
    _seasonality = seasonality if seasonality is not None else (7 if _freq == "D" else 52)
    return train_full_and_forecast(
        df_hist, make_features_fn, feature_cols,
        model_name="ElasticNet", steps=steps, seasonality=_seasonality,  # <- use _seasonality
    )

def forecast_lgbm(
    df_hist,
    make_features_fn,
    feature_cols,
    steps,
    seasonality=None,
    freq=None,
):
    # compute safe defaults at call time
    _freq = freq or "W"
    _seasonality = seasonality if seasonality is not None else (7 if _freq == "D" else 52)
    return train_full_and_forecast(
        df_hist, make_features_fn, feature_cols,
        model_name="LightGBM", steps=steps, seasonality=_seasonality,  # <- use _seasonality
    )

def forecast_xgb(
    df_hist,
    make_features_fn,
    feature_cols,
    steps,
    seasonality=None,
    freq=None,
):
    # compute safe defaults at call time
    _freq = freq or "W"
    _seasonality = seasonality if seasonality is not None else (7 if _freq == "D" else 52)
    return train_full_and_forecast(
        df_hist, make_features_fn, feature_cols,
        model_name="XGBoost", steps=steps, seasonality=_seasonality,  # <- use _seasonality
    )

def forecast_holt(
    df_hist,
    make_features_fn,
    feature_cols,
    steps,
    seasonality=None,
    freq=None,
):
    # compute safe defaults at call time
    _freq = freq or "W"
    _seasonality = seasonality if seasonality is not None else (7 if _freq == "D" else 52)
    return train_full_and_forecast(
        df_hist, make_features_fn, feature_cols,
        model_name="HoltWintersSafe", steps=steps, seasonality=_seasonality,  # <- use _seasonality
    )

def forecast_prophet(
    df_hist,
    make_features_fn,
    feature_cols,
    steps,
    seasonality=None,
    freq=None,
):
    # compute safe defaults at call time
    _freq = freq or "W"
    _seasonality = seasonality if seasonality is not None else (7 if _freq == "D" else 52)
    return train_full_and_forecast(
        df_hist, make_features_fn, feature_cols,
        model_name="Prophet", steps=steps, seasonality=_seasonality,  # <- use _seasonality
    )

# Some apps expect a helper to run a backtest directly
def run_backtest(
    feat,
    feature_cols,
    folds=5,
    horizon=4,
    metric="MAE",
    seasonality=None,
    freq=None,
):
    _freq = freq or "W"
    _seasonality = seasonality if seasonality is not None else (7 if _freq == "D" else 52)
    return backtest_models(
        feat, feature_cols,
        folds=folds, horizon=horizon, metric=metric,
        seasonality=_seasonality, freq=_freq
    )

# --- v1.2: LightGBM Quantile helpers ---
def fit_lgbm_quantile(Xtr: pd.DataFrame, ytr: pd.Series, alpha: float):
    from lightgbm import LGBMRegressor
    q = LGBMRegressor(
        objective="quantile",
        alpha=float(alpha),
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=15,
        min_data_in_leaf=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=0,
        verbose=-1,
        n_jobs=1
    )
    q.fit(Xtr, ytr)
    return q

def predict_lgbm_quantile(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X).astype(float)


