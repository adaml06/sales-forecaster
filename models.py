# models.py — safe backtesting + forecasting helpers
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

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
from sklearn.linear_model import ElasticNetCV

def fit_enet(Xtr: pd.DataFrame, ytr: pd.Series):
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("enet", ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            # Lighter inner-CV for small cloud CPUs; also uses new 'alphas' int form
            cv=3,
            alphas=100,
            max_iter=3000,
            tol=1e-3,
            random_state=0,
            n_jobs=1
        ))
    ])
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

    bundle = {
        "model": m,
        "regressors": regressors,
        "last_vals": {r: float(df[r].iloc[-1]) for r in regressors},
        "last_date": pd.to_datetime(df["ds"]).max(),
        "freq": "W",
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
) -> Tuple[Dict[str, float], pd.DataFrame, str]:
    """
    Rolling-origin backtest across multiple models. Skips seasonal models
    when there isn’t enough data.
    """
    assert metric in _METRICS, f"metric must be one of {list(_METRICS.keys())}"
    score_fn = _METRICS[metric]

    n = len(feat)
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
        train = feat.iloc[:end].copy()
        val   = feat.iloc[end:end + horizon].copy()
        ytr, yv = train["sales"], val["sales"]
        Xtr, Xv = train[feature_cols], val[feature_cols]

        # Naive
        naive_m = fit_naive(ytr)
        naive_pred = predict_naive(naive_m, horizon)
        per_model_errs["Naive"].append(score_fn(yv, naive_pred))
        fold_rows.append({"fold_end": end, "model": "Naive", "metric": metric, "score": per_model_errs["Naive"][-1]})

        # sNaive
        if "sNaive" in models_to_try:
            snaive_m = fit_snaive(ytr, seasonality)
            if snaive_m is not None:
                snaive_pred = predict_snaive(snaive_m, horizon)
                per_model_errs["sNaive"].append(score_fn(yv, snaive_pred))
                fold_rows.append({"fold_end": end, "model": "sNaive", "metric": metric, "score": per_model_errs["sNaive"][-1]})

        # OLS + ElasticNet + LGBM + XGB (need features)
        if len(Xtr.columns) > 0:
            ols_m = fit_ols(Xtr, ytr)
            per_model_errs["OLS"].append(score_fn(yv, predict_ols(ols_m, Xv)))
            fold_rows.append({"fold_end": end, "model": "OLS", "metric": metric, "score": per_model_errs["OLS"][-1]})

            try:
                enet_m = fit_enet(Xtr, ytr)
                per_model_errs["ElasticNet"].append(score_fn(yv, predict_enet(enet_m, Xv)))
                fold_rows.append({"fold_end": end, "model": "ElasticNet", "metric": metric, "score": per_model_errs["ElasticNet"][-1]})
            except Exception:
                pass

            try:
                lgb_m = fit_lgbm(Xtr, ytr)
                per_model_errs["LightGBM"].append(score_fn(yv, predict_lgbm(lgb_m, Xv)))
                fold_rows.append({"fold_end": end, "model": "LightGBM", "metric": metric, "score": per_model_errs["LightGBM"][-1]})
            except Exception:
                pass

            if _HAS_XGB:
                try:
                    xgb_m = fit_xgb(Xtr, ytr)
                    per_model_errs["XGBoost"].append(score_fn(yv, predict_xgb(xgb_m, Xv)))
                    fold_rows.append({"fold_end": end, "model": "XGBoost", "metric": metric, "score": per_model_errs["XGBoost"][-1]})
                except Exception:
                    pass

        # Holt-Winters (auto-safe)
        try:
            holt_m = fit_holt(ytr, seasonality)
            holt_pred = predict_holt(holt_m, horizon)
            per_model_errs["HoltWintersSafe"].append(score_fn(yv, holt_pred))
            fold_rows.append({"fold_end": end, "model": "HoltWintersSafe", "metric": metric, "score": per_model_errs["HoltWintersSafe"][-1]})
        except Exception:
            pass

        # Prophet (uses date/sales + optional price/is_promo; ignores engineered lags)
        if "Prophet" in models_to_try:
            try:
                df_tr = train[["date", "sales"]].copy()
                for c in ["price", "is_promo"]:
                    if c in train.columns:
                        df_tr[c] = train[c]
                p_m = fit_prophet(df_tr)  # fit from df with date/sales(+drivers)
                preds = predict_prophet(p_m, steps=horizon)
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

# -----------------------
# Forecast on full history
# -----------------------
def train_full_and_forecast(
    df_hist: pd.DataFrame,
    make_features_fn,
    feature_cols: List[str],
    model_name: str,
    steps: int,
    seasonality: int = 52
) -> pd.DataFrame:
    """
    Trains the chosen model on all data and produces a recursive forecast for `steps` weeks.
    Expects df_hist with columns: date, sales, price, is_promo (extra drivers optional)
    and a feature-making function identical to the one used in backtesting.
    """
    # Prophet path first (it doesn't use engineered lag features)
    if model_name == "Prophet":
        if not _HAS_PROPHET:
            raise ImportError("prophet is not installed. Install it with: pip install prophet")
        df_tr = df_hist[["date", "sales"]].copy()
        for c in ["price", "is_promo"]:
            if c in df_hist.columns:
                df_tr[c] = df_hist[c]
        p_m = fit_prophet(df_tr)
        preds = predict_prophet(p_m, steps=steps)
        future_dates = pd.date_range(df_hist["date"].max() + pd.Timedelta(days=7), periods=steps, freq="W")
        return pd.DataFrame({"date": future_dates, "forecast": preds})

    # All other models use engineered features
    feat_all, fcols = make_features_fn(df_hist.copy())
    X_all, y_all = feat_all[fcols], feat_all["sales"]

    # Fit chosen model on all data
    if model_name == "Naive":
        core_model = fit_naive(y_all)
    elif model_name == "sNaive":
        core_model = fit_snaive(y_all, seasonality)
        if core_model is None:
            core_model = fit_naive(y_all)
            model_name = "Naive"
    elif model_name == "OLS":
        core_model = fit_ols(X_all, y_all)
    elif model_name == "ElasticNet":
        try:
            core_model = fit_enet(X_all, y_all)
        except Exception:
            core_model = fit_ols(X_all, y_all)
            model_name = "OLS"
    elif model_name == "LightGBM":
        try:
            core_model = fit_lgbm(X_all, y_all)
        except Exception:
            core_model = fit_ols(X_all, y_all)
            model_name = "OLS"
    elif model_name == "XGBoost":
        try:
            core_model = fit_xgb(X_all, y_all)
        except Exception:
            core_model = fit_ols(X_all, y_all)
            model_name = "OLS"
    elif model_name == "HoltWintersSafe":
        core_model = fit_holt(y_all, seasonality)
    else:
        core_model = fit_ols(X_all, y_all)
        model_name = "OLS"

    # Recursive forecast (weekly)
    df_temp = df_hist.copy()
    preds = []
    last_date = df_temp["date"].max()

    for _ in range(steps):
        next_date = last_date + pd.Timedelta(days=7)
        new_row = {
            "date": next_date,
            "sales": np.nan,
            "price": df_temp["price"].iloc[-1] if "price" in df_temp.columns else np.nan,
            "is_promo": df_temp["is_promo"].iloc[-1] if "is_promo" in df_temp.columns else 0,
        }
        df_temp = pd.concat([df_temp, pd.DataFrame([new_row])], ignore_index=True)

        feat_tmp, fcols_tmp = make_features_fn(df_temp.copy())
        if len(feat_tmp) == 0:
            break
        rowX = feat_tmp.iloc[[-1]][fcols_tmp]

        if model_name == "Naive":
            yhat = float(predict_naive(core_model, 1)[0])
        elif model_name == "sNaive":
            yhat = float(predict_snaive(core_model, 1)[0])
        elif model_name in ("OLS", "ElasticNet", "LightGBM", "XGBoost"):
            yhat = float(core_model.predict(rowX)[0])
        elif model_name == "HoltWintersSafe":
            known = df_temp["sales"].dropna()
            holt_model = fit_holt(known, seasonality)
            yhat = float(predict_holt(holt_model, 1)[0])
        else:
            yhat = float(core_model.predict(rowX)[0])  # fallback

        df_temp.loc[df_temp["date"] == next_date, "sales"] = yhat
        preds.append({"date": next_date, "forecast": yhat})
        last_date = next_date

    return pd.DataFrame(preds)


# ===============================
# Aliases expected by app.py
# (thin wrappers that call our unified APIs)
# ===============================

def forecast_naive(df_hist, make_features_fn, feature_cols, steps, seasonality=52):
    return train_full_and_forecast(
        df_hist, make_features_fn, feature_cols, model_name="Naive", steps=steps, seasonality=seasonality
    )

def forecast_snaive(df_hist, make_features_fn, feature_cols, steps, seasonality=52):
    return train_full_and_forecast(
        df_hist, make_features_fn, feature_cols, model_name="sNaive", steps=steps, seasonality=seasonality
    )

def forecast_ols(df_hist, make_features_fn, feature_cols, steps, seasonality=52):
    return train_full_and_forecast(
        df_hist, make_features_fn, feature_cols, model_name="OLS", steps=steps, seasonality=seasonality
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
def forecast_elastic(df_hist, make_features_fn, feature_cols, steps, seasonality=52):
    return train_full_and_forecast(
        df_hist, make_features_fn, feature_cols, model_name="ElasticNet", steps=steps, seasonality=seasonality
    )

def forecast_lgbm(df_hist, make_features_fn, feature_cols, steps, seasonality=52):
    return train_full_and_forecast(
        df_hist, make_features_fn, feature_cols, model_name="LightGBM", steps=steps, seasonality=seasonality
    )

def forecast_xgb(df_hist, make_features_fn, feature_cols, steps, seasonality=52):
    return train_full_and_forecast(
        df_hist, make_features_fn, feature_cols, model_name="XGBoost", steps=steps, seasonality=seasonality
    )

def forecast_holt(df_hist, make_features_fn, feature_cols, steps, seasonality=52):
    # Holt-Winters safe wrapper
    return train_full_and_forecast(
        df_hist, make_features_fn, feature_cols, model_name="HoltWintersSafe", steps=steps, seasonality=seasonality
    )

def forecast_prophet(df_hist, make_features_fn, feature_cols, steps, seasonality=52):
    # Prophet ignores engineered features; we still pass them for signature compatibility
    return train_full_and_forecast(
        df_hist, make_features_fn, feature_cols, model_name="Prophet", steps=steps, seasonality=seasonality
    )

# Some apps expect a helper to run a backtest directly
def run_backtest(feat, feature_cols, folds=5, horizon=4, metric="MAPE", seasonality=52):
    return backtest_models(
        feat=feat,
        feature_cols=feature_cols,
        folds=folds,
        horizon=horizon,
        metric=metric,
        seasonality=seasonality,
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


