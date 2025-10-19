# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st

from features import build_features, data_quality_report
from models import (
    backtest_models,
    fit_naive, predict_naive,
    fit_snaive, predict_snaive,
    fit_holt, predict_holt,
    train_full_and_forecast
)
from utils import plot_history_forecast, combine_stats_row, score_table
from utils import stability_score_from_preds, detect_regime_shift
from sample_data_ml import gen_weekly_ml, gen_weekly_profile

def _final_choice_from_radio(choice, preds_dict, ens=None):
    """
    Returns (final_fc_df, chosen_name) based on the user's radio choice.
    - choice: "Best by backtest" | "Weighted ensemble" | <specific model name>
    - preds_dict: {"Naive": df, "LightGBM": df, ...}
    - ens: optional ensemble DataFrame (if already computed)
    """
    if not preds_dict:
        raise ValueError("No model forecasts available (preds_dict is empty).")

    # If user explicitly picked a model name that exists, return it directly
    if choice in preds_dict:
        return preds_dict[choice].copy(), choice

    if choice == "Weighted ensemble":
        if ens is not None:
            return ens.copy(), "Ensemble (1/error weights)"
        # No ensemble available → fall back to best by backtest
        choice = "Best by backtest"

    if choice == "Best by backtest":
        # Pick the model with the lowest backtest error among those we actually predicted
        bt = st.session_state.get("bt", {})
        errs = (bt.get("errors") or {})
        # Map UI names to backtest keys if needed
        name_map = {"Elastic": "ElasticNet", "HoltWinters": "HoltWintersSafe"}

        def err_for(name: str) -> float:
            key = name_map.get(name, name)
            val = errs.get(key, np.inf)
            try:
                return float(val)
            except Exception:
                return np.inf

        candidates = [(name, err_for(name)) for name in preds_dict.keys()]
        finite = [c for c in candidates if np.isfinite(c[1])]
        if finite:
            best_name = min(finite, key=lambda x: x[1])[0]
        else:
            # No finite errors recorded → just take the first available prediction
            best_name = next(iter(preds_dict.keys()))
        return preds_dict[best_name].copy(), best_name

    # Last-resort fallback: if the string wasn't recognized, choose first available
    return next(iter(preds_dict.values())).copy(), next(iter(preds_dict.keys()))

def weighted_ensemble(preds_dict: dict, error_by_model: dict):
    """
    v1.3 dynamic weighting:
    - Base weight = 1 / error  (lower error => higher weight)
    - Regime adjustment = 1 / (1 + lambda * volatility)
      where volatility = std(forecast) / (abs(mean(forecast)) + 1e-9)
    - Final weights normalized and applied row-wise across aligned horizons.
    Returns DataFrame(date, forecast) or None.
    """
    if not preds_dict:
        return None

    # union of all dates
    all_dates = []
    for df in preds_dict.values():
        all_dates.append(pd.to_datetime(df["date"]).to_numpy())
    if not all_dates:
        return None
    all_dates = pd.DatetimeIndex(np.sort(np.unique(np.concatenate(all_dates))))

    aligned = []
    dyn_weights = []
    lam = 0.75  # volatility penalty strength

    for name, df in preds_dict.items():
        err = error_by_model.get(name, np.nan)
        if not np.isfinite(err) or err <= 0:
            continue

        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.set_index("date").reindex(all_dates)
        tmp = tmp.rename(columns={"forecast": name})
        series = tmp[name].to_numpy(dtype=float)

        mu = np.nanmean(series)
        sd = np.nanstd(series)
        vol = (sd / (np.abs(mu) + 1e-9)) if np.isfinite(sd) else 0.0

        base = 1.0 / float(err)
        regime_adj = 1.0 / (1.0 + lam * float(vol))
        w = base * regime_adj

        aligned.append(tmp[name])
        dyn_weights.append((name, w))

    if not aligned or not dyn_weights:
        return None

    F = pd.concat(aligned, axis=1)  # columns = model names
    w_series = pd.Series({n: w for n, w in dyn_weights})
    w_series = w_series / w_series.sum()

    # Row-wise weighted average ignoring NaNs
    W = pd.DataFrame(
        np.broadcast_to(w_series.reindex(F.columns).fillna(0).values, F.shape),
        index=F.index, columns=F.columns
    )
    mask = ~F.isna()
    num = (F.fillna(0) * W).sum(axis=1)
    den = (mask * W).sum(axis=1).replace(0, np.nan)
    ens = (num / den).ffill().bfill()

    return pd.DataFrame({"date": all_dates.values, "forecast": ens.values})

# --- v1.4 Scenario Engine helpers (drop-in) ---------------------------------
from dataclasses import dataclass

@dataclass
class Scenario:
    name: str
    price_mult: float = 1.00          # e.g., 0.95 = -5% price
    promo_prob: float = 0.00          # 0.00..1.00 probability per week
    season_shift_pct: float = 0.0     # -20..+20 (%) → rotates woy_sin/cos phase
    notes: str = ""

def _rotate_seasonal_features(df_feat: pd.DataFrame, dates: pd.DatetimeIndex, shift_pct: float) -> pd.DataFrame:
    """
    Rotate week-of-year seasonal basis by a percentage of a full year.
    Keeps your engineered features intact and only adjusts future rows.
    """
    if abs(shift_pct) < 1e-9:
        return df_feat
    df = df_feat.copy()
    mask = df["date"].isin(dates)
    # full rotation = 52 weeks; percentage → radians offset
    delta_weeks = 52.0 * (shift_pct / 100.0)
    theta = 2 * np.pi * (delta_weeks / 52.0)
    # rotate sin/cos: sin' = sin*cosθ + cos*sinθ ; cos' = cos*cosθ - sin*sinθ
    if {"woy_sin","woy_cos"}.issubset(df.columns):
        s = df.loc[mask, "woy_sin"].astype(float).to_numpy()
        c = df.loc[mask, "woy_cos"].astype(float).to_numpy()
        s_new = s*np.cos(theta) + c*np.sin(theta)
        c_new = c*np.cos(theta) - s*np.sin(theta)
        df.loc[mask, "woy_sin"] = s_new
        df.loc[mask, "woy_cos"] = c_new
    return df

def _apply_scenario_future(future_df: pd.DataFrame, sc: Scenario, rng_seed: int = 0) -> pd.DataFrame:
    """Apply price multiplier and promo probability to the future driver stub."""
    out = future_df.copy()
    if "price" in out.columns:
        out["price"] = out["price"].astype(float) * float(sc.price_mult)
    if "is_promo" in out.columns:
        rng = np.random.default_rng(rng_seed)
        out["is_promo"] = (rng.random(len(out)) < float(sc.promo_prob)).astype(int)
    return out
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_monte_carlo_cached(
    raw_df: pd.DataFrame,
    horizon: int,
    base_price_mult: float,
    base_promo_prob: float,
    season_shift_pct: float,
    promo_std: float,
    price_std: float,
    N: int,
    seed: int = 0,
):

    from models import fit_lgbm, predict_lgbm

    # 1) Train ONCE on full history (features from your pipeline)
    feat_all, fcols_all = build_features(raw_df.copy())
    Xtr_all, ytr_all = feat_all[fcols_all], feat_all["sales"]
    base_model = fit_lgbm(Xtr_all, ytr_all)

    # 2) Build a base future driver stub once
    last_feat_date = pd.to_datetime(feat_all["date"]).max()
    future_dates = pd.date_range(last_feat_date + pd.Timedelta(weeks=1),
                             periods=int(horizon), freq="W")
    base_future = pd.DataFrame({"date": future_dates})
    for c in ["price", "is_promo"]:
        if c in raw_df.columns:
            val = raw_df[c].dropna().iloc[-1]
        else:
            val = 0 if c == "is_promo" else 1.0
        base_future[c] = val

    rng = np.random.default_rng(seed)
    totals = np.empty(N, dtype=float)

    for b in range(N):
        # 3) Randomize this run’s scenario knobs
        sc = Scenario(
            name=f"MC{b}",
            price_mult=max(0.5, base_price_mult * (1.0 + rng.normal(0, price_std))),
            promo_prob=float(np.clip(base_promo_prob + rng.normal(0, promo_std), 0.0, 1.0)),
            season_shift_pct=season_shift_pct,
        )

        # 4) Apply scenario to future, build features on (history + this future)
        sc_future = _apply_scenario_future(base_future, sc, rng_seed=b)
        sc_hist = pd.concat([raw_df, sc_future], ignore_index=True)
        sc_feat, sc_fcols = build_features(sc_hist)
        sc_fcols = _ensure_drivers(sc_fcols, sc_feat)
        sc_feat = _rotate_seasonal_features(sc_feat, future_dates, sc.season_shift_pct)
        

        fut = sc_feat[sc_feat["date"].isin(future_dates)]
        if fut.empty:
            # rare: if future rows got dropped, recompute with history-only model & stop-gap
            # (still using scenario’d future drivers)
            sc_hist2 = pd.concat([raw_df, sc_future.assign(sales=np.nan)], ignore_index=True)
            sc_feat2, sc_fcols2 = build_features(sc_hist2)
            sc_feat2 = _rotate_seasonal_features(sc_feat2, future_dates, sc.season_shift_pct)
            fut = sc_feat2[sc_feat2["date"].isin(future_dates)]
            if fut.empty:
                totals[b] = np.nan
                continue

        # 5) Predict with the ONE trained model
        xf_cols = [c for c in sc_fcols if c in fut.columns] or [c for c in fcols_all if c in fut.columns]
        if not xf_cols:
            totals[b] = np.nan
            continue
        Xf = fut[xf_cols]
        yhat = predict_lgbm(base_model, Xf)
        totals[b] = float(np.nansum(yhat))

    totals = np.asarray(totals, dtype=float)
    good = totals[np.isfinite(totals)]
    if good.size == 0:
        # graceful empty result
        return totals, float("nan"), float("nan"), float("nan")
    mu  = float(np.mean(good))
    p10 = float(np.percentile(good, 10))
    p90 = float(np.percentile(good, 90))
    return totals, mu, p10, p90

MODEL_MAP = {
    "Naive": "Naive",
    "sNaive": "sNaive",
    "OLS": "OLS",
    "Elastic": "ElasticNet",
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
    "HoltWinters": "HoltWintersSafe",
    "Prophet": "Prophet",
}
# --- ensure main drivers are included in the model feature set
def _ensure_drivers(fcols: list[str], df_feat: pd.DataFrame) -> list[str]:
    keep = []
    for c in ["price", "is_promo", "woy_sin", "woy_cos"]:
        if c in df_feat.columns:
            keep.append(c)
    # preserve order, put drivers first then everything else (no duplicates)
    return list(dict.fromkeys(keep + list(fcols)))
# v1.1 — simple confidence badge
def _confidence_badge(metric_name: str, best_error: float) -> tuple[str, str]:
    """
    Heuristic: lower errors => higher confidence.
    Returns (emoji_label, explanation).
    """
    m = (metric_name or "").upper()
    x = float(best_error if best_error is not None else 999.0)
    if m in ("MAPE", "SMAPE"):
        green, yellow = (10.0, 20.0) if m == "MAPE" else (12.0, 24.0)
    else:
        green, yellow = (0.10, 0.20)  # fallback for MAE/RMSE (relative sense)
    if x <= green:   return "🟢 High Confidence", f"{m} ≈ {x:.2f}"
    if x <= yellow:  return "🟡 Medium Confidence", f"{m} ≈ {x:.2f}"
    return "🔴 Low Confidence", f"{m} ≈ {x:.2f}"
st.set_page_config(page_title="Sales Forecaster", layout="wide")
st.title("🧠 Sales Forecaster (Weekly)")
# ---- Session defaults (so Results/Backtest never KeyError) ----
_DEFAULT_MODELS = ["Naive","sNaive","OLS","Elastic","LightGBM","XGBoost","HoltWinters","Prophet"]
st.session_state.setdefault("metric", "SMAPE")
st.session_state.setdefault("horizon", 12)
st.session_state.setdefault("folds", 4)
st.session_state.setdefault("include_models", _DEFAULT_MODELS)
# v1.1 — global UI mode + glossary
with st.sidebar:
    st.header("⚙️ Display Mode")
    simple_mode = st.toggle("Simple Mode", value=True, help="Hide advanced knobs until you need them.")
    st.divider()
    with st.expander("📖 Glossary", expanded=False):
        st.markdown("""
**MAPE** — Mean Absolute Percentage Error; avg % difference between actuals and predictions.  
**SMAPE** — Symmetric MAPE; bounded version of MAPE that treats over/under errors more evenly.  
**MAE** — Mean Absolute Error; avg absolute units off.  
**RMSE** — Root Mean Squared Error; emphasizes larger misses.  
**Elasticity** — How sensitive demand is to price changes.  
**Confidence Interval** — A range that likely contains the future value.  
        """)
tab_data, tab_models, tab_bpr = st.tabs(
    ["Data","Models & Settings","Backtest, Pick & Results"]
)

# ---------------- Data Tab ----------------
with tab_data:
    st.subheader("1) Load data or generate sample")
    col1, col2 = st.columns([2,1])
    with col1:
        file = st.file_uploader("Upload CSV with columns: date, sales, (optional: price, is_promo)", type=["csv"])
        if "raw_df" not in st.session_state:
            st.session_state.raw_df = None
        if file:
            df = pd.read_csv(file, parse_dates=["date"])
            st.session_state.raw_df = df
    with col2:
        if simple_mode:
            # Keep the existing one-click generator
            if st.button("Generate sample data"):
                st.session_state.raw_df = gen_weekly_ml(n_weeks=260, seed=np.random.randint(0, 99999))
        else:
            # Advanced Mode: presets + custom controls
            with st.expander("🔬 Advanced sample generator", expanded=True):
                preset = st.selectbox(
                    "Preset",
                    [
                        "ml_friendly",
                        "balanced",
                        "baseliney",
                        "steady_growth",
                        "holiday_spike",
                        "promo_driven",
                        "price_sensitive",
                        "post_launch_decline",
                        "volatile_market",
                        "flatline",
                        "spiky_outliers",
                        "season_switch",
                        "short_horizon",
                        "covid_drop",
                        "Custom",
                    ],
                    index=0,
                    help="Pick a starting profile or choose Custom to set every knob."
                )

                # --- Custom knobs (only used when preset == Custom) ---
                n_weeks = st.number_input("History length (weeks)", 52, 520, 260, 4)
                level = st.number_input("Base level", 0.0, 10000.0, 900.0, 50.0)
                trend_start = st.number_input("Initial weekly trend (units/week)", -5.0, 5.0, 0.8, 0.1)
                season_amp = st.number_input("Season amplitude (~52w)", 0.0, 500.0, 60.0, 5.0)
                price_base = st.number_input("Price base", 0.5, 1000.0, 10.0, 0.5)
                price_vol = st.slider("Price volatility (AR wiggle)", 0.0, 0.6, 0.07, 0.01)
                promo_rate = st.slider("Promo rate", 0.0, 0.95, 0.18, 0.01)
                promo_lift = st.slider("Avg promo lift", 0.0, 1.0, 0.35, 0.01)
                noise_level = st.slider("Noise level (multiplicative)", 0.0, 0.6, 0.12, 0.01)

                st.markdown("---")
                st.caption("Optional effects (applied after generation)")
                add_outliers = st.checkbox("Add outlier weeks (±50–100%)", value=False)
                add_season_switch = st.checkbox("Season amplitude switch after ~60%", value=False)
                add_covid_drop = st.checkbox("COVID-like drop & recovery", value=False)

                seed = st.number_input("Random seed", 0, 1_000_000, int(np.random.randint(0, 99999)), 1)

                if st.button("Generate sample data (advanced)"):
                    if preset != "Custom":
                        df = gen_weekly_profile(profile=preset, seed=int(seed))
                    else:
                        # Fully custom generation
                        df = gen_weekly_ml(
                            n_weeks=int(n_weeks),
                            level=float(level),
                            trend_start=float(trend_start),
                            season_amp=float(season_amp),
                            price_base=float(price_base),
                            price_vol=float(price_vol),
                            promo_rate=float(promo_rate),
                            promo_lift=float(promo_lift),
                            noise_level=float(noise_level),
                            seed=int(seed),
                        )

                        # Optional post-effects to mirror certain presets
                        import numpy as _np
                        _rng = _np.random.default_rng(int(seed))
                        if add_outliers and len(df) >= 20:
                            k = int(_rng.integers(3, 6))
                            idx = _rng.choice(len(df), size=k, replace=False)
                            mult = _rng.uniform(0.5, 2.0, size=k)
                            df.loc[idx, "sales"] = (df.loc[idx, "sales"].to_numpy() * mult).round(2)
                        if add_season_switch and len(df) >= 40:
                            cut = int(len(df) * 0.60)
                            t = _np.arange(len(df) - cut)
                            bump = 1.0 + 0.25 * _np.sin(2 * _np.pi * (t % 52) / 52.0)
                            df.loc[cut:, "sales"] = (df.loc[cut:, "sales"].to_numpy() * bump).round(2)
                        if add_covid_drop and len(df) >= 140:
                            a, b, c = 80, 100, 120
                            drop = 0.50
                            for i in range(a, min(b, len(df))):
                                df.loc[i, "sales"] = (df.loc[i, "sales"] * drop).round(2)
                            for i in range(b, min(c, len(df))):
                                alpha = (i - b) / max(1, c - b)
                                factor = drop + (1 - drop) * alpha
                                df.loc[i, "sales"] = (df.loc[i, "sales"] * factor).round(2)

                    st.session_state.raw_df = df
    if st.session_state.raw_df is not None:
        st.write("Preview:")
        st.dataframe(st.session_state.raw_df.head(12), use_container_width=True)
        q = data_quality_report(st.session_state.raw_df)
        color = "🟢" if q["missing"]==0 and q["gaps"]==0 else ("🟡" if q["missing"]<3 and q["gaps"]<2 else "🔴")
        st.info(f"{color} Rows: {q['rows']} | Range: {q['start']} → {q['end']} | Zeros: {q['zeros']} | Missing: {q['missing']} | Gaps: {q['gaps']}")
            # v1.1 — friendly warnings
        if q["rows"] < 52:
                st.warning("⚠️ Less than ~1 year of weekly data — expect wider forecast intervals and less stable backtests.")
        if q["missing"] > 0:
                st.warning("⚠️ Missing values in `sales` detected — consider filling or removing them for best results.")
        if q["gaps"] > 0:
                st.warning("⚠️ Irregular weekly gaps detected — the app will reindex, but results may be noisier.")
        # v1.1 — Quick Forecast
        st.subheader("🚀 Quick Forecast")
        if st.button("Run Quick Forecast"):
            if st.session_state.raw_df is None:
                st.warning("Upload or generate data first.")
            else:
                quick_h = 12
                df_hist = st.session_state.raw_df.copy()
                df_fc = train_full_and_forecast(
                    df_hist=df_hist,
                    make_features_fn=build_features,
                    feature_cols=[],            # models.py recomputes features internally
                    model_name="LightGBM",
                    steps=quick_h,
                    seasonality=52,
                )
                st.session_state.final_fc = df_fc
                fig = plot_history_forecast(df_hist, df_fc, title=f"Quick Forecast ({quick_h} weeks, LightGBM)")
                st.pyplot(fig, use_container_width=True)

                # confidence badge (uses prior backtest if available)
                badge, badge_sub = "🟡 Medium Confidence", "Heuristic default"
                if "bt" in st.session_state and "errors" in st.session_state.bt:
                    best_model = min(st.session_state.bt["errors"], key=st.session_state.bt["errors"].get)
                    best_err   = st.session_state.bt["errors"][best_model]
                    badge, badge_sub = _confidence_badge(st.session_state.get("metric","SMAPE"), best_err)
                st.markdown(f"**Confidence:** {badge}  \n_{badge_sub}_")
                st.success("Done! Switch to **Backtest, Pick & Results** to compare models or refine.")

# ---------------- Models Tab ----------------
with tab_models:
    st.subheader("2) Configure models & backtest")

    if simple_mode:
    # v1.1 — Simple Mode: minimal knobs
        metric  = st.selectbox("Metric (lower is better)", ["SMAPE","MAPE","MAE","RMSE"], index=0,
                               help="Pick how we score models.")
        horizon = st.selectbox("Forecast horizon (weeks)", [4,8,12,24], index=2)
        folds   = 4  # sensible default; hidden in simple mode
        include_models = ["Naive","sNaive","OLS","Elastic","LightGBM","XGBoost","HoltWinters","Prophet"]
        st.session_state.metric = metric
        st.session_state.horizon = int(horizon)
        st.session_state.folds = int(folds)
        st.session_state.include_models = include_models
        colm3 = st.container()
    else:
    # v1.1 — Advanced Mode: full control
        with st.expander("🔧 Advanced Settings", expanded=True):
            colm1, colm2 = st.columns(2)
            with colm1:
                metric  = st.selectbox("Metric (lower is better)", ["SMAPE","MAPE","MAE","RMSE"], index=0,
                                       help="Lower is better; SMAPE is robust for scale changes.")
                horizon = st.selectbox("Forecast horizon (weeks)", [4,8,12,24], index=2,
                                       help="How far ahead to predict.")
                folds   = st.slider("Backtest folds", 3, 10, 6, help="More folds = slower but stabler.")
            with colm2:
                use_holidays = st.checkbox("Use holidays (US)", value=True, disabled=True,
                                       help="Holiday regressor is already baked into the features.")
                include_models = st.multiselect(
                     "Models to consider",
                   ["Naive","sNaive","OLS","Elastic","LightGBM","XGBoost","HoltWinters","Prophet"],
                    default=["Naive","sNaive","OLS","Elastic","LightGBM","XGBoost","HoltWinters","Prophet"],
                    help="Uncheck anything you don’t want in the bake-off."
                )
                st.markdown("---")
                autotune = st.checkbox(
                    "🧮 Auto-tune LightGBM/XGBoost (v1.3)",
                    value=True,
                    help="Runs a small, fast search on the recent window to lower backtest error."
                )
                trials = st.slider(
                    "Auto-tune trials (fast)",
                    6, 24, 12, 2,
                    help="Fewer = faster. Each try trains a tiny model on a rolling validation split."
                )
        colm3 = st.container()


        # persist selections
        st.session_state.metric = metric
        st.session_state.horizon = int(horizon)
        st.session_state.folds = int(folds)
        st.session_state.include_models = include_models
        # -------------------------------------------------------------------------

    if st.button("Run backtest"):
            if st.session_state.raw_df is None:
                st.warning("Upload or generate data first.")
            else:
                feat, fcols = build_features(st.session_state.raw_df)
                ui = st.session_state.get("include_models", [])
                allowed = [MODEL_MAP[m] for m in ui if m in MODEL_MAP]
                # --- v1.3: optional auto-tuning before backtest ---
                if not simple_mode and 'autotune' in locals() and autotune:
                    from models import tune_lightgbm, tune_xgboost, set_tuned_params
                    if len(fcols) > 0 and len(feat) > 40:
                        if "LightGBM" in allowed:
                            best_lgbm = tune_lightgbm(feat, fcols, metric=metric, trials=int(trials))
                            if best_lgbm["params"]:
                                set_tuned_params("LightGBM", best_lgbm["params"])
                                st.info(f"LightGBM tuned ({metric}≈{best_lgbm['score']:.2f}): {best_lgbm['params']}")
                        if "XGBoost" in allowed:
                            try:
                                best_xgb = tune_xgboost(feat, fcols, metric=metric, trials=int(trials))
                                if best_xgb["params"]:
                                    set_tuned_params("XGBoost", best_xgb["params"])
                                    st.info(f"XGBoost tuned ({metric}≈{best_xgb['score']:.2f}): {best_xgb['params']}")
                            except Exception as _e:
                                st.caption(f"(XGBoost tuning skipped: {_e})")
                errs, fold_tbl, best = backtest_models(
                    feat, fcols, folds=folds, horizon=horizon, metric=metric, seasonality=52,
                    allowed_models=allowed
                )
                st.session_state.bt = {"errors": errs, "folds": fold_tbl, "best": best, "feat": feat, "fcols": fcols}
                st.success(f"Backtest done. Best (avg): **{best}**" if best else "Backtest done.")
                st.dataframe(fold_tbl.fillna("").round(2), use_container_width=True)
                mean_row = pd.DataFrame([{"Model": k, "Mean Error": round(v,2)} for k,v in errs.items() if not np.isnan(v)]).sort_values("Mean Error")
                st.caption("Average backtest error per model:")
                st.dataframe(mean_row, use_container_width=True)
with tab_bpr:
    st.subheader("3) Train best / ensemble & forecast")

    # ---- Hard guard: require data before building Results ----
    if st.session_state.get("raw_df") is None:
        st.info("Upload or generate data first (see Data tab), then run a backtest in Models.")
        st.stop()
    
    # ---- Guard: require backtest context for Results ----
    if "bt" not in st.session_state:
        st.info("Configure models and run a backtest on the Models tab first.")
        st.stop()

    # ===================== BEGIN FINAL FORECAST TOP BLOCK =====================
    st.markdown("## 📈 Final Forecast")

    # 1) Read all knobs from session (defined on the Models tab)
    metric            = st.session_state.get("metric", "SMAPE")
    horizon           = int(st.session_state.get("horizon", 12))
    folds             = int(st.session_state.get("folds", 6))
    include_models    = st.session_state.get(
        "include_models",
        ["Naive","sNaive","OLS","Elastic","LightGBM","XGBoost","HoltWinters","Prophet"]
    )
    price_adj         = float(st.session_state.get("price_adj", 1.0))
    promo_future      = float(st.session_state.get("promo_future", 0.0))
    season_shift_pct  = float(st.session_state.get("season_shift_pct", 0.0))



    # 2) Build features on full history once; anchor future dates to the features calendar
    hist_df = st.session_state.raw_df.copy().sort_values("date")
    feat, fcols = build_features(hist_df.copy())
    last_feat_date = pd.to_datetime(feat["date"]).max()
    future_dates = pd.date_range(last_feat_date + pd.Timedelta(weeks=1), periods=int(horizon), freq="W")

    # 3) Create a deterministic future driver stub (Monte Carlo will randomize later)
    future_df = pd.DataFrame({"date": future_dates})
    if "price" in hist_df.columns:
        future_df["price"] = float(hist_df["price"].dropna().iloc[-1]) * price_adj
    if "is_promo" in hist_df.columns:
        # deterministic stub: turns to 1’s when promo_prob>0.5, else 0’s
        future_df["is_promo"] = (np.full(len(future_df), promo_future) > 0.5).astype(int)

    # 4) Build features on (history + this future stub), rotate season for FUTURE rows only
    hist_plus_future = pd.concat([hist_df, future_df], ignore_index=True)
    final_feat, final_fcols = build_features(hist_plus_future)
    final_fcols = _ensure_drivers(final_fcols, final_feat)
    final_feat = _rotate_seasonal_features(final_feat, future_dates, season_shift_pct)

    # 5) Split to train / future
    future_feat = final_feat[final_feat["date"].isin(future_dates)]
    train_feat  = final_feat[~final_feat["date"].isin(future_dates)]
    Xtr, ytr = train_feat[final_fcols], train_feat["sales"]
    Xf       = future_feat[final_fcols]

    # 6) If features dropped rows (rare on short histories), use robust fallback for model training
    use_fallback = future_feat.empty or train_feat.empty or Xtr.shape[0]==0 or Xf.shape[0]==0

    # 7) Train candidate models on ALL data and build predictions dict
    preds = {}

    # Always provide at least Naive as a safe default
    from models import (
        fit_naive, predict_naive,
        fit_snaive, predict_snaive,
        fit_holt, predict_holt,
        train_full_and_forecast
    )
    # Naive (safe)
    naive_m = fit_naive(hist_df["sales"])
    preds["Naive"] = pd.DataFrame({"date": future_dates, "forecast": predict_naive(naive_m, horizon)})

    # sNaive (seasonal)
    if "sNaive" in include_models:
        snaive_m = fit_snaive(hist_df["sales"], seasonality=52)
        if snaive_m is not None:
            preds["sNaive"] = pd.DataFrame({"date": future_dates, "forecast": predict_snaive(snaive_m, horizon)})

    # OLS / Elastic / LightGBM / XGBoost use feature path if available; otherwise robust fallback
    if "OLS" in include_models:
        try:
            if use_fallback:
                df_fc = train_full_and_forecast(hist_df, build_features, final_fcols, "OLS", steps=horizon, seasonality=52)
                preds["OLS"] = df_fc
            else:
                from sklearn.linear_model import LinearRegression
                m = LinearRegression().fit(Xtr, ytr)
                preds["OLS"] = pd.DataFrame({"date": future_feat["date"], "forecast": m.predict(Xf)})
        except Exception as e:
            st.info(f"OLS skipped: {e}")

    if "Elastic" in include_models:
        try:
            if use_fallback:
                df_fc = train_full_and_forecast(hist_df, build_features, final_fcols, "ElasticNet", steps=horizon, seasonality=52)
                preds["Elastic"] = df_fc
            else:
                from sklearn.linear_model import ElasticNet
                m = ElasticNet(alpha=0.0005, l1_ratio=0.1, max_iter=5000).fit(Xtr, ytr)
                preds["Elastic"] = pd.DataFrame({"date": future_feat["date"], "forecast": m.predict(Xf)})
        except Exception as e:
            st.info(f"Elastic skipped: {e}")

    if "LightGBM" in include_models:
        try:
            from models import fit_lgbm, predict_lgbm
            if use_fallback:
                df_fc = train_full_and_forecast(hist_df, build_features, final_fcols, "LightGBM", steps=horizon, seasonality=52)
                preds["LightGBM"] = df_fc
            else:
                m = fit_lgbm(Xtr, ytr)
                preds["LightGBM"] = pd.DataFrame({"date": future_feat["date"], "forecast": predict_lgbm(m, Xf)})
        except Exception as e:
            st.info(f"LightGBM skipped: {e}")

    if "XGBoost" in include_models:
        try:
            from models import fit_xgb, predict_xgb
            if use_fallback:
                df_fc = train_full_and_forecast(hist_df, build_features, final_fcols, "XGBoost", steps=horizon, seasonality=52)
                preds["XGBoost"] = df_fc
            else:
                m = fit_xgb(Xtr, ytr)
                preds["XGBoost"] = pd.DataFrame({"date": future_feat["date"], "forecast": predict_xgb(m, Xf)})
        except Exception as e:
            st.info(f"XGBoost skipped: {e}")

    if "HoltWinters" in include_models:
        try:
            holt_m = fit_holt(hist_df["sales"], seasonality=52)
            preds["HoltWinters"] = pd.DataFrame({"date": future_dates, "forecast": predict_holt(holt_m, horizon)})
        except Exception as e:
            st.info(f"HoltWinters skipped: {e}")

    if "Prophet" in include_models:
        try:
            from models import fit_prophet, predict_prophet, _HAS_PROPHET
            if not _HAS_PROPHET:
                raise ImportError("prophet not available in this environment")

            # Train Prophet on (date, sales) and carry forward any drivers if present
            df_tr = hist_df[["date", "sales"]].copy()
            for c in ["price", "is_promo"]:
                if c in hist_df.columns:
                    df_tr[c] = hist_df[c]
            p_bundle = fit_prophet(df_tr)
            yhat = predict_prophet(p_bundle, steps=horizon)
            preds["Prophet"] = pd.DataFrame({"date": future_dates, "forecast": yhat})
        except Exception as e:
            st.info(f"Prophet skipped: {e}")

    # 8) Persist model forecasts for other modules (ROI/Risk, etc.)
    st.session_state["preds_dict"] = preds  # ✅ keep the per-model forecasts handy

    # 9) Build error map from backtest (names normalized)
    errs = st.session_state.bt["errors"]
    name_map = {"Elastic": "ElasticNet", "HoltWinters": "HoltWintersSafe"}
    error_for_weight = {
        k: float(errs.get(name_map.get(k, k), np.nan) if np.isfinite(errs.get(name_map.get(k, k), np.nan)) else 999.0)
        for k in preds.keys()
    }

    # 9) Ensemble (inverse-error × volatility penalty)
    ens = weighted_ensemble(preds, error_for_weight)

    # 10) User picks final — radio FIRST so chart appears immediately after
    choice = st.radio("Final forecast:", ["Best by backtest", "Weighted ensemble"], horizontal=True)
    st.caption(
    "💡 **Best by backtest** trains the single top-performing model from validation.  \n"
    "🤝 **Weighted ensemble** combines all models dynamically, giving higher weight to the "
    "ones with lower error and more stable forecasts."
    )
    with st.expander("🤝 How are the weights computed?"):
        st.write(
            "- **Inverse error:** Models with lower backtest error get **larger weights**.  \n"
            "- **Volatility penalty:** If a model’s forecast swings a lot (high std/mean), its weight is **reduced**.  \n"
            "- **Normalization:** Weights are scaled to sum to 1; the final forecast is the **weighted average** on each date."
        )

    final_fc, chosen_name = _final_choice_from_radio(choice, preds, ens)
    if choice == "Weighted ensemble" and ens is not None:
        final_fc = ens
        chosen_name = "Ensemble (1/error weights)"
        st.session_state.chosen_name = chosen_name

    # 11) Reliability (stability & intervals) — same logic for both paths
    from models import fit_lgbm_quantile, predict_lgbm_quantile
    st.markdown("### 🔍 Forecast Reliability Insights")
    st.caption(
        "This section helps you **trust** your forecasts by showing how consistent, "
        "uncertain, and stable they are across models and time."
    )

    stab_score, _stab_df = stability_score_from_preds(preds)
    if stab_score >= 80:
        stab_badge, stab_color = "🟢 Stable", "green"
    elif stab_score >= 60:
        stab_badge, stab_color = "🟡 Moderate", "orange"
    else:
        stab_badge, stab_color = "🔴 Unstable", "red"
    with st.expander("📊 What does Stability mean?", expanded=False):
        st.write(
            "The **Forecast Stability Score** (0–100) measures how much the different "
            "models agree with each other. If all models give similar results, the score "
            "is close to 100 (🟢 Stable). Large disagreements make it lower (🟡 Moderate "
            "or 🔴 Unstable)."
        )

    st.markdown(
        f"**Stability:** <span style='color:{stab_color}; font-weight:bold'>{stab_badge}</span> — {stab_score:.0f}/100",
        unsafe_allow_html=True,
    )
    # Simple regime-shift check (pattern change detector)
    try:
        shifted, shift_msg = detect_regime_shift(hist_df)
    except Exception:
        shifted, shift_msg = False, ""

    with st.expander("🧭 Regime shifts — what and why it matters", expanded=False):
        st.write(
            "A **regime shift** means the data’s underlying pattern changed (e.g., a new promo policy, a channel mix "
            "change, or an external shock). When a shift is detected, backtests become less representative and intervals "
            "may widen. Consider **retraining more often**, using **shorter horizons**, or adding **new drivers**."
        )
        if shift_msg:
            st.caption(f"Detector note: {shift_msg}")
    if shifted:
        st.warning("Possible **regime shift** detected — treat recent backtest scores with caution and prefer ensembles.")

    st.divider()
    st.markdown("### 🎯 Confidence Intervals")
    st.caption(
        "Confidence intervals show the **range of plausible outcomes** given model uncertainty or past forecast errors. "
        "A **wider band** means the model is less certain."
    )
    with st.expander("ℹ️ How are these intervals calculated?"):
        st.write(
            "- **Bootstrap residuals:** Re-samples recent forecast errors and adds them to the baseline forecast. "
            "This reflects the error you’ve actually seen in the past.\n"
            "- **LightGBM quantiles:** Trains special models to predict the **10th** and **90th** percentile forecasts. "
            "This is faster and more model-aware when feature rows are available."
        )

    interval_method = st.radio(
        "Choose uncertainty method:",
        ["Bootstrap residuals", "LightGBM quantiles"],
        horizontal=True,
        index=0,
        help="Bootstrap = resample past residuals. Quantiles = model-based 10th/90th percentiles."
    )
    fc = final_fc.copy()
    try:
        if interval_method == "LightGBM quantiles" and not use_fallback and chosen_name in ("LightGBM", "Ensemble (1/error weights)"):
            q10 = fit_lgbm_quantile(Xtr, ytr, alpha=0.10)
            q90 = fit_lgbm_quantile(Xtr, ytr, alpha=0.90)
            fc["lower"] = predict_lgbm_quantile(q10, Xf)
            fc["upper"] = predict_lgbm_quantile(q90, Xf)
        else:
            # small bootstrap on recent residuals
            k = min(26, max(8, len(hist_df) // 6))
            backcast_train = hist_df.iloc[:-k].copy()
            backcast_test  = hist_df.iloc[-k:].copy()
            feat_tr, fcols_tr = build_features(backcast_train)
            feat_te, fcols_te = build_features(backcast_test)
            common_cols = [c for c in fcols_tr if c in feat_te.columns]
            if (len(feat_tr)==0) or (len(feat_te)==0) or (len(common_cols)==0):
                s = hist_df["sales"].astype(float).values
                y_hat = pd.Series(s).rolling(8, min_periods=1).mean().to_numpy()
                resid = (s - y_hat)[-k:]
            else:
                from lightgbm import LGBMRegressor
                tr = feat_tr.dropna(subset=common_cols + ["sales"])
                te = feat_te.dropna(subset=common_cols + ["sales"])
                if (len(tr)==0) or (len(te)==0):
                    s = hist_df["sales"].astype(float).values
                    y_hat = pd.Series(s).rolling(8, min_periods=1).mean().to_numpy()
                    resid = (s - y_hat)[-k:]
                else:
                    m_resid = LGBMRegressor(
                        random_state=0, n_estimators=300, learning_rate=0.05, num_leaves=31, min_data_in_leaf=5
                    ).fit(tr[common_cols], tr["sales"])
                    y_hat = m_resid.predict(te[common_cols])
                    resid = te["sales"].to_numpy(dtype=float) - y_hat.astype(float)
            rng = np.random.default_rng(0)
            B, H = 500, len(fc)
            sim = np.empty((B, H), dtype=float)
            base = fc["forecast"].to_numpy(dtype=float)
            for b in range(B):
                noise = rng.choice(resid, size=H, replace=True)
                sim[b, :] = base + noise
            fc["lower"] = np.percentile(sim, 10, axis=0)
            fc["upper"] = np.percentile(sim, 90, axis=0)
    except Exception as e:
        st.info(f"Could not compute intervals: {e}")

    # 12) Plot at the TOP
    st.session_state.final_fc = fc
    fig = plot_history_forecast(hist_df, fc, title=f"Chosen: {chosen_name}")
    st.pyplot(fig, use_container_width=True)
    # ===================== END FINAL FORECAST TOP BLOCK =====================

    if "bt" not in st.session_state:
        st.info("Run backtest first.")
    else:
        # --- READ persisted selections from the Models tab ---
        metric = st.session_state.get("metric", "SMAPE")
        horizon = int(st.session_state.get("horizon", 12))
        folds = int(st.session_state.get("folds", 6))
        include_models = st.session_state.get("include_models", ["Naive","sNaive","OLS","Elastic","LightGBM","XGBoost","HoltWinters","Prophet"])
        price_adj = float(st.session_state.get("price_adj", 1.0))
        promo_future = float(st.session_state.get("promo_future", 0.0))
        season_shift_pct = float(st.session_state.get("season_shift_pct", 0.0))
        # ------------------------------------------------------
        feat = st.session_state.bt["feat"]
        fcols = st.session_state.bt["fcols"]
        best = st.session_state.bt["best"] or "OLS"

        # Train each candidate on ALL data for final forecast
        hist = feat[["date","sales"]].copy()
        last_feat_date = pd.to_datetime(feat["date"]).max()

        # Scenario: adjust future drivers
        future_dates = pd.date_range(last_feat_date + pd.Timedelta(weeks=1), periods=int(horizon), freq="W")
        future_df = pd.DataFrame({"date": future_dates})
        # naive driver fill (carry last known)
        for c in ["price", "is_promo"]:
            if c in st.session_state.raw_df.columns:
                val = st.session_state.raw_df[c].dropna().iloc[-1]
            else:
                val = 0 if c == "is_promo" else 1.0
            future_df[c] = val

        

        # --- v1.4: Scenario Lab & Comparison (add-on UI) ---------------------------
        base_scn = Scenario(
            name="Baseline",
            price_mult=float(price_adj),
            promo_prob=float(promo_future),
            season_shift_pct=float(season_shift_pct),
            notes="From sliders/preset"
        )

        # --- ONE Unified Scenario Lab (presets + custom) -----------------------------
        with st.expander("🎛️ Scenario Lab", expanded=not simple_mode):
            st.caption("Build scenarios for comparison: pick a Preset (with description) or choose **Custom** and tweak the sliders. Baseline is defined here too.")

            # Preset catalog (descriptions included)
            _SCENARIO_PRESETS = {
                "Custom": {
                    "desc": "Start from your current baseline values and adjust manually.",
                    "vals": None,  # None = use sliders/session values
                },
                "Holiday Push (+seasonality, +promo)": {
                    "desc": "Seasonal uplift and more promos during holiday weeks.",
                    "vals": {"price": 1.00, "promo": 0.35, "season": +12.0},
                },
                "Back-to-School (+seasonality)": {
                    "desc": "Phase shift toward late-summer / early-fall demand.",
                    "vals": {"price": 1.00, "promo": 0.20, "season": +6.0},
                },
                "Discount Blitz (−price, +promo)": {
                    "desc": "Aggressive discounting and frequent promos.",
                    "vals": {"price": 0.92, "promo": 0.50, "season": 0.0},
                },
                "Price Hike (+price, −promo)": {
                    "desc": "Nudge price upward while tightening promos.",
                    "vals": {"price": 1.05, "promo": 0.10, "season": 0.0},
                },
                "Clearance Fire Sale (−15% price, high promo)": {
                    "desc": "Short-term sell-through with deep discounts.",
                    "vals": {"price": 0.85, "promo": 0.60, "season": -5.0},
                },
                "Loyalty Lift (stable price, modest promo)": {
                    "desc": "Steady pricing; modest promos to drive repeats.",
                    "vals": {"price": 1.00, "promo": 0.25, "season": 0.0},
                },
                "Promo Freeze (no promos)": {
                    "desc": "Turn off promos to measure baseline demand.",
                    "vals": {"price": 1.00, "promo": 0.00, "season": 0.0},
                },
                "Inflation Squeeze (+2% price)": {
                    "desc": "Small price uptick; keep current promo cadence.",
                    "vals": {"price": 1.02, "promo": promo_future, "season": 0.0},
                },
                "E-commerce Surge (+seasonality)": {
                    "desc": "Stronger peak seasonality (e.g., Q4 online spikes).",
                    "vals": {"price": 1.00, "promo": 0.30, "season": +15.0},
                },
            }

            # --- Baseline definition (Preset OR Custom sliders) ---
            st.markdown("**Baseline scenario**")
            colB1, colB2 = st.columns([1,2])
            with colB1:
                base_choice = st.selectbox("Preset", list(_SCENARIO_PRESETS.keys()), index=0, key="baseline_choice")
            with colB2:
                st.caption(_SCENARIO_PRESETS[base_choice]["desc"])

            # Seed from session
            _bp  = float(st.session_state.get("price_adj", 1.00))
            _br  = float(st.session_state.get("promo_future", 0.00))
            _bs  = float(st.session_state.get("season_shift_pct", 0.0))
            # Apply preset if not Custom
            pv = _SCENARIO_PRESETS[base_choice]["vals"]
            if pv is not None:
                _bp, _br, _bs = float(pv["price"]), float(pv["promo"]), float(pv["season"])

            cb1, cb2, cb3 = st.columns(3)
            with cb1:
                _bp = st.slider("Price multiplier", 0.80, 1.20, _bp, 0.01, key="price_adj")
            with cb2:
                _br = st.slider("Promo probability (future)", 0.00, 0.60, _br, 0.05, key="promo_future")
            with cb3:
                _bs = st.slider("Seasonal shift (%)", -20.0, 20.0, _bs, 1.0, key="season_shift_pct")

            base_scn = Scenario(
                name="Baseline",
                price_mult=float(_bp),
                promo_prob=float(_br),
                season_shift_pct=float(_bs),
                notes=f"Baseline via Scenario Lab: {base_choice}",
            )

            st.markdown("---")
            st.markdown("**Add comparison scenarios**")
            n_extra = st.number_input("How many scenarios to compare (besides Baseline)?", 0, 5, 1)

            extra_scenarios = []
            for i in range(int(n_extra)):
                st.markdown(f"**Scenario {i+1}**")
                colS1, colS2 = st.columns([1,2])
                with colS1:
                    sc_choice = st.selectbox(
                        f"Preset {i+1}",
                        list(_SCENARIO_PRESETS.keys()),
                        index=0,
                        key=f"sc_choice_{i}"
                    )
                with colS2:
                    st.caption(_SCENARIO_PRESETS[sc_choice]["desc"])

                # Seed from Baseline (nice UX) then apply preset if chosen
                _p = float(_bp); _r = float(_br); _s = float(_bs)
                pv = _SCENARIO_PRESETS[sc_choice]["vals"]
                if pv is not None:
                    _p, _r, _s = float(pv["price"]), float(pv["promo"]), float(pv["season"])

                c1, c2, c3 = st.columns(3)
                with c1:
                    _p = st.slider(f"Price x ({i+1})", 0.80, 1.20, _p, 0.01, key=f"sc_price_{i}")
                with c2:
                    _r = st.slider(f"Promo prob ({i+1})", 0.00, 0.60, _r, 0.05, key=f"sc_promo_{i}")
                with c3:
                    _s = st.slider(f"Season shift % ({i+1})", -20.0, 20.0, _s, 1.0, key=f"sc_season_{i}")

                nm = st.text_input(f"Name ({i+1})", value=(sc_choice if sc_choice != "Custom" else f"Scenario {i+1}"), key=f"sc_name_{i}")
                extra_scenarios.append(Scenario(nm.strip(), float(_p), float(_r), float(_s), notes="Scenario Lab"))
        # ---------------------------------------------------------------------------

            all_scenarios = [base_scn] + extra_scenarios
            scenario_rows = []

            for idx, sc in enumerate(all_scenarios):
                # 1) Start from your future driver stub (we’ll apply sc)
                sc_future = _apply_scenario_future(future_df, sc, rng_seed=idx)

                # 2) Rebuild features on (hist + this scenario’s future)
                hist_raw_for_sc = st.session_state.raw_df.copy()
                sc_hist = pd.concat([hist_raw_for_sc, sc_future], ignore_index=True)
                sc_feat, sc_fcols = build_features(sc_hist)
                sc_fcols = _ensure_drivers(sc_fcols, sc_feat)

                # 3) Rotate seasonal basis for FUTURE rows only
                sc_feat = _rotate_seasonal_features(sc_feat, future_dates, sc.season_shift_pct)

                # 4) Split to train / future
                future_feat_sc = sc_feat[sc_feat["date"].isin(future_dates)]
                train_feat_sc  = sc_feat[~sc_feat["date"].isin(future_dates)]
                use_fallback_sc = future_feat_sc.empty or train_feat_sc.empty

                # 5) Fast forecast (LightGBM) for the table
                try:
                    if use_fallback_sc:
                        df_fc_sc = train_full_and_forecast(
                            df_hist=sc_hist,
                            make_features_fn=build_features,
                            feature_cols=sc_fcols,
                            model_name="LightGBM",
                            steps=int(horizon),
                            seasonality=52,
                        )
                    else:
                        from models import fit_lgbm, predict_lgbm
                        Xtr_sc, ytr_sc = train_feat_sc[sc_fcols], train_feat_sc["sales"]
                        Xf_sc          = future_feat_sc[sc_fcols]
                        m_sc = fit_lgbm(Xtr_sc, ytr_sc)
                        yhat_sc = predict_lgbm(m_sc, Xf_sc)
                        df_fc_sc = pd.DataFrame({"date": future_dates, "forecast": yhat_sc})
                except Exception:
                    # If anything fails, skip row gracefully
                    continue

                revenue = float(np.nansum(df_fc_sc["forecast"]))
                # Confidence badge vs current best by backtest
                errs_bt = st.session_state.bt["errors"]
                best_model = min(errs_bt, key=errs_bt.get) if errs_bt else None
                conf_badge, _ = _confidence_badge(st.session_state.get("metric","SMAPE"), errs_bt.get(best_model, 999.0)) if best_model else ("🟡 Medium Confidence","")

                # Profit (margin input only visible in advanced mode)
                margin = st.number_input("Gross Margin % (for profit)", 0.0, 100.0, 35.0, 1.0, key=f"margin_scn_{idx}", help="Used for Profit calc.") if not simple_mode else 35.0
                profit = revenue * (margin / 100.0)
                scenario_rows.append({"Scenario": sc.name, "Revenue": revenue, "Profit": profit, "Confidence": conf_badge, "Recommendation": "—"})

            # Show table (if any extra scenarios exist or just Baseline)
            if scenario_rows:
                st.subheader("🧮 Scenario Comparison")
                _tbl = pd.DataFrame(scenario_rows)
                # Recommend by Profit; break ties by Confidence bucket (map to numeric rank)
                def _conf_rank(lbl: str) -> int:
                    if lbl.startswith("🟢"): return 3
                    if lbl.startswith("🟡"): return 2
                    if lbl.startswith("🔴"): return 1
                    return 0

                _tbl["ConfidenceRank"] = _tbl["Confidence"].astype(str).map(_conf_rank).fillna(0).astype(int)
                winner = _tbl.sort_values(["Profit", "ConfidenceRank"], ascending=[False, False]).iloc[0]["Scenario"]
                _tbl.loc[_tbl["Scenario"] == winner, "Recommendation"] = "✅ Top pick"
                st.dataframe(_tbl.drop(columns=["ConfidenceRank"]).round(2), use_container_width=True)
                # Quick winner caption (and Δ vs Baseline if available)
                win_row = _tbl[_tbl["Scenario"] == winner].iloc[0]
                st.caption(
                    f"Top pick: **{winner}** — Revenue: {win_row['Revenue']:,.0f} | Profit: {win_row['Profit']:,.0f}"
                )
                if "Baseline" in _tbl["Scenario"].values:
                    base_row = _tbl[_tbl["Scenario"] == "Baseline"].iloc[0]
                    d_rev = win_row["Revenue"] - base_row["Revenue"]
                    d_prof = win_row["Profit"] - base_row["Profit"]
                    st.caption(f"Δ vs Baseline — Revenue: {d_rev:,.0f} | Profit: {d_prof:,.0f}")
            # ---------------------------------------------------------------------------

        

        # --- v1.4: Monte Carlo (around Baseline sliders) ---------------------------
        st.subheader("🎲 Monte Carlo (price/promo uncertainty)")
        with st.expander("Run Monte Carlo", expanded=False):
            N = st.slider("Simulations", 200, 2000, 600, 100)
            promo_std = st.slider("Promo volatility (±)", 0.0, 0.25, 0.10, 0.01, help="Randomize promo prob per run.")
            price_std = st.slider("Price volatility (±)", 0.0, 0.10, 0.03, 0.005, help="Randomize price multiplier per run.")
            if st.button("Run Simulation", use_container_width=True):
                # Pull the freshest values from session right before running
                _bprice  = float(st.session_state.get("price_adj", 1.0))
                _bpromo  = float(st.session_state.get("promo_future", 0.0))
                _bseason = float(st.session_state.get("season_shift_pct", 0.0))
                totals, mu, p10, p90 = run_monte_carlo_cached(
                    raw_df=st.session_state.raw_df,
                    horizon=int(horizon),
                    base_price_mult=float(price_adj),
                    base_promo_prob=float(promo_future),
                    season_shift_pct=float(season_shift_pct),
                    promo_std=float(promo_std),
                    price_std=float(price_std),
                    N=int(N),
                    seed=0,
                )
                st.write(f"**Mean:** {mu:,.0f} | **P10:** {p10:,.0f} | **P90:** {p90:,.0f}")

                totals = np.asarray(totals, dtype=float)
                totals = totals[np.isfinite(totals)]
                if totals.size == 0 or not np.isfinite(mu):
                    st.warning("Simulation produced no valid totals (feature pipeline dropped all future rows). "
                               "Try a shorter horizon or smaller volatility; also ensure drivers are included.")
                else:
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=(8,3.5))
                    plt.hist(totals, bins=30, density=True)
                    plt.axvline(mu, linestyle="--")
                    plt.axvline(p10, linestyle=":")
                    plt.axvline(p90, linestyle=":")
                    plt.title("Monte Carlo Revenue Distribution")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

        # --- v1.4: Price Elasticity (±10% around current price) --------------------
        st.subheader("📉 Price Elasticity (local)")
        with st.expander("Show sensitivity curve (±10%)", expanded=False):
            from models import fit_lgbm, predict_lgbm
            steps_pct = np.linspace(-0.10, 0.10, 11)
            revs = []
            for j, dp in enumerate(steps_pct):
                sc = Scenario("Elasticity", price_mult=price_adj*(1.0+dp),
                              promo_prob=promo_future, season_shift_pct=season_shift_pct)
                sc_future = _apply_scenario_future(future_df, sc, rng_seed=10+j)
                sc_hist = pd.concat([st.session_state.raw_df, sc_future], ignore_index=True)
                sc_feat, sc_fcols = build_features(sc_hist)
                sc_fcols = _ensure_drivers(sc_fcols, sc_feat)
                sc_feat = _rotate_seasonal_features(sc_feat, future_dates, sc.season_shift_pct)
                fut = sc_feat[sc_feat["date"].isin(future_dates)]
                tr  = sc_feat[~sc_feat["date"].isin(future_dates)]
                if fut.empty or tr.empty:
                    df_fc = train_full_and_forecast(
                        df_hist=sc_hist,
                        make_features_fn=build_features,
                        feature_cols=sc_fcols,
                        model_name="LightGBM",
                        steps=int(horizon), seasonality=52,
                    )
                    revs.append(float(np.nansum(df_fc["forecast"])))
                else:
                    Xtr_sc, ytr_sc = tr[sc_fcols], tr["sales"]
                    Xf_sc          = fut[sc_fcols]
                    m = fit_lgbm(Xtr_sc, ytr_sc)
                    yhat = predict_lgbm(m, Xf_sc)
                    revs.append(float(np.nansum(yhat)))

            # local slope ≈ ΔRev / ΔPrice near 0
            mid = len(steps_pct)//2
            slope = (revs[mid+1] - revs[mid-1]) / (steps_pct[mid+1] - steps_pct[mid-1])
            st.caption(f"Estimated local elasticity (revenue change per +1.0 price-mult): {slope:,.2f}")

            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8,3.5))
            plt.plot((steps_pct*100.0), revs)
            plt.title("Revenue vs Price Adjustment (±10%)")
            plt.xlabel("Price change (%)")
            plt.ylabel("Revenue (horizon sum)")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        # ---------------------------------------------------------------------------
    
        # === v1.5 — RECOMMENDATION ENGINE + AI EXECUTIVE SUMMARY ======================
        st.subheader("💡 Recommendation Engine + AI Executive Summary")

        # ---- Business inputs (used for Profit/ROI calculations) ----------------------
        with st.expander("Business assumptions (edit to your reality)", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            gross_margin_pct = col_a.slider("Gross margin % of revenue", 5, 95, 35, 1,
                                            help="Used to estimate profit from revenue deltas.")
            promo_budget_per_week = col_b.number_input(
                "Promo budget per promo-active week ($)", min_value=0.0, value=500.0, step=50.0,
                help="Allocated spend when a week is in promo (probabilistic)."
            )
            price_change_cost_per_pct = col_c.number_input(
                "Operational cost per 1% price change ($)", min_value=0.0, value=50.0, step=10.0,
                help="Execution cost to move price (catalog, signage, ops)."
            )
            st.caption("Tip: set these roughly—relative comparisons still hold.")

        # Pull freshest baseline knobs from Scenario Lab or session defaults
        _h = int(st.session_state.get("horizon", 12))
        _price_mult_base = float(st.session_state.get("price_adj", 1.00))
        _promo_prob_base = float(st.session_state.get("promo_future", 0.00))
        _season_shift = float(st.session_state.get("season_shift_pct", 0.0))

        # Convenience: last observed price (fallback to 10 if missing)
        _hist = st.session_state.raw_df.copy().sort_values("date")
        _last_price = float(_hist["price"].dropna().iloc[-1]) if "price" in _hist.columns and _hist["price"].notna().any() else 10.0
        _last_promo = float(_hist["is_promo"].rolling(12, min_periods=1).mean().iloc[-1]) if "is_promo" in _hist.columns else 0.0

        # --- small helpers ------------------------------------------------------------
        def _forecast_sum_units(price_mult: float, promo_prob: float) -> float:
            """
            Fast, local scenario evaluator (LightGBM) akin to your elasticity block:
            trains on history, forecasts horizon for a synthetic future with specified price/promo.
            Falls back to train_full_and_forecast if split is degenerate.
            """
            from models import fit_lgbm, predict_lgbm, train_full_and_forecast
            # Build synthetic future rows by reusing your Scenario helper path
            # 1) extend future dates
            future_dates = pd.date_range(_hist["date"].max() + pd.Timedelta(days=7), periods=_h, freq="W")
            # 2) clone hist; set future driver values
            df_future = pd.DataFrame({"date": future_dates})
            if "price" in _hist.columns:
                df_future["price"] = _last_price * price_mult
            if "is_promo" in _hist.columns:
                # Bernoulli probability as expectation ≈ promo_prob
                df_future["is_promo"] = promo_prob
            sc_hist = pd.concat([_hist, df_future], ignore_index=True)

            sc_feat, sc_fcols = build_features(sc_hist)
            # ensure core drivers stay
            sc_fcols = list(dict.fromkeys([c for c in ["price", "is_promo", "woy_sin", "woy_cos"] if c in sc_feat.columns] + sc_fcols))

            fut = sc_feat[sc_feat["date"].isin(future_dates)]
            tr  = sc_feat[~sc_feat["date"].isin(future_dates)]
            if fut.empty or tr.empty or len(tr) < 30 or len(sc_fcols) == 0:
                df_fc = train_full_and_forecast(
                    df_hist=sc_hist,
                    make_features_fn=build_features,
                    feature_cols=sc_fcols,
                    model_name="LightGBM",
                    steps=_h, seasonality=52,
                )
                return float(np.nansum(df_fc["forecast"]))
            else:
                Xtr, ytr = tr[sc_fcols], tr["sales"].astype(float)
                Xf       = fut[sc_fcols]
                m = fit_lgbm(Xtr, ytr)
                yhat = predict_lgbm(m, Xf)
                return float(np.nansum(yhat))

        def _revenue_profit(price_mult: float, promo_prob: float) -> tuple[float, float]:
            units = _forecast_sum_units(price_mult, promo_prob)
            # Expected future price level ≈ last observed price × multiplier
            revenue = units * (_last_price * price_mult)
            # Expected number of promo-active weeks ≈ promo_prob * horizon
            promo_cost = promo_budget_per_week * (promo_prob * _h)
            # Price change execution cost (absolute pct change)
            exec_cost = price_change_cost_per_pct * abs((price_mult - 1.0) * 100.0)
            profit = revenue * (gross_margin_pct / 100.0) - promo_cost - exec_cost
            return revenue, profit

        @st.cache_data(show_spinner=False)
        def _threshold_from_curve():
            """Recompute the local ±10% price curve and detect an inflection where revenue declines."""
            steps_pct = np.linspace(-0.10, 0.10, 11)
            revs = []
            for dp in steps_pct:
                pm = _price_mult_base * (1.0 + dp)
                rev, _ = _revenue_profit(pm, _promo_prob_base)
                revs.append(rev)
            # Find first point to the right of 0 where revenue decreases relative to previous
            idx0 = np.where(np.isclose(steps_pct, 0.0))[0][0]
            decline_at = None
            for k in range(idx0 + 1, len(steps_pct)):
                if revs[k] < revs[k - 1]:
                    decline_at = steps_pct[k]
                    break
            return steps_pct, revs, decline_at

        # Limit the upper bound to the first local decline if one exists
        steps_pct, revs, decline_at = _threshold_from_curve()
        upper_cap = 1.10
        if decline_at is not None:
            # decline_at is in fractional terms (e.g., 0.06 = +6%)
            upper_cap = min(upper_cap, 1.0 + float(decline_at))

        price_steps = np.linspace(0.90*_price_mult_base, upper_cap*_price_mult_base, 7)
        promo_steps = np.linspace(max(0.0, _promo_prob_base - 0.10), min(0.95, _promo_prob_base + 0.10), 7)
        def _grid_recommendations():
            # Baseline for deltas
            base_rev, base_prof = _revenue_profit(_price_mult_base, _promo_prob_base)
            # Search
            cand = []
            for pm in price_steps:
                for pp in promo_steps:
                    rev, prof = _revenue_profit(pm, pp)
                    cand.append({
                        "price_mult": pm,
                        "promo_prob": pp,
                        "Revenue": rev,
                        "Profit": prof,
                        "dRev": rev - base_rev,
                        "dProf": prof - base_prof,
                    })
            df = pd.DataFrame(cand)
            # distance from baseline (smaller is nicer when profits tie)
            df["move_size"] = np.hypot(
                (df["price_mult"]/_price_mult_base - 1.0) * 100.0,
                (df["promo_prob"] - _promo_prob_base) * 100.0
            )
            # Sort by ΔProfit desc, then smaller move
            df = df.sort_values(["dProf", "move_size"], ascending=[False, True]).reset_index(drop=True)
            return df, base_rev, base_prof

        def _threshold_from_curve():
            """Recompute the local ±10% price curve and detect an inflection where revenue declines."""
            steps_pct = np.linspace(-0.10, 0.10, 11)
            revs = []
            for dp in steps_pct:
                pm = _price_mult_base * (1.0 + dp)
                rev, _ = _revenue_profit(pm, _promo_prob_base)
                revs.append(rev)
            # Find first point to the right of 0 where revenue decreases relative to previous
            idx0 = np.where(np.isclose(steps_pct, 0.0))[0][0]
            decline_at = None
            for k in range(idx0+1, len(steps_pct)):
                if revs[k] < revs[k-1]:
                    decline_at = steps_pct[k]
                    break
            return steps_pct, revs, decline_at

        def _risk_level_from_stability():
            # Uses your stability score across model forecasts if available
            preds = st.session_state.get("preds_dict") or {}
            try:
                score, _ = stability_score_from_preds(preds)
            except Exception:
                score = 60.0
            if score >= 80: return "Low", score
            if score >= 60: return "Medium", score
            return "High", score

        # -------------------- Auto Recommendations ------------------------------------
        st.markdown("### 🤖 Auto Recommendations")
        rec_df, base_rev, base_prof = _grid_recommendations()

        # Risk level from forecast stability (global, simple & consistent)
        risk_level, stab_score = _risk_level_from_stability()

        # Best option by ΔProfit
        if rec_df.empty:
            st.info("No valid recommendations in the ±10% price / ±10pp promo window.")
            _best = None
        else:
            best_row = rec_df.iloc[0].copy()
            pm_pct_best = (best_row["price_mult"]/_price_mult_base - 1.0) * 100.0
            pp_pct_best = (best_row["promo_prob"] - _promo_prob_base) * 100.0

            if risk_level == "Low":
                # ✅ Low risk → give one clear optimal pick only
                st.success(
                    f"📈 **Optimal Strategy:** Price {pm_pct_best:+.1f}% & Promo {pp_pct_best:+.1f}pp → "
                    f"ΔProfit **{best_row['dProf']:+,.0f}**, ΔRevenue {best_row['dRev']:+,.0f} "
                    f"(Stability {stab_score:.0f}/100, {risk_level} risk)"
                )
                _best = best_row
            else:
                # ⚠️ Non-low risk → show optimal + safer alternatives
                # Define “safer” = smaller move magnitudes from baseline, among good ΔProf
                rec_df = rec_df.assign(
                    move_size = (rec_df["price_mult"]-_price_mult_base).abs()
                                + (rec_df["promo_prob"]-_promo_prob_base).abs()
                )
                # Consider top 10 by profit, then pick the 2 smallest moves (not equal to the best) with non-negative ΔProf
                top10 = rec_df.head(10)
                safer = (
                    top10[top10.index != top10.index[0]]
                    .query("dProf >= 0")
                    .sort_values(["move_size","dProf"], ascending=[True, False])
                    .head(2)
                    .copy()
                )

                # Show the optimal first
                st.warning(
                    f"🏁 **Highest Profit (higher risk):** Price {pm_pct_best:+.1f}% & Promo {pp_pct_best:+.1f}pp → "
                    f"ΔProfit **{best_row['dProf']:+,.0f}**, ΔRevenue {best_row['dRev']:+,.0f} "
                    f"(Stability {stab_score:.0f}/100, {risk_level} risk)"
                )

                # Then offer safer alternatives
                if not safer.empty:
                    st.markdown("**Lower-risk alternatives:**")
                    for i, r in safer.iterrows():
                        pm_pct = (r["price_mult"]/_price_mult_base - 1.0) * 100.0
                        pp_pct = (r["promo_prob"] - _promo_prob_base) * 100.0
                        st.write(
                            f"- Price {pm_pct:+.1f}%, Promo {pp_pct:+.1f}pp → "
                            f"ΔProfit **{r['dProf']:+,.0f}**, ΔRevenue {r['dRev']:+,.0f} "
                            f"(smaller move)"
                        )
                else:
                    st.caption("No sufficiently safer alternatives with positive profit found in this window.")
                _best = best_row

        # -------------------- Threshold Insights --------------------------------------
        st.markdown("### 📍 Threshold Insights")
        steps_pct, revs, decline_at = _threshold_from_curve()
        if decline_at is not None:
            st.info(f"Revenue **starts to decline above** ~**{decline_at*100:.1f}%** price change (relative to baseline).")
        else:
            st.info("No clear decline within ±10% price window — revenue is near flat/monotonic in this band.")

        # Show a small plot (no custom colors per your chart rules)
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,3.5))
        plt.plot(steps_pct*100.0, revs)
        plt.axvline(0, linestyle="--")
        plt.title("Projected Revenue vs Price Change (±10%)")
        plt.xlabel("Price change (%)")
        plt.ylabel("Revenue (horizon sum)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # -------------------- ROI & Risk Ranking --------------------------------------
        st.markdown("### 💹 ROI & Risk Ranking")
        risk_level, stab_score = _risk_level_from_stability()

        # Define two primary levers compared to baseline: PRICE and PROMO (one-at-a-time moves)
        lever_rows = []
        if _best is not None:
            # isolate price-only and promo-only deltas at same step as best (closest grid point)
            pm_only_rev, pm_only_prof = _revenue_profit(_best["price_mult"], _promo_prob_base)
            pr_only_rev, pr_only_prof = _revenue_profit(_price_mult_base, _best["promo_prob"])

            pm_cost = price_change_cost_per_pct * abs((_best["price_mult"] - 1.0)*100.0)
            pr_cost = promo_budget_per_week * ((_best["promo_prob"] - _promo_prob_base) * _h if _best["promo_prob"] > _promo_prob_base else 0.0)

            lever_rows.append({
                "Lever": "Price",
                "ROI_%": 100.0 * ((pm_only_prof - base_prof) / max(pm_cost, 1e-9)),
                "ΔProfit": pm_only_prof - base_prof,
                "ΔCost": pm_cost,
                "Risk": risk_level,
            })
            lever_rows.append({
                "Lever": "Promo",
                "ROI_%": 100.0 * ((pr_only_prof - base_prof) / max(pr_cost, 1e-9)) if pr_cost>0 else np.nan,
                "ΔProfit": pr_only_prof - base_prof,
                "ΔCost": pr_cost,
                "Risk": risk_level,
            })

        tbl = pd.DataFrame(lever_rows) if lever_rows else pd.DataFrame(columns=["Lever","ROI_%","ΔProfit","ΔCost","Risk"])
        st.dataframe(tbl.round(2), use_container_width=True)

        # -------------------- Executive Summary Paragraph -----------------------------
        st.markdown("### 🧾 AI Executive Summary")
        # Trend vs recent history
        _recent_k = min(12, len(_hist))
        recent_mean = float(_hist.tail(_recent_k)["sales"].mean()) if _recent_k>0 else np.nan
        # Produce the chosen (best) recommendation in words
        if _best is not None:
            pm_delta = (_best["price_mult"]/_price_mult_base - 1.0) * 100.0
            pp_delta = (_best["promo_prob"] - _promo_prob_base) * 100.0
            driver = "a targeted price move" if abs(pm_delta) >= abs(pp_delta) else "calibrated promo intensity"
            trend_pct = ( (base_rev - (recent_mean*_last_price*_h)) / max(recent_mean*_last_price*_h, 1e-9) ) * 100.0 if np.isfinite(recent_mean) else 0.0
            summary = (
                f"Sales are projected to change by {trend_pct:+.1f}% next quarter. "
                f"The model recommends focusing on {driver}, adjusting price by {pm_delta:+.1f}% "
                f"and promo probability by {pp_delta:+.1f}pp. "
                f"This yields an estimated profit impact of { _best['dProf']:+,.0f} "
                f"and revenue change of { _best['dRev']:+,.0f}. "
                f"Forecast stability is rated {stab_score:.0f}/100 ({risk_level} risk level)."
            )
        else:
            summary = "Forecast produced no dominant strategy within ±10%/±10pp; expect stable outcomes under current settings."
        st.write(summary)

        
        # ==============================================================================   
            

            
        # ---- end fallback guard ----

        # --------------------------------------------------------------------------

        # --- Signals ---
        lift_pct, price_elasticity, season_acf, trend_slope = 0.0, np.nan, np.nan, 0.0
        sales = _hist["sales"].astype(float).to_numpy()
        # Promo lift
        if "is_promo" in _hist.columns:
            # normalize the promo flag to clean 0/1 ints
            promo_flag = pd.to_numeric(_hist["is_promo"], errors="coerce").fillna(0).round().astype(int)
            if promo_flag.sum() >= 5 and (promo_flag == 0).sum() >= 5:
                promo_sales     = _hist.loc[promo_flag.eq(1), "sales"].astype(float).mean()
                nonpromo_sales  = _hist.loc[promo_flag.eq(0), "sales"].astype(float).mean()
                if pd.notna(nonpromo_sales) and nonpromo_sales > 0:
                    lift_pct = float((promo_sales - nonpromo_sales) / nonpromo_sales * 100.0)
                    st.write(f"**Promo Lift Estimate:** {lift_pct:.1f}%")

        # Price elasticity (quick OLS on logs with seasonal sin/cos and promo)
        if "price" in _hist.columns and _hist["price"].notna().sum() > 20:
            log_y = np.log(np.clip(_hist["sales"].astype(float).values, 1e-6, None))
            log_p = np.log(np.clip(_hist["price"].astype(float).ffill().values, 1e-6, None))
            week = pd.to_datetime(_hist["date"]).dt.isocalendar().week.astype(int).values
            s_sin = np.sin(2*np.pi * week / 52.0)
            s_cos = np.cos(2*np.pi * week / 52.0)
            promo = _hist["is_promo"].astype(float).values if "is_promo" in _hist.columns else np.zeros_like(log_y)
            X = np.column_stack([np.ones_like(log_y), log_p, promo, s_sin, s_cos])
            try:
                beta, *_ = np.linalg.lstsq(X, log_y, rcond=None)
                price_elasticity = float(beta[1])  # d log(y) / d log(price)
                st.write(f"**Estimated Price Elasticity:** {price_elasticity:.2f} (more negative = more sensitive)")
            except Exception:
                pass

        # Seasonality strength via ACF at lag 52
        if len(sales) >= 104:
            y = pd.Series(sales - np.nanmean(sales)).fillna(0.0).values
            lag = 52
            if len(y) > lag + 1 and np.std(y[:-lag]) > 0 and np.std(y[lag:]) > 0:
                season_acf = float(np.corrcoef(y[:-lag], y[lag:])[0,1])
                st.write(f"**Seasonality (lag-52 ACF):** {season_acf:.2f}")

        # Trend slope (units per week)
        if len(sales) > 5:
            t = np.arange(len(sales))
            slope, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(t), t]), sales, rcond=None)
            trend_slope = float(slope[1])
            st.write(f"**Trend Slope:** {trend_slope:+.2f} per week")

        # Improvement vs Naive (from backtest)
        errs_local = st.session_state.bt["errors"]
        cn = st.session_state.get("chosen_name")
        if cn and "Naive" in errs_local:
            best_err = errs_local.get(cn, np.nan)
            naive_err = errs_local["Naive"]
            if np.isfinite(best_err) and np.isfinite(naive_err) and naive_err > 0:
                imp = (naive_err - best_err)/naive_err * 100
                st.write(f"**Improvement vs Naive:** {imp:.1f}% better")
        # ---------------------------------------------------------------------------

        
        st.write("**Backtest average error by model**")
        bt_tbl = st.session_state.bt["folds"]
        st.dataframe(bt_tbl, use_container_width=True)


