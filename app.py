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
from sample_data_ml import gen_weekly_ml, gen_weekly_profile

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def weighted_ensemble(preds_dict: dict, error_by_model: dict):
    """
    preds_dict: {"ModelName": DataFrame(date, forecast), ...}
    error_by_model: {"ModelName": error_value, ...}  (lower is better)
    Returns a single DataFrame(date, forecast) or None if it cannot compute.
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
    weights = []
    for name, df in preds_dict.items():
        err = error_by_model.get(name, np.nan)
        if not np.isfinite(err) or err <= 0:
            continue
        w = 1.0 / err
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.set_index("date").reindex(all_dates)
        tmp = tmp.rename(columns={"forecast": name})
        aligned.append(tmp[name])
        weights.append((name, w))

    if not aligned or not weights:
        return None

    F = pd.concat(aligned, axis=1)  # cols are model names
    w_series = pd.Series({n: w for n, w in weights})
    w_series = w_series / w_series.sum()

    # Row-wise weighted average that ignores NaNs
    W = pd.DataFrame(
        np.broadcast_to(w_series.reindex(F.columns).fillna(0).values, F.shape),
        index=F.index, columns=F.columns
    )
    mask = ~F.isna()
    num = (F.fillna(0) * W).sum(axis=1)
    den = (mask * W).sum(axis=1).replace(0, np.nan)
    ens = (num / den).ffill().bfill()

    return pd.DataFrame({"date": all_dates.values, "forecast": ens.values})

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

st.set_page_config(page_title="Sales Forecaster", layout="wide")
st.title("🧠 Sales Forecaster (Weekly)")

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
        if st.button("Generate sample data"):
            st.session_state.raw_df = gen_weekly_ml(n_weeks=260, seed=np.random.randint(0,99999))
    if st.session_state.raw_df is not None:
        st.write("Preview:")
        st.dataframe(st.session_state.raw_df.head(12), use_container_width=True)
        q = data_quality_report(st.session_state.raw_df)
        color = "🟢" if q["missing"]==0 and q["gaps"]==0 else ("🟡" if q["missing"]<3 and q["gaps"]<2 else "🔴")
        st.info(f"{color} Rows: {q['rows']} | Range: {q['start']} → {q['end']} | Zeros: {q['zeros']} | Missing: {q['missing']} | Gaps: {q['gaps']}")

# ---------------- Models Tab ----------------
with tab_models:
    st.subheader("2) Configure models & backtest")
    colm1, colm2, colm3 = st.columns(3)
    with colm1:
        metric = st.selectbox("Metric (lower is better)", ["SMAPE","MAPE","MAE","RMSE"], index=0)
        horizon = st.selectbox("Forecast horizon (weeks)", [4,8,12,24], index=2)
        folds   = st.slider("Backtest folds", 3, 10, 6)
    with colm2:
        use_holidays = st.checkbox("Use holidays (US)", value=True, disabled=True)
        include_models = st.multiselect(
            "Models to consider",
            ["Naive","sNaive","OLS","Elastic","LightGBM","XGBoost","HoltWinters","Prophet"],
            default=["Naive","sNaive","OLS","Elastic","LightGBM","XGBoost","HoltWinters","Prophet"]
        )
    with colm3:
        st.markdown("**Scenario sliders (apply at forecast time):**")
        price_adj = st.slider("Price multiplier", 0.8, 1.2, 1.0, 0.01)
        promo_future = st.slider("Promo probability (future)", 0.0, 0.6, 0.0, 0.05)

    st.session_state.metric = metric
    st.session_state.horizon = int(horizon)
    st.session_state.folds = int(folds)
    st.session_state.include_models = include_models
    st.session_state.price_adj = float(price_adj)
    st.session_state.promo_future = float(promo_future)

    if st.button("Run backtest"):
        if st.session_state.raw_df is None:
            st.warning("Upload or generate data first.")
        else:
            feat, fcols = build_features(st.session_state.raw_df)
            errs, fold_tbl, best = backtest_models(
                feat, fcols, folds=folds, horizon=horizon, metric=metric, seasonality=52
            )
            st.session_state.bt = {"errors": errs, "folds": fold_tbl, "best": best, "feat": feat, "fcols": fcols}
            st.success(f"Backtest done. Best (avg): **{best}**" if best else "Backtest done.")
            st.dataframe(fold_tbl.fillna("").round(2), use_container_width=True)
            mean_row = pd.DataFrame([{"Model": k, "Mean Error": round(v,2)} for k,v in errs.items() if not np.isnan(v)]).sort_values("Mean Error")
            st.caption("Average backtest error per model:")
            st.dataframe(mean_row, use_container_width=True)
with tab_bpr:
    st.subheader("3) Train best / ensemble & forecast")
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
        # ------------------------------------------------------
        feat = st.session_state.bt["feat"]
        fcols = st.session_state.bt["fcols"]
        best = st.session_state.bt["best"] or "OLS"

        # Train each candidate on ALL data for final forecast
        hist = feat[["date", "sales"]].copy()
        last_date = hist["date"].max()

        # Scenario: adjust future drivers
        future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=int(horizon), freq="W")
        future_df = pd.DataFrame({"date": future_dates})
        # naive driver fill (carry last known)
        for c in ["price", "is_promo"]:
            if c in st.session_state.raw_df.columns:
                val = st.session_state.raw_df[c].dropna().iloc[-1]
            else:
                val = 0 if c == "is_promo" else 1.0
            future_df[c] = val

        # apply scenario knobs
        if "price" in future_df.columns:
            future_df["price"] = future_df["price"] * price_adj
        if "is_promo" in future_df.columns:
            rng = np.random.default_rng(0)
            future_df["is_promo"] = (rng.random(len(future_df)) < promo_future).astype(int)

        # Rebuild features on concatenated (hist+future placeholders)
        hist_raw = st.session_state.raw_df.copy()
        hist_raw = pd.concat([hist_raw, future_df], ignore_index=True)
        final_feat, final_fcols = build_features(hist_raw)

        # Select rows corresponding to future
        future_feat = final_feat[final_feat["date"].isin(future_dates)]
        train_feat  = final_feat[~final_feat["date"].isin(future_dates)]

        Xtr, ytr = train_feat[final_fcols], train_feat["sales"]
        Xf       = future_feat[final_fcols]

                # ---- SAFETY GUARD: if future or train features got dropped, switch to robust path ----
        if future_feat.empty or Xf.shape[0] == 0 or train_feat.empty or Xtr.shape[0] == 0:
            st.warning(
                "Some engineered feature rows were dropped (train or future), likely due to lags/rolls. "
                "Switching to a robust forecasting path."
            )
            preds = {}
            for ui_name in (include_models or ["Naive"]):
                model_name = MODEL_MAP.get(ui_name, ui_name)
                try:
                    df_fc = train_full_and_forecast(
                        df_hist=st.session_state.raw_df,
                        make_features_fn=build_features,
                        feature_cols=final_fcols,   # not used by Prophet; fine for others
                        model_name=model_name,
                        steps=int(horizon),
                        seasonality=52,
                    )
                    preds[ui_name] = df_fc
                except Exception as e:
                    st.info(f"{ui_name} could not produce a forecast in fallback path: {e}")

            # Build weights from backtest (map names so they match)
            errs = st.session_state.bt["errors"]
            name_map = {"Elastic": "ElasticNet", "HoltWinters": "HoltWintersSafe"}
            error_for_weight = {}
            for k in preds.keys():
                k_bt = name_map.get(k, k)
                err = errs.get(k_bt, np.nan)
                if err is None or not np.isfinite(err):
                    err = 999.0
                error_for_weight[k] = float(err)

            ens = weighted_ensemble(preds, error_for_weight)

            # Choose final forecast (same UI as before)
            choice = st.radio("Final forecast:", ["Best by backtest", "Weighted ensemble"], horizontal=True)

            final_fc = None
            chosen_name = None
            if choice == "Weighted ensemble" and ens is not None:
                final_fc = ens
                chosen_name = "Ensemble (1/error weights)"
            else:
                best_bt = st.session_state.bt["best"]
                fallback_order = ["LightGBM", "XGBoost", "Elastic", "OLS", "HoltWinters", "sNaive", "Naive", "Prophet"]
                choice_name = best_bt if best_bt in preds else next((m for m in fallback_order if m in preds), None)
                if choice_name is None:
                    st.error("No models produced predictions. Try enabling more models or reducing horizon.")
                else:
                    final_fc = preds[choice_name]
                    chosen_name = choice_name

            if final_fc is not None:
                st.session_state.final_fc = final_fc
                fig = plot_history_forecast(st.session_state.raw_df, final_fc, title=f"Chosen: {chosen_name}")
                st.pyplot(fig, use_container_width=True)

                # Stats footer
                st.subheader("📊 Stats")
                k = min(12, max(4, int(len(feat) * 0.15)))
                recent_actual = st.session_state.raw_df.set_index("date").tail(k)["sales"]
                recent_pred = final_fc.set_index("date").reindex(recent_actual.index)["forecast"].bfill().ffill()
                stats = score_table(recent_actual.values, recent_pred.values, seasonality=52)
                st.write(f"**Recent performance (vs last ~{len(recent_actual)} weeks)**")
                st.json(stats)
                    # Extra Data Insights
                # ===== Extra Data Insights & Strategy (enhanced) =====
                st.subheader("📈 Data Insights")
                df_hist = st.session_state.raw_df.copy().sort_values("date").reset_index(drop=True)
                sales = df_hist["sales"].astype(float).values
                dates = pd.to_datetime(df_hist["date"]).values

                expected_fc = float(np.nansum(final_fc["forecast"]))
                avg_fc = float(np.nanmean(final_fc["forecast"]))
                avg_sales = float(np.nanmean(sales))
                min_sales, max_sales = float(np.nanmin(sales)), float(np.nanmax(sales))
                std_sales = float(np.nanstd(sales))
                vol_ratio = std_sales / max(avg_sales, 1e-9)

                st.markdown(f"""
                - **Average Weekly Sales:** {avg_sales:,.0f}  
                - **Min / Max Weekly Sales:** {min_sales:,.0f} / {max_sales:,.0f}  
                - **Std Dev (Volatility):** {std_sales:,.0f} ({vol_ratio:.2f}× mean)  
                - **Forecast Horizon (Total Sales):** {expected_fc:,.0f}  
                - **Forecast Horizon (Avg per Week):** {avg_fc:,.0f}  
                """)

                # --- Signals ---
                lift_pct, price_elasticity, season_acf, trend_slope = 0.0, np.nan, np.nan, 0.0

                # Promo lift
                if "is_promo" in df_hist.columns and df_hist["is_promo"].sum() >= 5:
                    promo_sales = df_hist.loc[df_hist["is_promo"] == 1, "sales"].mean()
                    nonpromo_sales = df_hist.loc[df_hist["is_promo"] == 0, "sales"].mean()
                    if pd.notna(nonpromo_sales) and nonpromo_sales > 0:
                        lift_pct = float((promo_sales - nonpromo_sales) / nonpromo_sales * 100.0)
                        st.write(f"**Promo Lift Estimate:** {lift_pct:.1f}%")

                # Price elasticity (quick OLS on logs with seasonal sin/cos and promo)
                if "price" in df_hist.columns and df_hist["price"].notna().sum() > 20:
                    log_y = np.log(np.clip(df_hist["sales"].astype(float).values, 1e-6, None))
                    log_p = np.log(np.clip(df_hist["price"].astype(float).ffill().values, 1e-6, None))
                    # seasonal features
                    week = pd.to_datetime(df_hist["date"]).dt.isocalendar().week.astype(int).values
                    s_sin = np.sin(2*np.pi * week / 52.0)
                    s_cos = np.cos(2*np.pi * week / 52.0)
                    promo = df_hist["is_promo"].astype(float).values if "is_promo" in df_hist.columns else np.zeros_like(log_y)
                    X = np.column_stack([np.ones_like(log_y), log_p, promo, s_sin, s_cos])
                    try:
                        beta, *_ = np.linalg.lstsq(X, log_y, rcond=None)
                        # elasticity = d log(y) / d log(price) ≈ beta on log_p
                        price_elasticity = float(beta[1])
                        st.write(f"**Estimated Price Elasticity:** {price_elasticity:.2f} (more negative = more sensitive)")
                    except Exception:
                        pass

                # Seasonality strength via ACF at lag 52
                if len(sales) >= 104:  # need at least 2 years for a cleaner signal
                    y = pd.Series(sales - np.nanmean(sales)).fillna(0.0).values
                    lag = 52
                    if len(y) > lag + 1:
                        y1 = y[:-lag]
                        y2 = y[lag:]
                        if np.std(y1) > 0 and np.std(y2) > 0:
                            season_acf = float(np.corrcoef(y1, y2)[0,1])
                            st.write(f"**Seasonality (lag-52 ACF):** {season_acf:.2f}")

                # Trend slope (units per week)
                if len(sales) > 5:
                    t = np.arange(len(sales))
                    slope, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(t), t]), sales, rcond=None)
                    trend_slope = float(slope[1])
                    st.write(f"**Trend Slope:** {trend_slope:+.2f} per week")

                # Improvement vs Naive (from backtest)
                errs_local = st.session_state.bt["errors"]
                if "Naive" in errs_local:
                    best_err = errs_local.get(chosen_name, np.nan)
                    naive_err = errs_local["Naive"]
                    if np.isfinite(best_err) and np.isfinite(naive_err) and naive_err > 0:
                        imp = (naive_err - best_err)/naive_err * 100
                        st.write(f"**Improvement vs Naive:** {imp:.1f}% better")

                # --- AI Strategy: tailored multi-bullet plan ---
                st.subheader("🧭 Suggested Business Strategy")

                bullets = []

                # Promotions
                if lift_pct >= 15:
                    bullets.append("Increase promo frequency and concentrate in high-demand seasonal windows; expand promo depth cautiously to avoid eroding baseline.")
                elif lift_pct >= 5:
                    bullets.append("Use targeted promotions (e.g., around seasonal peaks) rather than always-on; A/B test promo depth and duration.")

                # Price
                if not np.isnan(price_elasticity):
                    if price_elasticity <= -0.8:
                        bullets.append("High price sensitivity: avoid list price hikes; focus on mix-shift and attach/accessories; explore temporary markdowns tied to events.")
                    elif price_elasticity <= -0.4:
                        bullets.append("Moderate price sensitivity: small price optimization can unlock revenue; test 1–3% price moves on low-elastic SKUs.")
                    else:
                        bullets.append("Low price sensitivity: consider modest list price increases, paired with loyalty value adds to preserve perceived fairness.")

                # Seasonality
                if not np.isnan(season_acf):
                    if season_acf >= 0.5:
                        bullets.append("Strong seasonality: front-load inventory and marketing ahead of seasonal peaks; align staffing and supply chain buffers.")
                    elif season_acf <= 0.2:
                        bullets.append("Weak seasonality: prioritize always-on demand programs (SEO/SEM/CRM) over seasonal bursts.")

                # Volatility / Trend
                if vol_ratio >= 0.6:
                    bullets.append("High volatility: implement weekly S&OP check-ins; use safety stock and flexible fulfillment to absorb demand shocks.")
                if trend_slope > 0:
                    bullets.append("Positive trend: reinvest in growth channels that correlate with recent lifts (promo/price levers as validated).")
                elif trend_slope < 0:
                    bullets.append("Negative trend: run root-cause analysis (assortment gaps, competition, out-of-stocks); test corrective promos on declining SKUs.")

                # Fallback if nothing fired
                if not bullets:
                    bullets.append("Maintain current mix; focus on incremental gains via targeted promos and continuous price testing.")

                for b in bullets:
                    st.markdown(f"- {b}")

                st.session_state.strategy = " ".join(bullets)

            
                    


                # Model comparison table (using backtest means you already computed)
                errs_all = st.session_state.bt["errors"]
                bt_tbl = pd.DataFrame(
                    [{"Model": m, "Mean Error": round(e, 2)} for m, e in errs_all.items() if np.isfinite(e)]
                ).sort_values("Mean Error")
                st.write("**Backtest average error by model**")
                st.dataframe(bt_tbl, use_container_width=True)

            # VERY IMPORTANT: stop the normal (feature-based) path after fallback finishes
            st.stop()
        # ---- end fallback guard ----



        # ========================
        # Fit models (properly indented under 'else')
        # ========================
        preds = {}
        if not include_models:
            include_models = ["Naive"]  # safe default

        # Naive
        naive_m = fit_naive(hist["sales"])
        naive_fc = predict_naive(naive_m, int(horizon))
        preds["Naive"] = pd.DataFrame({"date": future_dates, "forecast": naive_fc})

        # sNaive
        snaive_m = fit_snaive(hist["sales"], seasonality=52)
        if snaive_m is not None and "sNaive" in include_models:
            snaive_fc = predict_snaive(snaive_m, int(horizon))
            preds["sNaive"] = pd.DataFrame({"date": future_dates, "forecast": snaive_fc})

        # OLS
        if "OLS" in include_models:
            from sklearn.linear_model import LinearRegression
            m_ols = LinearRegression().fit(Xtr, ytr)
            preds["OLS"] = pd.DataFrame({"date": future_feat["date"], "forecast": m_ols.predict(Xf)})

        # Elastic
        if "Elastic" in include_models:
            from sklearn.linear_model import ElasticNet
            m_el = ElasticNet(alpha=0.0005, l1_ratio=0.1, max_iter=5000).fit(Xtr, ytr)
            preds["Elastic"] = pd.DataFrame({"date": future_feat["date"], "forecast": m_el.predict(Xf)})

        # LightGBM
        if "LightGBM" in include_models:
            from lightgbm import LGBMRegressor
            m_lgb = LGBMRegressor(
                random_state=0, n_estimators=500, learning_rate=0.05, num_leaves=31, min_data_in_leaf=5
            ).fit(Xtr, ytr)
            preds["LightGBM"] = pd.DataFrame({"date": future_feat["date"], "forecast": m_lgb.predict(Xf)})

        # XGBoost
        if "XGBoost" in include_models:
            try:
                from xgboost import XGBRegressor
                m_xgb = XGBRegressor(
                    n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9
                ).fit(Xtr, ytr)
                preds["XGBoost"] = pd.DataFrame({"date": future_feat["date"], "forecast": m_xgb.predict(Xf)})
            except Exception as e:
                st.info(f"XGBoost skipped: {e}")

        # Holt-Winters (safe)
        if "HoltWinters" in include_models:
            y_series = st.session_state.raw_df.sort_values("date")["sales"]
            holt_m = fit_holt(y_series, seasonality=52)
            holt_fc = predict_holt(holt_m, int(horizon))
            preds["HoltWinters"] = pd.DataFrame({"date": future_dates, "forecast": holt_fc})

        # Prophet (optional)
        if "Prophet" in include_models:
            try:
                from prophet import Prophet
                m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
                m.add_country_holidays(country_name="US")
                dfp = st.session_state.raw_df.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]
                m.fit(dfp)
                fut = pd.DataFrame({"ds": future_dates})
                pf = m.predict(fut)
                preds["Prophet"] = pd.DataFrame({"date": future_dates, "forecast": pf["yhat"].values})
            except Exception as e:
                st.info("Prophet could not run (it’s optional). Reason: {}".format(str(e)))


        # Build error dict from backtest means (map names so they match)
        errs = st.session_state.bt["errors"]
        name_map = {
            "Elastic": "ElasticNet",
            "HoltWinters": "HoltWintersSafe",
            # XGBoost isn't in backtest by default, so it will fall back to 999
            # "XGBoost": "XGBoost"  # left unmapped intentionally
        }
        error_for_weight = {}
        for k in preds.keys():
            k_bt = name_map.get(k, k)
            err = errs.get(k_bt, np.nan)
            if err is None or not np.isfinite(err):
                err = 999.0  # penalize models that weren't backtested
            error_for_weight[k] = float(err)


        ens = weighted_ensemble(preds, error_for_weight)

        # Pick best or ensemble
        choice = st.radio("Final forecast:", ["Best by backtest", "Weighted ensemble"], horizontal=True)

        final_fc = None
        chosen_name = None

        if choice == "Weighted ensemble" and ens is not None:
            final_fc = ens
            chosen_name = "Ensemble (1/error weights)"
        else:
            best_bt = st.session_state.bt["best"]
            fallback_order = ["LightGBM", "XGBoost", "Elastic", "OLS", "HoltWinters", "sNaive", "Naive", "Prophet"]
            choice_name = best_bt if best_bt in preds else next((m for m in fallback_order if m in preds), None)

            if choice_name is None:
                st.error("No models produced predictions. Try enabling more models or reducing horizon.")
            else:
                final_fc = preds[choice_name]
                chosen_name = choice_name

        # keep a copy for Export tab
        if final_fc is not None:
            st.session_state.final_fc = final_fc

        # Plot
        if final_fc is not None:
            fig = plot_history_forecast(st.session_state.raw_df, final_fc, title=f"Chosen: {chosen_name}")
            st.pyplot(fig, use_container_width=True)
            # ===== Extra Data Insights & Strategy =====
            st.subheader("📈 Data Insights")
            df_hist = st.session_state.raw_df.copy().sort_values("date")
            sales = df_hist["sales"].values
            expected_fc = final_fc["forecast"].sum()
            avg_fc = final_fc["forecast"].mean()
            st.markdown(f"""
            - **Average Weekly Sales:** {sales.mean():,.0f}  
            - **Min / Max Weekly Sales:** {sales.min():,.0f} / {sales.max():,.0f}  
            - **Std Dev (Volatility):** {sales.std():,.0f}  
            - **Forecast Horizon (Total Sales):** {expected_fc:,.0f}  
            - **Forecast Horizon (Avg per Week):** {avg_fc:,.0f}  
            """)

            lift, corr = 0.0, 0.0
            if "is_promo" in df_hist.columns and df_hist["is_promo"].sum() > 5:
                promo_sales = df_hist.loc[df_hist["is_promo"]==1,"sales"].mean()
                nonpromo_sales = df_hist.loc[df_hist["is_promo"]==0,"sales"].mean()
                lift = float(((promo_sales - nonpromo_sales)/max(nonpromo_sales, 1e-9))*100)
                st.write(f"**Promo Lift Estimate:** {lift:.1f}%")

            if "price" in df_hist.columns:
                corr = float(np.corrcoef(df_hist["price"].ffill(), df_hist["sales"])[0,1])
                st.write(f"**Price-Sales Correlation:** {corr:.2f} (negative = elastic)")

            errs_local = st.session_state.bt["errors"]
            if "Naive" in errs_local:
                # chosen_name already defined above when picking final model
                best_err = errs_local.get(chosen_name, np.nan)
                naive_err = errs_local["Naive"]
                if np.isfinite(best_err) and np.isfinite(naive_err) and naive_err > 0:
                    imp = (naive_err - best_err)/naive_err * 100
                    st.write(f"**Improvement vs Naive:** {imp:.1f}% better")

            st.subheader("🧭 Suggested Business Strategy")
            if lift > 10:
                strategy = "Promotions drive strong lifts — consider increasing promo frequency or targeting high-demand seasons."
            elif corr < -0.4:
                strategy = "Sales are price sensitive — avoid price hikes and explore price optimization."
            elif sales.std() > sales.mean()*0.5:
                strategy = "High volatility detected — advanced forecasting helps; stabilize supply and plan for swings."
            else:
                strategy = "Current patterns are stable — focus on maintaining consistency and gradual improvements."
            st.info(strategy)
            st.session_state.strategy = strategy

        # ===== Stats footer (comprehensive) =====
        st.subheader("📊 Stats")
        # Build a holdout to show recent performance (last ~15%, 4–12 weeks)
        k = min(12, max(4, int(len(feat) * 0.15)))
        recent_actual = st.session_state.raw_df.set_index("date").tail(k)["sales"]
        if final_fc is not None:
            recent_pred = final_fc.set_index("date").reindex(recent_actual.index)["forecast"].bfill().ffill()
            stats = score_table(recent_actual.values, recent_pred.values, seasonality=52)
            st.write(f"**Recent performance (vs last ~{len(recent_actual)} weeks)**")
            st.json(stats)

        # Model comparison table (using backtest means you already computed)
        # Model comparison table (using backtest means you already computed)
        errs_all = st.session_state.bt["errors"]
        bt_tbl = pd.DataFrame(
            [{"Model": m, "Mean Error": round(e, 2)} for m, e in errs_all.items() if np.isfinite(e)]
        ).sort_values("Mean Error")
        st.write("**Backtest average error by model**")
        st.dataframe(bt_tbl, use_container_width=True)


