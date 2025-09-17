# app.py — Weekly Sales Forecaster (upload or generate sample)
# Run: streamlit run app.py

import io
import math
import itertools
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

# LightGBM is optional; we fall back to OLS if features are too few
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

st.set_page_config(page_title="Weekly Sales Forecaster", page_icon="📈", layout="wide")

# ---------------------------
# UI Header
# ---------------------------
st.title("📈 Weekly Sales Forecaster")
st.caption("Upload your weekly sales CSV (date, sales, price, is_promo) or click **Generate Sample Data**.")

# ---------------------------
# Helpers
# ---------------------------
def mape(y_true, y_pred) -> float:
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs((yt - yp) / np.clip(np.abs(yt), 1e-9, None))) * 100.0)

def add_features(d: pd.DataFrame):
    d = d.copy()
    # Week-of-year cyclical features
    woy = d["date"].dt.isocalendar().week.astype(int)
    d["woy_sin"] = np.sin(2*np.pi*woy/52.0)
    d["woy_cos"] = np.cos(2*np.pi*woy/52.0)

    # Short lags + rolling means (kept small to work with short datasets)
    for L in [1, 2, 3, 4]:
        d[f"lag_{L}"] = d["sales"].shift(L)
    for W in [2, 3, 4]:
        d[f"roll_mean_{W}"] = d["sales"].shift(1).rolling(W, min_periods=2).mean()

    feature_cols = [c for c in d.columns if c not in ["date", "sales"]]
    d = d.dropna(subset=feature_cols).reset_index(drop=True)
    return d, feature_cols

def backtest(feat: pd.DataFrame, feature_cols: list, horizon=4, folds=3, use_lgbm=True):
    """Simple rolling-origin backtest: mean MAPE for OLS and (optionally) LightGBM."""
    n = len(feat)
    if n <= horizon + 4:
        return math.nan, math.nan

    step = max((n - horizon) // max(folds, 1), 1)
    ols_scores, lgb_scores = [], []

    for i in range(folds):
        end = min((i + 1) * step, n - horizon)
        if end <= 0 or (end + horizon) > n:
            continue
        train, val = feat.iloc[:end], feat.iloc[end:end + horizon]
        Xtr, ytr = train[feature_cols], train["sales"]
        Xv,  yv  = val[feature_cols],   val["sales"]

        ols = LinearRegression().fit(Xtr, ytr)
        ols_scores.append(mape(yv, ols.predict(Xv)))

        if use_lgbm and HAS_LGBM:
            lgb = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                random_state=0,
                min_data_in_leaf=1,
                verbose=-1
            )
            lgb.fit(Xtr, ytr)
            lgb_scores.append(mape(yv, lgb.predict(Xv)))

    mean_ols = float(np.mean(ols_scores)) if len(ols_scores) else math.nan
    mean_lgb = float(np.mean(lgb_scores)) if (use_lgbm and len(lgb_scores)) else math.nan
    return mean_ols, mean_lgb

def train_best(feat, feature_cols, ols_mape, lgb_mape):
    """Pick OLS if it wins or LightGBM otherwise; handle missing LightGBM."""
    use_ols = (math.isnan(lgb_mape) or (not math.isnan(ols_mape) and ols_mape <= lgb_mape) or not HAS_LGBM)
    if use_ols:
        model = LinearRegression().fit(feat[feature_cols], feat["sales"])
        model_name = "OLS Regression"
    else:
        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=0,
            min_data_in_leaf=1,
            verbose=-1
        ).fit(feat[feature_cols], feat["sales"])
        model_name = "LightGBM"
    return model, model_name

def recursive_forecast(df_hist: pd.DataFrame, steps: int, base_price=None, promo_plan=None, model=None):
    """Forecast next N weeks; optional what-ifs for price/promo."""
    df_temp = df_hist.copy()
    preds = []
    last_date = df_temp["date"].max()
    for i in range(steps):
        next_date = last_date + timedelta(days=7)
        price = base_price if base_price is not None else df_temp["price"].iloc[-1]
        is_promo = 0
        if promo_plan == "every_4th_week" and ((i + 1) % 4 == 0):
            is_promo = 1
        new_row = {"date": next_date, "sales": np.nan, "price": price, "is_promo": is_promo}
        df_temp = pd.concat([df_temp, pd.DataFrame([new_row])], ignore_index=True)
        feat_tmp, fcols = add_features(df_temp)
        row = feat_tmp.iloc[[-1]][fcols]
        yhat = float(model.predict(row)[0])
        df_temp.loc[df_temp["date"] == next_date, "sales"] = yhat
        preds.append({"date": next_date, "forecast": yhat})
        last_date = next_date
    return pd.DataFrame(preds)

def generate_synthetic(weeks=80, seed=42):
    """Create a realistic weekly dataset with seasonality, promos, price, noise."""
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=weeks, freq="W")

    seasonality = 200 * np.sin(np.linspace(0, 3*np.pi, weeks)) + 1000
    is_promo = np.random.choice([0, 1], size=weeks, p=[0.8, 0.2])
    promo_boost = is_promo * np.random.randint(100, 300, size=weeks)
    price = 50 + np.random.choice([-2, 0, 2], size=weeks)
    price_effect = -10 * (price - 50)
    noise = np.random.normal(0, 50, size=weeks)

    sales = (seasonality + promo_boost + price_effect + noise).clip(200, None).round(0)

    return pd.DataFrame({
        "date": dates,
        "sales": sales,
        "price": price,
        "is_promo": is_promo
    })

def format_pct(x):
    return "n/a" if (x is None or isinstance(x, float) and math.isnan(x)) else f"{x:.2f}%"

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Controls")
    horizon = st.selectbox("Forecast horizon (weeks)", options=[4, 8, 12, 16, 24], index=2)
    synth_weeks = st.slider("Sample data length (weeks)", min_value=40, max_value=156, value=80, step=4)
    promo_plan = st.selectbox("Promo plan for forecast (optional)", ["none", "every_4th_week"], index=0)
    base_price = st.number_input("Price for forecast (leave as is to auto-use last)", value=0.0, help="Enter 0 to auto-use the last observed price.")

# ---------------------------
# Data input area
# ---------------------------
col_upload, col_or, col_gen = st.columns([3, 1, 3])

with col_upload:
    uploaded = st.file_uploader("Upload CSV with columns: date, sales, price, is_promo", type=["csv"])

with col_or:
    st.write("")
    st.write("**OR**")

with col_gen:
    gen_clicked = st.button("🧪 Generate Sample Data")

# Load data
df = None
if gen_clicked:
    df = generate_synthetic(weeks=synth_weeks)
elif uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=["date"])
    # minimal validation / coerce types
    expected = {"date", "sales", "price", "is_promo"}
    if not expected.issubset(df.columns):
        st.error(f"Your CSV must include columns: {expected}. Found: {set(df.columns)}")
        st.stop()
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["is_promo"] = pd.to_numeric(df["is_promo"], errors="coerce").fillna(0)
    df = df.sort_values("date").reset_index(drop=True)

if df is not None:
    st.subheader("Dataset preview")
    st.dataframe(df.tail(15), use_container_width=True)

    # Build features
    feat, feature_cols = add_features(df)

    if len(feat) < 12 or len(feature_cols) < 4:
        st.warning(
            "Small effective training set after feature creation. "
            "Consider more weeks of data for more reliable results."
        )

    # Run backtest + train
    with st.spinner("Running backtest and training model..."):
        ols_mape, lgb_mape = backtest(feat, feature_cols, horizon=4, folds=3, use_lgbm=HAS_LGBM)
        model, model_name = train_best(feat, feature_cols, ols_mape, lgb_mape)

    # Stats box
    st.subheader("Stats (lower MAPE = better)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("OLS MAPE", format_pct(ols_mape))
    with col2:
        st.metric("LightGBM MAPE", "n/a" if (not HAS_LGBM or math.isnan(lgb_mape)) else f"{lgb_mape:.2f}%")
    with col3:
        if HAS_LGBM and not math.isnan(ols_mape) and not math.isnan(lgb_mape):
            imp = (ols_mape - lgb_mape) / ols_mape * 100.0
            st.metric("Improvement vs OLS", f"{imp:.1f}%")
        else:
            st.metric("Improvement vs OLS", "n/a")
    with col4:
        st.metric("Chosen model", model_name)

    # Forecast
    st.subheader(f"Forecast ({horizon} weeks)")
    price_for_forecast = None if base_price == 0 else base_price
    fcst_df = recursive_forecast(
        df_hist=df,
        steps=horizon,
        base_price=price_for_forecast,
        promo_plan=None if promo_plan == "none" else promo_plan,
        model=model
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], df["sales"], label="Actual")
    ax.plot(fcst_df["date"], fcst_df["forecast"], label="Forecast")
    ax.set_title("Weekly Sales: History + Forecast")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    # Show table + download
    st.dataframe(fcst_df, use_container_width=True)
    csv_bytes = fcst_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download forecast CSV",
        data=csv_bytes,
        file_name="forecast.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV *or* click **Generate Sample Data** to begin.")
