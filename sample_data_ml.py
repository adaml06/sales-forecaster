# sample_data_ml.py
import numpy as np
import pandas as pd

def gen_weekly_ml(
    n_weeks: int = 208,          # ~4 years so ML has enough history
    level: float = 900.0,
    trend_start: float = 0.8,    # baseline trend slope (units / week)
    season_amp: float = 60.0,    # smaller than your old 120 — sNaive still helps but won't dominate
    price_base: float = 10.0,
    price_vol: float = 0.07,     # price variation amplitude (↑ gives ML more to learn)
    promo_rate: float = 0.18,    # % of weeks with promos
    promo_lift: float = 0.35,    # average promo lift (interacts with price below)
    noise_level: float = 0.12,   # multiplicative noise; ML can smooth this out better than baselines
    seed: int = 0,
):
    """
    Returns a DataFrame with columns: date, sales, price, is_promo

    Design goals (ML-friendly but fair):
    - Seasonality present but NOT overwhelming (season_amp modest).
    - Price follows mean-reverting wiggles; sales include non-linear price elasticity.
    - Promo lift depends on both season and price (interaction) so tree/regularized models win.
    - Piecewise trend & occasional shocks so sNaive/Holt help but won't always be best.
    """

    rng = np.random.default_rng(seed)

    # --- calendar / index ---
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_weeks, freq="W")
    t = np.arange(n_weeks)

    # --- piecewise trend: slope changes mid-way (regime) ---
    slope = np.full(n_weeks, trend_start, float)
    cut1 = int(n_weeks * 0.35)
    cut2 = int(n_weeks * 0.70)
    slope[cut1:cut2] += 0.5         # growth speeds up
    slope[cut2:] -= 0.7             # then cools down
    trend = level + np.cumsum(slope)

    # --- seasonality (52w) kept moderate ---
    season = season_amp * np.sin(2 * np.pi * (t % 52) / 52.0)
    # a small second seasonal (26w) to add complexity
    season2 = (season_amp * 0.35) * np.sin(2 * np.pi * (t % 26) / 26.0)

    # --- price: mean-reverting wiggle around price_base ---
    # AR(1) with shrinkage toward 0, scaled, then centered on price_base
    z = np.zeros(n_weeks)
    for i in range(1, n_weeks):
        z[i] = 0.6 * z[i-1] + rng.normal(0, price_vol)
    price_series = price_base * (1.0 + z)
    price_series = np.clip(price_series, 0.6 * price_base, 1.6 * price_base)

    # --- promos: a bit seasonal (more likely in late Q4 / back-to-school) ---
    base_promo = np.full(n_weeks, promo_rate)
    # lift promo probability around weeks 32-36 and 47-52 (BTS + holiday)
    base_promo[(t % 52 >= 32) & (t % 52 <= 36)] += 0.15
    base_promo[(t % 52 >= 47)] += 0.20
    base_promo = np.clip(base_promo, 0.01, 0.95)
    is_promo = (rng.random(n_weeks) < base_promo).astype(int)

    # --- external shocks (rare) ---
    shocks = np.zeros(n_weeks)
    if n_weeks >= 52:
        shock_weeks = rng.choice(np.arange(10, n_weeks - 10), size=max(1, n_weeks // 104), replace=False)
        for sw in shock_weeks:
            shocks[sw: sw + 2] += rng.normal(120, 40)  # brief positive demand bump

    # --- latent base demand (before price/promo effects) ---
    latent = trend + season + season2 + shocks

    # --- non-linear price elasticity (elasticity depends on price level and promo) ---
    # elasticity becomes milder during promo and in high-demand seasonal peaks
    rel_price = price_series / price_base
    base_elasticity = -1.05 + 0.25 * is_promo + 0.20 * (season > 0).astype(float)
    price_effect = rel_price ** base_elasticity

    # --- promo effect with interaction to season & price ---
    promo_effect = 1.0 + is_promo * (promo_lift * (1.0 + 0.4 * (season > 30)) * (1.0 + 0.2 * (rel_price < 0.95)))

    # --- compose demand and add multiplicative noise ---
    sales = latent * price_effect * promo_effect
    sales = sales * (1.0 + noise_level * rng.standard_normal(n_weeks))

    # clamp & smooth occasional negatives
    sales = np.clip(sales, 0, None)

    df = pd.DataFrame({
        "date": dates,
        "sales": sales.round(2),
        "price": price_series.round(2),
        "is_promo": is_promo,
    })
    return df


def gen_weekly_profile(profile: str = "ml_friendly", seed: int = 0) -> pd.DataFrame:
    """
    Convenience wrapper with presets. All return the same schema: date, sales, price, is_promo.

    Realistic / educational presets:
      - 'ml_friendly'   : strong drivers & interactions; good demo for ML
      - 'balanced'      : moderate drivers; classical methods still competitive
      - 'baseliney'     : larger seasonality; sNaive will look strong
      - 'steady_growth' : gentle growth + mild seasonality
      - 'holiday_spike' : big Q4 seasonal amplitude, promo-friendly
      - 'promo_driven'  : frequent promos with large lift
      - 'price_sensitive': higher price volatility & noise
      - 'post_launch_decline': high level then declining trend
      - 'volatile_market': high noise & price wiggle
      - 'flatline'      : nearly no signal (robustness check)
      - 'spiky_outliers': inject a few extreme weeks
      - 'season_switch' : seasonality amplitude changes mid-series
      - 'short_horizon' : ~1 year of history (cold-start)
      - 'covid_drop'    : mid-window demand dip then recovery
    """
    p = (profile or "ml_friendly").lower()
    rng = np.random.default_rng(seed)

    # --- simple presets that are direct calls -------------------------------
    if p == "ml_friendly":
        return gen_weekly_ml(n_weeks=260, season_amp=55, price_vol=0.09,
                             promo_rate=0.22, promo_lift=0.45, noise_level=0.12, seed=seed)
    if p == "balanced":
        return gen_weekly_ml(n_weeks=208, season_amp=70, price_vol=0.06,
                             promo_rate=0.18, promo_lift=0.33, noise_level=0.10, seed=seed)
    if p == "baseliney":
        return gen_weekly_ml(n_weeks=156, season_amp=100, price_vol=0.04,
                             promo_rate=0.15, promo_lift=0.25, noise_level=0.08, seed=seed)
    if p == "steady_growth":
        return gen_weekly_ml(n_weeks=260, trend_start=0.5, season_amp=40,
                             promo_rate=0.15, promo_lift=0.30, noise_level=0.10, seed=seed)
    if p == "holiday_spike":
        return gen_weekly_ml(n_weeks=260, trend_start=0.2, season_amp=200,
                             promo_rate=0.25, promo_lift=0.50, noise_level=0.12, seed=seed)
    if p == "promo_driven":
        return gen_weekly_ml(n_weeks=208, trend_start=0.0, season_amp=30,
                             promo_rate=0.50, promo_lift=0.80, noise_level=0.10, seed=seed)
    if p == "price_sensitive":
        return gen_weekly_ml(n_weeks=208, season_amp=50, price_vol=0.20,
                             promo_rate=0.18, promo_lift=0.30, noise_level=0.14, seed=seed)
    if p == "post_launch_decline":
        return gen_weekly_ml(n_weeks=208, level=1200.0, trend_start=-0.6, season_amp=30,
                             promo_rate=0.12, promo_lift=0.25, noise_level=0.10, seed=seed)
    if p == "volatile_market":
        return gen_weekly_ml(n_weeks=260, season_amp=60, price_vol=0.25,
                             promo_rate=0.30, promo_lift=0.40, noise_level=0.30, seed=seed)
    if p == "flatline":
        return gen_weekly_ml(n_weeks=156, trend_start=0.0, season_amp=0.0,
                             promo_rate=0.0, promo_lift=0.0, noise_level=0.05, seed=seed)
    if p == "short_horizon":
        return gen_weekly_ml(n_weeks=52, season_amp=55, price_vol=0.08,
                             promo_rate=0.20, promo_lift=0.40, noise_level=0.12, seed=seed)

    # --- presets with light post-processing --------------------------------
    if p == "spiky_outliers":
        df = gen_weekly_ml(n_weeks=208, season_amp=55, price_vol=0.08,
                           promo_rate=0.18, promo_lift=0.35, noise_level=0.12, seed=seed)
        k = rng.integers(3, 6)
        idx = rng.choice(len(df), size=k, replace=False)
        mult = rng.uniform(0.5, 2.0, size=k)
        df.loc[idx, "sales"] = (df.loc[idx, "sales"].to_numpy() * mult).round(2)
        return df

    if p == "season_switch":
        # Start with a normal series, then boost seasonality amplitude for the last ~40%
        df = gen_weekly_ml(n_weeks=208, season_amp=45, price_vol=0.07,
                           promo_rate=0.18, promo_lift=0.35, noise_level=0.12, seed=seed)
        n = len(df)
        cut = int(n * 0.60)
        # multiplicative bump tied to week-of-year sine; mimics stronger seasonality
        t = np.arange(n - cut)
        bump = 1.0 + 0.25 * np.sin(2 * np.pi * (t % 52) / 52.0)
        df.loc[cut:, "sales"] = (df.loc[cut:, "sales"].to_numpy() * bump).round(2)
        return df

    if p == "covid_drop":
        df = gen_weekly_ml(n_weeks=208, season_amp=55, price_vol=0.08,
                           promo_rate=0.18, promo_lift=0.35, noise_level=0.12, seed=seed)
        # drop between week 80 and 100, recover linearly by week 120
        a, b, c = 80, 100, 120
        drop = 0.50
        for i in range(a, b):
            df.loc[i, "sales"] = (df.loc[i, "sales"] * drop).round(2)
        # linear recovery
        for i in range(b, c):
            alpha = (i - b) / max(1, c - b)
            factor = drop + (1 - drop) * alpha
            df.loc[i, "sales"] = (df.loc[i, "sales"] * factor).round(2)
        return df

    # default
    return gen_weekly_ml(seed=seed)
