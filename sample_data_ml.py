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
    Convenience wrapper with presets:
      - 'ml_friendly': stronger driver signal & interactions => ML shines
      - 'balanced':   moderate driver signal; HW/sNaive still competitive
      - 'baseliney':  closer to classic seasonal toy; sNaive often strong
    """
    profile = (profile or "ml_friendly").lower()
    if profile == "ml_friendly":
        return gen_weekly_ml(
            n_weeks=260, season_amp=55, price_vol=0.09,
            promo_rate=0.22, promo_lift=0.45, noise_level=0.12, seed=seed
        )
    if profile == "balanced":
        return gen_weekly_ml(
            n_weeks=208, season_amp=70, price_vol=0.06,
            promo_rate=0.18, promo_lift=0.33, noise_level=0.10, seed=seed
        )
    if profile == "baseliney":
        return gen_weekly_ml(
            n_weeks=156, season_amp=100, price_vol=0.04,
            promo_rate=0.15, promo_lift=0.25, noise_level=0.08, seed=seed
        )
    # default
    return gen_weekly_ml(seed=seed)
