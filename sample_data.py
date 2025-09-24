# sample_data.py
import numpy as np
import pandas as pd

def gen_weekly(n_weeks=104, level=800, trend=0.5, season_amp=120, promo_lift=0.25, price=10.0, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_weeks, freq="W")
    t = np.arange(n_weeks)
    season = season_amp * np.sin(2*np.pi*(t%52)/52)
    promos = (rng.random(n_weeks) < 0.15).astype(int)
    price_series = price * (1 + 0.05*np.sin(2*np.pi*(t%26)/26))
    base = level + trend*t + season
    sales = base * (1 + promo_lift*promos) * (price_series/price)**(-0.7)
    sales = sales * (1 + noise*rng.standard_normal(n_weeks))
    sales = np.clip(sales, 0, None)
    df = pd.DataFrame({"date": dates, "sales": sales.round(2), "price": price_series.round(2), "is_promo": promos})
    return df
