"""Create a synthetic house-price dataset (so the project runs without internet).
Replace data/housing.csv with your own CSV to use real data (keep a 'price' column or
change TARGET in config.py)."""
import numpy as np
import pandas as pd
from pathlib import Path

def make_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    area = rng.normal(1500, 400, n).clip(500, 4000)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n), 1, 5)
    age = rng.integers(0, 50, n)
    distance_city = rng.uniform(1, 30, n)
    city = rng.choice(["Mumbai", "Pune", "Nagpur"], n, p=[0.4, 0.35, 0.25])
    city_bonus = pd.Series(city).map({"Mumbai": 60000, "Pune": 25000, "Nagpur": 0}).values
    price = (50 * area + 8000 * bedrooms + 6000 * bathrooms - 900 * age
             - 1500 * distance_city + city_bonus + rng.normal(0, 15000, n))
    df = pd.DataFrame(dict(area=area.round(1), bedrooms=bedrooms, bathrooms=bathrooms,
                           age=age, distance_city=distance_city.round(2),
                           city=city, price=price.round(0)))
    # inject a few missing values to practise cleaning
    for col in ["area", "age", "city"]:
        df.loc[rng.choice(n, 15, replace=False), col] = np.nan
    return df

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "housing.csv"
    make_data().to_csv(out, index=False)
    print("Saved", out)
