"""Turn the raw Mumbai listings file into data/housing.csv (price in lakh rupees).
Run:  python -m src.prepare_data     (downloads the raw file if it is missing)
Source: public 'Mumbai House Prices' listings dataset (property-website asking prices)."""
import urllib.request
import pandas as pd
from . import config

RAW_URL = "https://raw.githubusercontent.com/Ranjit1401/Housecan/main/Mumbai%20House%20Prices.csv"
RAW_PATH = config.ROOT / "data" / "raw" / "Mumbai House Prices.csv"
MIN_REGION = 30          # regions with fewer listings are grouped as "Other"

def clean(raw):
    d = raw.drop_duplicates().copy()                      # the site repeats many listings
    d["price_lakh"] = d["price"] * d["price_unit"].map({"Cr": 100.0, "L": 1.0})   # 1 crore = 100 lakh
    d = d.dropna(subset=["price_lakh"])
    d = d[d["type"].isin(["Apartment", "Studio Apartment", "Villa", "Independent House"])]
    d = d[d["area"].between(200, 4000) & (d["bhk"] <= 6)]
    d = d[(d["price_lakh"] * 1e5 / d["area"]).between(2500, 70000)]   # drop typos / extreme rates per sq ft
    counts = d["region"].value_counts()
    d.loc[d["region"].map(counts) < MIN_REGION, "region"] = "Other"
    return d[["area", "bhk", "type", "region", "status", "age", "price_lakh"]].reset_index(drop=True)

def main():
    if not RAW_PATH.exists():
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAW_URL, RAW_PATH)
    raw = pd.read_csv(RAW_PATH)
    out = clean(raw)
    config.DATA_PATH.parent.mkdir(exist_ok=True)
    out.to_csv(config.DATA_PATH, index=False)
    print(f"raw rows {len(raw):,} -> cleaned rows {len(out):,}; saved {config.DATA_PATH}")

if __name__ == "__main__":
    main()
