"""Statistical inference: what drives the price? Log-linear OLS with robust (HC3) standard errors.
Each coefficient becomes 'percent change in price', with a 95% confidence interval.
Outputs: outputs/price_drivers.csv and outputs/region_effects.csv"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from . import config
from .preprocessing import load_data

def run():
    df = load_data()
    X = pd.get_dummies(df[config.NUMERIC + config.CATEGORICAL], drop_first=True).astype(float)
    res = sm.OLS(np.log(df[config.TARGET]), sm.add_constant(X)).fit(cov_type="HC3")      # robust to unequal error spread
    ci = res.conf_int()
    rows = []
    for term, b in res.params.items():
        if term == "const" or term.startswith("region_"):
            continue
        s = 100 if term == "area" else 1                               # area effect shown per +100 sq ft
        base = {"type": "Apartment", "status": "Ready to move", "age": "New"}.get(term.split("_")[0], "baseline")
        label = {"area": "+100 sq ft of area", "bhk": "+1 BHK (same area)"}.get(term, term.replace("_", " = ", 1) + f" (vs {base})")
        rows.append({"factor": label, "effect_pct": np.expm1(b * s) * 100, "ci_low_pct": np.expm1(ci.loc[term, 0] * s) * 100,
                     "ci_high_pct": np.expm1(ci.loc[term, 1] * s) * 100, "p_value": res.pvalues[term]})
    pd.DataFrame(rows).to_csv(config.OUT_DIR / "price_drivers.csv", index=False)
    # region effect relative to the average listing (the baseline region has coefficient 0)
    counts = df["region"].value_counts()
    coef = {r: res.params.get(f"region_{r}", 0.0) for r in counts.index}
    avg = sum(coef[r] * counts[r] for r in counts.index) / counts.sum()
    reg = pd.DataFrame({"region": counts.index, "effect_pct": [np.expm1(coef[r] - avg) * 100 for r in counts.index], "n": counts.values})
    reg.sort_values("effect_pct", ascending=False).to_csv(config.OUT_DIR / "region_effects.csv", index=False)
    print(f"R2 (log price) = {res.rsquared:.3f}, adjusted R2 = {res.rsquared_adj:.3f}")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nMost expensive areas vs average:", reg.sort_values("effect_pct").tail(3)["region"].tolist())

if __name__ == "__main__":
    run()
