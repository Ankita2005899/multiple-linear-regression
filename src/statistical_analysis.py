"""Statistical view of regression (p-values, confidence intervals, VIF, assumption tests)
using statsmodels. Fitted on log(price) with the main features; the ~150 region columns are
left out here to keep the table readable (the main model in train.py does use region)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from . import config
from .preprocessing import load_data

def run():
    df = load_data().dropna()
    cols = config.NUMERIC + [c for c in config.CATEGORICAL if c != "region"]
    X = pd.get_dummies(df[cols], drop_first=True).astype(float)
    y = np.log(df[config.TARGET])
    Xc = sm.add_constant(X); model = sm.OLS(y, Xc).fit()
    print(model.summary())
    print("\nVIF (multicollinearity, >10 is a problem):")
    for i, c in enumerate(Xc.columns[1:], 1): print(f"  {c:32s}{variance_inflation_factor(Xc.values, i):.2f}")
    bp = het_breuschpagan(model.resid, Xc)
    print(f"\nBreusch-Pagan p-value (homoscedasticity, want > 0.05): {bp[1]:.4f}")
    print(f"Durbin-Watson (independence, ~2 is good): {durbin_watson(model.resid):.3f}")

if __name__ == "__main__":
    run()
