"""Explain ONE prediction: how much does each group of features push the price above or below
the average listing? Works because the model is linear on log price (effect = coefficient x feature)."""
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from . import config
from .preprocessing import load_data

GROUPS = {"num__": "Size (area and BHK)", "cat__type_": "Property type", "cat__region_": "Area",
          "cat__status_": "Ready / under construction", "cat__age_": "New / resale"}

@lru_cache(maxsize=1)
def _setup():
    inner = joblib.load(config.MODEL_PATH).regressor_
    prep, lin = inner.named_steps["prep"], inner.named_steps["model"]
    T = prep.transform(load_data()[config.NUMERIC + config.CATEGORICAL])
    return prep, lin, np.asarray(T.mean(axis=0)).ravel(), list(prep.get_feature_names_out())

def explain(row):
    """row: dict with area, bhk, type, region, status, age -> baseline price and % effect per group (lakh rupees)."""
    prep, lin, mean, names = _setup()
    x = prep.transform(pd.DataFrame([row]))
    x = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.ravel(x)
    contrib = lin.coef_ * (x - mean)
    out = {g: 0.0 for g in GROUPS.values()}
    for n, c in zip(names, contrib):
        for prefix, g in GROUPS.items():
            if n.startswith(prefix):
                out[g] += c
    base = lin.intercept_ + lin.coef_ @ mean
    return {"baseline_lakh": float(np.expm1(base)), "price_lakh": float(np.expm1(base + sum(out.values()))),
            "effects_pct": {g: float(np.expm1(v) * 100) for g, v in out.items()}}
