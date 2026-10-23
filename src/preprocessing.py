"""Reusable preprocessing: imputation, scaling, one-hot encoding, optional polynomial features."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from . import config

def load_data(path=config.DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=[config.TARGET]).drop_duplicates()
    return df

def remove_outliers_iqr(df, cols, k=3.0):
    """Drop rows far outside the IQR fence (k=3 -> only extreme outliers)."""
    mask = pd.Series(True, index=df.index)
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask &= df[c].between(q1 - k * iqr, q3 + k * iqr) | df[c].isna()
    return df[mask]

def build_preprocessor(poly_degree=1):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if poly_degree > 1:
        num_steps.append(("poly", PolynomialFeatures(poly_degree, include_bias=False)))
    num_steps.append(("scale", StandardScaler()))
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))])
    return ColumnTransformer([("num", Pipeline(num_steps), config.NUMERIC),
                              ("cat", cat, config.CATEGORICAL)])
