import numpy as np
from src.from_scratch import LinearRegressionScratch

def test_scratch_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2)); y = 3 + 2 * X[:, 0] - 1 * X[:, 1]
    m = LinearRegressionScratch(method="normal").fit(X, y)
    assert abs(m.intercept_ - 3) < 1e-6 and np.allclose(m.coef_, [2, -1], atol=1e-6)

def test_clean_converts_crore_to_lakh():
    import pandas as pd
    from src.prepare_data import clean
    raw = pd.DataFrame(dict(bhk=[2], type=["Apartment"], locality=["x"], area=[1000], price=[1.5], price_unit=["Cr"],
                            region=["Thane West"], status=["Ready to move"], age=["New"]))
    out = clean(raw)
    assert out.loc[0, "price_lakh"] == 150.0

def test_explanation_adds_up_to_the_model_prediction():
    import pandas as pd, joblib, pytest
    from src import config, explain
    if not config.MODEL_PATH.exists():
        pytest.skip("model not trained yet")
    row = {"area": 900, "bhk": 2, "type": "Apartment", "region": "Thane West", "status": "Ready to move", "age": "New"}
    ex = explain.explain(row)
    model_price = float(joblib.load(config.MODEL_PATH).predict(pd.DataFrame([row]))[0])
    assert abs(ex["price_lakh"] - model_price) < 1e-6

def test_prices_are_positive_and_grow_with_area():
    import pandas as pd, joblib, pytest
    from src import config
    if not config.MODEL_PATH.exists():
        pytest.skip("model not trained yet")
    m = joblib.load(config.MODEL_PATH)
    base = {"bhk": 2, "type": "Apartment", "region": "Thane West", "status": "Ready to move", "age": "New"}
    small, big = (float(m.predict(pd.DataFrame([dict(base, area=a)]))[0]) for a in (600, 1200))
    assert 0 < small < big
