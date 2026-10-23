"""Load the saved model and predict for new data.  Usage: python -m src.predict"""
import joblib
import pandas as pd
from . import config

def predict(records):
    model = joblib.load(config.MODEL_PATH)
    return model.predict(pd.DataFrame(records))

if __name__ == "__main__":
    sample = [{"area": 850, "bhk": 2, "type": "Apartment", "region": "Thane West", "status": "Ready to move", "age": "New"}]
    print(f"Predicted price: {predict(sample)[0]:,.1f} lakh rupees")
