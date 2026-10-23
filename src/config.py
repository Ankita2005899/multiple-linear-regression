from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "housing.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
OUT_DIR = ROOT / "outputs"

TARGET = "price_lakh"   # price in lakh rupees (1 lakh = 100,000)
NUMERIC = ["area", "bhk"]
CATEGORICAL = ["type", "region", "status", "age"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
