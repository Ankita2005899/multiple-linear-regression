"""One command to run the whole pipeline:  python run_all.py"""
from src import prepare_data, eda, train, statistical_analysis, inference, predict, config

if not config.DATA_PATH.exists():
    prepare_data.main()          # downloads and cleans the real Mumbai listings
print("== EDA =="); eda.run()
print("\n== TRAINING =="); train.main()
print("\n== STATISTICS =="); statistical_analysis.run()
print("\n== PRICE DRIVERS (robust inference) =="); inference.run()
print("\n== PREDICTION DEMO ==")
print(predict.predict([{"area": 850, "bhk": 2, "type": "Apartment", "region": "Thane West", "status": "Ready to move", "age": "New"}]), "lakh rupees")
