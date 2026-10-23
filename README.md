# Linear Regression Project – House Price Prediction (real Mumbai listings)

**Data:** about 76,000 real property listings from the Mumbai region (asking prices from a property website, public
'Mumbai House Prices' dataset). `python -m src.prepare_data` downloads and cleans it into `data/housing.csv`
(price in lakh rupees; 55,295 listings after cleaning). The target is `price_lakh`, modelled on a log scale.

**Results (test set, 10,360 unseen listings):** best model Polynomial(2) + Ridge, R² about 0.85, average error about 19%.
Features: area, bhk, type, region, status (ready / under construction), age (new / resale).
Live app: `streamlit run app.py`.

**Advanced features:** log-price modelling, five linear models + a boosting benchmark, learning curve, error by group,
robust (HC3) inference with a plain-English 'what drives the price' table, a per-prediction explanation
(`src/explain.py`), prediction ranges from real test errors, photo-based room sizes, address-to-area matching,
USD/INR conversion, and a full report in `docs/PROJECT_REPORT.md`.

A complete, end-to-end linear regression project: data → EDA → preprocessing → training →
model comparison → evaluation → assumption checks → saved model → prediction.

## 1. Setup
```bash
pip install -r requirements.txt
python run_all.py          # runs everything
```
Individual steps:
```bash
python -m src.prepare_data         # downloads + cleans the real Mumbai data -> data/housing.csv
python -m src.eda                  # plots -> outputs/
python -m src.train                # trains/compares models, saves models/best_model.joblib
python -m src.statistical_analysis # p-values, VIF, Breusch-Pagan, Durbin-Watson
python -m src.predict              # predict for a new house
pytest                             # unit test for the from-scratch model
```

## 2. Folder structure
```
linear_regression_project/
├── data/housing.csv            dataset (swap in your own)
├── src/
│   ├── config.py               paths, column names, target, split settings
│   ├── prepare_data.py         download + clean the real listings
│   ├── generate_data.py        (old) synthetic data generator, no longer used
│   ├── preprocessing.py        impute, scale, one-hot, polynomial, outlier removal
│   ├── eda.py                  summary stats, correlation heatmap, histograms
│   ├── from_scratch.py         NumPy linear regression (Normal Eq. + Gradient Descent)
│   ├── train.py                Linear/Ridge/Lasso/ElasticNet/Poly, GridSearchCV, 5-fold CV
│   ├── evaluate.py             MAE, MSE, RMSE, R², Adj-R², diagnostic plots
│   ├── statistical_analysis.py statsmodels OLS summary + assumption tests
│   └── predict.py              load model & predict
├── models/best_model.joblib    saved sklearn Pipeline
├── outputs/                    plots, model_comparison.csv, best_metrics.json
├── tests/test_model.py
├── run_all.py
└── requirements.txt
```

## 3. Using your own dataset
1. Put your CSV in `data/housing.csv` (or change `DATA_PATH`).
2. Edit `src/config.py`: `TARGET`, `NUMERIC`, `CATEGORICAL`.
3. Run `python run_all.py`.

## 4. Steps of a linear regression task (what to write in your report)
1. **Problem definition** – predict a continuous target (here: price).
2. **Data collection** – CSV, database, API, Kaggle.
3. **EDA** – distributions, missing values, correlations, outliers.
4. **Cleaning** – impute missing values, remove duplicates/extreme outliers.
5. **Feature engineering** – encode categoricals, scale numerics, add polynomial/interaction terms, log-transform skewed targets.
6. **Split** – train/test (80/20), plus K-fold cross-validation.
7. **Train** – OLS, then regularised variants.
8. **Tune** – GridSearchCV for `alpha` / `l1_ratio`.
9. **Evaluate** – MAE, RMSE, R², Adjusted R² on the test set.
10. **Check assumptions** – linearity, independence, homoscedasticity, normal residuals, no multicollinearity.
11. **Interpret** – coefficients = change in target per unit change in feature (others fixed).
12. **Deploy** – save with joblib, predict on new data.

## 5. Maths in one place
- Model: ŷ = w₀ + w₁x₁ + … + wₙxₙ
- Loss (MSE): J = (1/n) Σ (yᵢ − ŷᵢ)²
- Normal equation: w = (XᵀX)⁻¹ Xᵀ y
- Gradient descent: w ← w − α · (2/n) Xᵀ(Xw − y)
- Ridge adds λ‖w‖² (shrinks), Lasso adds λ‖w‖₁ (zeroes out weak features).

## 6. Features you can integrate (extensions)
| Area | Ideas |
|---|---|
| Data | Load from SQL/API, real datasets (California Housing, Boston-style, Kaggle) |
| Preprocessing | KNN/iterative imputation, RobustScaler, log/Box-Cox transforms, binning |
| Feature work | Polynomial & interaction terms, feature selection (RFE, SelectKBest, Lasso), PCA |
| Models | Ridge, Lasso, ElasticNet, Huber (outlier-robust), SGDRegressor, compare with Random Forest / XGBoost |
| Validation | K-fold, repeated CV, learning curves, validation curves |
| Explainability | Standardised coefficients, permutation importance, SHAP |
| Diagnostics | Residual plots, Q-Q plot, VIF, Cook's distance, Breusch-Pagan |
| Deployment | Streamlit/Flask/FastAPI app, Docker, MLflow experiment tracking |
| Quality | Unit tests, logging, config file, CI (GitHub Actions) |

## 7. Output interpretation tips
- R² near 1 = good fit; always compare train vs test (a big gap = overfitting).
- RMSE is in the same unit as the target – easiest to explain.
- Residuals should look like random noise around 0.
- VIF > 10 → multicollinearity (on this data VIF is low, area and bhk are about 3).
