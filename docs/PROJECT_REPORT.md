# Multiple Linear Regression: Mumbai House Price Prediction

## 1. Problem
Predict the asking price of a house in the Mumbai region from its size, BHK, type, area (region), construction status and age
(new or resale), and explain which details drive the price.

## 2. Data
- Source: a public 'Mumbai House Prices' listings dataset (asking prices from a property website), 76,038 rows.
- Cleaning (`src/prepare_data.py`): repeated listings removed; price converted to lakh rupees (1 crore = 100 lakh);
  only apartments, studios, villas and independent houses kept; area 200 to 4,000 sq ft; BHK up to 6;
  price per sq ft between 2,500 and 70,000 rupees; regions with fewer than 30 listings grouped as "Other".
- Result: 55,295 listings (51,800 after removing rows identical in every kept column). Split: 41,440 training, 10,360 test.

## 3. Method
- Target: log(price), because prices are strongly skewed. Predictions are converted back to lakh rupees.
- Preprocessing in one Pipeline (no data leakage): median/most-frequent imputation, standard scaling, one-hot encoding.
- Models: Linear (OLS), Ridge, Lasso, ElasticNet, Polynomial(2) + Ridge, tuned with GridSearchCV and 5-fold cross-validation.
  The winner is chosen by cross-validation on the training data, not on the test set.
- Benchmark (not used in the app): a gradient-boosting model, to show how much accuracy the linear model gives up.

## 4. Results (test set)
| Model | RMSE (lakh) | R2 | Average % error |
|---|---|---|---|
| poly2_ridge | 68.1 | 0.847 | 19.3% |
| linear | 77.8 | 0.800 | 19.6% |
| ridge | 78.1 | 0.798 | 19.6% |
| elasticnet | 79.1 | 0.793 | 19.7% |
| lasso | 79.1 | 0.793 | 19.7% |
| boosting_benchmark | 62.5 | 0.871 | 17.2% |

Best linear model: **poly2_ridge** (R2 0.847, RMSE 68.1 lakh, average error 19.3%).
The boosting benchmark scores R2 0.871, so the linear model gives up about 2.4 R2 points and stays fully explainable.
In tests, 80% of predictions fell between 0.75 and 1.32 times the true price (the range shown in the app).

## 5. What drives the price (log-linear OLS, robust HC3 standard errors)
| Factor | Price change | 95% interval |
|---|---|---|
| +100 sq ft of area | +4.4% | +4.3% to +4.5% |
| +1 BHK (same area) | +35.6% | +34.8% to +36.4% |
| type = Independent House (vs Apartment) | -6.0% | -21.5% to +12.7% |
| type = Studio Apartment (vs Apartment) | -38.8% | -40.3% to -37.3% |
| type = Villa (vs Apartment) | +9.3% | +0.3% to +19.0% |
| status = Under Construction (vs Ready to move) | +0.9% | +0.3% to +1.5% |
| age = Resale (vs New) | -2.5% | -3.2% to -1.9% |
| age = Unknown (vs New) | -3.3% | -4.0% to -2.6% |

Area (region) matters most: the most expensive areas are several times the price of the cheapest ones at the same size.

## 6. Assumption checks
- Multicollinearity: VIF is about 3 for area and BHK, so there is no problem.
- Heteroscedasticity: the Breusch-Pagan test rejects equal variance (errors grow with price). The log target reduces this, and the
  inference table uses robust (HC3) standard errors to stay valid.
- Independence: Durbin-Watson is about 0.96. The listings have no real time order, so this is a weakness of the data, not a finding.
- Normality: residuals on the log scale are close to normal (skew about -0.1).

## 7. Where the model is weak
See `outputs/error_by_group.csv`: errors rise for 3+ BHK homes and for rare types (villas, independent houses), which have very few listings.
The learning curve (`outputs/learning_curve.png`) shows whether more data would help.

## 8. Limitations
- Asking prices, not sale prices. No floor, view, builder, amenities or exact address in the data.
- Mumbai region only. Pune and Nagpur are not covered.
- Photo option: a vision model guesses room sizes from photos; this is a rough estimate, not a measurement.
- Associations in listings are not proof of cause.

## 9. Reproduce
`pip install -r requirements.txt`, then `python run_all.py` (about one minute), and `streamlit run app.py` for the web app.
