# Car Price Prediction

Predicts the selling price of used cars using Linear Regression and Random Forest Regression, based on features like present price, kilometers driven, fuel type, transmission, and car age.

## Dataset
- 301 rows, 9 columns (`car_data.csv`)
- Features: Car_Name, Year, Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner
- Target: Selling_Price

## Steps Followed

1. **EDA** — checked value counts, nulls, statistical summary
2. **Feature Engineering**
   - Extracted `Brand` from `Car_Name`
   - Converted `Year` into `Age_of_car`
3. **Visualization** — pairplots, boxplots (Fuel Type / Transmission / Selling Type vs Price), correlation heatmap
4. **Train/Test Split** — done *before* encoding to avoid data leakage
5. **Encoding** — rare brands (<5 occurrences in training data) grouped into "Other", then one-hot encoded
6. **Modeling** — Linear Regression (baseline) and Random Forest Regressor
7. **Evaluation** — R², MAE, RMSE, and 5-fold cross-validation for both models
8. **Error Analysis** — inspected the top 5 largest prediction errors
9. **Visualization** — Actual vs Predicted scatter plot

## Results

| Model | Test R² | 5-Fold CV Avg R² |
|---|---|---|
| Linear Regression | 0.84 | 0.84 |
| Random Forest | 0.96 | **0.88** |

Random Forest performs better overall, but the cross-validation average (0.88) is reported as the primary metric rather than the single test-split score (0.96), since CV gives a more reliable estimate on this small dataset (301 rows).

## Key Insight
`Present_Price` is the strongest predictor of `Selling_Price` (correlation ≈ 0.88). The model is noticeably weaker on higher-priced cars, likely due to fewer high-price samples in the dataset.

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

