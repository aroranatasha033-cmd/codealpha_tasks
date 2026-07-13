# codealpha_tasks
Data Science internship tasks for CodeAlpha — includes sales prediction using Linear Regression and Random Forest with feature engineering and cross-validation.
# CodeAlpha Tasks

This repository contains my internship tasks for **CodeAlpha**.

---

## Task: Sales Prediction using Python

Predicting product sales based on advertising spend across TV, Radio, and Newspaper channels, using the classic Advertising dataset.

### Workflow

1. **Data Preprocessing & EDA**
   - Cleaned the dataset (removed index column, checked for nulls — none found)
   - Visualized feature correlations with a heatmap
   - Explored relationships between each ad channel and sales with pairplots
   - Checked for outliers using boxplots

2. **Modeling**
   - **Linear Regression (baseline)** — R² Score: 0.899
   - **Linear Regression + Interaction Term (TV × Radio)** — R² Score: 0.974
   - **Random Forest Regressor** — R² Score: 0.981

3. **Validation**
   - 5-fold cross-validation confirmed model stability:
     - Linear Regression average R²: 0.860
     - Random Forest average R²: 0.969

4. **Prediction**
   - Used the trained Random Forest model to forecast sales for new advertising budgets

### Key Insight
Adding an interaction term between TV and Radio spend significantly improved the linear model, suggesting these two channels have a combined effect on sales beyond their individual contributions. The Random Forest model captured this non-linearity even better, with the most stable cross-validation performance.

### Files
- `SalesPredictionUsingPython.py` — full analysis and modeling script
- `Advertising.csv` — dataset
- `Correlation HeatMap.png` — feature correlation heatmap
- `Pairplot.png` — advertising spend vs sales relationships
- `BoxPlot.png` — outlier detection
- `ActualandPredictedSales.png` — model performance visualization

### Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---
*Part of my Data Science internship tasks at CodeAlpha.*
