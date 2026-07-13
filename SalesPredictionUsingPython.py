# ____IMPORTING LIBRARIES___
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

# STEP 1: DATA PREPROCESSING AND VISUALISATION

df = pd.read_csv("Advertising.csv")

# DATA INSPECTION...
print("ADVERTISING DATASET SAMPLE")
print("PRINTING FIRST FIVE ROWS FIRST")
print(df.head())

# DATA CLEANING...
df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

print("\nCLEANED DATA")
print(df)

print("DATA INFO-COLUMNS, NON-NULL COUNT AND DATATYPE")
print(
    df.info()
)  # Result: 200 non null values , if it would be 150  non null and 50 null values then machine learning math coudlnt handled these-- so we got a proper idea now that how many are non null and how many are null values.

print("CHECKING FOR MISSING VALUES")
print(df.isnull().sum())

print("Stats of Data")
print(df.describe())

# ____VISUALISATION OF DATA___
plt.figure(figsize=(7, 6))
sns.heatmap(df.corr(), annot=True, cmap="PuOr", fmt=".2f",vmin=-1,vmax=1)
plt.title("Correlation Heatmap: Advertising v/s Sales")
plt.show()

sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='reg')
plt.suptitle("Pairplot", y=1.02, fontsize=14)
plt.show()

sns.boxplot(data=df[["TV", "Radio", "Newspaper"]],palette='PuOr')
plt.title("Detecting Outliers")
plt.ylabel("Advertising Spent")
plt.show()


# STEP 2: LINEAR REGRESSION AND INTERACTION EFFECT
# ___LINEAR_REGRESSION____
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_lr_model = lr_model.predict(X_test)

# ____Metrics_Evaluation____
r2score_lr = r2_score(y_test, y_lr_model)
mae_lr = mean_absolute_error(y_test, y_lr_model)
mse_lr = mean_squared_error(y_test, y_lr_model)
rmse_lr = np.sqrt(mse_lr)

print("\n----------")
print("\n____First Model:R2score,MAE,RMSE____")
print(f"R2 Score (Variance):{r2score_lr:.4f}")
print(f"MAE(Mean Absolute Error):{mae_lr:.4f}")
print(f"RMSE(Root mean square error):{rmse_lr:.4f}")

# ____FEATURE_ENGINEERING:Adding interaction Term____
df_copy = df.copy()
df_copy["TVandRadio"] = df_copy["TV"] * df_copy["Radio"]

X_new = df_copy[["TV", "Radio", "Newspaper", "TVandRadio"]]
y_new = df_copy["Sales"]

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42
)

lr_newmodel = LinearRegression()
lr_newmodel.fit(X_train_new, y_train_new)

y_pred_new = lr_newmodel.predict(X_test_new)

r2_new = r2_score(y_test_new, y_pred_new)
mae_new = mean_absolute_error(y_test_new, y_pred_new)
mse_new = mean_squared_error(y_test_new, y_pred_new)
rmse_new = np.sqrt(mse_new)

print("\n----------")
print("\n____Second Model (with new feature) R2score,MAE,RMSE____ ")
print(f"R2 Score (Variance) of new data:{r2_new:.4f}")
print(f"MAE(Mean Absolute Error) of new data:{mae_new:.4f}")
print(f"RMSE(Root mean square error) of new data:{rmse_new:.4f}")

# STEP 3:  RANDOM FOREST REGRESSOR
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

print("\n--------------")
print(f"Random Forest Test R2 Score:{r2_rf:.4f}")
print(f"Random Forest Test RMSE:{rmse_rf:.4f}")


# STEP 4: CROSS-VALIDATION----
print("\n----------------------------")
print('CV Scores for Linear Regression (first Model)')
print("\n----------------------------")
cross_valid_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring="r2")
average_r2score = cross_valid_scores.mean()

print(f"All 5 R2 Scores (Scores per Fold):{np.round(cross_valid_scores,4)}")
print(f"Average R2 Score (Mean R2 Score):{average_r2score:.4f}")

print("\n----------------------------")
print('CV Scores for Random Forest Model')
print("\n----------------------------")
rf_CV_scores=cross_val_score(rf,X_train,y_train,cv=5,scoring='r2')

print(f'Random forest Fold R2 Scores:{np.round(rf_CV_scores,4)}')
print(f'Random Forest Average CV R2 Score:{rf_CV_scores.mean():.4f} ')


#Actual vs Predicted Sales
comparison=pd.DataFrame({
    'Actual Sales':y_test,
    'Predicted Sales':y_pred_rf
})
print('\n---------------------------')
print('ACTUAL VS PREDICTED SALES')
print('\n---------------------------')
print(comparison.head(10))
plt.scatter(y_test,y_pred_rf,alpha=0.7)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2,
    label="Perfect Prediction"
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales (Random Forest)")
plt.legend()
plt.show()
#---------------------------------------------------------------------
# STEP 4: FUTURE SALES PREDICTION
#---------------------------------------------------------------------

budgets = pd.DataFrame(
    [[300.0, 100.0, 0.0], [120.0, 35.0, 17.0]], columns=["TV", "Radio", "Newspaper"]
)

# Reusing the already fitted 'rf' model from Phase 3 to predict future budgets
future_prediction = rf.predict(budgets)
print('\n---------------------------')
print('Future Sales Prediction')
print('\n---------------------------')
print(f"Budget 1 Forecasted Sales: {future_prediction[0]:.2f}")
print(f"Budget 2 Forecasted Sales: {future_prediction[1]:.2f}")