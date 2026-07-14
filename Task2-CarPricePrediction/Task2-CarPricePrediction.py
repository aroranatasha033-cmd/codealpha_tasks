#_____IMPORTING LIBRARIES______
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# STEP1: DATA PREPROCESSING AND VISUALISATION

df = pd.read_csv('car_data.csv')
print('\n----------------------')
print('Car Price Dataset')
print("Printing First five rows")
print(df.head())
print('\n----------------------')

#Data Inspection
print(df['Owner'].value_counts())
print(df['Selling_type'].value_counts())
print(df['Fuel_Type'].value_counts())
print(df['Transmission'].value_counts())

print("Information of whole dataset:")
print(df.info())

print("Checking for Missing Values:")
print(df.isnull().sum())

print("Statistics of whole data")
print(df.describe())


#Data Cleaning
df['Brand'] = df['Car_Name'].apply(lambda x: x.split()[0])

#Age of the car
present_year = datetime.datetime.now().year
df['Age_of_car'] = present_year - df['Year']

df.drop('Year', axis=1, inplace=True)
df.drop('Car_Name', axis=1, inplace=True)

print('\n---Data after preprocessing---')
print(df)

#-------VISUALISATION OF DATA----------
#PairPlot
sns.pairplot(df, x_vars=['Age_of_car', 'Present_Price', 'Driven_kms'], y_vars='Selling_Price', kind='reg')
plt.suptitle("Pairplot", y=1.02, fontsize=13)
plt.show()

#BoxPlot
plt.figure(figsize=(15, 5))

#Fuel Type vs Selling Price
plt.subplot(1, 3, 1)
sns.boxplot(x='Fuel_Type', y='Selling_Price', data=df, hue='Fuel_Type', palette='PuOr', legend=False)
plt.title('Fuel Type vs Selling Price')

#Transmission vs Selling Price
plt.subplot(1, 3, 2)
sns.boxplot(x='Transmission', y='Selling_Price', data=df, hue='Transmission', palette='PuOr', legend=False)
plt.title('Transmission vs Selling Price')

#Selling Type vs Selling Price
plt.subplot(1, 3, 3)
sns.boxplot(x='Selling_type', y='Selling_Price', data=df, hue='Selling_type', palette='PuOr', legend=False)
plt.title('Selling Type vs Selling Price')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
numeric_columns = ['Selling_Price', 'Present_Price', 'Driven_kms', 'Age_of_car']
correl_matrix = df[numeric_columns].corr()
sns.heatmap(correl_matrix, annot=True, cmap='PuOr', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()


# STEP2: TRAIN TEST SPLIT

X = df.drop(columns=['Selling_Price'])
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#grouping rare brands using train data only
brand_counts_train = X_train['Brand'].value_counts()
rare_brands = brand_counts_train[brand_counts_train < 5].index

print("Grouping rare brands into new column 'Other'")
X_train['Brand'] = X_train['Brand'].apply(lambda x: 'Other' if x in rare_brands else x)
X_test['Brand'] = X_test['Brand'].apply(lambda x: 'Other' if x in rare_brands else x)
print(X_train['Brand'].value_counts())

#One Hot Encoding to convert Categorical values to numeric values.
X_train = pd.get_dummies(X_train, columns=['Fuel_Type', 'Selling_type', 'Transmission', 'Brand'], drop_first=True, dtype=int)
X_test = pd.get_dummies(X_test, columns=['Fuel_Type', 'Selling_type', 'Transmission', 'Brand'], drop_first=True, dtype=int)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# STEP3: Using Linear Regression Model

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
r2score_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

print("\n----------")
print("\n____Linear Regression(First Model) R2score,MAE,RMSE____ ")
print(f"R2 Score (Variance):{r2score_lr:.4f}")
print(f"MAE(Mean Absolute Error):{mae_lr:.4f}")
print(f"RMSE(Root mean square error):{rmse_lr:.4f}")

# Cross validation for Linear Regression
lr_cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring="r2")
print("\n----------------")
print('CV Scores for Linear Regression')
print(f"All 5 R2 Scores:{lr_cv_scores}")
print(f"Average Linear Regression R2 Score:{lr_cv_scores.mean():.4f}")


# STEP 4: RANDOM FOREST REGRESSOR
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf_train = rf.predict(X_train)
y_pred_rf = rf.predict(X_test)

r2_rf_train = r2_score(y_train, y_pred_rf_train)
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

print("\n--------Random forest Regressor-------")
print(f"Random Forest Train R2 Score:{r2_rf_train:.4f}")
print(f"Random Forest Test R2 Score:{r2_rf:.4f}")
print(f"Random Forest Test RMSE:{rmse_rf:.4f}")

# Cross validation for Random Forest
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2")
print("\n----------------")
print('CV Scores for Random Forest')
print(f"All 5 R2 Scores:{rf_cv_scores}")
print(f"Average Random Forest R2 Score:{rf_cv_scores.mean():.4f}")

# STEP 5: ERROR ANALYSIS
analysis_df = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred_rf,
    'Absolute Error': abs(y_test - y_pred_rf)
}, index=X_test.index)

error_analysis = pd.concat([X_test, analysis_df], axis=1)

# top 5 large errors
print("____ TOP 5 LARGE ERRORS ____")
print(error_analysis.sort_values(by='Absolute Error', ascending=False).head(5))

# STEP 6: ACTUAL V/S PREDICTED PLOTS
comparison = pd.DataFrame({
    'Actual Selling_Price': y_test,
    'Predicted Selling_Price': y_pred_rf
})

print('ACTUAL VS PREDICTED SELLING_PRICE')
print(comparison.head(10))
plt.scatter(y_test, y_pred_rf, alpha=0.7)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="green",
    linewidth=2,
    label="Perfect Prediction"
)

plt.xlabel("Actual Selling_Price")
plt.ylabel("Predicted Selling_Price")
plt.title("Actual vs Predicted Selling_Price(Random Forest Regressor)")
plt.legend()
plt.show()