
# ============================================
# HOUSE PRICE PREDICTION PROJECT
# FINAL OPTIMIZED VERSION WITH RMSLE
# ============================================

# ==================================================
# STEP 1 — IMPORT LIBRARIES
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score
)


# ==================================================
# STEP 2 — LOAD DATASET
# ==================================================

df = pd.read_csv("house_prices_bangalore.csv")


# ==================================================
# STEP 3 — BASIC DATA INSPECTION
# ==================================================

print("\n================ FIRST 5 ROWS ================\n")
print(df.head())

print("\n================ DATASET SHAPE ================\n")
print(df.shape)

print("\n================ MISSING VALUES ================\n")
print(df.isnull().sum())


# ==================================================
# STEP 4 — DATA CLEANING
# ==================================================

# Remove missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

print("\n================ FINAL DATASET SHAPE ================\n")
print(df.shape)


# ==================================================
# STEP 5 — REMOVE WEAK FEATURES
# ==================================================

# Removing weak features
# Simpler model can improve generalization

df = df.drop(
    ['balcony', 'parking'],
    axis=1
)


# ==================================================
# STEP 6 — FEATURE ENGINEERING
# ==================================================

# Total rooms feature
df['total_rooms'] = df['bhk'] + df['bath']

# NOTE:
# Removed price_per_sqft because it creates
# target leakage using price column


# ==================================================
# STEP 7 — OUTLIER REMOVAL
# ==================================================

# Using IQR Method

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

df = df[
    (df['price'] >= lower_limit) &
    (df['price'] <= upper_limit)
]

print("\n================ DATASET SHAPE AFTER OUTLIER REMOVAL ================\n")
print(df.shape)


# ==================================================
# STEP 8 — VISUALIZATIONS
# ==================================================

# --------------------------------------------------
# PRICE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

df['price'].hist(bins=30)

plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")

plt.show()


# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap='coolwarm',
    annot=True
)

plt.title("Correlation Heatmap")

plt.show()


# --------------------------------------------------
# AREA VS PRICE
# --------------------------------------------------

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x='area',
    y='price',
    data=df
)

plt.title("Area vs Price")

plt.show()


# ==================================================
# STEP 9 — LOCATION ANALYSIS
# ==================================================

print("\n================ LOCATION WISE AVERAGE PRICE ================\n")

location_price = df.groupby('location')['price'].mean()

print(location_price.sort_values(ascending=False))


# ==================================================
# STEP 10 — SPLIT FEATURES & TARGET
# ==================================================

# Features
X = df.drop('price', axis=1)

# Target
y = df['price']


# ==================================================
# STEP 11 — ENCODE CATEGORICAL DATA
# ==================================================

X = pd.get_dummies(
    X,
    drop_first=True
)


# ==================================================
# STEP 12 — TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==================================================
# STEP 13 — LINEAR REGRESSION MODEL
# ==================================================

print("\n===================================================")
print("FINAL LINEAR REGRESSION MODEL")
print("===================================================\n")

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
prediction = model.predict(X_test)

# RMSLE requires non-negative predictions
prediction = np.maximum(prediction, 0)

# R2 Score
r2 = r2_score(y_test, prediction)

print("Model Accuracy (R2 Score):")
print(r2)

print("\nModel Accuracy Percentage:")
print(r2 * 100)


# ==================================================
# STEP 14 — ERROR METRICS
# ==================================================

# --------------------------------------------------
# MAE
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    prediction
)

print("\nMean Absolute Error (MAE):")
print(mae)


# --------------------------------------------------
# RMSE
# --------------------------------------------------

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediction
    )
)

print("\nRoot Mean Squared Error (RMSE):")
print(rmse)


# --------------------------------------------------
# RMSLE
# --------------------------------------------------

rmsle = np.sqrt(
    mean_squared_log_error(
        y_test,
        prediction
    )
)

print("\nRoot Mean Squared Log Error (RMSLE):")
print(rmsle)


# ==================================================
# STEP 15 — ACTUAL VS PREDICTED GRAPH
# ==================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    prediction
)

# Perfect Prediction Line

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red'
)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Linear Regression: Actual vs Predicted")

plt.show()


# ==================================================
# STEP 16 — IMPORTANT FEATURES
# ==================================================

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

feature_importance = feature_importance.sort_values(
    by='Coefficient',
    ascending=False
)

print("\n================ TOP IMPORTANT FEATURES ================\n")

print(feature_importance.head(10))


# ==================================================
# STEP 17 — FEATURE IMPORTANCE GRAPH
# ==================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    x='Coefficient',
    y='Feature',
    data=feature_importance.head(10)
)

plt.title("Top Important Features")

plt.show()


# ==================================================
# STEP 18 — SAVE MODEL
# ==================================================

joblib.dump(
    model,
    'house_price_model.pkl'
)

print("\nModel saved successfully as house_price_model.pkl")


# ==================================================
# STEP 19 — PROJECT COMPLETED
# ==================================================

print("\n===================================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("===================================================\n")


# ==================================================
# FINAL PROJECT SUMMARY
# ==================================================

# COMPLETED:
# 1. Data Cleaning
# 2. Duplicate Removal
# 3. Outlier Removal
# 4. Feature Engineering
# 5. Data Visualization
# 6. Correlation Analysis
# 7. Location Analysis
# 8. Encoding
# 9. Linear Regression Model
# 10. MAE Evaluation
# 11. RMSE Evaluation
# 12. RMSLE Evaluation
# 13. Feature Importance
# 14. Saved ML Model

