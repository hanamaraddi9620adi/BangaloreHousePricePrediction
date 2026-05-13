import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import sys

# Import our cleaning functions
from location_cleaning import apply_location_normalization, apply_log_transformation

def run_optimized_rf(data_path='../house_prices_bangalore.csv'):
    print("\n===================================================")
    print("PHASE 2: ADVANCED ENSEMBLE MODEL (MAX ACCURACY)")
    print("===================================================\n")
    
    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"Error: Could not find dataset at {data_path}.")
        return
        
    df = pd.read_csv(data_path)
    
    # 2. Data Cleaning
    df = df.dropna()
    df = df.drop_duplicates()
    
    # 3. Advanced Feature Engineering
    df['total_rooms'] = df['bhk'] + df['bath']
    df['area_per_bhk'] = df['area'] / df['bhk'].replace(0, 1) # Avoid division by zero
    df['bath_to_bhk_ratio'] = df['bath'] / df['bhk'].replace(0, 1)
    df['is_new_property'] = (df['age'] <= 5).astype(int)
    
    # 4. Outlier Removal (Using IQR on Price to remove luxury anomalies that skew learning)
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    df = df[(df['price'] >= lower_limit) & (df['price'] <= upper_limit)]
    
    # 5. Apply Team Integration Tools
    df = apply_location_normalization(df, col_name='location')
    df = apply_log_transformation(df, col_name='price')
    
    # 6. Prepare Features and Target
    # One-hot encode categorical features
    categorical_cols = ['Cleaned_Location', 'furnishing', 'property_type']
    cols_to_encode = [c for c in categorical_cols if c in df.columns]
    
    df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    
    # Drop original non-numeric and noisy columns
    features = df_encoded.drop(['location', 'price', 'Log_price', 'balcony', 'parking'], axis=1, errors='ignore')
    target = df_encoded['Log_price']
    
    # 7. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # 8. Train Advanced Model (HistGradientBoostingRegressor handles complex, non-linear relationships best)
    print("\nTraining HistGradientBoostingRegressor...")
    rf = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=500,
        max_depth=15,
        min_samples_leaf=5,
        l2_regularization=0.1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # 9. Evaluate
    predictions = rf.predict(X_test)
    
    rmsle = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("\n================ MODEL METRICS ================")
    print(f"Model Accuracy (R² Score): {r2 * 100:.2f}%")
    print(f"RMSLE (Track 2 Objective): {rmsle:.4f}")
    
    if rmsle < 0.20:
         print("[SUCCESS] RMSLE is below 0.20!")
    else:
         print("[WARNING] RMSLE is still above 0.20. Waiting for 3:00 PM secret metadata.")
    
    # 10. Save Model
    joblib.dump(rf, '../optimized_rf_model.pkl')
    print(f"\nSaved tuned model to optimized_rf_model.pkl")

if __name__ == "__main__":
    run_optimized_rf(data_path=os.path.join(os.path.dirname(__file__), '..', 'house_prices_bangalore.csv'))
