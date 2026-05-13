import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

from location_cleaning import apply_location_normalization

def handle_outliers(df):
    """
    Identifies properties with extreme price-to-area ratios.
    Flags them as 'Luxury Outliers' to prevent model skew, or caps them.
    """
    print("\n--- Outlier Audit & Handling ---")
    # Calculate price to area ratio
    df['price_area_ratio'] = df['price'] / df['area']
    
    # Define extreme outliers based on IQR of the ratio
    Q1 = df['price_area_ratio'].quantile(0.25)
    Q3 = df['price_area_ratio'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    
    # Flag luxury outliers
    df['is_luxury_outlier'] = (df['price_area_ratio'] > upper_bound).astype(int)
    print(f"Identified {df['is_luxury_outlier'].sum()} 'Luxury Outliers' with extreme price-to-area ratios.")
    
    # We will keep them but provide the flag 'is_luxury_outlier' to the model so it can learn this distinction
    df = df.drop(['price_area_ratio'], axis=1)
    return df

def train_baseline_model(data_path='../house_prices_bangalore.csv'):
    print("=====================================================")
    print("SUDDEN DEATH AUDIT: MATHEMATICAL & NOISE OPTIMIZATION")
    print("=====================================================")
    
    df = pd.read_csv(data_path)
    
    # 1. Noise Handling (Location Normalization)
    df = apply_location_normalization(df, col_name='location')
    
    # 2. Outlier Handling (Price-to-Area Ratios)
    df = handle_outliers(df)
    
    # Basic cleaning
    df = df.dropna()
    df = df.drop_duplicates()
    
    # Drop columns that are noisy or irrelevant
    df = df.drop(['balcony', 'parking'], axis=1, errors='ignore')
    
    # 3. Mathematical Audit (RMSLE Optimization via Log1p)
    print("\n--- Mathematical Audit ---")
    print("Transforming target variable 'price' using np.log1p() to minimize RMSLE.")
    df['log_price'] = np.log1p(df['price'])
    
    # 4. Prepare Pipeline
    categorical_cols = ['Cleaned_Location', 'furnishing', 'property_type']
    cols_to_encode = [c for c in categorical_cols if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    
    features = df_encoded.drop(['location', 'price', 'log_price'], axis=1, errors='ignore')
    target = df_encoded['log_price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # 5. Baseline Model Generation (Random Forest)
    print("\n--- Training Random Forest Regressor ---")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    predictions = rf.predict(X_test)
    
    # The RMSE on log_price is exactly the RMSLE on the original price
    rmsle = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"\n[VALIDATION METRIC] Random Forest RMSLE: {rmsle:.4f}")
    
    if rmsle < 0.20:
        print("[SUCCESS] RMSLE is below 0.20! The team is in a very strong position for the Top 10.")
    else:
        print("[WARNING] RMSLE is above 0.20. Further hyperparameter tuning is required.")
        
    # Feature Importance
    print("\n--- Feature Importance Analysis ---")
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print(feature_importance_df.head(10))

if __name__ == "__main__":
    train_baseline_model(os.path.join(os.path.dirname(__file__), '..', 'house_prices_bangalore.csv'))
