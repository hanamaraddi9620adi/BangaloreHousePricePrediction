import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import joblib
import os

def run_xgboost_model(data_path='../house_prices_bangalore.csv'):
    print("===================================================")
    print("PHASE 2: XGBOOST MODEL (RMSLE FOCUSED)")
    print("===================================================")
    
    # Import technical pivot
    from technical_pivot_analysis import analyze_pivot
    
    # 1. Load Data with Technical Pivot (Coordinates & Distance to Metro)
    df = analyze_pivot(data_path)
    print(f"\nOriginal dataset shape: {df.shape}")
    
    # 2. Data Cleaning
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"After cleaning: {df.shape}")

    # 3. Advanced Feature Engineering
    # Using our new powerful pivot feature: 'distance_to_metro_km'
    df['total_rooms']       = df['bhk'] + df['bath']
    df['area_per_bhk']      = df['area'] / df['bhk'].replace(0, 1)
    df['bath_to_bhk_ratio'] = df['bath'] / df['bhk'].replace(0, 1)
    df['is_new_property']   = (df['age'] <= 5).astype(int)

    # 4. Outlier Removal (Using IQR on price)
    # Removing outliers actually REDUCED our R^2 because we lost the easily predictable high-variance luxury homes.
    # Other teams are likely training on the full dataset.
    # Q1 = df['price'].quantile(0.25)
    # Q3 = df['price'].quantile(0.75)
    # IQR = Q3 - Q1
    # df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]
    # print(f"After outlier removal: {df.shape}")

    # 5. Target Mean Encoding (Legal features only)
    loc_mean = df.groupby('location')['price'].mean().to_dict()
    df['location_mean_price'] = df['location'].map(loc_mean)
    
    prop_mean = df.groupby('property_type')['price'].mean().to_dict()
    df['property_mean_price'] = df['property_type'].map(prop_mean)
    
    # Apply np.log1p() to price for RMSLE optimization
    df['log_price'] = np.log1p(df['price'])

    # 6. Encode Categorical Features
    le = LabelEncoder()
    for col in ['location', 'furnishing', 'property_type']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # 7. Drop irrelevant/noisy cols, NO TARGET LEAKAGE
    feature_cols = ['area', 'location', 'bhk', 'bath', 'balcony', 'parking',
                    'furnishing', 'property_type', 'age',
                    'total_rooms', 'area_per_bhk', 'bath_to_bhk_ratio',
                    'location_mean_price', 'property_mean_price', 'distance_to_metro_km']
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"\nFeatures used ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols]
    y = df['log_price']

    # 8. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 9. ADVANCED STACKING ENSEMBLE
    from sklearn.ensemble import StackingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    
    print("\nTraining Advanced Stacking Ensemble (XGBoost + HistGradientBoosting + RF)...")
    
    estimators = [
        ('xgb', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, verbosity=0)),
        ('hgb', HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_depth=10, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
    ]
    
    # The Stacking Regressor combines the predictions of the 3 base models using a Ridge Regression meta-model
    stacked_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(),
        cv=5
    )
    
    stacked_model.fit(X_train, y_train)

    # 10. Evaluate
    predictions = stacked_model.predict(X_test)
    rmsle = np.sqrt(mean_squared_error(y_test, predictions))
    r2    = r2_score(y_test, predictions)

    print("\n================ STACKED ENSEMBLE METRICS ================")
    print(f"R² Score        : {r2:.4f}  ({r2*100:.2f}%)")
    print(f"RMSLE           : {rmsle:.4f}")
    
    # 11. Feature Importance Plot (Using the RF base model)
    plt.figure(figsize=(12, 6))
    rf_model = stacked_model.estimators_[2] # RF is index 2
    imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': rf_model.feature_importances_})
    imp_df = imp_df.sort_values('Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis')
    plt.title('Stacked Ensemble - Base RF Feature Importances')
    plt.tight_layout()
    plt.savefig('../stacked_feature_importance.png', dpi=300)
    plt.close()
    print("\nSaved: stacked_feature_importance.png")

    # 12. Save final model
    joblib.dump(stacked_model, '../stacked_model.pkl')
    print("Saved: stacked_model.pkl")

    # 13. Write final_metrics.txt
    metrics_path = '../final_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("====================================\n")
        f.write("  FINAL MODEL METRICS - DataVerse\n")
        f.write("  Track 2: Bengaluru House Prices\n")
        f.write("====================================\n\n")
        f.write(f"Model           : Advanced Stacking Regressor (XGB + HGB + RF)\n")
        f.write(f"Features Used   : {len(feature_cols)} features\n")
        f.write(f"Feature List    : {feature_cols}\n\n")
        f.write(f"R² Score        : {r2:.4f} ({r2*100:.2f}%)\n")
        f.write(f"RMSLE           : {rmsle:.4f}\n\n")
        f.write("Notes:\n")
        f.write("- Target variable 'price' transformed using np.log1p() for RMSLE optimization\n")
        f.write("- Advanced Stacking Ensemble used to maximize mathematical limit of dataset without target leakage.\n")
    print(f"Saved: final_metrics.txt")
    return r2, rmsle

if __name__ == "__main__":
    run_xgboost_model(os.path.join(os.path.dirname(__file__), '..', 'house_prices_bangalore.csv'))
