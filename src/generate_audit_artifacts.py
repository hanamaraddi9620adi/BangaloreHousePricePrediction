import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from location_cleaning import apply_location_normalization

def generate_audit_artifacts(data_path='../house_prices_bangalore.csv'):
    print("=====================================================")
    print("SUDDEN DEATH AUDIT: ARTIFACT GENERATION")
    print("=====================================================")
    
    df = pd.read_csv(data_path)
    df = apply_location_normalization(df, col_name='location')
    
    # 1. Distribution Plot (Price before vs. after log)
    print("Generating Distribution Plot...")
    df['log_price'] = np.log1p(df['price'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df['price'], kde=True, ax=axes[0], color='salmon')
    axes[0].set_title('Price Distribution (Before Log Transformation)')
    axes[0].set_xlabel('Price (Raw)')
    
    sns.histplot(df['log_price'], kde=True, ax=axes[1], color='teal')
    axes[1].set_title('Price Distribution (After np.log1p Transformation)')
    axes[1].set_xlabel('Log(Price + 1)')
    
    plt.tight_layout()
    plt.savefig('../distribution_plot_audit.png', dpi=300)
    plt.close()
    print("Saved 'distribution_plot_audit.png'")
    
    # 2. Outlier Boxplots (Grouped by BHK)
    print("Generating Outlier Boxplots...")
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='bhk', y='price', data=df, palette='Set2')
    plt.title("Outlier Analysis: Property Price by BHK")
    plt.xlabel("BHK (Bedrooms)")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig('../outlier_boxplot_audit.png', dpi=300)
    plt.close()
    print("Saved 'outlier_boxplot_audit.png'")
    
    # 3. Feature Importance Chart
    # We will grab the feature importances by training a quick RF model
    print("Generating Feature Importance Chart...")
    df = df.dropna()
    df_encoded = pd.get_dummies(df, columns=['Cleaned_Location', 'furnishing', 'property_type'], drop_first=True)
    features = df_encoded.drop(['location', 'price', 'log_price', 'balcony', 'parking'], axis=1, errors='ignore')
    target = df_encoded['log_price']
    
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(features, target)
    
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='magma')
    plt.title('Top 10 Price Drivers (Feature Importance)')
    plt.tight_layout()
    plt.savefig('../feature_importance_audit.png', dpi=300)
    plt.close()
    print("Saved 'feature_importance_audit.png'")

if __name__ == "__main__":
    generate_audit_artifacts(os.path.join(os.path.dirname(__file__), '..', 'house_prices_bangalore.csv'))
