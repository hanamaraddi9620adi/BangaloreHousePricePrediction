import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import os

from technical_pivot_analysis import analyze_pivot

def run_efficiency_benchmark(data_path='../house_prices_bangalore.csv'):
    print("=====================================================")
    print("MODEL SELECTION & EFFICIENCY BENCHMARKING")
    print("=====================================================")
    
    # Load and prep data
    df = analyze_pivot(data_path)
    df = df.dropna().drop_duplicates()
    
    df['total_rooms'] = df['bhk'] + df['bath']
    df['area_per_bhk'] = df['area'] / df['bhk'].replace(0, 1)
    df['bath_to_bhk_ratio'] = df['bath'] / df['bhk'].replace(0, 1)
    
    df['log_price'] = np.log1p(df['price'])
    
    df = pd.get_dummies(df, columns=['location', 'furnishing', 'property_type'], drop_first=True)
    
    feature_cols = [c for c in df.columns if c not in ['price', 'log_price', 'balcony', 'parking']]
    X = df[feature_cols]
    y = df['log_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = (time.time() - start_time) * 1000 # in milliseconds
        
        preds = model.predict(X_test)
        rmsle = np.sqrt(mean_squared_error(y_test, preds))
        
        results.append({
            'Model': name,
            'Training Time (ms)': train_time,
            'RMSLE': rmsle
        })
        print(f"{name} -> Time: {train_time:.2f}ms | RMSLE: {rmsle:.4f}")
        
    results_df = pd.DataFrame(results)
    
    # Plotting Efficiency vs Accuracy for the Presentation
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Model Selection')
    ax1.set_ylabel('RMSLE (Lower is Better)', color=color)
    sns.barplot(x='Model', y='RMSLE', data=results_df, ax=ax1, palette='Reds_r')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Training Time in ms (Lower is Better)', color=color)  
    sns.lineplot(x='Model', y='Training Time (ms)', data=results_df, ax=ax2, color=color, marker='o', linewidth=3, markersize=10)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Model Selection Justification: Efficiency vs. Error (RMSLE)')
    fig.tight_layout()  
    plt.savefig('../efficiency_presentation_slide.png', dpi=300)
    plt.close()
    print("\nSaved 'efficiency_presentation_slide.png' for your pitch deck!")

if __name__ == "__main__":
    run_efficiency_benchmark(os.path.join(os.path.dirname(__file__), '..', 'house_prices_bangalore.csv'))
