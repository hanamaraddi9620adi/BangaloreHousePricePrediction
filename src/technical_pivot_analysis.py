import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Provided Pivot Data
coords_data = {
    'location': [
        'Bannerghatta Road', 'Yelahanka', 'Sarjapur Road', 'Whitefield', 'Banashankari',
        'KR Puram', 'JP Nagar', 'Uttarahalli', 'Rajajinagar', 'Marathahalli',
        'Jayanagar', 'HSR Layout', 'BTM Layout', 'Koramangala', 'Indiranagar',
        'Electronic City', 'Kengeri', 'Bellandur', 'Malleshwaram', 'Hebbal'
    ],
    'latitude': [
        12.9396275, 13.1006982, 12.9246491, 12.9957428, 12.9278196,
        13.007516, 12.9096941, 12.9055682, 13.0005359, 12.9552572,
        12.9292731, 12.9116225, 12.9140008, 12.9357366, 12.9732913,
        12.8487599, 12.9176571, 12.9320495, 13.0027353, 13.0382184
    ],
    'longitude': [
        77.6021696, 77.5963454, 77.6382779, 77.7579489, 77.556621,
        77.695935, 77.5866067, 77.5455438, 77.5496996, 77.6984163,
        77.5824229, 77.6388622, 77.6102821, 77.624081, 77.6404672,
        77.648253, 77.4837568, 77.6842915, 77.5703253, 77.5919
    ]
}

def analyze_pivot(data_path='../house_prices_bangalore.csv'):
    print("=====================================================")
    print("TECHNICAL PIVOT CHALLENGE: INFRASTRUCTURE PROXIMITY")
    print("=====================================================")
    
    df = pd.read_csv(data_path)
    coords_df = pd.DataFrame(coords_data)
    
    # Merge coordinates into main dataset
    df = pd.merge(df, coords_df, on='location', how='left')
    
    # The absolute heart of Bengaluru Metro is Majestic / MG Road.
    # We will use MG Road Metro Station (Lat: 12.9755, Lon: 77.6067) as the primary hub.
    MG_ROAD_LAT = 12.9755
    MG_ROAD_LON = 77.6067
    
    print("Calculating distance from each property to MG Road Metro Station...")
    df['distance_to_metro_km'] = df.apply(
        lambda row: haversine(row['latitude'], row['longitude'], MG_ROAD_LAT, MG_ROAD_LON), 
        axis=1
    )
    
    # Let's check if the price goes up when closer to the metro!
    # Correlation between distance and price
    correlation = df['distance_to_metro_km'].corr(df['price'])
    print(f"\nCorrelation between Distance to Metro and Price: {correlation:.4f}")
    if correlation < 0:
        print("-> Insight: Negative correlation! As distance INCREASES, price DECREASES.")
        print("-> Conclusion: Properties in close proximity to the Metro Hub are MORE EXPENSIVE.")
    else:
        print("-> Insight: Positive correlation. Distance increases price (unlikely for city hubs).")
        
    # Plotting the Pivot Challenge Answer
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='distance_to_metro_km', y='price', data=df, alpha=0.6, color='dodgerblue')
    sns.regplot(x='distance_to_metro_km', y='price', data=df, scatter=False, color='red')
    
    plt.title('Technical Pivot: Property Price vs. Distance to Metro Hub (MG Road)')
    plt.xlabel('Distance to Nearest Metro Hub (km)')
    plt.ylabel('Property Price (INR)')
    plt.tight_layout()
    plt.savefig('../pivot_metro_analysis.png', dpi=300)
    plt.close()
    print("\nSaved plot to 'pivot_metro_analysis.png' for the presentation.")
    
    # Return the new enriched dataframe so we can push it into our model
    return df

if __name__ == "__main__":
    analyze_pivot(os.path.join(os.path.dirname(__file__), '..', 'house_prices_bangalore.csv'))
