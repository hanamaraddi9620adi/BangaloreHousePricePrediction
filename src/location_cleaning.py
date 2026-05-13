import pandas as pd
import re

def clean_location(location: str) -> str:
    """
    Cleans and normalizes non-standardized location strings.
    This function targets specific variations (like Whitefield) and removes
    unnecessary characters, standardizing the text to lower case and stripping
    extra whitespace.
    
    Args:
        location (str): The raw location string.
        
    Returns:
        str: The cleaned and standardized location string.
    """
    # 1. Handle missing or non-string values
    if pd.isna(location) or not isinstance(location, str):
        return "unknown"
        
    # 2. Lowercase and strip whitespace for uniform processing
    loc = location.lower().strip()
    
    # 3. Remove unwanted characters (e.g., punctuation like commas, dots)
    loc = re.sub(r'[^\w\s]', ' ', loc)
    
    # 4. Collapse multiple spaces into a single space
    loc = re.sub(r'\s+', ' ', loc).strip()
    
    # 5. Group common variants (Rule 6.2 - Explainability)
    # Whitefield variants
    if 'whitefield' in loc or 'white field' in loc:
        return 'whitefield'
    
    # Electronic City variants
    if 'electronic city' in loc or 'e city' in loc or 'ecity' in loc:
        return 'electronic city'
        
    # Koramangala variants
    if 'koramangala' in loc or 'kormangala' in loc:
        return 'koramangala'
        
    # HSR Layout variants
    if 'hsr' in loc:
        return 'hsr layout'
        
    # BTM Layout variants
    if 'btm' in loc:
        return 'btm layout'
        
    # Indiranagar variants
    if 'indiranagar' in loc or 'indira nagar' in loc:
        return 'indiranagar'
        
    return loc

def apply_location_normalization(df: pd.DataFrame, col_name: str = 'Location') -> pd.DataFrame:
    """
    Applies the clean_location function to the specified column in the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        col_name (str): The name of the location column to clean. Default is 'Location'.
        
    Returns:
        pd.DataFrame: The DataFrame with the cleaned location column.
    """
    # Ensure column exists before applying
    if col_name in df.columns:
        # Apply the cleaning function and store in a new standardized column
        df['Cleaned_Location'] = df[col_name].apply(clean_location)
    else:
        print(f"Warning: Column '{col_name}' not found in DataFrame.")
    
    return df

def apply_log_transformation(df: pd.DataFrame, col_name: str = 'Price') -> pd.DataFrame:
    """
    Applies a log1p transformation to the specified numerical column to handle 
    high-variance and achieve the lowest RMSLE score.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        col_name (str): The column to transform. Default is 'Price'.
        
    Returns:
        pd.DataFrame: The DataFrame with an added 'Log_{col_name}' column.
    """
    import numpy as np
    if col_name in df.columns:
        # We use np.log1p (which computes log(1+x)) to gracefully handle zeros 
        # and ensure stability for the RMSLE metric.
        df[f'Log_{col_name}'] = np.log1p(df[col_name])
    else:
        print(f"Warning: Column '{col_name}' not found in DataFrame.")
    return df

def calculate_distance_from_hub(location: str) -> float:
    """
    Placeholder function to calculate the 'Distance from Hub' (e.g., Metro distance).
    This logic will be populated once the 3:00 PM secret metadata is provided.
    
    Args:
        location (str): The name of the cleaned location.
        
    Returns:
        float: Estimated distance in km from the primary infrastructure hub.
    """
    # TODO: Integrate 3:00 PM secret metadata API or mapping logic here.
    # Currently returning a placeholder value.
    return 0.0

def integrate_infrastructure_metadata(house_lat: float, house_lon: float, hub_coords: list) -> float:
    """
    Placeholder function to calculate the minimum distance between a house 
    and a list of infrastructure hubs (e.g., Metro stations) using the Haversine formula.
    Prepared for the 3:00 PM Technical Pivot.
    
    Args:
        house_lat (float): Latitude of the house.
        house_lon (float): Longitude of the house.
        hub_coords (list of tuples): List of (lat, lon) for infrastructure hubs.
        
    Returns:
        float: Minimum distance in km to the nearest hub.
    """
    import math
    
    def haversine(lat1, lon1, lat2, lon2):
        # Earth radius in kilometers
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2)**2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
        
    if not hub_coords:
        return 0.0
        
    distances = [haversine(house_lat, house_lon, hub_lat, hub_lon) for hub_lat, hub_lon in hub_coords]
    return min(distances)
