import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(df):
    """
    Load and clean the combined dataset.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The combined DataFrame containing mobile phone data
        
    Returns:
    --------
    pandas DataFrame
        Cleaned DataFrame
    """
    # Make a copy of the DataFrame
    df_clean = df.copy()
    
    # Drop duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Handle missing values
    df_clean['Rating'] = pd.to_numeric(df_clean['Rating'], errors='coerce')
    df_clean['Selling Price'] = pd.to_numeric(df_clean['Selling Price'], errors='coerce')
    df_clean['Original Price'] = pd.to_numeric(df_clean['Original Price'], errors='coerce')
    
    # Drop rows with missing Selling Price or Rating
    df_clean = df_clean.dropna(subset=['Selling Price'])
    
    # Fill missing ratings with median
    df_clean['Rating'].fillna(df_clean['Rating'].median(), inplace=True)
    
    # Fill missing Original Price with Selling Price
    mask = df_clean['Original Price'].isna()
    df_clean.loc[mask, 'Original Price'] = df_clean.loc[mask, 'Selling Price']
    
    # Extract numeric values from Memory and Storage
    df_clean['Memory'] = df_clean['Memory'].str.extract(r'(\d+)').astype(float)
    df_clean['Storage'] = df_clean['Storage'].str.extract(r'(\d+)').astype(float)
    
    # Create a discount percentage column
    df_clean['Discount %'] = ((df_clean['Original Price'] - df_clean['Selling Price']) / 
                             df_clean['Original Price'] * 100).round(2)
    
    # Replace negative discounts with 0
    df_clean['Discount %'] = df_clean['Discount %'].clip(lower=0)
    
    return df_clean

def preprocess_data(df):
    """
    Perform data preprocessing for analysis and modeling.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The cleaned DataFrame
        
    Returns:
    --------
    pandas DataFrame
        Preprocessed DataFrame ready for analysis
    """
    # Make a copy of the DataFrame
    df_processed = df.copy()
    
    # Create a price category column
    price_bins = [0, 10000, 20000, 30000, 40000, float('inf')]
    price_labels = ['Budget', 'Entry', 'Mid-range', 'Premium', 'Flagship']
    df_processed['Price Category'] = pd.cut(df_processed['Selling Price'], 
                                           bins=price_bins, 
                                           labels=price_labels)
    
    # Create a memory category column
    memory_bins = [0, 2, 4, 6, float('inf')]
    memory_labels = ['Low', 'Medium', 'High', 'Ultra']
    df_processed['Memory Category'] = pd.cut(df_processed['Memory'], 
                                            bins=memory_bins, 
                                            labels=memory_labels)
    
    # Create a storage category column
    storage_bins = [0, 32, 64, 128, float('inf')]
    storage_labels = ['Low', 'Medium', 'High', 'Ultra']
    df_processed['Storage Category'] = pd.cut(df_processed['Storage'], 
                                             bins=storage_bins, 
                                             labels=storage_labels)
    
    # Create a combined specs column
    df_processed['Specs'] = df_processed['Memory'].astype(str) + 'GB RAM, ' + df_processed['Storage'].astype(str) + 'GB Storage'
    
    # Count models by brand
    brand_counts = df_processed['Brand'].value_counts()
    major_brands = brand_counts[brand_counts >= 20].index.tolist()
    
    # Create a brand category column (Major brand or Other)
    df_processed['Brand Category'] = df_processed['Brand'].apply(lambda x: x if x in major_brands else 'Other')
    
    return df_processed
