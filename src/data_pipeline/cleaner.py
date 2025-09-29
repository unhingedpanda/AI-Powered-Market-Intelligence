"""
Data cleaning and normalization functions for the AI Market Intelligence system.
Handles cleaning and standardizing data from various sources.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_google_play_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize Google Play Store data.
    
    This function performs comprehensive data cleaning including:
    - Handling missing ratings with category median
    - Standardizing 'Installs' to numeric values
    - Converting 'Size' to numeric MB values
    - Cleaning 'Price' to numeric values
    - Correcting data types
    - Removing duplicates
    
    Args:
        df (pd.DataFrame): Raw Google Play Store data
        
    Returns:
        pd.DataFrame: Cleaned and standardized data
    """
    logger.info("Starting Google Play Store data cleaning...")
    
    # Create a copy to avoid modifying the original data
    df_clean = df.copy()
    
    # Log initial state
    logger.info(f"Initial data shape: {df_clean.shape}")
    logger.info(f"Initial missing values: {df_clean.isnull().sum().sum()}")
    
    # 1. Handle Missing Ratings - Fill with category median
    logger.info("Cleaning Rating column...")
    if 'Rating' in df_clean.columns:
        # First, ensure Rating is numeric (some might be strings)
        df_clean['Rating'] = pd.to_numeric(df_clean['Rating'], errors='coerce')
        
        # Calculate median rating by category
        category_median_rating = df_clean.groupby('Category')['Rating'].median()
        
        # Fill missing ratings with category median
        for category in df_clean['Category'].unique():
            if pd.notna(category):
                mask = (df_clean['Category'] == category) & (df_clean['Rating'].isna())
                if mask.any():
                    median_rating = category_median_rating.get(category, df_clean['Rating'].median())
                    df_clean.loc[mask, 'Rating'] = median_rating
        
        # Fill any remaining NaNs with overall median
        df_clean['Rating'].fillna(df_clean['Rating'].median(), inplace=True)
        
        logger.info(f"Rating column cleaned. Remaining missing values: {df_clean['Rating'].isnull().sum()}")
    
    # 2. Standardize Installs column
    logger.info("Cleaning Installs column...")
    if 'Installs' in df_clean.columns:
        def clean_installs(install_str):
            if pd.isna(install_str):
                return 0
            
            # Remove commas and plus signs, convert to string first
            install_str = str(install_str).replace(',', '').replace('+', '').replace('Free', '0')
            
            # Extract numeric part
            try:
                return int(install_str)
            except ValueError:
                # Handle edge cases like "Varies with device"
                return 0
        
        df_clean['Installs'] = df_clean['Installs'].apply(clean_installs)
        logger.info(f"Installs column cleaned. Sample values: {df_clean['Installs'].head().tolist()}")
    
    # 3. Standardize Size column (convert to MB)
    logger.info("Cleaning Size column...")
    if 'Size' in df_clean.columns:
        def clean_size(size_str):
            if pd.isna(size_str):
                return np.nan
            
            size_str = str(size_str).strip()
            
            # Handle special cases
            if size_str in ['Varies with device', 'varies with device']:
                return np.nan
            
            # Extract numeric part and unit
            if 'M' in size_str:
                try:
                    return float(size_str.replace('M', ''))
                except ValueError:
                    return np.nan
            elif 'k' in size_str or 'K' in size_str:
                try:
                    # Convert KB to MB
                    kb_value = float(size_str.replace('k', '').replace('K', ''))
                    return kb_value / 1024
                except ValueError:
                    return np.nan
            else:
                # Assume it's already in MB if no unit specified
                try:
                    return float(size_str)
                except ValueError:
                    return np.nan
        
        df_clean['Size_MB'] = df_clean['Size'].apply(clean_size)
        
        # Fill missing sizes with median size
        median_size = df_clean['Size_MB'].median()
        df_clean['Size_MB'].fillna(median_size, inplace=True)
        
        # Drop the original Size column
        df_clean = df_clean.drop('Size', axis=1)
        
        logger.info(f"Size column cleaned and converted to MB. Sample values: {df_clean['Size_MB'].head().tolist()}")
    
    # 4. Clean Price column
    logger.info("Cleaning Price column...")
    if 'Price' in df_clean.columns:
        def clean_price(price_str):
            if pd.isna(price_str):
                return 0.0
            
            price_str = str(price_str).strip()
            
            # Handle 'Free' apps
            if price_str.lower() == 'free' or price_str == '0':
                return 0.0
            
            # Remove dollar sign and convert to float
            price_str = price_str.replace('$', '')
            try:
                return float(price_str)
            except ValueError:
                return 0.0
        
        df_clean['Price'] = df_clean['Price'].apply(clean_price)
        logger.info(f"Price column cleaned. Sample values: {df_clean['Price'].head().tolist()}")
    
    # 5. Correct Data Types
    logger.info("Correcting data types...")
    
    # Reviews should be integer
    if 'Reviews' in df_clean.columns:
        df_clean['Reviews'] = pd.to_numeric(df_clean['Reviews'], errors='coerce').fillna(0).astype(int)
    
    # Last Updated should be datetime
    if 'Last Updated' in df_clean.columns:
        df_clean['Last Updated'] = pd.to_datetime(df_clean['Last Updated'], errors='coerce')
    
    # Ensure App names are strings
    if 'App' in df_clean.columns:
        df_clean['App'] = df_clean['App'].astype(str)
    
    # 6. Remove Duplicates - Keep the one with most reviews
    logger.info("Removing duplicates...")
    initial_count = len(df_clean)
    
    if 'App' in df_clean.columns and 'Reviews' in df_clean.columns:
        # Sort by reviews in descending order and keep first occurrence (highest reviews)
        df_clean = df_clean.sort_values('Reviews', ascending=False)
        df_clean = df_clean.drop_duplicates(subset=['App'], keep='first')
        df_clean = df_clean.sort_index()  # Restore original order
    
    final_count = len(df_clean)
    duplicates_removed = initial_count - final_count
    logger.info(f"Removed {duplicates_removed} duplicate apps")
    
    # 7. Additional Data Quality Checks
    logger.info("Performing additional data quality checks...")
    
    # Remove rows with invalid ratings (outside 0-5 range)
    if 'Rating' in df_clean.columns:
        invalid_ratings = (df_clean['Rating'] < 0) | (df_clean['Rating'] > 5)
        df_clean = df_clean[~invalid_ratings]
        if invalid_ratings.sum() > 0:
            logger.info(f"Removed {invalid_ratings.sum()} rows with invalid ratings")
    
    # Remove rows with negative reviews
    if 'Reviews' in df_clean.columns:
        negative_reviews = df_clean['Reviews'] < 0
        df_clean = df_clean[~negative_reviews]
        if negative_reviews.sum() > 0:
            logger.info(f"Removed {negative_reviews.sum()} rows with negative reviews")
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    # Log final state
    logger.info("Google Play Store data cleaning completed!")
    logger.info(f"Final data shape: {df_clean.shape}")
    logger.info(f"Final missing values: {df_clean.isnull().sum().sum()}")
    logger.info(f"Data reduction: {((len(df) - len(df_clean)) / len(df) * 100):.2f}%")
    
    return df_clean


def clean_d2c_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize D2C e-commerce dataset.
    
    Args:
        df (pd.DataFrame): Raw D2C dataset
        
    Returns:
        pd.DataFrame: Cleaned D2C data
    """
    logger.info("Starting D2C data cleaning...")
    
    df_clean = df.copy()
    
    # Log initial state
    logger.info(f"Initial D2C data shape: {df_clean.shape}")
    
    # Clean numeric columns
    numeric_columns = ['spend_usd', 'impressions', 'clicks', 'first_purchase', 'revenue_usd']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

    # Calculate derived metrics
    if all(col in df_clean.columns for col in ['spend_usd', 'first_purchase']):
        # Customer Acquisition Cost (CAC)
        df_clean['cac'] = df_clean['spend_usd'] / df_clean['first_purchase'].replace(0, 1)
        df_clean['cac'] = df_clean['cac'].replace([np.inf, -np.inf], 0)

    if all(col in df_clean.columns for col in ['revenue_usd', 'spend_usd']):
        # Return on Ad Spend (ROAS)
        df_clean['roas'] = df_clean['revenue_usd'] / df_clean['spend_usd'].replace(0, 1)
        df_clean['roas'] = df_clean['roas'].replace([np.inf, -np.inf], 0)

    if all(col in df_clean.columns for col in ['clicks', 'impressions']):
        # Click-through Rate (CTR) - already calculated in initial cleaning
        if 'ctr' not in df_clean.columns:
            df_clean['ctr'] = (df_clean['clicks'] / df_clean['impressions'].replace(0, 1)) * 100
        df_clean['ctr'] = df_clean['ctr'].replace([np.inf, -np.inf], 0)
    
    if all(col in df_clean.columns for col in ['conversions', 'clicks']):
        # Conversion Rate
        df_clean['conversion_rate'] = (df_clean['conversions'] / df_clean['clicks'].replace(0, 1)) * 100
        df_clean['conversion_rate'] = df_clean['conversion_rate'].replace([np.inf, -np.inf], 0)
    
    logger.info(f"D2C data cleaning completed. Final shape: {df_clean.shape}")
    return df_clean


def create_sample_app_store_data() -> pd.DataFrame:
    """
    Create sample App Store data for popular apps to demonstrate unified functionality.
    In a real implementation, this would fetch live data from the App Store API.

    Returns:
        pd.DataFrame: Sample App Store data
    """
    logger.info("Creating sample App Store data for unified dataset...")

    # Sample popular apps with their App Store IDs
    sample_apps = [
        {"app_name": "Instagram", "app_store_id": "389801252", "category": "SOCIAL_NETWORKING"},
        {"app_name": "YouTube", "app_store_id": "544007664", "category": "ENTERTAINMENT"},
        {"app_name": "WhatsApp", "app_store_id": "310633997", "category": "SOCIAL_NETWORKING"},
        {"app_name": "TikTok", "app_store_id": "835599320", "category": "ENTERTAINMENT"},
        {"app_name": "Facebook", "app_store_id": "284882215", "category": "SOCIAL_NETWORKING"},
        {"app_name": "Netflix", "app_store_id": "363590051", "category": "ENTERTAINMENT"},
        {"app_name": "Spotify", "app_store_id": "324684580", "category": "MUSIC"},
        {"app_name": "Amazon", "app_store_id": "297606951", "category": "SHOPPING"},
        {"app_name": "Uber", "app_store_id": "368677368", "category": "TRAVEL"},
        {"app_name": "Airbnb", "app_store_id": "401626263", "category": "TRAVEL"}
    ]

    app_store_data = []

    for app in sample_apps:
        # Create realistic sample data based on the app
        base_rating = 4.2 + (0.3 * (hash(app["app_name"]) % 10) / 10)  # 4.2-4.5 range
        base_reviews = 100000 + (hash(app["app_name"]) % 500000)  # 100k-600k range

        app_store_row = {
            'app_name': app['app_name'],
            'category': app['category'],
            'rating': round(base_rating, 1),
            'review_count': base_reviews,
            'review_text': f'Sample reviews for {app["app_name"]} from App Store',
            'size_mb': 150.0 + (hash(app["app_name"]) % 100),  # 150-250 MB range
            'price': 0.0,  # Most popular apps are free
            'content_rating': '17+' if app['category'] == 'SOCIAL_NETWORKING' else '12+',
            'last_updated': pd.Timestamp.now() - pd.Timedelta(days=30 + (hash(app["app_name"]) % 60))
        }
        app_store_data.append(app_store_row)

    app_store_df = pd.DataFrame(app_store_data)

    logger.info(f"Created sample App Store data with {len(app_store_df)} records")
    logger.info(f"Categories: {app_store_df['category'].value_counts().to_dict()}")

    return app_store_df


def create_unified_schema(google_play_df: pd.DataFrame, app_store_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Create a unified schema combining Google Play and App Store data.

    Args:
        google_play_df (pd.DataFrame): Cleaned Google Play Store data
        app_store_df (pd.DataFrame, optional): App Store data from API

    Returns:
        pd.DataFrame: Unified dataset with consistent schema
    """
    logger.info("Creating unified schema...")

    unified_data = []

    # Process Google Play data
    for _, row in google_play_df.iterrows():
        unified_row = {
            'app_name': row.get('App', ''),
            'category': row.get('Category', ''),
            'rating': row.get('Rating', 0.0),
            'reviews': row.get('Reviews', 0),
            'source': 'google_play',
            'review_text': '',  # Will be populated from API data
            'size_mb': row.get('Size_MB', 0.0),
            'price': row.get('Price', 0.0),
            'installs': row.get('Installs', 0),
            'content_rating': row.get('Content Rating', ''),
            'last_updated': row.get('Last Updated', pd.NaT)
        }
        unified_data.append(unified_row)

    # Process App Store data if provided, otherwise create sample data
    if app_store_df is not None and len(app_store_df) > 0:
        for _, row in app_store_df.iterrows():
            unified_row = {
                'app_name': row.get('app_name', ''),
                'category': row.get('category', ''),
                'rating': row.get('rating', 0.0),
                'reviews': row.get('review_count', 0),
                'source': 'app_store',
                'review_text': row.get('review_text', ''),
                'size_mb': row.get('size_mb', 0.0),
                'price': row.get('price', 0.0),
                'installs': 0,  # Not available in App Store API
                'content_rating': row.get('content_rating', ''),
                'last_updated': row.get('last_updated', pd.NaT)
            }
            unified_data.append(unified_row)
    else:
        # Create sample App Store data for demonstration
        logger.info("No App Store data provided, creating sample data for demonstration...")
        sample_app_store_df = create_sample_app_store_data()

        for _, row in sample_app_store_df.iterrows():
            unified_row = {
                'app_name': row.get('app_name', ''),
                'category': row.get('category', ''),
                'rating': row.get('rating', 0.0),
                'reviews': row.get('review_count', 0),
                'source': 'app_store',
                'review_text': row.get('review_text', ''),
                'size_mb': row.get('size_mb', 0.0),
                'price': row.get('price', 0.0),
                'installs': 0,  # Not available in App Store API
                'content_rating': row.get('content_rating', ''),
                'last_updated': row.get('last_updated', pd.NaT)
            }
            unified_data.append(unified_row)

    unified_df = pd.DataFrame(unified_data)

    # Ensure proper data types
    unified_df['rating'] = pd.to_numeric(unified_df['rating'], errors='coerce').fillna(0.0)
    unified_df['reviews'] = pd.to_numeric(unified_df['reviews'], errors='coerce').fillna(0).astype(int)
    unified_df['size_mb'] = pd.to_numeric(unified_df['size_mb'], errors='coerce').fillna(0.0)
    unified_df['price'] = pd.to_numeric(unified_df['price'], errors='coerce').fillna(0.0)
    unified_df['installs'] = pd.to_numeric(unified_df['installs'], errors='coerce').fillna(0).astype(int)

    logger.info(f"Unified schema created with {len(unified_df)} total records")
    logger.info(f"Sources: {unified_df['source'].value_counts().to_dict()}")

    return unified_df


def validate_cleaned_data(df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
    """
    Validate the quality of cleaned data and generate a quality report.
    
    Args:
        df (pd.DataFrame): Cleaned dataset to validate
        dataset_name (str): Name of the dataset for logging
        
    Returns:
        Dict: Quality validation report
    """
    logger.info(f"Validating {dataset_name}...")
    
    validation_report = {
        'dataset_name': dataset_name,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values_count': df.isnull().sum().sum(),
        'missing_values_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    
    # Column-specific validations
    column_quality = {}
    
    for column in df.columns:
        col_info = {
            'missing_count': df[column].isnull().sum(),
            'missing_percentage': (df[column].isnull().sum() / len(df)) * 100,
            'unique_values': df[column].nunique(),
            'data_type': str(df[column].dtype)
        }
        
        # Specific validations based on column type
        if column in ['rating', 'Rating']:
            if df[column].dtype in ['float64', 'int64']:
                col_info['min_value'] = df[column].min()
                col_info['max_value'] = df[column].max()
                col_info['invalid_range'] = ((df[column] < 0) | (df[column] > 5)).sum()
        
        column_quality[column] = col_info
    
    validation_report['column_quality'] = column_quality
    
    # Overall quality score (0-100)
    quality_score = 100
    quality_score -= validation_report['missing_values_percentage'] * 2  # Penalize missing values
    quality_score -= (validation_report['duplicate_rows'] / validation_report['total_rows']) * 100 * 10  # Penalize duplicates
    quality_score = max(0, quality_score)  # Ensure non-negative
    
    validation_report['quality_score'] = round(quality_score, 2)
    
    logger.info(f"{dataset_name} validation completed")
    logger.info(f"Quality Score: {validation_report['quality_score']}/100")
    logger.info(f"Missing Values: {validation_report['missing_values_percentage']:.2f}%")
    logger.info(f"Duplicate Rows: {validation_report['duplicate_rows']}")
    
    return validation_report
