"""
Data loading functions for the AI Market Intelligence system.
Handles loading raw data from various sources including CSV files and APIs.
"""

import pandas as pd
import os
from typing import Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_google_play_data(file_path: str) -> pd.DataFrame:
    """
    Load Google Play Store data from CSV file.
    
    Args:
        file_path (str): Path to the googleplaystore.csv file
        
    Returns:
        pd.DataFrame: Raw Google Play Store data
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        pd.errors.EmptyDataError: If the CSV file is empty
        Exception: For other data loading errors
    """
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Google Play Store data file not found: {file_path}")
        
        # Load the CSV with error handling for encoding issues
        logger.info(f"Loading Google Play Store data from: {file_path}")
        
        # Try multiple encodings as CSV files can have encoding issues
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully loaded data with {encoding} encoding")
                break
            except UnicodeDecodeError:
                logger.warning(f"Failed to load with {encoding} encoding, trying next...")
                continue
        
        if df is None:
            raise Exception("Failed to load CSV with any supported encoding")
        
        # Basic validation
        if df.empty:
            raise pd.errors.EmptyDataError("The CSV file is empty")
        
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading Google Play Store data: {e}")
        raise


def load_d2c_dataset(file_path: str) -> pd.DataFrame:
    """
    Load D2C e-commerce dataset from CSV file.
    
    Args:
        file_path (str): Path to the d2c_dataset.csv file
        
    Returns:
        pd.DataFrame: Raw D2C dataset
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        pd.errors.EmptyDataError: If the CSV file is empty
        Exception: For other data loading errors
    """
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"D2C dataset file not found: {file_path}")
        
        logger.info(f"Loading D2C dataset from: {file_path}")
        
        # Load the CSV with error handling
        df = pd.read_csv(file_path)
        
        # Basic validation
        if df.empty:
            raise pd.errors.EmptyDataError("The CSV file is empty")
        
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading D2C dataset: {e}")
        raise


def get_data_info(df: pd.DataFrame, dataset_name: str = "Dataset") -> dict:
    """
    Get comprehensive information about a dataset.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
        dataset_name (str): Name of the dataset for logging
        
    Returns:
        dict: Dictionary containing dataset information
    """
    info = {
        'name': dataset_name,
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'duplicates': df.duplicated().sum()
    }
    
    logger.info(f"{dataset_name} Info:")
    logger.info(f"  Shape: {info['shape']}")
    logger.info(f"  Missing values: {sum(info['missing_values'].values())} total")
    logger.info(f"  Duplicates: {info['duplicates']}")
    logger.info(f"  Memory usage: {info['memory_usage'] / (1024*1024):.2f} MB")
    
    return info
