"""
App Store API client for fetching app reviews and data via RapidAPI.
Includes comprehensive error handling, retry mechanisms, and rate limiting.
"""

import requests
import time
import os
from typing import Dict, List, Optional, Union
import logging
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppStoreClient:
    """
    App Store API client with robust error handling and retry mechanisms.
    
    This client interfaces with RapidAPI to fetch app store data including
    reviews, ratings, and app metadata. It includes comprehensive error handling,
    exponential backoff, and rate limiting to handle API constraints gracefully.
    """
    
    def __init__(self):
        """Initialize the App Store API client."""
        # Load environment variables
        load_dotenv()
        
        self.api_key = os.getenv('RAPIDAPI_KEY')
        if not self.api_key:
            raise ValueError("RAPIDAPI_KEY not found in environment variables")
        
        # API configuration (updated to match working example)
        self.base_url = "https://appstore-scrapper-api.p.rapidapi.com"
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "appstore-scrapper-api.p.rapidapi.com"
        }
        
        # Rate limiting and retry configuration
        self.max_retries = 3
        self.base_delay = 1.0  # Base delay in seconds
        self.max_delay = 60.0  # Maximum delay in seconds
        self.timeout = 30  # Request timeout in seconds
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        logger.info("AppStoreClient initialized successfully")
    
    def _exponential_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.
        
        Args:
            attempt (int): Current attempt number (0-indexed)
            
        Returns:
            float: Delay in seconds
        """
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make a robust API request with retry logic.
        
        Args:
            endpoint (str): API endpoint to call
            params (Dict): Query parameters
            
        Returns:
            Optional[Dict]: API response data or None if failed
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making request to {endpoint}, attempt {attempt + 1}")
                
                response = self.session.get(
                    url,
                    params=params or {},
                    timeout=self.timeout
                )
                
                # Handle different HTTP status codes
                if response.status_code == 200:
                    return response.json()
                
                elif response.status_code == 429:  # Rate limited
                    delay = self._exponential_backoff(attempt)
                    logger.warning(f"Rate limited. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                
                elif response.status_code == 404:
                    logger.error(f"App not found (404): {endpoint}")
                    return None
                
                elif response.status_code >= 500:  # Server errors
                    delay = self._exponential_backoff(attempt)
                    logger.warning(f"Server error {response.status_code}. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                
                else:
                    logger.error(f"Unexpected status code {response.status_code}: {response.text}")
                    return None
                    
            except requests.exceptions.Timeout:
                delay = self._exponential_backoff(attempt)
                logger.warning(f"Request timeout. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
                
            except requests.exceptions.ConnectionError:
                delay = self._exponential_backoff(attempt)
                logger.warning(f"Connection error. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request exception: {e}")
                return None
                
            except json.JSONDecodeError:
                logger.error("Invalid JSON response")
                return None
        
        logger.error(f"Failed to get response after {self.max_retries} attempts")
        return None
    
    def get_reviews(self, app_id: str, country: str = 'us', page: int = 1, 
                   sort: str = 'mostRecent', lang: str = 'en') -> Optional[List[Dict]]:
        """
        Fetch app reviews from the App Store.
        
        Args:
            app_id (str): App Store app ID
            country (str): Country code (default: 'us')
            page (int): Page number for pagination (default: 1)
            sort (str): Sort order (default: 'mostRecent')
            lang (str): Language code (default: 'en')
            
        Returns:
            Optional[List[Dict]]: List of reviews or None if failed
        """
        logger.info(f"Fetching reviews for app ID: {app_id}, country: {country}, page: {page}")
        
        # Validate inputs
        if not app_id or not app_id.isdigit():
            logger.error(f"Invalid app_id: {app_id}. Must be a numeric string.")
            return None
        
        params = {
            'id': app_id,
            'sort': sort,
            'page': str(page),
            'contry': country,  # Note: API uses 'contry' (typo in their API)
            'lang': lang
        }
        
        data = self._make_request('/v1/app-store-api/reviews', params)
        
        if data and isinstance(data, list):
            logger.info(f"Successfully fetched {len(data)} reviews")
            return data
        else:
            logger.warning(f"Failed to fetch reviews for app ID: {app_id}")
            return None
    
    def search_apps(self, term: str, country: str = 'us', limit: int = 50) -> Optional[Dict]:
        """
        Search for apps in the App Store.
        
        Args:
            term (str): Search term
            country (str): Country code (default: 'us')
            limit (int): Maximum number of results (default: 50)
            
        Returns:
            Optional[Dict]: Search results or None if failed
        """
        logger.info(f"Searching for apps with term: '{term}'")
        
        if not term or len(term.strip()) < 2:
            logger.error("Search term must be at least 2 characters long")
            return None
        
        params = {
            'term': term.strip(),
            'country': country,
            'limit': str(limit)
        }
        
        data = self._make_request('/search', params)
        
        if data:
            results_count = len(data.get('results', []))
            logger.info(f"Found {results_count} apps for search term: '{term}'")
        else:
            logger.warning(f"Failed to search for term: '{term}'")
        
        return data
    
    def create_app_store_dataframe(self, reviews_data: List[Dict], app_id: str = '') -> pd.DataFrame:
        """
        Convert API review data to a structured DataFrame.
        
        Args:
            reviews_data (List[Dict]): Raw review data from API (direct list of reviews)
            app_id (str): App ID for the reviews
            
        Returns:
            pd.DataFrame: Structured review data
        """
        logger.info("Converting API data to DataFrame")
        
        app_store_data = []
        
        # Handle new API format - direct list of reviews
        for review in reviews_data:
            row = {
                'app_id': app_id,
                'review_id': review.get('id', ''),
                'app_name': '',  # Will need to be filled from app details
                'rating': review.get('score', 0),
                'review_text': review.get('text', ''),
                'review_title': review.get('title', ''),
                'author': review.get('userName', ''),
                'author_url': review.get('userUrl', ''),
                'date': review.get('updated', ''),
                'version': review.get('version', ''),
                'review_url': review.get('url', ''),
                'country': 'us'  # Default, can be updated based on request
            }
            app_store_data.append(row)
        
        df = pd.DataFrame(app_store_data)
        
        # Clean and standardize data types
        if not df.empty:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Add derived fields
            df['review_length'] = df['review_text'].str.len().fillna(0)
            df['source'] = 'app_store'
            
            # Clean author names (remove special characters if needed)
            df['author'] = df['author'].astype(str).str.strip()
        
        logger.info(f"Created DataFrame with {len(df)} review records")
        return df
    
    def test_connection(self) -> bool:
        """
        Test the API connection and authentication.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        logger.info("Testing API connection...")
        
        try:
            # Try to get reviews for Apple Books app (ID: 364709193) as a test
            test_data = self.get_reviews("364709193", page=1)
            
            if test_data and isinstance(test_data, list) and len(test_data) > 0:
                logger.info("✅ API connection test successful")
                return True
            else:
                logger.error("❌ API connection test failed - no data returned")
                return False
                
        except Exception as e:
            logger.error(f"❌ API connection test failed: {e}")
            return False
    
    def close(self):
        """Close the session and cleanup resources."""
        if hasattr(self, 'session'):
            self.session.close()
            logger.info("AppStoreClient session closed")