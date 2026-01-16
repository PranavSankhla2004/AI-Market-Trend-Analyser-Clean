"""
Data Collection Module
Handles API calls, CSV loading, and data validation
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """Collects market data from multiple sources"""
    
    def __init__(self):
        pass
        
    def load_csv_data(self, filepath):
        """Load data from CSV files"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def create_synthetic_sales_data(self, n_records=1000, n_products=20):
        """Generate synthetic sales data for demonstration"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2023-01-01', periods=n_records, freq='D')
        products = [f"Product_{i}" for i in range(1, n_products + 1)]
        
        data = {
            'date': np.repeat(dates, n_products),
            'product': products * len(dates),
            'quantity_sold': np.random.poisson(lam=10, size=n_records * n_products),
            'price': np.random.uniform(50, 500, size=n_records * n_products),
            'customer_age': np.random.randint(18, 65, size=n_records * n_products),
            'customer_segment': np.random.choice(['Budget', 'Mid-Range', 'Premium'], 
                                                  size=n_records * n_products),
            'season': np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], 
                                       size=n_records * n_products),
        }
        
        df = pd.DataFrame(data)
        df['revenue'] = df['quantity_sold'] * df['price']
        logger.info(f"Generated synthetic sales data with {len(df)} records")
        return df
    
    def validate_data(self, df):
        """Validate data quality"""
        logger.info("Validating data...")
        
        # Check missing values
        missing_pct = (df.isnull().sum() / len(df) * 100)
        if (missing_pct > 30).any():
            logger.warning(f"High missing values detected:\n{missing_pct}")
        
        # Check duplicates
        duplicates = df.duplicated().sum()
        logger.info(f"Duplicate records: {duplicates}")
        
        return df


if __name__ == "__main__":
    collector = DataCollector()
    sales_data = collector.create_synthetic_sales_data()
    sales_data.to_csv('data/raw/sales_data.csv', index=False)
    print("✅ Data generated successfully!")
