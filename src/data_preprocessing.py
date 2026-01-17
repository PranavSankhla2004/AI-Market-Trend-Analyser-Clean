"""
Data Preprocessing Module
Cleaning, feature engineering, and transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data cleaning and feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def handle_missing_values(self, df, strategy='forward_fill'):
        """Handle missing values using specified strategy"""
        if strategy == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'drop':
            df = df.dropna()
        
        logger.info(f"Missing values handled using {strategy}")
        return df
    
    def remove_outliers(self, df, columns, method='iqr', threshold=1.5):
        """Remove outliers using IQR method"""
        df = df.copy()
        
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        
        logger.info(f"Outliers removed using {method}")
        return df
    
    def create_temporal_features(self, df, date_column):
        """Create time-based features from date column"""
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['dayofweek'] = df[date_column].dt.dayofweek
        df['quarter'] = df[date_column].dt.quarter
        df['weekofyear'] = df[date_column].dt.isocalendar().week
        
        # Cyclical encoding for seasonal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        logger.info("Temporal features created")
        return df
    
    def create_lag_features(self, df, target_col, lags=[1, 7, 30]):
        """Create lag features for time series"""
        df = df.copy()
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        logger.info(f"Lag features created with lags: {lags}")
        return df.dropna()
    
    def calculate_rfm_scores(self, df, date_col, customer_col, value_col):
        """Calculate RFM (Recency, Frequency, Monetary) scores safely"""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        ref_date = df[date_col].max() + pd.Timedelta(days=1)
        
        rfm = df.groupby(customer_col, as_index=False).agg({
            date_col: lambda x: (ref_date - x.max()).days,  # recency
            value_col: 'sum',                                # monetary
            customer_col: 'count'                            # frequency
        })
        
        rfm.rename(columns={
            date_col: 'recency',
            customer_col: 'frequency',
            value_col: 'monetary'
        }, inplace=True)
        
        # Calculate scores safely
        for col in ['recency', 'frequency', 'monetary']:
            rfm[f'{col[0]}_score'] = pd.qcut(
                rfm[col].rank(method='first'), 5, labels=False, duplicates='drop'
            )
            # Scale to 1-5
            rfm[f'{col[0]}_score'] = ((rfm[f'{col[0]}_score'] / rfm[f'{col[0]}_score'].max()) * 4 + 1).round().astype(int)
        
        rfm['rfm_score'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']
        
        logger.info("RFM scores calculated successfully")
        return rfm


    
    def normalize_features(self, df, columns, method='standard'):
        """Normalize numerical features"""
        df = df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        
        df[columns] = scaler.fit_transform(df[columns])
        logger.info(f"Features normalized using {method} scaling")
        
        return df, scaler
    
    def encode_categorical(self, df, columns):
        """One-hot encode categorical features"""
        df = pd.get_dummies(df, columns=columns, drop_first=True)
        logger.info(f"Categorical features encoded: {columns}")
        return df


if __name__ == "__main__":
    print("Data Preprocessing module ready!")
