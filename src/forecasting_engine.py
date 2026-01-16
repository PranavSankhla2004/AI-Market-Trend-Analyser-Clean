"""
Hybrid Forecasting Engine
Time series forecasting with Prophet
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class HybridForecaster:
    """Forecasting with Prophet"""
    
    def __init__(self):
        self.prophet_model = None
        self.forecast_df = None
    
    def train_prophet(self, df, target_col='revenue', date_col='date', periods=365):
        """Train Prophet model for time series forecasting"""
        import os
        os.environ['STAN_BACKEND'] = 'CMDSTANPY'
        
        # Prepare data for Prophet
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        # Initialize and fit Prophet with cmdstanpy backend
        try:
            self.prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95,
            )
        except:
            # Fallback if cmdstanpy not available
            self.prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95,
            )

        
        self.prophet_model.fit(prophet_df)
        
        # Create future dataframe
        future = self.prophet_model.make_future_dataframe(periods=periods)
        self.forecast_df = self.prophet_model.predict(future)
        
        logger.info(f"Prophet model trained. Forecast for {periods} periods")
        return self.forecast_df
    
    def get_forecast_summary(self, periods_ahead=30):
        """Get forecast summary for next N periods"""
        if self.forecast_df is None:
            logger.error("Model not trained yet")
            return None
        
        forecast = self.forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_ahead)
        forecast.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']
        
        return forecast
    
    def calculate_forecast_metrics(self, df, actual_col='revenue', date_col='date'):
        """Calculate forecast accuracy metrics"""
        if self.forecast_df is None:
            logger.error("Model not trained yet")
            return None
        
        # Merge forecast with actual values
        actual = df[[date_col, actual_col]].copy()
        actual['ds'] = pd.to_datetime(actual[date_col])
        
        comparison = pd.merge(
            self.forecast_df[['ds', 'yhat']], 
            actual[['ds', actual_col]], 
            on='ds', 
            how='inner'
        )
        
        if len(comparison) == 0:
            logger.error("No matching dates for comparison")
            return None
        
        mae = mean_absolute_error(comparison[actual_col], comparison['yhat'])
        rmse = np.sqrt(mean_squared_error(comparison[actual_col], comparison['yhat']))
        mape = np.mean(np.abs((comparison[actual_col] - comparison['yhat']) / 
                              comparison[actual_col])) * 100
        r2 = r2_score(comparison[actual_col], comparison['yhat'])
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
        
        logger.info(f"Forecast Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        return metrics


class AnomalyDetector:
    """Detect anomalies using statistical methods"""
    
    @staticmethod
    def detect_anomalies(df, target_col, threshold=2.5):
        """Detect anomalies using z-score method"""
        df = df.copy()
        mean = df[target_col].mean()
        std = df[target_col].std()
        
        df['z_score'] = np.abs((df[target_col] - mean) / std)
        df['anomaly'] = df['z_score'] > threshold
        
        anomalies = df[df['anomaly']].copy()
        logger.info(f"Detected {len(anomalies)} anomalies ({len(anomalies)/len(df)*100:.2f}%)")
        
        return df, anomalies


if __name__ == "__main__":
    print("Forecasting engine ready!")
