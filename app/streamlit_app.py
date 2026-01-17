"""
Main Streamlit Dashboard
Interactive visualization and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Include parent folder to import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection import DataCollector
from src.data_preprocessing import DataPreprocessor
from src.forecasting_engine import HybridForecaster, AnomalyDetector
from src.segmentation import CustomerSegmentation, ProductSegmentation

# Page configuration
st.set_page_config(
    page_title="Market Trend Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_data():
    """Load and preprocess synthetic sales data"""
    collector = DataCollector()
    df = collector.create_synthetic_sales_data(n_records=1000, n_products=20)
    
    preprocessor = DataPreprocessor()
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.create_temporal_features(df, 'date')
    
    return df

def main():
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("Select Page", [
        "Home",
        "Trend Analysis",
        "Demand Forecasting",
        "Customer Segmentation",
        "Anomaly Detection",
        "Recommendations"
    ])
    
    if page == "Home":
        show_home()
    elif page == "Trend Analysis":
        show_trend_analysis()
    elif page == "Demand Forecasting":
        show_forecasting()
    elif page == "Customer Segmentation":
        show_segmentation()
    elif page == "Anomaly Detection":
        show_anomalies()
    elif page == "Recommendations":
        show_recommendations()

def show_home():
    """Home page"""
    st.title("🎯 AI-Powered Market Trend Analysis")
    
    st.markdown("""
    ### Welcome to the Market Trend Analysis Platform
    
    This application leverages **AI & ML** to provide actionable insights into market trends, customer behavior, and pricing patterns.
    
    ---
    
    #### 🚀 Key Features
    
    **1. Trend Analysis** - Track product popularity over time  
    **2. Demand Forecasting** - Predict future sales with high accuracy  
    **3. Customer Segmentation** - Identify high-value customers  
    **4. Anomaly Detection** - Detect unusual market patterns  
    **5. Recommendations** - Get actionable business insights  
    """)
    
    df = load_data()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days} days")
    col3.metric("Products", df['product'].nunique())
    col4.metric("Total Revenue", f"₹{df['revenue'].sum():,.0f}")

def show_trend_analysis():
    """Trend analysis page"""
    st.title("📈 Product Trend Analysis")
    
    df = load_data()
    
    # Aggregate product metrics
    product_trends = df.groupby('product').agg({
        'quantity_sold': 'sum',
        'revenue': 'sum',
        'price': 'mean'
    }).sort_values('revenue', ascending=False)
    
    # Top products visualization
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Products by Revenue")
        fig = px.bar(product_trends.head(10), x='revenue', orientation='h', title="Top 10 Products")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Revenue Distribution")
        fig = px.pie(values=product_trends['revenue'], names=product_trends.index, title="Product Mix", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series trend
    st.subheader("Monthly Revenue Trend")
    monthly_revenue = df.groupby(df['date'].dt.to_period('M'))['revenue'].sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_revenue.index.astype(str), y=monthly_revenue.values, 
                             mode='lines+markers', name='Revenue', line=dict(color='#1f77b4', width=3)))
    fig.update_layout(title="Revenue Trend Over Time", xaxis_title="Month", yaxis_title="Revenue (₹)", hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

def show_forecasting():
    """Demand forecasting page"""
    st.title("🔮 Demand Forecasting")
    
    df = load_data()
    
    # Prepare aggregated daily revenue data
    forecast_data = df.groupby('date')['revenue'].sum().reset_index()
    
    # Ensure proper data types
    forecast_data['date'] = pd.to_datetime(forecast_data['date'], errors='coerce')
    forecast_data['revenue'] = pd.to_numeric(forecast_data['revenue'], errors='coerce')
    forecast_data.dropna(inplace=True)
    
    st.markdown("### Sales Forecast for Next 30 Days")
    
    # Debug: show data preview
    st.write("Forecast Data Preview (first 10 rows):")
    st.dataframe(forecast_data.head(10))
    
    # Check for sufficient data
    if len(forecast_data) < 2:
        st.error("Not enough data points to train the forecasting model.")
        return
    
    with st.spinner("Training forecasting model..."):
        forecaster = HybridForecaster()
        try:
            # Train Prophet model
            forecast_df = forecaster.train_prophet(
                forecast_data, target_col='revenue', date_col='date', periods=30
            )
            
            if forecast_df is None:
                st.error("Forecasting failed. Please check your data.")
                return
            
            # Get forecast summary and metrics
            summary = forecaster.get_forecast_summary(periods_ahead=30)
            metrics = forecaster.calculate_forecast_metrics(forecast_data)
        
        except Exception as e:
            st.error(f"Forecasting failed: {e}")
            return
    
    # Display metrics safely
    st.subheader("Forecast Accuracy Metrics")
    if metrics is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"₹{metrics.get('MAE', 0):,.0f}")
        col2.metric("RMSE", f"₹{metrics.get('RMSE', 0):,.0f}")
        col3.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%")
        col4.metric("R² Score", f"{metrics.get('R2', 0):.3f}")
    else:
        st.warning("Metrics could not be calculated.")
    
    # Forecast visualization
    st.subheader("Forecast Visualization")
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=forecast_data['date'], y=forecast_data['revenue'],
        mode='lines', name='Historical', line=dict(color='#1f77b4')
    ))
    
    # Forecast data
    if summary is not None:
        fig.add_trace(go.Scatter(
            x=summary['date'], y=summary['forecast'],
            mode='lines', name='Forecast', line=dict(color='#ff7f0e', dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=summary['date'], y=summary['upper_bound'],
            fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=summary['date'], y=summary['lower_bound'],
            fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
            name='95% Confidence', fillcolor='rgba(255,127,14,0.2)'
        ))
    
    fig.update_layout(
        title="Sales Forecast with Confidence Interval",
        xaxis_title="Date", yaxis_title="Revenue (₹)",
        hovermode='x unified', height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast summary table
    if summary is not None:
        st.subheader("Forecast Summary (Next 30 Days)")
        st.dataframe(summary.head(15))


def show_segmentation():
    """Customer segmentation page"""
    st.title("👥 Customer Segmentation")
    
    df = load_data()
    
    # Calculate RFM
    preprocessor = DataPreprocessor()
    rfm = preprocessor.calculate_rfm_scores(df, 'date', 'customer_age', 'revenue')
    
    # Segment customers
    with st.spinner("Segmenting customers..."):
        segmenter = CustomerSegmentation()
        df_segmented, metrics = segmenter.segment_customers(rfm)
    
    # Display metrics
    col1, col2 = st.columns(2)
    col1.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}")
    col2.metric("Davies-Bouldin Score", f"{metrics['davies_bouldin_score']:.3f}")
    
    # Segment distribution
    st.subheader("Segment Distribution")
    segment_dist = df_segmented['segment_name'].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(values=segment_dist.values, names=segment_dist.index, title="Customer Segments", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(segment_dist)

def show_anomalies():
    """Anomaly detection page"""
    st.title("⚠️ Anomaly Detection")
    
    df = load_data()
    
    daily_data = df.groupby('date')['revenue'].sum().reset_index()
    
    detector = AnomalyDetector()
    daily_data_with_anomalies, anomalies = detector.detect_anomalies(daily_data, 'revenue', threshold=2.5)
    
    st.metric("Anomalies Detected", len(anomalies))
    
    # Plot anomalies
    fig = go.Figure()
    normal = daily_data_with_anomalies[~daily_data_with_anomalies['anomaly']]
    fig.add_trace(go.Scatter(x=normal['date'], y=normal['revenue'], mode='markers', name='Normal', marker=dict(color='blue', size=5)))
    
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(x=anomalies['date'], y=anomalies['revenue'], mode='markers', name='Anomaly', marker=dict(color='red', size=10, symbol='star')))
    
    fig.update_layout(title="Revenue Anomalies Detection", xaxis_title="Date", yaxis_title="Revenue (₹)", height=500)
    st.plotly_chart(fig, use_container_width=True)

def show_recommendations():
    """Recommendations page"""
    st.title("💡 AI-Powered Recommendations")
    
    df = load_data()
    
    st.subheader("🛍️ Product Strategy")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""**High-Demand Products** 🚀
- Increase inventory
- Allocate more marketing
- Consider price increase (5-10%)""")
    with col2:
        st.warning("""**Low-Demand Products** 📉
- Promotions or discounts
- Phase out slow-moving items
- Bundle with popular products""")
    
    st.subheader("💰 Pricing Strategy")
    avg_price = df['price'].mean()
    st.metric("Average Price", f"₹{avg_price:.0f}")
    st.success("""**Pricing Recommendations:**
- Dynamic pricing during peak seasons
- Premium segment: +15% price tolerance
- Budget segment: Optimize for volume""")

if __name__ == "__main__":
    main()
