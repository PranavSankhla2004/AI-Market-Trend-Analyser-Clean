"""
Customer Segmentation Module
Clustering and behavioral analysis
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging

logger = logging.getLogger(__name__)


class CustomerSegmentation:
    """Segment customers using K-Means"""
    
    def __init__(self):
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
    
    def prepare_features(self, df, rfm_cols=['recency', 'frequency', 'monetary']):
        """Prepare and normalize features for clustering"""
        self.feature_cols = rfm_cols
        
        X = df[rfm_cols].copy()
        X = self.scaler.fit_transform(X)
        
        logger.info("Features prepared and normalized")
        return X
    
    def optimal_clusters(self, X, max_clusters=10):
        """Find optimal number of clusters using elbow method"""
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
        # Find optimal k
        optimal_k = list(k_range)[np.argmax(silhouette_scores)]
        logger.info(f"Optimal clusters: {optimal_k} (Silhouette: {max(silhouette_scores):.3f})")
        
        return optimal_k, silhouette_scores, inertias
    
    def fit_kmeans(self, X, n_clusters=3):
        """Fit K-Means clustering"""
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = self.kmeans_model.fit_predict(X)
        
        # Calculate metrics
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        metrics = {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'n_clusters': n_clusters
        }
        
        logger.info(f"K-Means fitted - Silhouette: {silhouette:.3f}")
        return labels, metrics
    
    def segment_customers(self, df, rfm_cols=['recency', 'frequency', 'monetary'], n_clusters=3):
        """Complete segmentation pipeline"""
        X = self.prepare_features(df, rfm_cols)
        labels, metrics = self.fit_kmeans(X, n_clusters)
        
        df_segmented = df.copy()
        df_segmented['segment'] = labels
        
        # Assign segment names
        segment_names = {0: 'At Risk', 1: 'Loyal', 2: 'Premium'}
        df_segmented['segment_name'] = df_segmented['segment'].map(segment_names)
        
        logger.info(f"Segmentation complete - {n_clusters} segments identified")
        return df_segmented, metrics
    
    def get_segment_profiles(self, df, segment_col='segment', rfm_cols=['recency', 'frequency', 'monetary']):
        """Generate segment profiles"""
        profiles = df.groupby(segment_col)[rfm_cols].agg(['mean', 'median', 'std'])
        
        logger.info("Segment profiles generated")
        return profiles


class ProductSegmentation:
    """Segment products based on sales patterns"""
    
    @staticmethod
    def abc_analysis(df, product_col, revenue_col, date_col=None):
        """ABC Analysis - classify products by sales contribution"""
        df = df.copy()
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            recent_date = df[date_col].max() - pd.Timedelta(days=90)
            df = df[df[date_col] >= recent_date]
        
        # Calculate revenue per product
        product_revenue = df.groupby(product_col)[revenue_col].sum().sort_values(ascending=False)
        
        # Calculate cumulative percentage
        cumsum = product_revenue.cumsum()
        cumsum_pct = cumsum / cumsum.iloc[-1] * 100
        
        # Classify
        classification = []
        for val in cumsum_pct:
            if val <= 70:
                classification.append('A')
            elif val <= 90:
                classification.append('B')
            else:
                classification.append('C')
        
        result = pd.DataFrame({
            'product': product_revenue.index,
            'revenue': product_revenue.values,
            'cumulative_revenue': cumsum.values,
            'cumulative_pct': cumsum_pct.values,
            'class': classification
        })
        
        logger.info(f"ABC Analysis: A={sum(result['class']=='A')}, B={sum(result['class']=='B')}, C={sum(result['class']=='C')}")
        return result


if __name__ == "__main__":
    print("Segmentation module ready!")
