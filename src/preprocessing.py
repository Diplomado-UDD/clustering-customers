"""
Data Preprocessing Module.

This module handles data cleaning, feature engineering, and transformation
for customer clustering analysis. Implements RFM analysis, CLV calculation,
and behavioral feature extraction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from datetime import datetime, timedelta
import logging
from typing import Tuple

from src.config import Config

logger = logging.getLogger(__name__)


class CustomerDataPreprocessor:
    """Preprocesses customer data for clustering analysis."""

    def __init__(self, config: Config):
        """
        Initialize the preprocessor.

        Args:
            config: Configuration object containing preprocessing parameters
        """
        self.config = config
        self.scaler = None
        self.feature_names = None

        logger.info("CustomerDataPreprocessor initialized")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw customer data.

        Args:
            df: Raw customer DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning data. Initial shape: {df.shape}")

        df = df.copy()

        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates(subset=['customer_id'])
        logger.info(f"Removed {initial_rows - len(df)} duplicate customers")

        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)

        # Convert date columns
        date_columns = ['registration_date', 'first_purchase_date', 'last_purchase_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        logger.info(f"Data cleaning complete. Final shape: {df.shape}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for clustering.

        Args:
            df: Cleaned customer DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features")

        df = df.copy()

        # Calculate customer age (days since registration)
        reference_date = pd.Timestamp(self.config.data_generator.end_date)
        df['customer_age_days'] = (reference_date - df['registration_date']).dt.days

        # Calculate days since last purchase
        df['days_since_last_purchase'] = (reference_date - df['last_purchase_date']).dt.days

        # Calculate customer lifetime (days between first and last purchase)
        df['customer_lifetime_days'] = (
            df['last_purchase_date'] - df['first_purchase_date']
        ).dt.days

        # Purchase frequency (transactions per day)
        df['purchase_frequency'] = df['total_transactions'] / (df['customer_lifetime_days'] + 1)

        # Average time between purchases
        df['avg_days_between_purchases'] = (
            df['customer_lifetime_days'] / (df['total_transactions'] + 1)
        )

        # Category diversity score
        df['category_diversity'] = df['unique_categories'] / 7.0  # Normalize by total categories

        # Spending consistency (inverse of coefficient of variation)
        df['spending_consistency'] = 1 / (
            1 + (df['std_transaction_value'] / (df['avg_transaction_value'] + 1))
        )

        # Discount sensitivity
        df['discount_sensitivity'] = df['avg_discount_used'] / 20.0  # Normalize by max discount

        # Fill any infinite or NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df

    def calculate_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics.

        Args:
            df: Customer DataFrame

        Returns:
            DataFrame with RFM features
        """
        if not self.config.preprocessing.enable_rfm:
            return df

        logger.info("Calculating RFM metrics")

        df = df.copy()

        # Recency: Days since last purchase (already calculated)
        df['rfm_recency'] = df['days_since_last_purchase']

        # Frequency: Total number of transactions
        df['rfm_frequency'] = df['total_transactions']

        # Monetary: Total amount spent
        df['rfm_monetary'] = df['total_spent']

        # Calculate RFM scores (1-5 scale)
        df['rfm_r_score'] = pd.qcut(
            df['rfm_recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop'
        ).astype(float)

        df['rfm_f_score'] = pd.qcut(
            df['rfm_frequency'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
        ).astype(float)

        df['rfm_m_score'] = pd.qcut(
            df['rfm_monetary'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
        ).astype(float)

        # Calculate overall RFM score
        df['rfm_score'] = df['rfm_r_score'] + df['rfm_f_score'] + df['rfm_m_score']

        logger.info("RFM calculation complete")
        return df

    def calculate_clv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Customer Lifetime Value (CLV).

        Args:
            df: Customer DataFrame

        Returns:
            DataFrame with CLV feature
        """
        if not self.config.preprocessing.enable_clv:
            return df

        logger.info("Calculating Customer Lifetime Value")

        df = df.copy()

        # Simple CLV calculation: Average transaction value * Purchase frequency * Customer lifetime
        # Projected over 1 year
        days_in_year = 365

        df['clv_simple'] = (
            df['avg_transaction_value'] *
            df['purchase_frequency'] *
            days_in_year
        )

        # Advanced CLV with retention rate estimation
        # Retention rate based on recency and frequency
        df['retention_rate'] = np.clip(
            (1 - df['days_since_last_purchase'] / 365) * (df['rfm_f_score'] / 5),
            0, 1
        )

        # CLV = (Average Transaction Value * Purchase Frequency * Customer Lifetime) * Retention Rate
        df['clv_advanced'] = df['clv_simple'] * df['retention_rate']

        logger.info("CLV calculation complete")
        return df

    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral segmentation features.

        Args:
            df: Customer DataFrame

        Returns:
            DataFrame with behavioral features
        """
        if not self.config.preprocessing.enable_behavioral_features:
            return df

        logger.info("Creating behavioral features")

        df = df.copy()

        # Engagement score (combination of frequency and recency)
        max_recency = df['days_since_last_purchase'].max()
        df['engagement_score'] = (
            (1 - df['days_since_last_purchase'] / max_recency) *
            (df['total_transactions'] / df['total_transactions'].max())
        )

        # Value tier based on spending
        df['value_tier'] = pd.qcut(
            df['total_spent'], q=4, labels=['Low', 'Medium', 'High', 'Premium'], duplicates='drop'
        )

        # Activity status
        df['activity_status'] = pd.cut(
            df['days_since_last_purchase'],
            bins=[0, 30, 90, 180, float('inf')],
            labels=['Active', 'Moderate', 'Dormant', 'Inactive']
        )

        # Shopping pattern: Regular vs Seasonal
        df['is_regular_buyer'] = (df['purchase_frequency'] > df['purchase_frequency'].median()).astype(int)

        # High-value flag
        df['is_high_value'] = (
            (df['total_spent'] > df['total_spent'].quantile(0.75)) &
            (df['total_transactions'] > df['total_transactions'].quantile(0.75))
        ).astype(int)

        # Churn risk score
        df['churn_risk'] = (
            (df['days_since_last_purchase'] / max_recency) * 0.5 +
            (1 - df['engagement_score']) * 0.5
        )

        logger.info("Behavioral features created")
        return df

    def handle_outliers(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Handle outliers in numeric features.

        Args:
            df: Customer DataFrame
            features: List of feature names to check for outliers

        Returns:
            DataFrame with outliers handled
        """
        if not self.config.preprocessing.handle_outliers:
            return df

        logger.info("Handling outliers")

        df = df.copy()
        method = self.config.preprocessing.outlier_method
        threshold = self.config.preprocessing.outlier_threshold

        for feature in features:
            if feature not in df.columns:
                continue

            if method == 'iqr':
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                # Cap outliers
                df[feature] = df[feature].clip(lower_bound, upper_bound)

            elif method == 'zscore':
                mean = df[feature].mean()
                std = df[feature].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std

                df[feature] = df[feature].clip(lower_bound, upper_bound)

        logger.info("Outlier handling complete")
        return df

    def select_clustering_features(self, df: pd.DataFrame) -> list:
        """
        Select relevant features for clustering.

        Args:
            df: Customer DataFrame

        Returns:
            List of feature names for clustering
        """
        features = [
            # Transaction metrics
            'total_transactions',
            'total_spent',
            'avg_transaction_value',
            'purchase_frequency',

            # Temporal features
            'customer_age_days',
            'days_since_last_purchase',
            'avg_days_between_purchases',

            # RFM features
            'rfm_recency',
            'rfm_frequency',
            'rfm_monetary',
            'rfm_score',

            # CLV features
            'clv_simple',
            'clv_advanced',
            'retention_rate',

            # Behavioral features
            'category_diversity',
            'spending_consistency',
            'discount_sensitivity',
            'engagement_score',
            'churn_risk',
        ]

        # Filter to only include features that exist in the dataframe
        available_features = [f for f in features if f in df.columns]

        logger.info(f"Selected {len(available_features)} features for clustering")
        return available_features

    def scale_features(self, df: pd.DataFrame, features: list, fit: bool = True) -> np.ndarray:
        """
        Scale features for clustering.

        Args:
            df: Customer DataFrame
            features: List of feature names to scale
            fit: Whether to fit the scaler or use existing one

        Returns:
            Scaled feature array
        """
        logger.info(f"Scaling {len(features)} features")

        method = self.config.preprocessing.scaling_method

        if fit:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")

            scaled_data = self.scaler.fit_transform(df[features])
            self.feature_names = features
        else:
            if self.scaler is None:
                raise ValueError("Scaler has not been fitted yet")
            scaled_data = self.scaler.transform(df[features])

        logger.info(f"Feature scaling complete using {method} method")
        return scaled_data

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, list]:
        """
        Complete preprocessing pipeline.

        Args:
            df: Raw customer DataFrame

        Returns:
            Tuple of (processed_df, scaled_features, feature_names)
        """
        logger.info("Starting complete preprocessing pipeline")

        # Clean data
        df = self.clean_data(df)

        # Engineer features
        df = self.engineer_features(df)

        # Calculate RFM
        df = self.calculate_rfm(df)

        # Calculate CLV
        df = self.calculate_clv(df)

        # Create behavioral features
        df = self.create_behavioral_features(df)

        # Select features for clustering
        features = self.select_clustering_features(df)

        # Handle outliers
        df = self.handle_outliers(df, features)

        # Scale features
        scaled_features = self.scale_features(df, features, fit=True)

        logger.info("Preprocessing pipeline complete")
        return df, scaled_features, features

    def save_processed_data(self, df: pd.DataFrame):
        """Save processed data to CSV."""
        output_path = self.config.paths.data_dir / 'processed_customer_data.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
