"""Feature engineering pipeline."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import List, Dict, Tuple, Optional
import logging

from ..utils.config import Config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for fraud detection."""
    
    def __init__(self, config: Config = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            config: Configuration object
        """
        from ..utils.config import get_config
        self.config = config or get_config()
        self.feature_config = self.config.get_feature_config()
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from datetime columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with temporal features
        """
        df_temp = df.copy()
        
        if 'claim_date' in df_temp.columns:
            # Convert to datetime
            df_temp['claim_date'] = pd.to_datetime(df_temp['claim_date'])
            
            # Extract components
            df_temp['claim_month'] = df_temp['claim_date'].dt.month
            df_temp['claim_day'] = df_temp['claim_date'].dt.day
            df_temp['claim_quarter'] = df_temp['claim_date'].dt.quarter
            df_temp['claim_year'] = df_temp['claim_date'].dt.year
            df_temp['claim_day_of_year'] = df_temp['claim_date'].dt.dayofyear
            
            # Cyclical encoding for month
            df_temp['claim_month_sin'] = np.sin(2 * np.pi * df_temp['claim_month'] / 12)
            df_temp['claim_month_cos'] = np.cos(2 * np.pi * df_temp['claim_month'] / 12)
            
            # Day of week is already present
            day_of_week_map = {
                'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                'Friday': 4, 'Saturday': 5, 'Sunday': 6
            }
            if 'claim_day_of_week' in df_temp.columns:
                df_temp['claim_day_of_week_num'] = df_temp['claim_day_of_week'].map(day_of_week_map)
                df_temp['is_weekend'] = df_temp['claim_day_of_week_num'].isin([5, 6]).astype(int)
            
            logger.info("Created temporal features")
        
        return df_temp
    
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio and interaction features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with ratio features
        """
        df_ratio = df.copy()
        
        # Financial ratios
        if 'total_claim' in df_ratio.columns and 'vehicle_price' in df_ratio.columns:
            df_ratio['claim_to_vehicle_price_ratio'] = df_ratio['total_claim'] / (df_ratio['vehicle_price'] + 1)
        
        if 'injury_claim' in df_ratio.columns and 'total_claim' in df_ratio.columns:
            df_ratio['injury_to_total_claim_ratio'] = df_ratio['injury_claim'] / (df_ratio['total_claim'] + 1)
        
        if 'policy deductible' in df_ratio.columns and 'total_claim' in df_ratio.columns:
            df_ratio['deductible_to_claim_ratio'] = df_ratio['policy deductible'] / (df_ratio['total_claim'] + 1)
        
        if 'annual premium' in df_ratio.columns and 'annual_income' in df_ratio.columns:
            df_ratio['premium_to_income_ratio'] = df_ratio['annual premium'] / (df_ratio['annual_income'] + 1)
        
        if 'vehicle_price' in df_ratio.columns and 'annual_income' in df_ratio.columns:
            df_ratio['vehicle_to_income_ratio'] = df_ratio['vehicle_price'] / (df_ratio['annual_income'] + 1)
        
        # Age-related features
        if 'age_of_driver' in df_ratio.columns and 'age_of_vehicle' in df_ratio.columns:
            df_ratio['driver_vehicle_age_diff'] = df_ratio['age_of_driver'] - df_ratio['age_of_vehicle']
            df_ratio['driver_vehicle_age_ratio'] = df_ratio['age_of_driver'] / (df_ratio['age_of_vehicle'] + 1)
        
        # Risk indicators
        if 'safety_rating' in df_ratio.columns and 'past_num_of_claims' in df_ratio.columns:
            df_ratio['risk_score'] = (100 - df_ratio['safety_rating']) * (df_ratio['past_num_of_claims'] + 1)
        
        if 'form defects' in df_ratio.columns and 'days open' in df_ratio.columns:
            df_ratio['defects_per_day'] = df_ratio['form defects'] / (df_ratio['days open'] + 1)
        
        logger.info(f"Created {len([c for c in df_ratio.columns if c not in df.columns])} ratio features")
        
        return df_ratio
    
    def create_categorical_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated features based on categorical variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with aggregate features
        """
        df_agg = df.copy()
        
        target_col = self.config.get('features.target')
        
        # Only create aggregates if target is available (training data)
        if target_col not in df_agg.columns or df_agg[target_col].isnull().all():
            logger.info("Skipping categorical aggregates (no target available)")
            return df_agg
        
        categorical_cols = ['accident_site', 'vehicle_category', 'vehicle_color']
        
        for cat_col in categorical_cols:
            if cat_col in df_agg.columns:
                # Fraud rate by category
                fraud_rate = df_agg.groupby(cat_col)[target_col].mean()
                df_agg[f'{cat_col}_fraud_rate'] = df_agg[cat_col].map(fraud_rate)
                
                # Claim amount by category
                if 'total_claim' in df_agg.columns:
                    avg_claim = df_agg.groupby(cat_col)['total_claim'].mean()
                    df_agg[f'{cat_col}_avg_claim'] = df_agg[cat_col].map(avg_claim)
        
        logger.info("Created categorical aggregate features")
        
        return df_agg
    
    def create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific fraud indicators.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with domain features
        """
        df_domain = df.copy()
        
        # High-value claim indicator
        if 'total_claim' in df_domain.columns:
            claim_threshold = df_domain['total_claim'].quantile(0.75)
            df_domain['is_high_value_claim'] = (df_domain['total_claim'] > claim_threshold).astype(int)
        
        # Young driver indicator (higher risk)
        if 'age_of_driver' in df_domain.columns:
            df_domain['is_young_driver'] = (df_domain['age_of_driver'] < 25).astype(int)
            df_domain['is_senior_driver'] = (df_domain['age_of_driver'] > 65).astype(int)
        
        # New vehicle indicator
        if 'age_of_vehicle' in df_domain.columns:
            df_domain['is_new_vehicle'] = (df_domain['age_of_vehicle'] <= 2).astype(int)
            df_domain['is_old_vehicle'] = (df_domain['age_of_vehicle'] > 10).astype(int)
        
        # Low safety rating indicator
        if 'safety_rating' in df_domain.columns:
            df_domain['is_low_safety'] = (df_domain['safety_rating'] < 70).astype(int)
        
        # High liability percentage
        if 'liab_prct' in df_domain.columns:
            df_domain['is_full_liability'] = (df_domain['liab_prct'] == 100).astype(int)
        
        # Multiple red flags
        red_flag_cols = [
            'address_change', 'form defects', 'past_num_of_claims'
        ]
        available_flags = [col for col in red_flag_cols if col in df_domain.columns]
        if available_flags:
            df_domain['total_red_flags'] = df_domain[available_flags].sum(axis=1)
            df_domain['has_multiple_red_flags'] = (df_domain['total_red_flags'] > 1).astype(int)
        
        # Witness and police report
        if 'witness_present' in df_domain.columns and 'police_report' in df_domain.columns:
            df_domain['no_evidence'] = ((df_domain['witness_present'] == 0) & 
                                       (df_domain['police_report'] == 0)).astype(int)
        
        logger.info(f"Created {len([c for c in df_domain.columns if c not in df.columns])} domain features")
        
        return df_domain
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()
        
        categorical_cols = self.feature_config.get('categorical_features', [])
        
        for col in categorical_cols:
            if col not in df_encoded.columns:
                continue
            
            if fit:
                # Fit and transform
                encoder = LabelEncoder()
                df_encoded[f'{col}_encoded'] = encoder.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = encoder
                logger.info(f"Fitted encoder for {col}")
            else:
                # Transform only
                if col in self.encoders:
                    # Handle unseen categories
                    encoder = self.encoders[col]
                    df_encoded[f'{col}_encoded'] = df_encoded[col].apply(
                        lambda x: encoder.transform([str(x)])[0] 
                        if str(x) in encoder.classes_ 
                        else -1
                    )
                else:
                    logger.warning(f"No encoder found for {col}")
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit scalers (True for training, False for inference)
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        numerical_cols = self.feature_config.get('numerical_features', [])
        
        # Add engineered numerical features
        engineered_num_cols = [
            col for col in df_scaled.columns 
            if any(keyword in col for keyword in ['_ratio', '_diff', '_score', '_rate', '_avg'])
            and df_scaled[col].dtype in ['float64', 'int64']
        ]
        
        all_num_cols = list(set(numerical_cols + engineered_num_cols))
        available_cols = [col for col in all_num_cols if col in df_scaled.columns]
        
        if fit:
            scaler = StandardScaler()
            df_scaled[available_cols] = scaler.fit_transform(df_scaled[available_cols])
            self.scalers['standard'] = scaler
            logger.info(f"Fitted scaler for {len(available_cols)} numerical features")
        else:
            if 'standard' in self.scalers:
                scaler = self.scalers['standard']
                df_scaled[available_cols] = scaler.transform(df_scaled[available_cols])
            else:
                logger.warning("No scaler found for numerical features")
        
        return df_scaled
    
    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformers (True for training, False for inference)
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Starting feature engineering. Input shape: {df.shape}")
        
        # Create new features
        df_features = self.create_temporal_features(df)
        df_features = self.create_ratio_features(df_features)
        df_features = self.create_domain_features(df_features)
        
        # Only create categorical aggregates during training
        if fit:
            df_features = self.create_categorical_aggregates(df_features)
        
        # Encode categorical features
        df_features = self.encode_categorical_features(df_features, fit=fit)
        
        # Scale numerical features
        df_features = self.scale_numerical_features(df_features, fit=fit)
        
        logger.info(f"Feature engineering complete. Output shape: {df_features.shape}")
        logger.info(f"New features created: {df_features.shape[1] - df.shape[1]}")
        
        # Store feature names
        if fit:
            target_col = self.config.get('features.target')
            self.feature_names = [col for col in df_features.columns if col != target_col]
        
        return df_features
    
    def select_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        k: int = 50,
        method: str = 'f_classif'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top k features using statistical tests.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            k: Number of features to select
            method: Selection method ('f_classif' or 'mutual_info')
            
        Returns:
            Tuple of (selected features DataFrame, selected feature names)
        """
        k = min(k, X.shape[1])
        
        if method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        else:
            selector = SelectKBest(mutual_info_classif, k=k)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected_features)} features using {method}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def get_feature_importance_from_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic feature importance metrics.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            DataFrame with feature importance
        """
        target_col = self.config.get('features.target')
        
        if target_col not in df.columns:
            logger.warning("Target column not found in DataFrame")
            return pd.DataFrame()
        
        feature_cols = [col for col in df.columns if col != target_col]
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        importance_data = []
        
        for feature in numeric_features:
            # Correlation with target
            corr = df[feature].corr(df[target_col])
            
            # Mean difference between classes
            fraud_mean = df[df[target_col] == 1][feature].mean()
            non_fraud_mean = df[df[target_col] == 0][feature].mean()
            mean_diff = abs(fraud_mean - non_fraud_mean)
            
            importance_data.append({
                'feature': feature,
                'correlation': abs(corr),
                'mean_difference': mean_diff
            })
        
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('correlation', ascending=False)
        
        return importance_df