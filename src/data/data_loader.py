"""Data loading utilities."""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

# Allow running this file both as a module (`python -m src.data.data_loader`)
# and as a script (`python src/data/data_loader.py`).
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = _Path(__file__).resolve().parents[2]  # .../insurance_fraud_detection
    if str(_repo_root) not in _sys.path:
        _sys.path.insert(0, str(_repo_root))

    from src.utils.config import Config  # type: ignore
else:
    from ..utils.config import Config

logger = logging.getLogger(__name__)

class DataLoader:
    """Load and prepare data for ML pipeline."""
    
    def __init__(self, config: Config = None):
        """
        Initialize DataLoader.
        
        Args:
            config: Configuration object
        """
        from ..utils.config import get_config
        self.config = config or get_config()
        self.data_config = self.config.get_data_config()
    def load_raw_data(self)->pd.DataFrame:
        raw_path=self.config.project_root/self.data_config['raw_path']
        try:
            df = pd.read_csv(raw_path)
            logger.info(f"Loaded raw data: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            raise
    def load_processed_data(self)->pd.DataFrame:
        processed_path=self.config.project_root/self.data_config['processed_path']
        try:
            df=pd.read_csv(processed_path)
            logger.info(f"Loaded processed data: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            raise
    def load_feature_data(self) -> pd.DataFrame:
        """
        Load feature-engineered data.
        
        Returns:
            Feature-engineered DataFrame
        """
        features_path = self.config.project_root / self.data_config['features_path']
        
        if not features_path.exists():
            logger.warning("Feature data not found.")
            raise FileNotFoundError(f"Feature data not found at {features_path}")
        
        try:
            df = pd.read_csv(features_path)
            logger.info(f"Loaded feature data: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading feature data: {e}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame):
        """
        Save processed data.
        
        Args:
            df: DataFrame to save
        """
        processed_path = self.config.project_root / self.data_config['processed_path']
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            df.to_csv(processed_path, index=False)
            logger.info(f"Saved processed data to {processed_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    
    def save_feature_data(self, df: pd.DataFrame):
        """
        Save feature-engineered data.
        
        Args:
            df: DataFrame to save
        """
        features_path = self.config.project_root / self.data_config['features_path']
        features_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            df.to_csv(features_path, index=False)
            logger.info(f"Saved feature data to {features_path}")
        except Exception as e:
            logger.error(f"Error saving feature data: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Log initial shape
        logger.info(f"Initial shape: {df_clean.shape}")
        
        # Handle missing values in target
        target_col = self.config.get('features.target')
        if target_col in df_clean.columns:
            initial_nulls = df_clean[target_col].isnull().sum()
            df_clean = df_clean.dropna(subset=[target_col])
            logger.info(f"Dropped {initial_nulls} rows with missing target")
        
        # Convert target to binary
        if target_col in df_clean.columns:
            df_clean[target_col] = (df_clean[target_col] == 'Y').astype(int)
            logger.info(f"Target distribution: {df_clean[target_col].value_counts().to_dict()}")
        
        # Handle numeric columns that might be strings
        numeric_cols = ['age_of_vehicle', 'injury_claim']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Handle outliers in age
        if 'age_of_driver' in df_clean.columns:
            # Cap age at 100 (278 seems like an error)
            outlier_count = (df_clean['age_of_driver'] > 100).sum()
            df_clean.loc[df_clean['age_of_driver'] > 100, 'age_of_driver'] = df_clean['age_of_driver'].median()
            if outlier_count > 0:
                logger.info(f"Capped {outlier_count} age outliers")
        
        # Convert binary columns
        binary_cols = self.config.get('features.binary_features', [])
        for col in binary_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(int)
        
        # Handle marital_status encoding
        if 'marital_status' in df_clean.columns:
            df_clean['marital_status'] = df_clean['marital_status'].map({0: 'Single', 1: 'Married'}).fillna('Unknown')
        
        logger.info(f"Final shape after cleaning: {df_clean.shape}")
        logger.info(f"Missing values per column:\n{df_clean.isnull().sum()[df_clean.isnull().sum() > 0]}")
        
        return df_clean
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = None,
        val_size: float = None,
        random_state: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            test_size: Test set size (fraction)
            val_size: Validation set size (fraction of remaining data)
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df) or (train_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        test_size = test_size or self.data_config.get('test_size', 0.2)
        val_size = val_size or self.data_config.get('val_size', 0.1)
        random_state = random_state or self.data_config.get('random_state', 42)
        
        target_col = self.config.get('features.target')
        
        # First split: train+val and test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df[target_col] if target_col in df.columns else None
        )
        
        if val_size > 0:
            # Second split: train and val
            val_size_adjusted = val_size / (1 - test_size)
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=train_val_df[target_col] if target_col in train_val_df.columns else None
            )
            
            logger.info(f"Data split - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
            return train_df, val_df, test_df
        else:
            logger.info(f"Data split - Train: {train_val_df.shape}, Test: {test_df.shape}")
            return train_val_df, test_df, None
