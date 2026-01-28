"""Model prediction and inference pipeline."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union
import logging
import joblib
from pathlib import Path

from ..utils.config import Config

logger = logging.getLogger(__name__)


class FraudPredictor:
    """Fraud detection predictor for inference."""
    
    def __init__(self, model_path: str = None, config: Config = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model
            config: Configuration object
        """
        from ..utils.config import get_config
        self.config = config or get_config()
        
        self.model = None
        self.feature_engineer = None
        self.threshold = 0.5
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load trained model and feature engineer.
        
        Args:
            model_path: Path to model file
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load model
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load feature engineer
        fe_path = model_path.parent / model_path.name.replace("_model.pkl", "_feature_engineer.pkl")
        if fe_path.exists():
            self.feature_engineer = joblib.load(fe_path)
            logger.info(f"Feature engineer loaded from {fe_path}")
        else:
            logger.warning("Feature engineer not found. Predictions may fail.")
    
    def preprocess_input(self, input_data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            input_data: Input data as dict or DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Convert dict to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Apply feature engineering
        if self.feature_engineer:
            df = self.feature_engineer.engineer_features(df, fit=False)
        
        # Select features used in training
        if hasattr(self.feature_engineer, 'feature_names'):
            feature_names = self.feature_engineer.feature_names
            # Use only features that exist in the input
            available_features = [f for f in feature_names if f in df.columns]
            df = df[available_features]
        
        return df
    
    def predict(
        self, 
        input_data: Union[Dict, pd.DataFrame],
        return_proba: bool = False
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Make fraud predictions.
        
        Args:
            input_data: Input data for prediction
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions or prediction dictionary
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess input
        X = self.preprocess_input(input_data)
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        if return_proba:
            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'fraud_probability': probabilities.tolist(),
                'is_fraud': (probabilities > self.threshold).tolist()
            }
        
        return predictions
    
    def predict_single(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud for a single claim.
        
        Args:
            input_dict: Dictionary with claim features
            
        Returns:
            Prediction result with probability and label
        """
        result = self.predict(input_dict, return_proba=True)
        
        return {
            'is_fraud': bool(result['is_fraud'][0]),
            'fraud_probability': float(result['fraud_probability'][0]),
            'confidence': float(abs(result['fraud_probability'][0] - 0.5) * 2),
            'risk_level': self._get_risk_level(result['fraud_probability'][0])
        }
    
    def predict_batch(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud for multiple claims.
        
        Args:
            input_df: DataFrame with claim features
            
        Returns:
            DataFrame with predictions
        """
        result = self.predict(input_df, return_proba=True)
        
        output_df = input_df.copy()
        output_df['fraud_prediction'] = result['predictions']
        output_df['fraud_probability'] = result['fraud_probability']
        output_df['is_fraud'] = result['is_fraud']
        output_df['risk_level'] = output_df['fraud_probability'].apply(self._get_risk_level)
        
        return output_df
    
    def _get_risk_level(self, probability: float) -> str:
        """
        Categorize fraud probability into risk levels.
        
        Args:
            probability: Fraud probability
            
        Returns:
            Risk level string
        """
        if probability >= 0.8:
            return "CRITICAL"
        elif probability >= 0.6:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        elif probability >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def set_threshold(self, threshold: float):
        """
        Set custom decision threshold.
        
        Args:
            threshold: Threshold value between 0 and 1
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.threshold = threshold
        logger.info(f"Prediction threshold set to {threshold}")
    
    def explain_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide explanation for prediction (basic version).
        
        Args:
            input_data: Input claim data
            
        Returns:
            Explanation dictionary
        """
        prediction_result = self.predict_single(input_data)
        
        # Get feature importances if available
        explanation = {
            'prediction': prediction_result,
            'input_features': input_data,
        }
        
        if hasattr(self.model, 'feature_importances_'):
            # Get preprocessed features
            X = self.preprocess_input(input_data)
            
            # Get top contributing features
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            explanation['top_features'] = feature_importance.to_dict('records')
        
        return explanation
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        if self.model is None:
            return {"error": "No model loaded"}
        
        info = {
            'model_type': type(self.model).__name__,
            'threshold': self.threshold,
        }
        
        if hasattr(self.model, 'n_features_in_'):
            info['n_features'] = self.model.n_features_in_
        
        if hasattr(self.model, 'feature_names_in_'):
            info['feature_names'] = self.model.feature_names_in_.tolist()
        
        return info