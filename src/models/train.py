"""Model training pipeline with MLflow integration."""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
import optuna
from typing import Dict, Any, Tuple, List
import logging
import joblib
from pathlib import Path
import json

from ..utils.config import Config
from ..data.data_loader import DataLoader
from ..features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """Fraud detection model trainer and evaluator."""
    
    def __init__(self, config: Config = None):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration object
        """
        from ..utils.config import get_config
        self.config = config or get_config()
        self.training_config = self.config.get_training_config()
        self.mlflow_config = self.config.get_mlflow_config()
        self.eval_config = self.config.get_evaluation_config()
        
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_engineer = FeatureEngineer(config)
        
        # Setup MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        tracking_uri = str(self.config.project_root / self.mlflow_config.get('tracking_uri', 'mlruns'))
        mlflow.set_tracking_uri(tracking_uri)
        
        experiment_name = self.mlflow_config.get('experiment_name', 'fraud_detection')
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment set: {experiment_name}")
        except Exception as e:
            logger.warning(f"Error setting up MLflow: {e}")
    
    def handle_imbalance(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using SMOTE or similar techniques.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Balanced training data
        """
        strategy = self.training_config.get('imbalance_strategy', 'smote')
        
        if strategy == 'none':
            return X_train, y_train
        
        logger.info(f"Original class distribution: {y_train.value_counts().to_dict()}")
        
        if strategy == 'smote':
            sampling_strategy = self.training_config.get('smote_sampling_strategy', 0.5)
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        elif strategy == 'smotetomek':
            smote_tomek = SMOTETomek(random_state=42)
            X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
        else:
            logger.warning(f"Unknown imbalance strategy: {strategy}")
            return X_train, y_train
        
        logger.info(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        params: Dict[str, Any] = None
    ) -> xgb.XGBClassifier:
        """Train XGBoost model."""
        if params is None:
            params = self.config.get_model_config('xgboost')
        
        # Handle class imbalance in model
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        params['scale_pos_weight'] = scale_pos_weight
        
        model = xgb.XGBClassifier(**params, random_state=42)
        
        if X_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        logger.info("XGBoost model trained")
        return model
    
    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        params: Dict[str, Any] = None
    ) -> lgb.LGBMClassifier:
        """Train LightGBM model."""
        if params is None:
            params = self.config.get_model_config('lightgbm')
        
        # Handle class imbalance
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        params['scale_pos_weight'] = scale_pos_weight
        
        model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
        
        if X_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(
                X_train, y_train,
                eval_set=eval_set
            )
        else:
            model.fit(X_train, y_train)
        
        logger.info("LightGBM model trained")
        return model
    
    def train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        params: Dict[str, Any] = None
    ) -> CatBoostClassifier:
        """Train CatBoost model."""
        if params is None:
            params = self.config.get_model_config('catboost')
        
        # Handle class imbalance
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        params['scale_pos_weight'] = scale_pos_weight
        
        model = CatBoostClassifier(**params, random_state=42)
        
        if X_val is not None:
            eval_set = (X_val, y_val)
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            model.fit(X_train, y_train, verbose=False)
        
        logger.info("CatBoost model trained")
        return model
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
        
        # Business metrics
        fp_cost = self.eval_config.get('false_positive_cost', 100)
        fn_cost = self.eval_config.get('false_negative_cost', 10000)
        total_cost = (metrics['fp'] * fp_cost) + (metrics['fn'] * fn_cost)
        metrics['total_cost'] = total_cost
        metrics['avg_cost_per_prediction'] = total_cost / len(y_test)
        
        logger.info(f"\n{model_name} Performance:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Total Cost: ${metrics['total_cost']:.2f}")
        
        return metrics
    
    def cross_validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation.
        
        Args:
            model: Model to validate
            X: Features
            y: Target
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = {}
        
        for metric in scoring:
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=metric)
            cv_results[f'cv_{metric}'] = scores.tolist()
            cv_results[f'cv_{metric}_mean'] = float(scores.mean())
            cv_results[f'cv_{metric}_std'] = float(scores.std())
            
            logger.info(f"CV {metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_results
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Train all configured models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of trained models and metrics
        """
        models_to_train = self.training_config.get('models', ['xgboost'])
        results = {}
        
        # Handle imbalance for training data
        imbalance_strategy = self.training_config.get('imbalance_strategy', 'none')
        if imbalance_strategy in ['smote', 'smotetomek']:
            X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        for model_name in models_to_train:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'='*50}")
            
            with mlflow.start_run(run_name=f"{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                model_params = self.config.get_model_config(model_name)
                mlflow.log_params(model_params)
                mlflow.log_param("imbalance_strategy", imbalance_strategy)
                
                # Train model
                if model_name == 'xgboost':
                    model = self.train_xgboost(X_train_balanced, y_train_balanced, X_val, y_val)
                elif model_name == 'lightgbm':
                    model = self.train_lightgbm(X_train_balanced, y_train_balanced, X_val, y_val)
                elif model_name == 'catboost':
                    model = self.train_catboost(X_train_balanced, y_train_balanced, X_val, y_val)
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    continue
                
                # Evaluate on validation set
                if X_val is not None:
                    val_metrics = self.evaluate_model(model, X_val, y_val, f"{model_name}_val")
                    for metric_name, value in val_metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"val_{metric_name}", value)
                
                # Evaluate on test set
                if X_test is not None:
                    test_metrics = self.evaluate_model(model, X_test, y_test, f"{model_name}_test")
                    for metric_name, value in test_metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"test_{metric_name}", value)
                else:
                    test_metrics = {}
                
                # Cross-validation
                cv_results = self.cross_validate_model(model, X_train, y_train)
                for metric_name, value in cv_results.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(metric_name, value)
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    # Log top 20 features
                    top_features = feature_importance.head(20)
                    mlflow.log_dict(top_features.to_dict(), "top_features.json")
                
                # Log model
                if model_name == 'xgboost':
                    mlflow.xgboost.log_model(model, "model")
                elif model_name == 'lightgbm':
                    mlflow.lightgbm.log_model(model, "model")
                elif model_name == 'catboost':
                    mlflow.catboost.log_model(model, "model")
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'val_metrics': val_metrics if X_val is not None else {},
                    'test_metrics': test_metrics,
                    'cv_results': cv_results
                }
                
                self.models[model_name] = model
        
        # Select best model based on test F1 score
        if X_test is not None and results:
            best_f1 = -1
            for model_name, result in results.items():
                f1 = result['test_metrics'].get('f1', 0)
                if f1 > best_f1:
                    best_f1 = f1
                    self.best_model = result['model']
                    self.best_model_name = model_name
            
            logger.info(f"\nBest model: {self.best_model_name} with F1={best_f1:.4f}")
        
        return results
    
    def save_model(self, model: Any = None, model_name: str = None, path: str = None):
        """
        Save model to disk.
        
        Args:
            model: Model to save (uses best model if None)
            model_name: Name for the model file
            path: Path to save the model
        """
        if model is None:
            model = self.best_model
            model_name = self.best_model_name
        
        if model is None:
            logger.error("No model to save")
            return
        
        if path is None:
            models_dir = self.config.models_dir
            path = models_dir / f"{model_name}_model.pkl"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
        
        # Save feature engineer
        fe_path = path.parent / f"{model_name}_feature_engineer.pkl"
        joblib.dump(self.feature_engineer, fe_path)
        logger.info(f"Feature engineer saved to {fe_path}")
    
    def load_model(self, path: str):
        """Load model from disk."""
        path = Path(path)
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        
        # Load feature engineer
        fe_path = path.parent / path.name.replace("_model.pkl", "_feature_engineer.pkl")
        if fe_path.exists():
            self.feature_engineer = joblib.load(fe_path)
            logger.info(f"Feature engineer loaded from {fe_path}")
        
        return model