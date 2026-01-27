"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the ML system."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        if config_path is None:
            # Default to project root config
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "model_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_paths()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _setup_paths(self):
        """Setup and validate paths."""
        project_root = Path(__file__).parent.parent.parent
        
        # Create necessary directories
        directories = [
            project_root / "data" / "raw",
            project_root / "data" / "processed",
            project_root / "data" / "features",
            project_root / "models",
            project_root / "logs",
            project_root / "mlruns",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key (e.g., 'model.name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model."""
        return self.config.get('training', {}).get(model_name, {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config.get('data', {})
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.config.get('features', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get('training', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get('evaluation', {})
    
    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow configuration."""
        return self.config.get('mlflow', {})
    
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return self.project_root / "data"
    
    @property
    def models_dir(self) -> Path:
        """Get models directory."""
        return self.project_root / "models"
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory."""
        return self.project_root / "logs"


def setup_logging(config: Config = None):
    """
    Setup logging configuration.
    
    Args:
        config: Configuration object
    """
    if config is None:
        config = Config()
    
    log_level = config.get('logging.level', 'INFO')
    log_format = config.get('logging.format', 
                           '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logs directory
    log_dir = config.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_dir / "app.log"),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")


# Global config instance
_config = None


def get_config(config_path: str = None) -> Config:
    """
    Get or create global configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config