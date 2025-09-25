#!/usr/bin/env python3
"""
Configuration Manager for Python-based Configuration
============================================================

This module provides a centralized way to manage all configuration parameters
"""

from . import config
from typing import Dict, Any


class ConfigManager:
    """
    Updated configuration manager using Python-based configuration.
    
    This class provides easy access to all configuration parameters
    with the ability to update them automatically.
    """
    
    def __init__(self):
        """Initialize configuration manager with Python config."""
        self.config_module = config
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature extraction configuration."""
        return self.config_module.get_feature_config()
    
    def get_data_processing_config(self) -> Dict[str, Any]:
        """Get data processing configuration."""
        return self.config_module.get_data_processing_config()
    
    def get_cv_config(self) -> Dict[str, Any]:
        """Get cross-validation configuration."""
        return self.config_module.get_cv_config()
    
    def get_threshold_config(self) -> Dict[str, Any]:
        """Get threshold optimization configuration."""
        return self.config_module.get_threshold_config()
    
    def get_voting_config(self) -> Dict[str, Any]:
        """Get voting threshold configuration."""
        return self.config_module.get_voting_config()
    
    def get_hyperparameter_config(self) -> Dict[str, Any]:
        """Get hyperparameter tuning configuration."""
        return self.config_module.get_hyperparameter_config()
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return self.config_module.get_model_defaults(model_name)
    
    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration."""
        return self.config_module.get_global_config()
    