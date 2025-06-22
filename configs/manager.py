"""
Configuration Manager Module

This module provides configuration management for weather analogue models,
including configuration loading, validation, and manipulation.
"""
import os
import copy
import yaml
import logging
from pathlib import Path
from datetime import datetime
from itertools import product

class ConfigManager:
    """
    Manager for model configurations.
    Handles loading, validation, and manipulation of configurations.
    """
    def __init__(self, config_path=None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file or directory (optional)
        """
        self.base_config = {}
        self.configs = []
        self.logger = logging.getLogger('ConfigManager')
        
        # Load configuration if provided
        if config_path:
            self.base_config = self.load_config(config_path)
    
    def load_base_config(self, config_path):
        """
        Load base configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                self.base_config = yaml.safe_load(f)
                self.logger.info(f"Loaded base configuration from {config_path}")
                return self.base_config
        except Exception as e:
            self.logger.error(f"Error loading base configuration: {str(e)}")
            raise
    
    def load_config(self, config_path):
        """
        Unified method to load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration dictionary
        """
        path = Path(config_path)
        
        if not path.exists():
            self.logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Add ID if not present
            if 'id' not in config:
                config['id'] = os.path.splitext(os.path.basename(config_path))[0]
                
            self.logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def load_configs(self, config_dir):
        """
        Load multiple configurations from directory.
        
        Args:
            config_dir: Directory containing configuration files
            
        Returns:
            List of loaded configurations
        """
        path = Path(config_dir)
        self.configs = []
        
        if not path.is_dir():
            self.logger.error(f"Not a directory: {config_dir}")
            raise NotADirectoryError(f"Not a directory: {config_dir}")
            
        config_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml') or f.endswith('.yml')]
        
        for file in config_files:
            file_path = os.path.join(config_dir, file)
            try:
                with open(file_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # Add ID if not present
                if 'id' not in config:
                    config['id'] = os.path.splitext(file)[0]
                    
                self.configs.append(config)
            except Exception as e:
                self.logger.error(f"Error loading configuration {file}: {str(e)}")
                
        self.logger.info(f"Loaded {len(self.configs)} configurations from {config_dir}")
        return self.configs
    
    def validate_config(self, config):
        """
        Validate configuration structure.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Boolean indicating if configuration is valid
        """
        # Check if config is a dictionary
        if not isinstance(config, dict):
            self.logger.error("Configuration must be a dictionary")
            return False
        
        # Required top-level sections
        required_sections = ['data', 'model', 'training']
        
        # Check for multi-model format
        if 'models' in config:
            # Multi-model format
            if not isinstance(config['models'], list):
                self.logger.error("'models' section must be a list of model configurations")
                return False
                
            # Check each model configuration
            for i, model_config in enumerate(config['models']):
                if not isinstance(model_config, dict):
                    self.logger.error(f"Model configuration at index {i} must be a dictionary")
                    return False
                    
                if 'model' not in model_config:
                    self.logger.error(f"Model configuration at index {i} missing 'model' section")
                    return False
                    
            # For multi-model configs, we only need 'data' at the top level
            required_sections = ['data']
        
        # Check for required sections
        for section in required_sections:
            if section not in config:
                self.logger.error(f"Missing required section: {section}")
                return False
                
        return True
    
    def update_config(self, base_config, updates):
        """
        Update a configuration with new values.
        
        Args:
            base_config: Base configuration to update
            updates: Dictionary of updates to apply
            
        Returns:
            Updated configuration dictionary
        """
        config = copy.deepcopy(base_config)
        
        for section, params in updates.items():
            if section not in config:
                config[section] = {}
                
            if isinstance(params, dict):
                for k, v in params.items():
                    config[section][k] = v
            else:
                config[section] = params
                
        return config
    
    def update_nested_config(self, config, updates):
        """
        Update a configuration with new values using dot notation.
        
        Args:
            config: Configuration dictionary to update
            updates: Dictionary of updates with dot notation keys
            
        Returns:
            Updated configuration dictionary
        """
        updated_config = copy.deepcopy(config)
        
        for key, value in updates.items():
            # Handle nested updates with dot notation (e.g., "model.learning_rate")
            if '.' in key:
                parts = key.split('.')
                c = updated_config
                
                # Navigate to the nested dictionary
                for part in parts[:-1]:
                    if part not in c:
                        c[part] = {}
                    c = c[part]
                
                # Update the value
                c[parts[-1]] = value
            else:
                updated_config[key] = value
        
        return updated_config
    
    def expand_config_for_multiple_models(self, config):
        """
        Expand a single-model configuration to support multiple models.
        
        Args:
            config: Single-model configuration
            
        Returns:
            Configuration with 'models' list
        """
        if 'models' in config:
            # Already in multi-model format
            return config
        
        multi_config = copy.deepcopy(config)
        
        # Extract model-specific configuration
        model_config = {
            'model': multi_config.pop('model', {}),
            'id': multi_config.get('id', 'model_1')
        }
        
        # Extract training-specific configuration that should go with the model
        if 'training' in multi_config:
            model_specific_training = {}
            model_specific_params = ['learning_rate', 'batch_size', 'weight_decay', 'optimizer']
            
            for param in model_specific_params:
                if param in multi_config['training']:
                    model_specific_training[param] = multi_config['training'].pop(param)
            
            if model_specific_training:
                model_config['training'] = model_specific_training
        
        # Create models list with the single model
        multi_config['models'] = [model_config]
        
        return multi_config
    
    def generate_experiment_id(self, prefix="exp"):
        """
        Generate a unique experiment ID.
        
        Args:
            prefix: Prefix for the experiment ID
            
        Returns:
            Unique experiment ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"
