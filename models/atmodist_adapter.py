"""
Atmodist Model Adapter Module

This module provides adapter classes for using Atmodist models
with the new training framework.
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.atmodist import Atmodist, OrdinalAtmodist, TripletAtmodist

class AtmodistModelAdapter:
    """
    Adapter class for Atmodist models to work with the new training framework.
    This class wraps PyTorch Lightning models to be compatible with the
    framework's ModelTrainer.
    """
    def __init__(self, config):
        """
        Initialize the Atmodist model adapter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.model_type = self.model_config.get('type', 'atmodist_standard')
        self.logger = logging.getLogger('AtmodistModelAdapter')
        
        # Get model parameters from config
        self.num_classes = self.model_config.get('num_classes', 8)  # Default for 3h frequency with 24h interval
        self.res_in_channels = self.model_config.get('res_in_channels', 5)  # Default for 5 variables
        self.res_out_channels_list = self.model_config.get('res_out_channels_list', (16, 32, 64, 128))
        self.lr = self.model_config.get('learning_rate', 0.01)
        self.momentum = self.model_config.get('momentum', 0.9)
        self.distance_metric = self.model_config.get('distance_metric', 'wasserstein')
        self.dropout = self.model_config.get('dropout', 0.2)
        
        # Initialize model
        self._init_model()
        
        # Loss function setup
        self._setup_loss_function()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def _init_model(self):
        """Initialize the appropriate model based on configuration"""
        if self.model_type == 'atmodist_standard':
            self.logger.info(f"Initializing standard Atmodist model with {self.res_in_channels} input channels")
            self.model = Atmodist(
                num_classes=self.num_classes,
                res_in_channels=self.res_in_channels,
                res_out_channels_list=self.res_out_channels_list,
                lr=self.lr,
                momentum=self.momentum
            )
        elif self.model_type == 'atmodist_ordinal':
            self.logger.info(f"Initializing ordinal Atmodist model with {self.res_in_channels} input channels")
            self.model = OrdinalAtmodist(
                num_classes=self.num_classes,
                res_in_channels=self.res_in_channels,
                res_out_channels_list=self.res_out_channels_list,
                lr=self.lr,
                momentum=self.momentum
            )
        elif self.model_type == 'atmodist_triplet':
            self.logger.info(f"Initializing triplet Atmodist model with {self.res_in_channels} input channels")
            self.model = TripletAtmodist(
                num_classes=self.num_classes,
                res_in_channels=self.res_in_channels,
                res_out_channels_list=self.res_out_channels_list,
                lr=self.lr,
                momentum=self.momentum
            )
        elif self.model_type == 'atmodist_revised':
            self.logger.info(f"Initializing revised Atmodist model with {self.res_in_channels} input channels")
            # You would need to implement or import a revised Atmodist model
            from models.atmodist_revised import AtmodistRevised
            self.model = AtmodistRevised(
                num_classes=self.num_classes,
                res_in_channels=self.res_in_channels,
                res_out_channels_list=self.res_out_channels_list,
                dropout=self.dropout,
                lr=self.lr,
                momentum=self.momentum
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _setup_loss_function(self):
        """Setup loss function based on configuration"""
        loss_type = self.model_config.get('loss_type', 'cross_entropy')
        
        if hasattr(self.model, 'loss') and self.model.loss is not None:
            self.logger.info(f"Using model's built-in loss function: {self.model.loss.__class__.__name__}")
            self.loss_fn = self.model.loss
        elif loss_type == 'cross_entropy':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_type == 'mse':
            self.loss_fn = torch.nn.MSELoss()
        elif loss_type == 'bce':
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            self.logger.warning(f"Unknown loss type: {loss_type}, using CrossEntropyLoss")
            self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def get_model(self):
        """Get the underlying model"""
        return self.model
    
    def forward(self, inputs):
        """Forward pass of the model"""
        return self.model(inputs)
    
    def compute_loss(self, outputs, targets):
        """Compute loss for outputs and targets"""
        if hasattr(self.model, 'compute_loss'):
            # Use the model's compute_loss method if available
            return self.model.compute_loss(outputs, targets)
        else:
            # Otherwise use the configured loss function
            return self.loss_fn(outputs, targets)
    
    def train(self):
        """Set model to training mode"""
        self.model.train()
        
    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()
        
    def parameters(self):
        """Get model parameters"""
        return self.model.parameters()
        
    def to(self, device):
        """Move model to device"""
        self.model.to(device)
        return self
        
    def state_dict(self):
        """Get model state dictionary"""
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        """Load model state dictionary"""
        return self.model.load_state_dict(state_dict)
    
    def save(self, filepath):
        """Save model to file"""
        torch.save(self.model.state_dict(), filepath)
        self.logger.info(f"Model saved to {filepath}")
        
    @staticmethod
    def load(filepath, config=None):
        """Load model from file"""
        if config is None:
            raise ValueError("Config is required to load model")
            
        adapter = AtmodistModelAdapter(config)
        adapter.model.load_state_dict(torch.load(filepath))
        return adapter
