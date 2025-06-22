"""
Lightning Trainer Module

This module provides a PyTorch Lightning based trainer for weather analogue models,
closely following the original exp_training.py training framework.
"""
import os
import sys
import logging
import numpy as np
import torch
import yaml
from datetime import datetime
from pathlib import Path

# 設置 matplotlib 使用非交互式後端，避免在非主線程繪圖時的問題
import matplotlib
matplotlib.use('Agg')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# PyTorch Lightning related imports
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Experiment tracking will be disabled.")

class LightningTrainer:
    """
    PyTorch Lightning based trainer for weather analogue models.
    Handles training, validation, and experiment tracking.
    Designed to work with the new modular framework while keeping 
    the original training logic from exp_training.py.
    """
    def __init__(self, lightning_model, data, config):
        """
        Initialize the Lightning trainer.
        
        Args:
            lightning_model: PyTorch Lightning model instance to train
            data: Dictionary containing data loaders ('train', 'validation')
            config: Configuration dictionary
        """
        self.model = lightning_model
        self.data_loaders = data
        self.config = config
        self.training_config = config.get('training', {})
        
        # Extract training parameters
        self.max_epochs = self.training_config.get('epochs', 100)
        self.check_val_every_n_epoch = self.training_config.get('validation_interval', 1)
        self.early_stopping = self.training_config.get('early_stopping', True)
        self.patience = self.training_config.get('patience', 10)
        
        # Setup logging
        self.logger = logging.getLogger('LightningTrainer')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
        # Initialize tracking
        self.tracking_enabled = False
        self.wandb_logger = None
        self.experiment_dir = None
        self.trainer = None
        
    def setup_tracking(self, experiment_name=None):
        """
        Setup experiment tracking with Weights & Biases.
        
        Args:
            experiment_name: Optional name for this experiment
        """
        if not WANDB_AVAILABLE:
            self.logger.warning("wandb not available, skipping tracking setup")
            return
            
        # Extract wandb configuration
        wandb_config = self.config.get('wandb', {})
        project = wandb_config.get('project', 'weather-analogue')
        entity = wandb_config.get('entity', None)
        tags = wandb_config.get('tags', [])
        api_key = wandb_config.get('api_key', None)
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = self.config['model']['type']
            experiment_name = f"{model_type}_{timestamp}"
            
        # Create experiment directory
        self.experiment_dir = os.path.join('experiments', experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(self.experiment_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
            
        # Initialize wandb logger
        if api_key:
            wandb.login(key=api_key)
            
        self.wandb_logger = WandbLogger(
            project=project,
            entity=entity,
            name=experiment_name,
            log_model="all"
        )
        
        self.tracking_enabled = True
        self.logger.info(f"Initialized experiment tracking: {experiment_name}")
        
    def setup_trainer(self):
        """
        Setup the PyTorch Lightning trainer.
        """
        # Setup callbacks
        callbacks = []
        
        # Model checkpoint callback
        checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints') if self.experiment_dir else 'checkpoints'
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            dirpath=checkpoint_dir,
            filename='{epoch:02d}-{val_acc:.2f}',
            save_top_k=1,
            mode='max'
        )
        callbacks.append(checkpoint_callback)
        
        # Learning rate monitor
        if self.tracking_enabled:
            lr_monitor = LearningRateMonitor(logging_interval='epoch')
            callbacks.append(lr_monitor)
        
        # Early stopping
        if self.early_stopping:
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                mode='min'
            )
            callbacks.append(early_stop_callback)
        
        # Create trainer
        logger = self.wandb_logger if self.tracking_enabled else True
        
        # 計算適當的記錄步驟間隔
        log_every_n_steps = 10  # 預設值
        if 'batch_size' in self.training_config:
            batch_size = self.training_config['batch_size']
            # 根據批次大小調整記錄頻率 - 較小的批次應該較少記錄
            if batch_size <= 16:
                log_every_n_steps = 20
            elif batch_size <= 32:
                log_every_n_steps = 15
            elif batch_size <= 64:
                log_every_n_steps = 10
            else:
                log_every_n_steps = 5
        
        self.trainer = Trainer(
            max_epochs=self.max_epochs,
            logger=logger,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            callbacks=callbacks,
            accelerator="auto",
            deterministic=True,
            log_every_n_steps=log_every_n_steps,
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=True
        )
        
        self.logger.info(f"Trainer initialized with max epochs: {self.max_epochs}")
        
    def train(self):
        """
        Train the model.
        
        Returns:
            Dictionary of training results
        """
        if self.trainer is None:
            self.setup_trainer()
            
        train_loader = self.data_loaders.get('train')
        val_loader = self.data_loaders.get('validation')
        
        if not train_loader:
            self.logger.error("No training data loader provided")
            return None
            
        self.logger.info(f"Starting training for {self.max_epochs} epochs")
        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        # Save the model
        self.save_model()
        
        # Return training results
        return {
            'best_model_path': self.trainer.checkpoint_callback.best_model_path,
            'best_model_score': self.trainer.checkpoint_callback.best_model_score,
            'final_model_path': os.path.join(self.experiment_dir, 'model.pt') if self.experiment_dir else 'model.pt'
        }
        
    def save_model(self, name='model.pt'):
        """
        Save model to experiment directory.
        
        Args:
            name: Name for the saved model file
        
        Returns:
            Path to the saved model
        """
        if self.experiment_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = os.path.join('experiments', f"exp_{timestamp}")
            os.makedirs(self.experiment_dir, exist_ok=True)
            
        model_path = os.path.join(self.experiment_dir, name)
        
        # Save the model
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Saved model to {model_path}")
        
        return model_path
        
    def finish(self):
        """
        Clean up and finish training.
        """
        if self.tracking_enabled and WANDB_AVAILABLE:
            wandb.finish()
            
    def evaluate(self, test_loader=None):
        """
        Evaluate the model on a test set.
        
        Args:
            test_loader: DataLoader for the test set. If None, uses validation set.
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.trainer is None:
            self.setup_trainer()
            
        # Use validation loader if test loader not provided
        eval_loader = test_loader if test_loader else self.data_loaders.get('validation')
        
        if not eval_loader:
            self.logger.error("No evaluation data loader provided")
            return None
            
        self.logger.info("Starting model evaluation")
        results = self.trainer.test(self.model, dataloaders=eval_loader)
        
        if results and len(results) > 0:
            metrics = results[0]
            self.logger.info(f"Evaluation results: {metrics}")
            
            # Log metrics to wandb if tracking is enabled
            if self.tracking_enabled and WANDB_AVAILABLE:
                wandb.log({"test_" + k: v for k, v in metrics.items()})
                
            return metrics
        else:
            self.logger.warning("No evaluation results returned")
            return None
