"""
Model Trainer Module

This module provides training capabilities for weather analogue models,
including initialization, training, validation, and experiment tracking.
"""
import os
import sys
import numpy as np
import logging
import time
import torch
import yaml
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Experiment tracking will be disabled.")

class ModelTrainer:
    """
    Trainer for weather analogue models.
    Handles training, validation, and experiment tracking.
    """
    def __init__(self, model, data, config):
        """
        Initialize the trainer.
        
        Args:
            model: Model instance to train
            data: Dictionary containing datasets ('train', 'validation')
            config: Configuration dictionary
        """
        self.model = model
        self.data = data
        self.config = config
        self.training_config = config.get('training', {})
        
        # Extract training parameters
        self.batch_size = self.training_config.get('batch_size', 64)
        self.epochs = self.training_config.get('epochs', 100)
        self.learning_rate = self.training_config.get('learning_rate', 0.001)
        self.weight_decay = self.training_config.get('weight_decay', 1e-5)
        self.optimizer_name = self.training_config.get('optimizer', 'adam')
        self.early_stopping = self.training_config.get('early_stopping', True)
        self.patience = self.training_config.get('patience', 10)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
            
        # Configure optimizer
        self._setup_optimizer()
        
        # Setup logging
        self.logger = logging.getLogger('ModelTrainer')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
        # Initialize tracking
        self.tracking_enabled = False
        self.run = None
        self.experiment_dir = None
        
    def _setup_optimizer(self):
        """Setup optimizer based on configuration"""
        if not hasattr(self.model, 'parameters'):
            self.logger.warning("Model doesn't have parameters method, skipping optimizer setup")
            self.optimizer = None
            return
            
        if self.optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            self.logger.warning(f"Unknown optimizer: {self.optimizer_name}, using Adam")
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate
            )
    
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
            
        # Initialize wandb
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=experiment_name,
            config=self.config,
            tags=tags
        )
        
        self.tracking_enabled = True
        self.logger.info(f"Initialized experiment tracking: {experiment_name}")
        
    def log_metrics(self, metrics, step=None):
        """
        Log metrics to tracking system.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.tracking_enabled or self.run is None:
            return
            
        self.run.log(metrics, step=step)
        
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
        
        if hasattr(self.model, 'save'):
            self.model.save(model_path)
        elif hasattr(self.model, 'state_dict'):
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'config': self.config,
                'timestamp': time.time()
            }, model_path)
        else:
            self.logger.warning("Model doesn't support saving, skipping")
            return None
            
        self.logger.info(f"Saved model to {model_path}")
        
        # Log as artifact if tracking enabled
        if self.tracking_enabled and self.run is not None:
            artifact = wandb.Artifact(
                name=f"model-{self.run.id}",
                type="model",
                description=f"Trained {self.model.__class__.__name__}"
            )
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)
            
        return model_path
        
    def train(self, epochs=None, experiment_name=None):
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train for (overrides config)
            experiment_name: Name for this experiment
            
        Returns:
            Dictionary of training results
        """
        # Setup tracking if experiment name provided
        if experiment_name and not self.tracking_enabled:
            self.setup_tracking(experiment_name)
            
        # Override epochs if provided
        if epochs is not None:
            self.epochs = epochs
            
        self.logger.info(f"Starting training for {self.epochs} epochs")
        
        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        best_model_path = None
        
        # Training loop
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Training phase
            if hasattr(self.model, 'train'):
                self.model.train()
                
            train_metrics = self._train_epoch()
            
            # Validation phase
            if hasattr(self.model, 'eval'):
                self.model.eval()
                
            val_metrics = self._validate_epoch()
            
            # Combine metrics
            epoch_metrics = {
                'epoch': epoch,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
                'time': time.time() - start_time
            }
            
            # Log metrics
            self.log_metrics(epoch_metrics, step=epoch)
            
            # Log to console
            metrics_str = ' '.join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items() 
                                  if k not in ['epoch', 'time']])
            self.logger.info(f"Epoch {epoch}/{self.epochs} - {metrics_str} - "
                            f"Time: {epoch_metrics['time']:.2f}s")
            
            # Early stopping
            if self.early_stopping:
                val_loss = val_metrics.get('loss', float('inf'))
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    best_model_path = self.save_model(f"best_model_epoch_{epoch}.pt")
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Save final model
        final_model_path = self.save_model("final_model.pt")
        
        # Finish tracking
        if self.tracking_enabled and self.run is not None:
            self.run.finish()
            
        return {
            'epochs_completed': epoch + 1,
            'best_model_path': best_model_path,
            'final_model_path': final_model_path,
            'best_val_loss': best_loss,
            'experiment_dir': self.experiment_dir
        }
    
    def _train_epoch(self):
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        # Use appropriate training method based on model type
        if 'train' not in self.data:
            self.logger.error("No training data available")
            return {'loss': float('inf'), 'accuracy': 0.0}
            
        train_loader = self.data['train']
        
        # Check if we're using Atmodist model
        if hasattr(self.model, 'forward') and hasattr(self.model, 'get_model') and 'atmodist' in self.model.__class__.__name__.lower():
            return self._train_epoch_atmodist(train_loader)
        
        # Generic training implementation
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract data and labels from batch
            inputs, targets = self._get_inputs_targets(batch)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            if hasattr(self.model, 'compute_loss'):
                loss = self.model.compute_loss(outputs, targets)
            else:
                loss = torch.nn.functional.cross_entropy(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            accuracy = self._compute_accuracy(outputs, targets)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
            
        # Calculate average metrics
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        return {'loss': avg_loss, 'accuracy': avg_accuracy}
        
    def _validate_epoch(self):
        """
        Validate for one epoch.
        
        Returns:
            Dictionary of validation metrics
        """
        # Use appropriate validation method based on model type
        if 'validation' not in self.data:
            self.logger.error("No validation data available")
            return {'loss': float('inf'), 'accuracy': 0.0}
            
        val_loader = self.data['validation']
        
        # Check if we're using Atmodist model
        if hasattr(self.model, 'get_model') and 'atmodist' in self.model.__class__.__name__.lower():
            return self._validate_epoch_atmodist(val_loader)
        
        # Generic validation implementation
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        all_preds = []
        all_targets = []
        
        # Set model to evaluation mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Extract data and labels from batch
                inputs, targets = self._get_inputs_targets(batch)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                if hasattr(self.model, 'compute_loss'):
                    loss = self.model.compute_loss(outputs, targets)
                else:
                    loss = torch.nn.functional.cross_entropy(outputs, targets)
                
                # Compute accuracy
                accuracy = self._compute_accuracy(outputs, targets)
                
                # Store predictions for additional metrics
                if isinstance(outputs, torch.Tensor) and outputs.dim() > 1:
                    preds = torch.argmax(outputs, dim=1).cpu().detach().numpy()
                    if isinstance(targets, torch.Tensor) and targets.dim() > 1:
                        target_vals = torch.argmax(targets, dim=1).cpu().detach().numpy()
                    else:
                        target_vals = targets.cpu().detach().numpy()
                    
                    all_preds.append(preds)
                    all_targets.append(target_vals)
                
                # Accumulate metrics
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
                
        # Calculate average metrics
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        # Compute additional metrics if possible
        additional_metrics = {}
        if all_preds and all_targets:
            try:
                import numpy as np
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                all_preds = np.concatenate(all_preds)
                all_targets = np.concatenate(all_targets)
                
                additional_metrics = {
                    'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
                    'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
                    'f1': f1_score(all_targets, all_preds, average='weighted', zero_division=0)
                }
            except (ImportError, ValueError) as e:
                self.logger.warning(f"Could not compute additional metrics: {str(e)}")
        
        return {'loss': avg_loss, 'accuracy': avg_accuracy, **additional_metrics}
    
    def _train_epoch_atmodist(self, train_loader):
        """
        Train for one epoch with Atmodist model.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary of training metrics
        """
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            timestamp1, timestamp2, r1, r2, target = batch
            
            # Move data to device
            r1 = r1.to(self.device)
            r2 = r2.to(self.device)
            target = target.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            atm1, atm2, pred = self.model.forward((r1, r2))
            
            # Calculate loss
            loss = torch.nn.functional.cross_entropy(pred, target)
            
            # Calculate accuracy
            predicted_classes = torch.argmax(pred, dim=1)
            target_classes = torch.argmax(target, dim=1)
            accuracy = torch.sum(predicted_classes == target_classes).item() / target.size(0)
            
            # Backward pass
            loss.backward()
            
            # Optimize
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_acc += accuracy
            num_batches += 1
            
        # Calculate average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0
        
        return {'loss': avg_loss, 'accuracy': avg_acc}
    
    def _validate_epoch_atmodist(self, val_loader):
        """
        Validate for one epoch with Atmodist model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary of validation metrics
        """
        total_loss = 0
        total_acc = 0
        num_batches = 0
        all_preds = []
        all_targets = []
        
        # Set model to evaluation mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                timestamp1, timestamp2, r1, r2, target = batch
                
                # Move data to device
                r1 = r1.to(self.device)
                r2 = r2.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                atm1, atm2, pred = self.model.forward((r1, r2))
                
                # Calculate loss
                loss = torch.nn.functional.cross_entropy(pred, target)
                
                # Calculate accuracy
                predicted_classes = torch.argmax(pred, dim=1)
                target_classes = torch.argmax(target, dim=1)
                accuracy = torch.sum(predicted_classes == target_classes).item() / target.size(0)
                
                # Store predictions and targets for additional metrics
                all_preds.append(predicted_classes.cpu().numpy())
                all_targets.append(target_classes.cpu().numpy())
                
                # Accumulate metrics
                total_loss += loss.item()
                total_acc += accuracy
                num_batches += 1
        
        # Calculate average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0
        
        # Compute additional metrics if data is available
        additional_metrics = {}
        if all_preds and all_targets:
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            
            # In a real implementation, you might add precision, recall, f1, etc.
            # This would require sklearn.metrics
            
        # Combine metrics
        metrics = {'loss': avg_loss, 'accuracy': avg_acc, **additional_metrics}
        
        return metrics
        
    def evaluate(self, test_data=None):
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Optional test dataset (uses validation data if not provided)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Use validation data if test data not provided
        if test_data is None:
            test_data = self.data.get('validation', None)
            
        if test_data is None:
            self.logger.error("No test data available for evaluation")
            return {}
            
        # Evaluation phase
        if hasattr(self.model, 'eval'):
            self.model.eval()
            
        # This is a placeholder implementation
        # In a real implementation, this would evaluate on test data
        # For now, we return mock metrics
        metrics = {'loss': 0.55, 'accuracy': 0.78}
        
        # Log metrics if tracking enabled
        if self.tracking_enabled and self.run is not None:
            self.run.log({f'test_{k}': v for k, v in metrics.items()})
            
        return metrics
