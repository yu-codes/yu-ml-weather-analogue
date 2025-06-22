"""
Lightning Model Adapter Module

This module provides adapter classes for using Atmodist models with PyTorch Lightning.
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 設置 matplotlib 使用非交互式後端，避免在非主線程繪圖時的問題
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.atmodist import Atmodist, OrdinalAtmodist, TripletAtmodist

class LightningModelAdapter(pl.LightningModule):
    """
    PyTorch Lightning adapter for Atmodist models.
    Converts standard PyTorch models to be compatible with PyTorch Lightning.
    """
    def __init__(self, config):
        """
        Initialize the Lightning model adapter.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        self.model_config = config.get('model', {})
        self.model_type = self.model_config.get('type', 'atmodist_standard')
        self.save_hyperparameters(config)
        
        # Get model parameters from config
        self.num_classes = self.model_config.get('num_classes', 8)  # Default for 3h frequency with 24h interval
        self.res_in_channels = self.model_config.get('res_in_channels', 5)  # Default for 5 variables
        self.res_out_channels_list = self.model_config.get('res_out_channels_list', (16, 32, 64, 128))
        self.lr = float(self.model_config.get('learning_rate', 0.01))
        self.momentum = float(self.model_config.get('momentum', 0.9))
        self.weight_decay = float(self.model_config.get('weight_decay', 1e-5))
        
        # Initialize model
        self._init_model()
        
        # Setup loss function
        self._setup_loss_function()
        
        # For validation metrics
        self.validation_step_outputs = []
    
    def _init_model(self):
        """Initialize the appropriate model based on configuration"""
        if self.model_type == 'atmodist_standard':
            self.model = Atmodist(
                num_classes=self.num_classes,
                res_in_channels=self.res_in_channels,
                res_out_channels_list=self.res_out_channels_list,
                lr=self.lr,
                momentum=self.momentum
            )
        elif self.model_type == 'atmodist_ordinal':
            self.model = OrdinalAtmodist(
                num_classes=self.num_classes,
                res_in_channels=self.res_in_channels,
                res_out_channels_list=self.res_out_channels_list,
                lr=self.lr,
                momentum=self.momentum
            )
        elif self.model_type == 'atmodist_triplet':
            self.model = TripletAtmodist(
                res_in_channels=self.res_in_channels,
                res_out_channels_list=self.res_out_channels_list,
                lr=self.lr,
                momentum=self.momentum
            )
        elif self.model_type == 'atmodist_revised':
            try:
                from models.atmodist_revised import Atmodist as AtmodistRevised
                self.model = AtmodistRevised(
                    num_classes=self.num_classes,
                    res_in_channels=self.res_in_channels,
                    res_out_channels_list=self.res_out_channels_list,
                    lr=self.lr,
                    momentum=self.momentum
                )
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Failed to import AtmodistRevised: {str(e)}")
        else:
            raise ValueError(f"Unknown or unsupported model type: {self.model_type}")
            
    def _setup_loss_function(self):
        """Setup loss function based on configuration"""
        loss_type = self.model_config.get('loss_type', 'cross_entropy')
        
        if hasattr(self.model, 'loss') and self.model.loss is not None:
            self.loss_fn = self.model.loss
        elif loss_type == 'cross_entropy':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_type == 'mse':
            self.loss_fn = torch.nn.MSELoss()
        elif loss_type == 'bce':
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
            
        # 初始化驗證輸出列表
        self.validation_step_outputs = []
        self.test_step_outputs = []  # 添加測試輸出列表
    
    def forward(self, inputs):
        """Forward pass through the model"""
        return self.model(inputs)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer_name = self.model_config.get('optimizer', 'sgd')
        
        # Ensure parameters are floats
        lr = float(self.lr)
        weight_decay = float(self.weight_decay)
        momentum = float(self.momentum) if hasattr(self, 'momentum') else 0.9
        
        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:  # default to SGD
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
            
        # Add scheduler if specified
        scheduler_name = self.model_config.get('scheduler', None)
        if scheduler_name is None:
            return optimizer
            
        if scheduler_name == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=8,
                cooldown=0,
                min_lr=1e-5,
                eps=4e-2,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }
        elif scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.model_config.get('t_max', 10),
                eta_min=self.model_config.get('min_lr', 1e-6)
            )
            return [optimizer], [scheduler]
            
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        timestamp1, timestamp2, r1, r2, target = batch
        atm1, atm2, pred = self.model((r1, r2))
        loss, accuracy = self._calculate_loss_and_accuracy(pred, target)
        
        # Log metrics - log less frequently on step to avoid cluttering W&B
        # but ensure every step contributes to epoch aggregation
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                
        return {'loss': loss, 'acc': accuracy}
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        timestamp1, timestamp2, r1, r2, target = batch
        atm1, atm2, pred = self.model((r1, r2))
        loss, accuracy = self._calculate_loss_and_accuracy(pred, target)
        
        # Log metrics - consistently log at epoch level for validation
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # Store outputs for epoch end processing
        self.validation_step_outputs.append((pred, target))
        
        return {'val_loss': loss, 'val_acc': accuracy}
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        timestamp1, timestamp2, r1, r2, target = batch
        atm1, atm2, pred = self.model((r1, r2))
        loss, accuracy = self._calculate_loss_and_accuracy(pred, target)
        
        # Log metrics - consistently log only at epoch level for test
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # Store outputs for epoch end processing in test_step_outputs instead of validation_step_outputs
        self.test_step_outputs.append((pred, target))
        
        return {'test_loss': loss, 'test_acc': accuracy}
    
    def on_validation_epoch_end(self):
        """Called at the end of validation to compute and log metrics"""
        all_preds = []
        all_targets = []
        
        for pred, target in self.validation_step_outputs:
            all_preds.append(pred)
            all_targets.append(target)
            
        if not all_preds:
            self.validation_step_outputs.clear()
            return
            
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Convert to class indices
        predicted_indices = torch.argmax(all_preds, dim=1).cpu().numpy()
        target_indices = torch.argmax(all_targets, dim=1).cpu().numpy()
        
        # Compute metrics with explicit zero_division handling
        conf_matrix = confusion_matrix(target_indices, predicted_indices)
        accuracy = accuracy_score(target_indices, predicted_indices)
        
        # Handle zero division explicitly for all metrics
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = precision_score(
                target_indices, predicted_indices, average='weighted', zero_division=0
            )
            recall = recall_score(
                target_indices, predicted_indices, average='weighted', zero_division=0
            )
            f1 = f1_score(target_indices, predicted_indices, average='weighted', zero_division=0)
            
        # Use older sklearn API if zero_division not supported
        try:
            class_report = classification_report(target_indices, predicted_indices, zero_division=0)
        except TypeError:
            # Fall back to default behavior for older scikit-learn versions
            class_report = classification_report(target_indices, predicted_indices)
        
        # Log metrics through Lightning
        self.log('val_accuracy', accuracy, on_epoch=True)
        self.log('val_precision', precision, on_epoch=True)
        self.log('val_recall', recall, on_epoch=True)
        self.log('val_f1', f1, on_epoch=True)
        
        # Print classification report
        print(f"Classification Report:\n{class_report}")
        
        # Log confusion matrix and class report
        self._log_confusion_matrix(conf_matrix)
        self._log_classification_report(class_report)
            
        # Clear outputs
        self.validation_step_outputs.clear()
        
    def _log_classification_report(self, class_report):
        """Log classification report to W&B"""
        try:
            import wandb
            report_lines = class_report.split('\n')
            report_table = wandb.Table(columns=["Class", "Precision", "Recall", "F1-Score", "Support"])
            for line in report_lines[2:-5]:  # Skip header and footer
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        class_name = parts[0]
                        precision = float(parts[1])
                        recall = float(parts[2])
                        f1 = float(parts[3])
                        support = int(parts[4])
                        report_table.add_data(class_name, precision, recall, f1, support)
            wandb.log({"classification_report": report_table})
        except (ImportError, Exception) as e:
            print(f"Failed to log classification report to W&B: {e}")
    
    def on_test_epoch_end(self):
        """Called at the end of testing to compute and log metrics"""
        # Save current validation outputs
        temp_outputs = self.validation_step_outputs
        # Use test outputs for calculations
        self.validation_step_outputs = self.test_step_outputs
        # Run validation logic
        self.on_validation_epoch_end()
        # Restore validation outputs
        self.validation_step_outputs = temp_outputs
        # Clear test outputs separately
        self.test_step_outputs.clear()
    
    def _calculate_loss_and_accuracy(self, pred, target):
        """Calculate loss and accuracy"""
        loss = self.loss_fn(pred, target)
        predicted_indices = torch.argmax(pred, dim=1)
        target_indices = torch.argmax(target, dim=1)
        correct_predictions = torch.sum(predicted_indices == target_indices).item()
        accuracy = correct_predictions / len(predicted_indices)
        return loss, accuracy
    
    def _log_confusion_matrix(self, conf_matrix):
        """Log confusion matrix to tracking system"""
        try:
            import wandb
            has_wandb = True
        except ImportError:
            has_wandb = False
            
        # 避免在非主線程中使用 matplotlib GUI
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式後端，避免 GUI 問題
            
        num_classes = conf_matrix.shape[0]
        # 設置標籤，从 3 开始，以 3 的倍數遞增
        labels = [
            f"{(i + 1) * 3}h" for i in range(num_classes)
        ]
        
        # 原始混淆矩陣
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xticks(np.arange(num_classes) + 0.5)
        ax.set_yticks(np.arange(num_classes) + 0.5)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Time Lag")
        ax.set_ylabel("Actual Time Lag")
        ax.set_title("Confusion Matrix (Original)")
        
        # 保存圖像並使用 Lightning logger 上傳
        if self.logger:
            cm_plot_path = f"confusion_matrix_original_epoch_{self.current_epoch}.png"
            plt.savefig(cm_plot_path)
            # Log using Lightning logger
            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({"confusion_matrix_original": wandb.Image(cm_plot_path)})
        
        plt.close(fig)
        
        # 歸一化混淆矩陣
        conf_matrix_normalized = (
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = sns.heatmap(
            conf_matrix_normalized,
            annot=False,
            cmap="Blues",
            norm=LogNorm(vmin=1e-2, vmax=1),
            ax=ax,
        )
        
        ax.set_xticks(np.arange(num_classes) + 0.5)
        ax.set_yticks(np.arange(num_classes) + 0.5)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Time Lag")
        ax.set_ylabel("Actual Time Lag")
        ax.set_title("Confusion Matrix (Normalized)")
        
        # 設置 color bar 刻度
        colorbar = heatmap.collections[0].colorbar
        colorbar.set_ticks([1e-2, 1e-1, 1])
        colorbar.set_ticklabels([r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])
        
        # 保存圖像並使用 Lightning logger 上傳
        if self.logger:
            cm_norm_path = f"confusion_matrix_normalized_epoch_{self.current_epoch}.png"
            plt.savefig(cm_norm_path)
            # Log using Lightning logger
            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({"confusion_matrix_normalized": wandb.Image(cm_norm_path)})
            
        plt.close(fig)
