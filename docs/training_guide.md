# Weather Analogue Model Training Guide - PyTorch Lightning Version

This document provides a comprehensive guide for training weather analogue models using PyTorch Lightning, covering configuration, data preparation, training processes, and result analysis.

## Framework Modules and Training Process Overview

This weather analogue model framework encompasses the following key modules and processes, from data preparation to model deployment:

### Core Files Involved in Training

| File Path | Role | Description |
|---------|------|----------|
| `scripts/train.py` | Main Training Script | Unified training entry point that handles command line arguments and coordinates the training process |
| `configs/manager.py` | Configuration Manager | Reads, parses, and validates configuration files |
| `configs/lightning_example.yaml` | Configuration File | Defines model, training parameters, and experiment settings |
| `models/lightning_model_adapter.py` | Model Adapter | Converts models to Lightning format, defines training/validation/testing steps |
| `models/atmodist_revised.py` | Model Definition | Implements the AtmoDist model architecture and forward pass logic |
| `trainers/lightning_trainer.py` | Trainer | Encapsulates PyTorch Lightning Trainer, configures callbacks and training environment |
| `preprocessing/data_processor.py` | Data Processor | Responsible for data loading, preprocessing, augmentation, and batch generation |
| `data/dataset.py` | Dataset Definition | Implements PyTorch Dataset class, handles raw data files |

### Complete Training Process

1. **Configuration Loading**:
   - `train.py` parses command line arguments
   - `configs/manager.py` reads and processes YAML configuration files
   - Determines whether to use single-model or multi-model training based on configuration

2. **Data Preparation**:
   - `preprocessing/data_processor.py` loads raw HDF5 data files
   - `data/dataset.py` creates and prepares PyTorch Datasets
   - Applies data preprocessing (standardization, normalization, etc.)
   - Generates training, validation, and test data loaders

3. **Model Initialization**:
   - Selects model class based on `model.type` in configuration (e.g., `atmodist_revised`)
   - `models/lightning_model_adapter.py` wraps the model as a Lightning module
   - Sets up loss functions, optimizers, and learning rate schedulers

4. **Training Setup**:
   - `trainers/lightning_trainer.py` configures the Lightning Trainer
   - Sets up checkpointing, early stopping, logging, and other callbacks
   - Initializes experiment tracking with Weights & Biases if configured

5. **Training Execution**:
   - Executes the training/validation/testing loops
   - Records metrics at each training and validation step
   - Periodically calculates and logs confusion matrices and classification reports
   - Automatically saves model checkpoints

6. **Result Analysis**:
   - Generates final evaluation metrics after training
   - Logs results to Weights & Biases
   - Saves trained models for inference

7. **Model Deployment**:
   - Uses trained models for weather analogue search
   - Analyzes model performance through `notebooks/analogue_search_atmodist.ipynb`

## 1. Configuration System

### Single-Model Configuration

The single-model configuration file defines all parameters for a single model:

```yaml
# configs/lightning_example.yaml
id: lightning_example
model:
  type: atmodist_revised
  num_classes: 8  # Based on time interval and frequency
  res_in_channels: 5  # Number of input variables
  res_out_channels_list: [16, 32, 64, 128]
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 1e-5
  optimizer: sgd  # Supported: sgd, adam
  scheduler: reduce_on_plateau  # Supported: reduce_on_plateau, cosine, null
  loss_type: cross_entropy
training:
  epochs: 20
  batch_size: 256
  train_val_split: 0.8
  validation_interval: 1
  early_stopping: true
  patience: 10
  log_every_n_steps: 10  # Log metrics every N steps
data:
  variables: [d2m, u, v, msl, r]
  frequency: 3h
  time_interval: 24
wandb:
  project: weather-analogue-lightning
  entity: your-username
  tags: [lightning, atmodist]
use_lightning: true  # Enable PyTorch Lightning
```

### Multi-Model Configuration

Multi-model configuration supports defining multiple models in a single file:

```yaml
# configs/multi_model_example.yaml
id: multi_model_example
# Shared settings for all models
training:
  epochs: 20
  early_stopping: true
data:
  variables: [d2m, u, v, msl, r]
  frequency: 3h
wandb:
  project: weather-analogue-lightning
# Model list
models:
  - id: model_small
    model:
      type: atmodist_revised
      num_classes: 8
      res_out_channels_list: [16, 32, 64]
      learning_rate: 0.01
    training:
      batch_size: 128
      
  - id: model_large
    model:
      type: atmodist_revised
      num_classes: 8
      res_out_channels_list: [32, 64, 128, 256]
      learning_rate: 0.005
    training:
      batch_size: 64
use_lightning: true
```

## 2. Training Process

### Basic Training Command

Use the unified training script `train.py` for all training:

```bash
python scripts/train.py --config <config_file_path> --data-path <data_file_path> --gpus <gpu_list>
```

### Single-Model Training Example

```bash
python scripts/train.py --config configs/lightning_example.yaml \
  --data-path data/processed/d2muvmslr_3h_none_log_20012020_weighted.h5 \
  --gpus 0
```

### Multi-Model Training Example

```bash
# Using multi-model configuration file
python scripts/train.py --config configs/multi_model_example.yaml \
  --data-path data/processed/d2muvmslr_3h_none_log_20012020_weighted.h5 \
  --gpus 0,1 --max-parallel 2

# Or using a configuration directory
python scripts/train.py --config-dir configs/models \
  --data-path data/processed/d2muvmslr_3h_none_log_20012020_weighted.h5 \
  --gpus 0,1,2,3 --max-parallel 4
```

### Command Line Arguments

| Parameter | Description |
|------|------|
| `--config` | Configuration file path |
| `--config-dir` | Directory containing multiple configuration files |
| `--data-path` | Path to the processed data file |
| `--gpus` | List of GPU IDs (comma-separated) |
| `--max-parallel` | Maximum number of parallel training tasks |
| `--seed` | Random seed |

## 3. PyTorch Lightning Core Components

### LightningModelAdapter

`LightningModelAdapter` adapts the Atmodist model to the PyTorch Lightning framework:

```python
class LightningModelAdapter(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._init_model()
        self._setup_loss_function()
        
    def training_step(self, batch, batch_idx):
        # Training step implementation
        
    def validation_step(self, batch, batch_idx):
        # Validation step implementation
        
    def configure_optimizers(self):
        # Configure optimizers and learning rate schedulers
```

Main responsibilities:

- Implement Lightning training/validation/testing methods
- Configure optimizers and learning rate schedulers
- Handle metric logging and visualization

### LightningTrainer

`LightningTrainer` encapsulates the PyTorch Lightning Trainer:

```python
class LightningTrainer:
    def __init__(self, config, data_loaders, model, experiment_id, gpus=None):
        self.config = config
        self.data_loaders = data_loaders
        self.model = model
        self.experiment_id = experiment_id
        self._setup_trainer(gpus)
        
    def train(self):
        # Start training
        
    def _setup_trainer(self, gpus):
        # Configure Lightning Trainer instance
```

Main responsibilities:

- Set up the training environment (devices, progress bars, logging, etc.)
- Configure callbacks (early stopping, checkpoints, etc.)
- Start and manage the training process

## 4. Important Implementation Details

### Loss and Metric Logging

```python
# In training_step
def training_step(self, batch, batch_idx):
    # Calculate loss
    loss, accuracy = self._calculate_loss_and_accuracy(pred, target)
    
    self.log('train_loss', loss, on_step=log_on_step, on_epoch=True, prog_bar=True)
    self.log('train_acc', accuracy, on_step=log_on_step, on_epoch=True, prog_bar=True)
    
    # Return loss, not a dictionary, to avoid implicit logging by Lightning
    return loss
```

### Visualization and Confusion Matrix

Calculate and log the confusion matrix and classification report at the end of each epoch:

```python
def on_validation_epoch_end(self):
    # Collect predictions and targets
    all_preds = torch.cat([p for p, _ in self.validation_step_outputs], dim=0)
    all_targets = torch.cat([t for _, t in self.validation_step_outputs], dim=0)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(target_indices, predicted_indices)
    
    # Plot using matplotlib and save to W&B
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    
    # Upload to W&B
    self.logger.experiment.log({"confusion_matrix": wandb.Image(fig)})
```

## 5. Using Trained Models

### Model Loading and Inference

After training, models are automatically saved in the `experiments/{experiment_id}` directory. You can load and use trained models with the following steps:

```python
import torch
from models.lightning_model_adapter import LightningModelAdapter

# Load model checkpoint
checkpoint_path = "experiments/lightning_example/checkpoints/best_model.ckpt"
model = LightningModelAdapter.load_from_checkpoint(checkpoint_path)
model.eval()

# Perform inference
with torch.no_grad():
    inputs = torch.randn(1, 5, 32, 32)  # Example input
    outputs = model(inputs)
    predictions = torch.argmax(outputs, dim=1)
```

## 6. Important Notes

- Multi-model configurations must include a `models` list
- Each model must have a unique `id`
- Shared settings will be applied to all models but can be overridden by model-specific settings
- The framework automatically handles experiment directories and model saving
