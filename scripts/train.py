#!/usr/bin/env python
"""
Training Script

This script trains weather analogue models based on the provided configuration using
PyTorch Lightning. Supports both single model and multi-model configurations.
"""
import argparse
import os
import sys
import logging
import yaml
import datetime
import threading
import time

# 設置 matplotlib 使用非交互式後端，避免在非主線程繪圖時的問題
import matplotlib
matplotlib.use('Agg')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from preprocessing.data_processor import DataProcessor
from models.lightning_model_adapter import LightningModelAdapter
from trainers.lightning_trainer import LightningTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train')

# Create a file handler for persistent logging
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'train_{timestamp}.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
logger.info(f"Logging to file: {log_file}")

def train_single_model(config, data_path, gpu_id, experiment_name=None):
    """
    Train a single model with the given configuration using PyTorch Lightning.
    
    Args:
        config: Model configuration dictionary
        data_path: Path to processed data file
        gpu_id: GPU ID to use
        experiment_name: Optional name for experiment tracking
        
    Returns:
        Dictionary of training results
    """
    # Validate configuration
    if not config or not isinstance(config, dict):
        logger.error(f"[{experiment_name}] Invalid configuration: {config}")
        return None
        
    if 'model' not in config:
        logger.error(f"[{experiment_name}] Missing model configuration in config: {config.keys()}")
        return None
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = config['model'].get('type', 'unknown')
        config_id = config.get('id', 'default')
        experiment_name = f"{model_type}_{config_id}_{timestamp}"
    
    try:
        # Initialize data processor
        logger.info(f"[{experiment_name}] Initializing data processor")
        if 'data' not in config:
            config['data'] = {}
        config['data']['processed_data_path'] = data_path
        data_processor = DataProcessor(config)
        
        # Load and prepare data
        logger.info(f"[{experiment_name}] Loading data from {data_path}")
        data = data_processor.load_processed_data()
        
        # Prepare data loaders for Lightning
        logger.info(f"[{experiment_name}] Preparing data loaders for Lightning")
        train_loader, val_loader = data_processor.prepare_lightning_data_loaders()
        
        # Initialize Lightning model
        logger.info(f"[{experiment_name}] Initializing Lightning model: {config['model'].get('type')}")
        logger.debug(f"[{experiment_name}] Model config: {config['model']}")
        model = LightningModelAdapter(config)
        
        # Create data dictionary for trainer
        data_loaders = {
            'train': train_loader,
            'validation': val_loader
        }
        
        # Initialize Lightning trainer
        logger.info(f"[{experiment_name}] Initializing Lightning trainer")
        trainer = LightningTrainer(model, data_loaders, config)
        
        # Setup experiment tracking
        logger.info(f"[{experiment_name}] Setting up experiment tracking")
        trainer.setup_tracking(experiment_name)
        
        # Train model
        logger.info(f"[{experiment_name}] Starting training")
        results = trainer.train()
        
        # Log results
        if results:
            logger.info(f"[{experiment_name}] Training completed")
            if 'best_model_path' in results:
                logger.info(f"[{experiment_name}] Best model saved to: {results['best_model_path']}")
            if 'final_model_path' in results:
                logger.info(f"[{experiment_name}] Final model saved to: {results['final_model_path']}")
        else:
            logger.warning(f"[{experiment_name}] Training completed but no results returned")
            
        # Evaluate on test set
        logger.info(f"[{experiment_name}] Evaluating model on test set")
        test_metrics = trainer.evaluate()
        
        if test_metrics:
            for metric, value in test_metrics.items():
                logger.info(f"[{experiment_name}] Test {metric}: {value:.4f}")
        
        return results
        
    except Exception as e:
        logger.exception(f"[{experiment_name}] Error during training: {str(e)}")
        return None
    
def train_model_with_thread(config_path, model_config, data_path, gpu_id, base_config=None):
    """
    Train a model in a separate thread.
    
    Args:
        config_path: Path to configuration file (for identification)
        model_config: Model configuration dictionary
        data_path: Path to processed data file
        gpu_id: GPU ID to use
        base_config: Base configuration to merge with model config
    """
    try:
        # Prepare complete configuration
        config = {}
        
        # Apply base config if provided
        if base_config:
            config.update(base_config)
        
        # For multi-model configs, extract model section
        if 'model' in model_config:
            config['model'] = model_config['model']
        
        # Extract other sections
        for section in ['training', 'data', 'wandb']:
            if section in model_config:
                config[section] = model_config[section]
        
        # Set experiment name
        experiment_name = model_config.get('id', os.path.basename(config_path).replace('.yaml', ''))
        
        # Train model
        train_single_model(config, data_path, gpu_id, experiment_name)
    
    except Exception as e:
        logger.exception(f"Error training model {model_config.get('id', 'unknown')}: {str(e)}")
        logger.error(f"Config used: {config}")  # Log the config to help debug

def main():
    parser = argparse.ArgumentParser(description='Train weather analogue models')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file (single model or multi-model)')
    parser.add_argument('--config-dir', type=str, 
                        help='Directory containing multiple configuration files')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the processed data file')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--max-parallel', type=int, default=1,
                        help='Maximum number of parallel training tasks')
    args = parser.parse_args()
    
    # Check data file
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return 1
    
    # Parse GPU IDs
    gpu_ids = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    
    # Configuration paths to process
    config_paths = []
    
    # Add single config file if provided
    if args.config and os.path.exists(args.config):
        config_paths.append(args.config)
    
    # Add config files from directory if provided
    if args.config_dir and os.path.exists(args.config_dir):
        for file in os.listdir(args.config_dir):
            if file.endswith('.yaml'):
                config_paths.append(os.path.join(args.config_dir, file))
    
    if not config_paths:
        logger.error("No configuration files found")
        return 1
    
    # Initialize threads
    threads = []
    available_gpus = list(gpu_ids)
    
    # Process each configuration
    for config_path in config_paths:
        logger.info(f"Processing configuration: {config_path}")
        
        try:
            # Direct YAML loading for simplicity and transparency
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Add ID if not present
            if 'id' not in config:
                config['id'] = os.path.splitext(os.path.basename(config_path))[0]
                
            logger.info(f"Configuration loaded: {config.keys()}")
            
            # Check if this is a multi-model configuration
            if 'models' in config and isinstance(config['models'], list):
                logger.info(f"Multi-model configuration detected in {config_path} with {len(config['models'])} models")
                
                # Extract base configuration (shared settings)
                base_config = {k: v for k, v in config.items() if k != 'models'}
                
                # Process each model configuration
                for model_config in config['models']:
                    # Wait for an available GPU
                    while not available_gpus and len(threads) >= args.max_parallel:
                        # Check if any threads have finished
                        for t in list(threads):
                            if not t.is_alive():
                                threads.remove(t)
                        
                        # If we still need to wait, sleep a bit
                        if not available_gpus and len(threads) >= args.max_parallel:
                            time.sleep(5)
                    
                    # Recycle GPUs if needed
                    if not available_gpus:
                        available_gpus = list(gpu_ids)
                    
                    # Get next GPU
                    gpu_id = available_gpus.pop(0)
                    
                    # Create thread for this model
                    thread = threading.Thread(
                        target=train_model_with_thread,
                        args=(config_path, model_config, args.data_path, gpu_id, base_config)
                    )
                    threads.append(thread)
                    thread.start()
                    
                    logger.info(f"Started training for model {model_config.get('id', 'unknown')} on GPU {gpu_id}")
            else:
                # Single model configuration
                logger.info(f"Single model configuration detected in {config_path}")
                
                # Wait for an available GPU
                while not available_gpus and len(threads) >= args.max_parallel:
                    # Check if any threads have finished
                    for t in list(threads):
                        if not t.is_alive():
                            threads.remove(t)
                    
                    # If we still need to wait, sleep a bit
                    if not available_gpus and len(threads) >= args.max_parallel:
                        time.sleep(5)
                
                # Recycle GPUs if needed
                if not available_gpus:
                    available_gpus = list(gpu_ids)
                
                # Get next GPU
                gpu_id = available_gpus.pop(0)
                
                # Create thread for this model
                thread = threading.Thread(
                    target=train_model_with_thread,
                    args=(config_path, config, args.data_path, gpu_id, None)
                )
                threads.append(thread)
                thread.start()
                
                logger.info(f"Started training for single model configuration on GPU {gpu_id}")
                
        except Exception as e:
            logger.error(f"Error processing configuration {config_path}: {str(e)}")
            continue
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    logger.info("All training tasks completed")
    return 0

if __name__ == '__main__':
    sys.exit(main())
