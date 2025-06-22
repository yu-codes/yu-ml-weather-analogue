#!/usr/bin/env python
"""
Test script for W&B logging in Lightning trainer

This script tests the W&B logging functionality in the Lightning trainer
to ensure that metrics are properly logged at both step and epoch level.
"""
import os
import sys
import yaml
import logging
from pathlib import Path

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
logger = logging.getLogger('test_wandb_logging')

def run_test(config_path, data_path, num_epochs=2):
    """
    Run a short training session to test W&B logging
    
    Args:
        config_path: Path to configuration file
        data_path: Path to processed data file
        num_epochs: Number of epochs to run (default: 2)
    """
    logger.info(f"Testing W&B logging with config: {config_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override epochs for quick test
    if 'training' not in config:
        config['training'] = {}
    config['training']['epochs'] = num_epochs
    
    # Ensure data path is set
    if 'data' not in config:
        config['data'] = {}
    config['data']['processed_data_path'] = data_path
    
    # Create a unique experiment name for testing
    experiment_name = f"wandb_log_test_{Path(config_path).stem}"
    
    # Initialize data processor
    logger.info("Initializing data processor")
    data_processor = DataProcessor(config)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = data_processor.load_processed_data()
    
    # Prepare data loaders
    logger.info("Preparing data loaders")
    train_loader, val_loader = data_processor.prepare_lightning_data_loaders()
    
    # Initialize model
    logger.info("Initializing Lightning model")
    model = LightningModelAdapter(config)
    
    # Create data loaders dictionary
    data_loaders = {
        'train': train_loader,
        'validation': val_loader
    }
    
    # Initialize trainer
    logger.info("Initializing Lightning trainer")
    trainer = LightningTrainer(model, data_loaders, config)
    
    # Setup tracking
    logger.info("Setting up W&B tracking")
    trainer.setup_tracking(experiment_name)
    
    # Run training
    logger.info(f"Starting training for {num_epochs} epochs")
    results = trainer.train()
    
    # Log results
    if results:
        logger.info("Training completed successfully")
        logger.info(f"Best model path: {results.get('best_model_path', 'N/A')}")
    else:
        logger.warning("Training failed or returned no results")
    
    # Finish training
    trainer.finish()
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test W&B logging in Lightning trainer")
    parser.add_argument('--config', type=str, default='configs/lightning_example.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data-path', type=str, 
                        required=True,
                        help='Path to processed data file')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs to run (default: 2)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)
    
    run_test(args.config, args.data_path, args.epochs)
