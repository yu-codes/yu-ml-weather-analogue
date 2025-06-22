#!/usr/bin/env python
"""
Test Model Loading

This script tests if models can be properly loaded and initialized.
"""
import os
import sys
import logging
import yaml

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.lightning_model_adapter import LightningModelAdapter
from preprocessing.data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_model')

def main():
    """Test model and data loading"""
    # Load configuration
    config_path = 'configs/lightning_example.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded: {config.keys()}")
    
    # Initialize data processor
    data_path = 'data/processed/d2muvmslr_3h_none_log_20012020_weighted.h5'
    config['data']['processed_data_path'] = data_path
    
    logger.info("Initializing data processor")
    data_processor = DataProcessor(config)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = data_processor.load_processed_data()
    
    logger.info(f"Data loaded with {len(data)} time points")
    
    # Prepare data loaders
    logger.info("Preparing data loaders")
    train_loader, val_loader = data_processor.prepare_lightning_data_loaders()
    
    logger.info(f"Data loaders created: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    # Initialize model
    logger.info("Initializing model")
    model_type = config['model'].get('type', 'unknown')
    logger.info(f"Model type: {model_type}")
    
    # Print critical model parameters for debugging
    logger.info(f"Model learning rate: {config['model'].get('learning_rate')}, type: {type(config['model'].get('learning_rate'))}")
    logger.info(f"Model momentum: {config['model'].get('momentum')}, type: {type(config['model'].get('momentum'))}")
    logger.info(f"Model weight decay: {config['model'].get('weight_decay')}, type: {type(config['model'].get('weight_decay'))}")
    
    model = LightningModelAdapter(config)
    logger.info("Model initialized successfully")
    
    # Check optimizer
    logger.info("Testing optimizer configuration")
    optimizer = model.configure_optimizers()
    logger.info("Optimizer configured successfully")
    
    logger.info("All tests passed successfully")
    return 0

if __name__ == '__main__':
    sys.exit(main())
