#!/usr/bin/env python
"""
Training Scheduler Script

This script schedules and runs multiple training tasks based on configuration files.
"""
import argparse
import os
import sys
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from trainers.scheduler import TrainingScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('run_training')

def main():
    parser = argparse.ArgumentParser(description='Run training scheduler')
    parser.add_argument('--config', type=str, help='Single configuration file path')
    parser.add_argument('--config-dir', type=str, help='Directory containing multiple configuration files')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the processed data file')
    parser.add_argument('--max-parallel', type=int, default=1,
                        help='Maximum number of parallel tasks')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Comma-separated list of GPU IDs to use')
    args = parser.parse_args()
    
    # Ensure either config or config-dir is provided
    if not args.config and not args.config_dir:
        logger.error("Either --config or --config-dir must be provided")
        return 1
    
    # Parse GPU IDs
    gpu_ids = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    
    # Initialize scheduler
    scheduler = TrainingScheduler(
        config_path=args.config,
        config_dir=args.config_dir,
        max_parallel=args.max_parallel,
        gpu_ids=gpu_ids
    )
    
    # Load configurations
    configs = scheduler.load_configs()
    
    if not configs:
        logger.error("No configurations found")
        return 1
    
    logger.info(f"Loaded {len(configs)} configurations")
    
    # Start scheduler
    results = scheduler.start(args.data_path)
    
    if results['status'] == 'error':
        logger.error(f"Scheduler error: {results['message']}")
        return 1
    
    logger.info(f"All tasks completed")
    logger.info(f"Completed: {results['completed']}, Failed: {results['failed']}")
    
    if results['report']:
        logger.info(f"Report saved to: {results['report']}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
