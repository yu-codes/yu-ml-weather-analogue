"""
Training Scheduler Module

This module provides scheduling capabilities for running multiple training tasks,
including resource management, task queueing, and result tracking.
"""
import os
import sys
import time
import subprocess
import threading
import logging
import yaml
import pandas as pd
from datetime import datetime

class TrainingScheduler:
    """
    Scheduler for managing multiple training tasks.
    Handles task queuing, GPU allocation, and result tracking.
    """
    def __init__(self, config_path=None, config_dir=None, max_parallel=1, gpu_ids=None):
        """
        Initialize the training scheduler.
        
        Args:
            config_path: Path to a single configuration file (optional)
            config_dir: Directory containing multiple configuration files (optional)
            max_parallel: Maximum number of parallel training tasks
            gpu_ids: List of GPU IDs to use
        """
        self.config_path = config_path
        self.config_dir = config_dir
        self.max_parallel = max_parallel
        self.gpu_ids = gpu_ids if gpu_ids else [0]
        self.running_tasks = {}
        self.completed_tasks = []
        self.failed_tasks = []
        self.queue = []
        self.lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger('TrainingScheduler')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def load_configs(self):
        """
        Load configuration files from the config directory or a single config file.
        
        Returns:
            List of configuration dictionaries
        """
        configs = []
        
        # Handle single configuration file
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    configs.append({
                        'path': self.config_path,
                        'id': config.get('id', os.path.basename(self.config_path).replace('.yaml', '')),
                        'priority': config.get('priority', 1)
                    })
                self.logger.info(f"Loaded configuration from: {self.config_path}")
            except Exception as e:
                self.logger.error(f"Error loading configuration {self.config_path}: {str(e)}")
        
        # Handle configuration directory
        if self.config_dir and os.path.exists(self.config_dir):
            config_files = [f for f in os.listdir(self.config_dir) if f.endswith('.yaml')]
            
            for file in config_files:
                file_path = os.path.join(self.config_dir, file)
                try:
                    with open(file_path, 'r') as f:
                        config = yaml.safe_load(f)
                        configs.append({
                            'path': file_path,
                            'id': config.get('id', file.replace('.yaml', '')),
                            'priority': config.get('priority', 1)
                        })
                except Exception as e:
                    self.logger.error(f"Error loading configuration {file}: {str(e)}")
            
            self.logger.info(f"Loaded {len(config_files)} configurations from directory: {self.config_dir}")
        
        if not configs:
            self.logger.warning("No configurations loaded")
            return []
        
        # Sort by priority
        self.queue = sorted(configs, key=lambda x: x['priority'], reverse=True)
        
        return self.queue
    
    def _train_process(self, config_path, gpu_id, data_path):
        """
        Run a training process.
        
        Args:
            config_path: Path to configuration file
            gpu_id: GPU ID to use
            data_path: Path to the processed data file
        """
        task_id = os.path.basename(config_path).replace('.yaml', '')
        log_dir = os.path.join('logs', 'training')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        start_time = time.time()
        
        try:
            with open(log_file, 'w') as f:
                cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/train.py --config {config_path} --data-path {data_path}"
                self.logger.info(f"Starting task {task_id} on GPU {gpu_id}: {cmd}")
                
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
                
                process.wait()
                
                if process.returncode != 0:
                    self.logger.error(f"Task {task_id} failed with exit code {process.returncode}")
                    with self.lock:
                        self.failed_tasks.append({
                            'id': task_id,
                            'config': config_path,
                            'gpu': gpu_id,
                            'start_time': start_time,
                            'end_time': time.time(),
                            'status': 'failed',
                            'log': log_file
                        })
                else:
                    self.logger.info(f"Task {task_id} completed successfully")
                    with self.lock:
                        self.completed_tasks.append({
                            'id': task_id,
                            'config': config_path,
                            'gpu': gpu_id,
                            'start_time': start_time,
                            'end_time': time.time(),
                            'status': 'completed',
                            'log': log_file
                        })
        except Exception as e:
            self.logger.exception(f"Error running task {task_id}: {str(e)}")
            with self.lock:
                self.failed_tasks.append({
                    'id': task_id,
                    'config': config_path,
                    'gpu': gpu_id,
                    'start_time': start_time,
                    'end_time': time.time(),
                    'status': 'error',
                    'error': str(e),
                    'log': log_file
                })
        finally:
            with self.lock:
                if gpu_id in self.running_tasks:
                    del self.running_tasks[gpu_id]
    
    def start(self, data_path):
        """
        Start the training scheduler.
        
        Args:
            data_path: Path to the processed data file
            
        Returns:
            Dictionary containing scheduling results
        """
        if not self.queue:
            self.load_configs()
            
        if not self.queue:
            self.logger.error("No configurations found")
            return {'status': 'error', 'message': 'No configurations found'}
        
        self.logger.info(f"Starting scheduler with {len(self.queue)} tasks")
        self.logger.info(f"Using GPUs: {self.gpu_ids}")
        self.logger.info(f"Maximum parallel tasks: {self.max_parallel}")
        
        while self.queue or self.running_tasks:
            # Check for available GPUs
            with self.lock:
                available_gpus = [gpu for gpu in self.gpu_ids if gpu not in self.running_tasks]
            
            # Start new tasks if GPUs are available
            while self.queue and available_gpus and len(self.running_tasks) < self.max_parallel:
                next_task = self.queue.pop(0)
                config_path = next_task['path']
                gpu_id = available_gpus.pop(0)
                
                with self.lock:
                    self.running_tasks[gpu_id] = next_task
                
                thread = threading.Thread(
                    target=self._train_process,
                    args=(config_path, gpu_id, data_path)
                )
                thread.daemon = True
                thread.start()
            
            # Wait before checking again
            time.sleep(5)
        
        # Generate report
        report = self.generate_report()
        self.logger.info("All tasks completed")
        self.logger.info(f"Completed: {len(self.completed_tasks)}, Failed: {len(self.failed_tasks)}")
        
        return {
            'status': 'completed',
            'report': report,
            'completed': len(self.completed_tasks),
            'failed': len(self.failed_tasks)
        }
    
    def generate_report(self):
        """
        Generate a report of training tasks.
        
        Returns:
            Path to the generated report
        """
        all_tasks = self.completed_tasks + self.failed_tasks
        if not all_tasks:
            return None
        
        # Create DataFrame
        df = pd.DataFrame(all_tasks)
        
        # Add duration
        df['duration'] = df['end_time'] - df['start_time']
        df['duration_minutes'] = df['duration'] / 60
        
        # Convert timestamps
        df['start_time'] = pd.to_datetime(df['start_time'], unit='s')
        df['end_time'] = pd.to_datetime(df['end_time'], unit='s')
        
        # Sort by start time
        df = df.sort_values('start_time')
        
        # Save report
        report_dir = os.path.join('logs', 'reports')
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(report_path, index=False)
        
        return report_path
