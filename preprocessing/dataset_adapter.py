"""
Dataset Adapter Module

This module provides dataset adapters for the AtmodistDataset to work with the new training framework.
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from preprocessing.dataset import AtmodistDataset, OrdinalDataset, TripletDataset

class DatasetAdapter:
    """
    Adapter for AtmodistDataset and related datasets to work with the new training framework.
    """
    def __init__(self, config):
        """
        Initialize the dataset adapter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.training_config = config.get('training', {})

        # Dataset parameters
        self.variables = self.data_config.get('variables', ['d2m', 'u', 'v', 'msl', 'r'])
        self.frequency = self.data_config.get('frequency', '3h')
        self.selected_frequency = int(self.frequency.replace('h', ''))
        self.time_unit = 'h'
        self.time_interval = self.data_config.get('time_interval', 24)
        self.num_samples = self.data_config.get('num_samples', 100000)
        self.dataset_type = self.data_config.get('dataset_type', 'atmodist')

        # Training parameters
        self.batch_size = self.training_config.get('batch_size', 64)
        self.train_val_split = self.training_config.get('train_val_split', 0.8)

        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def create_dataset(self, data, dataset_type=None):
        """
        Create the appropriate dataset based on type.
        
        Args:
            data: List of [timestamp, data_array] pairs
            dataset_type: Type of dataset to create (optional, uses config if not provided)
            
        Returns:
            Dataset instance
        """
        if dataset_type is None:
            dataset_type = self.dataset_type

        if dataset_type == 'atmodist':
            return AtmodistDataset(
                data=data,
                num_samples=self.num_samples,
                selected_frequency=self.selected_frequency,
                time_unit=self.time_unit,
                time_interval=self.time_interval
            )
        elif dataset_type == 'ordinal':
            return OrdinalDataset(
                data=data,
                num_samples=self.num_samples,
                selected_frequency=self.selected_frequency,
                time_unit=self.time_unit,
                time_interval=self.time_interval
            )
        elif dataset_type == 'triplet':
            return TripletDataset(
                data=data,
                num_samples=self.num_samples,
                selected_frequency=self.selected_frequency,
                time_unit=self.time_unit,
                time_interval=self.time_interval
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def setup_data_loaders(self, data):
        """
        Set up data loaders for training, validation, and testing.
        
        Args:
            data: List of [timestamp, data_array] pairs
            
        Returns:
            Dictionary of data loaders
        """
        # Create dataset
        full_dataset = self.create_dataset(data)

        # Split dataset
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * self.train_val_split)
        val_size = dataset_size - train_size

        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        return {
            'train': self.train_loader,
            'validation': self.val_loader
        }

    def get_input_shape(self, data):
        """
        Get the input shape for the model based on the data.
        
        Args:
            data: List of [timestamp, data_array] pairs
            
        Returns:
            Number of input channels
        """
        # Return number of variables (channels)
        if data and len(data) > 0:
            return len(data[0][1])
        return len(self.variables)
