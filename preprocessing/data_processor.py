import os
import sys
import numpy as np
import pandas as pd
import logging
import xarray as xr
from datetime import datetime
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from data.data_preprocessor import process_weather_data
from utils.utils_data import load_weather_data, merge_data_by_timestamp, list_to_netcdf, read_netcdf
from preprocessing.dataset import AtmodistDataset

class WeatherDataProcessor:
    """
    Enhanced processor for meteorological data, handling loading, preprocessing, and dataset creation.
    Builds on existing functionality while adding configuration-based processing and experiment tracking.
    """
    def __init__(self, config):
        """
        Initialize the data processor with configuration.
        
        Args:
            config: Configuration dictionary containing data parameters
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.variables = self.data_config.get('variables', [])
        self.preprocess_method = self.data_config.get('preprocess', 'standardize')
        self.alpha = self.data_config.get('alpha', 0.2)
        self.frequency = self.data_config.get('frequency', '6h')
        self.resample_method = self.data_config.get('resample_method', 'mean')
        self.weighted = self.data_config.get('weighted', True)
        
        # Configure logging
        self.logger = logging.getLogger('WeatherDataProcessor')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
    def load_data(self, path, variable=None):
        """
        Load weather data from file.
        
        Args:
            path: Path to the data file (netCDF)
            variable: Optional specific variable to load
            
        Returns:
            xarray Dataset containing the weather data
        """
        self.logger.info(f"Loading data from {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
            
        try:
            # Use existing utility function
            data = load_weather_data(path, variable)
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_variable(self, data, variable):
        """
        Preprocess a single variable according to configuration.
        
        Args:
            data: xarray DataArray or Dataset containing the variable
            variable: Name of the variable to process
            
        Returns:
            Processed xarray DataArray
        """
        self.logger.info(f"Preprocessing variable {variable} with method: {self.preprocess_method}")
        
        # Use existing process_weather_data function
        processed = process_weather_data(
            data,
            variable,
            crop_to_32x32=True,  # Can be made configurable
            freq=self.frequency,
            resample_method=self.resample_method,
            preprocessing_method=self.preprocess_method,
            alpha=self.alpha
        )
        
        return processed
    
    def process_all_variables(self, data_dir):
        """
        Process all variables specified in the configuration.
        
        Args:
            data_dir: Directory containing the data files
            
        Returns:
            Dictionary of processed data arrays by variable
        """
        processed_vars = {}
        
        for var in self.variables:
            var_path = os.path.join(data_dir, f"{var}.nc")
            if not os.path.exists(var_path):
                self.logger.warning(f"Variable file {var_path} not found, skipping")
                continue
                
            # Load and process
            var_data = self.load_data(var_path, var)
            processed_vars[var] = self.preprocess_variable(var_data, var)
            
        return processed_vars
    
    def merge_variables(self, processed_vars):
        """
        Merge processed variables into a single dataset.
        
        Args:
            processed_vars: Dictionary of processed variables
            
        Returns:
            Merged xarray Dataset
        """
        self.logger.info(f"Merging {len(processed_vars)} processed variables")
        
        # Convert to list of datasets for merging
        datasets = []
        for var, data in processed_vars.items():
            if isinstance(data, xr.DataArray):
                # Convert DataArray to Dataset
                data = data.to_dataset(name=var)
            datasets.append(data)
            
        # Use existing merge function if datasets not empty
        if datasets:
            # First attempt to use merge_data_by_timestamp
            try:
                merged = merge_data_by_timestamp(datasets)
            except Exception as e:
                self.logger.warning(f"Error using merge_data_by_timestamp: {str(e)}")
                self.logger.info("Falling back to xarray merge")
                merged = xr.merge(datasets)
        else:
            self.logger.warning("No datasets to merge")
            return None
            
        return merged
    
    def create_dataset(self, data, train_ratio=0.8, random_seed=42):
        """
        Split data into training and validation sets.
        
        Args:
            data: xarray Dataset to split
            train_ratio: Ratio of data to use for training
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing train and validation datasets
        """
        self.logger.info(f"Creating dataset with train ratio: {train_ratio}")
        np.random.seed(random_seed)
        
        # Get time dimension
        if 'time' not in data.dims:
            self.logger.error("No time dimension found in dataset")
            return {'full': data}
            
        n_samples = len(data.time)
        
        # Generate random indices for train/val split
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        train_size = int(train_ratio * n_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Split data
        train_data = data.isel(time=train_indices)
        val_data = data.isel(time=val_indices)
        
        return {
            'train': train_data,
            'validation': val_data,
            'full': data
        }
    
    def save_dataset(self, dataset_dict, output_dir):
        """
        Save the processed datasets to disk.
        
        Args:
            dataset_dict: Dictionary containing datasets
            output_dir: Directory to save datasets
            
        Returns:
            Dictionary of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = {}
        
        for split_name, data in dataset_dict.items():
            output_path = os.path.join(output_dir, f"{split_name}.nc")
            data.to_netcdf(output_path)
            saved_paths[split_name] = output_path
            self.logger.info(f"Saved {split_name} dataset to {output_path}")
            
        return saved_paths
    
    def process_pipeline(self, data_dir, output_dir=None, train_ratio=0.8):
        """
        Run the full data processing pipeline.
        
        Args:
            data_dir: Directory containing input data files
            output_dir: Directory to save processed data (optional)
            train_ratio: Ratio for train/validation split
            
        Returns:
            Processed dataset dictionary
        """
        # Process all variables
        processed_vars = self.process_all_variables(data_dir)
        
        # Merge variables
        merged_data = self.merge_variables(processed_vars)
        if merged_data is None:
            self.logger.error("Failed to create merged dataset")
            return None
            
        # Create train/val split
        dataset_dict = self.create_dataset(merged_data, train_ratio=train_ratio)
        
        # Save if output directory provided
        if output_dir:
            saved_paths = self.save_dataset(dataset_dict, output_dir)
            # Add paths to the dictionary
            dataset_dict['paths'] = saved_paths
        
        return dataset_dict
    
    def load_processed_data(self, file_path):
        """
        Load processed data in the format expected by the AtmodistDataset.
        
        Args:
            file_path: Path to the processed data file (netCDF or h5)
            
        Returns:
            List of [timestamp, data_array] pairs
        """
        self.logger.info(f"Loading processed data from {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # For h5 or netCDF files, use read_netcdf utility
        if file_path.endswith('.h5') or file_path.endswith('.nc'):
            try:
                result_list = read_netcdf(file_path)
                self.logger.info(f"Loaded data with {len(result_list)} time points")
                return result_list
            except Exception as e:
                self.logger.error(f"Error loading data: {str(e)}")
                raise
        else:
            self.logger.error(f"Unsupported file format: {file_path}")
            raise ValueError(f"Unsupported file format: {file_path}")
            
    def prepare_atmodist_data(self, variable_list, frequency, time_unit="h", 
                             resample_method="none", preprocessing_method="standardized",
                             year_range=(2001, 2020), output_dir=None):
        """
        Prepare data for Atmodist model training following the exp_training.py approach.
        
        Args:
            variable_list: List of variables to include
            frequency: Time frequency (e.g., 3, 6)
            time_unit: Time unit (e.g., "h" for hours)
            resample_method: Resampling method
            preprocessing_method: Preprocessing method
            year_range: Tuple of (start_year, end_year)
            output_dir: Output directory for processed data
            
        Returns:
            Path to the processed data file
        """
        self.logger.info(f"Preparing Atmodist data with variables: {variable_list}")
        
        # Construct file path based on parameters
        variables = ''.join(variable_list)
        file_name = f"{variables}_{frequency}{time_unit}_{resample_method}_{preprocessing_method}_{year_range[0]}{year_range[1]}.h5"
        
        # Use provided output_dir or default to data/processed
        if output_dir is None:
            output_dir = os.path.join(project_root, "data", "processed")
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)
        
        # Check if file already exists
        if os.path.exists(output_path):
            self.logger.info(f"Found existing processed data at {output_path}")
            return output_path
            
        # Process data for each variable
        processed_vars = {}
        for var in variable_list:
            var_path = os.path.join(project_root, "data", "raw", f"{var}.nc")
            if not os.path.exists(var_path):
                self.logger.warning(f"Variable file {var_path} not found, skipping")
                continue
                
            # Load and process
            var_data = self.load_data(var_path, var)
            processed_vars[var] = self.preprocess_variable(var_data, var)
            
        # Merge variables
        merged_data = self.merge_variables(processed_vars)
        if merged_data is None:
            self.logger.error("Failed to create merged dataset")
            return None
            
        # Save to h5 format
        merged_data.to_netcdf(output_path)
        self.logger.info(f"Saved processed data to {output_path}")
        
        return output_path

class DataProcessor:
    """
    Data processor for weather analogue models.
    Handles loading, preprocessing, and dataset creation.
    """
    def __init__(self, config):
        """
        Initialize the data processor with configuration.
        
        Args:
            config: Configuration dictionary containing data parameters
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.processed_data_path = self.data_config.get('processed_data_path')
        self.data = None

        # Dataset parameters
        self.num_samples = self.data_config.get('num_samples', 100000)
        self.selected_frequency = int(self.data_config.get('frequency', '3h').replace('h', ''))
        self.time_unit = 'h'
        self.time_interval = self.data_config.get('time_interval', 24)

        # Training parameters
        self.training_config = config.get('training', {})
        self.batch_size = self.training_config.get('batch_size', 64)
        self.train_val_split = self.training_config.get('train_val_split', 0.8)

        # Configure logging
        self.logger = logging.getLogger('DataProcessor')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def load_processed_data(self):
        """
        Load processed data from file.
        
        Returns:
            List of [timestamp, data_array] pairs
        """
        if not self.processed_data_path:
            self.logger.error("No processed data path provided")
            return None

        try:
            self.logger.info(f"Loading processed data from {self.processed_data_path}")
            self.data = read_netcdf(self.processed_data_path)
            self.logger.info(f"Loaded data with {len(self.data)} time points")
            return self.data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def prepare_atmodist_data(self):
        """
        Prepare data for Atmodist model training.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        if self.data is None:
            self.logger.warning("No data loaded, attempting to load from path")
            self.load_processed_data()

        if self.data is None:
            self.logger.error("Failed to load data")
            return None

        # Create dataset
        dataset = AtmodistDataset(
            data=self.data,
            num_samples=self.num_samples,
            selected_frequency=self.selected_frequency,
            time_unit=self.time_unit,
            time_interval=self.time_interval
        )

        # Split dataset
        total_size = len(dataset)
        train_size = int(self.train_val_split * total_size)
        val_size = int((total_size - train_size) * 0.8)
        test_size = total_size - train_size - val_size

        # Use fixed seed for reproducibility
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, 
            [train_size, val_size, test_size],
            generator=generator
        )

        # Create data tensors
        X_train, y_train = self._get_data_from_dataset(train_dataset)
        X_val, y_val = self._get_data_from_dataset(val_dataset)
        X_test, y_test = self._get_data_from_dataset(test_dataset)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _get_data_from_dataset(self, dataset):
        """
        Extract data tensors from dataset.
        
        Args:
            dataset: Dataset or Subset to extract data from
            
        Returns:
            Tuple of (X, y)
        """
        all_X1 = []
        all_X2 = []
        all_y = []

        for i in range(len(dataset)):
            timestamp1, timestamp2, r1, r2, target = dataset[i]
            all_X1.append(r1.unsqueeze(0))
            all_X2.append(r2.unsqueeze(0))
            all_y.append(target.unsqueeze(0))

        X1 = torch.cat(all_X1, dim=0)
        X2 = torch.cat(all_X2, dim=0)
        y = torch.cat(all_y, dim=0)

        return {'X1': X1, 'X2': X2}, y

    def prepare_lightning_data_loaders(self):
        """
        Prepare data loaders for PyTorch Lightning training.
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if self.data is None:
            self.logger.warning("No data loaded, attempting to load from path")
            self.load_processed_data()

        if self.data is None:
            self.logger.error("Failed to load data")
            return None, None

        # Create dataset
        dataset = AtmodistDataset(
            data=self.data,
            num_samples=self.num_samples,
            selected_frequency=self.selected_frequency,
            time_unit=self.time_unit,
            time_interval=self.time_interval
        )

        # Split dataset
        total_size = len(dataset)
        train_size = int(self.train_val_split * total_size)
        val_size = total_size - train_size

        # Use fixed seed for reproducibility
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=generator
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        self.logger.info(f"Created data loaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")

        return train_loader, val_loader
