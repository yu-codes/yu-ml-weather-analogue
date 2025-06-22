# Weather Analogue Model Data Preprocessing Guide

This document provides a comprehensive guide to data preprocessing for the weather analogue model, covering data sources, preprocessing steps, directory structure, and best practices.

## 1. Data Directory Structure

The organization of data directories in the project:

```bash
yu-ml-weather-analogue/
├── data/
│   ├── raw/                 # Raw meteorological data (NetCDF format)
│   │   ├── era5_d2m_*.nc    # 2m dew point temperature data
│   │   ├── era5_u_*.nc      # U-component of wind data
│   │   ├── era5_v_*.nc      # V-component of wind data
│   │   ├── era5_msl_*.nc    # Mean sea level pressure data
│   │   └── era5_r_*.nc      # Relative humidity data
│   ├── processed/           # Preprocessed data
│   │   └── d2muvmslr_3h_none_log_20012020_weighted.h5  # Merged and processed data
│   ├── era5_downloader.py   
│   ├── solpos_downloader.py 
│   └── data_preprocessor.py # Data preprocessing tools
```

## 2. Data Sources and Acquisition

### 2.1 ERA5 Reanalysis Data

This project uses ERA5 reanalysis data from the European Centre for Medium-Range Weather Forecasts (ECMWF):

- **Data Source**: ERA5 is one of the most comprehensive global meteorological reanalysis datasets
- **Spatial Resolution**: 0.25° × 0.25° (approximately 25km)
- **Temporal Resolution**: Hourly
- **Coverage Period**: Typically using data from 2001-2020

### 2.2 Data Acquisition Methods

ERA5 data can be acquired through the following methods:

```python
# Using the project's download script
from data.era5_downloader import download_era5_data

download_era5_data(
    variable='d2m',           # Variable name
    year_range=(2001, 2020),  # Year range
    output_dir='data/raw',    # Output directory
    area=[90, 0, 0, 360]      # Region range [North, West, South, East]
)
```

## 3. Preprocessing Workflow

The complete data preprocessing workflow includes the following steps:

### 3.1 Single Variable Processing

Use the `process_weather_data` function from `data/data_preprocessor.py` to process individual variables:

```python
from data.data_preprocessor import process_weather_data
import xarray as xr

# Load raw NetCDF data
raw_data = xr.open_dataset('data/raw/era5_d2m_2001.nc')

# Process single variable data
processed_d2m = process_weather_data(
    dataset=raw_data,          # Original dataset
    variable='d2m',            # Variable name
    crop_to_32x32=True,        # Crop to 32x32 grid
    freq='3h',                 # Resampling frequency
    resample_method='mean',    # Resampling method
    preprocessing_method='log' # Preprocessing method
)
```

Preprocessing steps include:

1. Loading raw NetCDF data
2. Extracting the specified variable
3. Temporal resampling (e.g., from hourly to 3-hourly data)
4. Spatial cropping (e.g., cropping to a 32×32 grid)
5. Data transformation (standardization, normalization, or log transformation)

### 3.2 Multi-variable Data Merging

Use the `merge_data_by_timestamp` function from `utils/utils_data.py` to merge multiple variables:

```python
from utils.utils_data import merge_data_by_timestamp

# Merge multiple variable data
merged_data = merge_data_by_timestamp(
    [processed_d2m, processed_u, processed_v, processed_msl, processed_r],
    variables=['d2m', 'u', 'v', 'msl', 'r']
)

# Save as HDF5 format
merged_data.to_netcdf('data/processed/d2muvmslr_3h_none_log_20012020.h5')
```

### 3.3 Dataset Creation

Use the `DatasetAdapter` class from `preprocessing/dataset_adapter.py` to create datasets suitable for training:

```python
from preprocessing.dataset_adapter import DatasetAdapter

# Initialize dataset adapter
dataset_adapter = DatasetAdapter(config)

# Create datasets and data loaders
train_loader, val_loader, test_loader = dataset_adapter.create_data_loaders(
    data_path='data/processed/d2muvmslr_3h_none_log_20012020_weighted.h5'
)
```

## 4. Supported Preprocessing Methods

### 4.1 Standardization

Transform data to have zero mean and unit standard deviation:

```python
def standardize(data):
    return (data - np.mean(data)) / np.std(data)
```

### 4.2 Normalization

Scale data to the [0, 1] interval:

```python
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
```

### 4.3 Log Transformation

Process skewed distribution data:

```python
def log_transform(data):
    # Ensure data is positive
    min_val = np.min(data)
    if min_val < 0:
        data = data - min_val + 1e-8
    return np.log1p(data)  # log(1+x)
```

## 5. Dataset Types

### 5.1 AtmodistDataset

Standard dataset for Atmodist model:

```python
from preprocessing.dataset import AtmodistDataset

dataset = AtmodistDataset(
    data=processed_data,
    variables=['d2m', 'u', 'v', 'msl', 'r'],
    frequency='3h',
    time_interval=24,
    num_samples=100000
)
```

Key parameter descriptions:

- `data`: Processed data
- `variables`: List of variables to use
- `frequency`: Data frequency (e.g., '3h')
- `time_interval`: Time interval (hours)
- `num_samples`: Number of samples to generate

### 5.2 OrdinalDataset and TripletDataset

Special datasets for other training approaches:

```python
from preprocessing.dataset import OrdinalDataset, TripletDataset

# Ordinal dataset (for ordinal regression)
ordinal_dataset = OrdinalDataset(processed_data, ...)

# Triplet dataset (for triplet loss training)
triplet_dataset = TripletDataset(processed_data, ...)
```

## 6. WeatherDataProcessor Class

The `WeatherDataProcessor` in `preprocessing/data_processor.py` is the main class for processing meteorological data, encapsulating the complete data processing workflow:

```python
from preprocessing.data_processor import WeatherDataProcessor

# Initialize data processor
processor = WeatherDataProcessor(config)

# Complete data preparation process
train_loader, val_loader, test_loader = processor.prepare_data(
    'data/processed/d2muvmslr_3h_none_log_20012020_weighted.h5'
)
```

Class method descriptions:

- `load_data(path)`: Load processed data
- `create_data_loaders(data)`: Create data loaders
- `prepare_data(data_path)`: Complete data preparation process

## 7. Configuration Parameters

Set data preprocessing parameters in the YAML configuration file:

```yaml
data:
  variables: [d2m, u, v, msl, r]  # Meteorological variables to use
  frequency: 3h                   # Data sampling frequency
  time_interval: 24               # Time interval (hours)
  preprocess: log                 # Preprocessing method: standardize, normalize, log, none
  weighted: true                  # Whether to use weighted sampling
  num_samples: 100000             # Number of samples to generate
```

## 8. Preprocessing Best Practices

### 8.1 Data Consistency

- Ensure all variables use the same preprocessing method
- Save preprocessing parameters for consistent application to new data
- Use unified file naming format, such as `{variables}_{frequency}_{preprocess}_{years}_weighted.h5`

### 8.2 Data Validation

- Check for missing values in the data and handle appropriately
- Confirm that data distribution meets expectations
- Perform data visualization to validate preprocessing effects

### 8.3 Efficiency Optimization

- Use the `num_workers` parameter to optimize data loading performance
- For large datasets, consider using memory mapping or chunk processing
- Save preprocessed data to avoid repeated processing

### 8.4 Common Problem Solutions

#### Memory Insufficiency

If you encounter memory issues when processing large data:

```python
# Chunk processing
chunks = {'time': 100, 'latitude': 32, 'longitude': 32}
data = xr.open_dataset('large_file.nc', chunks=chunks)
```

#### Data Skewness

For highly skewed data distributions:

```python
# Use more complex transformations
from sklearn.preprocessing import PowerTransformer
transformer = PowerTransformer(method='yeo-johnson')
transformed_data = transformer.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
```

## 9. Data Visualization Tools

Use `notebooks/data_eda.ipynb` for exploratory data analysis:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Data distribution visualization
plt.figure(figsize=(12, 6))
sns.histplot(data=processed_data.values.flatten(), kde=True)
plt.title('Data Distribution')
plt.savefig('data_distribution.png')

# Time series visualization
plt.figure(figsize=(15, 5))
time_series = processed_data.mean(dim=['latitude', 'longitude'])
plt.plot(time_series)
plt.title('Time Series Trend')
plt.savefig('time_series.png')
```

## 10. Data Quality Check

Use `utils/data_quality.py` for data quality checks:

```python
from utils.data_quality import check_data_quality

# Check data quality
quality_report = check_data_quality(
    data_path='data/processed/d2muvmslr_3h_none_log_20012020_weighted.h5',
    variables=['d2m', 'u', 'v', 'msl', 'r']
)

# Output quality report
print(quality_report)
```

Quality check items:

- Missing value proportion
- Anomaly detection
- Data distribution statistics
- Time continuity check
