# Data Download and Environment Setup Guide

This document provides a guide for data downloading and environment setup for the yu-ml-weather-analogue project.

## Table of Contents

- [Environment Setup](#environment-setup)
- [ERA5 Data Download](#era5-data-download)
- [Data Storage Location](#data-storage-location)
- [Common Issues](#common-issues)

## Environment Setup

### Installing Miniconda

If you haven't installed Conda yet, you can use the provided script to install Miniconda:

```bash
# Run from the scripts directory
./scripts/download_miniconda.sh

# Initialize conda
source ~/miniconda/bin/activate

# Create and activate the project environment
conda env create -f environment.yml
conda activate weather-analogue
```

### Setting up CDS API

ERA5 data needs to be downloaded through the Climate Data Store (CDS) API. Please follow these steps to set it up:

1. Register an account at [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
2. Go to your profile page and copy the API Key
3. Create a `.env` file in the project root directory and add the following:

```plaintext
CDS_API_KEY=xxxxx:xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

## ERA5 Data Download

### Available Commands

The `download_era5.sh` script provides a flexible command-line interface for downloading ERA5 data:

```bash
# Display help information
./scripts/download_era5.sh help

# Show available variables
./scripts/download_era5.sh vars
```

### Downloading Single Level Variables

Single level variables such as temperature, precipitation, etc. can be downloaded using the `single` command:

```bash
# Basic usage: variable name, start year, end year
./scripts/download_era5.sh single 2m_temperature 2010 2020

# Custom region and grid resolution
./scripts/download_era5.sh single total_precipitation 2000 2020 "56,-6,48,2" "0.25,0.25"
```

### Downloading Pressure Level Variables

Pressure level variables such as wind fields, geopotential height, etc. can be downloaded using the `pressure` command:

```bash
# Basic usage: variable name, pressure level (hPa), start year, end year
./scripts/download_era5.sh pressure u_component_of_wind 850 2010 2020

# Custom region and grid resolution
./scripts/download_era5.sh pressure temperature 500 2000 2020 "56,-6,48,2" "0.25,0.25"
```

### Parameter Descriptions

- **Region format**: `"North latitude, West longitude, South latitude, East longitude"` (Default: `"60,-10,40,5"` Europe region)
- **Grid resolution**: `"Latitude resolution, Longitude resolution"` (Default: `"0.25,0.25"`)

### Batch Downloading Multiple Variables

You can create a batch processing script to download multiple variables simultaneously:

```bash
#!/bin/bash
# Create a batch download script batch_download.sh

./scripts/download_era5.sh single 2m_temperature 2010 2020
./scripts/download_era5.sh single total_precipitation 2010 2020
./scripts/download_era5.sh pressure u_component_of_wind 850 2010 2020
./scripts/download_era5.sh pressure geopotential 500 2010 2020
```

Then execute:

```bash
chmod +x batch_download.sh
./batch_download.sh
```

## Data Storage Location

Downloaded data will be stored in the following structure:

```plaintext
yu-ml-weather-analogue/
├── data/
│   └── raw/
│       └── era5/
│           ├── single_level/
│           │   ├── 2m_temperature/
│           │   └── total_precipitation/
│           ├── pressure_level/
│           │   ├── 850hPa_u_component_of_wind/
│           │   └── 500hPa_geopotential/
│           └── logs/
```

## Common Issues

### Download Failures

If you encounter download failures, please check:

1. Whether the API Key in the `.env` file is correct
2. Whether the network connection is stable
3. Check the error logs in the `data/raw/era5/logs/` directory
4. **Storage space**: ERA5 data can be large, ensure you have enough storage space:

   ```zsh
   # Check disk space
   df -h
   ```

### Custom Downloads

If you need to customize more download parameters, you can directly modify the `data/era5_downloader.py` file.
