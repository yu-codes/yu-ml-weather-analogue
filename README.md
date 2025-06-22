# Yu-ML-Weather-Analogue

A machine learning-based weather analogue method for predicting extreme precipitation events. This project implements deep learning techniques to identify and analyze similar meteorological patterns from historical data, developed as part of a Master's thesis research.

## Overview

This framework enables efficient identification of weather analogues using deep learning techniques. The approach is designed to:

1. Process raw meteorological data from ERA5 and other sources
2. Train deep learning models to identify similar weather patterns
3. Support robust evaluation and visualization of results
4. Enable easy experiment tracking and reproducibility

## Features

- **PyTorch Lightning Integration**: Streamlined training and evaluation with the Lightning framework
- **Configuration-based experiments**: Define models and parameters in YAML files
- **Multi-configuration training**: Generate and run multiple model configurations
- **GPU-based scheduling**: Automatically schedule and run multiple experiments with GPU allocation
- **Experiment tracking**: Comprehensive tracking with Weights & Biases
- **Standardized preprocessing**: Process meteorological data consistently and efficiently
- **Analogue search capabilities**: Find similar weather patterns using trained models

## Project Structure

```text
yu-ml-weather-analogue/
├── README.md                  # Project introduction, installation, usage
├── LICENSE                    # MIT License
├── .gitignore                 # Ignore pycache, logs, data, wandb, etc.
├── environment.yaml           # Conda environment configuration
│
├── configs/                   # Configuration management
│   ├── __init__.py
│   ├── lightning_example.yaml # Example Lightning configuration
│   ├── multi_model_example.yaml # Multi-model configuration example
│   └── manager.py             # Configuration management utilities
│
├── data/                      # Data storage and processing
│   ├── __init__.py
│   ├── data_preprocessor.py   # Data preprocessing utilities
│   ├── era5_downloader.py     # ERA5 download script
│   ├── solpos_downloader.py   # Solar position data downloader
│   ├── raw/                   # Raw downloaded data (not uploaded to git)
│   └── processed/             # Processed data (not uploaded to git)
│
├── docs/                      # Documentation
│   ├── data_download_guide.md # Guide for downloading required data
│   ├── data_preprocessing_guide.md # Guide for data preprocessing
│   ├── env_setup_guide.md     # Environment setup instructions
│   ├── era5_variables.md      # ERA5 variables description
│   └── training_guide.md      # Model training guide
│
├── models/                    # Model architecture definitions
│   ├── __init__.py
│   ├── atmodist.py            # AtmoDist model implementation
│   ├── atmodist_revised.py    # Revised AtmoDist with improvements
│   ├── atmodist_adapter.py    # Legacy adapter
│   └── lightning_model_adapter.py # PyTorch Lightning model adapter
│
├── preprocessing/             # Enhanced data preprocessing modules
│   ├── __init__.py
│   └── data_processor.py      # Configurable data processor
│
├── trainers/                  # Training modules
│   ├── __init__.py
│   └── lightning_trainer.py   # Lightning trainer implementation
│
├── scripts/                   # Utility scripts
│   └── train.py               # Main training entry point
│
├── notebooks/                 # Analysis and visualization notebooks
│   ├── analogue_search_atmodist.ipynb   # Search using AtmoDist
│   ├── analogue_search_traditional.ipynb # Traditional methods
│   ├── data_distribution.ipynb          # Data analysis
│   └── data_eda.ipynb                   # Exploratory analysis
│
├── utils/                     # Utility functions
│
├── evaluation/                # Model evaluation modules
│
├── experiments/               # Experiment results storage
│
├── logs/                      # Training logs
│
└── test/                      # Test code
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/yu-ml-weather-analogue.git
   cd yu-ml-weather-analogue
   ```

2. Create the conda environment from the provided environment file:

   ```bash
   conda env create -f environment.yaml
   conda activate weather-analogue
   ```

3. (Optional) For faster dependency resolution, you can use mamba:

   ```bash
   conda install -c conda-forge mamba
   mamba env create -f environment.yaml
   ```

4. Set up Weights & Biases for experiment tracking:

   ```bash
   wandb login
   ```

For detailed environment setup instructions, see the [Environment Setup Guide](docs/env_setup_guide.md).

## Documentation

The project includes comprehensive documentation to help you get started:

- [Environment Setup Guide](docs/env_setup_guide.md) - Setting up your development environment
- [Data Download Guide](docs/data_download_guide.md) - Instructions for downloading required data
- [Data Preprocessing Guide](docs/data_preprocessing_guide.md) - Processing raw data for model training
- [Training Guide](docs/training_guide.md) - Training models with PyTorch Lightning
- [ERA5 Variables](docs/era5_variables.md) - Description of ERA5 meteorological variables used

## Usage

### Data Processing Workflow

1. **Download Data**: Download ERA5 reanalysis data using the ERA5 downloader

   ```bash
   python data/era5_downloader.py --variables t2m msl --years 2010-2020 --output data/raw
   ```

2. **Preprocess Data**: Process the raw data into the format required for training

   ```bash
   python data/data_preprocessor.py --input data/raw --output data/processed --variables t2m,msl --freq 3h
   ```

   For detailed data preprocessing options, refer to the [Data Preprocessing Guide](docs/data_preprocessing_guide.md).

### Model Training

Train a model using the PyTorch Lightning framework:

```bash
python scripts/train.py --config configs/lightning_example.yaml --data-path data/processed/your_processed_data.h5 --gpus 0
```

For multi-model training:

```bash
python scripts/train.py --config configs/multi_model_example.yaml --data-path data/processed/your_processed_data.h5 --gpus 0,1 --max-parallel 2
```

Command line options:

- `--config`: Path to the configuration file
- `--config-dir`: Directory containing multiple configuration files
- `--data-path`: Path to the processed data file
- `--gpus`: List of GPU IDs (comma-separated)
- `--max-parallel`: Maximum number of parallel training jobs
- `--seed`: Random seed for reproducibility

For detailed training instructions, see the [Training Guide](docs/training_guide.md).

### Using Trained Models

After training, you can use the trained models for analogue search and analysis:

```python
# Example code for loading and using a trained model
import torch
from models.lightning_model_adapter import LightningModelAdapter

# Load model checkpoint
checkpoint_path = "experiments/your_experiment_id/checkpoints/best_model.ckpt"
model = LightningModelAdapter.load_from_checkpoint(checkpoint_path)
model.eval()

# Perform inference
with torch.no_grad():
    inputs = torch.randn(1, 5, 32, 32)  # Example input
    outputs = model(inputs)
    predictions = torch.argmax(outputs, dim=1)
```

Example notebooks for working with trained models are provided in the `notebooks/` directory.

## Experiment Tracking

This project uses Weights & Biases (W&B) for experiment tracking. Key features include:

- Automatic logging of training and validation metrics
- Visualization of confusion matrices and classification reports
- Model checkpoints management
- Hyperparameter tracking

Training results are saved in:

- `experiments/`: Model checkpoints and configuration
- `logs/`: Training logs and progress information
- W&B online dashboard (requires account)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

This project is based on research conducted as part of a Master's thesis on weather analogue methods for precipitation forecasting. It builds upon several important works in the field of meteorological pattern recognition and deep learning.
