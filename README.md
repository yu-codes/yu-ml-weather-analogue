# yu-ml-weather-analogue

A machine learning-based weather analogue method for predicting extreme precipitation events. Developed as part of my Master's thesis.

## Project Structure

```
yu-ml-weather-analogue/
├── README.md                 # Project introduction, installation, usage
├── LICENSE                   # MIT License
├── .gitignore                # Ignore pycache, logs, data, wandb, etc.
├── environment.yml           # Conda environment configuration
│
├── data/                     # Data storage and processing
│   ├── raw/                  # Raw downloaded data (not uploaded)
│   ├── processed/            # Processed data (not uploaded)
│   ├── download_era5.py      # ERA5 download script
│   └── download_nimrod.py    # Nimrod data download script
│
├── notebooks/                # Data analysis and model results notebooks
│   ├── 01_explore_data.ipynb
│   ├── 02_train_atmodist.ipynb
│   └── 03_visualize_results.ipynb
│
├── experiments/              # Model training and evaluation scripts
│   ├── train_atmodist.py
│   ├── evaluate_baselines.py
│   └── run_all.sh            # Batch execution script (optional)
│
├── models/                   # Model architecture definitions
│   ├── atmodist.py           # Main model (Siamese)
│   ├── baseline.py           # Traditional distance methods
│   └── wrapper.py            # Model wrappers for explainability
│
├── utils/                    # Utility functions (reusable modules)
│   ├── metrics.py
│   ├── preprocessing.py
│   ├── distance.py
│   └── plot.py
│
├── results/                  # Results, figures, and logs
│   ├── figures/              # Prediction plots, SHAP plots, etc.
│   ├── logs/                 # Training/testing logs
│   └── summary.csv           # Final summary table
│
├── scripts/                # Utility scripts
│   ├── download_era5.sh      # ERA5 download shell script
│   ├── preprocess_data.sh    # 資料預處理 shell 腳本
│   └── ...                   # Other scripts
│
├── test/                     # 測試代碼
│   ├── test_data_processing.py # 資料預處理功能測試
│   └── ...                   # Other test scripts
│
└── configs/                  # Training and WandB configuration files
    ├── atmodist.yaml
    ├── wandb_config.yaml
    └── baseline.yaml
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/yu-ml-weather-analogue.git
    cd yu-ml-weather-analogue
    ```

2. Create the conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate yu-ml-weather-analogue
    ```

3. (Optional) Set up `.env` for sensitive information (API keys, etc.).

## Usage

### Data Processing

1. Download ERA5 data using the provided shell script:
   ```bash
   cd scripts
   ./download_era5.sh single 2m_temperature 2010 2020
   ```

2. Preprocess the downloaded data using the preprocessing script:
   ```bash
   cd scripts
   ./preprocess_data.sh --variables t2m,msl --freq 6h --years 2010-2020
   ```

   Options for `preprocess_data.sh`:
   - `-v, --variables`: Comma-separated list of variables (t2m, msl, tp, etc.)
   - `-f, --freq`: Time frequency (3h, 6h, 12h, etc.)
   - `-m, --method`: Resampling method (mean, sum, none)
   - `-p, --preprocess`: Preprocessing method (raw, log, normalized, standardized)
   - `-y, --years`: Year range (YYYY-YYYY)
   - `-o, --output`: Output directory
   - `-w, --weighted`: Use weighting
   - `--overlap`: Use overlapping windows

### Testing

Run tests to verify the functionality:
```bash
cd test
./test_data_processing.py
```

### Data Analysis and Modeling

- Explore and analyze data with notebooks in `notebooks/`.
- Train and evaluate models using scripts in `experiments/`.
- Model architectures are defined in `models/`.
- Utility functions are in `utils/`.
- Results and logs are saved in `results/`.
- Configuration files are in `configs/`.

## Logging & Tracking

- Training logs are saved in `results/logs/`.
- Weights & Biases (W&B) configuration is in `configs/wandb_config.yaml`.

## License

MIT License. See [LICENSE](LICENSE) for details.
