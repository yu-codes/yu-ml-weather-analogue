# Efficient Environment Configuration and Data Processing Guide

This document provides efficient methods for environment configuration and data processing for the yu-ml-weather-analogue project.

## Table of Contents

- [Conda Environment Setup](#conda-environment-setup)
- [Project Directory Structure](#project-directory-structure)
- [Data Download and Processing](#data-download-and-processing)
- [Common Issues](#common-issues)

## Conda Environment Setup

### Method 1: Using `environment.yaml` File

First, create a valid `environment.yaml` file:

```zsh
# First, ensure we're in the project root directory
cd ./yu-ml-weather-analogue

# Create or update the environment.yaml file
cat > environment.yaml << 'EOF'
name: weather-analogue
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pip
  - numpy
  - pandas
  - matplotlib
  - cartopy
  - xarray
  - netcdf4
  - cdsapi
  - scipy
  - scikit-learn
  - pytorch
  - torchvision
  - pytorch-lightning
  - jupyter
  - ipykernel
  - h5py
  - tqdm
  - python-dotenv
  - pip:
    - wandb
    - shap
EOF
```

Then, use this file to create the environment:

```zsh
# Create the environment using environment.yaml
conda env create -f environment.yaml

# Activate the environment
conda activate weather-analogue
```

### Method 2: Creating Environment Directly via Command Line

If you prefer to create the environment directly from the command line without editing a YAML file:

```zsh
# Create a new environment
conda create -n weather-analogue python=3.9 -y

# Activate the environment
conda activate weather-analogue

# Install necessary packages (using conda-forge channel)
conda install -c conda-forge numpy pandas matplotlib cartopy xarray netcdf4 cdsapi scipy scikit-learn pytorch torchvision pytorch-lightning jupyter ipykernel h5py tqdm python-dotenv -y

# Install pip packages
pip install wandb shap
```

### Accelerating Environment Setup

1. **Using mamba to speed up installation**:
   ```zsh
   # Install mamba
   conda install -c conda-forge mamba -y
   
   # Create environment using mamba
   mamba env create -f environment.yaml
   ```

2. **Optimizing CUDA support**:
   ```zsh
   # Install PyTorch with specific CUDA version
   conda install -c pytorch pytorch=1.12.1 cudatoolkit=11.3 -y
   ```

3. **Setting up Jupyter kernel**:
   ```zsh
   python -m ipykernel install --user --name weather-analogue --display-name "Python (weather-analogue)"
   ```

4. **Verifying the environment**:
   ```zsh
   # List installed packages
   conda list
   
   # Test Python imports
   python -c "import torch; import pandas; import matplotlib; import xarray; print('Environment setup successful!')"
   ```

### Exporting the Environment

After development, you may need to export the environment to share or rebuild it:

1. **Export the complete environment (including all dependencies)**:
   ```zsh
   # Activate your environment
   conda activate weather-analogue
   
   # Export the complete environment configuration
   conda env export > environment.yaml
   ```

2. **Export a more concise environment file (recommended for sharing)**:
   ```zsh
   # Export only explicitly installed packages (better for cross-platform)
   conda env export --from-history > environment.yaml
   ```

3. **Export a specific environment (without activating it first)**:
   ```zsh
   # Export the environment named weather-analogue
   conda env export -n weather-analogue > environment.yaml
   ```

4. **Export a list containing only package names**:
   ```zsh
   # Create a requirements file with only package names
   conda list -n weather-analogue --export > requirements.txt
   ```

To rebuild the environment from the exported file, simply run:

```zsh
conda env create -f environment.yaml
```


## Common Issues

### Environment Issues

1. **Package conflicts**: If you encounter package conflicts, try:
   ```zsh
   conda clean --all
   conda env remove -n weather-analogue
   conda env create -f environment.yaml
   ```

2. **CUDA issues**: Ensure the CUDA version is compatible with PyTorch:
   ```zsh
   # Check CUDA availability
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```


### Runtime Issues

1. **Memory insufficiency**: When processing large datasets, adjust batch size or use data generators.

2. **Long-running tasks**: Use `tmux` or `screen` to run long tasks in the background:
   ```zsh
   # Create a new tmux session
   tmux new -s train_session
   
   # Run commands in tmux
   ./scripts/run_training.sh
   
   # Detach session (Ctrl+B then D)
   # Reattach to session
   tmux attach -t train_session

   # List all tmux sessions
   tmux ls
   ```