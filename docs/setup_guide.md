# 高效環境配置與資料處理指南

本文檔提供 yu-ml-weather-analogue 專案的環境配置和資料處理的高效方法。

## 目錄

- [Conda 環境配置](#conda-環境配置)
- [專案目錄結構](#專案目錄結構)
- [資料下載與處理](#資料下載與處理)
- [常見問題](#常見問題)

## Conda 環境配置

### 方法一：使用 `environment.yaml` 檔案

首先創建一個有效的 `environment.yaml` 檔案：

```zsh
# 先確認我們在專案根目錄
cd ./yu-ml-weather-analogue

# 創建或更新 environment.yaml 檔案
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

然後，使用這個檔案創建環境：

```zsh
# 使用 environment.yaml 創建環境
conda env create -f environment.yaml

# 啟用環境
conda activate weather-analogue
```

### 方法二：直接使用命令行創建環境

如果你想直接從命令行創建，而不編輯 YAML 檔案：

```zsh
# 創建新環境
conda create -n weather-analogue python=3.9 -y

# 啟用環境
conda activate weather-analogue

# 安裝必要套件 (使用 conda-forge 頻道)
conda install -c conda-forge numpy pandas matplotlib cartopy xarray netcdf4 cdsapi scipy scikit-learn pytorch torchvision pytorch-lightning jupyter ipykernel h5py tqdm python-dotenv -y

# 安裝 pip 套件
pip install wandb shap
```

### 加速環境配置

1. **使用 mamba 加速安裝**：
   ```zsh
   # 安裝 mamba
   conda install -c conda-forge mamba -y
   
   # 使用 mamba 創建環境
   mamba env create -f environment.yaml
   ```

2. **優化 CUDA 支援**：
   ```zsh
   # 指定 CUDA 版本安裝 PyTorch
   conda install -c pytorch pytorch=1.12.1 cudatoolkit=11.3 -y
   ```

3. **設置 Jupyter 核心**：
   ```zsh
   python -m ipykernel install --user --name weather-analogue --display-name "Python (weather-analogue)"
   ```

4. **驗證環境**：
   ```zsh
   # 列出已安裝的套件
   conda list
   
   # 測試 Python 匯入
   python -c "import torch; import pandas; import matplotlib; import xarray; print('環境設置成功！')"
   ```

### 匯出環境

開發完成後，你可能需要將環境匯出以分享或重建：

1. **匯出完整環境（包含所有相依套件）**：
   ```zsh
   # 啟用你的環境
   conda activate weather-analogue
   
   # 匯出完整環境配置
   conda env export > environment.yaml
   ```

2. **匯出更簡潔的環境檔案（建議用於分享）**：
   ```zsh
   # 僅匯出明確安裝的套件（更適合跨平台）
   conda env export --from-history > environment.yaml
   ```

3. **匯出特定環境（無需先啟用）**：
   ```zsh
   # 匯出名為 weather-analogue 的環境
   conda env export -n weather-analogue > environment.yaml
   ```

4. **匯出僅包含套件名稱的清單**：
   ```zsh
   # 創建僅包含套件名稱的需求檔案
   conda list -n weather-analogue --export > requirements.txt
   ```

從匯出的環境檔案重建環境只需執行：

```zsh
conda env create -f environment.yaml
```


## 常見問題

### 環境問題

1. **包衝突**：如果遇到包衝突，先嘗試：
   ```zsh
   conda clean --all
   conda env remove -n weather-analogue
   conda env create -f environment.yaml
   ```

2. **CUDA 問題**：確保 CUDA 版本與 PyTorch 相容：
   ```zsh
   # 檢查 CUDA 可用性
   python -c "import torch; print('CUDA 可用:', torch.cuda.is_available())"
   ```


### 運行問題

1. **記憶體不足**：處理大型資料集時，可以調整批次大小或使用數據生成器。

2. **長時間運行**：使用 `tmux` 或 `screen` 在後台運行長時間任務：
   ```zsh
   # 使用 tmux 創建新會話
   tmux new -s train_session
   
   # 在 tmux 中運行命令
   ./scripts/run_training.sh
   
   # 分離會話 (Ctrl+B 然後按 D)
   # 重新連接會話
   tmux attach -t train_session

   # 查看所有 tmux 會話
   tmux ls
   ```