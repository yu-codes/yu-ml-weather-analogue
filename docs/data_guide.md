# 資料下載與環境設置指南

本文檔提供 yu-ml-weather-analogue 專案的資料下載與環境設置指南。

## 目錄

- [環境設置](#環境設置)
- [ERA5 資料下載](#era5-資料下載)
- [設置配置文件](#設置配置文件)

## 環境設置

### 安裝 Miniconda

如果您尚未安裝 Conda，可以使用提供的腳本安裝 Miniconda：

```bash
# 從 scripts 目錄執行
./scripts/download_miniconda.sh

# 初始化 conda
source ~/miniconda/bin/activate

# 創建並激活專案環境
conda env create -f environment.yml
conda activate weather-analogue
```

### 設置 CDS API

ERA5 資料需要通過 Climate Data Store (CDS) API 下載。請按照以下步驟設置：

1. 在 [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) 註冊帳號
2. 前往您的個人資料頁面，複製 API Key
3. 在專案根目錄創建 `.env` 文件，添加以下內容：

```
CDS_API_KEY=xxxxx:xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

## ERA5 資料下載

### 可用命令

`download_era5.sh` 腳本提供了靈活的命令行介面來下載 ERA5 資料：

```bash
# 顯示幫助信息
./scripts/download_era5.sh help

# 顯示可用變量
./scripts/download_era5.sh vars
```

### 下載單層變量

單層變量如溫度、降水等可使用 `single` 命令下載：

```bash
# 基本用法：變量名 起始年份 結束年份
./scripts/download_era5.sh single 2m_temperature 2010 2020

# 自定義區域和網格解析度
./scripts/download_era5.sh single total_precipitation 2000 2020 "56,-6,48,2" "0.25,0.25"
```

### 下載氣壓層變量

氣壓層變量如風場、位勢高度等可使用 `pressure` 命令下載：

```bash
# 基本用法：變量名 氣壓層(hPa) 起始年份 結束年份
./scripts/download_era5.sh pressure u_component_of_wind 850 2010 2020

# 自定義區域和網格解析度
./scripts/download_era5.sh pressure temperature 500 2000 2020 "56,-6,48,2" "0.25,0.25"
```

### 參數說明

- **區域格式**：`"北緯,西經,南緯,東經"` (預設：`"60,-10,40,5"` 歐洲區域)
- **網格解析度**：`"緯度解析度,經度解析度"` (預設：`"0.25,0.25"`)

### 批次下載多個變量

可以創建批次處理腳本同時下載多個變量：

```bash
#!/bin/bash
# 創建批次下載腳本 batch_download.sh

./scripts/download_era5.sh single 2m_temperature 2010 2020
./scripts/download_era5.sh single total_precipitation 2010 2020
./scripts/download_era5.sh pressure u_component_of_wind 850 2010 2020
./scripts/download_era5.sh pressure geopotential 500 2010 2020
```

然後執行：

```bash
chmod +x batch_download.sh
./batch_download.sh
```

## 資料存放位置

下載的資料將按以下結構存放：

```
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

## 常見問題

### 下載失敗

如果遇到下載失敗，請檢查：

1. `.env` 文件中的 API Key 是否正確
2. 網絡連接是否穩定
3. 查看 `data/raw/era5/logs/` 目錄下的錯誤日誌
4. **儲存空間**：ERA5 資料可能很大，確保有足夠儲存空間：
   ```zsh
   # 檢查磁碟空間
   df -h
   ```

### 自定義下載

如需自定義更多下載參數，可直接修改 `data/era5_downloader.py` 文件。
