"""
ERA5 氣象資料預處理模組

此模組提供氣象資料的預處理功能，包括資料加載、重採樣、標準化等處理，
並支援多變量合併及保存為 netCDF 檔案。
"""
import os
import sys
import numpy as np
import pandas as pd
import json
import gc
import xarray as xr

# 載入自定義模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils_data import (
    load_weather_data,
    extract_data_with_timestamps,
    merge_data_by_timestamp,
    list_to_netcdf,
)

def process_weather_data(
    dataset,
    variable,
    crop_to_32x32=True,
    freq="6h",
    resample_method="mean",
    preprocessing_method="raw",
    weight_matrix=None,
    year_range=None,
    alpha=0.2,
):
    """
    處理氣象數據，包括時間選擇、空間剪裁、重採樣和標準化
    
    Args:
        dataset: xarray 數據集
        variable: 要處理的變量名
        crop_to_32x32: 是否將數據剪裁為 32x32 網格
        freq: 重採樣頻率 ("3h", "6h", "12h" 等)
        resample_method: 重採樣方法 ("mean", "sum", "none")
        preprocessing_method: 預處理方法 ("raw", "log", "normalized", "standardized")
        weight_matrix: 可選的權重矩陣，用於加權計算
        year_range: 年份範圍 (start_year, end_year)
        alpha: log 轉換參數
        
    Returns:
        處理後的 xarray DataArray
    """
    data = dataset[variable]

    if year_range is not None:
        start_year, end_year = year_range
        data = data.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    if crop_to_32x32:
        # 檢查數據維度並適當裁剪
        lat_dim = 'latitude' if 'latitude' in data.dims else 'lat'
        lon_dim = 'longitude' if 'longitude' in data.dims else 'lon'
        
        if lat_dim in data.dims and lon_dim in data.dims:
            lat_size = min(32, len(data[lat_dim]))
            lon_size = min(32, len(data[lon_dim]))
            print(f"裁剪數據為 {lat_size}x{lon_size} 網格")
            data = data.isel({lat_dim: slice(0, lat_size), lon_dim: slice(0, lon_size)})
        else:
            print(f"警告: 找不到經緯度維度 ({lat_dim}, {lon_dim})，跳過裁剪")
            crop_to_32x32 = False

    resampled_data = data.resample(time=freq)
    if resample_method == "mean":
        resampled_data = resampled_data.mean()
    elif resample_method == "sum":
        resampled_data = resampled_data.sum()
    elif resample_method == "none":
        resampled_data = resampled_data.nearest()
    else:
        raise ValueError("Unsupported method. Choose 'mean', 'sum', or 'none'.")

    if weight_matrix is not None:
        if weight_matrix.shape != (32, 32):
            raise ValueError("The shape of the weight matrix must be (32, 32).")
        resampled_data = resampled_data * weight_matrix

    if preprocessing_method == "log":
        data_mean1 = resampled_data.mean()
        data_std1 = resampled_data.std()
        standardized_data = (resampled_data - data_mean1) / data_std1

        log_transformed_data = np.sign(standardized_data) * np.log(1 + alpha * np.abs(standardized_data))
        data_mean2 = log_transformed_data.mean()
        data_std2 = log_transformed_data.std()
        resampled_data = (log_transformed_data - data_mean2) / data_std2

    elif preprocessing_method == "normalized":
        data_min = resampled_data.min()
        data_max = resampled_data.max()
        resampled_data = (resampled_data - data_min) / (data_max - data_min)
    elif preprocessing_method == "standardized":
        data_mean = resampled_data.mean()
        data_std = resampled_data.std()
        resampled_data = (resampled_data - data_mean) / data_std
    # raw: do nothing

    return resampled_data

def preprocess_and_save(
    data_directories,
    freq,
    resample_method,
    preprocessing_method,
    year_range,
    save_data_directory="processed",
    overlap=0,
    weighted=False,
    alpha=0.2,
):
    """
    預處理並保存多個變量的氣象數據
    
    Args:
        data_directories: 元組列表 [(目錄路徑, 變量名)]
        freq: 時間頻率 (如 "3h", "6h")
        resample_method: 重採樣方法 ("mean", "sum", "none")
        preprocessing_method: 預處理方法 ("raw", "log", "normalized", "standardized")
        year_range: 年份範圍 (start_year, end_year)
        save_data_directory: 保存目錄
        overlap: 重疊參數
        weighted: 是否使用權重
        alpha: log 轉換參數
    """
    # 獲取所有變量的名稱組合
    variables = "".join([item[1] for item in data_directories])
    
    # 確保保存目錄是相對於項目根目錄的絕對路徑
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_dir = os.path.join(project_root, "data", save_data_directory)
    os.makedirs(save_dir, exist_ok=True)
    
    # 創建輸出文件名
    file_path = os.path.join(save_dir, 
                            f"{variables}_{freq}_{resample_method}_{preprocessing_method}_{year_range[0]}{year_range[1]}")
    
    # 加載權重矩陣（如果需要）
    if weighted:
        weights_path = os.path.join(project_root, "dataset", "weights.json")
        if os.path.exists(weights_path):
            with open(weights_path, "r") as f:
                loaded_data = json.load(f)
            weight_matrix = np.array(loaded_data["grid_weights"])
            file_path += "_weighted"
        else:
            print(f"警告: 權重文件不存在: {weights_path}，將不使用權重")
            weight_matrix = None
            weighted = False
    else:
        weight_matrix = None

    if overlap != 0:
        file_path += f"_overlap"
    file_path += ".h5"

    print(f"處理數據並保存到: {file_path}")
    print(f"數據變量: {variables}")
    print(f"時間頻率: {freq}, 重採樣方法: {resample_method}, 預處理方法: {preprocessing_method}")
    
    data_combined = []
    for data_dir, variable in data_directories:
        print(f"\n處理變量: {variable} 從目錄: {data_dir}")
        
        # 確保目錄路徑是絕對路徑或相對於正確位置的路徑
        if not os.path.isabs(data_dir):
            # 首先檢查是否是相對於項目根目錄的路徑
            abs_path = os.path.join(project_root, data_dir.lstrip('/'))
            if not os.path.exists(abs_path):
                # 然後檢查是否是相對於 raw/era5 目錄的路徑
                abs_path = os.path.join(project_root, "data", "raw", "era5", data_dir.lstrip('/'))
            
            if os.path.exists(abs_path):
                data_dir = abs_path
        
        # 加載數據
        try:
            loaded_data = load_weather_data(data_dir)
            
            # 檢查時間維度的名稱 (ERA5可能使用'valid_time'或'time')
            time_dim = 'time'
            if 'valid_time' in loaded_data.dims and 'time' not in loaded_data.dims:
                time_dim = 'valid_time'
                print(f"檢測到時間維度名稱為 '{time_dim}'，將其重命名為 'time'")
                loaded_data = loaded_data.rename({time_dim: 'time'})
            
            # 確保時間格式正確
            loaded_data["time"] = pd.to_datetime(loaded_data["time"].values)
            
            # 處理數據
            processed_data = process_weather_data(
                loaded_data,
                variable,
                crop_to_32x32=True,
                freq=freq,
                resample_method=resample_method,
                preprocessing_method=preprocessing_method,
                weight_matrix=weight_matrix,
                year_range=year_range,
                alpha=alpha,
            )
            
            # 提取帶時間戳的數據
            data_with_timestamp = extract_data_with_timestamps(processed_data)
            data_combined.append(data_with_timestamp)
            
            # 釋放內存
            del loaded_data
            del processed_data
            gc.collect()
            
        except Exception as e:
            print(f"處理變量 {variable} 時出錯: {e}")
            raise
    
    # 沒有數據可處理時提前返回
    if not data_combined:
        print("沒有數據可處理，退出")
        return
    
    # 合併所有變量的數據
    print("\n合併所有變量的數據...")
    atmosphere_data = merge_data_by_timestamp(data_combined)
    
    # 釋放內存
    del data_combined
    gc.collect()

    # 保存處理後的數據
    print(f"\n保存處理後的數據到 {file_path}")
    list_to_netcdf(atmosphere_data, file_path)
    
    # 釋放內存
    del atmosphere_data
    gc.collect()
    
    print(f"數據處理完成，已保存到: {file_path}")
    print(f"文件大小: {convert_size(os.path.getsize(file_path))}")

def convert_size(size_bytes):
    """將字節大小轉換為易讀的格式 (B, KB, MB, GB, TB)"""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    size = round(size_bytes / p, 2)
    return f"{size} {size_name[i]}"

# 範例使用（可以移到單獨的腳本中）
if __name__ == "__main__":
    # 數據目錄示例（可根據實際目錄結構調整）
    # 方式1：使用相對於項目根目錄的路徑
    data_directories = [
        ("data/raw/era5/single_level/2m_temperature", "t2m"),
        # 可以添加更多變量
    ]
    
    # 方式2：使用絕對路徑（不依賴於執行腳本的位置）
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # data_directories = [
    #     (os.path.join(project_root, "data/raw/era5/single_level/2m_temperature"), "t2m"),
    # ]
    
    # 處理參數
    freq = "3h"                 # 時間頻率
    resample_method = "mean"    # 重採樣方法: mean, sum, none
    preprocessing_method = "standardized"  # 預處理方法: raw, log, normalized, standardized
    year_range = (2010, 2010)   # 年份範圍
    
    # 調用預處理函數
    preprocess_and_save(
        data_directories,
        freq,
        resample_method,
        preprocessing_method,
        year_range,
        save_data_directory="processed",  # 保存目錄（相對於項目根目錄的 data 目錄）
        overlap=0,
        weighted=False,
        alpha=0.2,
    )