import os
import sys
import numpy as np
import pandas as pd
import json

import xarray as xr
import h5py
import netCDF4


def merge_files(data_dir, output_dir, merge_by="D"):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(data_dir) if f.endswith(".nc")]
    files_by_group = {}
    for file in files:
        if merge_by == "Y":
            group_key = file[:4]
        elif merge_by == "D":
            group_key = file[:8]
        else:
            raise ValueError(
                "Unsupported merge_by value. Use 'D' for days or 'Y' for years."
            )

        if group_key not in files_by_group:
            files_by_group[group_key] = []
        files_by_group[group_key].append(os.path.join(data_dir, file))

    for group_key in sorted(files_by_group.keys()):
        file_list = files_by_group[group_key]
        file_list.sort()
        output_file = os.path.join(output_dir, f"{group_key}.nc")

        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping.")
            continue

        ds = xr.open_mfdataset(file_list, combine="by_coords")
        ds.to_netcdf(output_file)
        print(f"Saved merged file for {group_key} to {output_file}")
        ds.close()


"""dirs = [
    (
        "../../../data_peichun/all/temperature",
        "../../../data_peichun/all/temperature_by_day",
    ),
]

for dir in dirs:
    data_dir, output_dir = dir
    merge_files(data_dir, output_dir, merge_by="D")"""


def load_weather_data(data_dir):
    """
    加載指定目錄中的所有 netCDF 文件並合併。
    支持按月份或按年份保存的文件格式。
    
    Args:
        data_dir: 數據目錄路徑（可以是絕對路徑或相對路徑）
    
    Returns:
        合併後的 xarray 數據集
    """
    # 檢查是否為絕對路徑，如果不是則轉換為絕對路徑
    if not os.path.isabs(data_dir):
        # 取得調用此函數的腳本所在目錄
        caller_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        # 將相對路徑轉換為絕對路徑
        data_dir = os.path.abspath(os.path.join(caller_dir, data_dir))
    
    print(f"Loading data from {data_dir}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"數據目錄不存在: {data_dir}")
    
    # 搜索所有 netCDF 文件
    nc_files = [f for f in os.listdir(data_dir) if f.endswith(".nc")]
    if not nc_files:
        raise ValueError(f"在 {data_dir} 中未找到 .nc 文件")
    
    nc_files = sorted(nc_files)  # 確保文件按名稱排序
    file_paths = [os.path.join(data_dir, filename) for filename in nc_files]
    
    print(f"Found {len(nc_files)} netCDF files, starting with {nc_files[0]}")
    
    # 使用 xarray 的 open_mfdataset 合併所有文件
    try:
        combined_data = xr.open_mfdataset(
            file_paths, combine="by_coords", chunks={"time": "auto"}
        )
        print(f"Data loaded successfully with dimensions: {combined_data.dims}")
        return combined_data
    except Exception as e:
        print(f"Error loading data: {e}")
        # 嘗試逐個加載並手動合併
        print("Trying to load files individually...")
        datasets = []
        for file_path in file_paths:
            try:
                ds = xr.open_dataset(file_path)
                datasets.append(ds)
                print(f"Successfully loaded {os.path.basename(file_path)}")
            except Exception as inner_e:
                print(f"Failed to load {os.path.basename(file_path)}: {inner_e}")
        
        if not datasets:
            raise ValueError("無法加載任何數據文件")
        
        # 手動合併數據集
        combined_data = xr.merge(datasets)
        print(f"Data loaded successfully with dimensions: {combined_data.dims}")
        return combined_data



def process_weather_data(
    dataset,
    variable,
    crop_to_32x32=True,
    freq="6h",
    resample_method="mean",
    preprocessing_method="raw",
    weight_matrix=None,
    year_range=None,
    alpha=0.2,  # 添加 alpha 參數來控制 log 壓縮
):
    data = dataset[variable]

    if year_range is not None:
        start_year, end_year = year_range
        data = data.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    if crop_to_32x32:
        data = data.isel(latitude=slice(0, 32), longitude=slice(0, 32))

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
        # 初次標準化 (z = (x - mean1) / std1)
        data_mean1 = resampled_data.mean()
        data_std1 = resampled_data.std()
        standardized_data = (resampled_data - data_mean1) / data_std1
        print(f"Step 1: Standardized Data - Mean: {data_mean1.values}, Std: {data_std1.values}")

        # 對數轉換 (w = sign(z) * log(1 + alpha * |z|))
        log_transformed_data = np.sign(standardized_data) * np.log(1 + alpha * np.abs(standardized_data))
        log_mean = log_transformed_data.mean()
        log_std = log_transformed_data.std()
        print(f"Step 2: Log-transformed Data - Mean: {log_mean.values}, Std: {log_std.values}")

        # 再次標準化 (y = (w - mean2) / std2)
        data_mean2 = log_transformed_data.mean()
        data_std2 = log_transformed_data.std()
        resampled_data = (log_transformed_data - data_mean2) / data_std2
        print(f"Step 3: Re-standardized Data - Mean: {data_mean2.values}, Std: {data_std2.values}")

    elif preprocessing_method == "normalized":
        data_min = resampled_data.min()
        data_max = resampled_data.max()
        resampled_data = (resampled_data - data_min) / (data_max - data_min)
    elif preprocessing_method == "standardized":
        data_mean = resampled_data.mean()
        data_std = resampled_data.std()
        resampled_data = (resampled_data - data_mean) / data_std
    else:
        pass

    return resampled_data


def process_weather_data_overlapped(
    dataset,
    variable,
    crop_to_32x32=True,
    freq="3h",
    resample_method="mean",
    preprocessing_method="raw",
    weight_matrix=None,
    year_range=None,
    overlap=2 / 3,  # Overlap for rolling windows
    alpha=0.2,  # 添加 alpha 參數來控制 log 壓縮
):
    data = dataset[variable]

    if year_range is not None:
        start_year, end_year = year_range
        data = data.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    if crop_to_32x32:
        data = data.isel(latitude=slice(0, 32), longitude=slice(0, 32))

    window_size = int(pd.to_timedelta(freq).total_seconds() / 3600)
    step_size = int(window_size * (1 - overlap))

    rolling_data = (
        data.rolling(time=window_size, center=False)
        .construct("window_dim")
        .isel(time=slice(0, None, step_size))
    )

    if resample_method == "mean":
        resampled_data = rolling_data.mean(dim="window_dim")
    elif resample_method == "sum":
        resampled_data = rolling_data.sum(dim="window_dim")
    else:
        raise ValueError("Unsupported method. Choose 'mean' or 'sum'.")

    if weight_matrix is not None:
        if weight_matrix.shape != (32, 32):
            raise ValueError("The shape of the weight matrix must be (32, 32).")
        resampled_data = resampled_data * weight_matrix

    if preprocessing_method == "log":
        # 初次標準化 (z = (x - mean1) / std1)
        data_mean1 = resampled_data.mean()
        data_std1 = resampled_data.std()
        standardized_data = (resampled_data - data_mean1) / data_std1
        print(f"Step 1: Standardized Data - Mean: {data_mean1.values}, Std: {data_std1.values}")

        # 對數轉換 (w = sign(z) * log(1 + alpha * |z|))
        log_transformed_data = np.sign(standardized_data) * np.log(1 + alpha * np.abs(standardized_data))
        log_mean = log_transformed_data.mean()
        log_std = log_transformed_data.std()
        print(f"Step 2: Log-transformed Data - Mean: {log_mean.values}, Std: {log_std.values}")

        # 再次標準化 (y = (w - mean2) / std2)
        data_mean2 = log_transformed_data.mean()
        data_std2 = log_transformed_data.std()
        resampled_data = (log_transformed_data - data_mean2) / data_std2
        # print(f"Step 3: Re-standardized Data - Mean: {data_mean2.values}, Std: {data_std2.values}")

    elif preprocessing_method == "normalized":
        data_min = resampled_data.min()
        data_max = resampled_data.max()
        resampled_data = (resampled_data - data_min) / (data_max - data_min)
    elif preprocessing_method == "standardized":
        data_mean = resampled_data.mean()
        data_std = resampled_data.std()
        resampled_data = (resampled_data - data_mean) / data_std
    else:
        pass

    return resampled_data


def extract_data_with_timestamps(variable_data):
    timestamps = variable_data["time"].values

    values_with_no_nan = np.nan_to_num(variable_data.values)

    raw_data_with_timestamp = [
        [timestamp, data] for timestamp, data in zip(timestamps, values_with_no_nan)
    ]

    return raw_data_with_timestamp


def merge_data_by_timestamp(data_list):
    merged_data = {}
    for data in data_list:
        for row in data:
            timestamp = row[0]
            if timestamp not in merged_data:
                merged_data[timestamp] = [row[1]]
            else:
                merged_data[timestamp].append(row[1])
    merged_data_list = [
        [timestamp] + [np.stack(values)] for timestamp, values in merged_data.items()
    ]
    return merged_data_list


def list_to_netcdf(data_list, file_path):
    times = [item[0] for item in data_list]
    example_array = data_list[0][1]
    n = example_array.shape[0]

    # data = np.array([item[1][:, :32, :32] for item in data_list])
    data = np.array([item[1] for item in data_list])
    data_array = xr.DataArray(
        data,
        dims=["time", "channel", "y", "x"],
        coords={
            "time": times,
            "channel": np.arange(n),
            "y": np.arange(32),
            "x": np.arange(32),
        },
    )
    data_array.to_netcdf(file_path)
    print(f"Data saved to {file_path} as NetCDF format.")


def read_netcdf(file_path):
    ds = xr.open_dataset(file_path, engine="netcdf4")
    result_list = []
    for t in range(len(ds["time"])):
        time_point = ds["time"].values[t]
        data_array = ds.isel(time=t).to_array().values.squeeze()
        if data_array.ndim == 2:
            data_array = np.expand_dims(data_array, axis=0)
        result_list.append([time_point, data_array])
    return result_list


def read_netcdf_raw(file_path):
    ds = xr.open_dataset(file_path, engine="netcdf4")
    return ds


def list_to_hdf5(file_path, atmosphere_data):
    with h5py.File(file_path, "w") as hdf:
        for idx, (timestamp, array) in enumerate(atmosphere_data):
            group = hdf.create_group(str(idx))
            group.create_dataset("timestamp", data=timestamp)
            group.create_dataset("data", data=array, compression="gzip")


def read_hdf5(file_path):
    atmosphere_data = []
    with h5py.File(file_path, "r") as hdf:
        for group_key in hdf.keys():
            group = hdf[group_key]
            timestamp = group["timestamp"][()]
            data = group["data"][:]
            atmosphere_data.append((timestamp, data))
    return atmosphere_data
