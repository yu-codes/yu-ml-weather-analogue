#!/usr/bin/env python
# filepath: /home/yuhan/Desktop/Master/yu-ml-weather-analogue/test/test_data_processing.py

"""
測試資料預處理功能
此腳本用於測試 data_preprocessor.py 的功能，特別是其對按月份儲存的 ERA5 資料的處理能力
"""

import os
import sys
import time

# 添加項目根目錄到 Python 路徑
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from data.data_preprocessor import preprocess_and_save

def test_single_variable():
    """測試單一變數的數據預處理功能"""
    print("開始測試單一變數的數據預處理功能...")
    
    # 定義測試參數
    data_directories = [
        ("data/raw/era5/single_level/2m_temperature", "t2m"),
    ]
    
    # 處理參數
    freq = "6h"                  # 時間頻率
    resample_method = "mean"     # 重採樣方法: mean, sum, none
    preprocessing_method = "standardized"  # 預處理方法: raw, log, normalized, standardized
    year_range = (2010, 2010)    # 測試單一年份
    
    # 記錄開始時間
    start_time = time.time()
    
    try:
        # 調用預處理函數
        preprocess_and_save(
            data_directories,
            freq,
            resample_method,
            preprocessing_method,
            year_range,
            save_data_directory="processed/test",  # 使用測試目錄
            overlap=0,
            weighted=False,
            alpha=0.2,
        )
        
        # 計算執行時間
        elapsed_time = time.time() - start_time
        print(f"\n測試完成！執行時間: {elapsed_time:.2f} 秒")
        print("請檢查生成的文件，確保數據正確處理。")
        
        return True
        
    except Exception as e:
        print(f"\n測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        print("\n請檢查錯誤並修正代碼。")
        
        return False

def test_multiple_variables():
    """測試多變數的數據預處理功能"""
    print("開始測試多變數的數據預處理功能...")
    
    # 定義測試參數
    data_directories = [
        ("data/raw/era5/single_level/2m_temperature", "t2m"),
        ("data/raw/era5/single_level/mean_sea_level_pressure", "msl"),
    ]
    
    # 處理參數
    freq = "6h"                  # 時間頻率
    resample_method = "mean"     # 重採樣方法: mean, sum, none
    preprocessing_method = "standardized"  # 預處理方法: raw, log, normalized, standardized
    year_range = (2010, 2010)    # 測試單一年份
    
    # 記錄開始時間
    start_time = time.time()
    
    try:
        # 調用預處理函數
        preprocess_and_save(
            data_directories,
            freq,
            resample_method,
            preprocessing_method,
            year_range,
            save_data_directory="processed/test",  # 使用測試目錄
            overlap=0,
            weighted=False,
            alpha=0.2,
        )
        
        # 計算執行時間
        elapsed_time = time.time() - start_time
        print(f"\n測試完成！執行時間: {elapsed_time:.2f} 秒")
        print("請檢查生成的文件，確保數據正確處理。")
        
        return True
        
    except Exception as e:
        print(f"\n測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        print("\n請檢查錯誤並修正代碼。")
        
        return False

def main():
    """主函數，運行所有測試"""
    print("=" * 50)
    print("開始運行所有測試")
    print("=" * 50)
    
    tests = [
        ("單一變數測試", test_single_variable),
        ("多變數測試", test_multiple_variables),
    ]
    
    results = []
    for test_name, test_func in tests:
        print("\n" + "=" * 50)
        print(f"運行測試: {test_name}")
        print("=" * 50)
        
        success = test_func()
        results.append((test_name, success))
        
    print("\n" + "=" * 50)
    print("測試結果摘要:")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "通過 ✅" if success else "失敗 ❌"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n所有測試通過！👏")
    else:
        print("\n部分測試失敗，請檢查錯誤並修正代碼。")

if __name__ == "__main__":
    main()
