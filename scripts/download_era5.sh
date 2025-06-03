#!/bin/bash

# filepath: /home/yuhan/Desktop/Master/yu-ml-weather-analogue/scripts/download_era5.sh

# 設置顏色輸出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 確保目錄結構存在
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_DIR}/data/raw/era5"

# 建立目錄
mkdir -p "${DATA_DIR}/single_level"
mkdir -p "${DATA_DIR}/pressure_level"
mkdir -p "${DATA_DIR}/logs"

# 檢查 .env 是否存在
if [ ! -f "${PROJECT_DIR}/.env" ]; then
    echo -e "${RED}Error: .env file not found in project directory.${NC}"
    echo -e "${YELLOW}Please create .env file with CDS_API_KEY=xxx:xxx${NC}"
    exit 1
fi

# 顯示可用變量
show_variables() {
    echo -e "${BLUE}Available single-level variables:${NC}"
    echo -e "  - 2m_temperature"
    echo -e "  - 2m_dewpoint_temperature"
    echo -e "  - 10m_u_component_of_wind"
    echo -e "  - 10m_v_component_of_wind"
    echo -e "  - surface_pressure"
    echo -e "  - mean_sea_level_pressure"
    echo -e "  - total_precipitation"
    
    echo -e "\n${BLUE}Available pressure-level variables:${NC}"
    echo -e "  - temperature"
    echo -e "  - u_component_of_wind"
    echo -e "  - v_component_of_wind"
    echo -e "  - geopotential"
    echo -e "  - relative_humidity"
    
    echo -e "\n${BLUE}Common pressure levels:${NC}"
    echo -e "  - 850 (hPa)"
    echo -e "  - 500 (hPa)"
    echo -e "  - 250 (hPa)"
}

# 下載單層變量
download_single_level() {
    local variable=$1
    local start_year=$2
    local end_year=$3
    local area=${4:-"60,-10,40,5"}  # 預設為歐洲區域
    local grid=${5:-"0.25,0.25"}    # 預設解析度
    
    local output_dir="${DATA_DIR}/single_level/${variable}"
    
    mkdir -p "$output_dir"
    
    echo -e "${GREEN}Downloading single level variable: ${variable} (${start_year}-${end_year})${NC}"
    echo -e "${YELLOW}Area: ${area}, Grid: ${grid}${NC}"
    
    # Split area and grid parameters
    IFS=',' read -r north west south east <<< "$area"
    IFS=',' read -r lat_res lon_res <<< "$grid"
    
    python "${PROJECT_DIR}/data/era5_downloader.py" \
        --variable "$variable" \
        --level_type "single" \
        --start_year "$start_year" \
        --end_year "$end_year" \
        --area $north $west $south $east \
        --grid $lat_res $lon_res \
        --output "$output_dir" \
        --file_prefix "${variable}"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully downloaded ${variable}${NC}"
    else
        echo -e "${RED}Failed to download ${variable}${NC}"
    fi
}

# 下載氣壓層變量
download_pressure_level() {
    local variable=$1
    local pressure=$2
    local start_year=$3
    local end_year=$4
    local area=${5:-"60,-10,40,5"}  # 預設為歐洲區域
    local grid=${6:-"0.25,0.25"}    # 預設解析度
    
    local output_dir="${DATA_DIR}/pressure_level/${pressure}hPa_${variable}"
    
    mkdir -p "$output_dir"
    
    echo -e "${GREEN}Downloading pressure level variable: ${variable} at ${pressure}hPa (${start_year}-${end_year})${NC}"
    echo -e "${YELLOW}Area: ${area}, Grid: ${grid}${NC}"
    
    # Split area and grid parameters
    IFS=',' read -r north west south east <<< "$area"
    IFS=',' read -r lat_res lon_res <<< "$grid"
    
    python "${PROJECT_DIR}/data/era5_downloader.py" \
        --variable "$variable" \
        --pressure_level "$pressure" \
        --level_type "pressure" \
        --start_year "$start_year" \
        --end_year "$end_year" \
        --area $north $west $south $east \
        --grid $lat_res $lon_res \
        --output "$output_dir" \
        --file_prefix "${pressure}hPa_${variable}"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully downloaded ${variable} at ${pressure}hPa${NC}"
    else
        echo -e "${RED}Failed to download ${variable} at ${pressure}hPa${NC}"
    fi
}

# 顯示使用說明
show_help() {
    echo -e "${BLUE}ERA5 Data Downloader${NC}"
    echo -e "Usage:"
    echo -e "  $0 [command] [options]"
    echo
    echo -e "Commands:"
    echo -e "  single VARIABLE START_YEAR END_YEAR [AREA] [GRID]   Download single-level variable"
    echo -e "  pressure VARIABLE LEVEL START_YEAR END_YEAR [AREA] [GRID]   Download pressure-level variable"
    echo -e "  help   Show this help message"
    echo -e "  vars   Show available variables"
    echo
    echo -e "Examples:"
    echo -e "  $0 single 2m_temperature 2010 2020"
    echo -e "  $0 single total_precipitation 2000 2020 \"56,-6,48,2\" \"0.25,0.25\""
    echo -e "  $0 pressure u_component_of_wind 850 2010 2020"
    echo -e "  $0 pressure temperature 500 2000 2020 \"56,-6,48,2\" \"0.25,0.25\""
    echo
    echo -e "Area format: \"North,West,South,East\" (default: \"60,-10,40,5\")"
    echo -e "Grid format: \"lat_resolution,lon_resolution\" (default: \"0.25,0.25\")"
}

# 主要邏輯
case "$1" in
    single)
        if [ $# -lt 4 ]; then
            echo -e "${RED}Error: Not enough arguments for single-level download${NC}"
            show_help
            exit 1
        fi
        download_single_level "$2" "$3" "$4" "${5:-60,-10,40,5}" "${6:-0.25,0.25}"
        ;;
    pressure)
        if [ $# -lt 5 ]; then
            echo -e "${RED}Error: Not enough arguments for pressure-level download${NC}"
            show_help
            exit 1
        fi
        download_pressure_level "$2" "$3" "$4" "$5" "${6:-60,-10,40,5}" "${7:-0.25,0.25}"
        ;;
    vars)
        show_variables
        ;;
    help|--help|-h|*)
        show_help
        ;;
esac