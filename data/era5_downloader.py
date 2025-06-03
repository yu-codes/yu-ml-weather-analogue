import os
import json
import argparse
import logging
from cdsapi import Client
from dotenv import load_dotenv
from datetime import datetime

SINGLE_LEVEL_VARS = [
    "2m_temperature", "2m_dewpoint_temperature", "10m_u_component_of_wind",
    "10m_v_component_of_wind", "surface_pressure", "mean_sea_level_pressure",
    "total_precipitation", "surface_runoff", "snow_depth", "cape",
    "geopotential", "volumetric_soil_water_layer_1"
]


def load_config():
    load_dotenv()
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    cds_key = os.getenv("CDS_API_KEY")
    cds_url = os.getenv("CDS_API_URL", "https://cds.climate.copernicus.eu/api/v2")
    return config, cds_key, cds_url


def parse_args():
    parser = argparse.ArgumentParser(description="Download ERA5 data")
    parser.add_argument("--variable", type=str, help="ERA5 variable")
    parser.add_argument("--pressure_level", type=str, help="Pressure level (e.g. 500)")
    parser.add_argument("--level_type", type=str, choices=["single", "pressure"], help="Level type")
    parser.add_argument("--start_year", type=int, help="Start year")
    parser.add_argument("--end_year", type=int, help="End year")
    parser.add_argument("--area", type=float, nargs=4, help="Area: N W S E")
    parser.add_argument("--grid", type=float, nargs=2, help="Grid resolution (lat lon)")
    parser.add_argument("--output", type=str, default="./", help="Output base directory")
    parser.add_argument("--file_prefix", type=str, default="era5", help="Output file name prefix")
    return parser.parse_args()


def merge_args_with_config(args, config):
    defaults = config.get("default_settings", {})
    return {
        "variable": args.variable or defaults.get("variable", "temperature"),
        "pressure_level": args.pressure_level or defaults.get("pressure_level"),
        "level_type": args.level_type or defaults.get("level_type"),
        "start_year": args.start_year or defaults.get("start_year", 1980),
        "end_year": args.end_year or defaults.get("end_year", 2020),
        "area": args.area or defaults.get("area", [60, -10, 40, 5]),
        "grid": args.grid or defaults.get("grid", [0.25, 0.25]),
        "output": args.output,
        "file_prefix": args.file_prefix
    }


def get_dataset_type(variable, pressure_level=None, level_type=None):
    if level_type == "single":
        return "reanalysis-era5-single-levels"
    if level_type == "pressure":
        return "reanalysis-era5-pressure-levels"
    # fallback 自動推斷
    if variable in SINGLE_LEVEL_VARS or pressure_level is None:
        return "reanalysis-era5-single-levels"
    return "reanalysis-era5-pressure-levels"


def validate_params(params):
    if params["level_type"] == "pressure" and params["variable"] in SINGLE_LEVEL_VARS:
        raise ValueError(f"❌ Variable `{params['variable']}` is not valid for pressure-level dataset.")
    if params["level_type"] == "single" and params.get("pressure_level"):
        print("⚠️ Warning: pressure_level will be ignored for single-level dataset.")


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "download_errors.log")
    logging.basicConfig(filename=log_path, level=logging.ERROR)
    return log_path


def download_era5(params, cds_key, cds_url):
    os.makedirs(params["output"], exist_ok=True)
    setup_logging(os.path.join(params["output"], "logs"))
    dataset = get_dataset_type(params["variable"], params["pressure_level"], params["level_type"])
    client = Client(url=cds_url, key=cds_key)

    for year in range(params["start_year"], params["end_year"] + 1):
        for month in range(1, 13):
            try:
                month_str = f"{month:02d}"
                print(f"⬇ Downloading {params['variable']} for {year}-{month_str} ...")

                file_name = f"{year}_{month_str}_{params['file_prefix']}.nc"
                file_path = os.path.join(params["output"], file_name)

                request = {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": params["variable"],
                    "year": str(year),
                    "month": [month_str],
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": params["area"],
                    "grid": params["grid"],
                }

                if dataset == "reanalysis-era5-pressure-levels":
                    request["pressure_level"] = [str(params["pressure_level"])]

                print(f"🔍 Request preview:\n{json.dumps(request, indent=2)}\n")

                client.retrieve(dataset, request, file_path)

            except Exception as e:
                logging.error(f"{datetime.now()} - Error downloading {year}-{month_str}: {e}")
                print(f"❌ Failed for {year}-{month_str}: {e}")




if __name__ == "__main__":
    args = parse_args()
    config, cds_key, cds_url = load_config()
    if not cds_key:
        raise ValueError("❌ CDS_API_KEY not found in .env")

    merged_params = merge_args_with_config(args, config)
    validate_params(merged_params)
    download_era5(merged_params, cds_key, cds_url)
