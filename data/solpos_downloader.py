import os
import sys
import pandas as pd
import pvlib
import pytz

def generate_solar_position_csv(
    latitude=52.5,
    longitude=-1.9,
    start="1940-01-01",
    end="2020-12-31",
    freq="H",
    timezone="Europe/London",
    output_csv="solar_position.csv"
):
    # 設定時區
    british_tz = pytz.timezone(timezone)

    # 產生時間戳
    date_range = pd.date_range(start=start, end=end, freq=freq)
    unix_timestamps = date_range.astype(int) // 10**9

    # 轉換為時區感知時間
    timestamps_utc = pd.to_datetime(unix_timestamps, unit="s", utc=True)
    timestamps_british = timestamps_utc.tz_convert(british_tz)

    # 計算太陽位置
    solpos = pvlib.solarposition.get_solarposition(timestamps_british, latitude, longitude)
    solpos.index = unix_timestamps

    # 去除重複索引
    df_final = solpos[~solpos.index.duplicated(keep="first")]

    # 儲存為 CSV
    df_final.to_csv(output_csv)
    print(f"Solar position calculation completed and saved to {output_csv}.")

if __name__ == "__main__":
    generate_solar_position_csv()