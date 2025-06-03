# 📘 ERA5 常用變量清單（Pressure Level + Single Level）

這份文檔提供了 ERA5 再分析資料中常用變數的描述及其與極端降水事件分析的相關性。

## ☁️ 壓力層變量（Pressure Level Variables）

這些變量需搭配 `pressure_level` 指定高度層（如 500, 850 hPa）：

| 變量名稱 (中文)       | CDS 變量名稱 (`variable`)           | 單位/說明                      | 與極端降水相關性                                         |
|------------------------|-------------------------------------|--------------------------------|----------------------------------------------------------|
| 溫度                   | `temperature`                       | K（開氏溫度）                  | 不同高度溫度差異影響大氣穩定度，暖層上方的冷層利於對流  |
| 位勢高度               | `geopotential`                      | m²/s²，可除以 9.8 近似為高度(m) | 高度場波動顯示槽脊系統，與天氣系統和降水區域密切相關    |
| 東西向風（U）         | `u_component_of_wind`               | m/s                           | 與水汽輸送和天氣系統移動有關，影響降水持續時間和範圍    |
| 南北向風（V）         | `v_component_of_wind`               | m/s                           | 與水汽輸送和天氣系統移動有關，南風常帶來暖濕氣流        |
| 相對濕度               | `relative_humidity`                 | %                             | 直接顯示大氣含水量，高相對濕度是降水形成的必要條件      |
| 比濕                   | `specific_humidity`                 | kg/kg                         | 表示單位空氣中的水汽含量，用於計算可降水量和水汽通量    |
| 垂直速度               | `vertical_velocity`                 | Pa/s                          | 上升運動（負值）有利於雲和降水形成                       |
| 位渦度                 | `potential_vorticity`               | K·m²/kg·s                     | 高位渦度區域與強對流和降水系統相關                       |
| 渦度                   | `vorticity`                         | 1/s                           | 正渦度區域與上升運動相關，有利於雲和降水發展             |
| 散度                   | `divergence`                        | 1/s                           | 低層輻合（負散度）和高層輻散（正散度）有利於強降水      |

---

## 🌍 單一層級變量（Single Level Variables）

這些變量不需要壓力層設定，直接可下載：

| 變量名稱 (中文)       | CDS 變量名稱 (`variable`)           | 單位/說明                     | 與極端降水相關性                                        |
|------------------------|-------------------------------------|-------------------------------|----------------------------------------------------------|
| 地表溫度               | `2m_temperature`                    | K，近地面 2 公尺              | 高溫增加大氣持水能力，暖空氣可攜帶更多水汽               |
| 地表比濕               | `2m_specific_humidity`              | kg/kg                        | 直接顯示近地表水汽含量，與降水潛力直接相關               |
| 地表露點溫度           | `2m_dewpoint_temperature`           | K                            | 溫度與露點溫度差值小表示濕度大，是降水形成重要指標       |
| 地表氣壓               | `surface_pressure`                  | Pa                           | 低壓系統常與降水相關聯，氣壓變化可指示天氣系統移動       |
| 海平面氣壓             | `mean_sea_level_pressure`           | Pa                           | 顯示高低壓系統分布，低壓中心和鋒面系統與強降水關聯       |
| 地表風速 U 分量        | `10m_u_component_of_wind`           | m/s                          | 與近地表水汽輸送有關，影響對流系統發展                   |
| 地表風速 V 分量        | `10m_v_component_of_wind`           | m/s                          | 與近地表水汽輸送有關，南風通常帶來高濕度空氣             |
| 地形高度（地形圖）     | `geopotential`                      | 固定地表高度                 | 地形抬升效應可增強降水，地形引導氣流影響降水分布         |
| 降水量（逐時）         | `total_precipitation`               | m（毫米為 m × 1000）         | 直接測量降水，是極端降水事件研究的基本變量               |
| 地表徑流               | `surface_runoff`                    | m                            | 與強降水後的地表水流有關，可指示土壤飽和程度             |
| 積雪深度               | `snow_depth`                        | m                            | 積雪融化可能導致洪水，也影響地表反照率和能量平衡         |
| 土壤濕度               | `volumetric_soil_water_layer_1`     | m³/m³                        | 高土壤濕度減少地表吸收能力，使後續降水更易形成徑流       |
| 雷暴指標（對流可用能） | `cape`                              | J/kg                         | 高 CAPE 值表示強對流潛力，與雷雨和強降水直接相關         |

---

## 📥 資料下載說明

- 若你需要 **壓力層變量**，請使用：
  - dataset: `reanalysis-era5-pressure-levels`
  - 加入 `pressure_level` 參數
- 若你需要 **單層變量**，請使用：
  - dataset: `reanalysis-era5-single-levels`

請參考 `data/download_era5.py` 的自動化下載工具。

---

## 🌧️ 極端降水事件分析推薦變數組合

根據氣象學原理和機器學習需求，以下變數組合特別適合極端降水事件分析：

### 1. 基礎組合（6個變數）
* `total_precipitation`：直接測量降水量
* `mean_sea_level_pressure`：識別天氣系統
* `2m_temperature` 和 `2m_dewpoint_temperature`：近地表溫度和水汽
* `10m_u_component_of_wind` 和 `10m_v_component_of_wind`：近地表風場

### 2. 進階組合（11個變數）
* 基礎組合的所有變數
* `850hPa` 層的 `temperature`、`relative_humidity`、`u_component_of_wind` 和 `v_component_of_wind`：低層大氣特性
* `500hPa` 層的 `geopotential`：中層大氣波動

### 3. 完整研究組合（15個變數以上）
* 進階組合的所有變數
* `cape`：對流潛力
* 多個壓力層（如 300hPa、700hPa）的動力和熱力變數
* 衍生變數：
  * 垂直整合水汽通量 (IVT)：由風場和比濕計算
  * 可降水量：垂直積分的大氣含水量
  * 溫度平流：溫度的水平輸送
  * 渦度平流：渦度的水平輸送

### 4. 常用壓力層
* **850hPa**：低層大氣（約 1.5 公里高）
* **500hPa**：中層大氣（約 5.5 公里高）
* **300hPa** 或 **250hPa**：高層大氣（近噴射氣流高度）

---

> 📌 資料來源：Copernicus Climate Data Store (CDS)  
> 🔗 官方連結：https://cds.climate.copernicus.eu/

