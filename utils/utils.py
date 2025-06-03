import os
import json


def set_working_directory():
    """
    將工作目錄設置為與 `.env` 文件相同的目錄。
    如果 `.env` 文件不存在，則默認為當前腳本所在目錄。
    """
    try:
        # 嘗試找到 .env 文件的路徑
        current_path = os.getcwd()
        while True:
            if ".env" in os.listdir(current_path):
                os.chdir(current_path)
                print(f"Working directory set to: {current_path}")
                return
            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:  # 已到達文件系統根目錄
                break
            current_path = parent_path

        # 如果未找到 .env 文件，默認設置為當前腳本所在目錄
        script_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_path)
        print(f".env not found. Working directory set to script location: {script_path}")
    except Exception as e:
        print(f"Error setting working directory: {e}")


def load_json_config(filename):
    """
    加載 JSON 配置文件，支持從環境變量、工作目錄、父目錄或絕對路徑加載。
    """
    # 從環境變量加載
    env_path = os.getenv("CONFIG_PATH_YUHAN")
    if env_path and os.path.isfile(os.path.join(env_path, filename)):
        print("Read Config from Environmental Variables")
        with open(os.path.join(env_path, filename), "r") as file:
            return json.load(file)

    # 從當前工作目錄加載
    if os.path.isfile(os.path.join(os.getcwd(), filename)):
        print("Read Config from Work Directory")
        with open(os.path.join(os.getcwd(), filename), "r") as file:
            return json.load(file)

    # 從父目錄加載
    parent_dir = os.path.dirname(os.getcwd())
    grandparent_dir = os.path.dirname(parent_dir)
    grandparent_file_path = os.path.join(grandparent_dir, filename)
    if os.path.isfile(grandparent_file_path):
        print("Read Config from Parent Directory")
        with open(grandparent_file_path, "r") as file:
            return json.load(file)

    # 從絕對路徑加載
    if os.path.isfile(filename):
        print("Read Config from Absolute Directory")
        with open(filename, "r") as file:
            return json.load(file)

    raise FileNotFoundError(f"Not Found: {filename}")


def ensure_directory_exists(path):
    """
    確保指定的目錄存在，如果不存在則創建。
    """
    if not os.path.exists(path):
        os.makedirs(path)
