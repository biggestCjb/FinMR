import os
from dotenv import load_dotenv

load_dotenv() 

file_path = os.path.abspath(__file__)
root_dir = os.path.abspath(os.path.join(file_path, "../../../..")) + "/"


def load_env_vars():
    """
    Loads environment variables and returns common paths and configuration information.
    Returns:
         dict: A dictionary containing information such as API keys, path configurations, and more.
    """
    return {
        "api_key": os.getenv("API_KEY"),
        "base_url": os.getenv("BASE_URL"),
        "root_dir": root_dir,
        "dataset": root_dir + "data/data_v5.json",
        "temp_path": root_dir + "images/tmp/resized_image.png",
        "model_path": root_dir + "model_weight/",
        "ErrorLogPath": root_dir + "errorLog/image+text/",
        "cot_path": root_dir + "outputs/image+text/",
    }
