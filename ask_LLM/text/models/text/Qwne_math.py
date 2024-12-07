import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from langchain_openai import ChatOpenAI


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../tools")))
from config import load_env_vars
from processing import process_questions


if __name__ == '__main__':

    LLMmodel = ChatOpenAI(
            model = "qwen-math",
            api_key = load_env_vars().get("api_key"),
            base_url = load_env_vars().get("base_url"),
            temperature = 0.7
        )

    paths = {
    "error_log_path": load_env_vars().get("ErrorLogPath") + "qwen/qwen_math_ErrorLog.json",
    "zeroshot_path": load_env_vars().get("cot_path") + "qwen/qwen_math_zeroshot.json",
    "fewshot_path": load_env_vars().get("cot_path") + "qwen/qwen_math__fewshot.json",
    "dataset_path": load_env_vars().get("dataset"),
    }

    process_questions(LLMmodel, paths)
