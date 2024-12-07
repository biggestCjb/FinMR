from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import base64
import re
from IPython.display import display, Markdown,Image
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json
from pathlib import Path
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import Generation
from IPython.display import display, Markdown
from langchain.tools import Tool

import warnings

from sentence_transformers import SentenceTransformer
# import torch
from langchain_ollama.llms import OllamaLLM

warnings.filterwarnings("ignore")
from pathlib import Path
from langchain_openai import ChatOpenAI
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../tools")))
from config import load_env_vars
from processing import process_questions


if __name__ == '__main__':

    LLMmodel = ChatOpenAI(
        model = "gpt-o1", 
        api_key = load_env_vars().get("api_key"),
        base_url = load_env_vars().get("base_url"), 
        temperature = 0.7
    )

    paths = {
    "error_log_path": load_env_vars().get("ErrorLogPath") + "gpt/gpt_o1_ErrorLog.json",
    "zeroshot_path": load_env_vars().get("cot_path") + "gpt/gpt_o1_zeroshot.json",
    "fewshot_path": load_env_vars().get("cot_path") + "gpt/gpt_o1__fewshot.json",
    "dataset_path": load_env_vars().get("dataset"),
    }

    process_questions(LLMmodel, paths)




