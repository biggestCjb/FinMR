# from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage
import base64
import re
from IPython.display import display, Markdown,Image
from langchain_core.prompts import ChatPromptTemplate
import json
from pathlib import Path
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import Generation
from IPython.display import display, Markdown
from langchain.tools import Tool
from sentence_transformers import SentenceTransformer
from langchain_ollama.llms import OllamaLLM
import warnings
warnings.filterwarnings("ignore")
from langchain_openai import ChatOpenAI
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../tools")))
from config import load_env_vars
from processing import process_questions


if __name__ == '__main__':

    LLMmodel = ChatOpenAI(
        model = "claude-3-5-sonnet-20241022",
        api_key = load_env_vars().get("api_key"),
        base_url = load_env_vars().get("base_url"),
        temperature = 0.7
    )

    paths = {
    "error_log_path": load_env_vars().get("ErrorLogPath") + "claude/claude_3.5_ErrorLog.json",
    "zeroshot_path": load_env_vars().get("cot_path") + "claude/claude_3.5_zeroshot.json",
    "fewshot_path": load_env_vars().get("cot_path") + "claude/claude_3.5_fewshot.json",
    "dataset_path": load_env_vars().get("dataset"),
    }

    process_questions(LLMmodel, paths)

