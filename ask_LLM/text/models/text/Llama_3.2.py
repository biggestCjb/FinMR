from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from IPython.display import display, Markdown,Image
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pathlib import Path
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import Generation
from IPython.display import display, Markdown
from langchain.tools import Tool
import warnings
from sentence_transformers import SentenceTransformer
# import torch
from langchain_ollama.llms import OllamaLLM
import warnings
warnings.filterwarnings("ignore")
from langchain_ollama import OllamaLLM
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../tools")))
from config import load_env_vars
from processing import process_questions


if __name__ == '__main__':

    LLMmodel = OllamaLLM(
        model = "llama3.2",
        temperature = 0.7,
    )

    paths = {
    "error_log_path": load_env_vars().get("ErrorLogPath") + "Llama/Llama_3.2_ErrorLog.json",
    "zeroshot_path": load_env_vars().get("cot_path") + "Llama/Llama_3.2_zeroshot.json",
    "fewshot_path": load_env_vars().get("cot_path") + "Llama/Llama_3.2__fewshot.json",
    "dataset_path": load_env_vars().get("dataset"),
    }

    process_questions(LLMmodel, paths)






