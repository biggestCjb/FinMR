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



model = SentenceTransformer('all-MiniLM-L6-v2')


def init_faiss():
    """
        Initialize the Faiss vector database.

        Returns:
            faiss. IndexFlatIP: The initialized Faiss index object.
    """
    index = faiss.IndexFlatIP(1024)  
    if index.ntotal==0:
        pass
    
    else:
        index.reset()
    print("Number of vectors after reset:", index.ntotal)
    return index


def index_faiss(index, file_path):
    """
    Load the data and add it to the Faiss index.

    Args:
        index (faiss. IndexFlatIP): Faiss index object.
        file_path (str): Error log file path.

    Returns:
        faiss. IndexFlatIP: The updated Faiss index object.
    """
    data = json.loads(Path(file_path).read_text())

    for error in data:
        storeEmbedding(index, error)

    print("Number of vectors after adding:", index.ntotal)

    return index


def textEmbedding(data):
    textdata = "Question:" + data.get("Question Text") + "image caption" +data.get("description") +"Options:" + str(data.get("Options")) + " Correct Answer:" + data.get("Answer") 
    
    text_embedding = model.encode([textdata], convert_to_tensor=True)
    
    return text_embedding


def normalize(embeddings):
    norms = torch.norm(embeddings, dim=1, keepdim=True)
    return embeddings / norms



def normalize(embeddings):
    """
    Normalize embedding vectors.

    Args:
        embeddings (torch.Tensor): Original embedding vectors.

    Returns:
        torch.Tensor: Normalized embedding vectors.
    """
    norms = torch.norm(embeddings, dim=1, keepdim=True)  
    return embeddings / norms  


def storeEmbedding(index, data):
    error_log_embedding = textEmbedding(data)
    error_log_embedding = normalize(error_log_embedding)
    error_log_embedding_np = error_log_embedding.detach().cpu().numpy() 
    index.add(error_log_embedding_np)  
    return index


def query_embedding_faiss(query_data, index, k=5):
    """
    Query the Faiss index for the most similar vectors to the input data.

    Args:
        query_data (dict): Query data containing text and image path.
        index (faiss.IndexFlatIP): Faiss index object.
        k (int): Number of similar vectors to return.

    Returns:
        tuple: (List of similarity scores, List of indices)
    """    
    query_embedding = textEmbedding(query_data)
    query_embedding = normalize(query_embedding)
    query_embedding_np = query_embedding.detach().cpu().numpy()
    
    D, I = index.search(query_embedding_np, k)
    
    return D, I
