import json
import faiss
import torch
import os
import pandas as pd

from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

from config import load_env_vars


def init_faiss():
    """
        Initialize the Faiss vector database.

        Returns:
            faiss. IndexFlatIP: The initialized Faiss index object.
    """
    index = faiss.IndexFlatIP(1024)  # 使用内积度量
    if index.ntotal==0:
        pass
    
    else:
        index.reset()
    # 检查索引是否清空
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


def clipEmbedding(data):
    """
    Generate multimodal embeddings for text and images.

    Args:
        data (dict): Dictionary containing question text and image path.

    Returns:
        torch.Tensor: Concatenated multimodal embedding vector.
    """
    # Initialize model and processor
    clip_path = load_env_vars().get("model_path") + "clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(clip_path)
    processor = CLIPProcessor.from_pretrained(clip_path)
    
    textdata = "Question:" + data.get("Question Text") + " Options:" + str(data.get("Options")) + " Correct Answer:" + data.get("Answer")
    
    # Check if an image is provided
    if data.get("Image") != '':
        image_path = load_env_vars().get("root_dir") + data.get("Image") 
        
        image = Image.open(image_path)
        inputs = processor(text=[textdata], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
        
        # Generate embeddings using the CLIP model
        outputs = model(**inputs)
        image_embedding = outputs.image_embeds  # Image embeddings
        text_embedding = outputs.text_embeds  # Text embeddings
    else:
        inputs = processor(text=[textdata], return_tensors="pt", padding=True, truncation=True, max_length=77)
        text_embedding = model.get_text_features(**inputs)
        
        image_embedding = torch.zeros((text_embedding.shape[0], 512)) 
    
    # Concatenate text and image embeddings
    combined_embedding = torch.cat((text_embedding, image_embedding), dim=-1)
    
    return combined_embedding


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
    """
    Add embeddings generated from a single data point to the Faiss index.

    Args:
        index (faiss.IndexFlatIP): Faiss index object.
        data (dict): Single data point containing text and image path.

    Returns:
        faiss.IndexFlatIP: Updated Faiss index object.
    """
    error_log_embedding = clipEmbedding(data)
    
    error_log_embedding = normalize(error_log_embedding)
    
    error_log_embedding_np = error_log_embedding.detach().numpy()  # Ensure conversion to numpy format
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
    query_embedding = clipEmbedding(query_data)
    query_embedding = normalize(query_embedding) 
    
    query_embedding_np = query_embedding.detach().numpy()
    
    # Search the Faiss index for the top-k most similar vectors
    D, I = index.search(query_embedding_np, k)  # D is similarity scores, I is the corresponding indices
    
    return D, I
