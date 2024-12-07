import json
import os
from tqdm import tqdm
from pathlib import Path
from utils import write_output, extract_answer, MarkdownParser
from prompts import inputPrompt, FeedbackPrompt, ICLPrompt
from embedding import init_faiss, index_faiss, query_embedding_faiss


def load_data_and_logs(dataset_path, error_log_path, zeroshot_path, fewshot_path):
    """
    Load dataset, error logs, and zero-shot and few-shot files. 
    If files do not exist, default files will be created.

    Args:
        dataset_path (str): Path to the dataset file.
        error_log_path (str): Path to the error log file.
        zeroshot_path (str): Path to the zero-shot inference file.
        fewshot_path (str): Path to the few-shot inference file.

    Returns:
        tuple: Contains the following:
            - data (list): Dataset content.
            - errorLOG (list): Error log content.
    """
   
    data = json.loads(Path(dataset_path).read_text())

    # Check and load error log. If it doesn't exist, create a default file.
    if os.path.exists(error_log_path):
        errorLOG = json.loads(Path(error_log_path).read_text())
    else:
        print("ErrorLOG does not exist!")
        print("Creating ErrorLOG ...")

        errorLOG = [{
            "ID": 9999,
            "Question Number": 9999,
            "Share Context": "",
            "Share Image": "",
            "Question Text": "text",
            "Image": "images/QuantitativeAnalysis1_images/40u.png",
            "Options": {"A": " -0.215", "B": " -0.113", "C": " 0.113", "D": " 0.215"},
            "Answer": "C",
            "Explanation": "text",
            "QA Type": "text",
            "Question Type": "text",
            "Level of Difficulty": "text",
            "Knowledge Topics": "text",
            "General Topics": "text",
            "Book Label": "text",
            "Model Answer": "C",
            "Model Reasoning": "text",
            "Feedback": "text"
        }]

        with open(error_log_path, 'w', encoding='utf-8') as f:
            json.dump(errorLOG, f, ensure_ascii=False, indent=4)
        
        print("Successfully created ErrorLOG!")

    # Check if the zero-shot inference file exists. If not, create it.
    if not os.path.exists(zeroshot_path):
        print("ZeroshotFile does not exist!")
        print("Creating ZeroshotFile ...")

        with open(zeroshot_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
            
        print("Successfully created ZeroshotFile!")

    # Check if the few-shot inference file exists. If not, create it.
    if not os.path.exists(fewshot_path):
        print("FewshotshotFile does not exist!")
        print("Creating FewshotshotFile ...")

        with open(fewshot_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)

        print("Successfully created FewshotshotFile!")
    
    return data, errorLOG


def initialize_vector_database(error_log_path):
    """
    Initialize the vector database and load the error log.

    Args:
        error_log_path (str): Path to the error log.

    Returns:
        index: Vector database initialized and loaded with data.
    """
    index = init_faiss()  
    index_faiss(index, error_log_path)  
    return index


def process_questions(LLMmodel, paths):
    """
    Reasoning process: Use the language model to process questions 
    and update zero-shot, few-shot files, and error logs based on results.

    Args:
        LLMmodel (ChatOpenAI): Instance of the loaded language model.
        paths (dict): Dictionary containing all paths, 
                      including dataset, error log, zero-shot, and few-shot file paths.
    """
    
    data, errorLOG = load_data_and_logs(
        paths["dataset_path"], 
        paths["error_log_path"], 
        paths["zeroshot_path"], 
        paths["fewshot_path"]
        )

    
    index = initialize_vector_database(paths["error_log_path"])

    # Filter data marked for processing
    # filtered_data = [item for item in data if item.get("Filter") == 1.0]
    filtered_data = [item for item in data if item.get("Datasplit") == "train"]

    # Initialize model output parser and reasoning chain
    outputParser = MarkdownParser()
    chain = LLMmodel | outputParser

    for question in tqdm(filtered_data, desc="Processing Questions"):
        # First reasoning
        reasoning_first = chain.invoke(inputPrompt(question))
        answer_first = extract_answer(reasoning_first)
        
        
        modelOutput = question
        modelOutput["Model Answer"] = answer_first
        modelOutput["Model Reasoning"] = reasoning_first
        write_output(modelOutput, paths["zeroshot_path"])  # Save to zero-shot inference file
        
        
        if answer_first == question["Answer"]:
            
            write_output(modelOutput, paths["fewshot_path"])
        else:
            
            cos, I = query_embedding_faiss(question, index, k=5)
            
            errorexample = errorLOG[I[0][0]]
            
            # Second reasoning
            reasoning_second = chain.invoke(ICLPrompt(question, errorexample))
            answer_second = extract_answer(reasoning_second)
            

            modelOutput["Model Answer"] = answer_second
            modelOutput["Model Reasoning"] = reasoning_second
            write_output(modelOutput, paths["fewshot_path"])  # Save to few-shot inference file
            

            modelOutput["Model Answer"] = answer_first
            modelOutput["Model Reasoning"] = reasoning_first
            feedback = chain.invoke(FeedbackPrompt(modelOutput))
            modelOutput["Feedback"] = feedback
            write_output(modelOutput, paths["error_log_path"])  # Save to error log
