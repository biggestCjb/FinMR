import json
import os
import re
from pathlib import Path
from typing import List
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import Generation
from langchain.tools import Tool



class MarkdownParser(BaseGenerationOutputParser[str]):
    """
    A custom parser that formats the model output for Markdown display
    by replacing LaTeX-style delimiters \[ and \] with $.
    """
    def parse_result(self, result: List[Generation], *, partial: bool = False) -> str:
        """Parse the model output and format it as Markdown.

        Args:
            result: A list of Generations (assumed to contain only one string).
            partial: Whether to allow partial results (for streaming, not used here).

        Returns:
            A Markdown-formatted string with LaTeX-style delimiters replaced.
        """
        # Ensure there's only one generation
        if len(result) != 1:
            raise ValueError("This parser only supports a single generation.")
        
        # Extract the generation content
        generation = result[0]
        if not isinstance(generation.text, str):
            raise ValueError("Expected text output for Markdown formatting.")
        
        # Replace  \\[ and \\] with $ for LaTeX-style display
        formatted_text = generation.text.replace('\\[', '$').replace('\\]', '$').replace('\\(', '$').replace('\\)', '$')
        return formatted_text
    



def extract_answer(text: str) -> str:
    """
    Extract answer options (A, B, C, D) in parentheses in the text.
    Args:
        text (str): A string containing the text of the answer.
        str: Extracted answer options (A, B, C, D), if not found, return "Answer not found".
    Returns:
        Answer
    """
    # Regular expression to find the answer in brackets, e.g., [C]
    match = re.search(r"\【([A-D])\】", text)
    if match:
        return match.group(1)  
    else:
        return "Answer not found"  

# Wrap extract_answer in a LangChain Tool to make it invokable
extract_answer_tool = Tool.from_function(
    func=extract_answer,
    name="Extract Answer Tool",
    description="Extracts the answer option in brackets (e.g., 【C】) from the provided text."
)



def write_output(data, file_path):
    """
    Writes data to the specified file, and appends the data if the file already exists.
    Args:
        data (dict or list): A single piece of data or a list of data to be written.
        file_path (str): The path to the file.
    """
            
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    if isinstance(existing_data, list):
        existing_data.append(data)
    else:
        existing_data = data

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

