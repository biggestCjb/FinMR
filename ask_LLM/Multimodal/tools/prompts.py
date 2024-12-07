import os
import json
import base64

from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage

from config import load_env_vars

# Define the size of the image that meets the standard
ACCEPTED_SIZES = [
    (1092, 1092),  # 1:1
    (951, 1268),   # 3:4
    (896, 1344),   # 2:3
    (819, 1456),   # 9:16
    (784, 1568)    # 1:2
]

def find_best_fit_size(width, height):
    """
    Based on the width and height of the input image, find the closest standard size.
    Args:
        width (int): The width of the image.
        height (int): The height of the image.    
    Returns:
        tuple: The nearest standard size (width, height).
    """
    aspect_ratio = width / height
    best_fit = min(ACCEPTED_SIZES, key=lambda size: abs(aspect_ratio - (size[0] / size[1])))
    return best_fit

def resize_image(image_path, max_tokens=1600):
    """
    Resize the image so that it fits the standard size and meets the token limit.
    Args:
        max_tokens (int): The maximum number of tokens allowed.
        image_path (str): The path to the image.

    Returns:
        PIL. Image: The adjusted image.
    
    """
    with Image.open(image_path) as img:

        img = img.convert("RGB")

        width, height = img.size
        while True:
            
            target_width, target_height = find_best_fit_size(width, height)
            
            # Calculate the current number of tokens
            current_tokens = (width * height) / 750
            
            
            if (width <= target_width and height <= target_height) and current_tokens <= max_tokens:
                break  

            # Resize the image proportionally
            scaling_factor_width = target_width / width
            scaling_factor_height = target_height / height
            scaling_factor_tokens = (max_tokens * 750 / (width * height)) ** 0.5
            
            scaling_factor = min(scaling_factor_width, scaling_factor_height, scaling_factor_tokens)
            width = int(width * scaling_factor)
            height = int(height * scaling_factor)
            img = img.resize((width, height))
        
        return img

def image_to_base64(image_path, max_tokens=1600):
    """
    Resize the picture and convert it to Base64 format.
    Args:
        str: Base64 encoding of the adjusted image.
        image_path (str): The path to the image.
    Returns:
        max_tokens (int): The maximum number of tokens allowed.
    """
    resized_image = resize_image(image_path, max_tokens)
    temp_path = load_env_vars().get("temp_path")
    resized_image.save(temp_path, format="PNG")

    with open(temp_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Delete temporary files
    os.remove(temp_path)
    return base64_data


def inputPrompt(question):
    """
    Build an input prompt including the question and options.

    Args:
        question (dict): Data containing the question, context, and image paths.

    Returns:
        list: A list of system message and user message.
    """       
    # Build the system message
    system_message = SystemMessage(
        content="You are a financial expert. You will be given questions and options, possibly with context information and images. Please answer the question."
    )

    # Build the user message
    human_message = HumanMessage(content=[])

    if len(question["Share Context"]) != 0:
        human_message.content.append({"type": "text", "text": "Context: " + question["Share Context"]})

    if len(question["Share Image"]) != 0:
        for path in question["Share Image"]:
            image_url = load_env_vars().get("root_dir") + path
            image_data = image_to_base64(image_url)
            human_message.content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}})

    human_message.content.append({"type": "text", "text": "Question: " + question["Question Text"]})

    if len(question["Image"]) != 0:
        image_url = load_env_vars().get("root_dir") + question["Image"]
        image_data = image_to_base64(image_url)
        human_message.content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}})

    human_message.content.append({"type": "text", "text": "Options: " + str(question["Options"])})

    human_message.content.append({"type": "text", "text": "Let's think step by step. The output reasoning steps are in Markdown format. Finally, must put the correct option (A, B, C, or D) in【 】. e.g.Therefore, the correct option is 【B】."})

    response = [system_message, human_message]
    return response


def FeedbackPrompt(question):
    """
    Build a feedback prompt to generate feedback information.

    Args:
        question (dict): Data containing incorrect reasoning and correct reasoning.

    Returns:
        list: A list of system message and user message.
    """       
    system_message = SystemMessage(
            content="""You are a financial expert. You will be given questions and options, possibly with context information and images. Also, you will be given wrong reasoning steps and correct reasoning hints. You are supposed to give feedback."""
    )

    # Build the user message
    human_message = HumanMessage(content=[])

    if len(question["Share Context"]) != 0:
        human_message.content.append({"type": "text", "text": "Context: " + question["Share Context"]})

    if len(question["Share Image"]) != 0:
        for path in question["Share Image"]:
            image_url = load_env_vars().get("root_dir") + path
            image_data = image_to_base64(image_url)
            human_message.content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}})

    human_message.content.append({"type": "text", "text": "Question: " + question["Question Text"]})

    if len(question["Image"]) != 0:
        image_url = load_env_vars().get("root_dir") + question["Image"]
        image_data = image_to_base64(image_url)
        human_message.content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}})

    human_message.content.append({"type": "text", "text": "Options: " + str(question["Options"])})
    human_message.content.append({"type": "text", "text": "Wrong Reasoning Steps: " + question["Model Reasoning"]})
    human_message.content.append({"type": "text", "text": "Wrong Answer: " + question["Model Answer"]})
    human_message.content.append({"type": "text", "text": "Correct Reasoning Steps: " + question["Explanation"]})
    human_message.content.append({"type": "text", "text": "Correct Answer: " + question["Answer"]})

    human_message.content.append({"type": "text", "text": """ Please give the feedback in Markdown format. 1. Please output correct reasoning steps according to hints. 2. Compare the correct reasoning step with the model's wrong reasoning step, and point out the difference. 3. Summarize the hint for future similar questions."""})

    response = [system_message, human_message]
    return response


def ICLPrompt(question, example):
    """
    Build a few-shot learning prompt.

    Args:
        question (dict): Current question data.
        example (dict): Learning example for the prompt.

    Returns:
        list: A list of system message and user message.
    """       
    # Build the system message
    system_message = SystemMessage(
        content="You are a financial expert. You will be given previous learning document including questions and options, possibly with context information and images. Please answer the current question."
    )

    # Build the user message
    human_message = HumanMessage(content=[])
    human_message.content.append({"type": "text", "text": "Previous Learning Document: "})
    if len(example["Share Context"]) != 0:
        human_message.content.append({"type": "text", "text": "Context: " + example["Share Context"]})

    if len(example["Share Image"]) != 0:
        for path in example["Share Image"]:
            image_url = load_env_vars().get("root_dir") + path
            image_data = image_to_base64(image_url)
            human_message.content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}})

    human_message.content.append({"type": "text", "text": "Question: " + example["Question Text"]})

    if len(example["Image"]) != 0:
        image_url = load_env_vars().get("root_dir") + example["Image"]
        image_data = image_to_base64(image_url)
        human_message.content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}})

    human_message.content.append({"type": "text", "text": "Options: " + str(example["Options"])})
    human_message.content.append({"type": "text", "text": "Wrong Reasoning Steps: " + example["Model Reasoning"]})
    human_message.content.append({"type": "text", "text": "Feedback: " + example["Feedback"]})


    human_message.content.append({"type": "text", "text": "Current Question is as follows: "})
    if len(question["Share Context"]) != 0:
        human_message.content.append({"type": "text", "text": "Context: " + question["Share Context"]})

    if len(question["Share Image"]) != 0:
        for path in question["Share Image"]:
            image_url = load_env_vars().get("root_dir") + path
            image_data = image_to_base64(image_url)
            human_message.content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}})

    human_message.content.append({"type": "text", "text": "Question: " + question["Question Text"]})

    if len(question["Image"]) != 0:
        image_url = load_env_vars().get("root_dir") + question["Image"]
        image_data = image_to_base64(image_url)
        human_message.content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}})

    human_message.content.append({"type": "text", "text": "Options: " + str(question["Options"])})

    human_message.content.append({"type": "text", "text": "Let's think step by step. The output reasoning steps are in Markdown format. Finally, must put the correct option (A, B, C, or D) in【 】. e.g.Therefore, the correct option is 【B】."})

    response = [system_message, human_message]
    return response
