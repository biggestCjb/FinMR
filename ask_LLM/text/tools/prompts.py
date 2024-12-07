import os
import json
import base64

from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage

from config import load_env_vars


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
        content="You are a financial expert. You will be given questions and options, possibly with context information and images caption. Please answer the question."
    )

    human_message=HumanMessage(content=[])

    if len(question["Share Context"]) != 0:
        human_message.content.append({"type": "text", "text": "Context: " + question["Share Context"]})

    if len(question["shared_description"])!= 0:
        human_message.content.append({"type": "text", "type": "image caption of context: "+question["shared_description"]})

    human_message.content.append({"type": "text", "text": "Question: "+ question["Question Text"]})

    if len(question["description"]) != 0:
        human_message.content.append({"type": "text", "type": "image caption: "+question["description"]})

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
            content="""You are a financial expert. You will be given questions and options, possibly with context information and images. Also, you will be given wrong reasoning steps and correct reasoning hints.You are supposed to give feedback.""")

    human_message=HumanMessage(content=[])

    if len(question["Share Context"]) != 0:
        human_message.content.append({"type": "text", "text": "Context: " + question["Share Context"]})

    if len(question["shared_description"])!= 0:
        human_message.content.append({"type": "text", "type": "image caption of context: "+question["shared_description"]})


    human_message.content.append({"type": "text", "text": "Question: "+ question["Question Text"]})

    if len(question["description"]) != 0:
        human_message.content.append({"type": "text", "type": "image caption: "+question["description"]})

    human_message.content.append({"type": "text", "text": "Options: " + str(question["Options"])})
    human_message.content.append({"type": "text", "text": "Wrong Reasoning Steps: " + question["Model Reasoning"]})
    human_message.content.append({"type": "text", "text": "Wrong Answer: " + question["Model Answer"]})
    human_message.content.append({"type": "text", "text": "Correct Reasoning Steps: " + question["Explanation"]})
    human_message.content.append({"type": "text", "text": "Correct Answer: " + question["Answer"]})

    human_message.content.append({"type": "text", "text": """ Please give the feedback in Markdown format. 1. Please output correct reasoning steps according to hints. 2. compare the correct reasoning step with the model's wrong reasoning step, and point out the difference. 3. summarize the hint for future simalar questions."""})

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
       
    system_message = SystemMessage(
        content="You are a financial expert. You will be given previous learning document including questions and options, possibly with context information and images. Please answer the current question."
    )

    human_message=HumanMessage(content=[])
    human_message.content.append({"type": "text", "text": "Previous Learning Document: "})
    if len(example["Share Context"]) != 0:
        human_message.content.append({"type": "text", "text": "Context: " + example["Share Context"]})

    if len(example["shared_description"])!= 0:
        human_message.content.append({"type": "text", "type": "image caption of context: "+example["shared_description"]})

    human_message.content.append({"type": "text", "text": "Question: "+ example["Question Text"]})

    if len(example["description"]) != 0:
        human_message.content.append({"type": "text", "type": "image caption: "+example["description"]})

    human_message.content.append({"type": "text", "text": "Options: " + str(example["Options"])})
    human_message.content.append({"type": "text", "text": "Wrong Reasoning Steps: " + example["Model Reasoning"]})
    human_message.content.append({"type": "text", "text": "Feedback: " + example["Feedback"]})


    human_message.content.append({"type": "text", "text": "Current Question is as follows: "})
    if len(question["Share Context"]) != 0:
        human_message.content.append({"type": "text", "text": "Context: " + question["Share Context"]})

    if len(question["shared_description"])!= 0:
        human_message.content.append({"type": "text", "type": "image caption of context: "+question["shared_description"]})

    human_message.content.append({"type": "text", "text": "Question: "+ question["Question Text"]})

    if len(question["description"]) != 0:
        human_message.content.append({"type": "text", "type": "image caption: "+question["description"]})

    human_message.content.append({"type": "text", "text": "Options: " + str(question["Options"])})

    human_message.content.append({"type": "text", "text": "Let's think step by step. The output reasoning steps are in Markdown format. Finally, must put the correct option (A, B, C, or D) in【 】. e.g.Therefore, the correct option is 【B】."})

    response = [system_message, human_message]
    return response

