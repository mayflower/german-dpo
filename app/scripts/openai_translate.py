from datasets import load_dataset
import json
from langchain_community.callbacks.manager import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import math
import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm

llm = OpenAI(
    model_name=os.environ.get("OPENAI_GPT_VERSION"),
    max_tokens=2500,
    temperature=0.0,
)

template = """Pretend to be a professional translation assistant.
Given the english conversations below as JSON array of objects, create a valid JSON array of objects containing the translated informal german texts.
Each child object has a property named 'system' and a property named 'human'.
The english conversations: {text}
The german translations:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["text"],
)

chain = prompt | llm

# 364k dataset length
dataset = load_dataset("Open-Orca/SlimOrca-Dedup",
                        cache_dir="./cache")
total_dataset_size = len(dataset['train']["conversations"])

# Batch size configurations
batch_size = 5

# Covers translations
translated_conversations = []

# Define the total costs
total_costs = []

def run_batch_translations(total_batches: int = None):
    '''
    Run the translation script
    '''
    # Covers translations
    conv_to_translate = []
    # Check if the batch size is valid
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0")
    # Prepare the translation
    iterator = tqdm(range(0, total_dataset_size, batch_size), total=math.ceil(total_dataset_size/batch_size), desc="Start translating")
    for i in iterator:
        # Stop if we have reached the total batch itterations
        if (total_batches is not None) and (i == (batch_size * total_batches)):
            iterator.close()
            break
        conv_entries = dataset['train']["conversations"][i:i+batch_size]
        # Prepare the batch
        for entry in conv_entries:
            system_conv = next((conv for conv in entry if conv["from"] == "system"), None)
            human_conv = next((conv for conv in entry if conv["from"] == "human"), None)
            if system_conv and human_conv:
                conv_to_translate.append({
                        "system": system_conv['value'],
                        "human": human_conv['value']
                })
        # Translate the batch
        translated_conversations.append(
            translate_openai(conv_to_translate))
        conv_to_translate = []

def translate_openai(conv_to_translate: list[str]) -> list[str]:
    """
    A translation function that translates a list of inputs into German using the OpenAI API.
    """
    # Prepare the translation string for OpenAI
    text = json.dumps(conv_to_translate)
    result = ''
    with get_openai_callback() as cb:
        response = chain.invoke({"text":text})
        # Calculate single entry cost
        print(f"Batch Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Batch Cost (USD): ${cb.total_cost}")
        total_costs.append(cb.total_cost)
    try:
        # Convert the response back to a JSON object
        result = json.loads(response)
    except:
        print("Invalid JSON")
    if (len(result) != len(conv_to_translate)):
        print("Wrong number of result lines: ", len(conv_to_translate), " (english) and ", len(result), " (german):\n", prompt.format(text=text))

    return result

def save_output():
    '''
    Save the output json file
    '''
    # 4 columns, system prompt e.g., “You are a system that can math”, user prompt “What is 5 x 5?”, 
    # Reject result (Mistral), accept result (GPT4).
    # Save JSON-file
    print("Writing to translation file...")
    json_filename = './output/dpo_translation.json'
    pd.DataFrame(translated_conversations).to_json(json_filename, index=False, encoding='utf-8', ensure_ascii=False)
    print(f"Done translation. Stored results into file: {json_filename}")

def estimate_metrics():
    '''
    Run the translation and save the output
    '''
    # Get the start time
    st = time.time()
    # Run the translation with total batch size of 5, e.g., batch size of 2 times 5 = 10 conversation entries
    run_batch_translations(total_batches = 2)
    total_batch_iterations = np.round(math.ceil(total_dataset_size / batch_size))
    mean_batch_cost = np.mean(total_costs)
    # Estimate total cost
    print(f"Estimated Total Cost (USD): ${np.round(total_batch_iterations * mean_batch_cost, 2)}")
    # Get the end time
    et = time.time()
    # Get the execution time
    elapsed_time = et - st
    print(f"Estimated Total Runtime: {np.round(elapsed_time * total_batch_iterations, 2)} seconds)")

def run_translations():
    '''
    Run the translation and save the output
    '''
    # Get the start time
    st = time.time()
    # Run the translation and save the output
    run_batch_translations(total_batches=2)
    save_output()
    # Get the end time
    et = time.time()
    # Get the execution time
    elapsed_time = et - st
    print(f"Total Runtime: {np.round(elapsed_time, 2)} seconds)")
