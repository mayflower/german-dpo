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

json_filename = './output/dpo_translation.json'
cache_index_filename = './cache/dpo_translation_index.csv'

# Batch size configurations
# Run the translation with max batch size of 2,
# e.g., batch size of 2 times 2 = 4 conversation entries in total
# Set max_batches to None to run the full dataset
batch_size = 2
max_batches = None

# Get the total dataset size and calculate the total batch iterations
total_dataset_size = len(dataset['train']["conversations"])
total_batch_iterations = math.ceil(total_dataset_size/batch_size)

# Covers translations
translated_conversations = []

# Saves the total costs
total_costs = []

def run_batch_translations():
    '''
    Run the translation script
    '''
    # Covers translations
    conv_to_translate = []
    # Check if the batch size is valid
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0")
    # Prepare the translation
    iterator = tqdm(range(0, total_dataset_size, batch_size), total=total_batch_iterations, desc="Start translating")
    for i in iterator:
        # Stop if we have reached the total batch itterations
        if (max_batches is not None) and (i == (batch_size * max_batches)):
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
        # Save the batch output
        save_output(i)
        translated_conversations.clear()
        conv_to_translate.clear()

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

def save_output(index: int = None):
    '''
    Save the output json file
    '''
    # 4 columns, system prompt e.g., “You are a system that can math”, user prompt “What is 5 x 5?”, 
    # Reject result (Mistral), accept result (GPT4).
    # Save JSON-file
    print("Writing to translation file ...")
    def reduce_translated_conversations():
        merged_list = []
        for conv in translated_conversations:
            merged_list.extend(conv)
        return merged_list
    
    pd.DataFrame(reduce_translated_conversations()).to_json(
        json_filename, force_ascii=False, mode='a', orient='records', lines=True)
    print("Done writing to translation file")
    
    # Writing index to file to keep track of progress and resume if necessary
    print("Writing index to file ...")
    if index is not None:
        pd.DataFrame([{
            'current_index':index,
            'max_batches':max_batches,
            'total_batch_iterations':total_batch_iterations}]).to_csv(
            cache_index_filename, encoding='utf-8', mode='w', index=False)
    print("Done writing index to file")

def estimate_metrics():
    '''
    Run the translation and save the output
    '''
    # Get the start time
    st = time.time()
    # Run the translation and save the output
    run_batch_translations()
    mean_batch_cost = np.mean(total_costs)
    # Estimate total cost
    print(f"Estimated total cost (USD): ${np.round(total_batch_iterations * mean_batch_cost, 2)}")
    # Get the end time
    et = time.time()
    # Get the execution time
    elapsed_time = et - st
    print(f"Estimated total runtime: {np.round(elapsed_time * total_batch_iterations, 2)} seconds)")

def run_translations():
    '''
    Run the translation and save the output
    '''
    # Get the start time
    st = time.time()
    # Run the translation and save the output
    run_batch_translations()
    # Get the end time
    et = time.time()
    # Get the execution time
    elapsed_time = et - st
    print(f"Done translation in: {np.round(elapsed_time, 2)} seconds)")
    
