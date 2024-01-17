from langchain_community.callbacks.manager import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm

llm = OpenAI(
    model_name=os.environ.get("OPENAI_GPT_INFERENCE_VERSION"),
    max_tokens=2500,
    temperature=0.0,
)

template = """{system}

{human}"""

prompt = PromptTemplate(
    template=template,
    input_variables=[
        "system",
        "human"],
)

chain = prompt | llm

json_filename = os.environ.get("DPO_TRANSLATION_FILENAME")

# Load the previous built translations
pdo_translations = pd.read_json(json_filename, orient="records").to_dict()
total_iterations = len(pdo_translations)

# Set max_iterations to None to run the full dataset
max_iterations = 2

# Get the total dataset size
total_pdo_translation_size = len(pdo_translations)

# Covers the inferenced conversations
inferenced_conversations = []

# Saves the total costs
total_costs = []

def run_inference_openai():
    '''
    Run the inference script
    '''
    iterator = tqdm(range(0, total_iterations, 1), total=total_iterations, desc="Start inferencing")
    for i in iterator:
        # Stop if we have reached the total batch itterations
        if (max_iterations is not None) and (i == max_iterations):
            iterator.close()
            break
        translated_entry = pdo_translations[i]
        with get_openai_callback() as cb:
            response = chain.invoke({
                    "system":translated_entry['system'],
                    "human":translated_entry['human']})
            # Calculate single entry cost
            print(f"Batch Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Batch Cost (USD): ${cb.total_cost}")
            total_costs.append(cb.total_cost)

        translated_entry['choosen'] = response['choices'][0]['text']
        inferenced_conversations.append(translated_entry)

def estimate_inference():
    '''
    Check the cost and runtime of the inference
    '''
    # Get the start time
    st = time.time()
    # Run the translation and save the output
    run_inference_openai()
    mean_cost = np.mean(total_costs)
    # Estimate total cost
    print(f"Estimated total cost (USD): ${np.round(total_iterations * mean_cost, 2)}")
    # Get the end time
    et = time.time()
    # Get the execution time
    elapsed_time = et - st
    print(f"Estimated total runtime: {np.round(elapsed_time * total_iterations, 2)} seconds)")