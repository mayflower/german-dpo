from transformers import T5ForConditionalGeneration, T5Tokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import json
import torch

model_name = 'jbochi/madlad400-3b-mt'
model = T5ForConditionalGeneration.from_pretrained(
    model_name, 
    device_map="cuda:0", 
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ),
    load_in_4bit=True)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def translate(sentence: str) -> str:
    texts = [sentence]
    with torch.no_grad():
        madlad_texts = [f'<2de> ' + text.replace("\n", " ") for text in texts]
        encoded_batch = tokenizer(madlad_texts, return_tensors="pt", padding=True).to("cuda:0")
        outputs = model.generate(input_ids=encoded_batch['input_ids'], max_new_tokens=2048)
        translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return translated_texts[0]


dataset = load_dataset("Open-Orca/SlimOrca-Dedup",
                       cache_dir="./cache")

new_format = []
for entry in tqdm(dataset['train']["conversations"][:1], total=len(dataset['train']["conversations"][:1]), desc="Translating",):
    for conv in entry:
        new_entry = {
            "prompt": translate(conv["value"]),
        }
        new_format.append(new_entry)

with open("translated_prompts.json", 'w', encoding='utf-8') as file:
        json.dump(new_format, file, indent=4, ensure_ascii=False)


