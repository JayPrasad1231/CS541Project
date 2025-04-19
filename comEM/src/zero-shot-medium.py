import pandas as pd
import time
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# === Load your local model ===
model_name = "google/flan-t5-base"  
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")
llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)

amazon = pd.read_csv("../datasets/Amazon.csv", encoding='unicode_escape')
google = pd.read_csv("../datasets/GoogleProducts.csv", encoding='unicode_escape')
perfect_mapping = pd.read_csv("../datasets/Amzon_GoogleProducts_perfectMapping.csv", encoding='unicode_escape')

ground_truth_matches = set(zip(perfect_mapping['idAmazon'], perfect_mapping['idGoogleBase']))

def to_blocking_text(row):
    return f"{row.get('title', row.get('name', ''))} {row.get('brand', '')} {row.get('manufacturer', '')}"

amazon['blocking_key'] = amazon.apply(to_blocking_text, axis=1)
google['blocking_key'] = google.apply(to_blocking_text, axis=1)

def block_records(amazon_record, google_df, threshold=0.3):
    amazon_tokens = set(str(amazon_record['blocking_key']).lower().split())

    def compute_overlap(row):
        google_tokens = set(str(row['blocking_key']).lower().split())
        if not google_tokens:
            return 0
        return len(amazon_tokens & google_tokens) / len(amazon_tokens | google_tokens)

    google_df['similarity'] = google_df.apply(compute_overlap, axis=1)
    candidates = google_df[google_df['similarity'] >= threshold].drop(columns=['similarity'])
    return candidates

def format_record(row):
    return f"{row.get('title', row.get('name', ''))} {row.get('brand', '')} {row.get('manufacturer', '')}"

def query_llm(anchor, candidate):
    prompt = f"""
    Are the following two product records referring to the same entity? Answer with only 'Yes' or 'No'.

    Record 1: {anchor}
    Record 2: {candidate}
    """
    start = time.time()
    output = llm(prompt, max_new_tokens=5, truncation=True)[0]['generated_text']
    duration = time.time() - start
    return output.strip(), len(prompt.split()), duration