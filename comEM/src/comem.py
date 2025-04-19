# comEM Pipeline: Blocking → Matching/Comparing (Small LLM) → Selecting (Gemini with CoT)
import pandas as pd
import time
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# === Config ===
model_name = "google/flan-t5-base"  # small LLM for matching/comparing
print("Loading small model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
small_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")
small_llm = pipeline("text2text-generation", model=small_model, tokenizer=tokenizer, device=0)

# Gemini setup
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)  # Replace with your actual key
gemini = genai.GenerativeModel("gemini-2.0-flash-lite-001")

amazon = pd.read_csv("../datasets/Amazon.csv", encoding='unicode_escape')
google = pd.read_csv("../datasets/GoogleProducts.csv", encoding='unicode_escape')
perfect_mapping = pd.read_csv("../datasets/Amzon_GoogleProducts_perfectMapping.csv", encoding='unicode_escape')

ground_truth_matches = set(zip(perfect_mapping['idAmazon'], perfect_mapping['idGoogleBase']))

def format_record(row):
    return f"{row.get('title', row.get('name', ''))} {row.get('brand', '')} {row.get('manufacturer', '')}"

def count_tokens(text):
    return gemini.count_tokens(text).total_tokens

# === Blocking ===
def to_blocking_text(row):
    return f"{row.get('title', row.get('name', ''))} {row.get('brand', '')} {row.get('manufacturer', '')}"

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

# === Step 1: Matching or Comparing (Small LLM with CoT) ===
def small_llm_matching(anchor_text, candidate_text):
    prompt = f"""
    Do the following two product records refer to the same entity?
    Think step-by-step about the title, brand, and manufacturer.

    Product A: {anchor_text}
    Product B: {candidate_text}

    Final answer: Yes or No.
    """
    output = small_llm(prompt, max_new_tokens=20, truncation=True)[0]['generated_text']
    return output.strip(), prompt

# === Step 2: Selecting (Gemini with CoT) ===
def gemini_select(anchor_text, top_candidates):
    candidate_texts = [format_record(row) for row in top_candidates]
    candidate_ids = [row['id'] for row in top_candidates]

    prompt = f"""
    You are given a product and a list of candidates. Analyze each candidate step-by-step and choose the best match.

    Product: {anchor_text}

    Candidates:
    """ + "\n".join([f"{i+1}. {text}" for i, text in enumerate(candidate_texts)]) + "\n\nRespond with reasoning and end with 'Answer: X' where X is the best match or 0 for no match."

    start = time.time()
    response = gemini.generate_content(prompt)
    duration = time.time() - start
    num_tokens = count_tokens(prompt + response.text)

    try:
        answer_line = next(line for line in response.text.split("\n") if "answer:" in line.lower())
        selected_index = int(''.join(filter(str.isdigit, answer_line.strip())))
    except:
        selected_index = 0

    if 1 <= selected_index <= len(candidate_ids):
        return candidate_ids[selected_index - 1], duration, num_tokens
    return None, duration, num_tokens

# === Full comEM Pipeline ===
def run_comem_pipeline(amazon, google, ground_truth_matches):
    amazon['blocking_key'] = amazon.apply(to_blocking_text, axis=1)
    google['blocking_key'] = google.apply(to_blocking_text, axis=1)

    predictions = []

    for _, a_row in tqdm(amazon.iterrows(), total=len(amazon)):
        candidates = block_records(a_row, google)
        if candidates.empty:
            continue

        anchor_text = format_record(a_row)
        top_candidates = []

        for _, g_row in candidates.iterrows():
            candidate_text = format_record(g_row)
            pred, prompt = small_llm_matching(anchor_text, candidate_text)
            if "yes" in pred.lower():
                top_candidates.append(g_row)

        if not top_candidates:
            predictions.append({
                "idAmazon": a_row['id'],
                "idGoogle": None,
                "prediction": "No Match",
                "duration": 0,
                "num_tokens": 0
            })
            continue

        selected_id, duration, tokens = gemini_select(anchor_text, top_candidates)

        predictions.append({
            "idAmazon": a_row['id'],
            "idGoogle": selected_id,
            "prediction": "Yes" if selected_id else "No Match",
            "duration": duration,
            "num_tokens": tokens
        })

    return pd.DataFrame(predictions)

# === Evaluation ===
def evaluate_predictions(pred_df, ground_truth_matches, name="comEM"):
    matched_pairs = set(
        zip(
            pred_df[pred_df['prediction'] == "Yes"]['idAmazon'],
            pred_df[pred_df['prediction'] == "Yes"]['idGoogle']
        )
    )

    y_true = []
    y_pred = []
    for (a_id, g_id) in matched_pairs.union(ground_truth_matches):
        y_true.append((a_id, g_id) in ground_truth_matches)
        y_pred.append((a_id, g_id) in matched_pairs)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    print(f"\n{name} → Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Total Time: {pred_df['duration'].sum():.2f} seconds")
    print(f"Total Tokens: {pred_df['num_tokens'].sum()}")
    return pred_df

evaluate_predictions(run_comem_pipeline(amazon, google, ground_truth_matches), ground_truth_matches)