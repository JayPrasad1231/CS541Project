# Chain of Verification (CoV) - Selecting Strategy Only
# Requires shared context: model, amazon/google tables, blocking, etc.

import pandas as pd
import time
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# === Load your local model ===
model_name = "google/flan-t5-base"  # Or larger if your GPU allows
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


# === Chain-of-Verification Selecting ===
def cov_selecting():
    predictions = []

    for idx, a_row in tqdm(amazon.iterrows(), total=len(amazon)):
        candidates = block_records(a_row, google)
        if candidates.empty:
            continue

        anchor_text = format_record(a_row)
        candidate_texts = [format_record(g_row) for _, g_row in candidates.iterrows()]
        candidate_ids = [g_row['id'] for _, g_row in candidates.iterrows()]

        # Step 1: Verification reasoning
        verification_prompt = f"""
        Analyze the following product and candidate list. For each candidate, reason about whether it refers to the same entity as the anchor.
        Think about title, brand, and manufacturer. List reasoning for each, then make a final decision.

        Anchor Product: {anchor_text}

        Candidates:
        """ + "\n".join([f"{i+1}. {text}" for i, text in enumerate(candidate_texts)]) + "\n\nRespond with reasoning for each and finally: 'Answer: X' where X is the number or 0 if none match."

        start = time.time()
        response = llm(verification_prompt, max_new_tokens=100, truncation=True)[0]['generated_text']
        duration = time.time() - start
        num_tokens = len(verification_prompt.split()) + len(response.split())

        # Step 2: Extract decision
        try:
            lines = response.strip().split("\n")
            answer_line = next(line for line in lines if "answer:" in line.lower())
            selected_index = int(''.join(filter(str.isdigit, answer_line.strip())))
        except:
            selected_index = 0

        if 1 <= selected_index <= len(candidate_ids):
            selected_id = candidate_ids[selected_index - 1]
            predictions.append({
                "idAmazon": a_row['id'],
                "idGoogle": selected_id,
                "prediction": "Yes",
                "duration": duration,
                "num_tokens": num_tokens
            })
        else:
            predictions.append({
                "idAmazon": a_row['id'],
                "idGoogle": None,
                "prediction": "No Match",
                "duration": duration,
                "num_tokens": num_tokens
            })

    return predictions

# === Run and Evaluate ===
def run_and_evaluate(strategy_fn, name="Strategy"):
    print(f"\n=== Running {name} ===")
    predictions = strategy_fn()
    pred_df = pd.DataFrame(predictions)

    matched_pairs = set(
        zip(
            pred_df[pred_df['prediction'].str.lower() == "yes"]['idAmazon'],
            pred_df[pred_df['prediction'].str.lower() == "yes"]['idGoogle']
        )
    )

    y_true = []
    y_pred = []
    for (a_id, g_id) in matched_pairs.union(ground_truth_matches):
        y_true.append((a_id, g_id) in ground_truth_matches)
        y_pred.append((a_id, g_id) in matched_pairs)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    print(f"{name} → Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    total_time = sum(p['duration'] for p in predictions)
    total_tokens = sum(p['num_tokens'] for p in predictions)
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Total Tokens: {total_tokens}")

    return pred_df

run_and_evaluate(cov_selecting, "Chain-of-Verification")