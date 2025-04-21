# Chain of Verification (CoV) - Using Hermes Mistral for Selecting Only
import pandas as pd
import time
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# === Load Hermes-1 Mistral ===
model_name = "NousResearch/hermes-1-mistral"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").to("cuda")
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# === Load Data ===
amazon = pd.read_csv("../datasets/Amazon.csv", encoding='unicode_escape')
google = pd.read_csv("../datasets/GoogleProducts.csv", encoding='unicode_escape')
perfect_mapping = pd.read_csv("../datasets/Amzon_GoogleProducts_perfectMapping.csv", encoding='unicode_escape')
ground_truth_matches = set(zip(perfect_mapping['idAmazon'], perfect_mapping['idGoogleBase']))

# === Helper Functions ===
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

        # Step 1: Verification reasoning prompt
        verification_prompt = f"""Analyze the following product and candidate list. For each candidate, reason about whether it refers to the same entity as the anchor.
Think about title, brand, and manufacturer. List reasoning for each, then make a final decision.

Anchor Product: {anchor_text}

Candidates:
""" + "\n".join([f"{i+1}. {text}" for i, text in enumerate(candidate_texts)]) + "\n\nRespond with reasoning for each and finally: 'Answer: X' where X is the number or 0 if none match."

        start = time.time()
        output = llm(verification_prompt, max_new_tokens=300, truncation=True)[0]['generated_text']
        duration = time.time() - start
        num_tokens = len(verification_prompt.split()) + len(output.split())

        # Step 2: Extract decision
        try:
            lines = output.strip().split("\n")
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
    print(f"Total Time: {sum(p['duration'] for p in predictions):.2f} seconds")
    print(f"Total Tokens: {sum(p['num_tokens'] for p in predictions)}")

    return pred_df

# === Execute ===
run_and_evaluate(cov_selecting, "Chain-of-Verification (Hermes)")
