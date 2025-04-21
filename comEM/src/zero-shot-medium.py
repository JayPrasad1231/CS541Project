import pandas as pd
import time
import re
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from collections import defaultdict

# === Load Hermes model ===
model_name = "NousResearch/hermes-1-mistral"
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, batch_size=4)

# === Load data ===
amazon = pd.read_csv("../datasets/Amazon.csv", encoding='unicode_escape')
google = pd.read_csv("../datasets/GoogleProducts.csv", encoding='unicode_escape')
perfect_mapping = pd.read_csv("../datasets/Amzon_GoogleProducts_perfectMapping.csv", encoding='unicode_escape')
ground_truth_matches = set(zip(perfect_mapping['idAmazon'], perfect_mapping['idGoogleBase']))

# === Helpers ===
def to_blocking_text(row):
    return f"{row.get('title', row.get('name', ''))} {row.get('brand', '')} {row.get('manufacturer', '')}"
amazon['blocking_key'] = amazon.apply(to_blocking_text, axis=1)
google['blocking_key'] = google.apply(to_blocking_text, axis=1)

def block_records(amazon_record, google_df, threshold=0.3):
    amazon_tokens = set(str(amazon_record['blocking_key']).lower().split())
    def compute_overlap(row):
        google_tokens = set(str(row['blocking_key']).lower().split())
        return len(amazon_tokens & google_tokens) / len(amazon_tokens | google_tokens) if google_tokens else 0
    google_df['similarity'] = google_df.apply(compute_overlap, axis=1)
    return google_df[google_df['similarity'] >= threshold].drop(columns=['similarity'])

def format_record(row):
    return f"{row.get('title', row.get('name', ''))} {row.get('brand', '')} {row.get('manufacturer', '')}"

# === Zero-Shot Matching ===
def zero_shot_medium_matching():
    prompts, meta = [], []
    for idx, a_row in tqdm(amazon.iterrows(), total=len(amazon)):
        candidates = block_records(a_row, google)
        anchor_text = format_record(a_row)
        for _, g_row in candidates.iterrows():
            candidate_text = format_record(g_row)
            prompt = f"""
Are the following two product records referring to the same entity? Answer with only 'Yes' or 'No'.

Record 1: {anchor_text}
Record 2: {candidate_text}

Answer:"""
            prompts.append(prompt)
            meta.append((a_row['id'], g_row['id'], prompt))

    sequences = llm(prompts, max_new_tokens=10, do_sample=False)
    predictions = []
    for (a_id, g_id, prompt), result in zip(meta, sequences):
        output = result['generated_text'].replace(prompt, '').strip().lower()
        answer = output.split()[0] if output else "no"
        predictions.append({
            "idAmazon": a_id,
            "idGoogle": g_id,
            "prediction": "Yes" if answer.startswith("yes") else "No",
            "duration": 0,
            "num_tokens": len(prompt.split()) + len(output.split())
        })
    return predictions

# === Zero-Shot Selecting ===
def zero_shot_medium_selecting():
    prompts, meta = [], []
    for idx, a_row in tqdm(amazon.iterrows(), total=len(amazon)):
        candidates = block_records(a_row, google)
        if candidates.empty:
            continue
        anchor_text = format_record(a_row)
        candidate_texts = [format_record(g_row) for _, g_row in candidates.iterrows()]
        candidate_ids = [g_row['id'] for _, g_row in candidates.iterrows()]
        prompt = f"""
You are given a product description and a list of candidate products.
Choose the number of the candidate that matches best. If none match, respond with '0'.

Product: {anchor_text}

Candidates:
""" + "\n".join([f"{i+1}. {text}" for i, text in enumerate(candidate_texts)]) + "\n0. None of the above\n\nAnswer:"
        prompts.append(prompt)
        meta.append((a_row['id'], candidate_ids, prompt))

    sequences = llm(prompts, max_new_tokens=10, do_sample=False)
    predictions = []
    for (a_id, candidate_ids, prompt), result in zip(meta, sequences):
        output = result['generated_text'].replace(prompt, '').strip()
        match = re.search(r"\b(\d+)\b", output)
        selected_index = int(match.group(1)) if match else 0
        if 1 <= selected_index <= len(candidate_ids):
            selected_id = candidate_ids[selected_index - 1]
            predictions.append({"idAmazon": a_id, "idGoogle": selected_id, "prediction": "Yes", "duration": 0, "num_tokens": len(prompt.split()) + len(output.split())})
        else:
            predictions.append({"idAmazon": a_id, "idGoogle": None, "prediction": "No Match", "duration": 0, "num_tokens": len(prompt.split()) + len(output.split())})
    return predictions

# === Zero-Shot Comparing ===
def zero_shot_medium_comparing():
    prompts, meta = [], []
    for idx, a_row in tqdm(amazon.iterrows(), total=len(amazon)):
        candidates = block_records(a_row, google)
        if len(candidates) < 2:
            continue
        anchor_text = format_record(a_row)
        candidate_list = list(candidates.iterrows())
        candidate_ids = [row['id'] for _, row in candidate_list]
        candidate_texts = [format_record(row) for _, row in candidate_list]
        for i in range(len(candidate_texts)):
            for j in range(len(candidate_texts)):
                if i == j:
                    continue
                prompt = f"""
Compare the two candidates for the anchor product.
Anchor: {anchor_text}
Candidate A: {candidate_texts[i]}
Candidate B: {candidate_texts[j]}

Which one is a better match? Answer 'A' or 'B'.
"""
                prompts.append(prompt)
                meta.append((a_row['id'], candidate_ids[i], prompt))

    sequences = llm(prompts, max_new_tokens=10, do_sample=False)
    vote_map = defaultdict(lambda: defaultdict(int))
    for (a_id, cand_id, prompt), result in zip(meta, sequences):
        output = result['generated_text'].replace(prompt, '').strip().lower()
        if output.startswith("a"):
            vote_map[a_id][cand_id] += 1

    predictions = []
    for anchor_id, cand_votes in vote_map.items():
        best_id = max(cand_votes.items(), key=lambda x: x[1])[0]
        predictions.append({"idAmazon": anchor_id, "idGoogle": best_id, "prediction": "Yes", "duration": 0, "num_tokens": 0})
    return predictions

# === Evaluation ===
def run_and_evaluate(strategy_fn, name="Strategy"):
    print(f"\n=== Running {name} ===")
    predictions = strategy_fn()
    pred_df = pd.DataFrame(predictions)
    matched_pairs = set(zip(pred_df[pred_df['prediction'].str.lower() == "yes"]['idAmazon'], pred_df[pred_df['prediction'].str.lower() == "yes"]['idGoogle']))
    y_true, y_pred = [], []
    for (a_id, g_id) in matched_pairs.union(ground_truth_matches):
        y_true.append((a_id, g_id) in ground_truth_matches)
        y_pred.append((a_id, g_id) in matched_pairs)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f"{name} → Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    total_time = sum(p['duration'] for p in predictions)
    total_tokens = sum(p['num_tokens'] for p in predictions)
    print(f"Total Time: {total_time:.2f} seconds\nTotal Tokens: {total_tokens}")
    return pred_df

# === Run All ===
run_and_evaluate(zero_shot_medium_matching, "Zero-Shot Matching")
run_and_evaluate(zero_shot_medium_selecting, "Zero-Shot Selecting")
run_and_evaluate(zero_shot_medium_comparing, "Zero-Shot Comparing")