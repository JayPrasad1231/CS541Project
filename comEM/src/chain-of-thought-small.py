# Chain-of-Thought Prompting Versions of Matching, Selecting, and Comparing
# Assumes shared setup (model, data loading, blocking, etc.) is already handled.

import pandas as pd
import time
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# === Load your local model ===
model_name = "declare-lab/flan-alpaca-base"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer, batch_size = 16, device=0)

amazon = pd.read_csv("Amazon.csv", encoding='unicode_escape')
google = pd.read_csv("GoogleProducts.csv", encoding='unicode_escape')
perfect_mapping = pd.read_csv("Amzon_GoogleProducts_perfectMapping.csv", encoding='unicode_escape')

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

# === Chain-of-Thought Matching ===
def cot_matching():
    prompts = []
    meta = []

    for idx, a_row in tqdm(amazon.iterrows(), total=len(amazon)):
        candidates = block_records(a_row, google)
        anchor_text = format_record(a_row)

        for _, g_row in candidates.iterrows():
            candidate_text = format_record(g_row)
            prompt = f"""
            Determine if these two products refer to the same entity.
            Compare title, brand, and manufacturer.

            Product A: {anchor_text}
            Product B: {candidate_text}

            Respond with one word only: Yes or No.
            """
            prompts.append(prompt)
            meta.append((a_row['id'], g_row['id']))

    sequences = llm(prompts, max_new_tokens=5, batch_size=16)

    predictions = []
    for (a_id, g_id), result, prompt in zip(meta, sequences, prompts):
        output = result['generated_text'].strip().lower()
        if output not in ["yes", "no"]:
            print(f"Unexpected output: {output}")
            continue
        prediction = "Yes" if output.startswith("yes") else "No"
        predictions.append({
            "idAmazon": a_id,
            "idGoogle": g_id,
            "prediction": prediction,
            "duration": 0,  # optional
            "num_tokens": len(prompt.split()) + len(output.split())
        })

    return predictions


# === Chain-of-Thought Selecting ===
import re

def cot_selecting():
    prompts = []
    meta = []

    for idx, a_row in tqdm(amazon.iterrows(), total=len(amazon)):
        candidates = block_records(a_row, google)
        if candidates.empty:
            continue

        anchor_text = format_record(a_row)
        candidate_texts = [format_record(g_row) for _, g_row in candidates.iterrows()]
        candidate_ids = [g_row['id'] for _, g_row in candidates.iterrows()]

        prompt = f"""
        Analyze the following product and choose the best matching candidate.
        Think step-by-step about the title, brand, and manufacturer.

        Product: {anchor_text}

        Candidates:
        """ + "\n".join([f"{i+1}. {text}" for i, text in enumerate(candidate_texts)]) + \
                 f"\n0. None of the above\n\nRespond with the number only. Answer:"

        prompts.append(prompt)
        meta.append((a_row['id'], candidate_ids))

    # === Generate all responses in batch
    sequences = llm(prompts, max_new_tokens=10, batch_size=16)

    predictions = []
    for (a_id, candidate_ids), result, prompt in zip(meta, sequences, prompts):
        output = result['generated_text'].strip()
        match = re.search(r"\b(\d+)\b", output)
        selected_index = int(match.group(1)) if match else 0

        if 1 <= selected_index <= len(candidate_ids):
            selected_id = candidate_ids[selected_index - 1]
            predictions.append({
                "idAmazon": a_id,
                "idGoogle": selected_id,
                "prediction": "Yes",
                "duration": 0,
                "num_tokens": len(prompt.split()) + len(output.split())
            })
        else:
            predictions.append({
                "idAmazon": a_id,
                "idGoogle": None,
                "prediction": "No Match",
                "duration": 0,
                "num_tokens": len(prompt.split()) + len(output.split())
            })

    return predictions


# === Chain-of-Thought Comparing ===
def cot_comparing():
    prompts = []
    meta = []  # (anchor_id, candidate_i_id, candidate_j_id, candidate_i_index)

    for idx, a_row in tqdm(amazon.iterrows(), total=len(amazon)):
        candidates = block_records(a_row, google)
        if len(candidates) < 2:
            continue

        anchor_text = format_record(a_row)
        candidate_list = list(candidates.iterrows())  # [(i, row), ...]
        candidate_ids = [row['id'] for _, row in candidate_list]
        candidate_texts = [format_record(row) for _, row in candidate_list]

        for i in range(len(candidate_texts)):
            for j in range(len(candidate_texts)):
                if i == j:
                    continue
                prompt = f"""
                Compare the two candidates for the anchor product below.
                Think step-by-step about their titles, brands, and features.

                Anchor: {anchor_text}
                Candidate A: {candidate_texts[i]}
                Candidate B: {candidate_texts[j]}

                Which one matches better? Answer 'A' or 'B'.
                """
                prompts.append(prompt)
                meta.append((a_row['id'], candidate_ids[i]))

    # === Run in batch
    sequences = llm(prompts, max_new_tokens=10, batch_size=16)

    # === Tally votes per anchor + candidate
    from collections import defaultdict
    vote_map = defaultdict(lambda: defaultdict(int))  # anchor_id -> candidate_id -> vote count

    for (anchor_id, candidate_id), result, prompt in zip(meta, sequences, prompts):
        output = result['generated_text'].strip().lower()
        if output.startswith("a"):
            vote_map[anchor_id][candidate_id] += 1

    # === Choose the candidate with the most votes per anchor
    predictions = []
    for anchor_id, candidates_votes in vote_map.items():
        best_candidate_id = max(candidates_votes.items(), key=lambda x: x[1])[0]
        predictions.append({
            "idAmazon": anchor_id,
            "idGoogle": best_candidate_id,
            "prediction": "Yes",
            "duration": 0,
            "num_tokens": 0
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

run_and_evaluate(cot_matching, "Chain-Of-Thought Matching")
run_and_evaluate(cot_selecting, "Chain-Of-Thought Selecting")
run_and_evaluate(cot_comparing, "Chain-Of-Thought Comparing")