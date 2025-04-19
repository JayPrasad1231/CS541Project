# Gemini-based Entity Matching using Chain-of-Thought and Chain-of-Verification
import time
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)  # Replace with your actual key
model = genai.GenerativeModel("gemini-2.0-flash-lite-001")

# === Utility Functions ===
def format_record(row):
    return f"{row.get('title', row.get('name', ''))} {row.get('brand', '')} {row.get('manufacturer', '')}"

def count_tokens(text):
    return model.count_tokens(text).total_tokens

# === Chain-of-Thought Matching ===
def gemini_cot_matching(amazon, google, block_records):
    predictions = []
    for _, a_row in tqdm(amazon.iterrows(), total=len(amazon)):
        candidates = block_records(a_row, google)
        anchor_text = format_record(a_row)

        for _, g_row in candidates.iterrows():
            candidate_text = format_record(g_row)
            prompt = f"""
            Determine if these two products refer to the same entity.
            Think step-by-step about the title, brand, and manufacturer.

            Product A: {anchor_text}
            Product B: {candidate_text}

            Final Answer: Yes or No.
            """
            start = time.time()
            response = model.generate_content(prompt)
            duration = time.time() - start
            num_tokens = count_tokens(prompt + response.text)

            predictions.append({
                "idAmazon": a_row['id'],
                "idGoogle": g_row['id'],
                "prediction": "Yes" if "yes" in response.text.lower() else "No",
                "duration": duration,
                "num_tokens": num_tokens
            })

    return predictions

# === Chain-of-Thought Comparing ===
def gemini_cot_comparing(amazon, google, block_records):
    predictions = []
    for _, a_row in tqdm(amazon.iterrows(), total=len(amazon)):
        candidates = block_records(a_row, google)
        if len(candidates) < 2:
            continue

        anchor_text = format_record(a_row)
        candidate_ids = [row['id'] for _, row in candidates.iterrows()]
        candidate_texts = [format_record(row) for _, row in candidates.iterrows()]

        scored = []
        for i, text_i in enumerate(candidate_texts):
            votes = 0
            for j, text_j in enumerate(candidate_texts):
                if i == j:
                    continue
                prompt = f"""
                Compare the following two candidates for the given anchor product.
                Think step-by-step based on title, brand, and manufacturer.

                Anchor: {anchor_text}
                Candidate A: {text_i}
                Candidate B: {text_j}

                Which matches better? Answer 'A' or 'B'.
                """
                response = model.generate_content(prompt)
                if response.text.strip().lower().startswith('a'):
                    votes += 1
            scored.append((votes, candidate_ids[i]))

        best_candidate = max(scored)[1]
        predictions.append({
            "idAmazon": a_row['id'],
            "idGoogle": best_candidate,
            "prediction": "Yes",
            "duration": 0,
            "num_tokens": 0
        })

    return predictions

# === Chain-of-Thought + Chain-of-Verification Selecting ===
def gemini_cov_selecting(amazon, google, block_records):
    predictions = []
    for _, a_row in tqdm(amazon.iterrows(), total=len(amazon)):
        candidates = block_records(a_row, google)
        if candidates.empty:
            continue

        anchor_text = format_record(a_row)
        candidate_texts = [format_record(g_row) for _, g_row in candidates.iterrows()]
        candidate_ids = [g_row['id'] for _, g_row in candidates.iterrows()]

        prompt = f"""
        You are given a product and a list of candidates. Think step-by-step about whether each candidate matches the product.
        Consider title, brand, and manufacturer for each.

        Product: {anchor_text}

        Candidates:
        """ + "\n".join([f"{i+1}. {text}" for i, text in enumerate(candidate_texts)]) + "\n\nProvide reasoning and then state the number of the best match, or 0 if none match."

        start = time.time()
        response = model.generate_content(prompt)
        duration = time.time() - start
        num_tokens = count_tokens(prompt + response.text)

        try:
            lines = response.text.strip().split("\n")
            answer_line = next(line for line in lines if "answer" in line.lower())
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

# === Evaluation ===
def run_and_evaluate(strategy_fn, name, amazon, google, ground_truth_matches, block_records):
    print(f"\n=== Running {name} ===")
    predictions = strategy_fn(amazon, google, block_records)
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
