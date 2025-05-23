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

# === Chain-of-Thought Matching ===
def gemini_cot_matching(amazon_df, google_df, block_fn):
    prompts = []
    metadata = []

    for _, a_row in tqdm(amazon_df.iterrows(), total=len(amazon_df)):
        candidates = block_fn(a_row, google_df)
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
            prompts.append(prompt)
            metadata.append((a_row['id'], g_row['id']))

    predictions = []
    for prompt, (idAmazon, idGoogle) in tqdm(zip(prompts, metadata), total=len(prompts)):
        start = time.time()
        response = model.generate_content(prompt)
        duration = time.time() - start
        num_tokens = count_tokens(prompt + response.text)

        prediction = "Yes" if "yes" in response.text.lower() else "No"

        predictions.append({
            "idAmazon": idAmazon,
            "idGoogle": idGoogle,
            "prediction": prediction,
            "duration": duration,
            "num_tokens": num_tokens
        })

    return predictions


def gemini_cov_selecting(amazon_df, google_df, block_fn):
    prompts = []
    metadata = []

    for _, a_row in tqdm(amazon_df.iterrows(), total=len(amazon_df)):
        candidates = block_fn(a_row, google_df)
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

        prompts.append(prompt)
        metadata.append((a_row['id'], candidate_ids))

    predictions = []
    for prompt, (idAmazon, candidate_ids) in tqdm(zip(prompts, metadata), total=len(prompts)):
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
            idGoogle = candidate_ids[selected_index - 1]
            prediction = "Yes"
        else:
            idGoogle = None
            prediction = "No Match"

        predictions.append({
            "idAmazon": idAmazon,
            "idGoogle": idGoogle,
            "prediction": prediction,
            "duration": duration,
            "num_tokens": num_tokens
        })

    return predictions


def gemini_cot_comparing(amazon_df, google_df, block_fn):
    predictions = []
    for _, a_row in tqdm(amazon_df.iterrows(), total=len(amazon_df)):
        candidates = block_fn(a_row, google_df)
        if len(candidates) < 2:
            continue

        anchor_text = format_record(a_row)
        candidate_ids = [row['id'] for _, row in candidates.iterrows()]
        candidate_texts = [format_record(row) for _, row in candidates.iterrows()]

        prompts = []
        comparisons = []

        for i, text_i in enumerate(candidate_texts):
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
                prompts.append(prompt)
                comparisons.append((i, j))

        votes = [0] * len(candidate_texts)
        for prompt, (i, j) in zip(prompts, comparisons):
            response = model.generate_content(prompt)
            if response.text.strip().lower().startswith("a"):
                votes[i] += 1

        best_index = votes.index(max(votes))
        predictions.append({
            "idAmazon": a_row['id'],
            "idGoogle": candidate_ids[best_index],
            "prediction": "Yes",
            "duration": 0,
            "num_tokens": 0
        })

    return predictions

# === Evaluation ===
def run_and_evaluate(strategy_fn, name):
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

run_and_evaluate(gemini_cot_matching, "Chain of Thought Matching with Gemini")
run_and_evaluate(gemini_cov_selecting, "Chain of Verification + Chain of Thought Selecting")
run_and_evaluate(gemini_cot_matching, "Chain of THought Comparing with Gemini")