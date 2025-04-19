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

# === Load data ===
amazon = pd.read_csv("../datasets/Amazon.csv", encoding='unicode_escape')
google = pd.read_csv("../datasets/GoogleProducts.csv", encoding='unicode_escape')
perfect_mapping = pd.read_csv("../datasets/Amzon_GoogleProducts_perfectMapping.csv", encoding='unicode_escape')

# === Create ground truth set ===
ground_truth_matches = set(zip(perfect_mapping['idAmazon'], perfect_mapping['idGoogleBase']))

# === Blocking preparation ===
def to_blocking_text(row):
    return f"{row.get('title', row.get('name', ''))} {row.get('brand', '')} {row.get('manufacturer', '')}"

amazon['blocking_key'] = amazon.apply(to_blocking_text, axis=1)
google['blocking_key'] = google.apply(to_blocking_text, axis=1)

# === Improved token-based blocking ===
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

# === Format records ===
def format_record(row):
    return f"{row.get('title', row.get('name', ''))} {row.get('brand', '')} {row.get('manufacturer', '')}"

# === Local model query ===
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

# === Inference loop ===
predictions = []
for idx, a_row in tqdm(amazon.iterrows(), total=len(amazon)):
    candidates = block_records(a_row, google)
    anchor_text = format_record(a_row)

    for _, g_row in candidates.iterrows():
        candidate_text = format_record(g_row)
        pred, num_tokens, duration = query_llm(anchor_text, candidate_text)

        predictions.append({
            "idAmazon": a_row['id'],
            "idGoogle": g_row['id'],
            "prediction": pred,
            "duration": duration,
            "num_tokens": num_tokens
        })

# === Evaluation ===
pred_df = pd.DataFrame(predictions)
matched_pairs = set(zip(pred_df[pred_df['prediction'].str.lower() == "yes"]['idAmazon'],
                        pred_df[pred_df['prediction'].str.lower() == "yes"]['idGoogle']))

y_true = []
y_pred = []

for (a_id, g_id) in matched_pairs.union(ground_truth_matches):
    y_true.append((a_id, g_id) in ground_truth_matches)
    y_pred.append((a_id, g_id) in matched_pairs)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
print(f"Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}")

# === Time and token usage ===
total_time = sum(pred['duration'] for pred in predictions)
total_tokens = sum(pred['num_tokens'] for pred in predictions)

print(f"Total time: {total_time:.2f} seconds")
print(f"Total tokens processed (approx): {total_tokens}")