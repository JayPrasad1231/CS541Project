import time
import os
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
from transformers import pipeline

load_dotenv(find_dotenv())

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Load models
model_large = pipeline("text2text-generation", model="google/flan-t5-base", device=0)
model_small = pipeline("text2text-generation", model="google/flan-t5-small", device=0)
genai.configure(api_key=GOOGLE_API_KEY)
model_llm = genai.GenerativeModel("gemini-2.0-flash-lite-001")

def format_record(record):
    return ', '.join(f"{k}: {v}" for k, v in record.items())

def create_match_prompt(anchor, candidate):
    return (
        "Do these two records refer to the same real-world entity? Answer Yes or No.\n"
        f"Record A: {format_record(anchor)}\n"
        f"Record B: {format_record(candidate)}"
    )

def ask_verifier(question, anchor, candidate, model):
    prompt = f"{question}\n\nRecord A: {format_record(anchor)}\nRecord B: {format_record(candidate)}"
    output = model(prompt, max_new_tokens=50, truncation=True)[0]['generated_text']
    return output


def direct_match(anchor, candidate, model):
    prompt = create_match_prompt(anchor, candidate)
    start = time.time()
    output = model(prompt, max_new_tokens=10, truncation=True)[0]['generated_text']
    duration = time.time() - start
    tokens = len(prompt.split()) + len(output.split())
    return {"match": output.strip(), "time": duration, "tokens": tokens}

def query_llm(anchor, candidate):
    prompt = create_match_prompt(anchor, candidate)
    start = time.time()

    # Count tokens first
    token_info = model_llm.count_tokens(prompt)
    num_tokens = token_info.total_tokens  # total prompt tokens

    # Generate response
    output = model_llm.generate_content(prompt)
    duration = time.time() - start

    return {
        "match": output.text.strip(),
        "time": duration,
        "tokens": num_tokens
    }



# Chain of Verification Questions:

verification_questions = [
    "Are the product titles semantically equivalent?",
    "Do the brands match or are they known aliases?",
    "Are the sizes or quantities compatible?",
    "Is there any contradicting information between the records?"
]

def chain_of_verification(anchor, candidate, verifier_model, final_model):
    answers = []
    total_tokens = 0
    start = time.time()

    for q in verification_questions:
        answer = ask_verifier(q, anchor, candidate, verifier_model)
        answers.append(answer)
        total_tokens += len(answer.split()) + len(q.split())

    # Summarize evidence for final judgment
    evidence = "\n".join([f"- {a}" for a in answers])
    final_prompt = (
        f"The following questions were asked to verify a match between two records:\n{evidence}\n\n"
        "Based on this evidence, do the two records refer to the same entity? Answer Yes or No."
    )

    final_output = final_model(final_prompt, max_new_tokens=10, truncation=True)[0]['generated_text']
    duration = time.time() - start
    total_tokens += len(final_prompt.split()) + len(final_output.split())

    return {"match": final_output.strip(), "time": duration, "tokens": total_tokens}

# Example pair
anchor = {"title": "SanDisk Cruzer Force 32GB USB 2.0", "brand": "SanDisk"}
candidate = {"title": "Cruzer Force 32GB Flash Drive", "brand": "SanDisk"}

# Run
# baseline_result = direct_match(anchor, candidate, model_small)
# cov_result = chain_of_verification(anchor, candidate, model_small, model_large)

llm_result = query_llm(anchor, candidate)

print("=== LLM Result ===")
print(f"Decision: {llm_result['match']}")
print(f"Time: {llm_result['time']:.2f}s")
print(f"Tokens used: {llm_result['tokens']}")

# Display
# print("=== DIRECT MATCH ===")
# print(f"Decision: {baseline_result['match']}")
# print(f"Time: {baseline_result['time']:.2f}s")
# print(f"Tokens used: {baseline_result['tokens']}")

# print("\n=== CHAIN OF VERIFICATION ===")
# print(f"Decision: {cov_result['match']}")
# print(f"Time: {cov_result['time']:.2f}s")
# print(f"Tokens used: {cov_result['tokens']}")

