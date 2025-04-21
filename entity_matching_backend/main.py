from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
# One-off Entity Matching API (Single Record Based on User Request)
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import google.generativeai as genai
import time
from typing import Optional, List


app = FastAPI()

# 👇 CORS should go here, immediately after app is created
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# === Model Setup ===
models = {
    "small": "google/flan-t5-base",
    "medium": "google/flan-t5-xl"
}

# Load small and medium models
print("Loading small and medium models...")
loaded_models = {}
for size, name in models.items():
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to("cuda")
    loaded_models[size] = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)

# Load Gemini
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini = genai.GenerativeModel("gemini-2.0-flash-lite-001")

def count_tokens(text):
    return gemini.count_tokens(text).total_tokens

# === FastAPI Setup ===
app = FastAPI()

class MatchRequest(BaseModel):
    size: str  # 'small', 'medium', or 'large'
    task: str  # 'matching', 'comparing', or 'selecting'
    anchor: str
    candidate: Optional[str] = None  # Optional for matching
    candidates: Optional[List[str]] = None  # Optional for selecting/comparing


# === Prompt Functions ===
def prompt_matching(anchor, candidate):
    return f"""
    Do the following two product records refer to the same entity?
    Think step-by-step about the title, brand, and manufacturer.

    Product A: {anchor}
    Product B: {candidate}

    Final answer: Yes or No.
    """

def prompt_comparing(anchor, candidate1, candidate2):
    return f"""
    Compare the following two candidates for the given anchor product.
    Think step-by-step based on title, brand, and manufacturer.

    Anchor: {anchor}
    Candidate A: {candidate1}
    Candidate B: {candidate2}

    Which matches better? Answer 'A' or 'B'.
    """

def prompt_selecting(anchor, candidates):
    candidate_list = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
    return f"""
    You are given a product and a list of candidates. Analyze each candidate step-by-step and choose the best match.

    Product: {anchor}

    Candidates:
    {candidate_list}

    Respond with reasoning and end with 'Answer: X' where X is the best match or 0 for no match.
    """

# === Main Endpoint ===
@app.post("/entity-match")
def entity_match(request: MatchRequest):
    start = time.time()

    size = request.size.lower()
    task = request.task.lower()
    anchor = request.anchor

    # Select model
    if size in ["small", "medium"]:
        model = loaded_models[size]
        def call_model(prompt):
            return model(prompt, max_new_tokens=50, truncation=True)[0]['generated_text']
    elif size == "large":
        def call_model(prompt):
            return gemini.generate_content(prompt).text
    else:
        return {"error": "Invalid model size. Choose from 'small', 'medium', 'large'."}

    # Build prompt based on task
    if task == "matching":
        if not request.candidate:
            return {"error": "'candidate' is required for matching."}
        prompt = prompt_matching(anchor, request.candidate)
    elif task == "comparing":
        if not request.candidates or len(request.candidates) != 2:
            return {"error": "Provide exactly 2 candidates for comparing."}
        prompt = prompt_comparing(anchor, request.candidates[0], request.candidates[1])
    elif task == "selecting":
        if not request.candidates or len(request.candidates) == 0:
            return {"error": "'candidates' list is required for selecting."}
        prompt = prompt_selecting(anchor, request.candidates)
    else:
        return {"error": "Invalid task. Choose from 'matching', 'comparing', 'selecting'."}

    # Generate response
    output = call_model(prompt)
    duration = time.time() - start

    token_count = count_tokens(prompt + output) if size == "large" else len(prompt.split()) + len(output.split())

    return {
        "response": output.strip(),
        "tokens": token_count,
        "duration_sec": duration,
        "model": size,
        "task": task
    }

