from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
# One-off Entity Matching API (Single Record Based on User Request)
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
import google.generativeai as genai
import time
from typing import Optional, List
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())



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

# Load small and medium models
model_name = "declare-lab/flan-alpaca-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
llm_small = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
llm_medium = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)

loaded_models = {}
loaded_models["small"] = llm_small
loaded_models["medium"] = llm_medium

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
# Load Gemini
genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash-lite-001")

def count_tokens(text):
    return gemini.count_tokens(text).total_tokens


class MatchRequest(BaseModel):
    size: str
    task: str
    anchor: str
    candidate: Optional[str] = None
    candidates: Optional[List[str]] = None
    background: Optional[str] = None  # <-- NEW


matching_prompt = """
EXAMPLE Task: Determine whether the following two product records refer to the same entity. 
Compare title, brand, and manufacturer. Answer with only one word: Yes or No.

Record 1:
Title: Canon EOS 5D Mark III 22.3 MP Full Frame CMOS Digital SLR Camera (Body)
Brand: Canon
Manufacturer: Canon

Record 2:
Title: Canon EOS 5D Mark III DSLR Camera (Body Only)
Brand: Canon
Manufacturer: Canon USA

Assistant: Yes
"""

# === Entity Comparing Example ===
comparing_prompt = """
Think step-by-step, but ONLY OUTPUT the final answer as either:
"Candidate 1" or "Candidate 2"

Format:
Candidate 1: [attributes]
Candidate 2: [attributes]
Conclusion: Be concise and respond with Candidate A or Candidate B

ONLY RESPOND with Candidate A or Candidate B
"""

# === Entity Selecting Example ===
selecting_prompt = """
EXAMPLE Task: Determine which of the following candidate products refer to the same entity as the anchor product. 
Compare title, brand, and manufacturer. Think step by step. At the end, respond with: Answer: X, 
where X is the number of the matching candidate, or 0 if none match.

Anchor Product:
Title: Canon EOS 5D Mark III 22.3 MP Full Frame CMOS Digital SLR Camera (Body)
Brand: Canon
Manufacturer: Canon

Candidates:
1.
Title: Canon EOS 5D Mark III DSLR Camera (Body Only)
Brand: Canon
Manufacturer: Canon USA

2.
Title: Nikon D850 FX-format Digital SLR Camera Body
Brand: Nikon
Manufacturer: Nikon Inc

Assistant:
1. Candidate 1 closely matches in title, same model name and function. Brand is the same, and manufacturer is a regional variant of Canon.
2. Candidate 2 is a different brand and product entirely.

Answer: 1
"""


# === Prompt Functions ===
def prompt_matching(anchor, candidate, background=""):
    return f"""
    {matching_prompt}

    HERE IS SOME BACKGROUND INFORMATION TO CONSIDER: {background}

    Do the following two product records refer to the same entity?
    Think step-by-step based on the information provided.

    Product A: {anchor}
    Product B: {candidate}

    Final answer: Yes or No.
    """

def prompt_comparing(anchor, candidate1, candidate2, background=""):
    return f"""
    {comparing_prompt}

    HERE IS SOME BACKGROUND INFORMATION TO CONSIDER: {background}

    Compare the following two candidates for the given anchor product.
    Think step-by-step based on the information provided.

    Anchor: {anchor}
    Candidate A: {candidate1}
    Candidate B: {candidate2}

    Which matches better? Answer 'Candidate A' or 'Candidate B'.
    """

def prompt_selecting(anchor, candidates, background=""):
    candidate_list = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
    return f"""
    {selecting_prompt}

    HERE IS SOME BACKGROUND INFORMATION TO CONSIDER: {background}

    You are given a product and a list of candidates. 
    Analyze each candidate step-by-step and choose the best match.

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
    background = request.background

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
        prompt = prompt_matching(anchor, request.candidate, background)
    elif task == "comparing":
        if not request.candidates or len(request.candidates) != 2:
            return {"error": "Provide exactly 2 candidates for comparing."}
        prompt = prompt_comparing(anchor, request.candidates[0], request.candidates[1], background)
    elif task == "selecting":
        if not request.candidates or len(request.candidates) == 0:
            return {"error": "'candidates' list is required for selecting."}
        prompt = prompt_selecting(anchor, request.candidates, background)
    else:
        return {"error": "Invalid task. Choose from 'matching', 'comparing', 'selecting'."}

    # Generate response
    print(prompt)
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

