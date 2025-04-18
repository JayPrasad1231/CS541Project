import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

model.eval()

def compute_yes_no_logprobs(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    # Get logit scores for the first generated token
    scores = outputs.scores[0]  # shape: (1, vocab_size)
    logits = scores[0]

    yes_token_id = tokenizer("Yes")["input_ids"][0]
    no_token_id = tokenizer("No")["input_ids"][0]

    probs = F.softmax(logits, dim=-1)
    logprobs = F.log_softmax(logits, dim=-1)

    yes_prob = probs[yes_token_id].item()
    no_prob = probs[no_token_id].item()
    yes_logprob = logprobs[yes_token_id].item()
    no_logprob = logprobs[no_token_id].item()

    return {
        "yes_prob": yes_prob,
        "no_prob": no_prob,
        "yes_logprob": yes_logprob,
        "no_logprob": no_logprob,
        "pred": "yes" if yes_prob > no_prob else "no"
    }

# Example
anchor = {"title": "Cruzer Force USB Flash Drive 32GB", "brand": "SanDisk"}
candidate = {"title": "Sandisk USB Flash Drive 32GB Cruzer Glide", "brand": "SanDisk"}

prompt = f"""Do these two records refer to the same real-world entity? Answer Yes or No.
Record A: Title: {anchor['title']}, Brand: {anchor['brand']}
Record B: Title: {candidate['title']}, Brand: {candidate['brand']}"""

score_info = compute_yes_no_logprobs(prompt)
print(score_info)
