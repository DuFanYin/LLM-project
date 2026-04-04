#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('pip', 'install datasets transformers torch accelerate')


# In[1]:


# Datasets
from datasets import load_dataset, concatenate_datasets

# Output
import json
from collections import Counter
import pandas as pd

# Model
import torch
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import time


# In[2]:


# Load dataset
dataset = load_dataset("gsm8k", "main")

# Combine train and test datasets
dataset = concatenate_datasets([dataset["train"], dataset["test"]])

# Filter datasets
dataset = dataset.filter(
    lambda x: len(x["answer"]) > 404
)


dataset = dataset.shuffle(seed=42)
print(dataset)


# In[3]:


# Load model
model_name = "Qwen/Qwen2-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)


# In[4]:


# Helper function
def extract_true_answer(answer):
    return answer.split("####")[-1].strip()

def extract_true_reasoning(answer):
    return answer.split("####")[0].strip()

def count_tokens(input_text, output_text):
    input_tokens = len(tokenizer.encode(input_text))
    output_tokens = len(tokenizer.encode(output_text))
    return input_tokens, output_tokens

def extract_intermediate_results(reasoning):
    """
    Extract numbers appearing immediately after >> in the reasoning.
    Returns as strings with commas removed.
    """
    matches = re.findall(r">>([\d,\.]+)", reasoning)
    return [m.replace(",", "").strip() for m in matches]


# In[5]:


def build_prompt(question, true_answer, true_reasoning):

    return f"""
Solve the math problem step by step.

Return your answer in EXACTLY this JSON format:
{{
  "model_reasoning": "...",
  "model_answer": "..."
}}

Rules:
- model_reasoning: step-by-step reasoning only
- model_answer: final numeric answer only (no words, no symbols)
- Do NOT include any text outside the JSON
- Do NOT include markdown
- Do NOT repeat the question
- Do NOT include ####

Problem:
{question}
"""


# In[6]:


pipe = pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device=0 if torch.cuda.is_available() else -1
)


# In[ ]:


results = []

start_time = time.time()

for i, example in enumerate(dataset.select(range(100))):
    question = example["question"]

    true_answer = extract_true_answer(example["answer"])
    true_reasoning = extract_true_reasoning(example["answer"])

    prompt = build_prompt(
        question=question,
        true_answer=true_answer,
        true_reasoning=true_reasoning
    )

    # Run model
    output = pipe(
        prompt,
        max_new_tokens=320,
        return_full_text=False,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.1
    )[0]

    response = output["generated_text"]

    def extract_json(text):
        """
        Extract the first valid JSON object from the text.
        Returns {} if no valid JSON found.
        """
        matches = re.finditer(r"\{.*?\}", text, flags=re.DOTALL)
        for match in matches:
            candidate = match.group()
            try:
                parsed = json.loads(candidate)
                # Ensure keys exist to avoid null later
                if "model_answer" not in parsed:
                    parsed["model_answer"] = ""
                if "model_reasoning" not in parsed:
                    parsed["model_reasoning"] = ""
                return parsed
            except json.JSONDecodeError:
                continue
        return {"model_answer": "", "model_reasoning": ""}

    parsed = extract_json(response)
        
    if parsed:
        model_answer = parsed["model_answer"].strip()      
        model_reasoning = parsed["model_reasoning"].strip()
    else:
        print("PARSE FAILED:\n", response)
        model_answer = ""
        model_reasoning = ""

    # Accuracy
    is_correct = (model_answer == true_answer)

    # Token usage
    input_tokens, output_tokens = count_tokens(prompt, response)

    results.append({
        "question": question,
        "true_reasoning": true_reasoning,
        "true_answer": true_answer,
        "model_reasoning": model_reasoning,
        "model_answer": model_answer,
        "correct": is_correct,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "response": response
    })

end_time = time.time()


# In[ ]:


df = pd.DataFrame(results)
# df.to_csv("qwen_gsm8k_results.csv", index=False)

df["true_answer"] = pd.to_numeric(df["true_answer"], errors="coerce")

df.to_json("results.json", orient="records", indent=4)


# In[ ]:


# Intermediate Step Numeric Matching

def compute_step_accuracy(results):
    total_steps = 0
    correct_steps = 0

    for r in results:
        true_steps = extract_intermediate_results(r["true_reasoning"])
        model_steps = extract_intermediate_results(r["model_reasoning"])

        for t, m in zip(true_steps, model_steps):
            total_steps += 1
            if t == m:
                correct_steps += 1

    return correct_steps / total_steps if total_steps > 0 else 0


# In[ ]:


# Formatting

def compute_format_score(results):
    """
    Compute the format score for a list of results without changing them.
    
    Criteria for being correctly formatted:
    1. model_answer:
       - not empty
       - contains only numbers (may include decimal points)
       - does NOT include any reasoning or extra text
    2. model_reasoning:
       - not empty
       - does NOT include the final answer (no numbers after #### or similar)
    
    Returns:
        float: proportion of entries meeting the correct format
    """
    valid = 0

    for r in results:
        # --- Check model_answer ---
        answer = r["model_answer"]

        if isinstance(answer, (int, float)):
            answer_ok = True
        elif isinstance(answer, str):
            answer_ok = bool(answer) and bool(re.fullmatch(r"\d+(\.\d+)?", answer))
        else:
            answer_ok = False

        # --- Check model_reasoning ---
        reasoning = r["model_reasoning"].strip()
        # Non-empty and must not contain final answer markers
        reasoning_ok = bool(reasoning) and "####" not in reasoning

        if answer_ok and reasoning_ok:
            valid += 1

    format_score = valid / len(results) if results else 0
    return format_score


# In[ ]:


# Consistency

# def get_multiple_attempts(prompt, k):
#     outputs = []

#     for _ in range(k):
#         output = pipe(
#             prompt,
#             max_new_tokens=320,
#             do_sample=False,
#             temperature=0.0
#         )[0]["generated_text"]

#         # Extract numeric model_answer from plain text
#         match = re.search(r"\b\d+(\.\d+)?\b", output)
#         if match:
#             outputs.append(match.group(0))
#         else:
#             outputs.append("")

#     return outputs

# def compute_consistency(results_with_attempts):
#     scores = []

#     for attempts in results_with_attempts:
#         counts = Counter(attempts)
#         most_common = counts.most_common(1)[0][1]
#         scores.append(most_common / len(attempts))

#     return sum(scores) / len(scores) if scores else 0


# In[ ]:


# Metrics

total_runtime = end_time - start_time
avg_runtime = total_runtime / len(results)
print(f"Total runtime for {len(results)} examples: {total_runtime:.3f} sec")
print(f"Average runtime per example: {avg_runtime:.3f} sec")

accuracy = sum(r["correct"] for r in results) / len(results)
print("Final answer accuracy:", accuracy)

step_accuracy = compute_step_accuracy(results)
print("Intermediate step accuracy:", step_accuracy)

avg_input_tokens = sum(r["input_tokens"] for r in results) / len(results)
avg_output_tokens = sum(r["output_tokens"] for r in results) / len(results)

print("Avg input tokens:", avg_input_tokens)
print("Avg output tokens:", avg_output_tokens)

format_score = compute_format_score(results)
print("Format score:", format_score)

# results_with_attempts = []

# for r in results:
#     # Build prompt
#     prompt = build_prompt(question=r["question"], true_answer=true_answer, true_reasoning=true_reasoning)

#     # Get multiple answers from the model
#     attempts = get_multiple_attempts(prompt, k=3)
#     results_with_attempts.append(attempts)

# # Compute consistency
# consistency_score = compute_consistency(results_with_attempts)
# print("Consistency:", consistency_score)


# In[ ]:




