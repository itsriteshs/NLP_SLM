# Install first:
# pip install transformers accelerate torch pypdf sentencepiece

from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

PDF_FILE = "English_AI_Glossary.pdf"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# 1. Read the English glossary PDF
reader = PdfReader(PDF_FILE)
english_text = "\n".join(page.extract_text() or "" for page in reader.pages)

# 2. Load a Small Language Model (SLM)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

# 3. Ask the SLM to translate the glossary to Tamil
prompt = f"""
Translate the following Artificial Intelligence glossary from English to Tamil.

Rules:
1. Preserve every English technical term.
2. Give a natural Tamil translation for the term.
3. Translate each definition accurately into simple Tamil.
4. Do not omit any item.
5. Return the result in this format:

English Term | Tamil Term | Tamil Definition

GLOSSARY:
{english_text}
"""

messages = [{"role": "user", "content": prompt}]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1800,
        temperature=0.2,
        do_sample=True
    )

result = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)

# 4. Display and save translated glossary
print("\n===== TAMIL GLOSSARY =====\n")
print(result)

with open("Tamil_AI_Glossary.txt", "w", encoding="utf-8") as f:
    f.write(result)

print("\nSaved as Tamil_AI_Glossary.txt")
