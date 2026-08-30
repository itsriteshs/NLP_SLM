# NLP SLM Tamil Glossary Translator

Translate an English AI glossary PDF into Tamil using a small language model.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python slm_glossary_translate_tamil.py
```

The first run downloads `Qwen/Qwen2.5-1.5B-Instruct` from Hugging Face. The translated output is saved to `Tamil_AI_Glossary.txt`.
