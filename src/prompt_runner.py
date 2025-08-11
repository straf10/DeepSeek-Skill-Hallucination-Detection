"""
prompt_runner.py  –  Deepseek prompts per job row
--------------------------------------------------------------------------
• Διαβάζει το Excel `job_samples` με στήλες: job_id, title, experience_level, description
• Στέλνει το Template-A prompt σε Ollama Deepseek
• Αποθηκεύει αποτελέσματα σε JSONL `OUT_JSONL`
"""

import json
import logging
import re
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path


# ----------------------------------------------------------------------
# 0. Hard-coded ρυθμίσεις
# ----------------------------------------------------------------------
job_samples   = r"C:\Python\THESIS\skillab_job_fetcher\input\job_samples.xlsx"
OUT_JSONL  = r"C:\Python\THESIS\skillab_job_fetcher\output\deepseek_skill_results.jsonl"
MODEL_NAME = "deepseek-r1:7b"
BASE_URL   = "http://localhost:11434"      # Ollama 0.9.6

MAX_TOKENS   = 256
TEMPERATURE  = 0.0
HIDE_THINK   = True

# ----------------------------------------------------------------------
# 1. Deepseek wrapper
# ----------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
def _strip_think(txt: str) -> str:
    return _THINK_RE.sub("", txt).strip()

def generate(prompt: str) -> str:
    payload = {
        "model":  MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predicts": MAX_TOKENS
        }
    }
    if HIDE_THINK:
        payload["think"] = False

    r = requests.post(f"{BASE_URL.rstrip('/')}/api/generate", json=payload, timeout=210)
    r.raise_for_status()
    return _strip_think(r.json().get("response", ""))

# ----------------------------------------------------------------------
# 2. Prompt template
# ----------------------------------------------------------------------
TEMPLATE = """You are a career-research assistant.
Task: List the skills explicitly required for the job below. The skills can be either hard or soft skills.
Rules:
• Return only a bullet list, one skill per line.
• All lowercase, no sentences, no numbering.
• Maximum 12 skills.
• Include obvious synonyms (e.g. “oop” ⇒ “object-oriented programming”).
• Do NOT add responsibilities or commentary.
• Do NOT reveal chain-of-thought.

<TITLE>: {title}
<EXPERIENCE_LEVEL>: {level}
<DESCRIPTION>:
{description}
"""

# ----------------------------------------------------------------------
# 3. Main run
# ----------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # φόρτωσε Excel
    df = pd.read_excel(job_samples, dtype=str)
    must = {"job_id", "title", "experience_level", "description"}
    if missing := must - set(df.columns.str.lower()):
        raise SystemExit(f"ERROR: Missing columns {', '.join(missing)} in {job_samples}")

    out_path = Path(OUT_JSONL).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Loaded %d jobs from %s", len(df), job_samples)
    logging.info("Model=%s  BaseURL=%s", MODEL_NAME, BASE_URL)

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, row in df.iterrows():
            prompt = TEMPLATE.format(
                title=row["title"],
                level=row.get("experience_level", "unknown"),
                description=(row.get("description") or "").strip()
            )
            try:
                resp = generate(prompt)
            except Exception as e:
                logging.error("Error on job %s: %s", row["job_id"], e)
                resp = f"ERROR: {e}"

            fout.write(json.dumps({
                "job_id"   : row["job_id"],
                "prompt"   : prompt,
                "response" : resp,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds")+"Z"
            }, ensure_ascii=False) + "\n")

            logging.info("Job %s done (%d/%d)", row["job_id"], idx+1, len(df))

    logging.info("FINISHED → %s", out_path)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
