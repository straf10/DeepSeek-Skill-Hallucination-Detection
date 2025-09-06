import re
import requests
import json
import logging
import time
import pandas as pd

from pathlib import Path
from datetime import datetime
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ---------- Paths / Config ----------
job_samples   = r"C:\Python\THESIS\skillab_job_fetcher\input\job_samples.xlsx"
OUT_JSONL  = r"C:\Python\THESIS\skillab_job_fetcher\output\prompt_results.jsonl"

THINK_RE = re.compile(r"<think>[\s\S]*?(?:</think>|$)", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")

HIDE_THINK = True
TEMPERATURE = 0.3
MAX_TOKENS = 400
TIMEOUT = (5, 300)
MODEL_NAME = "deepseek-r1:7b"
BASE_URL = "http://localhost:11434"

session = requests.Session()
retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset({"POST"}))
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()

def write_jsonl(record: dict, path: str = OUT_JSONL) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")

def call_deepseek(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": float(TEMPERATURE), "num_predict": int(MAX_TOKENS)}
    }
    if HIDE_THINK:
        payload["think"] = False

    try:
        logging.info(f"Sending request to {BASE_URL} with payload: {json.dumps(payload)}")
        resp = session.post(f"{BASE_URL.rstrip('/')}/api/generate", json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        logging.info(f"Raw response: {data}")
        return strip_think(data.get("response", ""))
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return ""
    except json.JSONDecodeError:
        logging.error("Invalid JSON response from API")
        return ""

def make_prompt(title: str, level: str, description: str, max_skills: int = 10) -> str:
    return f"""
You are a skill extraction assistant.

Task:
From the job description below, extract up to {max_skills} distinct skills.
Return ONLY a JSON array of objects with the following fields:
  - "skill_label": canonical lowercase form of the skill
  - "evidence": short phrase copied exactly from the description
  - "confidence": float between 0 and 1

Rules:
- Rules:
- Absolutely NO markdown, no ``` fences, no prose. Output ONLY JSON array.
- One skill per object. Do not include responsibilities or tasks.
- Use canonical names (e.g. "js" -> "javascript", "oop" -> "object-oriented programming").
- Confidence reflects certainty of correct skill extraction.
- Evidence must be verbatim from the job description.

Job:
<TITLE>: {title}
<EXPERIENCE_LEVEL>: {level}
<DESCRIPTION>:
{description}
"""


def run_jobs():
    df = pd.read_excel(job_samples, dtype=str)
    must = {"job_id", "title", "experience_level", "description"}
    if not must.issubset(df.columns):
        raise ValueError("Missing required columns in Excel")

    for i, row in df.iterrows():
        t0 = time.perf_counter()
        prompt = make_prompt(row["title"], row["experience_level"], row["description"])
        answer = call_deepseek(prompt)
        if not answer:
            answer = "[]"

        write_jsonl({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "job_id": row["job_id"],
            "answer": answer,
        })

        elapsed = time.perf_counter() - t0
        print(f"[{i+1}/{len(df)}] Job {row['job_id']} done in {elapsed:.1f}s")

if __name__ == '__main__':
    run_jobs()