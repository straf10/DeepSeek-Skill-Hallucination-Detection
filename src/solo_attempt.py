import re
import requests
import pandas as pd
import logging
import json
import time

from unidecode import unidecode
from pathlib import Path
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ---------- Paths / Config ----------
INPUT_QUERIES = r"C:\Python\THESIS\skillab_job_fetcher\input\queries_test"
SKILLS= r"C:\Python\THESIS\skillab_job_fetcher\output\skill_counts.xlsx"
OUT_JSONL = r"C:\Python\THESIS\skillab_job_fetcher\output\ds_query_bullets.jsonl"

THINK_RE = re.compile(r"<think>[\s\S]*?(?:</think>|$)", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")

HIDE_THINK = True
TEMPERATURE = 0.2
MAX_TOKENS = 512
TIMEOUT = (5, 600)
MODEL_NAME = "deepseek-r1:7b"
BASE_URL = "http://localhost:11434"

session = requests.Session()
retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"])
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

def strip_think(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()

def normalise(s: str) -> str:
    s = unidecode(str(s)).lower().strip()
    return WHITESPACE_RE.sub(" ", s)

def load_queries() -> list[str]:
    path = Path(INPUT_QUERIES)
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out

def load_filtered_skills( min_freq: int = 3, max_allowed_skills: int = 500) -> set[str]:
    """Load skills and return normalized set for O(1) complexity lookup - Give minimum skill frequency and maximum allowed skills"""
    path = Path(SKILLS)
    if not path.exists():
        return set()

    try:
        xls = pd.ExcelFile(path, engine="openpyxl")
        sheet = "filtered_skills" if "filtered_skills" in xls.sheet_names else "skill_counts"
        df = pd.read_excel(xls, sheet_name=sheet)

        if "count" in df.columns and sheet == "skill_counts":
            # Sort by frequency descending
            df = df[df["count"] >= min_freq].sort_values("count", ascending=False)
            skills = df["skill_label"].astype(str).tolist()
        else:
            skills = df["skill_label"].astype(str).tolist()

        # Create normalized set for O(1) lookups
        allowed_set = {normalise(s) for s in skills[:max_allowed_skills]}
        logging.info(
            "Loaded %d skills. Sample: %s",
            len(allowed_set),
            list(allowed_set)[:5]
        )
        return allowed_set

    except Exception as e:
        logging.error(f"Error loading skills: {e}")
        return set()

def make_prompt(query: str, max_points: int) -> str:
    """Always use open prompt for better generation quality"""
    return (
        "You are a skill matching expert. \n"
        f"Answer the following question in up to {max_points} bullet points.\n"
        "Only output bullet points (one skill per bullet). No explanations.\n"
        "Do NOT reveal chain-of-thought.\n\n"
        f"Question: {query}\n"
    )

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

def make_constrained_prompt(query: str, allowed_set: set[str], max_points: int) -> str:
    skills_block = "\n".join(f"- {s}" for s in sorted(allowed_set))
    return (
        "You are a skill matching expert.\n"
        f"Select up to {max_points} skills STRICTLY from the list below.\n"
        "Output ONLY bullet points, one skill per bullet, EXACTLY as written.\n"
        "Do NOT invent, do NOT rephrase. Do NOT reveal chain-of-thought.\n\n"
        f"Question: {query}\n\n"
        f"ALLOWED SKILLS:\n{skills_block}\n"
    )

def answer_with_allowed(query: str, allowed_set: set[str], max_points: int = 5) -> str:
    prompt = make_constrained_prompt(query, allowed_set, max_points)
    return call_deepseek(prompt)

def smoke_test(max_points: int=10, delay_s: float=0.0):
    MIN_FREQ = 3
    MAX_ALLOWED_SKILLS = 465

    allowed = load_filtered_skills(MIN_FREQ, MAX_ALLOWED_SKILLS)
    if not allowed:
        print("Δεν βρέθηκαν allowed skills. Έλεγξε το COUNTS_XLSX ή τα φίλτρα.")
        return

    qpath = Path(INPUT_QUERIES)
    if not qpath.exists():
        print(f"Το αρχείο queries δεν βρέθηκε: {qpath}")
        return

    queries = load_queries()
    if not queries:
        print("Το αρχείο queries είναι κενό ή περιέχει μόνο σχόλια.")
        return

    print(f"Running {len(queries)} smoke queries...\n")
    for i, q in enumerate(queries, 1):
        tstamp = time.strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.perf_counter()

        try:
            ans = answer_with_allowed(q, allowed_set=allowed, max_points=max_points).strip()
            elapsed = time.perf_counter() - start_time

            if not ans:
                ans = "- (no matching skills)"
            print(f"[{i}/{len(queries)}] {tstamp} Q: {q}\n{ans}\n")

        except Exception as e:
            elapsed = (time.perf_counter() - start_time) if 't0' in locals() else 0.0
            print(f"[{i}/{len(queries)}] Q: {q}\n! Error: {e}\n")

        if delay_s > 0:
            time.sleep(delay_s)
# def main():
#     query = "Top skills for a Python backend developer"
#     prompt = make_prompt(query, max_points=5)
#     print("=== Prompt ===")
#     print(prompt)
#     print("==============")
#
#     response = call_deepseek(prompt)
#     print("=== Response ===")
#     print(response if response else "[No response received]")

def main():
    allowed = load_filtered_skills(min_freq=3, max_allowed_skills=465)
    query = "Top skills for a Python backend developer"
    resp = answer_with_allowed(query, allowed, max_points=5)
    print("=== Response ===")
    print(resp)

# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    smoke_test(max_points=10, delay_s=0.5)
