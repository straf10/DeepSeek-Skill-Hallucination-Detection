import re
import requests
import json
import logging
import time

from pathlib import Path
from datetime import datetime
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


# ---------- Paths / Config ----------
INPUT_QUERIES = r"C:\Python\THESIS\skillab_job_fetcher\input\queries"
OUT_JSONL = r"C:\Python\THESIS\skillab_job_fetcher\output\open_mode_results.jsonl"

THINK_RE = re.compile(r"<think>[\s\S]*?(?:</think>|$)", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")

HIDE_THINK = True
TEMPERATURE = 0.3
MAX_TOKENS = 512
TIMEOUT = (5, 600)
MODEL_NAME = "deepseek-r1:7b"
BASE_URL = "http://localhost:11434"

session = requests.Session()
retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"])
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

# --- helper for JSON ---
def write_jsonl(record: dict, path: str = OUT_JSONL) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")

def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()

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

def make_concise_prompt(query, max_points=10):
    return (
        f"List up to {max_points} skills as plain bullet points.\n"
        "Only output the bullet list, one skill per line.\n"
        "No numbering, no explanations, no bold/markdown, no extra text.\n"
        "Each line must start with '- ' followed by the skill label.\n"
        "Return only the bullet lines.\n\n"
        f"Question: {query}"
    )

def run_queries(max_points: int = 10) -> None:
    """Loads queries, asks model and writes JSONL: {timestamp, question, answer}."""
    queries = load_queries()
    if not queries:
        raise RuntimeError("No queries found. Check INPUT_QUERIES path.")

    print(f"Running {len(queries)} queries...\n")
    for i, q in enumerate(queries, 1):
        tstamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.perf_counter()

        prompt = make_concise_prompt(q, max_points=max_points)
        answer = call_deepseek(prompt).strip()
        if not answer:
            answer = "- (no matching skills)"

        write_jsonl({
            "timestamp": tstamp,
            "question": q,
            "answer": answer,
        })

        elapsed = time.perf_counter() - start_time

        print(f"[{i}/{len(queries)}] {tstamp} Q: {q}\n{answer}\n(Elapsed: {elapsed:.2f}s)\n")

if __name__ == '__main__':
    run_queries(10)