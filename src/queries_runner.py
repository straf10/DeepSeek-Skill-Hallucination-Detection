import os
import json
import argparse
import requests
import logging
import re

from datetime import datetime
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Hardcoded path για το αρχείο των queries
DEFAULT_QUERIES = r"C:\Python\THESIS\skillab_job_fetcher\input\queries"

THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()

def make_concise_prompt(query, max_points=10):
    return (
        f"Answer the following question in up to {max_points} bullet points.\n"
        "Only output the bullet list, no explanations.\n"
        "Do NOT include any chain-of-thought or hidden reasoning in your output.\n\n"
        f"Question: {query}"
    )

class DeepseekModel:
    """Wrapper για local Ollama-based Deepseek μέσω HTTP API."""
    def __init__(self, model_name, device='cpu', base_url=None, timeout=60):
        self.model_name = model_name
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.timeout = timeout

        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)


    def generate(self, prompt, max_tokens=256, temperature=0.2, hide_think=True):
        """Καλεί το Ollama HTTP API στο /api/generate για single-turn inference."""

        prompt = make_concise_prompt(prompt,10)
        url = f"{self.base_url.rstrip('/')}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "5m",
            "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
            }
        }

        if hide_think:
            payload["think"] = False

        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            txt = data.get("response", "")
            return strip_think(txt) if hide_think else txt
        except requests.HTTPError as e:
            r = getattr(e, "response", None)
            if r is not None and r.status_code == 400 and "no such model" in r.text.lower():
                raise RuntimeError(f"Model '{self.model_name}' not found on Ollama at {self.base_url}.") from e
            raise

def load_queries(path):
    """Φορτώνει λίστα queries από text (ένα ανά γραμμή). Αγνοεί κενές και #σχόλια."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Queries file not found: {p}")
    with p.open('r', encoding='utf-8') as f:
        lines = []
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
        return lines


def run_queries(model, queries, max_tokens=512, temperature=0.6,hide_think=True):
    """Τρέχει όλα τα ερωτήματα και επιστρέφει λίστα από dicts με τα αποτελέσματα."""
    results = []
    total = len(queries)
    for i, q in enumerate(queries, 1):
        logging.info(f"Running query {i}/{total}: {q}")
        try:
            resp = model.generate(
                prompt=q,
                max_tokens=max_tokens,
                temperature=temperature,
                hide_think=hide_think
            )
            logging.info(f"Received response for query {i}/{total} (length: {len(resp)} chars)")
        except Exception as e:
            resp = f"ERROR: {e}"
            logging.error(f"Error on query {i}/{total}: {e}")
        results.append({
            'id': i,
            'query': q,
            'response': resp,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    return results


def save_results(results, out_path):
    """Αποθηκεύει το αποτέλεσμα σε JSONL, ένα αντικείμενο JSON ανά γραμμή."""
    with open(out_path, 'w', encoding='utf-8') as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    logging.info(f"All results saved to {out_path}")

def main():

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser(
        description="Run Deepseek queries locally via Ollama with hardcoded queries file and save responses in JSONL"
    )

    parser.add_argument(
        '--show-think', action='store_true',
        help="Αν δοθεί, εμφανίζει το thinking block (default: hidden)"
    )

    parser.add_argument(
        '--queries', '-q',
        default=os.getenv('QUERIES_PATH', DEFAULT_QUERIES),
        help="Text file με ένα query ανά γραμμή"
    )
    parser.add_argument(
        '--out', '-o', default='open_mode_results.jsonl',
        help="Αρχείο εξόδου JSONL (default: ds_query_results.jsonl)"
    )
    parser.add_argument(
        '--model', '-m', default=os.getenv('DEEPSEEK_MODEL', 'deepseek-r1:7b'),
        help="Όνομα Deepseek μοντέλου (env DEEPSEEK_MODEL)"
    )
    parser.add_argument(
        '--base-url', '-b',
        default=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        help="Base URL για ollama server"
    )
    parser.add_argument(
        '--max-tokens', type=int, default=256,
        help="Max tokens ανά απάντηση"
    )
    parser.add_argument(
        '--temperature', type=float, default=0.6,
        help="Sampling temperature"
    )
    args = parser.parse_args()

    # Hardcoded φόρτωση queries
    queries = load_queries(args.queries)
    logging.info(f"Loaded {len(queries)} queries from {args.queries}")

    model = DeepseekModel(
        model_name=args.model,
        base_url=args.base_url,
        timeout=90
    )

    logging.info(f"Model {args.model} ready at {model.base_url}")

    results = run_queries(
        model,
        queries,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        hide_think=not args.show_think
    )
    save_results(results, args.out)

if __name__ == '__main__':
    main()
