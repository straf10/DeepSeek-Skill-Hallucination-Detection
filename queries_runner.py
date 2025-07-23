import os
import json
import argparse
import requests
import logging
import re
from datetime import datetime

# Hardcoded path για το αρχείο των queries
DEFAULT_QUERIES = r"C:\Python\THESIS\skillab_job_fetcher\queries"

THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()

def make_concise_prompt(query, max_points=10):
    return (
        f"Answer the following question in up to {max_points} bullet points.\n"
        "Do NOT include any chain-of-thought or reasoning in your output.\n\n"
        "You are extremely concise.\n"
        f"Question: {query}"
    )

class DeepseekModel:
    """Wrapper για local Ollama-based Deepseek μέσω HTTP API."""
    def __init__(self, model_name, device='cpu', base_url=None):
        self.model_name = model_name
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

    def generate(self, prompt, max_tokens=256, temperature=0.6, hide_think=True):
        """Καλεί το Ollama HTTP API στο /api/generate για single-turn inference."""

        prompt = make_concise_prompt(prompt,10)
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        if hide_think:
            payload["think"] = False
        payload["options"] = {
            "temperature": temperature,
            "num_predicts": max_tokens
        }

        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        txt = data.get("response", "")
        return strip_think(txt)

def load_queries(path):
    """Φορτώνει τη λίστα των queries από ένα απλό text αρχείο, ένα ανά γραμμή."""
    with open(path, 'r', encoding='utf-8') as f:
        return [q.strip() for q in f if q.strip()]


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
    # Ενεργοποίηση logging
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
        help="Text file με ένα  querie ανά γραμμή"
    )
    parser.add_argument(
        '--out', '-o', default='results_v2.jsonl',
        help="Αρχείο εξόδου JSONL (default: results_v2.jsonl)"
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
    logging.info(f"Loaded {len(queries)} queries from {DEFAULT_QUERIES}")

    model = DeepseekModel(
        model_name=args.model,
        base_url=args.base_url
    )

    logging.info(f"Model {args.model} ready at {model.base_url}")

    results = run_queries(
        model,
        queries,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    save_results(results, args.out)

if __name__ == '__main__':
    main()
