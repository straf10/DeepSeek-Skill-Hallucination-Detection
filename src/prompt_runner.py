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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# PATHS
job_samples   = r"C:\Python\THESIS\skillab_job_fetcher\input\job_samples.xlsx"
OUT_JSONL  = r"C:\Python\THESIS\skillab_job_fetcher\output\ds_prompt_results.jsonl"
MODEL_NAME = "deepseek-r1:7b"
BASE_URL   = "http://localhost:11434"      # Ollama 0.11.4

COUNTS_XLSX = Path(r"C:\Python\THESIS\skillab_job_fetcher\output\skill_counts.xlsx")
MIN_GLOBAL_FREQ = 3
MAX_SHORTLIST   = 120
RETRY_SHORTLISTS = [120, 80, 50, 30]
INCLUDE_EVIDENCE = False

MAX_TOKENS   = 160  # Default 256
TEMPERATURE  = 0.0
HIDE_THINK   = True

_session = requests.Session()
_retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504], allowed_methods=["POST"])
_adapter = HTTPAdapter(max_retries=_retry)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


# Deepseek wrapper
_THINK_RE = re.compile(r"<think>[\s\S]*?(?:</think>|$)", re.IGNORECASE)
def _strip_think(txt: str) -> str:
    return _THINK_RE.sub("", txt or "").strip()

_ARRAY_RE = re.compile(r"\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)?\]", re.MULTILINE)

def _extract_json_array(txt: str) -> str:
    m = _ARRAY_RE.search(txt or "")
    return m.group(0) if m else "[]"

def generate(prompt: str) -> str:
    payload = {
        "model":  MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "temperature": float(TEMPERATURE),
            "num_predict": int(MAX_TOKENS)  # <-- 0.11.x
        }
    }
    if HIDE_THINK:
        payload["think"] = False

    r = _session.post(f"{BASE_URL.rstrip('/')}/api/generate", json=payload, timeout=210)
    r.raise_for_status()
    return _strip_think(r.json().get("response", ""))




def load_filtered_labels(min_count: int = MIN_GLOBAL_FREQ) -> list[str]:
    xls = pd.ExcelFile(COUNTS_XLSX, engine="openpyxl")
    if "filtered_skills" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="filtered_skills")
        return df["skill_label"].astype(str).tolist()

    df = pd.read_excel(xls, sheet_name="skill_counts")
    return df.loc[df["count"] >= min_count, "skill_label"].astype(str).tolist()

def load_label_counts() -> dict[str, int]:
    xls = pd.ExcelFile(COUNTS_XLSX, engine="openpyxl")
    df = pd.read_excel(xls, sheet_name="skill_counts")
    return dict(zip(df["skill_label"].astype(str), df["count"].astype(int)))

_WS = re.compile(r"\w+", re.UNICODE)
STOP = {"and","or","the","for","with","to","of","in","on","at","by","as","an","a",
        "skills","skill","developer","engineer"}

def norm_tokens(s: str) -> set[str]:
    return {w.lower() for w in _WS.findall(s or "") if len(w) >= 3 and w.lower() not in STOP}

def build_shortlist_from_labels(description: str,
                                labels: list[str],
                                counts: dict[str,int],
                                max_k: int = MAX_SHORTLIST) -> list[tuple[int,str]]:
    text_toks = norm_tokens(description)
    cand = []
    for lbl in labels:
        toks = [t for t in norm_tokens(lbl) if len(t) >= 3]
        if not toks:
            continue
        hits = sum(1 for t in toks if t in text_toks)
        if hits == 0:
            continue

        score = hits*10 + sum(len(t) for t in toks if t in text_toks)
        if counts:
            df = max(counts.get(lbl, 1), 1)
            score += 5.0 / (1.0 + df**0.5)
        cand.append((score, lbl))
    cand.sort(reverse=True, key=lambda x: x[0])
    top = [lbl for _s, lbl in cand[:max_k]]
    return list(enumerate(top, start=1))  # [(ID, label)]


# Prompt template
TEMPLATE = """You are a career-research assistant.

Task:
From the job below, extract up to 12 distinct skills (hard or soft). 
Return a pure JSON array (no prose) of objects with fields:
  - "skill_label": string (lowercase, canonical form)
  - "skill_id": string (empty if unknown)
  - "evidence": short phrase copied from the description that supports this skill
  - "confidence": number in [0,1]

Rules:
- Output ONLY a JSON array. No markdown, no code fences, no extra text.
- One skill per object.
- Prefer canonical names (e.g., "oop" -> "object-oriented programming").
- Do NOT include responsibilities or commentary.
- Do NOT reveal chain-of-thought.

Job:
<TITLE>: {title}
<EXPERIENCE_LEVEL>: {level}
<DESCRIPTION>:
{description}
"""



# Main run
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # φόρτωσε Excel
    df = pd.read_excel(job_samples, dtype=str);
    df.columns = df.columns.str.lower()
    # έλεγχος ότι υπάρχουν οι σωστές στήλες
    must = {"job_id", "title", "experience_level", "description"}
    missing = must - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: Missing columns {', '.join(sorted(missing))} in {job_samples}")

    # προετοιμασία output αρχείου
    out_path = Path(OUT_JSONL).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_labels = load_filtered_labels(MIN_GLOBAL_FREQ)  # ~500 labels με κάλυψη >80%
    label_counts = load_label_counts()  # χάρτης label→count

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, row in df.iterrows():
            desc = (row.get("description") or "").strip()

            resp = "[]";
            used_idmap = [];
            err = None
            for k in RETRY_SHORTLISTS:
                idlab = build_shortlist_from_labels(desc, filtered_labels, label_counts, max_k=k)
                if not idlab:
                    resp = "[]";
                    used_idmap = [];
                    err = None
                    break
                ids_block = "\n".join(f"{i} : {lbl}" for i, lbl in idlab)
                prompt = TEMPLATE.format(
                    title=row.get("title", ""),
                    level=row.get("experience_level", ""),
                    description=desc
                )

                try:
                    resp = generate(prompt)
                    try:
                        _ = json.loads(resp)
                    except Exception:
                        logging.warning("Non-JSON response for job %s; saving raw text.", row["job_id"])

                    used_idmap = [{"id": i, "label": lbl} for i, lbl in idlab]
                    err = None
                    break
                except requests.exceptions.ReadTimeout:
                    err = f"timeout@k={k}"
                    continue
                except Exception as e:
                    err = str(e);
                    break

            fout.write(json.dumps({
                "job_id": row["job_id"],
                "response": resp,
                "shortlist": used_idmap,  # [{id,label}] για post-map
                "shortlist_k": len(used_idmap),
                "error": err,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z"
            }, ensure_ascii=False) + "\n")

            logging.info("Job %s done (%d/%d)", row["job_id"], idx+1, len(df))

    logging.info("FINISHED → %s", out_path)

if __name__ == "__main__":
    main()
