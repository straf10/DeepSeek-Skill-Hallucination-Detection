"""
Extracts skills from a Deepseek JSONL results file into CSV and Excel,
with hardcoded input/output paths.
"""

import json
import re
from pathlib import Path

import pandas as pd

# ─── Paths ───────────────────────────────────────────────
INPUT_PATH = Path(r"C:\Python\THESIS\skillab_job_fetcher\output\ds_prompt_results.jsonl")
XLSX_OUTPUT = Path(r"C:\Python\THESIS\skillab_job_fetcher\output\ds_prompt_res_normalised.xlsx")
# ─────────────────────────────────────────────────────────────────────────────

def parse_response_to_list(resp: str) -> list[str]:
    """
    Παίρνει το raw response string (με bullets) και επιστρέφει λίστα καθαρών skill-strings.
    """
    lines = resp.splitlines()
    skills = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # αφαίρεση bullets ή αριθμήσεων στην αρχή
        clean = re.sub(r'^[\s\-\•\d\.\)]+', '', line).strip()
        if clean:
            skills.append(clean)
    return skills

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    records = []
    # Διαβάζουμε το JSONL
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            job_id = obj.get("job_id") or obj.get("id")
            response = obj.get("response", "")
            if job_id is None:
                continue
            for skill in parse_response_to_list(response):
                records.append({
                    "job_id": job_id,
                    "skill": skill
                })

    df = pd.DataFrame.from_records(records, columns=["job_id", "skill"])

    df.to_excel(XLSX_OUTPUT, index=False)
    print(f"[OK] Excel written to: {XLSX_OUTPUT}")

if __name__ == "__main__":
    main()
