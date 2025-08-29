import re
import json
import pandas as pd
from pathlib import Path
from typing import Iterable, List, Dict, Any

# ---------- Paths / Config ----------
OPEN_MODE_JSONL   = r"C:\Python\THESIS\skillab_job_fetcher\output\open_mode_results.jsonl"
CLOSED_MODE_JSONL = r"C:\Python\THESIS\skillab_job_fetcher\output\closed_mode.jsonl"
JOBS_SAMPLE_JSONL = r"C:\Python\THESIS\skillab_job_fetcher\output\prompt_results.jsonl"
OUT_XLSX          = r"C:\Python\THESIS\skillab_job_fetcher\output\normalised.xlsx"

# ---------- Regex helpers ----------
THINK_RE        = re.compile(r"<think>[\s\S]*?(?:</think>|$)", re.IGNORECASE)
CODE_FENCE_RE   = re.compile(r"^```.*?\n|\n```$", re.DOTALL | re.IGNORECASE)
WHITESPACE_RE   = re.compile(r"\s+")
BULLET_LINE_RE  = re.compile(r"""^\s*(?:[-*•]|\d+[\.)])\s*(.+)$""")

def norm(s: str) -> str:
    """Lower + strip + whitespaces."""
    s = (s or "").strip()
    s = WHITESPACE_RE.sub(" ", s)
    return s.lower()

def strip_meta(text: str) -> str:
    """Removes <think>…</think> block and code fences from answers."""
    t = text or ""
    t = THINK_RE.sub("", t)
    t = CODE_FENCE_RE.sub("", t)
    return t.strip()

# ---------- IO ----------
def _read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] File not found: {p}")
        return []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping invalid JSONL line {i} in {p.name}: {e}")

def _dedup_preserve_order(xs: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

# ---------- Parsing of “open/closed mode” answers ----------
def parse_answer_to_skill_list(answer_text: str) -> List[str]:
    """
    Returns normalised skills list
    """
    if not answer_text:
        return []
    txt = strip_meta(answer_text)

    skills: List[str] = []

    for raw in txt.splitlines():
        m = BULLET_LINE_RE.match(raw)
        if m:
            skills.append(norm(m.group(1)))
        else:
            if "," in raw and len(raw) < 200:
                parts = [norm(p) for p in raw.split(",")]
                skills.extend([p for p in parts if p])
            else:

                if raw.strip().startswith("- "):
                    skills.append(norm(raw.strip()[2:]))

    # Remove empty - duplicates
    skills = [s for s in skills if s]
    skills = _dedup_preserve_order(skills)

    return skills

def load_open_or_closed(jsonl_path: str | Path) -> pd.DataFrame:
    """
    Reads JSONL lines and returns DF with columns: question, skills (semicolon-joined).
    """
    rows: List[Dict[str, str]] = []
    for rec in _read_jsonl(jsonl_path):
        q = (rec.get("question") or "").strip()
        a = rec.get("answer") or ""
        if not q and not a:
            continue
        skills = parse_answer_to_skill_list(a)
        rows.append({
            "question": q,
            "skills": ";".join(skills)
        })
    return pd.DataFrame(rows)

# ---------- Parsing of “job samples” answers ----------
def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    """
    Returns 1st JSON array from text.
    """
    if not text:
        return []

    t = strip_meta(text)
    t = re.sub(r"^\s*json\s*", "", t, flags=re.IGNORECASE)
    t = t.replace("```", "").strip()

    start = t.find("[")
    end   = t.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = t[start:end+1].strip()
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:

            candidate2 = re.sub(r",(\s*[\]\}])", r"\1", candidate)
            try:
                data = json.loads(candidate2)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

    # --- Fallback for unfinished answers ---
    if start == -1:
        return []

    rest = t[start:]
    positions = [m.start() for m in re.finditer(r"\}", rest)]
    for pos in reversed(positions[-20:]):
        cand = rest[:pos+1]
        cand = re.sub(r",\s*$", "", cand)

        if not cand.lstrip().startswith('['):
            cand = '[' + cand
        if not cand.rstrip().endswith(']'):
            cand = cand + ']'

        cand = re.sub(r",(\s*[\]\}])", r"\1", cand)
        try:
            data = json.loads(cand)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            continue

    return []


def load_job_samples(jsonl_path: str | Path) -> pd.DataFrame:
    """
    Waits JSONL lines and returns DF with columns: job_id, skill_label, confidence, evidence.
    """
    out_rows: List[Dict[str, Any]] = []
    for rec in _read_jsonl(jsonl_path):
        job_id = (str(rec.get("job_id") or "")).strip()
        ans    = rec.get("answer") or ""
        arr = _extract_json_array(ans)
        if not arr:
            continue

        for obj in arr:
            # only if dict
            if not isinstance(obj, dict):
                continue

            # normalisation
            label_raw = obj.get("skill_label")
            label = norm(label_raw) if label_raw is not None else ""
            if not label:
                # skip empty lines
                continue

            conf_raw = obj.get("confidence")
            conf = None
            if isinstance(conf_raw, (int, float)):
                conf = float(conf_raw)
            elif isinstance(conf_raw, str):
                s = conf_raw.strip()
                # only values : 0.85, 1, .9, 1e-3 etc
                if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s):
                    conf = float(s)

            evidence = (obj.get("evidence") or "").strip()

            out_rows.append({
                "job_id": job_id,
                "skill_label": label,
                "confidence": conf,  # float ή None
                "evidence": evidence
            })

    return pd.DataFrame(out_rows)

# ---------- Orchestration ----------
def write_excel(
    open_df: pd.DataFrame,
    closed_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    out_xlsx: str | Path = OUT_XLSX
) -> None:
    out_path = Path(out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    open_df   = open_df.fillna("")
    closed_df = closed_df.fillna("")
    jobs_df   = jobs_df.fillna({"confidence": pd.NA, "evidence": ""})

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        open_df.to_excel(writer,   sheet_name="open_mode",   index=False)
        closed_df.to_excel(writer, sheet_name="closed_mode", index=False)
        jobs_df.to_excel(writer,   sheet_name="job_samples", index=False)

    print(f"[OK] Excel written → {out_path.resolve()}")

def main():
    print("[1/4] Loading OPEN MODE…")
    open_df = load_open_or_closed(OPEN_MODE_JSONL)
    print(f"  → {len(open_df)} rows")

    print("[2/4] Loading CLOSED MODE…")
    closed_df = load_open_or_closed(CLOSED_MODE_JSONL)
    print(f"  → {len(closed_df)} rows")

    print("[3/4] Loading JOB SAMPLES…")
    jobs_df = load_job_samples(JOBS_SAMPLE_JSONL)
    print(f"  → {len(jobs_df)} skill rows")

    print("[4/4] Writing Excel…")
    write_excel(open_df, closed_df, jobs_df, OUT_XLSX)

if __name__ == "__main__":
    main()
