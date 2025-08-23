import pandas as pd
import numpy as np

from pathlib import Path
from typing import List, Tuple, Set
from rapidfuzz import process, fuzz

# ---------- Paths ----------
NORMALISED_XLSX = r"C:\Python\THESIS\skillab_job_fetcher\output\normalised.xlsx"
COUNTS_XLSX     = r"C:\Python\THESIS\skillab_job_fetcher\output\skill_counts.xlsx"
OUTPUT_XLSX     = r"C:\Python\THESIS\skillab_job_fetcher\output\fuzzy_candidates.xlsx"

# ---------- Config ----------
USE_ALL_SKILLS = True
TOP_K = 3
SCORER = fuzz.token_sort_ratio  # εναλλακτικά fuzz.ratio
MIN_DISPLAY_SCORE = 70
CAND_THRESHOLD        = 70
AUTO_ACCEPT_THRESHOLD = 90
BANDS = [
    ("High",    90),
    ("Medium",  80),
    ("Low",     70),
    ("VeryLow",  0),
]

NO_MATCH_TOKENS = {
    "(no matching skills)",
    "no matching skills",
    "- (no matching skills)",
}

def band_of(score: int | float) -> str:
    try:
        s = int(score)
    except (TypeError, ValueError):
        s = 0
    if s >= 90: return "High"
    if s >= 80: return "Medium"
    if s >= 70: return "Low"
    return "VeryLow"

def load_allowed_set(path: str | Path, use_all: bool) -> Set[str]:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Missing: {p}")
        return set()
    xls = pd.ExcelFile(p, engine="openpyxl")
    if use_all:
        sheet = "skill_counts" if "skill_counts" in xls.sheet_names else xls.sheet_names[0]
    else:
        sheet = "filtered_skills" if "filtered_skills" in xls.sheet_names else (
            "skill_counts" if "skill_counts" in xls.sheet_names else xls.sheet_names[0]
        )
    df = pd.read_excel(xls, sheet_name=sheet)
    col = "skill_label" if "skill_label" in df.columns else df.columns[0]
    allowed = {s for s in df[col].astype(str).tolist() if s}
    print(f"[OK] Allowed from '{sheet}': {len(allowed)} skills")
    return allowed

def split_semicolon_cell(cell: str) -> List[str]:
    if not isinstance(cell, str):
        return []

    parts = [x.strip() for x in cell.split(";")]
    return [x for x in parts if x and x not in NO_MATCH_TOKENS]

def topk_matches(label: str, allowed_list: List[str], k: int) -> List[Tuple[str, int]]:
    if not allowed_list:
        return []

    res = process.extract(label, allowed_list, scorer=SCORER, limit=k, processor=None)
    out = []
    for cand, score, _ in res:
        if score >= MIN_DISPLAY_SCORE:
            out.append((cand, int(score)))
    return out

def augment_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Προσθέτει:
      - predicted_label = candidate_1
      - predicted_score = score_1
      - band (High/Medium/Low/VeryLow)
      - accept_suggestion: auto_accept / review / no_candidate
    """
    if df.empty:
        return df
    df = df.copy()
    # ασφαλής numeric score_1
    s1 = pd.to_numeric(df.get("score_1", np.nan), errors="coerce")
    df["predicted_label"] = df.get("candidate_1", "")
    df["predicted_score"] = s1
    df["band"] = df["predicted_score"].apply(band_of)

    def suggest(row):
        sc = row.get("predicted_score")
        if pd.isna(sc):
            return "no_candidate"
        if sc >= AUTO_ACCEPT_THRESHOLD:
            return "auto_accept"
        return "review"

    df["accept_suggestion"] = df.apply(suggest, axis=1)
    return df

def split_by_threshold(df: pd.DataFrame, threshold: int):
    """
    Επιστρέφει (candidates_df, no_match_df) με βάση το αν score_1 >= threshold.
    Εφόσον το RapidFuzz επιστρέφει φθίνουσα σειρά, το score_1 είναι πάντα το μέγιστο.
    """
    if df.empty:
        return df.copy(), df.copy()
    s1 = pd.to_numeric(df.get("score_1"), errors="coerce").fillna(-1)
    mask = s1 >= threshold
    return df[mask].copy(), df[~mask].copy()

def process_open_or_closed(df: pd.DataFrame, sheet_name: str, allowed: Set[str]) -> pd.DataFrame:
    if "question" not in df.columns or "skills" not in df.columns:
        return pd.DataFrame()
    allowed_list = list(allowed)
    rows = []
    for i, row in df.iterrows():
        q = str(row.get("question") or "")
        preview = q.replace("\n", " ")[:120]
        skills = split_semicolon_cell(str(row.get("skills") or ""))
        # optional per-row dedup (χωρίς normalization)
        seen = set()
        for lab in skills:
            if lab in seen:
                continue
            seen.add(lab)
            # exact?
            if lab in allowed:
                continue
            # fuzzy
            matches = topk_matches(lab, allowed_list, TOP_K)
            best = max((s for _, s in matches), default=0)
            rec = {
                "source": sheet_name,
                "row_idx": i + 1,
                "question_preview": preview,
                "original_skill": lab,
                "candidate_1": matches[0][0] if len(matches) > 0 else "",
                "score_1":     matches[0][1] if len(matches) > 0 else None,
                "candidate_2": matches[1][0] if len(matches) > 1 else "",
                "score_2":     matches[1][1] if len(matches) > 1 else None,
                "candidate_3": matches[2][0] if len(matches) > 2 else "",
                "score_3":     matches[2][1] if len(matches) > 2 else None,
                "best_score":  best,
            }
            rows.append(rec)
    return pd.DataFrame(rows)

def process_job_samples(df: pd.DataFrame, allowed: Set[str]) -> pd.DataFrame:
    need = {"job_id", "skill_label"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    allowed_list = list(allowed)
    rows = []
    for _, row in df.iterrows():
        jid = str(row.get("job_id") or "")
        lab = str(row.get("skill_label") or "")
        if not lab or lab in NO_MATCH_TOKENS:
            continue
        if lab in allowed:
            continue  # κρατάμε μόνο τα non-exact
        matches = topk_matches(lab, allowed_list, TOP_K)
        best = max((s for _, s in matches), default=0)
        rec = {
            "job_id": jid,
            "original_skill": lab,
            "evidence": str(row.get("evidence") or ""),
            "confidence_model": row.get("confidence"),
            "candidate_1": matches[0][0] if len(matches) > 0 else "",
            "score_1":     matches[0][1] if len(matches) > 0 else None,
            "candidate_2": matches[1][0] if len(matches) > 1 else "",
            "score_2":     matches[1][1] if len(matches) > 1 else None,
            "candidate_3": matches[2][0] if len(matches) > 2 else "",
            "score_3":     matches[2][1] if len(matches) > 2 else None,
            "best_score":  best,
        }
        rows.append(rec)
    return pd.DataFrame(rows)

def main():
    xls = pd.ExcelFile(NORMALISED_XLSX, engine="openpyxl")
    need = {"open_mode", "closed_mode", "job_samples"}
    if not need.issubset(set(xls.sheet_names)):
        raise SystemExit(f"Missing sheets in {NORMALISED_XLSX}: {need - set(xls.sheet_names)}")

    open_df   = pd.read_excel(xls, sheet_name="open_mode")
    closed_df = pd.read_excel(xls, sheet_name="closed_mode")
    jobs_df   = pd.read_excel(xls, sheet_name="job_samples")

    allowed = load_allowed_set(COUNTS_XLSX, use_all=USE_ALL_SKILLS)

    REQUIRED_COLS = ["original_skill", "candidate_1", "score_1", "candidate_2", "score_2", "candidate_3", "score_3",
                     "best_score"]

    def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in REQUIRED_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        return df

    # build base outputs (non-exact με top-k)
    open_out = process_open_or_closed(open_df, "open_mode", allowed)
    closed_out = process_open_or_closed(closed_df, "closed_mode", allowed)
    jobs_out = process_job_samples(jobs_df, allowed)

    # εμπλουτισμός με predicted_* & suggestions
    open_out = augment_predictions(open_out)
    closed_out = augment_predictions(closed_out)
    jobs_out = augment_predictions(jobs_out)

    # split σε candidates/no-match με βάση το CAND_THRESHOLD
    open_cand, open_nomatch = split_by_threshold(open_out, CAND_THRESHOLD)
    closed_cand, closed_nomatch = split_by_threshold(closed_out, CAND_THRESHOLD)
    jobs_cand, jobs_nomatch = split_by_threshold(jobs_out, CAND_THRESHOLD)

    # write 6 sheets
    out_path = Path(OUTPUT_XLSX)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        open_cand.to_excel(writer, sheet_name="open_mode_candidates", index=False)
        closed_cand.to_excel(writer, sheet_name="closed_mode_candidates", index=False)
        jobs_cand.to_excel(writer, sheet_name="job_samples_candidates", index=False)

        open_nomatch.to_excel(writer, sheet_name="open_mode_no_match", index=False)
        closed_nomatch.to_excel(writer, sheet_name="closed_mode_no_match", index=False)
        jobs_nomatch.to_excel(writer, sheet_name="job_samples_no_match", index=False)

    print(f"[OK] Wrote {out_path.resolve()}")
    print(f"  open_mode_candidates:   {len(open_cand)} rows | no_match: {len(open_nomatch)}")
    print(f"  closed_mode_candidates: {len(closed_cand)} rows | no_match: {len(closed_nomatch)}")
    print(f"  job_samples_candidates: {len(jobs_cand)} rows | no_match: {len(jobs_nomatch)}")


if __name__ == "__main__":
    main()