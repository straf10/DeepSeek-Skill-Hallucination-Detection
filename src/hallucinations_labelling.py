from pathlib import Path
from typing import List, Tuple, Set
import pandas as pd

# --------------- ΡΥΘΜΙΣΕ ΤΑ PATHS ---------------
SKILL_COUNTS_XLSX   = r"C:\Python\THESIS\skillab_job_fetcher\output\skill_counts.xlsx"
FUZZY_XLSX          = r"C:\Python\THESIS\skillab_job_fetcher\output\fuzzy_candidates.xlsx"
SYNONYMS_XLSX       = r"C:\Python\THESIS\skillab_job_fetcher\output\test_data\synonyms.xlsx"
NORMALISED_XLSX     = r"C:\Python\THESIS\skillab_job_fetcher\output\normalised.xlsx"
OUT_XLSX            = r"C:\Python\THESIS\skillab_job_fetcher\output\H0_skills.xlsx"
OUT_LOW70_XLSX = str(Path(OUT_XLSX).with_name("low_confidence_skills.xlsx"))

FUZZY_H0_THRESHOLD = 90
FUZZY_NEAR_MIN = 70
FUZZY_NEAR_MAX = 89
REASON_RANK = {"exact": 0, "synonym": 1, "fuzzy>=90": 2}

# --------------- Helpers ---------------
def resolve_conflicts(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    """
    Κρατάει ένα μόνο row ανά group keys, επιλέγοντας τον «καλύτερο» λόγο (reason)
    με βάση την προτεραιότητα: exact < synonym < fuzzy>=90.
    """
    if df.empty:
        return df
    df = df.copy()
    df["reason_rank"] = df["reason"].map(REASON_RANK).fillna(99)
    # ταξινόμηση ώστε το καλύτερο reason να έρθει πρώτο ανά group
    df = df.sort_values(group_keys + ["reason_rank"])
    # κρατάμε την 1η εγγραφή ανά group
    df = df.drop_duplicates(subset=group_keys, keep="first")
    return df.drop(columns=["reason_rank"])

def load_allowed_counts(path: str | Path) -> Set[str]:
    xls = pd.ExcelFile(path, engine="openpyxl")
    if "skill_counts" not in xls.sheet_names:
        raise SystemExit("Το 'skill_counts' sheet δεν βρέθηκε στο skill_counts.xlsx")
    df = pd.read_excel(xls, sheet_name="skill_counts")
    col = "skill_label" if "skill_label" in df.columns else df.columns[0]
    allowed = set(df[col].astype(str))
    print(f"[OK] Allowed (skill_counts): {len(allowed)}")
    return allowed

def safe_to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def split_semicolon(s: str) -> List[str]:
    if not isinstance(s, str) or not s:
        return []
    return [x.strip() for x in s.split(";") if x.strip()]

# --------------- Load FUZZY>=90 as H0 ---------------
def h0_from_fuzzy(fuzzy_path: str | Path, allowed: Set[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Επιστρέφει 3 DF (open, closed, jobs) με H0 από fuzzy>=90.
    canonical_skill = candidate_1 (προϋπόθεση: candidate_1 ∈ allowed).
    """
    xls = pd.ExcelFile(fuzzy_path, engine="openpyxl")
    def load_sheet(name: str) -> pd.DataFrame:
        return pd.read_excel(xls, sheet_name=name) if name in xls.sheet_names else pd.DataFrame()

    # μόνο *_candidates
    open_c   = load_sheet("open_mode_candidates")
    closed_c = load_sheet("closed_mode_candidates")
    jobs_c   = load_sheet("job_samples_candidates")

    def pick_h0_from_cand(df: pd.DataFrame, source_kind: str) -> pd.DataFrame:
        if df.empty:
            return df
        cols_ctx_open = ["source","row_idx","question_preview"]
        cols_ctx_jobs = ["job_id","evidence","confidence_model"]
        common_cols = ["original_skill","candidate_1","score_1","score_2","score_3","best_score","band","accept_suggestion"]

        df = ensure_cols(df, common_cols)
        # fuzzy >= 90
        s1 = safe_to_numeric(df["score_1"])
        mask = (s1 >= FUZZY_H0_THRESHOLD)
        df = df[mask].copy()

        # canonical = candidate_1 και πρέπει να είναι στο allowed
        df["canonical_skill"] = df["candidate_1"].astype(str)
        df = df[df["canonical_skill"].isin(allowed)]
        if df.empty:
            return df

        df["reason"] = "fuzzy>=90"

        keep = ["original_skill","canonical_skill","reason","candidate_1","score_1","score_2","score_3","best_score","band","accept_suggestion"]
        if source_kind in ("open_mode","closed_mode"):
            df = ensure_cols(df, cols_ctx_open)
            keep = cols_ctx_open + keep
        else:
            df = ensure_cols(df, cols_ctx_jobs)
            keep = cols_ctx_jobs + keep

        return df[keep].copy()

    open_h0   = pick_h0_from_cand(open_c,   "open_mode")
    closed_h0 = pick_h0_from_cand(closed_c, "closed_mode")
    jobs_h0   = pick_h0_from_cand(jobs_c,   "job_samples")

    return open_h0, closed_h0, jobs_h0


def fuzzy_band_70_89(fuzzy_path: str | Path, allowed: Set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Συλλέγει από τα *_candidates όσα έχουν 70 ≤ score_1 ≤ 89 και επιστρέφει 3 DF:
    open, closed, jobs — με context πεδία + manual_label (κενό αρχικά).
    Περιέχει επίσης ένα βοηθητικό boolean 'in_allowed' για γρήγορο φιλτράρισμα.
    """
    xls = pd.ExcelFile(fuzzy_path, engine="openpyxl")
    def load_sheet(name: str) -> pd.DataFrame:
        return pd.read_excel(xls, sheet_name=name) if name in xls.sheet_names else pd.DataFrame()

    open_c   = load_sheet("open_mode_candidates")
    closed_c = load_sheet("closed_mode_candidates")
    jobs_c   = load_sheet("job_samples_candidates")

    common_cols = ["original_skill","candidate_1","score_1","score_2","score_3","best_score","band","accept_suggestion"]
    cols_ctx_open = ["source","row_idx","question_preview"]
    cols_ctx_jobs = ["job_id","evidence","confidence_model"]

    def pick(df: pd.DataFrame, source_kind: str) -> pd.DataFrame:
        if df.empty:
            return df
        df = ensure_cols(df, common_cols)
        s1 = safe_to_numeric(df["score_1"])
        df = df[(s1 >= 70) & (s1 <= 89)].copy()
        if df.empty:
            return df

        # canonical για αναφορά/έλεγχο σε allowed
        df["canonical_skill"] = df["candidate_1"].astype(str)
        df["in_allowed"] = df["canonical_skill"].isin(allowed)

        # manual labeling fields
        df["manual_label"] = ""      # συμπλήρωσέ το εσύ: H0 ή μία από τις H1_* υποκατηγορίες
        df["manual_notes"] = ""

        keep = ["original_skill","canonical_skill","candidate_1","score_1","score_2","score_3",
                "best_score","band","accept_suggestion","in_allowed","manual_label","manual_notes"]

        if source_kind in ("open_mode","closed_mode"):
            df = ensure_cols(df, cols_ctx_open)
            keep = cols_ctx_open + keep
        else:
            df = ensure_cols(df, cols_ctx_jobs)
            keep = cols_ctx_jobs + keep

        return df[keep].copy()

    return (
        pick(open_c,   "open_mode"),
        pick(closed_c, "closed_mode"),
        pick(jobs_c,   "job_samples"),
    )

def export_no_match_for_labelling(fuzzy_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Διαβάζει τα sheets *_no_match από το fuzzy_candidates.xlsx και επιστρέφει 3 DF
    (open/closed/jobs) με πεδία manual_label/manual_notes για χειροκίνητο labelling.
    """
    xls = pd.ExcelFile(fuzzy_path, engine="openpyxl")

    def load_sheet(name: str) -> pd.DataFrame:
        return pd.read_excel(xls, sheet_name=name) if name in xls.sheet_names else pd.DataFrame()

    open_nm   = load_sheet("open_mode_no_match")
    closed_nm = load_sheet("closed_mode_no_match")
    jobs_nm   = load_sheet("job_samples_no_match")

    cols_ctx_open = ["source","row_idx","question_preview"]
    cols_ctx_jobs = ["job_id","evidence","confidence_model"]
    common_cols   = [
        "original_skill","candidate_1","score_1","candidate_2","score_2","candidate_3","score_3",
        "best_score","band","accept_suggestion"
    ]

    def prep(df: pd.DataFrame, source_kind: str) -> pd.DataFrame:
        if df.empty:
            return df
        df = ensure_cols(df, common_cols)
        # manual πεδία
        df["manual_label"] = ""   # H0 ή H1a/H1b/H1c/H1d/H1e
        df["manual_notes"] = ""

        keep = ["original_skill","candidate_1","score_1","candidate_2","score_2","candidate_3","score_3",
                "best_score","band","accept_suggestion","manual_label","manual_notes"]

        if source_kind in ("open_mode","closed_mode"):
            df = ensure_cols(df, cols_ctx_open)
            keep = cols_ctx_open + keep
        else:
            df = ensure_cols(df, cols_ctx_jobs)
            keep = cols_ctx_jobs + keep

        return df[keep].copy()

    return (
        prep(open_nm,   "open_mode"),
        prep(closed_nm, "closed_mode"),
        prep(jobs_nm,   "job_samples"),
    )


# --------------- Load SYNONYMS rescued as H0 ---------------
def h0_from_synonyms(syn_path: str | Path, allowed: Set[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Παίρνει *_rescued από synonyms.xlsx και σημειώνει reason = "synonym"
    Κρατά μόνο όσα canonical_skill ∈ allowed.
    """
    xls = pd.ExcelFile(syn_path, engine="openpyxl")
    def load_sheet(name: str) -> pd.DataFrame:
        return pd.read_excel(xls, sheet_name=name) if name in xls.sheet_names else pd.DataFrame()

    open_r   = load_sheet("open_mode_rescued")
    closed_r = load_sheet("closed_mode_rescued")
    jobs_r   = load_sheet("job_samples_rescued")

    def pick_h0_from_rescued(df: pd.DataFrame, source_kind: str) -> pd.DataFrame:
        if df.empty:
            return df

        if "canonical_skill" not in df.columns:
            return pd.DataFrame()
        df = df[df["canonical_skill"].astype(str).isin(allowed)].copy()
        if df.empty:
            return df

        df["reason"] = "synonym"

        cols_ctx_open = ["source","row_idx","question_preview"]
        cols_ctx_jobs = ["job_id","evidence","confidence_model"]
        keep_common   = ["original_skill","canonical_skill","reason"]

        if source_kind in ("open_mode","closed_mode"):
            df = ensure_cols(df, cols_ctx_open)
            keep = cols_ctx_open + keep_common
        else:
            df = ensure_cols(df, cols_ctx_jobs)
            keep = cols_ctx_jobs + keep_common

        return df[keep].copy()

    open_h0   = pick_h0_from_rescued(open_r,   "open_mode")
    closed_h0 = pick_h0_from_rescued(closed_r, "closed_mode")
    jobs_h0   = pick_h0_from_rescued(jobs_r,   "job_samples")

    return open_h0, closed_h0, jobs_h0

# --------------- EXACT from normalised.xlsx ---------------
def h0_from_exact(normalised_path: str | Path, allowed: Set[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Διαβάζει normalised.xlsx για να εντοπίσει exact matches.
    - open_mode / closed_mode: col 'skills' με 'a;b;c'
    - job_samples: col 'skill_label'
    Επιστρέφει 3 DF με reason = "exact".
    """
    xls = pd.ExcelFile(normalised_path, engine="openpyxl")
    need = {"open_mode","closed_mode","job_samples"}
    if not need.issubset(set(xls.sheet_names)):
        missing = need - set(xls.sheet_names)
        raise SystemExit(f"Λείπουν sheets από το normalised.xlsx: {missing}")

    open_df   = pd.read_excel(xls, sheet_name="open_mode")
    closed_df = pd.read_excel(xls, sheet_name="closed_mode")
    jobs_df   = pd.read_excel(xls, sheet_name="job_samples")

    def exact_from_openclosed(df: pd.DataFrame, source_kind: str) -> pd.DataFrame:
        if df.empty or "skills" not in df.columns:
            return pd.DataFrame()
        rows = []
        for i, r in df.iterrows():
            qprev = str(r.get("question","") or "").replace("\n"," ")[:120]
            for lab in split_semicolon(str(r.get("skills","") or "")):
                if lab in allowed:
                    rows.append({
                        "source": source_kind,
                        "row_idx": i+1,
                        "question_preview": qprev,
                        "original_skill": lab,
                        "canonical_skill": lab,
                        "reason": "exact",
                    })
        return pd.DataFrame(rows)

    def exact_from_jobs(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "skill_label" not in df.columns:
            return pd.DataFrame()
        rows = []
        for _, r in df.iterrows():
            lab = str(r.get("skill_label","") or "")
            if not lab or lab not in allowed:
                continue
            rows.append({
                "job_id": str(r.get("job_id","") or ""),
                "evidence": str(r.get("evidence","") or ""),
                "confidence_model": r.get("confidence"),
                "original_skill": lab,
                "canonical_skill": lab,
                "reason": "exact",
            })
        return pd.DataFrame(rows)

    open_h0   = exact_from_openclosed(open_df,   "open_mode")
    closed_h0 = exact_from_openclosed(closed_df, "closed_mode")
    jobs_h0   = exact_from_jobs(jobs_df)

    return open_h0, closed_h0, jobs_h0

# --------------- Συγχώνευση & Εξαγωγή ---------------
def dedup(df: pd.DataFrame, subset_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df.drop_duplicates(subset=subset_cols).reset_index(drop=True)

def main():
    allowed = load_allowed_counts(SKILL_COUNTS_XLSX)

    # 1) fuzzy>=90
    f_open, f_closed, f_jobs = h0_from_fuzzy(FUZZY_XLSX, allowed)

    # 2) synonyms rescued
    s_open, s_closed, s_jobs = h0_from_synonyms(SYNONYMS_XLSX, allowed)

    # 3) exact (προαιρετικό)
    if NORMALISED_XLSX:
        try:
            e_open, e_closed, e_jobs = h0_from_exact(NORMALISED_XLSX, allowed)
        except Exception as e:
            print(f"[WARN] Δεν έγινε exact-pass: {e}")
            e_open = e_closed = e_jobs = pd.DataFrame()
    else:
        print("[INFO] Δεν δόθηκε NORMALISED_XLSX → παραλείπεται το exact-pass.")
        e_open = e_closed = e_jobs = pd.DataFrame()

    # Merge per source
    open_df   = pd.concat([e_open, f_open, s_open],   ignore_index=True)
    closed_df = pd.concat([e_closed, f_closed, s_closed], ignore_index=True)
    jobs_df   = pd.concat([e_jobs, f_jobs, s_jobs],   ignore_index=True)

    open_df = resolve_conflicts(open_df, ["source", "row_idx", "canonical_skill"])
    closed_df = resolve_conflicts(closed_df, ["source", "row_idx", "canonical_skill"])
    jobs_df = resolve_conflicts(jobs_df, ["job_id", "canonical_skill"])

    # Dedup (κρατάμε μοναδικούς συνδυασμούς)
    open_df   = dedup(open_df,   ["original_skill","canonical_skill","reason","row_idx"])
    closed_df = dedup(closed_df, ["original_skill","canonical_skill","reason","row_idx"])
    jobs_df   = dedup(jobs_df,   ["original_skill","canonical_skill","reason","job_id"])

    # Save
    # outp = Path(OUT_XLSX)
    # outp.parent.mkdir(parents=True, exist_ok=True)
    # with pd.ExcelWriter(outp, engine="openpyxl") as w:
    #     open_df.to_excel(w,   sheet_name="open_mode",   index=False)
    #     closed_df.to_excel(w, sheet_name="closed_mode", index=False)
    #     jobs_df.to_excel(w,   sheet_name="job_samples", index=False)

    open_7089, closed_7089, jobs_7089 = fuzzy_band_70_89(FUZZY_XLSX, allowed)

    outp = Path(OUT_XLSX)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(outp, engine="openpyxl") as w:
        # H0
        open_df.to_excel(w, sheet_name="open_mode", index=False)
        closed_df.to_excel(w, sheet_name="closed_mode", index=False)
        jobs_df.to_excel(w, sheet_name="job_samples", index=False)
        # 70–89
        open_7089.to_excel(w, sheet_name="open 70-89", index=False)
        closed_7089.to_excel(w, sheet_name="closed 70-89", index=False)
        jobs_7089.to_excel(w, sheet_name="job_samples 70-89", index=False)

    nm_open, nm_closed, nm_jobs = export_no_match_for_labelling(FUZZY_XLSX)

    out_nm = Path(OUT_LOW70_XLSX)
    out_nm.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_nm, engine="openpyxl") as wnm:
        nm_open.to_excel(wnm, sheet_name="open_mode", index=False)
        nm_closed.to_excel(wnm, sheet_name="closed_mode", index=False)
        nm_jobs.to_excel(wnm, sheet_name="job_samples", index=False)

    print(f"[OK] Wrote no‑match workbook → {out_nm.resolve()}")

    # Console summary
    def stats(name, df):
        total = len(df)
        exact = (df["reason"] == "exact").sum() if "reason" in df.columns else 0
        fuzzy = (df["reason"] == "fuzzy>=90").sum() if "reason" in df.columns else 0
        syn   = (df["reason"] == "synonym").sum() if "reason" in df.columns else 0
        print(f"\n{name}: {total} rows  |  exact={exact}  fuzzy>=90={fuzzy}  synonym={syn}")

    print(f"[OK] Wrote {outp.resolve()}")
    stats("open_mode",   open_df)
    stats("closed_mode", closed_df)
    stats("job_samples", jobs_df)

if __name__ == "__main__":
    main()
